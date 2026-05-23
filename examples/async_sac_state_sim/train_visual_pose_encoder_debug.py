#!/usr/bin/env python3
"""Debug-only overfit test for VisualPoseEncoder.

This script does not connect pose_hat to the SAC actor. It only verifies whether
depth images can supervise socket-relative pose prediction on a fixed dataset.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Dict, Tuple

if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MUJOCO_ENV_ROOT = PROJECT_ROOT / "mujoco-env"
for path in (MUJOCO_ENV_ROOT, PROJECT_ROOT, Path(__file__).resolve().parent):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import numpy as np
import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from networks.visual_pose_encoder import VisualPoseEncoder
from train_visual_sac_with_history import make_visual_history_env

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


def _write_gray_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = image.astype(np.uint8)
    if cv2 is not None:
        cv2.imwrite(str(path), image)
    else:
        pgm_path = path.with_suffix(".pgm")
        with pgm_path.open("wb") as f:
            f.write(f"P5\n{image.shape[1]} {image.shape[0]}\n255\n".encode("ascii"))
            f.write(image.tobytes())


def make_env(args: argparse.Namespace):
    env_args = Namespace(
        control_dt=args.control_dt,
        physics_dt=args.physics_dt,
        render_dt=args.render_dt,
        max_episode_steps=args.max_episode_steps,
        ik_regularization=args.ik_regularization,
        ik_radius=args.ik_radius,
        history_len=5,
        image_size=args.image_size,
        camera_name=args.camera_name,
        max_depth=args.max_depth,
        save_depth_dir="./debug_depth_pose_encoder",
        save_depth_count=0,
        show_depth=False,
        log_every=args.num_samples + 1,
        render_human=False,
        inspect_raw_obs=False,
    )
    return make_visual_history_env(env_args)


def get_history_wrapper(env):
    current = env
    while current is not None:
        if hasattr(current, "base_env") and hasattr(current, "_render_depth"):
            return current
        current = getattr(current, "env", None)
    raise AttributeError("Could not find DepthProprioHistoryObservationWrapper in env chain")


def get_pose_sample(env) -> Dict[str, np.ndarray]:
    wrapper = get_history_wrapper(env)
    base_env = wrapper.base_env

    depth = wrapper._render_depth(tag="dataset")
    tcp_pos, tcp_quat = wrapper._get_tcp_pose()
    socket_pos = getattr(base_env, "_desired_goal", None)
    if socket_pos is None and getattr(base_env, "task", None) is not None:
        socket_pos = getattr(base_env.task, "hole_position", None)
    if socket_pos is None:
        socket_pos = np.zeros(3, dtype=np.float32)
    socket_pos = np.asarray(socket_pos[:3], dtype=np.float32)

    rel_world = socket_pos - tcp_pos
    site_name = getattr(base_env.controller, "ee_site_name", "grip_site")
    tcp_rotm = base_env.get_site_rotm(site_name).astype(np.float32)
    rel_tcp = tcp_rotm.T @ rel_world

    return {
        "depth": depth.astype(np.float32),
        "relative_pos_world": rel_world.astype(np.float32),
        "relative_pos_tcp": rel_tcp.astype(np.float32),
        "tcp_pos": tcp_pos.astype(np.float32),
        "tcp_quat": tcp_quat.astype(np.float32),
        "socket_pos_world": socket_pos.astype(np.float32),
    }


def collect_dataset(args: argparse.Namespace) -> Dict[str, np.ndarray]:
    env = make_env(args)
    data = []
    obs, _ = env.reset()
    del obs

    while len(data) < args.num_samples:
        data.append(get_pose_sample(env))
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset()

    env.close()
    return {
        key: np.stack([sample[key] for sample in data], axis=0).astype(np.float32)
        for key in data[0].keys()
    }


def load_or_collect_dataset(args: argparse.Namespace) -> Dict[str, np.ndarray]:
    dataset_path = Path(args.dataset_path)
    if args.recollect or not dataset_path.exists():
        print(f"[dataset] collecting {args.num_samples} samples from env...")
        dataset = collect_dataset(args)
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dataset_path, **dataset)
        print(f"[dataset] saved to {dataset_path}")
        return dataset

    print(f"[dataset] loading {dataset_path}")
    with np.load(dataset_path) as data:
        return {key: data[key].astype(np.float32) for key in data.files}


def print_range(name: str, value: np.ndarray) -> None:
    print(
        f"[{name}] shape={value.shape}, "
        f"min={value.min(axis=0)}, max={value.max(axis=0)}, "
        f"mean={value.mean(axis=0)}, std={value.std(axis=0)}"
    )


def inspect_depth(depth: np.ndarray, output_dir: Path, count: int) -> None:
    print(
        "[depth] "
        f"shape={depth.shape}, min={depth.min():.6f}, max={depth.max():.6f}, "
        f"mean={depth.mean():.6f}, std={depth.std():.6f}, "
        f"nan={np.isnan(depth).any()}, inf={np.isinf(depth).any()}, "
        f"all_zero={np.allclose(depth, 0.0)}, all_constant={np.allclose(depth.min(), depth.max())}"
    )
    for idx in range(min(count, len(depth))):
        depth_hw = depth[idx, 0]
        vis = ((1.0 - depth_hw) * 255.0).clip(0, 255).astype(np.uint8)
        _write_gray_image(output_dir / f"depth_{idx:03d}.png", vis)
    if count > 0:
        print(f"[depth] saved {min(count, len(depth))} debug images to {output_dir}")


def split_dataset(
    depth: np.ndarray,
    gt: np.ndarray,
    train_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(depth))
    rng.shuffle(indices)
    split = int(len(indices) * train_ratio)
    train_idx, val_idx = indices[:split], indices[split:]
    return depth[train_idx], gt[train_idx], depth[val_idx], gt[val_idx]


def make_loader(depth: np.ndarray, gt_norm: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        th.as_tensor(depth, dtype=th.float32),
        th.as_tensor(gt_norm, dtype=th.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def denormalize(value_norm: th.Tensor, mean: th.Tensor, std: th.Tensor) -> th.Tensor:
    return value_norm * std + mean


def eval_model(
    model: VisualPoseEncoder,
    loader: DataLoader,
    mean: th.Tensor,
    std: th.Tensor,
    device: th.device,
) -> Dict[str, float]:
    model.eval()
    losses = []
    errors = []
    axis_errors = []
    preds = []
    with th.no_grad():
        for depth, gt_norm in loader:
            depth = depth.to(device)
            gt_norm = gt_norm.to(device)
            pred_norm, _ = model(depth)
            loss = F.mse_loss(pred_norm, gt_norm)
            pred = denormalize(pred_norm, mean, std)
            gt = denormalize(gt_norm, mean, std)
            abs_error = th.abs(pred - gt)
            losses.append(loss.item())
            errors.append(th.linalg.norm(pred - gt, dim=1).mean().item() * 1000.0)
            axis_errors.append(abs_error.mean(dim=0).cpu().numpy() * 1000.0)
            preds.append(pred.cpu().numpy())

    axis_error = np.mean(np.stack(axis_errors, axis=0), axis=0)
    pred_std = np.concatenate(preds, axis=0).std(axis=0)
    return {
        "pose_loss": float(np.mean(losses)),
        "pose_error_mm": float(np.mean(errors)),
        "pose_error_x_mm": float(axis_error[0]),
        "pose_error_y_mm": float(axis_error[1]),
        "pose_error_z_mm": float(axis_error[2]),
        "pose_hat_std_mean": float(pred_std.mean()),
    }


def module_grad_norm(module: th.nn.Module) -> float:
    total = 0.0
    for param in module.parameters():
        if param.grad is not None:
            total += float(param.grad.detach().pow(2).sum().item())
    return total ** 0.5


def module_param_norm(module: th.nn.Module) -> float:
    total = 0.0
    for param in module.parameters():
        total += float(param.detach().pow(2).sum().item())
    return total ** 0.5


def model_param_delta(model: th.nn.Module, initial_state: Dict[str, th.Tensor]) -> float:
    total = 0.0
    current_state = model.state_dict()
    for key, initial_value in initial_state.items():
        delta = current_state[key].detach().cpu() - initial_value
        total += float(delta.pow(2).sum().item())
    return total ** 0.5


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_or_collect_dataset(args)
    depth = dataset["depth"]
    rel_world = dataset["relative_pos_world"]
    rel_tcp = dataset["relative_pos_tcp"]
    gt = rel_tcp if args.label_frame == "tcp" else rel_world

    print_range("relative_pos_world", rel_world)
    print_range("relative_pos_tcp", rel_tcp)
    print(f"[label] training label frame: {args.label_frame}")
    inspect_depth(depth, output_dir / "depth_vis", args.save_depth_count)

    train_depth, train_gt, val_depth, val_gt = split_dataset(
        depth,
        gt,
        train_ratio=0.8,
        seed=args.seed,
    )
    pos_mean_np = train_gt.mean(axis=0)
    pos_std_np = train_gt.std(axis=0)
    pos_std_np = np.maximum(pos_std_np, 1e-6)
    print(f"[normalization] pos_mean={pos_mean_np}, pos_std={pos_std_np}")

    train_gt_norm = (train_gt - pos_mean_np) / pos_std_np
    val_gt_norm = (val_gt - pos_mean_np) / pos_std_np

    mean_pred = np.repeat(pos_mean_np[None, :], len(val_gt), axis=0)
    mean_baseline_error_mm = float(np.linalg.norm(mean_pred - val_gt, axis=1).mean() * 1000.0)
    print(f"[baseline] mean_baseline_error_mm={mean_baseline_error_mm:.6f}")

    device = th.device(args.device if th.cuda.is_available() or args.device == "cpu" else "cpu")
    train_loader = make_loader(train_depth, train_gt_norm, args.batch_size, shuffle=True)
    val_loader = make_loader(val_depth, val_gt_norm, args.batch_size, shuffle=False)
    model = VisualPoseEncoder(depth_shape=tuple(depth.shape[1:])).to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mean = th.as_tensor(pos_mean_np, dtype=th.float32, device=device)
    std = th.as_tensor(pos_std_np, dtype=th.float32, device=device)
    initial_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    csv_path = output_dir / "pose_encoder_debug_log.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train/pose_loss",
                "val/pose_loss",
                "train/pose_error_mm",
                "val/pose_error_mm",
                "val/pose_error_x_mm",
                "val/pose_error_y_mm",
                "val/pose_error_z_mm",
                "mean_baseline_error_mm",
                "visual_encoder_grad_norm",
                "pose_head_grad_norm",
                "visual_encoder_param_norm",
                "pose_head_param_norm",
                "pose_hat_std_mean",
                "param_delta_norm",
            ],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            model.train()
            train_losses = []
            train_errors = []
            last_encoder_grad_norm = 0.0
            last_head_grad_norm = 0.0

            for batch_depth, batch_gt_norm in train_loader:
                batch_depth = batch_depth.to(device)
                batch_gt_norm = batch_gt_norm.to(device)
                pred_norm, _ = model(batch_depth)
                loss = F.mse_loss(pred_norm, batch_gt_norm)

                optimizer.zero_grad()
                loss.backward()
                last_encoder_grad_norm = module_grad_norm(model.cnn) + module_grad_norm(model.mlp)
                last_head_grad_norm = module_grad_norm(model.pose_head)
                optimizer.step()

                with th.no_grad():
                    pred = denormalize(pred_norm, mean, std)
                    target = denormalize(batch_gt_norm, mean, std)
                    train_errors.append(th.linalg.norm(pred - target, dim=1).mean().item() * 1000.0)
                    train_losses.append(loss.item())

            train_metrics = {
                "pose_loss": float(np.mean(train_losses)),
                "pose_error_mm": float(np.mean(train_errors)),
            }
            val_metrics = eval_model(model, val_loader, mean, std, device)
            param_delta = model_param_delta(model, initial_state)
            row = {
                "epoch": epoch,
                "train/pose_loss": train_metrics["pose_loss"],
                "val/pose_loss": val_metrics["pose_loss"],
                "train/pose_error_mm": train_metrics["pose_error_mm"],
                "val/pose_error_mm": val_metrics["pose_error_mm"],
                "val/pose_error_x_mm": val_metrics["pose_error_x_mm"],
                "val/pose_error_y_mm": val_metrics["pose_error_y_mm"],
                "val/pose_error_z_mm": val_metrics["pose_error_z_mm"],
                "mean_baseline_error_mm": mean_baseline_error_mm,
                "visual_encoder_grad_norm": last_encoder_grad_norm,
                "pose_head_grad_norm": last_head_grad_norm,
                "visual_encoder_param_norm": module_param_norm(model.cnn) + module_param_norm(model.mlp),
                "pose_head_param_norm": module_param_norm(model.pose_head),
                "pose_hat_std_mean": val_metrics["pose_hat_std_mean"],
                "param_delta_norm": param_delta,
            }
            writer.writerow(row)

            if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
                print(
                    f"epoch={epoch:04d} "
                    f"train/pose_loss={row['train/pose_loss']:.6f} "
                    f"val/pose_loss={row['val/pose_loss']:.6f} "
                    f"train/pose_error_mm={row['train/pose_error_mm']:.3f} "
                    f"val/pose_error_mm={row['val/pose_error_mm']:.3f} "
                    f"mean_baseline_error_mm={mean_baseline_error_mm:.3f} "
                    f"visual_encoder_grad_norm={last_encoder_grad_norm:.6f} "
                    f"pose_head_grad_norm={last_head_grad_norm:.6f} "
                    f"param_delta_norm={param_delta:.6f} "
                    f"pose_hat_std_mean={row['pose_hat_std_mean']:.6f}"
                )

    final_val = eval_model(model, val_loader, mean, std, device)
    final_param_delta = model_param_delta(model, initial_state)
    print("\n[acceptance]")
    print(f"val_error_mm={final_val['pose_error_mm']:.6f}")
    print(f"mean_baseline_error_mm={mean_baseline_error_mm:.6f}")
    print(f"val below mean baseline: {final_val['pose_error_mm'] < mean_baseline_error_mm}")
    print(f"pose_hat not constant: {final_val['pose_hat_std_mean'] > 1e-6}")
    print(f"parameters updated: {final_param_delta > 1e-8}")
    print(f"log csv: {csv_path}")

    th.save(
        {
            "model": model.state_dict(),
            "pos_mean": pos_mean_np,
            "pos_std": pos_std_np,
            "label_frame": args.label_frame,
        },
        output_dir / "visual_pose_encoder_debug.pt",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=str, default="./debug_pose_dataset/pose_debug_dataset.npz")
    parser.add_argument("--output-dir", type=str, default="./debug_pose_encoder")
    parser.add_argument("--recollect", action="store_true")
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-frame", choices=["tcp", "world"], default="tcp")
    parser.add_argument("--save-depth-count", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=10)

    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--camera-name", type=str, default="ee_cam")
    parser.add_argument("--max-depth", type=float, default=2.0)
    parser.add_argument("--control-dt", type=float, default=0.01)
    parser.add_argument("--physics-dt", type=float, default=0.001)
    parser.add_argument("--render-dt", type=float, default=0.006)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--ik-regularization", type=float, default=0.0001)
    parser.add_argument("--ik-radius", type=float, default=0.01)
    return parser.parse_args()


if __name__ == "__main__":
    main()
