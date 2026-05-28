#!/usr/bin/env python3
"""Minimal SB3 SAC baseline for Aubo peg insertion with depth + proprio input."""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MUJOCO_ENV_ROOT = PROJECT_ROOT / "mujoco-env"
for path in (MUJOCO_ENV_ROOT, PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import gymnasium as gym
import mujoco
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch import nn

from mujoco_env.mujoco_env.envs.env_instance import make_aubo_i5_assemble_hole_env

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


def _configure_quiet_imports() -> None:
    """Reduce third-party noise on the terminal (TensorFlow, Gym deprecation, etc.)."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")


class TrainLogger:
    """Lightweight structured console logger for training."""

    WIDTH = 72

    @staticmethod
    def _use_color() -> bool:
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    @classmethod
    def _c(cls, code: str, text: str) -> str:
        if not cls._use_color():
            return text
        return f"\033[{code}m{text}\033[0m"

    @classmethod
    def banner(cls, title: str) -> None:
        line = "═" * cls.WIDTH
        print(cls._c("1;36", line))
        print(cls._c("1;36", f"  {title}"))
        print(cls._c("1;36", line))

    @classmethod
    def section(cls, title: str) -> None:
        print()
        print(cls._c("1;34", f"▶ {title}"))
        print(cls._c("0;37", "─" * cls.WIDTH))

    @classmethod
    def kv(cls, key: str, value: Any, indent: int = 2) -> None:
        pad = " " * indent
        print(f"{pad}{key:<22} {value}")

    @classmethod
    def info(cls, message: str) -> None:
        print(cls._c("0;32", f"  ✓ {message}"))

    @classmethod
    def warn(cls, message: str) -> None:
        print(cls._c("0;33", f"  ! {message}"))

    @classmethod
    def progress(
        cls,
        timestep: int,
        total: int,
        fps: float,
        ep_rew_mean: float,
        ep_len_mean: float,
        n_episodes: int,
    ) -> None:
        pct = 100.0 * timestep / max(total, 1)
        bar_len = 28
        filled = int(bar_len * min(timestep / max(total, 1), 1.0))
        bar = "█" * filled + "░" * (bar_len - filled)
        rew = "n/a" if np.isnan(ep_rew_mean) else f"{ep_rew_mean:8.2f}"
        elen = "n/a" if np.isnan(ep_len_mean) else f"{ep_len_mean:6.1f}"
        print(
            cls._c(
                "0;36",
                f"  [{bar}] {pct:5.1f}%  "
                f"step {timestep:>9,d}/{total:<9,d}  "
                f"fps {fps:6.0f}  "
                f"ep_rew {rew}  ep_len {elen}  "
                f"episodes {n_episodes:4d}",
            )
        )

    @classmethod
    def done(cls, message: str) -> None:
        print()
        print(cls._c("1;32", f"  ✓ {message}"))
        print(cls._c("1;36", "═" * cls.WIDTH))


class TrainProgressCallback(BaseCallback):
    """Periodic one-line training summary (replaces noisy SB3 verbose spam)."""

    def __init__(self, total_timesteps: int, log_interval: int = 10_000):
        super().__init__(verbose=0)
        self.total_timesteps = int(total_timesteps)
        self.log_interval = max(1, int(log_interval))
        self._t0 = time.time()
        self._last_log_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_log_step < self.log_interval:
            return True
        self._last_log_step = self.num_timesteps

        elapsed = max(time.time() - self._t0, 1e-6)
        fps = self.num_timesteps / elapsed

        ep_rew_mean = float("nan")
        ep_len_mean = float("nan")
        n_episodes = 0
        if hasattr(self.model, "ep_info_buffer") and len(self.model.ep_info_buffer) > 0:
            n_episodes = len(self.model.ep_info_buffer)
            ep_rew_mean = float(np.mean([ep["r"] for ep in self.model.ep_info_buffer]))
            ep_len_mean = float(np.mean([ep["l"] for ep in self.model.ep_info_buffer]))

        TrainLogger.progress(
            timestep=self.num_timesteps,
            total=self.total_timesteps,
            fps=fps,
            ep_rew_mean=ep_rew_mean,
            ep_len_mean=ep_len_mean,
            n_episodes=n_episodes,
        )
        return True


def _shape_tree(obs: Any) -> Any:
    if isinstance(obs, dict):
        return {key: _shape_tree(value) for key, value in obs.items()}
    if hasattr(obs, "shape"):
        return tuple(obs.shape)
    return type(obs).__name__


def inspect_raw_observation_once(env: gym.Env) -> None:
    """Print the current environment reset/step observation structure."""
    TrainLogger.section("Raw env probe (rank 0)")
    obs, info = env.reset()
    TrainLogger.kv("reset obs", _shape_tree(obs))
    TrainLogger.kv("reset info.obs", _shape_tree(info.get("obs", {})))

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    TrainLogger.kv("step obs", _shape_tree(obs))
    TrainLogger.kv("step info.obs", _shape_tree(info.get("obs", {})))
    TrainLogger.kv(
        "step",
        f"action={tuple(action.shape)}, reward={float(reward):.4f}, "
        f"done={bool(terminated or truncated)}",
    )


class MultiModalObservationWrapper(gym.Wrapper):
    """Expose selectable proprio/vision observations for SB3 MultiInputPolicy.

    The wrapper deliberately does not put desired_goal/socket_pose_gt into the
    policy observation. The original env info is kept only for diagnostics.
    """

    def __init__(
        self,
        env: gym.Env,
        obs_mode: str = "depth_proprio",
        image_size: Tuple[int, int] = (64, 64),
        camera_name: str = "ee_cam",
        max_depth: float = 2.0,
        save_depth_dir: str | Path = "./debug_depth",
        save_depth_count: int = 8,
        show_depth: bool = False,
        show_rgb: bool = False,
        log_every: int = 1,
        enable_step_log: bool = False,
    ):
        super().__init__(env)
        valid_modes = {"proprio", "depth_proprio", "rgb_proprio", "rgbd_proprio"}
        if obs_mode not in valid_modes:
            raise ValueError(f"obs_mode must be one of {sorted(valid_modes)}, got {obs_mode}")
        self.obs_mode = obs_mode
        self.height, self.width = image_size
        self.camera_name = camera_name
        self.max_depth = float(max_depth)
        self.save_depth_dir = Path(save_depth_dir)
        self.save_depth_count = int(save_depth_count)
        self.show_depth = bool(show_depth)
        self.show_rgb = bool(show_rgb)
        self.log_every = max(1, int(log_every))
        self.enable_step_log = bool(enable_step_log)

        self.base_env = self.env.unwrapped
        self.rgb_renderer = None
        self.depth_renderer = None
        if self.uses_rgb:
            self.rgb_renderer = mujoco.Renderer(self.base_env.model, height=self.height, width=self.width)
        if self.uses_depth:
            self.depth_renderer = mujoco.Renderer(self.base_env.model, height=self.height, width=self.width)
            self.depth_renderer.enable_depth_rendering()
        self.camera_id = mujoco.mj_name2id(
            self.base_env.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name
        )
        if self.camera_id < 0:
            raise ValueError(f"Camera '{self.camera_name}' not found in MuJoCo model")

        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self._step_count = 0
        self._saved_depth = 0

        proprio_dim = self._get_proprio().shape[0]
        obs_spaces = {
            "proprio": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(proprio_dim,),
                dtype=np.float32,
            ),
        }
        if self.uses_depth:
            obs_spaces["depth"] = spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(1, self.height, self.width),
                    dtype=np.float32,
            )
        if self.uses_rgb:
            obs_spaces["rgb"] = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(3, self.height, self.width),
                dtype=np.float32,
            )
        self.observation_space = spaces.Dict(obs_spaces)

    @property
    def uses_depth(self) -> bool:
        return self.obs_mode in {"depth_proprio", "rgbd_proprio"}

    @property
    def uses_rgb(self) -> bool:
        return self.obs_mode in {"rgb_proprio", "rgbd_proprio"}

    def reset(self, **kwargs):
        raw_obs, info = self.env.reset(**kwargs)
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self._step_count = 0
        obs = self._build_obs(tag="reset")
        if self.enable_step_log and self._step_count == 0:
            self._log_shapes(obs, action=None, reward=None, done=False, tag="reset")
        return obs, info

    def step(self, action):
        raw_obs, reward, terminated, truncated, info = self.env.step(action)
        self.prev_action = np.asarray(action, dtype=np.float32).copy()
        self._step_count += 1
        done = bool(terminated or truncated)
        obs = self._build_obs(tag="step")

        if self.enable_step_log and (
            self._step_count <= 3 or self._step_count % self.log_every == 0 or done
        ):
            self._log_shapes(obs, action=action, reward=reward, done=done, tag="step")

        return obs, reward, terminated, truncated, info

    def close(self):
        if self.show_depth and cv2 is not None:
            cv2.destroyWindow("depth")
        if self.show_rgb and cv2 is not None:
            cv2.destroyWindow("rgb")
        if self.rgb_renderer is not None:
            self.rgb_renderer.close()
        if self.depth_renderer is not None:
            self.depth_renderer.close()
        return self.env.close()

    def _build_obs(self, tag: str) -> Dict[str, np.ndarray]:
        obs = {"proprio": self._get_proprio()}
        if self.uses_depth:
            obs["depth"] = self._render_depth(tag=tag)
        if self.uses_rgb:
            obs["rgb"] = self._render_rgb()
        return obs

    def _render_depth(self, tag: str) -> np.ndarray:
        if self.depth_renderer is None:
            raise RuntimeError("Depth renderer was not initialized for this obs_mode")
        self.depth_renderer.update_scene(self.base_env.data, camera=self.camera_id)
        depth_hw = self.depth_renderer.render().astype(np.float32)

        self._check_depth(depth_hw, tag=tag)

        depth_hw = np.nan_to_num(depth_hw, nan=self.max_depth, posinf=self.max_depth, neginf=0.0)
        depth_hw = np.clip(depth_hw, 0.0, self.max_depth) / self.max_depth
        depth_chw = depth_hw[None, :, :].astype(np.float32)

        if self._saved_depth < self.save_depth_count:
            self._save_depth_image(depth_chw, tag=tag)

        if self.show_depth:
            self._show_depth_image(depth_chw)

        return depth_chw

    def _render_rgb(self) -> np.ndarray:
        if self.rgb_renderer is None:
            raise RuntimeError("RGB renderer was not initialized for this obs_mode")
        self.rgb_renderer.update_scene(self.base_env.data, camera=self.camera_id)
        rgb_hwc = self.rgb_renderer.render().astype(np.float32) / 255.0
        if self.show_rgb:
            self._show_rgb_image(rgb_hwc)
        rgb_chw = np.transpose(rgb_hwc, (2, 0, 1)).astype(np.float32)
        return rgb_chw

    def _check_depth(self, depth_hw: np.ndarray, tag: str) -> None:
        has_nan = bool(np.isnan(depth_hw).any())
        has_inf = bool(np.isinf(depth_hw).any())
        finite = depth_hw[np.isfinite(depth_hw)]
        all_zero = bool(depth_hw.size > 0 and np.allclose(depth_hw, 0.0))
        all_constant = bool(finite.size > 0 and np.allclose(finite.min(), finite.max()))

        if finite.size > 0:
            min_depth = float(finite.min())
            max_depth = float(finite.max())
        else:
            min_depth = float("nan")
            max_depth = float("nan")

        if self.enable_step_log and (has_nan or has_inf or all_zero or all_constant):
            TrainLogger.warn(
                f"depth:{tag} nan={has_nan} inf={has_inf} zero={all_zero} "
                f"const={all_constant} min={min_depth:.4f} max={max_depth:.4f}"
            )

    def _save_depth_image(self, depth_chw: np.ndarray, tag: str) -> None:
        self.save_depth_dir.mkdir(parents=True, exist_ok=True)
        depth_hw = depth_chw[0]
        # Nearer pixels are brighter in the debug PNG for easier socket inspection.
        vis = ((1.0 - depth_hw) * 255.0).clip(0, 255).astype(np.uint8)
        if cv2 is not None:
            filename = self.save_depth_dir / f"{self._saved_depth:03d}_{tag}_depth.png"
            cv2.imwrite(str(filename), vis)
        else:
            filename = self.save_depth_dir / f"{self._saved_depth:03d}_{tag}_depth.pgm"
            with filename.open("wb") as f:
                f.write(f"P5\n{self.width} {self.height}\n255\n".encode("ascii"))
                f.write(vis.tobytes())
        if self.enable_step_log:
            TrainLogger.info(f"saved depth debug image: {filename}")
        self._saved_depth += 1

    def _show_depth_image(self, depth_chw: np.ndarray) -> None:
        if cv2 is None:
            print("[depth show] cv2 is not installed; cannot open depth window")
            self.show_depth = False
            return

        depth_hw = depth_chw[0]
        vis = ((1.0 - depth_hw) * 255.0).clip(0, 255).astype(np.uint8)
        vis = cv2.resize(vis, (256, 256), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("depth", vis)
        cv2.waitKey(1)

    def _show_rgb_image(self, rgb_hwc: np.ndarray) -> None:
        if cv2 is None:
            print("[rgb show] cv2 is not installed; cannot open rgb window")
            self.show_rgb = False
            return

        vis = (rgb_hwc * 255.0).clip(0, 255).astype(np.uint8)
        vis = cv2.resize(vis, (256, 256), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("rgb", vis[:, :, ::-1])
        cv2.waitKey(1)

    def _get_proprio(self) -> np.ndarray:
        qpos, qvel = self._get_joint_state()
        tcp_pos, tcp_quat = self._get_tcp_pose()
        tcp_lin_vel, tcp_ang_vel = self._get_tcp_velocity()

        proprio = np.concatenate(
            [
                qpos,
                qvel,
                tcp_pos,
                tcp_quat,
                tcp_lin_vel,
                tcp_ang_vel,
                self.prev_action.astype(np.float32).ravel(),
            ],
            axis=0,
        )
        return proprio.astype(np.float32)

    def _get_joint_state(self) -> Tuple[np.ndarray, np.ndarray]:
        dof = int(self.base_env.robot_config.dof)
        joint_names = getattr(self.base_env.robot_config, "joint_names", None)
        qpos = np.zeros(dof, dtype=np.float32)
        qvel = np.zeros(dof, dtype=np.float32)

        if joint_names is not None and len(joint_names) >= dof:
            for idx, name in enumerate(joint_names[:dof]):
                joint_id = mujoco.mj_name2id(self.base_env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if joint_id >= 0:
                    qpos_adr = self.base_env.model.jnt_qposadr[joint_id]
                    qvel_adr = self.base_env.model.jnt_dofadr[joint_id]
                    qpos[idx] = self.base_env.data.qpos[qpos_adr]
                    qvel[idx] = self.base_env.data.qvel[qvel_adr]
            return qpos, qvel

        return (
            self.base_env.data.qpos[:dof].astype(np.float32).copy(),
            self.base_env.data.qvel[:dof].astype(np.float32).copy(),
        )

    def _get_tcp_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        site_name = getattr(self.base_env.controller, "ee_site_name", "grip_site")
        site_id = mujoco.mj_name2id(self.base_env.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            return np.zeros(3, dtype=np.float32), np.array([1, 0, 0, 0], dtype=np.float32)

        tcp_pos = self.base_env.data.site_xpos[site_id].copy()
        tcp_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(tcp_quat, self.base_env.data.site_xmat[site_id])
        return tcp_pos.astype(np.float32), tcp_quat.astype(np.float32)

    def _get_tcp_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        site_name = getattr(self.base_env.controller, "ee_site_name", "grip_site")
        try:
            lin_vel = self.base_env.get_site_xvelp(site_name).astype(np.float32)
            ang_vel = self.base_env.get_site_xvelr(site_name).astype(np.float32)
            return lin_vel, ang_vel
        except Exception:
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    def _log_shapes(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray | None,
        reward: float | None,
        done: bool,
        tag: str,
    ) -> None:
        action_shape = None if action is None else tuple(np.asarray(action).shape)
        reward_text = "None" if reward is None else f"{float(reward):.4f}"
        parts = [f"{key}={tuple(value.shape)}" for key, value in obs.items()]
        TrainLogger.kv(
            f"env {tag}",
            f"{self.obs_mode} | "
            + ", ".join(parts)
            + f" | action={action_shape} reward={reward_text} done={done}",
            indent=4,
        )


DepthProprioObservationWrapper = MultiModalObservationWrapper


class MultiModalExtractor(BaseFeaturesExtractor):
    """CNN visual encoders + proprio MLP feature extractor for MultiInputPolicy."""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        proprio_dim = int(np.prod(observation_space["proprio"].shape))
        self.visual_keys = [key for key in ("depth", "rgb") if key in observation_space.spaces]
        self.visual_cnns = nn.ModuleDict()
        visual_feature_dims = []

        for key in self.visual_keys:
            visual_shape = observation_space[key].shape
            cnn = nn.Sequential(
                nn.Conv2d(visual_shape[0], 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            self.visual_cnns[key] = cnn
            with th.no_grad():
                sample_visual = th.zeros(1, *visual_shape)
                visual_feature_dims.append(cnn(sample_visual).shape[1])

        total_visual_features = int(sum(visual_feature_dims))

        self.proprio_mlp = nn.Sequential(
            nn.Linear(proprio_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(total_visual_features + 128, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        visual_features = [
            self.visual_cnns[key](observations[key])
            for key in self.visual_keys
        ]
        proprio_features = self.proprio_mlp(observations["proprio"])
        if len(visual_features) > 0:
            fused = th.cat([*visual_features, proprio_features], dim=1)
        else:
            fused = proprio_features
        return self.fusion(fused)


DepthProprioExtractor = MultiModalExtractor


def make_single_visual_env(args: argparse.Namespace, rank: int = 0) -> gym.Env:
    render_mode = "human" if getattr(args, "render_human", False) else None
    env = make_aubo_i5_assemble_hole_env(
        render_mode=render_mode,
        mode="sim",
        include_image=False,
        include_depth=False,
        control_dt=args.control_dt,
        physics_dt=args.physics_dt,
        render_dt=args.render_dt,
        max_episode_steps=args.max_episode_steps,
        ik_regularization=args.ik_regularization,
        ik_radius=args.ik_radius,
    )

    if args.inspect_raw_obs and rank == 0:
        inspect_raw_observation_once(env)

    env = DepthProprioObservationWrapper(
        env,
        obs_mode=args.obs_mode,
        image_size=(args.image_size, args.image_size),
        camera_name=args.camera_name,
        max_depth=args.max_depth,
        save_depth_dir=args.save_depth_dir,
        save_depth_count=args.save_depth_count,
        show_depth=args.show_depth,
        show_rgb=args.show_rgb,
        log_every=args.log_every,
        enable_step_log=getattr(args, "env_step_log", False) and rank == 0,
    )
    monitor_dir = Path(getattr(args, "monitor_dir", "./logs/visual_sac_baseline"))
    monitor_dir.mkdir(parents=True, exist_ok=True)
    env = Monitor(env, filename=str(monitor_dir / f"env_{rank}"))

    # Per-worker seed so parallel envs do not share identical trajectories.
    env_seed = int(getattr(args, "seed", 0)) + int(rank)
    env.reset(seed=env_seed)
    return env


def resolve_gradient_steps(n_envs: int, gradient_steps: int | None) -> int:
    """Match SB3 VecEnv sample/update ratio: n_envs transitions per vec step."""
    if gradient_steps is not None:
        return max(1, int(gradient_steps))
    return int(n_envs) if n_envs > 1 else 1


def make_visual_env(args: argparse.Namespace):
    n_envs = int(getattr(args, "n_envs", 1))
    if n_envs <= 1:
        return make_single_visual_env(args, rank=0)

    if args.render_human or args.show_depth or args.show_rgb:
        raise ValueError(
            "Parallel training does not support render_human/show_depth/show_rgb. "
            "Disable visual windows when using --n-envs > 1."
        )

    def make_env_fn(rank: int):
        def _init():
            # Suppress per-worker MuJoCo assembly logs (only rank 0 stays verbose).
            if rank > 0:
                with open(os.devnull, "w") as devnull:
                    with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                        return make_single_visual_env(args, rank=rank)
            return make_single_visual_env(args, rank=rank)

        return _init

    return SubprocVecEnv(
        [make_env_fn(rank) for rank in range(n_envs)],
        start_method=getattr(args, "vec_env_start_method", "forkserver"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-timesteps", type=int, default=1000_000)
    parser.add_argument("--save-freq", type=int, default=20_000)
    parser.add_argument("--save-path", type=str, default="./checkpoints/visual_sac_baseline_rgb")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed (env i uses seed+i).")
    parser.add_argument(
        "--gradient-steps",
        type=int,
        default=None,
        help="Gradient updates per vec step. Default: n_envs when n_envs>1 else 1.",
    )
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument(
        "--vec-env-start-method",
        type=str,
        default="forkserver",
        choices=["forkserver", "spawn", "fork"],
    )
    parser.add_argument("--monitor-dir", type=str, default="./logs/visual_sac_baseline_rgb")

    parser.add_argument(
        "--obs-mode",
        type=str,
        default="rgb_proprio",
        choices=["proprio", "depth_proprio", "rgb_proprio", "rgbd_proprio"],
        help=(
            "选择输入模式：proprio=单本体状态；depth_proprio=本体+深度；"
            "rgb_proprio=本体+RGB；rgbd_proprio=本体+RGBD。视觉输入均经过 CNN。"
        ),
    )
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--camera-name", type=str, default="ee_cam")
    parser.add_argument("--max-depth", type=float, default=2.0)
    parser.add_argument("--save-depth-dir", type=str, default="./debug_depth")
    parser.add_argument("--save-depth-count", type=int, default=0)
    parser.add_argument("--show-depth", action="store_true")
    parser.add_argument("--show-rgb", action="store_true")
    parser.add_argument("--log-every", type=int, default=100, help="Env step log interval when --env-step-log.")
    parser.add_argument(
        "--env-step-log",
        action="store_true",
        help="Print per-step env obs/reward logs (rank 0 only; off by default).",
    )
    parser.add_argument(
        "--train-log-interval",
        type=int,
        default=10_000,
        help="Print training progress every N total timesteps.",
    )
    parser.add_argument(
        "--sb3-verbose",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Stable-Baselines3 internal verbosity (0=quiet, use progress bar in script).",
    )
    parser.add_argument("--render-human", action="store_true")
    parser.add_argument(
        "--inspect-raw-obs",
        action="store_true",
        help="Run one raw-env reset/step shape probe on rank 0 before training.",
    )

    parser.add_argument("--control-dt", type=float, default=0.01)
    parser.add_argument("--physics-dt", type=float, default=0.001)
    parser.add_argument("--render-dt", type=float, default=0.006)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--ik-regularization", type=float, default=0.0001)
    parser.add_argument("--ik-radius", type=float, default=0.01)
    return parser.parse_args()


def _print_obs_space(space: spaces.Space, indent: int = 2) -> None:
    if isinstance(space, spaces.Dict):
        for key, sub in space.spaces.items():
            TrainLogger.kv(str(key), sub, indent=indent)
    else:
        TrainLogger.kv("space", space, indent=indent)


def main() -> None:
    _configure_quiet_imports()
    args = parse_args()
    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    TrainLogger.banner("Visual SAC Baseline")
    TrainLogger.section("Run configuration")
    TrainLogger.kv("obs_mode", args.obs_mode)
    TrainLogger.kv("image_size", f"{args.image_size}x{args.image_size}")
    TrainLogger.kv("n_envs", args.n_envs)
    TrainLogger.kv("total_timesteps", f"{args.total_timesteps:,d}")
    TrainLogger.kv("device", args.device)
    TrainLogger.kv("save_path", args.save_path)
    TrainLogger.kv("monitor_dir", args.monitor_dir)

    gradient_steps = resolve_gradient_steps(args.n_envs, args.gradient_steps)
    TrainLogger.section("SAC / parallel")
    TrainLogger.kv("gradient_steps", gradient_steps)
    TrainLogger.kv("buffer_size", f"{args.buffer_size:,d}")
    TrainLogger.kv("batch_size", args.batch_size)
    TrainLogger.kv("learning_starts", f"{args.learning_starts:,d}")
    TrainLogger.kv("seed", f"{args.seed} (env {args.seed}..{args.seed + max(args.n_envs - 1, 0)})")
    TrainLogger.kv("train_log_interval", f"{args.train_log_interval:,d}")
    if args.n_envs > 1:
        TrainLogger.info("parallel workers >0 run quiet during env construction")

    TrainLogger.section("Building environments")
    env = make_visual_env(args)
    TrainLogger.kv("observation_space", "")
    _print_obs_space(env.observation_space, indent=4)
    TrainLogger.kv("action_space", env.action_space)

    policy_kwargs = dict(
        features_extractor_class=MultiModalExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 256], qf=[256, 256]),
    )

    TrainLogger.section("Initializing SAC")
    model = SAC(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=args.sb3_verbose,
        train_freq=1,
        gradient_steps=gradient_steps,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        device=args.device,
        seed=args.seed,
    )
    TrainLogger.info("model ready")

    checkpoint_save_freq = max(args.save_freq // max(args.n_envs, 1), 1)
    callbacks: List[BaseCallback] = [
        TrainProgressCallback(
            total_timesteps=args.total_timesteps,
            log_interval=args.train_log_interval,
        ),
        CheckpointCallback(
            save_freq=checkpoint_save_freq,
            save_path=args.save_path,
            name_prefix="visual_sac_baseline",
            verbose=0,
        ),
    ]
    TrainLogger.section("Training")
    TrainLogger.kv("checkpoint_every", f"{args.save_freq:,d} env steps (~{checkpoint_save_freq} vec steps)")
    use_progress_bar = False
    try:
        import rich  # noqa: F401
        import tqdm  # noqa: F401

        use_progress_bar = True
    except ImportError:
        TrainLogger.warn("install tqdm + rich for a live progress bar (pip install tqdm rich)")

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=CallbackList(callbacks),
        progress_bar=use_progress_bar,
    )
    final_path = Path(args.save_path) / "visual_sac_baseline_final"
    model.save(str(final_path))
    env.close()
    TrainLogger.done(f"saved final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
