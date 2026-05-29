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
from typing import Any, Dict, List

if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MUJOCO_ENV_ROOT = PROJECT_ROOT / "mujoco-env"
for path in (MUJOCO_ENV_ROOT, PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch import nn

from gated_fusion_models import GatedFusionExtractor
from mujoco_env.mujoco_env.envs.env_instance import make_aubo_i5_assemble_hole_env
from wrappers import (
    ALL_OBS_MODES,
    FORCE_OBS_MODES,
    DepthProprioObservationWrapper,
    ForceProprioObservationWrapper,
)


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


class MultiModalExtractor(BaseFeaturesExtractor):
    """CNN visual encoders + proprio/force MLP feature extractor for MultiInputPolicy."""

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
        fusion_dim = total_visual_features + 128

        self.force_mlp = None
        if "force" in observation_space.spaces:
            force_dim = int(np.prod(observation_space["force"].shape))
            self.force_mlp = nn.Sequential(
                nn.Linear(force_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
            )
            fusion_dim += 128

        self.phase_mlp = None
        if "phase" in observation_space.spaces:
            phase_dim = int(np.prod(observation_space["phase"].shape))
            self.phase_mlp = nn.Sequential(
                nn.Linear(phase_dim, 32),
                nn.ReLU(),
            )
            fusion_dim += 32

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        visual_features = [
            self.visual_cnns[key](observations[key])
            for key in self.visual_keys
        ]
        proprio_features = self.proprio_mlp(observations["proprio"])
        features = [*visual_features, proprio_features]
        if self.force_mlp is not None:
            features.append(self.force_mlp(th.flatten(observations["force"], start_dim=1)))
        if self.phase_mlp is not None:
            features.append(self.phase_mlp(th.flatten(observations["phase"], start_dim=1)))
        fused = th.cat(features, dim=1) if len(features) > 1 else features[0]
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

    wrapper_kwargs = dict(
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
    if args.obs_mode in FORCE_OBS_MODES:
        env = ForceProprioObservationWrapper(
            env,
            wrench_history_len=args.wrench_history_len,
            force_scale=args.force_scale,
            torque_scale=args.torque_scale,
            contact_force_threshold=args.contact_force_threshold,
            jam_force_threshold=args.jam_force_threshold,
            **wrapper_kwargs,
        )
    else:
        if args.use_wrench_history:
            raise ValueError(
                "--use-wrench-history requires one of the force obs modes: "
                f"{sorted(FORCE_OBS_MODES)}"
            )
        env = DepthProprioObservationWrapper(env, **wrapper_kwargs)

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


def build_policy_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """Select the SB3 feature extractor without changing the SAC wiring."""
    if args.fusion_arch == "concat":
        extractor_class = MultiModalExtractor
        extractor_kwargs = dict(features_dim=args.fusion_features_dim)
    elif args.fusion_arch == "gated":
        extractor_class = GatedFusionExtractor
        extractor_kwargs = dict(
            features_dim=args.fusion_features_dim,
            visual_latent_dim=args.visual_latent_dim,
            force_latent_dim=args.force_latent_dim,
            proprio_latent_dim=args.proprio_latent_dim,
            pred_latent_dim=args.pred_latent_dim,
            expert_dim=args.expert_dim,
            gate_temperature=args.gate_temperature,
            enable_predictive_encoder=args.enable_predictive_encoder,
            use_phase_gate_prior=args.use_phase_gate_prior,
        )
    else:
        raise ValueError(f"Unsupported fusion_arch={args.fusion_arch!r}")

    return dict(
        features_extractor_class=extractor_class,
        features_extractor_kwargs=extractor_kwargs,
        net_arch=dict(pi=[256, 256], qf=[256, 256]),
    )


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
    parser.add_argument("--save-path", type=str, default="./checkpoints/visual_sac_baseline_depth_force_proprio")
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
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument(
        "--vec-env-start-method",
        type=str,
        default="forkserver",
        choices=["forkserver", "spawn", "fork"],
    )
    parser.add_argument("--monitor-dir", type=str, default="./logs/visual_sac_baseline_depth_force_proprio")

    parser.add_argument(
        "--obs-mode",
        type=str,
        default="depth_force_proprio",
        choices=sorted(ALL_OBS_MODES),
        help=(
            "选择输入模式：proprio=单本体状态；depth_proprio=本体+深度；"
            "rgb_proprio=本体+RGB；force_proprio=本体+六维力历史；"
            "也支持 depth/rgb/rgbd + force_proprio 组合。视觉输入均经过 CNN。"
        ),
    )
    parser.add_argument(
        "--use-wrench-history",
        action="store_true",
        help="Enable wrench-history observations; use with force_* obs modes.",
    )
    parser.add_argument("--wrench-history-len", type=int, default=16)
    parser.add_argument("--force-scale", type=float, default=50.0)
    parser.add_argument("--torque-scale", type=float, default=5.0)
    parser.add_argument("--contact-force-threshold", type=float, default=3.0)
    parser.add_argument("--jam-force-threshold", type=float, default=15.0)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--camera-name", type=str, default="ee_cam")
    parser.add_argument("--max-depth", type=float, default=2.0)
    parser.add_argument("--save-depth-dir", type=str, default="./debug_depth")
    parser.add_argument("--save-depth-count", type=int, default=0)
    parser.add_argument("--show-depth", action="store_true")
    parser.add_argument("--show-rgb", action="store_true")
    parser.add_argument(
        "--fusion-arch",
        type=str,
        default="gated",
        choices=["concat", "gated"],
        help="Feature fusion architecture: original concat extractor or gated multimodal extractor.",
    )
    parser.add_argument("--fusion-features-dim", type=int, default=256)
    parser.add_argument("--visual-latent-dim", type=int, default=128)
    parser.add_argument("--force-latent-dim", type=int, default=128)
    parser.add_argument("--proprio-latent-dim", type=int, default=64)
    parser.add_argument("--pred-latent-dim", type=int, default=64)
    parser.add_argument("--expert-dim", type=int, default=128)
    parser.add_argument("--gate-temperature", type=float, default=1.0)
    parser.add_argument(
        "--enable-predictive-encoder",
        action="store_true",
        default=True,
        help="Enable the predictive expert in gated fusion (enabled by default).",
    )
    parser.add_argument(
        "--use-phase-gate-prior",
        action="store_true",
        default=True,
        help="Add phase-based prior logits to the gated fusion gate (enabled by default).",
    )
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
    if args.obs_mode in FORCE_OBS_MODES:
        args.use_wrench_history = True
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
    TrainLogger.kv("fusion_arch", args.fusion_arch)
    TrainLogger.kv("fusion_features_dim", args.fusion_features_dim)
    TrainLogger.kv("visual_latent_dim", args.visual_latent_dim)
    TrainLogger.kv("force_latent_dim", args.force_latent_dim)
    TrainLogger.kv("proprio_latent_dim", args.proprio_latent_dim)
    TrainLogger.kv("pred_latent_dim", args.pred_latent_dim)
    TrainLogger.kv("expert_dim", args.expert_dim)
    TrainLogger.kv("gate_temperature", args.gate_temperature)
    TrainLogger.kv("enable_predictive_encoder", args.enable_predictive_encoder)
    TrainLogger.kv("use_phase_gate_prior", args.use_phase_gate_prior)
    if args.obs_mode in FORCE_OBS_MODES or args.use_wrench_history:
        TrainLogger.kv("use_wrench_history", args.use_wrench_history)
        TrainLogger.kv("wrench_history_len", args.wrench_history_len)
        TrainLogger.kv("force_scale", args.force_scale)
        TrainLogger.kv("torque_scale", args.torque_scale)

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

    policy_kwargs = build_policy_kwargs(args)

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
