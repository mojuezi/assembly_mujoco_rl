#!/usr/bin/env python3
"""SB3 SAC baseline with depth + proprio + proprio history observation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MUJOCO_ENV_ROOT = PROJECT_ROOT / "mujoco-env"
for path in (MUJOCO_ENV_ROOT, PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from mujoco_env.mujoco_env.envs.env_instance import make_aubo_i5_assemble_hole_env
from networks.visual_pose_encoder import VisualPoseEncoder
from train_visual_sac_baseline import (
    DepthProprioObservationWrapper,
    inspect_raw_observation_once,
)


class DepthProprioHistoryObservationWrapper(DepthProprioObservationWrapper):
    """Add a fixed-length proprio history buffer to the visual observation."""

    def __init__(
        self,
        env: gym.Env,
        history_len: int = 5,
        **kwargs,
    ):
        self.history_len = int(history_len)
        if self.history_len <= 0:
            raise ValueError("history_len must be positive")

        super().__init__(env, **kwargs)

        proprio_dim = int(np.prod(self.observation_space["proprio"].shape))
        self.proprio_dim = proprio_dim
        self.history_buffer = np.zeros((self.history_len, proprio_dim), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "depth": self.observation_space["depth"],
                "proprio": self.observation_space["proprio"],
                "history": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.history_len, proprio_dim),
                    dtype=np.float32,
                ),
                "socket_relative_pos_gt": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(3,),
                    dtype=np.float32,
                ),
            }
        )

    def reset(self, **kwargs):
        raw_obs, info = self.env.reset(**kwargs)
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self._step_count = 0

        proprio = self._get_proprio()
        self.history_buffer[:] = proprio[None, :]
        obs = self._build_obs_from_proprio(proprio=proprio, tag="reset")
        self._log_shapes(obs, action=None, reward=None, done=False, tag="reset")
        return obs, info

    def step(self, action):
        raw_obs, reward, terminated, truncated, info = self.env.step(action)
        self.prev_action = np.asarray(action, dtype=np.float32).copy()
        self._step_count += 1

        proprio = self._get_proprio()
        self.history_buffer[:-1] = self.history_buffer[1:]
        self.history_buffer[-1] = proprio

        done = bool(terminated or truncated)
        obs = self._build_obs_from_proprio(proprio=proprio, tag="step")

        if self._step_count <= 5 or self._step_count % self.log_every == 0 or done:
            self._log_shapes(obs, action=action, reward=reward, done=done, tag="step")

        return obs, reward, terminated, truncated, info

    def _build_obs_from_proprio(self, proprio: np.ndarray, tag: str) -> Dict[str, np.ndarray]:
        depth = self._render_depth(tag=tag)
        return {
            "depth": depth,
            "proprio": proprio.astype(np.float32),
            "history": self.history_buffer.copy(),
            "socket_relative_pos_gt": self._get_socket_relative_pos_gt(),
        }

    def _get_socket_relative_pos_gt(self) -> np.ndarray:
        """Return socket position relative to TCP, expressed in TCP frame."""
        socket_pos = getattr(self.base_env, "_desired_goal", None)
        if socket_pos is None and getattr(self.base_env, "task", None) is not None:
            socket_pos = getattr(self.base_env.task, "hole_position", None)
        if socket_pos is None:
            return np.zeros(3, dtype=np.float32)

        tcp_pos, _ = self._get_tcp_pose()
        rel_world = np.asarray(socket_pos[:3], dtype=np.float32) - tcp_pos

        site_name = getattr(self.base_env.controller, "ee_site_name", "grip_site")
        try:
            tcp_rotm = self.base_env.get_site_rotm(site_name).astype(np.float32)
            rel_tcp = tcp_rotm.T @ rel_world
        except Exception:
            rel_tcp = rel_world

        return rel_tcp.astype(np.float32)

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
        flattened_history_shape = (int(np.prod(obs["history"].shape)),)
        print(
            f"[{tag}] depth shape={obs['depth'].shape}, "
            f"proprio shape={obs['proprio'].shape}, "
            f"history shape={obs['history'].shape}, "
            f"flattened history shape={flattened_history_shape}, "
            f"action shape={action_shape}, reward={reward_text}, done={done}"
        )


class DepthProprioHistoryExtractor(BaseFeaturesExtractor):
    """Depth CNN + proprio MLP + flattened-history MLP feature extractor."""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        depth_shape = observation_space["depth"].shape
        proprio_dim = int(np.prod(observation_space["proprio"].shape))
        history_shape = observation_space["history"].shape
        history_flat_dim = int(np.prod(history_shape))
        self._logged_shapes = False

        self.depth_cnn = nn.Sequential(
            nn.Conv2d(depth_shape[0], 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample_depth = th.zeros(1, *depth_shape)
            depth_features = self.depth_cnn(sample_depth).shape[1]

        self.proprio_mlp = nn.Sequential(
            nn.Linear(proprio_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.history_mlp = nn.Sequential(
            nn.Linear(history_flat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(depth_features + 128 + 128, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        depth_features = self.depth_cnn(observations["depth"])
        proprio_features = self.proprio_mlp(observations["proprio"])
        flat_history = th.flatten(observations["history"], start_dim=1)
        history_features = self.history_mlp(flat_history)
        final_features = self.fusion(
            th.cat([depth_features, proprio_features, history_features], dim=1)
        )

        if not self._logged_shapes:
            print(
                "[feature extractor] "
                f"depth shape={tuple(observations['depth'].shape)}, "
                f"proprio shape={tuple(observations['proprio'].shape)}, "
                f"history shape={tuple(observations['history'].shape)}, "
                f"flattened history shape={tuple(flat_history.shape)}, "
                f"final feature shape={tuple(final_features.shape)}"
            )
            self._logged_shapes = True

        return final_features


class AuxSAC(SAC):
    """Minimal SAC variant with an auxiliary depth-to-relative-pose loss."""

    def __init__(
        self,
        *args,
        pose_loss_coef: float = 1.0,
        pose_lr: float = 3e-4,
        **kwargs,
    ):
        self.pose_loss_coef = float(pose_loss_coef)
        self.pose_lr = float(pose_lr)
        self.visual_pose_encoder = None
        self.visual_pose_optimizer = None
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        if not isinstance(self.observation_space, spaces.Dict):
            raise TypeError("AuxSAC requires a Dict observation space")

        depth_shape = self.observation_space["depth"].shape
        self.visual_pose_encoder = VisualPoseEncoder(depth_shape=depth_shape).to(self.device)
        self.visual_pose_optimizer = th.optim.Adam(
            self.visual_pose_encoder.parameters(),
            lr=self.pose_lr,
        )

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        super().train(gradient_steps=gradient_steps, batch_size=batch_size)

        if self.visual_pose_encoder is None or self.visual_pose_optimizer is None:
            return

        pose_losses = []
        pose_errors = []
        pose_axis_errors = []
        self.visual_pose_encoder.train()

        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(  # type: ignore[union-attr]
                batch_size,
                env=self._vec_normalize_env,
            )
            observations = replay_data.observations
            if "socket_relative_pos_gt" not in observations:
                continue

            depth = observations["depth"]
            pose_gt = observations["socket_relative_pos_gt"]
            pose_hat, _ = self.visual_pose_encoder(depth)
            pose_loss = F.mse_loss(pose_hat, pose_gt)

            self.visual_pose_optimizer.zero_grad()
            (self.pose_loss_coef * pose_loss).backward()
            self.visual_pose_optimizer.step()

            with th.no_grad():
                abs_error = th.abs(pose_hat - pose_gt)
                pose_losses.append(float(pose_loss.item()))
                pose_errors.append(float(th.linalg.norm(pose_hat - pose_gt, dim=1).mean().item()))
                pose_axis_errors.append(abs_error.mean(dim=0).detach().cpu().numpy())

        if len(pose_losses) == 0:
            return

        mean_axis_error = np.mean(np.stack(pose_axis_errors, axis=0), axis=0)
        self.logger.record("train/pose_loss", float(np.mean(pose_losses)))
        self.logger.record("train/pose_error_mean", float(np.mean(pose_errors)))
        self.logger.record("train/pose_error_x", float(mean_axis_error[0]))
        self.logger.record("train/pose_error_y", float(mean_axis_error[1]))
        self.logger.record("train/pose_error_z", float(mean_axis_error[2]))

    def _get_torch_save_params(self):
        state_dicts, saved_pytorch_variables = super()._get_torch_save_params()
        state_dicts += ["visual_pose_encoder", "visual_pose_optimizer"]
        return state_dicts, saved_pytorch_variables


def make_visual_history_env(args: argparse.Namespace) -> gym.Env:
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

    if args.inspect_raw_obs:
        inspect_raw_observation_once(env)

    env = DepthProprioHistoryObservationWrapper(
        env,
        history_len=args.history_len,
        image_size=(args.image_size, args.image_size),
        camera_name=args.camera_name,
        max_depth=args.max_depth,
        save_depth_dir=args.save_depth_dir,
        save_depth_count=args.save_depth_count,
        show_depth=args.show_depth,
        log_every=args.log_every,
    )
    return Monitor(env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-timesteps", type=int, default=1000_000)
    parser.add_argument("--save-freq", type=int, default=10_000)
    parser.add_argument("--save-path", type=str, default="./checkpoints/visual_sac_with_history_v2")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--pose-loss-coef", type=float, default=1.0)
    parser.add_argument("--pose-lr", type=float, default=3e-4)

    parser.add_argument("--history-len", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--camera-name", type=str, default="ee_cam")
    parser.add_argument("--max-depth", type=float, default=2.0)
    parser.add_argument("--save-depth-dir", type=str, default="./debug_depth_history")
    parser.add_argument("--save-depth-count", type=int, default=0)
    parser.add_argument("--show-depth", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--render-human", action="store_true")
    parser.add_argument("--no-inspect-raw-obs", dest="inspect_raw_obs", action="store_false")
    parser.set_defaults(inspect_raw_obs=True)

    parser.add_argument("--control-dt", type=float, default=0.01)
    parser.add_argument("--physics-dt", type=float, default=0.001)
    parser.add_argument("--render-dt", type=float, default=0.006)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--ik-regularization", type=float, default=0.0001)
    parser.add_argument("--ik-radius", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    env = make_visual_history_env(args)
    print(f"[env] observation_space={env.observation_space}")
    print(f"[env] action_space={env.action_space}")

    policy_kwargs = dict(
        features_extractor_class=DepthProprioHistoryExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 256], qf=[256, 256]),
    )

    model = AuxSAC(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        train_freq=1,
        gradient_steps=1,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        device=args.device,
        pose_loss_coef=args.pose_loss_coef,
        pose_lr=args.pose_lr,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=args.save_path,
        name_prefix="visual_sac_with_history",
    )
    model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_callback)
    model.save(str(Path(args.save_path) / "visual_sac_with_history_final"))
    env.close()


if __name__ == "__main__":
    main()
