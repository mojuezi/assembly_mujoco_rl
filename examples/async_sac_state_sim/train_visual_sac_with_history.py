#!/usr/bin/env python3
"""SB3 SAC baseline with depth + proprio + proprio history observation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from mujoco_env.mujoco_env.envs.env_instance import make_aubo_i5_assemble_hole_env
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
        }

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
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--save-freq", type=int, default=10_000)
    parser.add_argument("--save-path", type=str, default="./checkpoints/visual_sac_with_history")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-starts", type=int, default=1_000)

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

    model = SAC(
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
