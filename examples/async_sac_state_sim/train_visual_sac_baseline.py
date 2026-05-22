#!/usr/bin/env python3
"""Minimal SB3 SAC baseline for Aubo peg insertion with depth + proprio input."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import gymnasium as gym
import mujoco
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from mujoco_env.mujoco_env.envs.env_instance import make_aubo_i5_assemble_hole_env

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


def _shape_tree(obs: Any) -> Any:
    if isinstance(obs, dict):
        return {key: _shape_tree(value) for key, value in obs.items()}
    if hasattr(obs, "shape"):
        return tuple(obs.shape)
    return type(obs).__name__


def inspect_raw_observation_once(env: gym.Env) -> None:
    """Print the current environment reset/step observation structure."""
    obs, info = env.reset()
    print(f"[raw reset] obs shapes: {_shape_tree(obs)}")
    print(f"[raw reset] info obs shapes: {_shape_tree(info.get('obs', {}))}")

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"[raw step] obs shapes: {_shape_tree(obs)}")
    print(f"[raw step] info obs shapes: {_shape_tree(info.get('obs', {}))}")
    print(
        "[raw step] "
        f"action shape={action.shape}, reward={float(reward):.4f}, "
        f"done={bool(terminated or truncated)}"
    )


class DepthProprioObservationWrapper(gym.Wrapper):
    """Expose obs as {'depth': [1, H, W], 'proprio': low-dimensional robot state}.

    The wrapper deliberately does not put desired_goal/socket_pose_gt into the
    policy observation. The original env info is kept only for diagnostics.
    """

    def __init__(
        self,
        env: gym.Env,
        image_size: Tuple[int, int] = (64, 64),
        camera_name: str = "ee_cam",
        max_depth: float = 2.0,
        save_depth_dir: str | Path = "./debug_depth",
        save_depth_count: int = 8,
        log_every: int = 1,
    ):
        super().__init__(env)
        self.height, self.width = image_size
        self.camera_name = camera_name
        self.max_depth = float(max_depth)
        self.save_depth_dir = Path(save_depth_dir)
        self.save_depth_count = int(save_depth_count)
        self.log_every = max(1, int(log_every))

        self.base_env = self.env.unwrapped
        self.renderer = mujoco.Renderer(self.base_env.model, height=self.height, width=self.width)
        self.renderer.enable_depth_rendering()
        self.camera_id = mujoco.mj_name2id(
            self.base_env.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name
        )
        if self.camera_id < 0:
            raise ValueError(f"Camera '{self.camera_name}' not found in MuJoCo model")

        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self._step_count = 0
        self._saved_depth = 0

        proprio_dim = self._get_proprio().shape[0]
        self.observation_space = spaces.Dict(
            {
                "depth": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(1, self.height, self.width),
                    dtype=np.float32,
                ),
                "proprio": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(proprio_dim,),
                    dtype=np.float32,
                ),
            }
        )

    def reset(self, **kwargs):
        raw_obs, info = self.env.reset(**kwargs)
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self._step_count = 0
        obs = self._build_obs(tag="reset")
        self._log_shapes(obs, action=None, reward=None, done=False, tag="reset")
        return obs, info

    def step(self, action):
        raw_obs, reward, terminated, truncated, info = self.env.step(action)
        self.prev_action = np.asarray(action, dtype=np.float32).copy()
        self._step_count += 1
        done = bool(terminated or truncated)
        obs = self._build_obs(tag="step")

        if self._step_count <= 5 or self._step_count % self.log_every == 0 or done:
            self._log_shapes(obs, action=action, reward=reward, done=done, tag="step")

        return obs, reward, terminated, truncated, info

    def close(self):
        if hasattr(self, "renderer"):
            self.renderer.close()
        return self.env.close()

    def _build_obs(self, tag: str) -> Dict[str, np.ndarray]:
        depth = self._render_depth(tag=tag)
        proprio = self._get_proprio()
        return {"depth": depth, "proprio": proprio}

    def _render_depth(self, tag: str) -> np.ndarray:
        self.renderer.update_scene(self.base_env.data, camera=self.camera_id)
        depth_hw = self.renderer.render().astype(np.float32)

        self._check_depth(depth_hw, tag=tag)

        depth_hw = np.nan_to_num(depth_hw, nan=self.max_depth, posinf=self.max_depth, neginf=0.0)
        depth_hw = np.clip(depth_hw, 0.0, self.max_depth) / self.max_depth
        depth_chw = depth_hw[None, :, :].astype(np.float32)

        if self._saved_depth < self.save_depth_count:
            self._save_depth_image(depth_chw, tag=tag)

        return depth_chw

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

        if has_nan or has_inf or all_zero or all_constant:
            print(
                f"[depth check:{tag}] nan={has_nan}, inf={has_inf}, "
                f"all_zero={all_zero}, all_constant={all_constant}, "
                f"min={min_depth:.4f}, max={max_depth:.4f}"
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
        print(f"[depth save] {filename}")
        self._saved_depth += 1

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
        print(
            f"[{tag}] depth shape={obs['depth'].shape}, "
            f"proprio shape={obs['proprio'].shape}, "
            f"action shape={action_shape}, reward={reward_text}, done={done}"
        )


class DepthProprioExtractor(BaseFeaturesExtractor):
    """Small depth CNN + proprio MLP feature extractor for MultiInputPolicy."""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        depth_shape = observation_space["depth"].shape
        proprio_dim = int(np.prod(observation_space["proprio"].shape))

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
        self.fusion = nn.Sequential(
            nn.Linear(depth_features + 128, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        depth_features = self.depth_cnn(observations["depth"])
        proprio_features = self.proprio_mlp(observations["proprio"])
        return self.fusion(th.cat([depth_features, proprio_features], dim=1))


def make_visual_env(args: argparse.Namespace) -> gym.Env:
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

    env = DepthProprioObservationWrapper(
        env,
        image_size=(args.image_size, args.image_size),
        camera_name=args.camera_name,
        max_depth=args.max_depth,
        save_depth_dir=args.save_depth_dir,
        save_depth_count=args.save_depth_count,
        log_every=args.log_every,
    )
    return Monitor(env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--save-freq", type=int, default=10_000)
    parser.add_argument("--save-path", type=str, default="./checkpoints/visual_sac_baseline")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-starts", type=int, default=1_000)

    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--camera-name", type=str, default="ee_cam")
    parser.add_argument("--max-depth", type=float, default=2.0)
    parser.add_argument("--save-depth-dir", type=str, default="./debug_depth")
    parser.add_argument("--save-depth-count", type=int, default=8)
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

    env = make_visual_env(args)
    print(f"[env] observation_space={env.observation_space}")
    print(f"[env] action_space={env.action_space}")

    policy_kwargs = dict(
        features_extractor_class=DepthProprioExtractor,
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
        name_prefix="visual_sac_baseline",
    )
    model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_callback)
    model.save(str(Path(args.save_path) / "visual_sac_baseline_final"))
    env.close()


if __name__ == "__main__":
    main()
