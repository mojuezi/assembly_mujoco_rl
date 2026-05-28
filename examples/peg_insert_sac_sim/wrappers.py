"""Observation wrappers for visual/proprio/force SAC inputs."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Dict, Tuple

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


BASE_OBS_MODES = {"proprio", "depth_proprio", "rgb_proprio", "rgbd_proprio"}
FORCE_OBS_MODES = {
    "force_proprio",
    "depth_force_proprio",
    "rgb_force_proprio",
    "rgbd_force_proprio",
}
ALL_OBS_MODES = BASE_OBS_MODES | FORCE_OBS_MODES


class MultiModalObservationWrapper(gym.Wrapper):
    """Expose selectable proprio/vision observations for SB3 MultiInputPolicy."""

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
        if obs_mode not in ALL_OBS_MODES:
            raise ValueError(f"obs_mode must be one of {sorted(ALL_OBS_MODES)}, got {obs_mode}")
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
        return self.obs_mode in {"depth_proprio", "rgbd_proprio", "depth_force_proprio", "rgbd_force_proprio"}

    @property
    def uses_rgb(self) -> bool:
        return self.obs_mode in {"rgb_proprio", "rgbd_proprio", "rgb_force_proprio", "rgbd_force_proprio"}

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
            print(
                f"[env warn] depth:{tag} nan={has_nan} inf={has_inf} zero={all_zero} "
                f"const={all_constant} min={min_depth:.4f} max={max_depth:.4f}"
            )

    def _save_depth_image(self, depth_chw: np.ndarray, tag: str) -> None:
        self.save_depth_dir.mkdir(parents=True, exist_ok=True)
        depth_hw = depth_chw[0]
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
            print(f"[env info] saved depth debug image: {filename}")
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
        print(
            f"[env {tag}] {self.obs_mode} | "
            + ", ".join(parts)
            + f" | action={action_shape} reward={reward_text} done={done}"
        )


class ForceProprioObservationWrapper(MultiModalObservationWrapper):
    """Add normalized wrench history and force phase features to SAC observations."""

    def __init__(
        self,
        env: gym.Env,
        wrench_history_len: int = 16,
        force_scale: float = 50.0,
        torque_scale: float = 5.0,
        contact_force_threshold: float = 3.0,
        jam_force_threshold: float = 15.0,
        **kwargs,
    ):
        obs_mode = kwargs.get("obs_mode", "force_proprio")
        if obs_mode not in FORCE_OBS_MODES:
            raise ValueError(f"ForceProprioObservationWrapper requires force obs_mode, got {obs_mode}")

        self.wrench_history_len = int(wrench_history_len)
        if self.wrench_history_len <= 0:
            raise ValueError("wrench_history_len must be positive")
        self.force_scale = float(force_scale)
        self.torque_scale = float(torque_scale)
        self.contact_force_threshold = float(contact_force_threshold)
        self.jam_force_threshold = float(jam_force_threshold)
        self._raw_wrench = np.zeros(6, dtype=np.float32)
        self._last_tcp_z = None
        self._z_progress = deque(maxlen=min(8, self.wrench_history_len))

        super().__init__(env, **kwargs)

        self.force_history = np.zeros((self.wrench_history_len, 6), dtype=np.float32)
        spaces_dict = dict(self.observation_space.spaces)
        spaces_dict["force"] = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(self.wrench_history_len, 6),
            dtype=np.float32,
        )
        spaces_dict["phase"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(5,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(spaces_dict)

    def reset(self, **kwargs):
        raw_obs, info = self.env.reset(**kwargs)
        info_obs = info.get("obs", {}) if isinstance(info, dict) else {}
        raw_for_phase = info_obs if isinstance(info_obs, dict) else raw_obs
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self._step_count = 0
        self._raw_wrench = self._extract_wrench(raw_obs, info)
        normalized = self._normalize_wrench(self._raw_wrench)
        self.force_history[:] = normalized[None, :]
        self._last_tcp_z = self._extract_tcp_z(raw_for_phase)
        self._z_progress.clear()
        obs = self._build_obs(tag="reset")
        assert obs["force"].shape == (self.wrench_history_len, 6)
        assert obs["phase"].shape == (5,)
        if self.enable_step_log and self._step_count == 0:
            self._log_shapes(obs, action=None, reward=None, done=False, tag="reset")
        return obs, info

    def step(self, action):
        raw_obs, reward, terminated, truncated, info = self.env.step(action)
        info_obs = info.get("obs", {}) if isinstance(info, dict) else {}
        raw_for_phase = info_obs if isinstance(info_obs, dict) else raw_obs
        self.prev_action = np.asarray(action, dtype=np.float32).copy()
        self._step_count += 1
        self._raw_wrench = self._extract_wrench(raw_obs, info)
        normalized = self._normalize_wrench(self._raw_wrench)
        self.force_history[:-1] = self.force_history[1:]
        self.force_history[-1] = normalized
        self._update_z_progress(raw_for_phase)

        done = bool(terminated or truncated)
        obs = self._build_obs(tag="step")
        if self.enable_step_log and (
            self._step_count <= 3 or self._step_count % self.log_every == 0 or done
        ):
            self._log_shapes(obs, action=action, reward=reward, done=done, tag="step")
        return obs, reward, terminated, truncated, info

    def _build_obs(self, tag: str) -> Dict[str, np.ndarray]:
        obs = super()._build_obs(tag=tag)
        obs["force"] = self.force_history.copy()
        obs["phase"] = self._get_phase().astype(np.float32)
        return obs

    def _extract_wrench(self, raw_obs: Dict[str, np.ndarray], info: Dict) -> np.ndarray:
        def _as_wrench6(value: np.ndarray) -> np.ndarray:
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            wrench = np.zeros(6, dtype=np.float32)
            wrench[: min(arr.size, 6)] = arr[:6]
            return wrench

        if isinstance(raw_obs, dict) and raw_obs.get("wrench") is not None:
            return _as_wrench6(raw_obs["wrench"])
        info_obs = info.get("obs", {}) if isinstance(info, dict) else {}
        if isinstance(info_obs, dict) and info_obs.get("wrench") is not None:
            return _as_wrench6(info_obs["wrench"])
        return np.zeros(6, dtype=np.float32)

    def _normalize_wrench(self, wrench: np.ndarray) -> np.ndarray:
        normalized = np.asarray(wrench, dtype=np.float32).copy()
        normalized[:3] /= max(self.force_scale, 1e-6)
        normalized[3:] /= max(self.torque_scale, 1e-6)
        return np.clip(normalized, -5.0, 5.0).astype(np.float32)

    def _extract_tcp_z(self, raw_obs: Dict[str, np.ndarray]) -> float | None:
        if isinstance(raw_obs, dict) and raw_obs.get("tcp_pos") is not None:
            tcp_pos = np.asarray(raw_obs["tcp_pos"], dtype=np.float32)
            if tcp_pos.size >= 3:
                return float(tcp_pos[2])
        return None

    def _update_z_progress(self, raw_obs: Dict[str, np.ndarray]) -> None:
        tcp_z = self._extract_tcp_z(raw_obs)
        if tcp_z is None:
            return
        if self._last_tcp_z is not None:
            self._z_progress.append(abs(float(self._last_tcp_z) - tcp_z))
        self._last_tcp_z = tcp_z

    def _insertion_progress(self) -> float:
        if not self._z_progress:
            return 0.0
        return float(np.sum(self._z_progress))

    def _get_phase(self) -> np.ndarray:
        force_norm = float(np.linalg.norm(self._raw_wrench[:3]))
        torque_norm = float(np.linalg.norm(self._raw_wrench[3:]))
        contact_flag = float(force_norm > self.contact_force_threshold)

        insertion_progress = self._insertion_progress()
        if self._z_progress:
            jam_flag = float(abs(float(self._raw_wrench[2])) > self.jam_force_threshold and insertion_progress < 1e-4)
        else:
            # TODO: Replace this first-step fallback with a task-specific insertion progress estimate.
            jam_flag = float(abs(float(self._raw_wrench[2])) > self.jam_force_threshold)

        return np.array(
            [
                contact_flag,
                jam_flag,
                np.clip(force_norm / max(self.force_scale, 1e-6), 0.0, 5.0),
                np.clip(torque_norm / max(self.torque_scale, 1e-6), 0.0, 5.0),
                insertion_progress,
            ],
            dtype=np.float32,
        )


DepthProprioObservationWrapper = MultiModalObservationWrapper
