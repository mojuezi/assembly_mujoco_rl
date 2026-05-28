#!/usr/bin/env python3
"""Evaluate a checkpoint from train_visual_sac_baseline.py.

obs_mode / image_size must match training, or SAC.load will fail on observation_space.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from stable_baselines3 import SAC

from train_visual_sac_baseline import make_visual_env
from wrappers import ALL_OBS_MODES, FORCE_OBS_MODES


@dataclass
class EpisodeMetrics:
    """Terminal-state diagnostics aligned with PegInsertionTask success criteria."""

    xy_distance_m: float = float("nan")
    z_above_m: float = float("nan")
    xy_ok: bool = False
    z_ok: bool = False
    min_xy_distance_m: float = float("nan")
    tcp_pos: Optional[np.ndarray] = None
    desired_goal: Optional[np.ndarray] = None

    @property
    def xy_distance_mm(self) -> float:
        return self.xy_distance_m * 1000.0

    @property
    def z_above_mm(self) -> float:
        return self.z_above_m * 1000.0

    @property
    def min_xy_distance_mm(self) -> float:
        return self.min_xy_distance_m * 1000.0


@dataclass
class EpisodeResult:
    success: bool
    terminated: bool
    truncated: bool
    reward: float
    length: int
    last_step_reward: float = 0.0
    failure: str = ""
    metrics: EpisodeMetrics = field(default_factory=EpisodeMetrics)


@dataclass
class EvalSummary:
    n_episodes: int = 0
    n_success: int = 0
    n_timeout: int = 0
    n_collision: int = 0
    n_out_of_workspace: int = 0
    n_other_fail: int = 0
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.n_episodes == 0:
            return 0.0
        return self.n_success / self.n_episodes

    def record(self, result: EpisodeResult) -> None:
        self.n_episodes += 1
        self.episode_rewards.append(result.reward)
        self.episode_lengths.append(result.length)
        if result.success:
            self.n_success += 1
        elif result.truncated and not result.terminated:
            self.n_timeout += 1
        elif result.failure == "collision":
            self.n_collision += 1
        elif result.failure == "out_of_workspace":
            self.n_out_of_workspace += 1
        else:
            self.n_other_fail += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate visual SAC baseline")
    parser.add_argument(
        "--obs-mode",
        type=str,
        default="force_proprio",
        choices=sorted(ALL_OBS_MODES),
        help="Must match the mode used during training",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./checkpoints/visual_sac_baseline_rgb/visual_sac_baseline_800000_steps",
    )
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument(
        "--use-wrench-history",
        action="store_true",
        help="Enable wrench-history observations; automatically enabled for force_* obs modes.",
    )
    parser.add_argument("--wrench-history-len", type=int, default=16)
    parser.add_argument("--force-scale", type=float, default=50.0)
    parser.add_argument("--torque-scale", type=float, default=5.0)
    parser.add_argument("--contact-force-threshold", type=float, default=3.0)
    parser.add_argument("--jam-force-threshold", type=float, default=15.0)
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode.")
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base env seed (episode i uses seed+i).")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--render-human", action="store_false") #默认测试渲染窗口是打开的
    parser.add_argument("--show-rgb", action="store_true")
    parser.add_argument("--show-depth", action="store_true")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print final summary (no per-episode report).",
    )
    return parser.parse_args()


def build_args(ns: argparse.Namespace):
    from argparse import Namespace

    use_wrench_history = bool(ns.use_wrench_history or ns.obs_mode in FORCE_OBS_MODES)
    return Namespace(
        control_dt=0.01,
        physics_dt=0.001,
        render_dt=0.006,
        max_episode_steps=ns.max_steps,
        ik_regularization=0.0001,
        ik_radius=0.01,
        obs_mode=ns.obs_mode,
        image_size=ns.image_size,
        use_wrench_history=use_wrench_history,
        wrench_history_len=ns.wrench_history_len,
        force_scale=ns.force_scale,
        torque_scale=ns.torque_scale,
        contact_force_threshold=ns.contact_force_threshold,
        jam_force_threshold=ns.jam_force_threshold,
        camera_name="ee_cam",
        max_depth=2.0,
        save_depth_dir="./debug_depth_eval",
        save_depth_count=0,
        show_depth=ns.show_depth,
        show_rgb=ns.show_rgb,
        log_every=10_000,
        render_human=ns.render_human,
        inspect_raw_obs=False,
        n_envs=1,
        seed=ns.seed,
        env_step_log=False,
    )


def _unwrap_env(env: Any) -> Any:
    base = env
    for _ in range(16):
        if hasattr(base, "env"):
            base = base.env
        else:
            break
    return base


def _use_color() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    if not _use_color():
        return text
    return f"\033[{code}m{text}\033[0m"


def classify_episode(
    *,
    terminated: bool,
    truncated: bool,
    last_reward: float,
    success: bool,
) -> str:
    if success:
        return ""
    if truncated and not terminated:
        return "timeout"
    if terminated and last_reward <= -90.0:
        return "out_of_workspace"
    if terminated and last_reward <= -40.0:
        return "collision"
    if terminated:
        return "terminated_other"
    return "unknown"


def compute_metrics_from_obs(env: Any, raw_obs: Dict[str, Any]) -> EpisodeMetrics:
    base = _unwrap_env(env)
    task = getattr(base, "task", None)
    metrics = EpisodeMetrics()
    if task is None or raw_obs is None:
        return metrics

    tcp = np.asarray(raw_obs.get("tcp_pos", np.zeros(3)), dtype=np.float32)[:3]
    goal = np.asarray(raw_obs.get("desired_goal", np.zeros(3)), dtype=np.float32)[:3]
    metrics.tcp_pos = tcp
    metrics.desired_goal = goal

    metrics.xy_distance_m = float(np.linalg.norm(tcp[:2] - goal[:2]))
    metrics.z_above_m = float(tcp[2] - goal[2])
    metrics.xy_ok = metrics.xy_distance_m < float(task.xy_success_tolerance)
    metrics.z_ok = float(task.z_insert_min) <= metrics.z_above_m <= float(task.z_insert_max)
    return metrics


def is_task_success(env: Any, info: Dict[str, Any]) -> bool:
    raw_obs = info.get("obs")
    if raw_obs is None:
        return False
    metrics = compute_metrics_from_obs(env, raw_obs)
    return bool(metrics.xy_ok and metrics.z_ok)


def failure_hints(result: EpisodeResult, task: Any) -> List[str]:
    """Short human-readable hints for what went wrong."""
    hints: List[str] = []
    m = result.metrics

    if result.success:
        hints.append("insertion criteria met (XY & Z within tolerance)")
        return hints

    if result.failure == "collision":
        hints.append("episode ended on collision penalty (hard contact / bad approach)")
    elif result.failure == "out_of_workspace":
        hints.append("TCP left allowed workspace bounds")
    elif result.failure == "timeout":
        hints.append("reached max steps without success termination")

    if task is not None:
        xy_tol_mm = float(task.xy_success_tolerance) * 1000.0
        z_min_mm = float(task.z_insert_min) * 1000.0
        z_max_mm = float(task.z_insert_max) * 1000.0

        if not m.xy_ok:
            hints.append(
                f"XY misaligned: {m.xy_distance_mm:.2f} mm from hole center "
                f"(need < {xy_tol_mm:.2f} mm)"
            )
        if not m.z_ok:
            if m.z_above_mm > z_max_mm:
                hints.append(
                    f"TCP too high: Z offset {m.z_above_mm:+.2f} mm "
                    f"(target band [{z_min_mm:+.2f}, {z_max_mm:+.2f}] mm)"
                )
            elif m.z_above_mm < z_min_mm:
                hints.append(
                    f"TCP too low / overshoot: Z offset {m.z_above_mm:+.2f} mm "
                    f"(target band [{z_min_mm:+.2f}, {z_max_mm:+.2f}] mm)"
                )
            else:
                hints.append(f"Z offset {m.z_above_mm:+.2f} mm outside success band")

        if not np.isnan(m.min_xy_distance_mm) and m.min_xy_distance_mm > xy_tol_mm:
            hints.append(
                f"closest XY during episode: {m.min_xy_distance_mm:.2f} mm "
                f"(never entered {xy_tol_mm:.2f} mm tolerance)"
            )

    if not hints:
        hints.append("see terminated/truncated flags and last-step reward")
    return hints


def _bar(ok: bool, width: int = 12) -> str:
    filled = width if ok else max(1, width // 4)
    ch = "█" if ok else "░"
    return ch * filled


def run_episode(
    env: Any,
    model: SAC,
    *,
    max_steps: int,
    seed: int,
    deterministic: bool = True,
) -> EpisodeResult:
    obs, info = env.reset(seed=seed)
    episode_reward = 0.0
    last_reward = 0.0
    terminated = False
    truncated = False
    length = 0
    min_xy = float("inf")

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        last_reward = float(reward)
        episode_reward += last_reward
        length += 1

        raw_obs = info.get("obs")
        if raw_obs is not None and "tcp_pos" in raw_obs and "desired_goal" in raw_obs:
            tcp = np.asarray(raw_obs["tcp_pos"], dtype=np.float32)[:3]
            goal = np.asarray(raw_obs["desired_goal"], dtype=np.float32)[:3]
            xy_d = float(np.linalg.norm(tcp[:2] - goal[:2]))
            min_xy = min(min_xy, xy_d)

        if bool(terminated or truncated):
            break

    metrics = compute_metrics_from_obs(env, info.get("obs", {}))
    if np.isfinite(min_xy):
        metrics.min_xy_distance_m = min_xy

    success = bool(metrics.xy_ok and metrics.z_ok)
    failure = classify_episode(
        terminated=bool(terminated),
        truncated=bool(truncated),
        last_reward=last_reward,
        success=success,
    )
    return EpisodeResult(
        success=success,
        terminated=bool(terminated),
        truncated=bool(truncated),
        reward=episode_reward,
        length=length,
        last_step_reward=last_reward,
        failure=failure,
        metrics=metrics,
    )


def print_episode_report(
    ep_index: int,
    result: EpisodeResult,
    summary: EvalSummary,
    *,
    task: Any,
) -> None:
    """Per-episode console report with alignment bars and failure hints."""
    width = 64
    m = result.metrics

    if result.success:
        title = _c("1;32", f"Episode {ep_index:04d}  ✓ SUCCESS")
    else:
        fail_tag = result.failure or "fail"
        title = _c("1;31", f"Episode {ep_index:04d}  ✗ FAIL ({fail_tag})")

    print()
    print("─" * width)
    print(title)
    print("─" * width)

    print(f"  total reward     {result.reward:10.2f}")
    print(f"  last-step reward {result.last_step_reward:10.2f}")
    print(f"  steps            {result.length:10d}  "
          f"terminated={int(result.terminated)}  truncated={int(result.truncated)}")

    if m.tcp_pos is not None and m.desired_goal is not None:
        print(f"  TCP (m)          [{m.tcp_pos[0]:+.4f}, {m.tcp_pos[1]:+.4f}, {m.tcp_pos[2]:+.4f}]")
        print(f"  hole goal (m)    [{m.desired_goal[0]:+.4f}, {m.desired_goal[1]:+.4f}, {m.desired_goal[2]:+.4f}]")

    print()
    print("  Alignment (success thresholds)")
    xy_bar = _bar(m.xy_ok)
    z_bar = _bar(m.z_ok)
    print(f"    XY  [{xy_bar}]  {m.xy_distance_mm:7.2f} mm  {'OK' if m.xy_ok else 'MISS'}")
    print(f"    Z   [{z_bar}]  {m.z_above_mm:+7.2f} mm  {'OK' if m.z_ok else 'MISS'}")

    if np.isfinite(m.min_xy_distance_mm):
        print(f"    best XY in ep     {m.min_xy_distance_mm:7.2f} mm")

    print()
    print("  Diagnosis")
    for line in failure_hints(result, task):
        print(f"    • {line}")

    running_rate = summary.success_rate * 100.0
    print()
    print(
        f"  Running success: {summary.n_success}/{summary.n_episodes} "
        f"= {running_rate:.1f}%"
    )
    print("─" * width)


def print_summary(summary: EvalSummary, cli: argparse.Namespace) -> None:
    width = 64
    print()
    print("=" * width)
    print("  Evaluation summary")
    print("=" * width)
    print(f"  episodes          {summary.n_episodes}")
    print(f"  success           {summary.n_success}")
    print(f"  success rate      {summary.success_rate * 100:.2f}%")
    print(f"  timeout           {summary.n_timeout}")
    print(f"  collision         {summary.n_collision}")
    print(f"  out_of_workspace  {summary.n_out_of_workspace}")
    print(f"  other fail        {summary.n_other_fail}")
    if summary.episode_rewards:
        print(f"  mean reward       {np.mean(summary.episode_rewards):.2f}")
        print(f"  std reward        {np.std(summary.episode_rewards):.2f}")
        print(f"  mean length       {np.mean(summary.episode_lengths):.1f}")
    print("=" * width)
    print(f"  model={cli.model_path}")
    print(f"  obs_mode={cli.obs_mode}  n_episodes={cli.n_episodes}  seed={cli.seed}")
    if cli.obs_mode in FORCE_OBS_MODES or cli.use_wrench_history:
        print(
            f"  wrench_history_len={cli.wrench_history_len}  "
            f"force_scale={cli.force_scale}  torque_scale={cli.torque_scale}"
        )
    print("=" * width)


def main() -> None:
    cli = parse_args()
    if cli.n_episodes < 1:
        raise ValueError("--n-episodes must be >= 1")

    env_args = build_args(cli)
    print(f"[eval] obs_mode={cli.obs_mode} | model={cli.model_path}")
    print(f"[eval] n_episodes={cli.n_episodes} | max_steps={cli.max_steps} | seed={cli.seed}")

    env = make_visual_env(env_args)
    base_env = _unwrap_env(env)
    task = getattr(base_env, "task", None)

    print(f"[eval] observation_space={env.observation_space}")
    model = SAC.load(cli.model_path, env=env, device=cli.device)

    summary = EvalSummary()
    for ep in range(cli.n_episodes):
        result = run_episode(
            env,
            model,
            max_steps=cli.max_steps,
            seed=cli.seed + ep,
        )
        summary.record(result)

        if not cli.quiet:
            print_episode_report(ep, result, summary, task=task)

    env.close()
    print_summary(summary, cli)


if __name__ == "__main__":
    main()
