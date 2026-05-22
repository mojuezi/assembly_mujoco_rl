from argparse import Namespace
from stable_baselines3 import SAC

from train_visual_sac_baseline import make_visual_env

args = Namespace(
    control_dt=0.01,
    physics_dt=0.001,
    render_dt=0.006,
    max_episode_steps=500,
    ik_regularization=0.0001,
    ik_radius=0.01,

    image_size=64,
    camera_name="ee_cam",
    max_depth=2.0,
    save_depth_dir="./debug_depth_eval",
    save_depth_count=0,
    show_depth=True,
    log_every=1,
    render_human=True,
    inspect_raw_obs=False,
)

env = make_visual_env(args)

model = SAC.load(
    "./checkpoints/visual_sac_baseline/visual_sac_baseline_final",
    env=env,
    device="cuda",
)

obs, info = env.reset()
episode_reward = 0.0

for step in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    episode_reward += reward
    done = terminated or truncated

    print(
        f"step={step}, action={action}, reward={reward:.4f}, "
        f"episode_reward={episode_reward:.4f}, done={done}"
    )

    if done:
        break

env.close()