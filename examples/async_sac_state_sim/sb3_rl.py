#!/usr/bin/env python3
from pathlib import Path
import sys
import os

import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import time
from functools import partial

import gymnasium
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import time

from mujoco import viewer as mj_viewer
from mujoco_env.mujoco_env.envs import env_instance  # noqa: F401 触发 register
from mujoco_env.mujoco_env.envs.env_instance import make_aubo_i5_assemble_hole_env
from mujoco_env.mujoco_env.envs.env_instance import make_franka_panda_pick_cube_env
from mujoco_env.mujoco_env.tasks.peg_insertion.trajectory_recorder import TrajectoryRecorder
from mujoco_env.mujoco_env.tasks.peg_insertion.aubo_i5_config import AuboI5Config
from mujoco_env.mujoco_env.robot_config.aubo_i5 import AuboI5Robot

from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback



def print_green(x):
    return print("\033[92m {}\033[00m".format(x))

#train model 
def train(agent_name="ppo", total_timesteps=100_000, save_freq=10_000, env=None, 
          save_path="./checkpoints/test_image_sim_v2"):
    
    # Create a directory to store the log files if it doesn't exist
    log_dir = "./logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = None

    # For sim env
    # Create the environment and wrap it with Monitor to log performance
    env = Monitor(env, log_dir)  # Monitor the environment and store logs in the specified directory

    # For real env
    # env = PegInHoleGymEnvReal()

    
    # Choose the algorithm
    agent_name = agent_name.lower()
    if agent_name == "ppo":
        model_class = PPO
        policy = "MultiInputPolicy"
    elif agent_name == "sac":
        model_class = SAC
        policy = "MultiInputPolicy"
    elif agent_name == "a2c":
        model_class = A2C
        policy = "MultiInputPolicy"
    else:
        raise ValueError(f"Unsupported agent: {agent_name}")

    # Create the model
    model = model_class(policy, env, verbose=1, train_freq=1, gradient_steps=1, device="cuda", tensorboard_log=log_dir, 
                        buffer_size=1_000_00, batch_size=128)  

    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=save_path,
                                             name_prefix=f"{agent_name}_model")

    # Train the model and save checkpoints
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    # Save the final model after training
    model.save(os.path.join(save_path, f"{agent_name}_final_model"))

    return model, env



#test model 1000 times episode
def test_rl_model(agent_name, env):
    # Create the environment
    # env = PegInHoleGymEnv()
    # env = PegInHoleGymEnvReal()

    # Choose the algorithm
    if agent_name == "ppo":
        model_class = PPO
        policy = "MultiInputPolicy"
    elif agent_name == "sac":
        model_class = SAC
        policy = "MultiInputPolicy"
    elif agent_name == "a2c":
        model_class = A2C
        policy = "MultiInputPolicy"
    else:
        raise ValueError(f"Unsupported agent: {agent_name}")

    # Load the trained model
    model = model_class(policy, env, verbose=1, device="cuda", buffer_size=1_000_00)
    model = model.load(f"./checkpoints/test_image_sim_v1/sac_final_model")

    success_count = 0
    failure_count = 0
    episode_count = 0
    max_episodes = 1000

    obs, info = env.reset()

    while episode_count < max_episodes:
        print("obs_pre: ", info["obs"]["tcp_pos"])
        print("obs_pre_norm: ", obs["tcp_pos"])
        action, _ = model.predict(obs, deterministic=True)
        # action = np.array([-0.005, 0.005, -0.005])
        obs, reward, terminated, truncated, info = env.step(action)
        print("obs_post: ", info["obs"]["tcp_pos"])
        print("obs_post_norm: ", obs["tcp_pos"])

        if terminated or truncated:
            episode_count += 1

            if info.get("insertion_success", True):
                success_count += 1
                print(f"Episode {episode_count}: Success")
            else:
                failure_count += 1
                print(f"Episode {episode_count}: Failure")

            obs, info = env.reset() 

    success_rate = (success_count / max_episodes) * 100
    print(f"\n Test completed: {max_episodes} episodes in total")
    print(f"Successes: {success_count}")
    print(f"Failures: {failure_count}")
    print(f"Success Rate: {success_rate:.2f}%")




if __name__ == "__main__":

    gymnasium.register(
        id="Auboi5_assemble_hole_env-v0", 
        entry_point="mujoco_env.mujoco_env.envs.env_instance:make_aubo_i5_assemble_hole_env", 
        kwargs={'render_mode': "rgb_array", 
                "mode":"sim", 
                "include_image": True,
                "image_size": (128, 128),
                "include_depth": False,
                "control_dt": 0.01, 
                "physics_dt": 0.0005, 
                "render_dt": 0.006,
                "max_episode_steps": 500, 
                "ik_regularization": 0.0001, 
                "ik_radius": 0.01}
    )
    gymnasium.register(
        id="Auboi5_assemble_hole_env-state", 
        entry_point="mujoco_env.mujoco_env.envs.env_instance:make_aubo_i5_assemble_hole_env", 
        kwargs={'render_mode': "human", 
                "mode":"sim", 
                "control_dt": 0.01, 
                "physics_dt": 0.001, 
                "render_dt": 0.006,
                "max_episode_steps": 500, 
                "ik_regularization": 0.0001, 
                "ik_radius": 0.01}
    )
    gymnasium.register(
        id="Auboi5_assemble_hole_env-v1", 
        entry_point="mujoco_env.mujoco_env.envs.env_instance:make_aubo_i5_assemble_hole_env", 
        kwargs={'render_mode': None, 
                "mode":"sim", 
                "control_dt": 0.001, 
                "physics_dt": 0.001, 
                "max_episode_steps": 20000, 
                "ik_regularization": 0.001, 
                "ik_radius": 0.01}
    )

    gymnasium.register(
        id="Auboi5_assemble_hole_env-v2", 
        entry_point="mujoco_env.mujoco_env.envs.env_instance:make_aubo_i5_assemble_hole_env", 
        kwargs={'render_mode': None, 
                'mode': "real", 
                'robot_ip': "192.168.1.100", 
                "control_dt": 0.001, 
                "physics_dt": 0.001, 
                "max_episode_steps": 20000, 
                "ik_regularization": 0.001, 
                "ik_radius": 0.01, 
                "hole_position_real": np.array([-0.518,0.103,0.055]), 
                "workspace_low_real": np.array([-0.67,-0.03,0.04]),
                "workspace_high_real": np.array([-0.354,0.41,0.3617]), }
    )

    env = gymnasium.make("Auboi5_assemble_hole_env-v0")
    # env = gymnasium.wrappers.FlattenObservation(env)

    agent_name = "sac"

    # Train the RL model
    train(agent_name=agent_name, total_timesteps=1000000, save_freq=20000, env=env)

    # Test the trained RL model
    # test_rl_model(agent_name, env)
