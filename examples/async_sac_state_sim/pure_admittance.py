#!/usr/bin/env python3
from pathlib import Path
import sys

import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import time
from functools import partial

import gym
import gymnasium
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import threading

from agentlace.data.data_store import QueuedDataStore
from agentlace.trainer import TrainerClient, TrainerServer
from serl_launcher.utils.launcher import (
    make_sac_agent,
    make_trainer_config,
    make_wandb_logger,
    make_replay_buffer,
)

from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics
from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.common.evaluation import evaluate
from serl_launcher.utils.timer_utils import Timer

from mujoco import viewer as mj_viewer
from mujoco_env.mujoco_env.envs import env_instance  # noqa: F401 触发 register
from mujoco_env.mujoco_env.envs.env_instance import make_aubo_i5_assemble_hole_env
from mujoco_env.mujoco_env.envs.env_instance import make_franka_panda_pick_cube_env
from dm_robotics.transformations import transformations as tr
from mujoco_env.mujoco_env.tasks.peg_insertion.trajectory_recorder import TrajectoryRecorder
from mujoco_env.mujoco_env.tasks.peg_insertion.aubo_i5_config import AuboI5Config
from mujoco_env.mujoco_env.robot_config.aubo_i5 import AuboI5Robot

import pdb
# import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue

FLAGS = flags.FLAGS

flags.DEFINE_float('utd_ratio', 1.0, 'Update-to-data ratio.')
flags.DEFINE_string("env", "HalfCheetah-v4", "Name of environment.")
flags.DEFINE_string("agent", "sac", "Name of agent.")
flags.DEFINE_string("exp_name", None, "Name of the experiment for wandb logging.")
flags.DEFINE_integer("max_traj_length", 100, "Maximum length of trajectory.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_bool("save_model", False, "Whether to save model.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_integer("critic_actor_ratio", 8, "critic to actor update ratio.")

flags.DEFINE_integer("max_steps", 1000000, "Maximum number of training steps.")
flags.DEFINE_integer("replay_buffer_capacity", 1000000, "Replay buffer capacity.")
flags.DEFINE_integer("learner_update_per_step", 3000, "After how many steps make an update")
flags.DEFINE_integer("learner_update_steps", 10, "How many steps each update")
flags.DEFINE_integer("random_steps", 1000, "Sample random actions for this many steps.")
flags.DEFINE_integer("training_starts", 1000, "Training starts after this step.")
flags.DEFINE_integer("steps_per_update", 50, "Number of steps per update the server.")

flags.DEFINE_integer("log_period", 10, "Logging period.")
flags.DEFINE_integer("eval_period", 2000, "Evaluation period.")
flags.DEFINE_integer("eval_n_trajs", 5, "Number of trajectories for evaluation.")

# flag to indicate if this is a leaner or a actor
flags.DEFINE_boolean("learner", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("actor", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("render", False, "Render the environment.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_integer("checkpoint_period", 0, "Period to save checkpoints.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging

flags.DEFINE_string("log_rlds_path", None, "Path to save RLDS logs.")
flags.DEFINE_string("preload_rlds_path", None, "Path to preload RLDS data.")

flags.DEFINE_boolean("record_trajectory", False, "启用轨迹录制功能")
flags.DEFINE_string("trajectory_dir", "./trajectories", "轨迹文件保存目录")


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def main(_):


    gymnasium.register(
        id="Auboi5_assemble_hole_env-v0", 
        entry_point="mujoco_env.mujoco_env.envs.env_instance:make_aubo_i5_assemble_hole_env", 
        kwargs={'render_mode': "human", 
                "mode":"sim", 
                "control_dt": 0.01, 
                "physics_dt": 0.001, 
                "render_dt": 0.0016,
                "max_episode_steps": 2000, 
                "ik_regularization": 0.001, 
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

    # create env and load dataset
    # env = make_aubo_i5_assemble_hole_env(
    #         scene_name="assemble_hole",
    #         include_image=False,
    #         image_size=(128, 128),
    #         include_depth=False,
    #         control_dt=0.02,
    #         physics_dt=0.002,
    #         max_episode_steps=1000,
    #         render_mode="human", 
    #     )
    
    env = gymnasium.make("Auboi5_assemble_hole_env-v2")

    env = gymnasium.wrappers.FlattenObservation(env)

    queue = Queue()

    action = env.controller.robot_interface.get_robot_state()["tcp_pose"]
    euler = np.array(action[3:]).astype(np.float32)
    quat = tr.euler_to_quat(euler, 'ZYX')
    robot_action = np.zeros(7)
    robot_action[:3] = action[:3]
    robot_action[3:] = quat

    robot_action_1 = robot_action.copy()
    robot_action_2 = robot_action.copy()
    robot_action_2[0] += 0.2

    print("Collison level: ", env.controller.robot_interface.robot_config.getCollisionLevel())
    print("Collison stop type: ", env.controller.robot_interface.robot_config.getCollisionStopType())
    # print("Collison threshold: ", env.controller.robot_interface.robot_config.getCollisionThreshold())
    print("Collison occured: ", env.controller.robot_interface.robot_state.isCollisionOccurred())

    env.controller.robot_interface.robot_config.setCollisionLevel(1)

    def control_loop(queue): 
        
        controller = env.controller
        wrench_offset = np.array(controller.robot_interface.get_robot_state()["tcp_wrench"]).astype(np.float32)
        init_qpos = np.array([177.0 / 180.0 * np.pi,  4.0 / 180.0 * np.pi, 95.0 / 180.0 * np.pi, 1.7 / 180.0 * np.pi, 89 / 180.0 * np.pi, 20.0 / 180.0 * np.pi]) # The VLA arm
        # init_qpos = np.array([123.0 / 180.0 * np.pi,  -26.0 / 180.0 * np.pi, 89.0 / 180.0 * np.pi, 24 / 180.0 * np.pi, 89 / 180.0 * np.pi, -20.0 / 180.0 * np.pi]) # The VLA arm
        init_tcppose = np.array(controller.robot_interface.robot_algorithm.forwardKinematics(init_qpos)[0])

        ret = controller.robot_interface.enable_servo_mode(False)
        controller.robot_interface.move_to_joint_positions(init_qpos, 
                                                        velocity=1.0, 
                                                        acceleration=0.5, 
                                                        blocking=True)



        controller.set_selection_vector(np.array([1, 1, 1, 0, 0, 0]))
        
        # For admittance 0.005, 0.01, 0.05, 0.1
        # controller.set_admittance_params(mass=np.array([20.0, 20.0, 10.0, 1.0, 1.0, 1.0]), 
        #                                     damping=np.array([400.0, 340.0, 180.0, 5.0, 5.0, 5.0]), 
        #                                     stiffness=np.array([60.0, 60.0, 50.0, 9.0, 9.0, 9.0])) 
        # For admittance None
        controller.set_admittance_params(mass=np.array([20.0, 20.0, 10.0, 1.0, 1.0, 1.0]), 
                                            damping=np.array([450.0, 400.0, 300.0, 5.0, 5.0, 5.0]), 
                                            stiffness=np.array([60.0, 60.0, 50.0, 9.0, 9.0, 9.0])) 
        
        controller.set_admittance_params(mass=np.array([20.0, 20.0, 10.0, 1.0, 1.0, 1.0]), 
                                            damping=np.array([450.0, 400.0, 300.0, 5.0, 5.0, 5.0]), 
                                            stiffness=np.array([0.0, 0.0, 0.0, 9.0, 9.0, 9.0])) 
        


        action = controller.robot_interface.get_robot_state()["tcp_pose"]
        euler = np.array(action[3:]).astype(np.float32)
        quat = tr.euler_to_quat(euler, 'ZYX')
        robot_action = np.zeros(7)
        robot_action[:3] = action[:3]
        robot_action[3:] = quat

        robot_action_1 = robot_action.copy()
        robot_action_2 = robot_action.copy()
        robot_action_2[0] += 0.5


        if hasattr(controller, 'reset'):
            controller.reset()
        controller.robot_interface.clear_errors()
        time.sleep(1)
        # ret = controller.robot_interface.enable_servo_mode(True)
        # print(controller.robot_interface.robot_motion.isServoModeEnabled())

        cnt = 0
        flag = 1

        while True: 

            # cnt += 1

            # if cnt % 150 == 0: 
            #     if flag == 1:
            #         flag = 2
            #         robot_action = robot_action_2
            #     else: 
            #         flag = 1
            #         robot_action = robot_action_1

            # print("time_1: ", time.time())
            current_state = {
                    "qpos": controller.robot_interface.get_joint_positions(), 
                    "qvel": controller.robot_interface.get_joint_velocities(), 
                    "wrench": controller.robot_interface.get_tcp_wrench() - wrench_offset
                }
            
            for i in range(current_state['wrench'].shape[0]): 
                    if current_state['wrench'][i] > -0.5 and current_state['wrench'][i] < 0.5: current_state['wrench'][i] = 0
            
            # print("wrench: ", current_state['wrench'])
            
            # print("time_2: ", time.time())

            if not queue.empty():
                robot_action = queue.get()


            ctrl_signal = controller.compute_control(
                    target=robot_action,
                    current_state=current_state, 
                    external_force= current_state["wrench"], 
                )
            # print("time_3: ", time.time())
            # controller.robot_interface.move_tcp_pose(
            #         ctrl_signal,
            #         velocity=0.5,
            #         acceleration=5,
            #         blocking=True,
            #     )
            # print(ctrl_signal)
            while not controller.robot_interface.robot_motion.isServoModeEnabled():
                ret = controller.robot_interface.enable_servo_mode(True)
            if ret:
                ret_servootion = controller.robot_interface.servo_to_tcp_positions(ctrl_signal)
                # print("ret_servootion: ", ret_servootion)
                # print("ret: ", ret)
            # print("time_4: ", time.time())

    
    p = Process(target=control_loop, args=(queue, ) ,daemon=True)
    p.start()



    # cnt = 0
    # flag = 1
    # pre = time.time()

    

    
    while True: 
        pass
        # cur = time.time()
        # if cur - pre > 2: 
        #     if flag == 1:
        #         flag = 2
        #         queue.put(robot_action_2)
        #     else: 
        #         flag = 1
        #         queue.put(robot_action_1)
            
        #     pre = cur
    



    

if __name__ == "__main__":
    app.run(main)
