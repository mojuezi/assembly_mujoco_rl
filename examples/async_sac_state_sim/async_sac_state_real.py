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

flags.DEFINE_string("loaded_model", None, "Path to the trained model")


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


##############################################################################


def actor(agent: SACAgent, data_store, env, sampling_rng):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_store,
        wait_for_server=True,
    )

    # Function to update the agent with new params
    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    eval_env = gymnasium.make("Auboi5_assemble_hole_env-v1")
    eval_env = gymnasium.wrappers.FlattenObservation(eval_env)
    # if FLAGS.env == "PandaPickCube-v0":
    #     eval_env = gym.wrappers.FlattenObservation(eval_env)
    eval_env = RecordEpisodeStatistics(eval_env)

    env_traj_recorder = make_aubo_i5_assemble_hole_env(
            scene_name="assemble_hole",
            include_image=False,
            image_size=(128, 128),
            include_depth=False,
            control_dt=0.01, 
            physics_dt=0.001, 
            max_episode_steps=1000,
            render_mode="human", 
        )


    trajectory_recorder = None
    if FLAGS.record_trajectory:
        print(f"\n📹 初始化轨迹记录器...")
        trajectory_recorder = TrajectoryRecorder(
            robot_name="aubo_i5",
            dof=6,
            save_dir=FLAGS.trajectory_dir,
            control_freq=1.0 / env_traj_recorder.control_dt,
            auto_save_on_episode_end=True
        )
        # 设置环境信息用于回放
        trajectory_recorder.set_env_info(
            xml_path=str(env_traj_recorder.xml_path),
            scene_name=env_traj_recorder.task.scene_name,
            task_name=env_traj_recorder.task.name,
            gripper_name=AuboI5Config.get_config().gripper_name,
            joint_names=["shoulder_joint", "upperArm_joint", "foreArm_joint", "wrist1_joint", "wrist2_joint", "wrist3_joint"],
            initial_qpos=AuboI5Robot.INIT_QPOS
        )
        print(f"✅ 轨迹记录器初始化成功")
        print(f"   - 保存目录: {FLAGS.trajectory_dir}")
        print(f"   - 控制频率: {1.0 / env_traj_recorder.control_dt:.1f} Hz")




    env.controller.robot_interface.robot_config.setCollisionLevel(1)

    obs, info = env.reset()
    done = False
    pos_ori = info["obs"]["tcp_pos"].copy()
    quat_ori = info["obs"]["tcp_quat"].copy()
    # euler_ori = info["euler"].copy()
    pos_ref = info["obs"]["desired_goal"].copy()
    pos_ref[-1] += 0.07
    action_input_pos_quat = np.concatenate([pos_ori, quat_ori], -1)
    # action_input_pos_euler = np.concatenate([pos_ori, euler_ori], -1)
    robot_action = np.concatenate([pos_ori, quat_ori], -1)
    robot_action = np.concatenate([pos_ref, quat_ori], -1)
    rewards_list_total = []


    # Make a thread func

    queue_action = Queue()
    queue_observation = Queue()
    queue_observation_reset = Queue()

    def control_loop(env, queue_action, queue_observation, queue_observation_reset): 
        controller = env.controller
        wrench_offset = np.array(controller.robot_interface.get_robot_state()["tcp_wrench"]).astype(np.float32)
        init_qpos = np.array([177.0 / 180.0 * np.pi,  4.0 / 180.0 * np.pi, 95.0 / 180.0 * np.pi, 1.7 / 180.0 * np.pi, 89 / 180.0 * np.pi, 20.0 / 180.0 * np.pi]) # The VLA arm
        # init_qpos = np.array([123.0 / 180.0 * np.pi,  -26.0 / 180.0 * np.pi, 89.0 / 180.0 * np.pi, 24 / 180.0 * np.pi, 89 / 180.0 * np.pi, -20.0 / 180.0 * np.pi]) 
        init_tcppose = np.array(controller.robot_interface.robot_algorithm.forwardKinematics(init_qpos)[0])


        controller.robot_interface.move_to_joint_positions(env.init_qpos, 
                                                        velocity=0.5, 
                                                        acceleration=0.5, 
                                                        blocking=True)
        
        controller.set_selection_vector(np.array([1, 1, 1, 0, 0, 0]))
        
        controller.set_admittance_params(mass=np.array([20.0, 20.0, 10.0, 1.0, 1.0, 1.0]), 
                                            damping=np.array([450.0, 400.0, 300.0, 5.0, 5.0, 5.0]), 
                                            stiffness=np.array([60.0, 60.0, 50.0, 9.0, 9.0, 9.0])) 
        
        controller.set_admittance_params(mass=np.array([20.0, 20.0, 10.0, 1.0, 1.0, 1.0]), 
                                            damping=np.array([450.0, 400.0, 300.0, 5.0, 5.0, 5.0]), 
                                            stiffness=np.array([0.0, 0.0, 0.0, 9.0, 9.0, 9.0])) 
        
        # action = controller.robot_interface.get_robot_state()["tcp_pose"]
        # euler = np.array(action[3:]).astype(np.float32)
        # quat = tr.euler_to_quat(euler, 'ZYX')
        # robot_action = np.zeros(7)
        # robot_action[:3] = action[:3]
        # robot_action[3:] = quat

        desired_force = np.zeros(6)


        if hasattr(controller, 'reset'):
            controller.reset()

        controller.robot_interface.clear_errors()
        ret = controller.robot_interface.enable_servo_mode(True)

        while True: 

            # print("process_time1: ", time.time())

            current_state = {
                    "qpos": controller.robot_interface.get_joint_positions(), 
                    "qvel": controller.robot_interface.get_joint_velocities(), 
                    "wrench": controller.robot_interface.get_tcp_wrench() - wrench_offset
                }
            
            for i in range(current_state['wrench'].shape[0]): 
                if current_state['wrench'][i] > -0.5 and current_state['wrench'][i] < 0.5: current_state['wrench'][i] = 0
            
            if not queue_action.empty():
                desired_force = queue_action.get()
            
            controller.set_desired_force(desired_force)
            # print("desired_force: ", desired_force)

            ctrl_signal = controller.compute_control(
                    target=robot_action,
                    current_state=current_state, 
                    external_force= current_state["wrench"], 
                )

            while not controller.robot_interface.robot_motion.isServoModeEnabled():
                ret = controller.robot_interface.enable_servo_mode(True)
                # print("ret: ", ret)

            if ret:
                ret_servootion = controller.robot_interface.servo_to_tcp_positions(ctrl_signal)
                # print("ret_servootion: ", ret_servootion)
            
            
            next_obs, reward, done, truncated, info = env.step(np.zeros(7)) # make a check
            queue_observation.put((next_obs, reward, done, truncated, info))

            if done or truncated: 
                
                ret = controller.robot_interface.enable_servo_mode(False)

                if done: 
                    # TODO: Move tcp 0.1m upwards
                    cur_tcp = controller.robot_interface.get_tcp_pose()
                    cur_tcp[2] += 0.08
                    controller.robot_interface.move_tcp_pose(cur_tcp, 
                                                             velocity=0.5, 
                                                            acceleration=0.5, 
                                                            blocking=True)
                
                controller.robot_interface.move_to_joint_positions(env.init_qpos, 
                                                        velocity=0.5, 
                                                        acceleration=0.5, 
                                                        blocking=True)
                if hasattr(controller, 'reset'):
                    controller.reset()
                
                controller.set_desired_force(np.zeros(6))
                desired_force = np.zeros(6)

                obs, info = env.reset()
                # TODO
                queue_observation_reset.put((obs, reward, done, truncated, info))
                reset_start = time.time()
                
                # while not controller.robot_interface.robot_motion.isServoModeEnabled():
                #     ret = controller.robot_interface.enable_servo_mode(True)
                
            
    
    p = Process(target=control_loop, args=(env, queue_action, queue_observation, queue_observation_reset) ,daemon=True)
    p.start()


    
    

    # training loop
    timer = Timer()
    running_return = 0.0
    # action_prev = np.zeros_like(env.action_space.sample())

    
    for step in tqdm.tqdm(range(FLAGS.max_steps*10), dynamic_ncols=True):
        timer.tick("total")
        # pdb.set_trace()
        # print("time_1: ", time.time())

        if trajectory_recorder and step % 100000==0 and step!=0:
            trajectory_recorder.new_episode(step)

        with timer.context("sample_actions"):
            # if step < FLAGS.random_steps:
            #     actions = env.action_space.sample()
            #     # actions = np.clip(actions, -0.1, 0.1)
            # else:
            #     print("time_jax1: ", time.time())
            sampling_rng, key = jax.random.split(sampling_rng)
            actions = agent.sample_actions(
                observations=jax.device_put(obs),
                seed=key,
                deterministic=False,
            )
            actions = np.asarray(jax.device_get(actions))
            # actions = np.clip(actions, -0.1, 0.1)
            ##################################################### dim 6: position control

            # action_input =  actions.copy() * 0.01
            # action_input[3:] = 0

            # action_input_euler = action_input[3:]
            # action_input_quat = tr.euler_to_quat(action_input_euler)
            # # action_input_pos_quat = np.zeros(7)

            # obs_dict = info["obs"]
            # current_rot_quat = obs_dict["tcp_quat"]
            # current_pos = obs_dict["tcp_pos"]
            # current_hmat = tr.pos_quat_to_hmat(current_pos, current_rot_quat)
            # action_input_hmat = tr.pos_quat_to_hmat(action_input[:3], action_input_quat)

            # action_input_hmat_final = action_input_hmat @ current_hmat
            # action_input_pos, action_input_quat = tr.hmat_to_pos_quat(action_input_hmat_final)
            # action_input_pos_quat = np.concatenate([action_input_pos, action_input_quat], -1)

            # action_input_pos_quat[3:] = quat_ori
            # action_input_pos_quat[2] = pos_ori[-1]
            ############################################################ dim 2: position control
            # actions = np.array([0, -0.01])
            # action_input =  actions.copy() * 0.01
            # obs_dict = info["obs"]
            # current_pos = obs_dict["tcp_pos"]
            # action_input_pos_quat[:3] += action_input
            ###########################################
            # action_input_pos_quat[2] -= 0.001
            desired_force = np.zeros(6)
            # desired_force[2] = -actions[2].copy()

            desired_force[0] = -10 * actions[1].copy()
            desired_force[1] = 10 * actions[0].copy()
            desired_force[2] = -5 * actions[2].copy()

            # desired_force[:2] = -5 * actions[:2].copy()
            # desired_force[2] = -3 * actions[2].copy()
            print("desired_force: ", desired_force)
            # time.sleep(0.04)
            
            # desired_force[:3] = [0, -10, -10]
            # desired_force[0] = 10
            

            # action_input = actions[:2].copy() * 0.01

            queue_action.put(desired_force)


            # Staring here: action includes ref pos
            # pdb.set_trace()
            # pos_ref_incre_action = actions.copy()[3:] * 0.01
            # action_input_pos_quat[:3] += pos_ref_incre_action

        # Step environment
        with timer.context("step_env"):
            # next_obs, reward, done, truncated, info = env.step(action_input_pos_quat)
            # if not queue_observation.empty(): 
            next_obs, reward, done, truncated, info = queue_observation.get()
            # print(f"reward: {reward} done: {done}, truncated: {truncated}" )
            # wrench = info["obs"]["wrench"]
            # print(f"wrench: {wrench}" )
            # print("desired_goal: ", info['obs']['desired_goal'])
            print("reward: ", reward)
            # print("action: ", action_input_pos_quat)
            # print("current pos: ", info["obs"]["tcp_pos"])
            # pdb.set_trace()
            next_obs = np.asarray(next_obs, dtype=np.float32)
            reward = np.asarray(reward, dtype=np.float32)

            running_return += reward

            data_store.insert(
                dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=reward,
                    masks=1.0 - done,
                    dones=done or truncated,
                )
            )

            if trajectory_recorder:
                qpos = info["qpos"][:6]  # 只记录前6个关节（机械臂关节）
                trajectory_recorder.record(qpos)

            obs = next_obs
            if done or truncated:
                running_return = 0.0
                obs, reward, done, truncated, info = queue_observation_reset.get()
            #     obs, info = env.reset()
            #     action_input_pos_quat = np.concatenate([pos_ori, quat_ori], -1)
                # action_prev = np.zeros_like(env.action_space.sample())

        # if FLAGS.render:
            # env.render()
        # print("time_2: ", time.time())
        
        # if step % FLAGS.steps_per_update == 0:
        #     client.update()

        if step % FLAGS.eval_period == 0 and step != 0:
            # with timer.context("eval"):
            #     evaluate_info = evaluate(
            #         policy_fn=partial(agent.sample_actions, argmax=True),
            #         env=eval_env,
            #         num_episodes=FLAGS.eval_n_trajs,
            #     )
            # stats = {"eval": evaluate_info}
            # client.request("send-stats", stats)
            pass
            # policy_fn = partial(agent.sample_actions, argmax=True)
            # rewards_list = []
            # for _ in range(FLAGS.eval_n_trajs):
            #     step_eval = 0
            #     reward_epi = 0.0
            #     observation_eval, info_eval = eval_env.reset()
            #     done_eval = False
            #     while not done_eval:
            #         step_eval += 1
            #         action_eval = policy_fn(observation_eval)
            #         observation_eval, reward_eval, done_eval, truncated_eval, info_eval = eval_env.step(action_eval)
            #         reward_epi += reward_eval
            #         if done_eval or truncated_eval: 
            #             rewards_list.append(reward_epi / step)
            #             break
            
            # rewards_list_total.append(rewards_list)

            # rewards_list_total_array = np.array(rewards_list_total)
            # if rewards_list_total_array[-1].mean():
            #     pass
            # # df = pd.DataFrame(rewards_list_total_array)
            # # fig, ax = plt.plot() 
            # # ax.plot(rewards_list_total_array.mean(axis=1))
            # # ax.grid()
            # # plt.tight_layout()
            # print("rewards_list_total_array: ", rewards_list_total_array)


        timer.tock("total")
        # print("time_jax2: ", time.time())

        if step % FLAGS.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)

    
##############################################################################


def learner(rng, agent: SACAgent, replay_buffer, replay_iterator):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    # set up wandb and logging
    wandb_logger = make_wandb_logger(
        project="serl_dev",
        description=FLAGS.exp_name or FLAGS.env,
        debug=FLAGS.debug,
    )
    wandb_logger=None

    # To track the step in the training loop
    update_steps = 0

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=update_steps)
        return {}  # not expecting a response

    # Create server
    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.start(threaded=True)

    # Loop to wait until replay_buffer is filled
    pbar = tqdm.tqdm(
        total=FLAGS.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < FLAGS.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
    pbar.close()

    # send the initial network to the actor
    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    # wait till the replay buffer is filled with enough data
    timer = Timer()

    # show replay buffer progress bar during training
    pbar = tqdm.tqdm(
        total=FLAGS.replay_buffer_capacity,
        initial=len(replay_buffer),
        desc="replay buffer",
    )

    for step in tqdm.tqdm(range(FLAGS.max_steps * FLAGS.steps_per_update), dynamic_ncols=True, desc="learner"):
        # Train the networks

        if step % int(FLAGS.steps_per_update)==0: 

            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)

            with timer.context("train"):
                utd_ratio_int = int(FLAGS.utd_ratio)
            
            
            # for _ in range(int(FLAGS.learner_update_steps)): 
            agent, update_info = agent.update_high_utd(batch, utd_ratio=utd_ratio_int)
            agent = jax.block_until_ready(agent)
            
        # publish the updated network
            server.publish_network(agent.state.params)

        if update_steps % FLAGS.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=update_steps)
            wandb_logger.log({"timer": timer.get_average_times()}, step=update_steps)

        if FLAGS.checkpoint_period and update_steps % FLAGS.checkpoint_period == 0:
            assert FLAGS.checkpoint_path is not None
            checkpoints.save_checkpoint(
                FLAGS.checkpoint_path, agent.state, step=update_steps, keep=20
            )

        pbar.update(len(replay_buffer) - pbar.n)  # update replay buffer bar
        update_steps += 1


##############################################################################


def main(_):
    devices = jax.local_devices()
    num_devices = len(devices)
    sharding = jax.sharding.PositionalSharding(devices)
    assert FLAGS.batch_size % num_devices == 0

    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)

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
    if jax.default_backend() == "cpu": 
        env = gymnasium.make("Auboi5_assemble_hole_env-v0")
    else:
        env = gymnasium.make("Auboi5_assemble_hole_env-v1")
    
    env = gymnasium.make("Auboi5_assemble_hole_env-v2")

    env = gymnasium.wrappers.FlattenObservation(env)



    # if FLAGS.render:
    #     env = gym.make(FLAGS.env, render_mode="human")
    # else:
    #     env = gym.make(FLAGS.env)

    # if FLAGS.env == "PandaPickCube-v0":
    #     env = gym.wrappers.FlattenObservation(env)

    rng, sampling_rng = jax.random.split(rng)
    agent: SACAgent = make_sac_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
    )

    if FLAGS.loaded_model: 
        assert FLAGS.loaded_model is not None
        poi = checkpoints.restore_checkpoint(
            FLAGS.loaded_model, target=agent.state, step=22000000
        )
        agent = agent.replace(state=poi)

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent: SACAgent = jax.device_put(
        jax.tree_util.tree_map(jnp.array, agent), sharding.replicate()
    )

    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        replay_buffer = make_replay_buffer(
            env,
            capacity=FLAGS.replay_buffer_capacity,
            rlds_logger_path=FLAGS.log_rlds_path,
            type="replay_buffer",
            preload_rlds_path=FLAGS.preload_rlds_path,
        )
        replay_iterator = replay_buffer.get_iterator(
            sample_args={
                "batch_size": FLAGS.batch_size * FLAGS.critic_actor_ratio,
            },
            device=sharding.replicate(),
        )
        # learner loop
        print_green("starting learner loop")
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            replay_iterator=replay_iterator,
        )

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(2000)  # the queue size on the actor

        # actor loop
        print_green("starting actor loop")
        actor(agent, data_store, env, sampling_rng)

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)
