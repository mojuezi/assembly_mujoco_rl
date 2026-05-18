#!/usr/bin/env python3
"""
Aubo i5 

基于 PegInsertionTask 任务和 AuboI5Config 配置。
演示机器人在仿真环境中保持初始位姿，使用 CartesianIKController。

支持轨迹录制功能：
- 添加 --record-trajectory 参数启用轨迹录制
- 使用 --trajectory-dir 指定保存目录
- 轨迹自动保存为 .npz 格式
- 可使用 trajectory_replay.py 回放录制的轨迹

使用示例:
    # 录制轨迹
    python demo_aubo_i5_assemble_hole.py --record-trajectory --steps 300

    # 回放轨迹
    python mujoco_env/tasks/peg_insertion/trajectory_replay.py trajectories/episode_0000.npz

作者: Liu Gang
日期: 2026-01-07
"""

import sys
import time
import argparse
import numpy as np
import mujoco
from pathlib import Path
from mujoco import viewer as mj_viewer
from dm_robotics.transformations import transformations as tr

import pdb

# 将项目根目录添加到 sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 使用新的模块结构（直接从 envs 导入，避免包命名歧义）
from mujoco_env.mujoco_env.envs.env_instance import make_aubo_i5_assemble_hole_env
from mujoco_env.mujoco_env.tasks.peg_insertion.aubo_i5_config import AuboI5Config
from mujoco_env.mujoco_env.robot_config.aubo_i5 import AuboI5Robot
from mujoco_env.mujoco_env.utils import KeyboardIO
from mujoco_env.mujoco_env.tasks.peg_insertion.trajectory_recorder import TrajectoryRecorder
import mujoco_env.mujoco_env.utils.transform as T


def sample_action(action_space, target_pose=None, gripper_action=0.0):
    """
    返回固定的目标动作

    Args:
        action_space: 动作空间
        target_pose: 目标位姿 (7,) [pos, quat]
        gripper_action: 夹爪动作

    Returns:
        action: 动作向量
    """
    if target_pose is None:
        return np.zeros(action_space.shape)

    # 构造动作：[pos(3), quat(4), gripper(1)]
    # 确保 target_pose 是 (7,)
    if len(target_pose) != 7:
        raise ValueError(f"Expected target_pose of length 7, got {len(target_pose)}")

    action = np.concatenate([target_pose, [gripper_action]])
    return action


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Aubo i5 装配孔演示')
    parser.add_argument('--no-render', action='store_true', help='禁用MuJoCo渲染窗口')
    parser.add_argument('--steps', type=int, default=100000, help='运行步数')
    parser.add_argument('--episodes', type=int, default=5, help='演示的 episode 数量')
    parser.add_argument('--record-trajectory', action='store_true', help='启用轨迹录制功能')
    parser.add_argument('--trajectory-dir', type=str, default='./trajectories', help='轨迹文件保存目录')
    args = parser.parse_args()

    render = args.no_render

    print("="*60)
    print("Aubo i5 装配")
    print("="*60)

    try:
        # === 1. 创建机器人配置 ===
        print("\n🤖 创建机器人配置...")
        robot_config = AuboI5Config.get_config()

        # 获取初始关节位置
        init_qpos = np.array([167.0 / 180.0 * np.pi,  -2.40 / 180.0 * np.pi, 120.0 / 180.0 * np.pi, 31.0 / 180.0 * np.pi, 90.7 / 180.0 * np.pi, -3.0 / 180.0 * np.pi])
        print(f"初始关节配置 (INIT): {init_qpos}")

        print(f"✅ 机器人配置创建成功")
        print(f"   - 机器人: {robot_config.name}")
        print(f"   - 夹爪: {robot_config.gripper_name}")
        print(f"   - 控制器: {robot_config.controller_type}")

        # === 3. 创建仿真环境 ===
        print("\n🌍 创建仿真环境...")

        # 确定末端执行器site名称
        ee_site_name = "grip_site" if robot_config.gripper_name else "attachment_site"
        print(f"末端执行器Site: {ee_site_name}")

        env = make_aubo_i5_assemble_hole_env(
            scene_name="assemble_hole",
            include_image=False,
            image_size=(128, 128),
            include_depth=False,
            control_dt=0.02,
            physics_dt=0.002,
            max_episode_steps=args.steps,
            render_mode="human" if render else "rgb_array", 
            mode="real", 
            robot_ip="192.168.1.100", 
            hole_position_real= np.array([-0.518,0.103,0.055]), # all the positions are from the view of mount
            workspace_low_real= np.array([-0.67,-0.03,0.04]), 
            workspace_high_real= np.array([-0.354,0.41,0.317]),
        )

        print(f"✅ 环境创建成功")
        print(f"   - 动作空间: {env.action_space}")
        print(f"✅ 任务配置创建成功")
        print(f"   - 任务: {env.task.name}")
        print(f"   - 场景: {env.task.scene_name}")

        # pdb.set_trace()

        if env.robot_interface.connect(): 
            env.robot_interface.power_on()

        # === 4. 初始化轨迹记录器 ===
        trajectory_recorder = None
        if args.record_trajectory:
            print(f"\n📹 初始化轨迹记录器...")
            trajectory_recorder = TrajectoryRecorder(
                robot_name="aubo_i5",
                dof=6,
                save_dir=args.trajectory_dir,
                control_freq=1.0 / env.control_dt,
                auto_save_on_episode_end=True
            )
            # 设置环境信息用于回放
            trajectory_recorder.set_env_info(
                xml_path=str(env.xml_path),
                scene_name=env.task.scene_name,
                task_name=env.task.name,
                gripper_name=robot_config.gripper_name,
                joint_names=["shoulder_joint", "upperArm_joint", "foreArm_joint", "wrist1_joint", "wrist2_joint", "wrist3_joint"],
                initial_qpos=AuboI5Robot.INIT_QPOS
            )
            print(f"✅ 轨迹记录器初始化成功")
            print(f"   - 保存目录: {args.trajectory_dir}")
            print(f"   - 控制频率: {1.0 / env.control_dt:.1f} Hz")

        # === 5. 运行 Episodes（与 pick_cube 演示保持一致） ===

        # 初始化键盘控制器
        print("初始化键盘控制器...")
        keyboard_recorder = KeyboardIO()
        keyboard_recorder.start()

        # 获取机器人关节名
        arm_joint_names = ["shoulder_joint", "upperArm_joint", "foreArm_joint", "wrist1_joint", "wrist2_joint", "wrist3_joint"]

        for episode in range(getattr(args, "episodes", 1)):
            print(f"\n🚀 开始 Episode {episode + 1}/{getattr(args, 'episodes', 1)}...")

            # 开始新的轨迹录制
            if trajectory_recorder:
                trajectory_recorder.new_episode(episode)

            # --- Stable Reset ---
            obs, info = env.reset()

            # # 强制设置机器人到初始关节配置
            # for i, name in enumerate(arm_joint_names):
            #     try:
            #         joint_id = env.model.joint(name).id
            #         qpos_adr = env.model.jnt_qposadr[joint_id]
            #         env.data.qpos[qpos_adr] = init_qpos[i]
            #         # 同时重置速度
            #         qvel_adr = env.model.jnt_dofadr[joint_id]
            #         env.data.qvel[qvel_adr] = 0
            #     except Exception as e:
            #         print(f"Warning: Could not set joint {name}: {e}")

            # ret = env.robot_interface.enable_servo_mode(True)
            # print(ret)
            # if ret:
            #     ret = env.robot_interface.servo_to_joint_positions(init_qpos)
            #     print(ret)
            # env.robot_interface.enable_servo_mode(False)

            env.robot_interface.move_to_joint_positions(positions=init_qpos, 
                                                        velocity=0.5,
                                                        acceleration=0.5,
                                                        blocking=True,)
            # env.controller.reset()

            # # 获取初始末端位姿作为目标
            # try:
            #     site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
            #     if site_id != -1:
            #         initial_pos = env.data.site_xpos[site_id].copy()
            #         initial_mat = env.data.site_xmat[site_id].reshape(3, 3).copy()
            #         initial_quat = tr.mat_to_quat(initial_mat)
            #         current_target_pose = np.concatenate([initial_pos, initial_quat])
            #     else:
            #         print(f"Warning: Site '{ee_site_name}' not found, using default pose")
            #         current_target_pose = np.array([0.5, 0, 0.6, 1, 0, 0, 0])  # 默认位姿
            #         initial_pos = current_target_pose[:3]
            # except Exception as e:
            #     print(f"Warning: Failed to get EE pose: {e}, using default")
            #     current_target_pose = np.array([0.5, 0, 0.6, 1, 0, 0, 0])  # 默认位姿
            #     initial_pos = current_target_pose[:3]

            current_target_pose = env.robot_interface.get_tcp_pose()
            initial_pos = current_target_pose[:3]

            print(f"   Episode {episode+1} 重置完成. 初始 TCP: {initial_pos}")

            # 运行循环
            
            for i in range(args.steps):

                pos_offset = keyboard_recorder.get_end_pos_offset()
                rot_offset = keyboard_recorder.get_end_rot_offset()

                # pos_offset = np.array([0.001, 0.001, 0.001])

                current_target_pose[:3] += pos_offset

                curr_quat = current_target_pose[3:]
                curr_mat = T.quat_2_mat(curr_quat)
                new_mat = curr_mat @ rot_offset
                current_target_pose[3:] = T.mat_2_quat(new_mat)

                gripper_val = 1.0 if keyboard_recorder.gripper_flag else 0.0
                action = sample_action(env.action_space, target_pose=current_target_pose, gripper_action=gripper_val)

                obs, reward, terminated, truncated, info = env.step(action[:-1])
                desired_goal = env._desired_goal
                print(action)

                # 记录关节位置
                if trajectory_recorder:
                    qpos = obs["qpos"][:6]  # 只记录前6个关节（机械臂关节）
                    trajectory_recorder.record(qpos)

                time.sleep(env.control_dt)

                if i % 10 == 0:
                    curr_pos = env.robot_interface.get_tcp_pose()[:3]
                    err = np.linalg.norm(curr_pos - initial_pos)
                    pos_offset = keyboard_recorder.get_end_pos_offset()
                    # print(f"   Step {i}: target_pos={current_target_pose[:3]}, keyboard_offset={pos_offset}, current_tcp={curr_pos}, reward={reward:.3f}")
                # print(f"current_tcp={curr_pos} action={action}")
                # print(f"terminated={terminated} truncated={truncated}")
                # wrench = obs["wrench"]
                # print(wrench[:3])

                if terminated or truncated:
                    print(f"   Episode {episode+1} finished at step {i}")
                    break
           
            print(f"✅ Episode {episode+1} 演示结束")

        print("\n🎉 所有演示任务完成")

        # 轨迹录制总结
        if trajectory_recorder:
            total_saved = trajectory_recorder.get_episode_count()
            print(f"\n📊 轨迹录制总结:")
            print(f"   - 录制 episodes: {total_saved}")
            print(f"   - 保存目录: {args.trajectory_dir}")
            print(f"   - 使用轨迹回放: python mujoco_env/tasks/peg_insertion/trajectory_replay.py {args.trajectory_dir}/episode_0000.npz")

        # 清理
        if 'keyboard_recorder' in locals():
            keyboard_recorder.stop()
        env.close()

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
