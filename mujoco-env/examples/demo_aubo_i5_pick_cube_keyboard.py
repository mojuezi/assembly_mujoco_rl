#!/usr/bin/env python3
"""
Aubo i5 抓取方块演示

基于 PickCubeTask 任务和 AuboI5Config 配置。
演示机器人在仿真环境中保持初始位姿，使用 CartesianIKController。

作者: Liu Gang
日期: 2026-01-03
"""

import sys
import time
import argparse
import numpy as np
import mujoco
from pathlib import Path
from mujoco import viewer as mj_viewer
from dm_robotics.transformations import transformations as tr

# 将项目根目录添加到 sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 使用新的模块结构
from mujoco_env.mujoco_env.envs.env_instance import make_aubo_i5_pick_cube_env
from mujoco_env.mujoco_env.robot_config.aubo_i5 import AuboI5Robot
from mujoco_env.mujoco_env.utils import KeyboardIO
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
    parser = argparse.ArgumentParser(description='Aubo i5 抓取方块演示')
    parser.add_argument('--no-render', action='store_true', help='禁用MuJoCo渲染窗口')
    parser.add_argument('--steps', type=int, default=1000, help='每个 episode 的运行步数')
    parser.add_argument('--episodes', type=int, default=5, help='演示的 episode 数量')
    parser.add_argument('--no-plot', action='store_true', help='禁用力传感器示波器')
    args = parser.parse_args()

    render = not args.no_render
    use_sensor = not args.no_plot  # 默认启用传感器和绘图

    print("="*60)
    print("Aubo i5 抓取方块演示 (Cartesian IK + 初始位姿保持)")
    print(f"模式: {'渲染' if render else '无渲染'}, Episode数: {args.episodes}, 每Episode步数: {args.steps}")
    print(f"力传感器: {'启用' if use_sensor else '禁用'}")
    print("="*60)
    
    # 如果启用示波器，导入绘图工具
    plotter = None
    if use_sensor:
        print("正在初始化示波器...")
        try:
            from mujoco_env.mujoco_env.utils.plot_utils import ForceSensorPlotter
            plotter = ForceSensorPlotter()
            print("示波器初始化完成")
        except Exception as e:
            print(f"示波器初始化失败: {e}")

    try:
        # === 获取初始关节配置 ===
        init_qpos = AuboI5Robot.INIT_QPOS
        print(f"初始关节配置 (INIT): {init_qpos}")
        
        # === 创建仿真环境 ===
        print("\n🌍 创建仿真环境...")
        
        env = make_aubo_i5_pick_cube_env(
            scene_name="cubes",  # 使用抓取场景
            include_image=False,    # 可视化不需要图像观测
            image_size=(128, 128),
            include_depth=False,
            use_ft_sensor=use_sensor, # 根据参数决定是否启用力传感器
            control_dt=0.02,
            physics_dt=0.002,
            max_episode_steps=args.steps,
            render_mode="human" if render else "rgb_array",
        )
        
        # 获取机器人配置信息用于显示
        robot_config = env.task.robot_config
        print(f"✅ 仿真环境创建成功")
        print(f"   - 机器人: {robot_config.name}")
        print(f"   - 夹爪: {robot_config.gripper_name}")
        print(f"   - 力传感器: {robot_config.use_ft_sensor}")
        print(f"   - 控制器: {robot_config.controller_type}")
        print(f"   - 任务: {env.task.name}")

        # 验证力传感器是否加载
        if robot_config.use_ft_sensor:
            try:
                force_id = env.model.sensor("force_ee").id
                torque_id = env.model.sensor("torque_ee").id
                print(f"✅ 力传感器加载成功!")
                print(f"   - Force Sensor ID: {force_id}")
                print(f"   - Torque Sensor ID: {torque_id}")
            except Exception as e:
                print(f"❌ 力传感器加载失败: {e}")
        
        print(f"   - 场景: {env.task.scene_name}")
        
        # 确定末端执行器site名称
        ee_site_name = "grip_site" if robot_config.gripper_name else "attachment_site"
        print(f"末端执行器Site: {ee_site_name}")
        print(f"   - 动作空间: {env.action_space}")
        
        # === 4. 运行 Episodes ===
        # 初始化键盘控制器
        print("初始化键盘控制器...")
        try:
            keyboard_recorder = KeyboardIO()
            keyboard_recorder.start()
        except Exception as e:
            print(f"Warning: 无法初始化键盘控制器 (可能是因为在无头模式/Docker中运行): {e}")
            keyboard_recorder = None

        # 获取机器人关节名
        arm_joint_names = ["shoulder_joint", "upperArm_joint", "foreArm_joint", "wrist1_joint", "wrist2_joint", "wrist3_joint"]
        
        for episode in range(args.episodes):
            print(f"\n🚀 开始 Episode {episode + 1}/{args.episodes}...")
            
            # --- Stable Reset (参考 SERL 恢复/重置逻辑) ---
            # 1. 环境重置 (重置物理仿真和任务状态)
            obs, info = env.reset()
            
            # 2. 强制设置机器人到初始关节配置 (确保稳定性)
            for i, name in enumerate(arm_joint_names):
                try:
                    joint_id = env.model.joint(name).id
                    qpos_adr = env.model.jnt_qposadr[joint_id]
                    env.data.qpos[qpos_adr] = init_qpos[i]
                    # 同时重置速度
                    qvel_adr = env.model.jnt_dofadr[joint_id]
                    env.data.qvel[qvel_adr] = 0
                except Exception as e:
                    print(f"Warning: Could not set joint {name}: {e}")
            
            # 3. 推进仿真以更新运动学状态
            mujoco.mj_forward(env.model, env.data)
            
            # 4. 重置控制器状态，同步当前位姿作为控制起点
            env.controller.reset()
            
            # 5. 获取初始末端位姿，作为 Cartesian IK 的目标
            site_id = env.model.site(ee_site_name).id
            initial_pos = env.data.site_xpos[site_id].copy()
            initial_mat = env.data.site_xmat[site_id].reshape(3, 3).copy()
            initial_quat = tr.mat_to_quat(initial_mat)  # [w, x, y, z]
            
            current_target_pose = np.concatenate([initial_pos, initial_quat])
            
            print(f"   Episode {episode+1} 重置完成. 初始 TCP: {initial_pos}")
            
            # --- 运行循环 ---
            if render:
                # 使用MuJoCo原生viewer
                with mj_viewer.launch_passive(env.model, env.data) as viewer:
                    for i in range(args.steps):
                        if not viewer.is_running():
                            break
                            
                        # 处理键盘输入
                        if keyboard_recorder:
                            pos_offset = keyboard_recorder.get_end_pos_offset()
                            rot_offset = keyboard_recorder.get_end_rot_offset()
                            gripper_val = 1.0 if keyboard_recorder.gripper_flag else 0.0
                        else:
                            pos_offset = np.zeros(3)
                            rot_offset = np.eye(3)
                            gripper_val = 0.0
                        
                        # 更新当前目标位姿
                        current_target_pose[:3] += pos_offset
                        
                        # 更新姿态
                        curr_quat = current_target_pose[3:]
                        curr_mat = T.quat_2_mat(curr_quat)
                        new_mat = curr_mat @ rot_offset
                        current_target_pose[3:] = T.mat_2_quat(new_mat)
                        
                        # 获取夹爪状态
                        # gripper_val already set above
                        
                        # 采样动作
                        action = sample_action(env.action_space, target_pose=current_target_pose, gripper_action=gripper_val)

                        obs, reward, terminated, truncated, info = env.step(action)
                        
                        # 更新示波器
                        if plotter is not None:
                            wrench = env.task.get_ft_sensor_data(env.data)
                            plotter.update(wrench[:3], wrench[3:])

                        # 渲染
                        viewer.sync()
                        
                        # 控制帧率
                        time.sleep(env.control_dt)
                        
                        if i % 100 == 0:
                            # 计算当前位姿误差
                            curr_pos = env.data.site_xpos[site_id]
                            err = np.linalg.norm(curr_pos - initial_pos)
                            print(f"   Step {i}: Reward={reward:.3f}, Pos Error={err:.4f}m")
                        
                        if terminated or truncated:
                            print(f"   Episode {episode+1} finished at step {i}")
                            break
            else:
                # 无渲染模式
                for i in range(args.steps):
                    action = sample_action(env.action_space, target_pose=current_target_pose, gripper_action=0.0)
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    if i % 200 == 0:
                        curr_pos = env.data.site_xpos[site_id]
                        err = np.linalg.norm(curr_pos - initial_pos)
                        print(f"   Step {i}: Reward={reward:.3f}, Pos Error={err:.4f}m")
                    
                    if terminated or truncated:
                        print(f"   Episode {episode+1} finished at step {i}")
                        break
            
            print(f"✅ Episode {episode+1} 演示结束")
        
        print("\n🎉 所有演示任务完成")
        
        # 清理
        if 'keyboard_recorder' in locals() and keyboard_recorder:
            keyboard_recorder.stop()
        env.close()
        
        # 等待示波器关闭
        if plotter is not None:
            print("\n请手动关闭示波器窗口以退出程序...")
            plotter.close()
        
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

