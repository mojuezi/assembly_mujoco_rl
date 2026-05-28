#!/usr/bin/env python3
"""
Franka FR3 抓取方块演示

基于 PickCubeTask 任务和 FrankaFR3PickCubeConfig 配置。
演示机器人在仿真环境中保持初始位姿，使用 OperationalSpaceController。

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
from mujoco_env.mujoco_env.envs.env_instance import make_franka_panda_pick_cube_env
from mujoco_env.mujoco_env.robot_config.franka_panda import FrankaPandaRobot
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
    # 注意：PickCubeTask的动作空间通常是 7+1=8 维 (如果使用OSC/Cartesian)
    # 但如果是joint_position，则是 7+1=8 维 (7个关节 + 1个夹爪)
    # 这里我们使用 OSC，所以动作应该是笛卡尔目标
    
    # 确保 target_pose 是 (7,)
    if len(target_pose) != 7:
        raise ValueError(f"Expected target_pose of length 7, got {len(target_pose)}")
        
    action = np.concatenate([target_pose, [gripper_action]])
    return action


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Franka Panda 抓取方块演示')
    parser.add_argument('--no-render', action='store_true', help='禁用MuJoCo渲染窗口')
    parser.add_argument('--steps', type=int, default=1000, help='运行步数')
    args = parser.parse_args()

    render = not args.no_render

    print("="*60)
    print("Franka Panda 抓取方块演示 (OSC + 初始位姿保持)")
    print("="*60)
    
    try:
        # === 获取初始关节配置 ===
        # HOME_QPOS 定义在 FrankaPandaRobot 中
        home_qpos = FrankaPandaRobot.HOME_QPOS
        print(f"初始关节配置 (HOME): {home_qpos}")
        
        # === 创建仿真环境 ===
        print("\n🌍 创建仿真环境...")
        
        env = make_franka_panda_pick_cube_env(
            scene_name="cubes",  # 使用抓取场景
            include_image=False,    # 可视化不需要图像观测
            image_size=(128, 128),
            include_depth=False,
            control_dt=0.02,  # 提高控制频率到 50Hz
            physics_dt=0.002,
            max_episode_steps=500,
            render_mode="human" if render else "rgb_array",
            # 传递零空间目标关节配置（opspace函数的joint参数）
            target_joint=home_qpos,  # 传递给OperationalSpaceController的target_joint参数
            # 调整OSC参数以减少震荡
            # pos_gains=(100.0, 100.0, 100.0), # 降低Kp (原200)
            # ori_gains=(100.0, 100.0, 100.0), # 降低Kp (原200)
            # damping_ratio=1.5,               # 增加阻尼 (原1.0)
            # nullspace_stiffness=0.1,         # 降低零空间刚度 (原0.5)
        )
        
        # 获取机器人配置信息用于显示
        robot_config = env.task.robot_config
        print(f"✅ 仿真环境创建成功")
        print(f"   - 机器人: {robot_config.name}")
        print(f"   - 夹爪: {robot_config.gripper_name}")
        print(f"   - 控制器: {robot_config.controller_type}")
        print(f"   - 任务: {env.task.name}")
        print(f"   - 场景: {env.task.scene_name}")
        
        # 确定末端执行器site名称 (用于内部计算奖励等)
        ee_site_name = "grip_site" if robot_config.gripper_name else "attachment_site"
        print(f"末端执行器Site: {ee_site_name}")
        print(f"   - 动作空间: {env.action_space}")
        
        # DEBUG: 打印XML路径
        print(f"DEBUG: XML Path: {env.xml_path}")
        
        # === 4. 初始化状态 ===
        obs, info = env.reset()
        
        # 设置机器人到初始关节配置 (修复索引偏移问题)
        print(f"设置机器人到初始 HOME 配置...")
        # 自动获取机械臂关节名：寻找前 7 个非 free 关节
        arm_joint_names = []
        for i in range(env.model.njnt):
            if env.model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE:
                arm_joint_names.append(env.model.joint(i).name)
            if len(arm_joint_names) == 7:
                break
        
        print(f"检测到机械臂关节: {arm_joint_names}")
        
        for i, name in enumerate(arm_joint_names):
            try:
                joint_id = env.model.joint(name).id
                qpos_adr = env.model.jnt_qposadr[joint_id]
                env.data.qpos[qpos_adr] = home_qpos[i]
                qvel_adr = env.model.jnt_dofadr[joint_id]
                env.data.qvel[qvel_adr] = 0
            except Exception as e:
                print(f"Warning: Could not set joint {name}: {e}")
        
        # 推进仿真以更新运动学
        mujoco.mj_forward(env.model, env.data)

        # 重置控制器状态，使用当前位置作为初始参考
        env.controller.reset()
        
        # 获取当前(初始)末端位姿，作为OSC的目标
        site_id = env.model.site(ee_site_name).id
        initial_pos = env.data.site_xpos[site_id].copy()
        initial_mat = env.data.site_xmat[site_id].reshape(3, 3).copy()
        initial_quat = tr.mat_to_quat(initial_mat) # [w, x, y, z]
        
        # 构建初始目标位姿 (7,)
        initial_target_pose = np.concatenate([initial_pos, initial_quat])
        current_target_pose = initial_target_pose.copy()

        # 初始化键盘控制器
        print("初始化键盘控制器...")
        try:
            keyboard_recorder = KeyboardIO()
            keyboard_recorder.start()
        except Exception as e:
            print(f"Warning: 无法初始化键盘控制器 (可能是因为在无头模式/Docker中运行): {e}")
            keyboard_recorder = None
        
        print(f"初始 TCP 位置: {initial_pos}")
        print(f"初始 TCP 四元数: {initial_quat}")
        
        print(f"\n🚀 开始运行...")
        
        if render:
            # 使用MuJoCo原生viewer
            print(f"启动 MuJoCo Viewer...")
            
            with mj_viewer.launch_passive(env.model, env.data) as viewer:
                start_time = time.time()
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
                    
                    # 渲染
                    viewer.sync()
                    
                    # 控制帧率
                    time.sleep(env.control_dt)
                    
                    if i % 50 == 0:
                        # 计算当前位姿误差
                        curr_pos = env.data.site_xpos[site_id]
                        err = np.linalg.norm(curr_pos - initial_pos)
                        print(f"Step {i}: Reward={reward:.3f}, Pos Error={err:.4f}m")
                    
                    if terminated or truncated:
                        print(f"Episode finished at step {i}")
                        obs, info = env.reset()
                        # 重置后需要重新设置到 HOME 吗？通常 reset 会处理，但为了 demo 效果，我们这里不再强制设置
                        # 如果需要严格保持 HOME，应该在每次 reset 后再次设置 qpos
                
                print("演示结束")
        else:
            # 无渲染模式
            start_time = time.time()
            for i in range(args.steps):
                action = sample_action(env.action_space, target_pose=initial_target_pose, gripper_action=0.0)
                obs, reward, terminated, truncated, info = env.step(action)
                
                if i % 100 == 0:
                    curr_pos = env.data.site_xpos[site_id]
                    err = np.linalg.norm(curr_pos - initial_pos)
                    print(f"Step {i}: Reward={reward:.3f}, Pos Error={err:.4f}m")
                
                if terminated or truncated:
                    obs, info = env.reset()
            
            print(f"运行完成，耗时: {time.time() - start_time:.2f}s")
        
        # 清理
        if 'keyboard_recorder' in locals() and keyboard_recorder:
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
