#!/usr/bin/env python3
"""
Aubo i5 水果采摘演示

基于 FruitPickTask 任务和 AuboI5FruitPickConfig 配置。
支持自动模式和键盘遥操控模式。

快捷键说明 (需要 --keyboard 参数):
- <ARROW> : 沿 x/y 轴移动末端执行器
- <CTRL + ARROW> : 沿 z 轴移动末端执行器
- <SHIFT + ARROW> : 绕 x/y 轴旋转末端执行器
- <CTRL + SHIFT + ARROW> : 绕 z 轴旋转末端执行器
- <CAPSLOCK> : 切换剪刀开合

始终有效:
- <R> : 重置环境
- <ESC> : 退出

日期: 2026-01-15
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
from mujoco_env.mujoco_env.envs.env_instance import make_aubo_i5_fruit_pick_env
from mujoco_env.mujoco_env.tasks.fruit_pick.aubo_i5_config import AuboI5FruitPickConfig
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
    if len(target_pose) != 7:
        raise ValueError(f"Expected target_pose of length 7, got {len(target_pose)}")
        
    action = np.concatenate([target_pose, [gripper_action]])
    return action


def reset_environment(env, init_qpos, arm_joint_names):
    """
    重置环境和机器人状态
    
    Returns:
        (obs, info, initial_pos, initial_quat, target_pos, pick_site_id)
    """
    obs, info = env.reset()
    
    # 强制设置机器人到初始关节配置
    for i, name in enumerate(arm_joint_names):
        try:
            joint_id = env.model.joint(name).id
            qpos_adr = env.model.jnt_qposadr[joint_id]
            env.data.qpos[qpos_adr] = init_qpos[i]
            qvel_adr = env.model.jnt_dofadr[joint_id]
            env.data.qvel[qvel_adr] = 0
        except Exception:
            pass
    
    # 推进仿真
    mujoco.mj_forward(env.model, env.data)
    
    # 重置控制器
    env.controller.reset()
    
    # 恢复苹果连接
    env.task.restore_apple(env.model, env.data)
    
    # 获取初始末端位姿
    ee_site_name = getattr(env.controller, "ee_site_name", "attachment_site")
    site_id = env.model.site(ee_site_name).id
    initial_pos = env.data.site_xpos[site_id].copy()
    initial_mat = env.data.site_xmat[site_id].reshape(3, 3).copy()
    initial_quat = tr.mat_to_quat(initial_mat)
    
    # 获取目标位置
    pick_site_name = getattr(env.task, "pick_site_name", "pick_site")
    try:
        pick_site_id = env.model.site(pick_site_name).id
        target_pos = env.data.site_xpos[pick_site_id].copy()
    except Exception:
        pick_site_id = None
        target_pos = np.array([0.65, 0.15, 0.85], dtype=np.float32)
    
    return obs, info, initial_pos, initial_quat, target_pos, pick_site_id


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Aubo i5 水果采摘演示')
    parser.add_argument('--no-render', action='store_true', help='禁用MuJoCo渲染窗口')
    parser.add_argument('--steps', type=int, default=10000, help='最大运行步数')
    parser.add_argument('--keyboard', '-k', action='store_true', 
                        help='启用键盘遥操控模式 (方向键控制末端)')
    args = parser.parse_args()

    render = not args.no_render
    keyboard_mode = args.keyboard

    print("=" * 60)
    print("Aubo i5 水果采摘演示")
    print(f"模式: {'键盘遥操控' if keyboard_mode else '自动'}, 渲染: {'是' if render else '否'}")
    print("=" * 60)
    
    try:
        # === 获取初始关节配置 ===
        init_qpos = AuboI5FruitPickConfig.INIT_QPOS
        print(f"初始关节配置 (INIT): {np.rad2deg(init_qpos).round(1)} 度")
        
        # === 创建仿真环境 ===
        print("\n🌍 创建仿真环境...")
        
        env = make_aubo_i5_fruit_pick_env(
            scene_name="fruitpick",
            include_image=False,
            image_size=(128, 128),
            include_depth=False,
            control_dt=0.02,
            physics_dt=0.002,
            max_episode_steps=args.steps,
            render_mode="human" if render else "rgb_array",
        )
        
        robot_config = env.task.robot_config
        print(f"✅ 仿真环境创建成功")
        print(f"   - 机器人: {robot_config.name}")
        print(f"   - 夹爪: {robot_config.gripper_name}")
        print(f"   - 控制器: {robot_config.controller_type}")
        print(f"   - 任务: {env.task.name}")
        print(f"   - 场景: {env.task.scene_name}")
        
        # 确定末端执行器site名称
        ee_site_name = getattr(env.controller, "ee_site_name", "attachment_site")
        print(f"   - 末端执行器Site: {ee_site_name}")
        print(f"   - 动作空间: {env.action_space}")
        
        # 获取机器人关节名
        arm_joint_names = ["shoulder_joint", "upperArm_joint", "foreArm_joint", 
                          "wrist1_joint", "wrist2_joint", "wrist3_joint"]
        
        # 初始化键盘控制器
        print("初始化键盘控制器...")
        keyboard_recorder = None
        try:
            keyboard_recorder = KeyboardIO()
            keyboard_recorder.start()
        except Exception as e:
            print(f"Warning: 无法初始化键盘控制器 (可能是因为在无头模式/Docker中运行): {e}")
            if keyboard_mode:
                print("Error: 键盘模式已启用但无法初始化键盘。切换回自动模式。")
                keyboard_mode = False
        
        # 插值参数 (自动模式)
        duration_steps = 500
        
        # 初始重置
        print(f"\n🚀 开始演示...")
        obs, info, initial_pos, initial_quat, target_pos, pick_site_id = \
            reset_environment(env, init_qpos, arm_joint_names)
        
        site_id = env.model.site(ee_site_name).id
        distance_threshold = getattr(env.task, "distance_threshold", 0.01)
        try:
            pinch_site_id = env.model.site("pinch").id
        except Exception:
            pinch_site_id = site_id
        
        print(f"   重置完成. 初始 TCP: {initial_pos}")
        print(f"   🎯 抓取目标: {target_pos}")
        
        # 当前目标位姿 (用于键盘模式)
        current_target_pose = np.concatenate([initial_pos.copy(), initial_quat.copy()])
        
        # 剪刀状态
        scissors_closing = False
        prev_scissors_closing = False
        close_start_time = None
        
        # --- 运行循环 ---
        if render:
            with mj_viewer.launch_passive(env.model, env.data) as viewer:
                task_done = False
                step = 0
                
                while viewer.is_running():
                    if keyboard_recorder:
                        # ESC 退出
                        if keyboard_recorder.exit_flag:
                            print("收到退出信号 (ESC)")
                            break
                        
                        # R 重置
                        if keyboard_recorder.reset_flag:
                            keyboard_recorder.reset_flag = False
                            print("\n🔄 重置环境...")
                            obs, info, initial_pos, initial_quat, target_pos, pick_site_id = \
                                reset_environment(env, init_qpos, arm_joint_names)
                            current_target_pose = np.concatenate([initial_pos.copy(), initial_quat.copy()])
                            task_done = False
                            step = 0
                            scissors_closing = False
                            prev_scissors_closing = False
                            close_start_time = None
                            print(f"   重置完成. 初始 TCP: {initial_pos}")
                            print(f"   🎯 抓取目标: {target_pos}")
                            continue
                        
                        # === 更新目标位姿 ===
                        if keyboard_mode:
                            # 键盘遥操控模式
                            pos_offset = keyboard_recorder.get_end_pos_offset()
                            rot_offset = keyboard_recorder.get_end_rot_offset()
                            
                            # 更新位置
                            current_target_pose[:3] += pos_offset
                            
                            # 更新姿态
                            curr_quat = current_target_pose[3:]
                            curr_mat = T.quat_2_mat(curr_quat)
                            new_mat = curr_mat @ rot_offset
                            current_target_pose[3:] = T.mat_2_quat(new_mat)
                            
                            # CAPSLOCK 控制剪刀开合
                            scissors_closing = bool(keyboard_recorder.gripper_flag)
                        else:
                            # 自动模式：线性插值到目标
                            if not env.task.is_cut and pick_site_id is not None:
                                target_pos = env.data.site_xpos[pick_site_id].copy()
                            
                            alpha = min(step / duration_steps, 1.0)
                            current_target_pos = (1 - alpha) * initial_pos + alpha * target_pos
                            current_target_pose = np.concatenate([current_target_pos, initial_quat])
                    else:
                        # 没有键盘控制器，强制使用自动模式逻辑
                        if not env.task.is_cut and pick_site_id is not None:
                            target_pos = env.data.site_xpos[pick_site_id].copy()
                        
                        alpha = min(step / duration_steps, 1.0)
                        current_target_pos = (1 - alpha) * initial_pos + alpha * target_pos
                        current_target_pose = np.concatenate([current_target_pos, initial_quat])
                    
                    # 自动模式下：靠近目标后触发一次剪刀闭合
                    curr_pinch_pos = env.data.site_xpos[pinch_site_id]
                    dist_to_target = np.linalg.norm(curr_pinch_pos - target_pos)
                    if not keyboard_mode and not scissors_closing and dist_to_target < distance_threshold:
                        scissors_closing = True
                    
                    closing_edge = (not prev_scissors_closing) and scissors_closing
                    if closing_edge:
                        close_start_time = time.monotonic()
                    if not scissors_closing:
                        close_start_time = None

                    # === 执行动作 ===
                    gripper_action = 1.0 if scissors_closing else 0.0
                    
                    action = sample_action(env.action_space, 
                                          target_pose=current_target_pose, 
                                          gripper_action=gripper_action)
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    # === 渲染 ===
                    viewer.sync()
                    time.sleep(env.control_dt)
                    
                    # === 判断任务完成 ===
                    curr_pos = env.data.site_xpos[pinch_site_id]
                    dist = np.linalg.norm(curr_pos - target_pos)
                    
                    if step % 100 == 0:
                        print(f"   Step {step}: 距离目标 {dist:.3f}m, Reward={reward:.3f}")
                    
                    if (not task_done and scissors_closing and close_start_time is not None and
                            (time.monotonic() - close_start_time) >= 1.0 and dist < distance_threshold):
                        print(f"   ✅ 任务完成! 距离 {dist:.4f}m < {distance_threshold:.3f}m 且剪刀闭合 1s")
                        if env.task.cut_apple(env.model, env.data):
                            print("   🍎 苹果已剪断，正在掉落...")
                        task_done = True
                    
                    if truncated and not task_done:
                        print("   ⏱️ 任务超时")
                        task_done = True
                    
                    prev_scissors_closing = scissors_closing
                    step += 1
                
                print("\n🎯 演示结束")
        else:
            # 无渲染模式 (仅自动模式)
            for step in range(args.steps):
                if pick_site_id is not None and not env.task.is_cut:
                    target_pos = env.data.site_xpos[pick_site_id].copy()

                alpha = min(step / duration_steps, 1.0)
                current_target_pos = (1 - alpha) * initial_pos + alpha * target_pos
                current_target_pose = np.concatenate([current_target_pos, initial_quat])

                curr_pinch_pos = env.data.site_xpos[pinch_site_id]
                dist_to_target = np.linalg.norm(curr_pinch_pos - target_pos)
                if not scissors_closing and dist_to_target < distance_threshold:
                    scissors_closing = True
                closing_edge = (not prev_scissors_closing) and scissors_closing
                if closing_edge:
                    close_start_time = time.monotonic()
                if not scissors_closing:
                    close_start_time = None

                gripper_action = 1.0 if scissors_closing else 0.0
                
                action = sample_action(env.action_space, 
                                       target_pose=current_target_pose, 
                                       gripper_action=gripper_action)
                obs, reward, terminated, truncated, info = env.step(action)
                
                curr_pos = env.data.site_xpos[pinch_site_id]
                dist = np.linalg.norm(curr_pos - target_pos)
                
                if step % 100 == 0:
                    print(f"   Step {step}: 距离目标 {dist:.3f}m, Reward={reward:.3f}")

                if (scissors_closing and close_start_time is not None and
                        (time.monotonic() - close_start_time) >= 1.0 and dist < distance_threshold):
                    print(f"   ✅ 任务完成! 距离 {dist:.4f}m < {distance_threshold:.3f}m 且剪刀闭合 1s")
                    if env.task.cut_apple(env.model, env.data):
                        print("   🍎 苹果已剪断")
                    break
                if truncated:
                    print("   ⏱️ 任务超时")
                    break
                
                prev_scissors_closing = scissors_closing
        
        # 清理
        if keyboard_recorder:
            keyboard_recorder.stop()
        env.close()

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        if keyboard_recorder:
            try:
                keyboard_recorder.stop()
            except:
                pass
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
