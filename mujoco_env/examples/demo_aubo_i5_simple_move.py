#!/usr/bin/env python3
"""
# Aubo i5机器人 MoveL / MoveJ 演示脚本
#
# 本示例支持两种运动模式：
# - MoveL：在笛卡尔空间（末端执行器直线插值）实现直线路径运动，通过逆解控制末端沿空间直线移动，适合精密轨迹跟踪
# - MoveJ：在关节空间采用插值，实现各关节角的平滑移动，末端轨迹通常是曲线，适合一般点到点运动
#
# 功能演示包括：
# - 选择MoveL或MoveJ模式，规划和仿真不同类型的机器人运动
# - MoveL模式下自动生成并可视化末端空间直线轨迹及其跟踪效果
# - MoveJ模式下展示关节插值移动及相关路径表现
#
# 本脚本适用于机器人路径规划、直线插补、轨迹可视化等教学和测试场景。
"""

import sys
from pathlib import Path

# 将项目根目录添加到 sys.path
# 脚本位于 mujoco_env/examples/ 下，向上两级即为项目根目录 mujoco_env/
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import mujoco
from mujoco import viewer as mj_viewer
import argparse

# 使用新的模块结构
from mujoco_env.mujoco_env.envs.env_instance import make_aubo_i5_simple_move_env
from mujoco_env.mujoco_env.planners.otg_interpolator import OTG


def _add_trajectory_point(viewer, position, point_id):
    """
    在MuJoCo viewer中添加轨迹点
    
    Args:
        viewer: MuJoCo viewer实例
        position: 3D位置 [x, y, z]
        point_id: 点的ID（用于颜色变化）
    """
    # 颜色渐变：从红色到绿色
    color_ratio = min(point_id / 50.0, 1.0)  # 最多50个点后颜色饱和
    rgba = [1.0 - color_ratio, color_ratio, 0.2, 0.8]  # [R, G, B, A]
    
    # 在viewer的user scene中添加几何体
    if hasattr(viewer, 'user_scn') and viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
        geom_id = viewer.user_scn.ngeom
        
        # 初始化几何体为球体
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[geom_id],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.01, 0, 0],  # 球体半径 1cm
            pos=position,
            mat=np.eye(3).flatten(),
            rgba=rgba
        )
        
        # 增加几何体计数
        viewer.user_scn.ngeom += 1


def move_linear(env, start_pose, end_pose, steps=200, viewer=None):
    """
    执行笛卡尔空间直线运动
    
    Args:
        env: 机器人环境
        start_pose: 起始位姿 [x, y, z, qw, qx, qy, qz]
        end_pose: 结束位姿 [x, y, z, qw, qx, qy, qz]
        steps: 运动步数
        viewer: MuJoCo viewer实例 (可选)
    """
    print(f"  执行 {steps} 步运动: 从 [{start_pose[0]:.3f}, {start_pose[1]:.3f}, {start_pose[2]:.3f}] 到 [{end_pose[0]:.3f}, {end_pose[1]:.3f}, {end_pose[2]:.3f}]")
    
    controller = env.controller
    
    # 初始化轨迹点计数器
    trajectory_points = []
    
    for i in range(steps):
        # 如果提供了viewer且viewer已关闭，则停止
        if viewer is not None and not viewer.is_running():
            break

        # 线性插值位姿
        alpha = (i + 1) / steps  # 使用 i+1 确保最后一步到达目标
        interpolated_pose = start_pose + alpha * (end_pose - start_pose)
        
        action = interpolated_pose
        
        # 执行环境步进
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 检查是否需要重置
        if terminated or truncated:
            obs, info = env.reset()
        
        # 添加轨迹点可视化（每10步添加一个点）
        if i % 10 == 0 and viewer is not None:
            # 获取当前末端执行器位置
            current_tcp_pos, _ = controller.forward_kinematics()
            trajectory_points.append(current_tcp_pos.copy())
            
            # 在viewer中添加可视化球体
            _add_trajectory_point(viewer, current_tcp_pos, len(trajectory_points))
        
        # 渲染
        if viewer is not None:
            viewer.sync()
            # 控制仿真速度
            import time
            time.sleep(0.02)  # 50Hz显示频率


def run_moveJ_demo(env, viewer=None):
    """
    运行关节空间运动演示（参考 demo_aubo_movej.py）
    
    Args:
        env: 仿真环境
        viewer: MuJoCo viewer
    """
    print("\n🎯 开始关节空间运动演示")
    print("循环5次: home -> joint1 -> joint2 -> home")

    # 定义目标关节位置
    # Home位置
    joint_home = np.array([
        0.0 / 180.0 * np.pi,           # J1: 0°
        -15.0 / 180.0 * np.pi,         # J2: -15°
        100.0 / 180.0 * np.pi,         # J3: 100°
        25.0 / 180.0 * np.pi,          # J4: 25°
        90.0 / 180.0 * np.pi,          # J5: 90°
        0.0 / 180.0 * np.pi            # J6: 0°
    ])
    
    # 位置1 - 所有关节90°
    joint1 = np.array([
        90.0 / 180.0 * np.pi,          # J1: 90°
        90.0 / 180.0 * np.pi,          # J2: 90°
        90.0 / 180.0 * np.pi,          # J3: 90°
        90.0 / 180.0 * np.pi,          # J4: 90°
        90.0 / 180.0 * np.pi,          # J5: 90°
        90.0 / 180.0 * np.pi           # J6: 90°
    ])
    
    # 位置2 - 所有关节-90°
    joint2 = np.array([
        -90.0 / 180.0 * np.pi,         # J1: -90°
        -90.0 / 180.0 * np.pi,         # J2: -90°
        -90.0 / 180.0 * np.pi,         # J3: -90°
        -90.0 / 180.0 * np.pi,         # J4: -90°
        -90.0 / 180.0 * np.pi,         # J5: -90°
        -90.0 / 180.0 * np.pi          # J6: -90°
    ])

    print(f"Home位置: {np.rad2deg(joint_home).round(1)} 度")
    print(f"位置1: {np.rad2deg(joint1).round(1)} 度")
    print(f"位置2: {np.rad2deg(joint2).round(1)} 度")

    # 初始化OTG
    print("初始化在线轨迹生成器")
    # 注意：这里的参数来自 demo_aubo_movej.py
    otg = OTG(
        OTG_dim=env.robot_config.dof,  # 这里改成机器人的自由度
        control_cycle=0.005,  
        max_velocity=1.0, 
        max_acceleration=5.0, 
        max_jerk=8.0
    )
    
    # 设置当前状态（假设从Home开始）
    # 但首先我们要确保机器人处于Home位置
    env.data.qpos[:6] = joint_home
    env.data.qvel[:6] = 0
    mujoco.mj_forward(env.model, env.data)
    
    otg.set_params(joint_home, np.zeros(6))

    def _move_to_joint_target(target_joint, description):
        """执行关节空间运动到目标位置"""
        print(f"目标: {description}")
        otg.update_target_position(target_joint)
        
        step_count = 0
        max_steps = 2000  # 防止死循环
        
        while step_count < max_steps:
            step_count += 1
            
            # 检查viewer状态
            if viewer is not None and not viewer.is_running():
                return False

            target_q, _ = otg.update_state()
            target_q = np.array(target_q)
            env.step(target_q)
            
            # 添加轨迹点可视化（每10步添加一个点）
            if step_count % 10 == 0 and viewer is not None:
                # 获取当前末端执行器位置
                if hasattr(env.controller, 'forward_kinematics'):
                    current_tcp_pos, _ = env.controller.forward_kinematics()
                else:
                    # 对于JointPositionController，直接从data中获取
                    try:
                        current_tcp_pos = env.data.site("grip_site").xpos.copy()
                    except:
                        # 如果找不到site，尝试使用最后一个body
                        current_tcp_pos = env.data.xpos[-1].copy()

                # 在viewer中添加可视化球体
                _add_trajectory_point(viewer, current_tcp_pos, step_count // 10)
            
            # 获取当前关节位置
            curr_joint = env.data.qpos[:6]  # 前6个关节
            
            # 计算误差
            error = np.abs(curr_joint - target_joint)
            
            # 渲染
            if viewer is not None:
                viewer.sync()
                import time
                time.sleep(0.005) # 稍微快一点，匹配control_cycle
            
            # 检查是否到达（放宽误差阈值：0.01 rad ≈ 0.57°）
            if np.all(error < 0.01):
                print(f"  到达 {description.split()[0]}, 最大误差: {np.rad2deg(error.max()):.3f}°")
                # 更新OTG状态以准备下一次运动
                otg.set_params(curr_joint, env.data.qvel[:6])
                return True
        
        otg.set_params(curr_joint, env.data.qvel[:6])
        return True

    try:
        # 循环2次
        for cycle in range(2):
            print(f"\n=== 循环 {cycle + 1}/2 ===")
            
            # 运动1: 移动到 joint1
            if not _move_to_joint_target(joint1, "joint1 (所有关节 90°)"):
                return
            
            # 运动2: 移动到 joint2
            if not _move_to_joint_target(joint2, "joint2 (所有关节 -90°)"):
                return
            
            # 运动3: 回到 Home
            if not _move_to_joint_target(joint_home, "Home"):
                return

    except KeyboardInterrupt:
        print("\n用户中断演示")


def run_moveL_demo(env, viewer=None):
    """
    运行笛卡尔直线运动演示
    
    Args:
        env: 仿真环境
        viewer: MuJoCo viewer
    """
    print("\n🎯 开始笛卡尔直线运动演示")
    print("执行矩形轨迹: 上 -> 右 -> 下 -> 左")

    joint = np.array([
        0.0,                           # J1: 0°
        -15.0 / 180.0 * np.pi,        # J2: -15°
        100.0 / 180.0 * np.pi,        # J3: 100°
        25.0 / 180.0 * np.pi,         # J4: 25°
        90.0 / 180.0 * np.pi,         # J5: 90°
        0.0 / 180.0 * np.pi           # J6: 0°
    ])

    
    # 设置机器人到初始配置
    controller = env.controller
    
    # 先将机器人移动到指定的初始关节配置
    print(f"设置机器人到初始配置: {np.degrees(joint)} 度")
    env.data.qpos[:6] = joint
    env.data.qvel[:6] = 0
    import mujoco
    mujoco.mj_forward(env.model, env.data)
    
    # 获取当前TCP位姿作为起始点
    current_pos, current_quat = controller.forward_kinematics()
    
    # 构建初始位姿 [x, y, z, qw, qx, qy, qz]
    pose = np.concatenate([current_pos, current_quat])
    print(f"当前TCP位置: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}] m")
    print(f"当前TCP姿态: [{current_quat[0]:.3f}, {current_quat[1]:.3f}, {current_quat[2]:.3f}, {current_quat[3]:.3f}]")
    print(f"DEBUG: current_pos维度: {current_pos.shape}, current_quat维度: {current_quat.shape}")
    print(f"DEBUG: pose维度: {pose.shape}, pose内容: {pose}")
    
    try:
        # 运动1: 向上移动 0.2m
        print("\n运动1: 向上移动 0.2m")
        pose1 = pose.copy()
        pose1[2] += 0.2  # Z方向 +0.2m
        move_linear(env, pose, pose1, steps=100, viewer=viewer)
        print(f"  完成运动1")
        
        # 运动2: 向右移动 0.2m  
        print("\n运动2: 向右移动 0.2m")
        pose2 = pose1.copy()
        pose2[1] += 0.2  # Y方向 +0.2m
        move_linear(env, pose1, pose2, steps=100, viewer=viewer)
        print(f"  完成运动2")
        
        # 运动3: 向下移动 0.2m
        print("\n运动3: 向下移动 0.2m")
        pose3 = pose2.copy()
        pose3[2] -= 0.2  # Z方向 -0.2m
        move_linear(env, pose2, pose3, steps=100, viewer=viewer)
        print(f"  完成运动3")
        
        # 运动4: 向左移动 0.2m（回到起始Y位置）
        print("\n运动4: 向左移动 0.2m")
        pose4 = pose3.copy()
        pose4[1] -= 0.2  # Y方向 -0.2m
        move_linear(env, pose3, pose4, steps=100, viewer=viewer)
        print(f"  完成运动4")
        
        print("\n✅ 完成矩形轨迹！")
        print("末端执行器已完成一个矩形路径运动")
        
        # 验证最终位置
        final_pos, final_quat = controller.forward_kinematics()
        print(f"最终TCP位置: [{final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}] m")
        print(f"位置误差: [{abs(final_pos[0]-current_pos[0]):.3f}, {abs(final_pos[1]-current_pos[1]):.3f}, {abs(final_pos[2]-current_pos[2]):.3f}] m")
    
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断演示")


def main():
    """主函数：使用新的4层架构创建并演示机器人运动"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Aubo i5机器人运动演示')
    parser.add_argument('--no-render', action='store_true', help='禁用MuJoCo渲染窗口')
    parser.add_argument('--mode', type=str, choices=['moveL', 'moveJ'], help='运动模式: moveL 或 moveJ')
    args = parser.parse_args()

    # 确定模式
    if args.mode:
        mode = args.mode
    else:
        # 交互式选择模式
        print("\n请选择运动模式:")
        print("1: moveL (笛卡尔直线运动)")
        print("2: moveJ (关节空间运动)")
        
        try:
            user_input = input("请输入选项 (1 或 2): ").strip()
            if user_input == '2':
                mode = 'moveJ'
            else:
                mode = 'moveL'
                if user_input != '1':
                    print("输入无效或为空，默认使用 moveL 模式")
        except EOFError:
            print("无法获取输入 (可能是CI环境)，默认使用 moveL 模式")
            mode = 'moveL'

    render = not args.no_render

    print("="*60)
    print(f"Aubo i5 机器人运动演示 - 模式: {mode}")
    print("="*60)
    
    try:
        # === 创建仿真环境 ===
        print("\n🌍 创建仿真环境...")
        
        env = make_aubo_i5_simple_move_env(
            mode=mode,
            scene_name="default",  # 使用default场景，包含地面和光照
            include_image=False,  # 可视化暂时不需要图像
            image_size=(128, 128),
            include_depth=False,
            mount_name="pedestal",
            gripper_name="assemble_axle",
            use_ft_sensor=False,
            render_mode="rgb_array",  # 手动管理 viewer，使用 rgb_array
        )
        
        # 获取机器人配置信息用于显示
        robot_config = env.task.robot_config
        print(f"✅ 仿真环境创建成功")
        print(f"   - 环境类型: {type(env).__name__}")
        print(f"   - 机器人名称: {robot_config.name}")
        print(f"   - 机器人类型: {robot_config.robot_type}")
        print(f"   - 自由度: {robot_config.dof}")
        print(f"   - 控制器: {robot_config.controller_type}")
        print(f"   - 控制频率: {robot_config.control_freq} Hz")
        print(f"   - 底座: {robot_config.mount_name}")
        print(f"   - 夹爪: {robot_config.gripper_name}")
        print(f"   - 力传感器: {robot_config.use_ft_sensor}")
        print(f"   - 任务名称: {env.task.name}")
        print(f"   - 场景: {env.task.scene_name}")
        print(f"   - 观测配置:")
        print(f"     - 图像: {env.task.obs_config.include_image}")
        print(f"     - 深度: {env.task.obs_config.include_depth}")
        print(f"     - 本体感知: {env.task.obs_config.include_proprioception}")
        print(f"   - XML路径: {env.task.model_path}")
        print(f"   - 动作空间: {env.action_space}")
        print(f"   - 观测空间: {list(env.observation_space.spaces.keys())}")
        print(f"   - 最大步数: {env.max_episode_steps}")
        
        # === 4. 直接使用MuJoCo原生可视化 ===
        if render:
            print("\n🎮 开始MuJoCo原生可视化...")
            print("="*60)
            print("\n💡 操作提示:")
            print("   - 鼠标左键拖拽: 旋转视角")
            print("   - 鼠标右键拖拽: 平移视角") 
            print("   - 滚轮: 缩放")
            print("   - ESC: 退出查看器")
            print("   - Space: 暂停/继续")
            print("\n⏱️  启动原生MuJoCo查看器...")
            
            # 获取MuJoCo模型和数据
            model = env.model
            data = env.data
            
            print(f"✅ MuJoCo模型信息:")
            print(f"   - 自由度: {model.nv}")
            print(f"   - 物体数量: {model.nbody}") 
            print(f"   - 几何体数量: {model.ngeom}")
            print(f"   - 关节数量: {model.njnt}")
            
            # 重置环境并设置初始状态
            obs, info = env.reset()
            
            # 使用MuJoCo原生viewer
            print(f"\n🚀 启动MuJoCo Viewer...")
            with mj_viewer.launch_passive(model, data) as viewer:
                print(f"✅ MuJoCo Viewer启动成功!")
                
                if mode == 'moveL':
                    run_moveL_demo(env, viewer)
                else:
                    run_moveJ_demo(env, viewer)
                
                # 演示结束后保持窗口直到关闭
                print(f"\n🎯 演示完成，保持窗口打开...")
                while viewer.is_running():
                    viewer.sync()
                    import time
                    time.sleep(0.1)
            
            print(f"\n✅ 可视化完成")
        else:
            # 无渲染模式
            print("\n🚀 开始无渲染运行...")
            obs, info = env.reset()
            if mode == 'moveL':
                run_moveL_demo(env, None)
            else:
                run_moveJ_demo(env, None)
            print(f"\n✅ 运行完成")
        
        # 清理
        env.close()
        
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
