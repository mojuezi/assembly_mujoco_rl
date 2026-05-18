"""
真机接口演示脚本

演示如何使用统一的机器人接口控制 Franka 和 Aubo 机器人
"""

import numpy as np
import time
import logging
from typing import Optional

from mujoco_env.mujoco_env.real import FrankaInterface, AuboInterface, RobotInterface


def demo_get_state(robot: RobotInterface):
    """演示获取机器人状态"""
    print("\n" + "=" * 50)
    print("演示: 获取机器人状态")
    print("=" * 50)
    
    state = robot.get_robot_state()
    
    print(f"\n关节位置 (rad):")
    print(f"  {state['joint_positions']}")
    print(f"\n关节速度 (rad/s):")
    print(f"  {state['joint_velocities']}")
    print(f"\nTCP 位置 (m):")
    print(f"  x={state['tcp_pose'][0]:.4f}, y={state['tcp_pose'][1]:.4f}, z={state['tcp_pose'][2]:.4f}")
    print(f"\nTCP 姿态 (四元数):")
    print(f"  qw={state['tcp_pose'][3]:.4f}, qx={state['tcp_pose'][4]:.4f}, "
          f"qy={state['tcp_pose'][5]:.4f}, qz={state['tcp_pose'][6]:.4f}")
    print(f"\nTCP 力 (N):")
    print(f"  fx={state['tcp_force'][0]:.2f}, fy={state['tcp_force'][1]:.2f}, fz={state['tcp_force'][2]:.2f}")
    print(f"\nTCP 力矩 (Nm):")
    print(f"  tx={state['tcp_torque'][0]:.2f}, ty={state['tcp_torque'][1]:.2f}, tz={state['tcp_torque'][2]:.2f}")


def demo_tcp_motion(robot: RobotInterface):
    """演示TCP笛卡尔空间运动"""
    print("\n" + "=" * 50)
    print("演示: TCP笛卡尔空间运动")
    print("=" * 50)
    
    # 获取当前位姿
    current_pose = robot.get_tcp_pose()
    print(f"\n当前 TCP 位姿:")
    print(f"  位置: [{current_pose[0]:.4f}, {current_pose[1]:.4f}, {current_pose[2]:.4f}]")
    
    # 确认是否执行移动
    response = input("\n是否执行小幅移动（Z轴上移5cm）? (y/n): ")
    if response.lower() != 'y':
        print("已取消移动")
        return
    
    # 目标位姿（Z轴上移5cm）
    target_pose = current_pose.copy()
    target_pose[2] += 0.05
    
    print(f"\n目标 TCP 位姿:")
    print(f"  位置: [{target_pose[0]:.4f}, {target_pose[1]:.4f}, {target_pose[2]:.4f}]")
    
    # 执行移动
    print("\n开始移动...")
    try:
        robot.move_tcp_pose(target_pose, velocity=0.05)
        print("移动命令已发送")
        time.sleep(2.0)
        
        # 检查位置
        new_pose = robot.get_tcp_pose()
        error = np.linalg.norm(new_pose[:3] - target_pose[:3])
        print(f"\n移动完成！位置误差: {error * 1000:.2f} mm")
        
        # 返回起始位姿
        response = input("\n是否返回起始位姿? (y/n): ")
        if response.lower() == 'y':
            print("\n返回起始位姿...")
            robot.move_tcp_pose(current_pose, velocity=0.05)
            time.sleep(2.0)
            print("返回完成！")
    except Exception as e:
        print(f"移动失败: {e}")


def demo_joint_motion(robot: RobotInterface):
    """演示关节空间运动（仅Aubo支持）"""
    print("\n" + "=" * 50)
    print("演示: 关节空间运动")
    print("=" * 50)
    
    if not isinstance(robot, AuboInterface):
        print("\n此功能仅 Aubo 机器人支持")
        return
    
    # 获取当前关节位置
    current_joints = robot.get_joint_positions()
    print(f"\n当前关节位置 (rad):")
    print(f"  {current_joints}")
    
    # 确认是否执行移动
    response = input("\n是否执行小幅关节运动（第6关节旋转10度）? (y/n): ")
    if response.lower() != 'y':
        print("已取消移动")
        return
    
    # 目标关节位置
    target_joints = current_joints.copy()
    target_joints[5] += np.deg2rad(10)  # 第6关节旋转10度
    
    print(f"\n目标关节位置 (rad):")
    print(f"  {target_joints}")
    
    # 执行移动
    print("\n开始移动...")
    try:
        robot.move_to_joint_positions(target_joints, velocity=0.3, blocking=True)
        print("移动完成！")
        
        # 检查位置
        new_joints = robot.get_joint_positions()
        error = np.linalg.norm(new_joints - target_joints)
        print(f"关节位置误差: {np.rad2deg(error):.2f} deg")
        
        # 返回起始位置
        response = input("\n是否返回起始位置? (y/n): ")
        if response.lower() == 'y':
            print("\n返回起始位置...")
            robot.move_to_joint_positions(current_joints, velocity=0.3, blocking=True)
            print("返回完成！")
    except Exception as e:
        print(f"移动失败: {e}")


def demo_compliance_mode(robot: RobotInterface):
    """演示柔顺模式"""
    print("\n" + "=" * 50)
    print("演示: 柔顺模式切换")
    print("=" * 50)
    
    # 确认是否执行
    response = input("\n是否切换到柔顺模式? (y/n): ")
    if response.lower() != 'y':
        print("已取消")
        return
    
    try:
        # 切换到柔顺模式
        print("\n切换到柔顺模式（低刚度）...")
        robot.set_compliance_mode(stiffness=200.0, damping=20.0)
        print("✓ 柔顺模式已启用，您可以手动推动机器人")
        print("  等待10秒...")
        time.sleep(10.0)
        
        # 切换回精确模式
        print("\n切换回精确模式（高刚度）...")
        robot.set_compliance_mode(stiffness=600.0, damping=60.0)
        print("✓ 精确模式已启用")
    except Exception as e:
        print(f"模式切换失败: {e}")


def demo_freedrive_mode(robot: RobotInterface):
    """演示示教模式（仅Aubo支持）"""
    print("\n" + "=" * 50)
    print("演示: 示教模式（拖动示教）")
    print("=" * 50)
    
    if not isinstance(robot, AuboInterface):
        print("\n此功能仅 Aubo 机器人支持")
        return
    
    # 确认是否执行
    response = input("\n是否启用示教模式? (y/n): ")
    if response.lower() != 'y':
        print("已取消")
        return
    
    try:
        # 启用示教模式
        print("\n启用示教模式...")
        robot.set_freedrive_mode(True)
        print("✓ 示教模式已启用，请拖动机器人到目标位置")
        print("  等待20秒...")
        time.sleep(20.0)
        
        # 记录目标位姿
        target_pose = robot.get_tcp_pose()
        print(f"\n记录的目标位姿:")
        print(f"  位置: [{target_pose[0]:.4f}, {target_pose[1]:.4f}, {target_pose[2]:.4f}]")
        
        # 退出示教模式
        print("\n退出示教模式...")
        robot.set_freedrive_mode(False)
        print("✓ 示教模式已退出")
    except Exception as e:
        print(f"示教模式失败: {e}")


def demo_error_handling(robot: RobotInterface):
    """演示错误处理"""
    print("\n" + "=" * 50)
    print("演示: 错误处理")
    print("=" * 50)
    
    print("\n清除机器人错误...")
    try:
        robot.clear_errors()
        print("✓ 错误已清除")
    except Exception as e:
        print(f"清除错误失败: {e}")


def main_franka():
    """Franka 机器人演示"""
    print("\n" + "=" * 60)
    print("Franka Panda 机器人接口演示")
    print("=" * 60)
    
    # 创建接口
    robot = FrankaInterface(
        robot_ip="192.168.1.1",
        server_ip="127.0.0.1",
        server_port=5000
    )
    
    # 连接
    print(f"\n连接到 Franka 服务器: {robot.base_url}")
    if not robot.connect():
        print("连接失败！请确保 Franka 服务器已启动")
        return
    
    print("✓ 连接成功")
    
    try:
        # 运行演示
        demo_get_state(robot)
        demo_tcp_motion(robot)
        demo_compliance_mode(robot)
        demo_error_handling(robot)
    finally:
        robot.disconnect()
        print("\n已断开连接")


def main_aubo():
    """Aubo 机器人演示"""
    print("\n" + "=" * 60)
    print("Aubo i5 机器人接口演示")
    print("=" * 60)
    
    # 创建接口
    robot = AuboInterface(robot_ip="192.168.1.6")
    
    # 连接
    print(f"\n连接到 Aubo 机器人: {robot.robot_ip}")
    if not robot.connect():
        print("连接失败！请确保机器人已开机且网络连接正常")
        return
    
    print("✓ 连接成功")
    
    try:
        # 运行演示
        demo_get_state(robot)
        demo_tcp_motion(robot)
        demo_joint_motion(robot)
        demo_compliance_mode(robot)
        demo_freedrive_mode(robot)
        demo_error_handling(robot)
    finally:
        robot.disconnect()
        print("\n已断开连接")


def main():
    """主函数"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "=" * 60)
    print("真机接口演示程序")
    print("=" * 60)
    print("\n请选择要测试的机器人:")
    print("  1. Franka Panda")
    print("  2. Aubo i5")
    print("  3. 退出")
    
    choice = input("\n请输入选项 (1/2/3): ")
    
    if choice == "1":
        main_franka()
    elif choice == "2":
        main_aubo()
    elif choice == "3":
        print("退出程序")
    else:
        print("无效选项")


def demo_real_robot_env():
    """
    演示如何使用 RealRobotEnv 进行强化学习
    
    这个演示展示了：
    1. 如何使用 get_robot_interface 工厂函数创建机器人接口
    2. 如何创建任务（Task）
    3. 如何创建 RealRobotEnv 环境
    4. 如何使用标准的 Gymnasium API 进行交互
    """
    print("\n" + "=" * 60)
    print("RealRobotEnv 演示（Gymnasium 环境）")
    print("=" * 60)
    
    # 导入必要的模块
    from mujoco_env.mujoco_env.core.real_env import RealRobotEnv
    from mujoco_env.mujoco_env.real import get_robot_interface
    from mujoco_env.mujoco_env.tasks.base_task import RobotConfig, ObservationConfig
    from mujoco_env.mujoco_env.tasks.pick_cube import PickCubeTask
    
    # 1. 创建机器人接口
    robot_type = input("\n请选择机器人类型 (franka/aubo): ")
    robot_ip = input("请输入机器人IP地址: ")
    
    print(f"\n创建 {robot_type} 机器人接口...")
    try:
        robot_interface = get_robot_interface(
            robot_type=robot_type,
            robot_ip=robot_ip
        )
        print("✓ 机器人接口创建成功")
    except Exception as e:
        print(f"✗ 创建失败: {e}")
        return
    
    # 2. 创建机器人配置
    robot_config = RobotConfig(
        name=robot_type,
        robot_type=robot_type,
        dof=7 if robot_type == "franka" else 6,
        control_freq=20,
        controller_type="joint_position",
        has_gripper=True
    )
    
    obs_config = ObservationConfig(
        include_image=False,
        include_depth=False,
        include_proprioception=True,
        include_goal=True
    )
    
    # 3. 创建任务
    print("\n创建 PickCubeTask...")
    task = PickCubeTask(
        robot_config=robot_config,
        obs_config=obs_config,
        max_episode_steps=100
    )
    print("✓ 任务创建成功")
    
    # 4. 创建环境
    print("\n创建 RealRobotEnv...")
    try:
        env = RealRobotEnv(
            robot_interface=robot_interface,
            task=task,
            control_freq=20.0
        )
        print("✓ 环境创建成功")
        print(f"  - Action space: {env.action_space}")
        print(f"  - Observation space keys: {list(env.observation_space.spaces.keys())}")
    except Exception as e:
        print(f"✗ 创建失败: {e}")
        return
    
    # 5. 运行环境
    try:
        print("\n开始 episode...")
        obs, info = env.reset()
        print(f"✓ Reset 成功")
        print(f"  - Observation keys: {list(obs.keys())}")
        print(f"  - TCP pose: {obs['tcp_pose']}")
        print(f"  - Achieved goal: {obs['achieved_goal']}")
        print(f"  - Desired goal: {obs['desired_goal']}")
        
        # 运行几步
        num_steps = int(input("\n请输入要运行的步数 (推荐 5-10): "))
        
        for step in range(num_steps):
            # 随机动作（实际应该使用策略）
            action = env.action_space.sample() * 0.1  # 缩小幅度，更安全
            
            print(f"\nStep {step + 1}/{num_steps}")
            print(f"  Action: {action[:3]}...")  # 只显示前3个值
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"  Reward: {reward:.4f}")
            print(f"  Success: {info['is_success']}")
            print(f"  TCP pose: {obs['tcp_pose'][:3]}")
            
            if terminated:
                print("  ✓ 任务完成！")
                break
            
            if truncated:
                print("  ⚠ Episode 超时")
                break
            
            time.sleep(0.1)  # 小延迟，方便观察
        
        print("\n✓ Episode 完成")
        
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n✗ 运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 6. 关闭环境
        print("\n关闭环境...")
        env.close()
        print("✓ 环境已关闭")


if __name__ == "__main__":
    # main()  # 原来的演示
    
    # 新增：选择演示类型
    print("\n" + "=" * 60)
    print("真机演示程序")
    print("=" * 60)
    print("\n请选择演示类型:")
    print("  1. 机器人接口演示 (RobotInterface)")
    print("  2. 强化学习环境演示 (RealRobotEnv)")
    print("  3. 退出")
    
    choice = input("\n请输入选项 (1/2/3): ")
    
    if choice == "1":
        main()
    elif choice == "2":
        demo_real_robot_env()
    elif choice == "3":
        print("退出程序")
    else:
        print("无效选项")


