#!/usr/bin/env python
"""
导纳控制演示

演示Aubo i5机械臂的导纳控制功能，包括：
- 力控制模式
- 实时力反馈
- 动态力/力矩可视化（可选）

导纳控制（Admittance Control）是一种基于力反馈的控制方法，
允许机器人根据外部力进行柔顺运动。

作者: Liu Gang
日期: 2025-12-20
修订: 添加详细注释和文档
"""

import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import multiprocessing as mp
import time
import sys
import argparse
from pathlib import Path

# 添加路径以便导入mujoco_env模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from mujoco_env.mujoco_env.core import SimulationRobotEnv, RobotConfig, ObservationConfig
from mujoco_env.mujoco_env.controllers import get_controller

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# ============================================================================
# 力数据可视化（用于调试）
# ============================================================================

# 初始化力数据缓存
force_x = []
force_y = []
force_z = []
time_data = []
index = 0


def generate_data(data_queue, force, lock):
    """
    生成力数据供绘图使用
    
    Args:
        data_queue: 多进程队列，用于传递数据
        force: 力向量 [Fx, Fy, Fz]
        lock: 线程锁
    """
    global index
    time_data.append(index)
    force_x.append(force[0])
    force_y.append(force[1])
    force_z.append(force[2])
    index += 1
    
    data_queue.put((time_data.copy(), force_x.copy(), force_y.copy(), force_z.copy()))
    
    # 保持数据窗口大小为100
    if index > 100:
        force_x.pop(0)
        force_y.pop(0)
        force_z.pop(0)
        time_data.pop(0)


def update_plot(data_queue, lock):
    """
    更新力数据的实时绘图
    
    Args:
        data_queue: 多进程队列
        lock: 线程锁
    """
    fig, ax = plt.subplots()
    
    def animate(i):
        """动画更新函数"""
        with lock:
            if not data_queue.empty():
                time, fx, fy, fz = data_queue.get()
                ax.clear()
                ax.plot(time, fx, label='Force X', color='r')
                ax.plot(time, fy, label='Force Y', color='g')
                ax.plot(time, fz, label='Force Z', color='b')
                ax.legend(loc='upper left')
                ax.set_xlabel('Time Steps')
                ax.set_ylabel('Force (N)')
                ax.set_title('Real-time Force Feedback')
    
    ani = animation.FuncAnimation(fig, animate, interval=100)
    plt.tight_layout()
    plt.show()


# ============================================================================
# 主程序
# ============================================================================

def main(render=True, debug_plot=False):
    """
    主函数：演示导纳控制
    
    Args:
        render: 是否显示MuJoCo渲染窗口（默认True）
        debug_plot: 是否启用力数据可视化（默认False，会降低性能）
    """
    
    # 如果启用可视化，创建独立进程进行绘图
    if debug_plot:
        logging.info("启用力反馈可视化")
        data_queue = mp.Queue()
        lock = mp.Lock()
        
        # 启动绘图进程
        data_process = mp.Process(
            name="data_plot",
            target=update_plot,
            args=(data_queue, lock)
        )
        data_process.daemon = False
        data_process.start()
    
    # ========================================================================
    # 步骤1: 初始化环境
    # ========================================================================
    
    logging.info("初始化Aubo i5机械臂环境")
    
    # 根据参数设置渲染模式
    render_mode = 'human' if render else None
    if render:
        logging.info("MuJoCo渲染窗口已启用")
    else:
        logging.info("MuJoCo渲染窗口已禁用")
    
    # 获取场景XML路径
    assets_path = Path(__file__).parent.parent / "mujoco_env" / "assets"
    scene_xml = assets_path / "scenes" / "assemble.xml"
    
    # 创建机器人配置
    robot_config = RobotConfig(
        name="aubo_i5",
        robot_type="aubo",
        dof=6,
        control_freq=200,
        controller_type="admittance",
        has_gripper=True
    )
    
    # 创建观测配置
    obs_config = ObservationConfig(
        include_proprioception=True,
        include_image=False
    )
    
    # 创建环境
    env = SimulationRobotEnv(
        xml_path=str(scene_xml),
        robot_config=robot_config,
        obs_config=obs_config,
        control_dt=1.0/200.0,  # 200 Hz
        physics_dt=0.002,
        max_episode_steps=10000,
        render_mode=render_mode
    )
    
    env.reset()
    
    # 创建导纳控制器
    controller = get_controller(
        'admittance',
        model=env.model,
        data=env.data,
        dof=6,
        control_freq=200.0,
        ee_site_name="0_grip_site"
    )
    
    # ========================================================================
    # 步骤2: 配置导纳控制器参数
    # ========================================================================
    
    logging.info("配置导纳控制参数")
    
    # 配置导纳控制器参数
    # 力控开启方向（基于基坐标系）
    # [1, 1, 1, 1, 1, 1] = [Fx, Fy, Fz, Mx, My, Mz]
    # 1 表示该方向启用力控，0 表示位置控制
    controller.set_selection_vector(np.array([1, 1, 1, 1, 1, 1]))
    
    # 目标力/力矩
    controller.set_desired_force(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    
    # ========================================================================
    # 步骤3: 设置初始位姿
    # ========================================================================
    
    logging.info("移动到初始位置")
    
    # 定义初始关节角度（单位：弧度）
    joint = np.array([
        0.0,                           # J1: 0°
        -15.0 / 180.0 * np.pi,        # J2: -15°
        100.0 / 180.0 * np.pi,        # J3: 100°
        25.0 / 180.0 * np.pi,         # J4: 25°
        90.0 / 180.0 * np.pi,         # J5: 90°
        0.0 / 180.0 * np.pi           # J6: 0°
    ])
    
    # 正向运动学：关节空间 -> 笛卡尔空间
    position, quaternion = controller.forward_kinematics(joint)
    pose = np.concatenate([position, quaternion], axis=None)
    
    # 先使用位置控制移动到初始位置
    controller.set_selection_vector(np.array([0, 0, 0, 0, 0, 0]))  # 全位置控制
    logging.info("位置控制阶段（100步）")
    for i in range(100):
        action = controller.compute_control(pose)
        env.step(action)
        if i % 20 == 0:
            logging.info(f"  步数: {i}/100")
    
    # ========================================================================
    # 步骤4: 切换到导纳控制并保持
    # ========================================================================
    
    logging.info("切换到导纳控制模式")
    controller.set_selection_vector(np.array([1, 1, 1, 1, 1, 1]))  # 全力控
    
    logging.info("开始导纳控制（按Ctrl+C停止）")
    try:
        step_count = 0
        max_steps = 8000000  # 大数值，实际由用户Ctrl+C终止
        
        for i in range(max_steps):
            # 执行控制步
            action = controller.compute_control(pose)
            env.step(action)
            
            # 定期打印状态信息
            if i % 1000 == 0:
                    logging.info(f"步数: {i}")
            
            step_count = i
    
    except KeyboardInterrupt:
        logging.info(f"\n用户中断，执行了 {step_count} 步")
    
    # ========================================================================
    # 步骤5: 清理资源
    # ========================================================================
    
    logging.info("关闭环境")
    env.close()
    
    if debug_plot:
        logging.info("等待绘图进程结束")
        data_process.terminate()
        data_process.join(timeout=2)
    
    logging.info("演示完成！")


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='Aubo i5导纳控制演示',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 显示MuJoCo渲染窗口（默认）
  python demo_admittance_control.py
  
  # 不显示MuJoCo渲染窗口
  python demo_admittance_control.py --no-render
  
  # 启用力数据可视化（会降低性能）
  python demo_admittance_control.py --debug-plot
  
  # 显示帮助信息
  python demo_admittance_control.py -h
        """
    )
    parser.add_argument(
        '--no-render',
        action='store_true',
        help='禁用MuJoCo渲染窗口（默认显示）'
    )
    parser.add_argument(
        '--debug-plot',
        action='store_true',
        help='启用力数据实时可视化（会降低性能）'
    )
    
    args = parser.parse_args()
    
    try:
        # 默认显示渲染，除非指定 --no-render
        main(render=not args.no_render, debug_plot=args.debug_plot)
    except Exception as e:
        logging.error(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

