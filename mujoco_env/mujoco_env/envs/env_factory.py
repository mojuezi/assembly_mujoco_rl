"""
环境工厂函数

提供统一的环境创建接口，支持仿真和真机环境

作者: Liu Gang
日期: 2025-12-24
"""

from typing import Optional, Dict, Any, Union
from pathlib import Path
import numpy as np
import pdb

from mujoco_env.mujoco_env.core.sim_env import SimulationRobotEnv
from mujoco_env.mujoco_env.core.real_env import RealRobotEnv
from mujoco_env.mujoco_env.tasks.base_task import BaseTask
from mujoco_env.mujoco_env.robot_config.base import RobotConfig
from mujoco_env.mujoco_env.real import get_robot_interface


# ============================================================================
# 辅助函数
# ============================================================================

def _get_scene_xml_path(
    robot_name: str,
    scene_name: Optional[str] = None
) -> str:
    """
    获取场景 XML 文件路径
    
    Args:
        robot_name: 机器人名称 (franka_panda, aubo_i5, ur5e, etc.)
        scene_name: 场景名称，如果为 None，使用默认场景
    
    Returns:
        xml_path: XML 文件的绝对路径
    
    Raises:
        FileNotFoundError: 如果 XML 文件不存在
    """
    # 获取 assets/scenes 目录
    current_dir = Path(__file__).parent.parent
    scenes_dir = current_dir / "assets" / "scenes"
    
    if scene_name is None:
        # 使用默认场景
        scene_name = "default.xml"
    elif not scene_name.endswith(".xml"):
        scene_name = f"{scene_name}.xml"
    
    xml_path = scenes_dir / scene_name
    
    if not xml_path.exists():
        raise FileNotFoundError(
            f"Scene XML file not found: {xml_path}\n"
            f"Available scenes: {list(scenes_dir.glob('*.xml'))}"
        )
    
    return str(xml_path)


# 移除不需要的辅助函数，推荐用户直接创建配置和任务对象


# ============================================================================
# 主要工厂函数
# ============================================================================

def make_sim_env(
    # 简化的3层参数架构：
    # 1. 任务配置参数（包含robot_config、观测空间和场景）
    task: 'BaseTask',

    # 2. 强化学习环境参数
    max_episode_steps: int = 500,
    render_mode: str = None,
    control_dt: Optional[float] = None,
    ee_site_name: Optional[str] = None,

    # 3. 其他参数
    **controller_kwargs
) -> SimulationRobotEnv:
    """
    创建仿真环境 - 简化的3层参数架构
    
    ## 1. 任务配置参数
        task: BaseTask对象，包含：
            - robot_config: 机器人配置（包含底座、夹爪、力传感器等）
            - 任务逻辑和奖励函数  
            - 观测空间配置（图像、深度等）
            - 场景配置
            - 任务相关参数
            - model_path: 自动构建的完整XML文件路径
    
    ## 2. 强化学习环境参数
        max_episode_steps: 最大episode步数
        render_mode: 渲染模式
        control_dt: 控制周期（秒），如果为None则使用 1.0/robot_config.control_freq
        **controller_kwargs: 控制器参数
    
    Returns:
        SimulationRobotEnv实例
    
    Examples:
        >>> from mujoco_env.mujoco_env.tasks.pick_cube.aubo_i5_config import AuboI5Robot
        >>> from mujoco_env.mujoco_env.tasks.pick_cube.pick_cube import PickCubeTask
        >>> 
        >>> # 1. 创建机器人配置
        >>> robot_config = AuboI5Robot.get_config(
        ...     controller_type="joint_position",
        ...     mount_name="cylinder",
        ...     gripper_name="robotiq_2f85", 
        ...     use_ft_sensor=False
        ... )
        >>> 
        >>> # 2. 创建任务（自动构建完整XML：机器人+底座+夹爪+传感器+场景）
        >>> task = PickCubeTask(
        ...     robot_config=robot_config,
        ...     scene_name="grasping",
        ...     include_image=True,
        ...     image_size=(84, 84)
        ... )
        >>> 
        >>> # 3. 创建环境（使用task中构建的完整XML）
        >>> env = make_sim_env(
        ...     task=task,
        ...     max_episode_steps=1000,
        ...     render_mode="rgb_array"
        ... )
    """
    # 从task中获取robot_config和完整的XML路径
    robot_config = task.robot_config
    
    # 确定 control_dt
    if control_dt is None:
        control_dt = 1.0 / robot_config.control_freq
    
    # 创建仿真环境（使用task构建的完整XML文件）
    env = SimulationRobotEnv(
        xml_path=str(task.model_path),  # 完整的环境XML（机器人+底座+夹爪+传感器+场景）
        task=task,
        control_dt=control_dt,
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
        ee_site_name=ee_site_name,
        **controller_kwargs
    )
    
    return env


def make_real_env(
    task: 'BaseTask',
    robot_ip: str,
    max_episode_steps: int = 500,
    **interface_kwargs
) -> RealRobotEnv:
    """
    创建真机环境 - 简化架构
    
    Args:
        task: BaseTask对象，包含机器人配置和任务逻辑
        robot_ip: 机器人 IP 地址
        max_episode_steps: 最大 episode 步数
        **interface_kwargs: 传递给机器人接口的额外参数
    
    Returns:
        env: RealRobotEnv 实例
    
    Examples:
        >>> robot_config = AuboI5Robot.get_config()
        >>> task = PickCubeTask(robot_config=robot_config)
        >>> env = make_real_env(
        ...     task=task,
        ...     robot_ip="192.168.1.1"
        ... )
    """
    # 从task中获取robot_config
    robot_config = task.robot_config
    
    # 创建机器人接口
    robot_interface = get_robot_interface(
        robot_type=robot_config.name,
        robot_ip=robot_ip,
        **interface_kwargs
    )
    
    # 创建真机环境
    env = RealRobotEnv(
        robot_interface=robot_interface,
        task=task,
        max_episode_steps=max_episode_steps
    )
    
    return env


def make_env(
    task: 'BaseTask',
    mode: str = "sim",
    robot_ip: Optional[str] = None,
    **kwargs
) -> Union[SimulationRobotEnv, RealRobotEnv]:
    """
    通用环境创建接口
    
    Args:
        task: BaseTask对象，包含机器人配置和任务逻辑
        mode: 环境模式 ("sim"/"simulation" 或 "real"/"robot")
        robot_ip: 机器人IP地址（真机模式必需）
        **kwargs: 传递给具体环境创建函数的参数
    
    Returns:
        env: SimulationRobotEnv 或 RealRobotEnv 实例
    
    Examples:
        >>> robot_config = AuboI5Robot.get_config()
        >>> task = PickCubeTask(robot_config=robot_config)
        >>> env = make_env(task=task, mode="sim")
    """
    mode_lower = mode.lower()
    
    if mode_lower in ["sim", "simulation"]:
        return make_sim_env(task=task, **kwargs)
    
    elif mode_lower in ["real", "robot"]:
        if robot_ip is None:
            raise ValueError(
                "robot_ip is required for real robot mode. "
                "Usage: make_env(task=task, mode='real', robot_ip='192.168.1.1')"
            )
        
        return make_real_env(
            task=task,
            robot_ip=robot_ip,
            **kwargs
        )
    
    else:
        raise ValueError(
            f"Unknown mode: '{mode}'. "
            f"Supported modes: 'sim', 'simulation', 'real', 'robot'"
        )

