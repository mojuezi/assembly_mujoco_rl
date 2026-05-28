"""
Robot definitions module

机器人定义模块，包含所有支持的机器人配置。

支持的机器人：
- Franka Panda (7-DOF协作机器人)
- Franka FR3 (第三代Franka机器人)  
- Aubo i5 (6-DOF工业机器人)
- UR5e (Universal Robots协作机器人)
- DianaMed (Comau 7-DOF冗余机器人)

作者: Liu Gang
日期: 2025-12-20
"""

from typing import Type, Dict, List

# 导入所有机器人类
from .franka_panda import FrankaPandaRobot
from .franka_fr3 import FrankaFR3Robot
from .aubo_i5 import AuboI5Robot
from .ur5e import (
    UR5eRobot,
    UR5eConveyorRobot,
    UR5eGraspRobot
)
from .diana_med import (
    DianaMedRobot,
    DianaArucoRobot,
    DianaCollideRobot,
    DianaCalibRobot,
    DianaPickAndPlaceRobot,
    DianaReachRobot,
    DianaTeleopRobot
)


# 机器人注册表
ROBOT_REGISTRY: Dict[str, Type] = {
    # Franka系列
    "franka_panda": FrankaPandaRobot,
    "panda": FrankaPandaRobot,  # 别名
    "franka_fr3": FrankaFR3Robot,
    "fr3": FrankaFR3Robot,  # 别名
    
    # Aubo系列
    "aubo_i5": AuboI5Robot,
    "aubo": AuboI5Robot,  # 别名
    
    # UR系列
    "ur5e": UR5eRobot,
    "ur5e_conveyor": UR5eConveyorRobot,
    "ur5e_grasp": UR5eGraspRobot,
    
    # DianaMed系列
    "diana_med": DianaMedRobot,
    "diana": DianaMedRobot,  # 别名
    "diana_aruco": DianaArucoRobot,
    "diana_collide": DianaCollideRobot,
    "diana_calib": DianaCalibRobot,
    "diana_pick": DianaPickAndPlaceRobot,
    "diana_reach": DianaReachRobot,
    "diana_teleop": DianaTeleopRobot,
}


def get_robot(robot_name: str) -> Type:
    """
    根据名称获取机器人类
    
    Args:
        robot_name: 机器人名称
        
    Returns:
        机器人类
        
    Raises:
        ValueError: 如果机器人不存在
        
    Examples:
        >>> robot_class = get_robot("franka_panda")
        >>> robot_class = get_robot("ur5e")
        >>> robot_class = get_robot("diana_med")
    """
    if robot_name not in ROBOT_REGISTRY:
        available = list(ROBOT_REGISTRY.keys())
        raise ValueError(
            f"未知的机器人名称: '{robot_name}'.\n"
            f"可用的机器人: {available}"
        )
    
    return ROBOT_REGISTRY[robot_name]


def list_robots() -> List[str]:
    """
    列出所有注册的机器人名称
    
    Returns:
        机器人名称列表，按字母顺序排序
    """
    return sorted(ROBOT_REGISTRY.keys())


__all__ = [
    # 机器人类 - Franka系列
    "FrankaPandaRobot",
    "FrankaFR3Robot",
    
    # 机器人类 - Aubo系列
    "AuboI5Robot",
    
    # 机器人类 - UR系列
    "UR5eRobot",
    "UR5eConveyorRobot",
    "UR5eGraspRobot",
    
    # 机器人类 - DianaMed系列
    "DianaMedRobot",
    "DianaArucoRobot",
    "DianaCollideRobot",
    "DianaCalibRobot",
    "DianaPickAndPlaceRobot",
    "DianaReachRobot",
    "DianaTeleopRobot",
    
    # 工具函数和注册表
    "ROBOT_REGISTRY",
    "get_robot",
    "list_robots",
]
