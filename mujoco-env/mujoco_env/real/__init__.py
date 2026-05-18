"""
真机接口模块

提供统一的机器人接口，支持 Franka 和 Aubo 等真实机器人

作者: Liu Gang
日期: 2025-12-20
更新: 2025-12-24 - 添加 get_robot_interface 工厂函数
"""

from typing import Dict, Any
from .robot_interface import RobotInterface
from .franka_interface import FrankaInterface
from .aubo_interface import AuboInterface


# 机器人接口注册表
_ROBOT_INTERFACE_REGISTRY: Dict[str, type] = {
    "franka": FrankaInterface,
    "franka_panda": FrankaInterface,
    "panda": FrankaInterface,
    "aubo": AuboInterface,
    "aubo_i5": AuboInterface,
}


def get_robot_interface(
    robot_type: str,
    robot_ip: str,
    **kwargs: Any
) -> RobotInterface:
    """
    工厂函数：根据机器人类型创建对应的接口实例
    
    Args:
        robot_type: 机器人类型，支持:
            - "franka", "franka_panda", "panda" -> FrankaInterface
            - "aubo", "aubo_i5" -> AuboInterface
        robot_ip: 机器人IP地址
        **kwargs: 传递给接口构造函数的额外参数
    
    Returns:
        RobotInterface: 机器人接口实例
    
    Raises:
        ValueError: 如果机器人类型不支持
    
    Examples:
        >>> # 创建 Franka 接口
        >>> franka = get_robot_interface("franka", "192.168.1.1")
        >>> 
        >>> # 创建 Aubo 接口
        >>> aubo = get_robot_interface("aubo", "192.168.1.2", server_port=8080)
        >>> 
        >>> # 使用上下文管理器
        >>> with get_robot_interface("franka", "192.168.1.1") as robot:
        ...     state = robot.get_robot_state()
        ...     print(state)
    """
    robot_type_lower = robot_type.lower()
    
    if robot_type_lower not in _ROBOT_INTERFACE_REGISTRY:
        available_types = list(_ROBOT_INTERFACE_REGISTRY.keys())
        raise ValueError(
            f"Unknown robot type: '{robot_type}'. "
            f"Available types: {available_types}"
        )
    
    interface_class = _ROBOT_INTERFACE_REGISTRY[robot_type_lower]
    return interface_class(robot_ip=robot_ip, **kwargs)


def register_robot_interface(name: str, interface_class: type):
    """
    注册自定义机器人接口
    
    Args:
        name: 机器人类型名称
        interface_class: 接口类（必须继承自 RobotInterface）
    
    Examples:
        >>> class MyRobotInterface(RobotInterface):
        ...     # 实现接口
        ...     pass
        >>> 
        >>> register_robot_interface("my_robot", MyRobotInterface)
        >>> robot = get_robot_interface("my_robot", "192.168.1.1")
    """
    if not issubclass(interface_class, RobotInterface):
        raise TypeError(
            f"{interface_class.__name__} must inherit from RobotInterface"
        )
    
    _ROBOT_INTERFACE_REGISTRY[name.lower()] = interface_class


def list_available_robots() -> list:
    """
    列出所有可用的机器人类型
    
    Returns:
        list: 可用的机器人类型名称列表
    """
    return list(_ROBOT_INTERFACE_REGISTRY.keys())


__all__ = [
    "RobotInterface",
    "FrankaInterface",
    "AuboInterface",
    "get_robot_interface",
    "register_robot_interface",
    "list_available_robots",
]

