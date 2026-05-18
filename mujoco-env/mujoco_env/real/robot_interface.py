"""
机器人接口基类

定义统一的机器人接口，用于仿真和真机环境
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import numpy as np
import logging


class RobotInterface(ABC):
    """
    机器人接口基类
    
    定义了所有机器人必须实现的标准接口，确保仿真和真机环境使用一致的API。
    
    坐标系约定:
    - 位姿格式: [x, y, z, qw, qx, qy, qz] (位置 + 四元数)
    - 速度格式: [vx, vy, vz, wx, wy, wz] (线速度 + 角速度)
    - 力/力矩格式: [fx, fy, fz, tx, ty, tz]
    
    Examples:
        >>> robot = FrankaInterface(robot_ip="192.168.1.1")
        >>> if robot.connect():
        ...     state = robot.get_robot_state()
        ...     print(f"TCP位姿: {state['tcp_pose']}")
        ...     robot.disconnect()
    """
    
    def __init__(self, robot_ip: str, **kwargs):
        """
        初始化机器人接口
        
        Args:
            robot_ip: 机器人IP地址
            **kwargs: 其他配置参数
        """
        self.robot_ip = robot_ip
        self.logger = logging.getLogger(self.__class__.__name__)
        self.is_connected = False
    
    @abstractmethod
    def connect(self) -> bool:
        """
        连接到机器人
        
        Returns:
            bool: 连接成功返回True，否则返回False
        """
        pass
    
    @abstractmethod
    def disconnect(self):
        """断开机器人连接"""
        pass
    
    def power_on(self) -> bool:
        """
        机器人上电
        
        Returns:
            bool: 上电成功返回True，否则返回False
        """
        self.logger.warning(f"{self.__class__.__name__} does not support power on")
        return False

    def power_off(self) -> bool:
        """
        机器人下电
        
        Returns:
            bool: 下电成功返回True，否则返回False
        """
        self.logger.warning(f"{self.__class__.__name__} does not support power off")
        return False
    
    def get_joint_positions(self) -> np.ndarray:
        """
        获取当前关节位置
        
        Returns:
            np.ndarray: 关节位置 (弧度), shape=(dof,)
        """
        pass
    
    @abstractmethod
    def get_joint_velocities(self) -> np.ndarray:
        """
        获取当前关节速度
        
        Returns:
            np.ndarray: 关节速度 (rad/s), shape=(dof,)
        """
        pass
    
    @abstractmethod
    def get_tcp_pose(self) -> np.ndarray:
        """
        获取TCP (Tool Center Point) 位姿
        
        Returns:
            np.ndarray: TCP位姿 [x, y, z, qw, qx, qy, qz], shape=(7,)
                - x, y, z: 位置 (米)
                - qw, qx, qy, qz: 姿态四元数
        """
        pass
    
    @abstractmethod
    def get_tcp_velocity(self) -> np.ndarray:
        """
        获取TCP速度
        
        Returns:
            np.ndarray: TCP速度 [vx, vy, vz, wx, wy, wz], shape=(6,)
                - vx, vy, vz: 线速度 (m/s)
                - wx, wy, wz: 角速度 (rad/s)
        """
        pass
    
    @abstractmethod
    def get_tcp_force(self) -> np.ndarray:
        """
        获取TCP力传感器读数
        
        Returns:
            np.ndarray: TCP力 [fx, fy, fz], shape=(3,)
        """
        pass
    
    @abstractmethod
    def get_tcp_torque(self) -> np.ndarray:
        """
        获取TCP力矩传感器读数
        
        Returns:
            np.ndarray: TCP力矩 [tx, ty, tz], shape=(3,)
        """
        pass
    
    def enable_servo_mode(self, enable: bool = True)-> bool:
        """
        使能或关闭伺服运动模式

        Args:
            enable: True 使能伺服模式，False 关闭伺服模式
        Returns:
            bool: 使能或关闭伺服运动模式成功返回True，否则返回False
        """
        self.logger.warning(f"{self.__class__.__name__} does not support servo mode")
        return False

    def servo_to_joint_positions(
        self,
        positions: np.ndarray
    ) -> int:
        """
        伺服到目标关节位置
        
        Args:
            positions: 目标关节位置 (弧度), shape=(dof,)
        Returns:
            int: 返回码
        """
        self.logger.warning(
            f"{self.__class__.__name__} does not support servo joint positions"
        )
        return -1

    @abstractmethod
    def move_to_joint_positions(
        self,
        positions: np.ndarray,
        velocity: float = 0.5,
        acceleration: float = 0.5,
        blocking: bool = False
    ):
        """
        移动到目标关节位置
        
        Args:
            positions: 目标关节位置 (弧度), shape=(dof,)
            velocity: 速度因子 [0, 1]
            acceleration: 加速度因子 [0, 1]
            blocking: 是否阻塞等待完成
        """
        pass
    
    @abstractmethod
    def move_tcp_pose(
        self,
        pose: np.ndarray,
        velocity: float = 0.1,
        acceleration: float = 0.1,
        blocking: bool = False
    ):
        """
        移动到目标TCP位姿
        
        Args:
            pose: 目标位姿 [x, y, z, qw, qx, qy, qz], shape=(7,)
            velocity: 速度 (m/s)
            acceleration: 加速度 (m/s^2)
            blocking: 是否阻塞等待完成
        """
        pass
    
    @abstractmethod
    def get_robot_state(self) -> Dict:
        """
        获取完整机器人状态
        
        Returns:
            Dict: 机器人状态字典，包含:
                - joint_positions: 关节位置
                - joint_velocities: 关节速度
                - tcp_pose: TCP位姿
                - tcp_velocity: TCP速度
                - tcp_force: TCP力
                - tcp_torque: TCP力矩
                - errors: 错误列表
        """
        pass
    
    @abstractmethod
    def stop_motion(self):
        """立即停止机器人运动"""
        pass
    
    @abstractmethod
    def clear_errors(self):
        """清除机器人错误"""
        pass
    
    # 可选功能
    
    def set_freedrive_mode(self, enable: bool):
        """
        设置示教模式（拖动示教）
        
        Args:
            enable: True启用示教模式，False退出示教模式
        
        Note:
            不是所有机器人都支持此功能
        """
        self.logger.warning(f"{self.__class__.__name__} does not support freedrive mode")
    
    def get_joint_torques(self) -> Optional[np.ndarray]:
        """
        获取关节力矩
        
        Returns:
            np.ndarray or None: 关节力矩 (Nm), shape=(dof,)，不支持则返回None
        """
        self.logger.warning(f"{self.__class__.__name__} does not support joint torque sensing")
        return None
    
    def set_compliance_mode(self, stiffness: float, damping: float):
        """
        设置柔顺模式参数
        
        Args:
            stiffness: 刚度参数
            damping: 阻尼参数
        
        Note:
            不是所有机器人都支持此功能
        """
        self.logger.warning(f"{self.__class__.__name__} does not support compliance mode")
    
    def get_jacobian(self) -> Optional[np.ndarray]:
        """
        获取雅可比矩阵
        
        Returns:
            np.ndarray or None: 雅可比矩阵 shape=(6, dof)，不支持则返回None
        """
        self.logger.warning(f"{self.__class__.__name__} does not support jacobian computation")
        return None
    
    def __enter__(self):
        """上下文管理器入口"""
        if not self.connect():
            raise ConnectionError(f"Failed to connect to robot at {self.robot_ip}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.disconnect()
    
    def __repr__(self):
        status = "connected" if self.is_connected else "disconnected"
        return f"{self.__class__.__name__}(ip={self.robot_ip}, status={status})"

