"""
DianaMed机器人定义

DianaMed是Comau公司的7自由度协作机器人，
具有高灵活性和精确度。

作者: Liu Gang  
日期: 2025-12-20
"""

from pathlib import Path
from typing import Optional, Dict
import numpy as np


class DianaMedRobot:
    """
    DianaMed机器人类
    
    特点：
    - 7自由度冗余机械臂
    - 工作半径1630mm
    - 负载6kg
    - 高灵活性，适合复杂任务
    """
    
    # 机器人基本信息
    NAME = "diana_med"
    DOF = 7
    
    # 关节限位 (rad)
    JOINT_LIMITS = np.array([
        [-2.97, 2.97],    # j1
        [-2.09, 2.09],    # j2
        [-2.97, 2.97],    # j3
        [-2.09, 2.09],    # j4
        [-2.97, 2.97],    # j5
        [-2.09, 2.09],    # j6
        [-2.97, 2.97],    # j7
    ])
    
    # 速度限位 (rad/s)
    VELOCITY_LIMITS = np.array([2.09, 2.09, 2.09, 2.09, 2.79, 2.79, 2.79])
    
    # 力矩限位 (Nm)  
    TORQUE_LIMITS = np.array([100, 100, 50, 50, 20, 20, 20])
    
    # Home位置 (rad)
    HOME_QPOS = np.array([0.0, -np.pi/4, 0.0, np.pi/2, 0.0, np.pi/4, 0.0])
    
    # 工作空间限制 (m)
    WORKSPACE_MIN = np.array([0.3, -0.4, 0.0])
    WORKSPACE_MAX = np.array([1.0, 0.4, 0.6])
    
    @classmethod
    def get_config(cls, **kwargs) -> Dict:
        """
        获取机器人配置
        
        Returns:
            配置字典
        """
        from .base import RobotConfig
        
        return RobotConfig(
            name=cls.NAME,
            dof=cls.DOF,
            joint_names=[f'j{i+1}' for i in range(7)],
            actuator_names=[f'a{i+1}' for i in range(7)],
            home_qpos=cls.HOME_QPOS,
            **kwargs
        )
    
    @classmethod
    def get_xml_path(cls) -> Optional[Path]:
        """
        获取机器人XML文件路径
        
        Returns:
            XML文件路径，如果不存在返回None
        """
        xml_path = Path(__file__).parent.parent / "assets" / "models" / "manipulators" / "DianaMed" / "DianaMed.xml"
        return xml_path if xml_path.exists() else None


class DianaArucoRobot(DianaMedRobot):
    """
    DianaMed + RealSense + Aruco标记场景
    
    适用于：
    - 视觉伺服
    - Aruco标记跟踪
    - 相机标定
    """
    
    @classmethod
    def get_xml_path(cls) -> Optional[Path]:
        """Aruco场景的XML"""
        xml_path = Path(__file__).parent.parent / "assets" / "scenes" / "DianaAruco" / "diana_aruco.xml"
        return xml_path if xml_path.exists() else None


class DianaCollideRobot(DianaMedRobot):
    """
    DianaMed + 障碍物场景
    
    适用于：
    - 避障规划
    - 碰撞检测测试
    - 路径规划
    """
    
    @classmethod
    def get_xml_path(cls) -> Optional[Path]:
        """障碍物场景的XML"""
        xml_path = Path(__file__).parent.parent / "assets" / "scenes" / "DianaCollide" / "diana_collide.xml"
        return xml_path if xml_path.exists() else None


class DianaCalibRobot(DianaMedRobot):
    """
    DianaMed + 标定板场景
    
    适用于：
    - 相机标定
    - 手眼标定
    - 视觉系统校准
    """
    
    @classmethod
    def get_xml_path(cls) -> Optional[Path]:
        """标定场景的XML"""
        xml_path = Path(__file__).parent.parent / "assets" / "scenes" / "DianaCalib" / "diana_calib.xml"
        return xml_path if xml_path.exists() else None


class DianaPickAndPlaceRobot(DianaMedRobot):
    """
    DianaMed + 抓取场景
    
    适用于：
    - Pick and Place
    - 物体操作
    - 精细装配
    """
    
    HOME_QPOS = np.array([0.0, -np.pi/6, 0.0, np.pi/2.5, 0.0, np.pi/3, 0.0])
    
    WORKSPACE_MIN = np.array([0.2, -0.3, 0.0])
    WORKSPACE_MAX = np.array([0.8, 0.3, 0.5])
    
    @classmethod
    def get_xml_path(cls) -> Optional[Path]:
        """抓取场景的XML"""
        xml_path = Path(__file__).parent.parent / "assets" / "scenes" / "DianaPickPlace" / "diana_pick.xml"
        return xml_path if xml_path.exists() else None


class DianaReachRobot(DianaMedRobot):
    """
    DianaMed + 到达目标场景
    
    适用于：
    - 运动规划训练
    - 强化学习
    - 轨迹优化
    """
    
    HOME_QPOS = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    @classmethod
    def get_xml_path(cls) -> Optional[Path]:
        """到达任务场景的XML"""
        xml_path = Path(__file__).parent.parent / "assets" / "scenes" / "DianaReach" / "diana_reach.xml"
        return xml_path if xml_path.exists() else None


class DianaTeleopRobot(DianaMedRobot):
    """
    DianaMed + 遥操作场景
    
    适用于：
    - 遥操作
    - 示教学习
    - 人机交互
    """
    
    @classmethod
    def get_xml_path(cls) -> Optional[Path]:
        """遥操作场景的XML"""
        xml_path = Path(__file__).parent.parent / "assets" / "scenes" / "DianaTeleop" / "diana_teleop.xml"
        return xml_path if xml_path.exists() else None

