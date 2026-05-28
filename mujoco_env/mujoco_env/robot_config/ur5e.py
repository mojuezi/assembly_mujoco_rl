"""
UR5e机器人定义

Universal Robots UR5e是一款6自由度协作机器人，
适用于中等负载的操作任务。

作者: Liu Gang
日期: 2025-12-20
"""

from pathlib import Path
from typing import Optional, Dict
import numpy as np


class UR5eRobot:
    """
    UR5e机器人类
    
    特点：
    - 6自由度机械臂
    - 工作半径850mm
    - 负载5kg
    - 适合中等精度操作任务
    """
    
    # 机器人基本信息
    NAME = "ur5e"
    DOF = 6
    
    # 关节限位 (rad)
    JOINT_LIMITS = np.array([
        [-2*np.pi, 2*np.pi],      # shoulder_pan
        [-2*np.pi, 2*np.pi],      # shoulder_lift
        [-np.pi, np.pi],          # elbow
        [-2*np.pi, 2*np.pi],      # wrist_1
        [-2*np.pi, 2*np.pi],      # wrist_2
        [-2*np.pi, 2*np.pi],      # wrist_3
    ])
    
    # 速度限位 (rad/s)
    VELOCITY_LIMITS = np.array([3.15, 3.15, 3.15, 3.2, 3.2, 3.2])
    
    # 力矩限位 (Nm)
    TORQUE_LIMITS = np.array([150, 150, 150, 28, 28, 28])
    
    # Home位置 (rad)
    HOME_QPOS = np.array([0.0, -np.pi/2, 0.0, -np.pi/2, 0.0, 0.0])
    
    # 工作空间限制 (m)
    WORKSPACE_MIN = np.array([0.3, -0.4, 0.0])
    WORKSPACE_MAX = np.array([0.8, 0.4, 0.6])
    
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
            joint_names=[
                'shoulder_pan_joint',
                'shoulder_lift_joint',
                'elbow_joint',
                'wrist_1_joint',
                'wrist_2_joint',
                'wrist_3_joint'
            ],
            actuator_names=[
                'actuator1',
                'actuator2',
                'actuator3',
                'actuator4',
                'actuator5',
                'actuator6'
            ],
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
        # 假设XML文件位置
        xml_path = Path(__file__).parent.parent / "assets" / "models" / "manipulators" / "UR5e" / "UR5e.xml"
        return xml_path if xml_path.exists() else None


class UR5eConveyorRobot(UR5eRobot):
    """
    UR5e机器人 + 传送带场景
    
    适用于：
    - 物料搬运
    - 装箱任务
    - 传送带抓取
    """
    
    HOME_QPOS = np.array([
        1.46345588e-05,
        -6.87047296e-01,
        2.10020717e+00,
        -2.98390247e+00,
        -1.57080312e+00,
        1.57079752e+00
    ])
    
    @classmethod
    def get_xml_path(cls) -> Optional[Path]:
        """传送带场景的XML"""
        xml_path = Path(__file__).parent.parent / "assets" / "scenes" / "UR5eConveyor" / "ur5e_conveyor.xml"
        return xml_path if xml_path.exists() else None


class UR5eGraspRobot(UR5eRobot):
    """
    UR5e机器人 + 抓取场景
    
    适用于：
    - Pick and Place
    - 抓取任务
    - 物体操作
    """
    
    HOME_QPOS = np.array([
        -0.27131313,
        -1.58681262,
        1.45363338,
        -1.43761664,
        -1.57079275,
        1.29926298
    ])
    
    # 抓取任务的工作空间
    WORKSPACE_MIN = np.array([0.2, -0.3, 0.0])
    WORKSPACE_MAX = np.array([0.6, 0.3, 0.5])
    
    @classmethod
    def get_xml_path(cls) -> Optional[Path]:
        """抓取场景的XML"""
        xml_path = Path(__file__).parent.parent / "assets" / "scenes" / "UR5eGrasp" / "ur5e_grasp.xml"
        return xml_path if xml_path.exists() else None

