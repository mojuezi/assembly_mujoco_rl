"""
Franka FR3 机器人定义

Franka FR3是Franka Emika公司的第三代协作机器人，相比Panda有更高的性能。

作者: Liu Gang
日期: 2025-12-21
"""

from pathlib import Path
from typing import Dict, Optional
import numpy as np
from .base import RobotConfig


class FrankaFR3Robot(RobotConfig):
    """
    Franka FR3机器人配置和工具类
    
    特点:
    - 7自由度协作机器人
    - 工作空间: 855mm半径
    - 负载: 3kg
    - 重复定位精度: ±0.05mm
    - 支持力矩控制和阻抗控制
    """
    
    # 机器人参数
    NAME = "franka_fr3"
    DOF = 7
    
    # 关节限位 (弧度) - 与Panda类似但略有不同
    JOINT_LIMITS = np.array([
        [-2.7437, 2.7437],   # joint1
        [-1.7837, 1.7837],   # joint2
        [-2.9007, 2.9007],   # joint3
        [-3.0421, -0.1518],  # joint4
        [-2.8065, 2.8065],   # joint5
        [0.5445, 4.5169],    # joint6
        [-3.0159, 3.0159],   # joint7
    ])
    
    # 速度限位 (rad/s)
    VELOCITY_LIMITS = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])
    
    # 力矩限位 (Nm)
    TORQUE_LIMITS = np.array([87, 87, 87, 87, 12, 12, 12])
    
    # Home位置
    HOME_QPOS = np.array([0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4])
    
    # 工作空间边界（基座坐标系）
    WORKSPACE_MIN = np.array([0.2, -0.5, 0.1])
    WORKSPACE_MAX = np.array([0.8, 0.5, 0.8])
    
    # 关节名称（对应 MuJoCo XML 中的 joint name）
    JOINT_NAMES = [
        "fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
        "fr3_joint5", "fr3_joint6", "fr3_joint7"
    ]

    # 支持的底座类型
    AVAILABLE_MOUNTS = ["pedestal", "table", "cylinder", "floor_left", "top_point2", "default"]
    
    # 支持的夹爪类型
    AVAILABLE_GRIPPERS = ["panda_hand", "robotiq_2f85", None]

    # 底座位置 (待根据实际XML调整，这里使用通用默认值)
    MOUNT_POS = {
        "pedestal": [0.4, -0.4, 0.092],
        "default": [0, 0, 0],
    }
    
    # 机器人安装偏移 (相对于底座)
    ROBOT_OFFSET = {
        "pedestal": [0, 0, 0.392],
        "default": [0, 0, 0],
    }

    def __init__(
        self,
        controller_type: str = "joint_position",
        control_freq: int = 20,
        mount_name: str = "default",
        gripper_name: Optional[str] = "panda_hand",
        use_ft_sensor: bool = False,
        **kwargs
    ):
        """
        初始化Franka FR3机器人配置
        
        Args:
            controller_type: 控制器类型
            control_freq: 控制频率 (Hz)
            mount_name: 底座名称
            gripper_name: 夹爪名称
            use_ft_sensor: 是否使用力传感器
            **kwargs: 额外配置参数
        """
        # 计算底座位置和机器人偏移
        mount_pos = self.MOUNT_POS.get(mount_name, [0, 0, 0])
        robot_offset = self.ROBOT_OFFSET.get(mount_name, [0, 0, 0])

        super().__init__(
            name=self.NAME,
            robot_type="franka_fr3",
            dof=self.DOF,
            controller_type=controller_type,
            control_freq=control_freq,
            mount_name=mount_name,
            mount_pos=mount_pos,
            robot_offset=robot_offset,
            gripper_name=gripper_name,
            use_ft_sensor=use_ft_sensor,
            joint_names=self.JOINT_NAMES,
            **kwargs
        )
    
    @classmethod
    def get_config(
        cls,
        controller_type: str = "joint_position",
        control_freq: int = 20,
        mount_name: str = "default",
        gripper_name: Optional[str] = "panda_hand",
        use_ft_sensor: bool = False,
        **kwargs
    ) -> "FrankaFR3Robot":
        """
        获取Franka FR3机器人配置
        
        Args:
            controller_type: 控制器类型
            control_freq: 控制频率 (Hz)
            mount_name: 底座名称
            gripper_name: 夹爪名称
            use_ft_sensor: 是否使用力传感器
            **kwargs: 额外配置参数
            
        Returns:
            FrankaFR3Robot实例
        """
        return cls(
            controller_type=controller_type,
            control_freq=control_freq,
            mount_name=mount_name,
            gripper_name=gripper_name,
            use_ft_sensor=use_ft_sensor,
            **kwargs
        )
    
    @staticmethod
    def get_xml_path(scene: str = "default") -> Optional[Path]:
        """
        获取机器人XML文件路径
        
        Args:
            scene: 场景名称
            
        Returns:
            Path对象或None
        """
        base_path = Path(__file__).parent.parent / "assets" / "models"
        
        # 尝试从manipulators目录查找
        xml_path = base_path / "manipulators" / "FrankaFR3" / "FrankaFR3.xml"
        
        if xml_path.exists():
            return xml_path
        
        # 尝试从franka_fr3目录查找
        xml_path = base_path / "manipulators" / "franka_fr3" / "franka_fr3.xml"
        if xml_path.exists():
            return xml_path
        
        return None
    
    @classmethod
    def validate_qpos(cls, qpos: np.ndarray) -> bool:
        """
        检查关节位置是否在限位范围内
        
        Args:
            qpos: 关节位置 (7,)
            
        Returns:
            True如果在限位内，False否则
        """
        if len(qpos) != cls.DOF:
            return False
            
        return np.all(
            (qpos >= cls.JOINT_LIMITS[:, 0]) &
            (qpos <= cls.JOINT_LIMITS[:, 1])
        )
    
    @classmethod
    def check_velocity_limits(cls, qvel: np.ndarray) -> bool:
        """
        检查关节速度是否在限位范围内
        
        Args:
            qvel: 关节速度 (7,)
            
        Returns:
            True如果在限位内，False否则
        """
        return np.all(np.abs(qvel) <= cls.VELOCITY_LIMITS)
    
    @classmethod
    def check_torque_limits(cls, tau: np.ndarray) -> bool:
        """
        检查关节力矩是否在限位范围内
        
        Args:
            tau: 关节力矩 (7,)
            
        Returns:
            True如果在限位内，False否则
        """
        return np.all(np.abs(tau) <= cls.TORQUE_LIMITS)
    
    @classmethod
    def clip_qpos(cls, qpos: np.ndarray) -> np.ndarray:
        """
        限制关节位置在限位内
        
        Args:
            qpos: 关节位置数组 [dof]
            
        Returns:
            限制后的关节位置
        """
        qpos_clipped = np.copy(qpos)
        for i in range(cls.DOF):
            qpos_clipped[i] = np.clip(
                qpos[i],
                cls.JOINT_LIMITS[i, 0],
                cls.JOINT_LIMITS[i, 1]
            )
        return qpos_clipped


# ============================================================================
# 导出
# ============================================================================


__all__ = [
    "FrankaFR3Robot",
]
