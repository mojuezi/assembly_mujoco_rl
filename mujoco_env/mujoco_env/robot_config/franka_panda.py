"""
Franka Panda机器人定义

作者: Liu Gang
日期: 2025-12-20
"""

from pathlib import Path
from typing import Dict, Optional
import numpy as np
from .base import RobotConfig


class FrankaPandaRobot(RobotConfig):
    """Franka Panda机器人配置和工具类"""
    
    # 机器人参数
    NAME = "franka_panda"
    DOF = 7
    
    # 关节限位 (弧度)
    JOINT_LIMITS = np.array([
        [-2.8973, 2.8973],   # joint1
        [-1.7628, 1.7628],   # joint2
        [-2.8973, 2.8973],   # joint3
        [-3.0718, -0.0698],  # joint4
        [-2.8973, 2.8973],   # joint5
        [-0.0175, 3.7525],   # joint6
        [-2.8973, 2.8973],   # joint7
    ])
    
    # 速度限位 (rad/s)
    VELOCITY_LIMITS = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])
    
    # 力矩限位 (Nm)
    TORQUE_LIMITS = np.array([87, 87, 87, 87, 12, 12, 12])
    
    # Home位置
    HOME_QPOS = np.array([1.57, -0.785, 0, -2.356, 0, 1.571, 0.785])
    
    # 关节名称（对应 MuJoCo XML 中的 joint name）
    JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
    
    # DH参数 (可选，用于运动学计算)
    DH_PARAMS = {
        'a': [0, 0, 0, 0.0825, -0.0825, 0, 0.088],
        'd': [0.333, 0, 0.316, 0, 0.384, 0, 0],
        'alpha': [0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2],
    }

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
        初始化Franka Panda机器人配置
        
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
            robot_type="Panda", # 注意：assets目录名可能是 "Panda" 或 "franka_panda"，需确认
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
    ) -> "FrankaPandaRobot":
        """
        获取机器人配置
        
        Args:
            controller_type: 控制器类型
            control_freq: 控制频率
            mount_name: 底座名称
            gripper_name: 夹爪名称
            use_ft_sensor: 是否使用力传感器
            **kwargs: 额外配置
            
        Returns:
            FrankaPandaRobot: 机器人配置对象
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
        xml_path = base_path / "franka_panda" / f"{scene}.xml"
        
        if xml_path.exists():
            return xml_path
        
        # 如果不存在，尝试从impl/assets查找
        impl_path = Path(__file__).parent.parent.parent / "impl" / "assets" / "models" / "manipulators" / "Panda"
        if impl_path.exists():
            return impl_path / "panda.xml"
        
        return None
    
    @staticmethod
    def validate_qpos(qpos: np.ndarray) -> bool:
        """
        验证关节位置是否在限位内
        
        Args:
            qpos: 关节位置数组 [dof]
            
        Returns:
            bool: 是否有效
        """
        if len(qpos) != FrankaPandaRobot.DOF:
            return False
        
        for i, q in enumerate(qpos):
            if q < FrankaPandaRobot.JOINT_LIMITS[i, 0] or q > FrankaPandaRobot.JOINT_LIMITS[i, 1]:
                return False
        
        return True
    
    @staticmethod
    def clip_qpos(qpos: np.ndarray) -> np.ndarray:
        """
        限制关节位置在限位内
        
        Args:
            qpos: 关节位置数组 [dof]
            
        Returns:
            限制后的关节位置
        """
        qpos_clipped = np.copy(qpos)
        for i in range(FrankaPandaRobot.DOF):
            qpos_clipped[i] = np.clip(
                qpos[i],
                FrankaPandaRobot.JOINT_LIMITS[i, 0],
                FrankaPandaRobot.JOINT_LIMITS[i, 1]
            )
        return qpos_clipped
    
    @staticmethod
    def get_info() -> Dict[str, any]:
        """获取机器人信息"""
        return {
            "name": FrankaPandaRobot.NAME,
            "dof": FrankaPandaRobot.DOF,
            "manufacturer": "Franka Emika",
            "payload": 3.0,  # kg
            "reach": 0.855,  # meters
            "joint_limits": FrankaPandaRobot.JOINT_LIMITS.tolist(),
            "velocity_limits": FrankaPandaRobot.VELOCITY_LIMITS.tolist(),
            "torque_limits": FrankaPandaRobot.TORQUE_LIMITS.tolist(),
        }
