"""
Aubo i5机器人定义

作者: Liu Gang
日期: 2025-12-20
"""

from pathlib import Path
from typing import Dict, Optional
import numpy as np
from .base import RobotConfig


class AuboI5Robot(RobotConfig):
    """Aubo i5机器人基类，包含所有通用属性和方法"""
    
    # 机器人参数
    NAME = "aubo_i5"
    DOF = 6
    
    # 支持的底座类型
    AVAILABLE_MOUNTS = ["pedestal", "table", "cylinder", "floor_left", "top_point2"]
    
    # 支持的夹爪类型
    AVAILABLE_GRIPPERS = ["robotiq_2f85", "panda_hand", "robotiq_gripper"]
    
    # 底座位置
    MOUNT_POS = {
        "pedestal": [0.6, -0.4, 0.09],  # 参考 assemble.xml [0.4, -0.4, 0.092]
        "pedestal_new": [0.00052423, -0.0008827, 0.04912453],  # 0.32974+0.06225997-0.34287544
        "cylinder": [0, 0, 0],      # 默认位置
        "table": [0, 0, 0],         # 默认位置
        "floor_left": [0, 0, 0],    # 默认位置
        "top_point2": [0, 0, 0],    # 默认位置
    }
    
    # 机器人安装偏移 (相对于底座)
    ROBOT_OFFSET = {
        "pedestal": [0, 0, 0.392],  # 假设pedestal STL原点在顶部，不需要Z轴偏移
        "cylinder": [0, 0, 0.14],   # cylinder高度约0.14m
        "table": [0, 0, 0],         # table表面
        "floor_left": [0, 0, 0],    # 地面
        "top_point2": [0, 0, 0],    # 默认偏移
    }
    
    # 关节限位 (弧度)
    JOINT_LIMITS = np.array([
        [-3.05, 3.05],       # joint1
        [-3.05, 3.05],       # joint2
        [-3.05, 3.05],       # joint3
        [-3.05, 3.05],       # joint4
        [-3.05, 3.05],       # joint5
        [-6.28, 6.28],       # joint6
    ])
    
    # 速度限位 (rad/s)
    VELOCITY_LIMITS = np.array([3.14, 3.14, 3.14, 3.14, 3.14, 3.14])
    
    # 力矩限位 (Nm)
    TORQUE_LIMITS = np.array([150, 150, 150, 28, 28, 28])
    
    # Home位置
    HOME_QPOS = np.array([0, -0.262, 1.745, 0.436, 1.571, 0])

    INIT_QPOS = np.array([1.57,  -15.0 / 180.0 * np.pi, 100.0 / 180.0 * np.pi, 25.0 / 180.0 * np.pi, 90.0 / 180.0 * np.pi, 0])
    
    # 关节名称（对应 MuJoCo XML 中的 joint name）
    JOINT_NAMES = [
        "shoulder_joint", "upperArm_joint", "foreArm_joint",
        "wrist1_joint", "wrist2_joint", "wrist3_joint"
    ]
    
    # DH参数
    DH_PARAMS = {
        'a': [0, 0.408, 0.376, 0, 0, 0],
        'd': [0.122, 0, 0, 0.1215, 0.1025, 0.094],
        'alpha': [np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0],
    }
    
    # 传感器配置
    SENSOR_CONFIG = {
        "KWR75B": {
            "model_path": "models/sensors/ftsensor/KWR75B.stl",
            "height": 0.038,  # 传感器厚度
            "scale": [0.001, 0.001, 0.001], # STL单位为mm
            "mass": 0.2, # 质量 kg
        }
    }
    
    def __init__(
        self,
        controller_type: str = "joint_position",
        control_freq: int = 20,
        mount_name: str = "pedestal",
        gripper_name: Optional[str] = None,
        use_ft_sensor: Optional[str] = None,
        **kwargs
    ):
        """
        初始化Aubo i5机器人配置
        
        Args:
            controller_type: 控制器类型
            control_freq: 控制频率
            mount_name: 底座名称
            gripper_name: 夹爪名称
            use_ft_sensor: 是否使用力传感器(传感器名称)
            **kwargs: 其他配置参数
        """
        # 计算底座位置和机器人偏移
        mount_pos = self.MOUNT_POS.get(mount_name, [0, 0, 0])
        robot_offset = self.ROBOT_OFFSET.get(mount_name, [0, 0, 0])
        
        super().__init__(
            name=self.NAME,
            robot_type="Aubo_i5",
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
        mount_name: str = "pedestal",
        gripper_name: Optional[str] = None,
        use_ft_sensor: Optional[str] = None,
        **kwargs
    ) -> "AuboI5Robot":
        """
        获取通用机器人配置
        
        Args:
            controller_type: 控制器类型
            control_freq: 控制频率
            mount_name: 底座名称
            gripper_name: 夹爪名称
            use_ft_sensor: 是否使用力传感器(传感器名称)
            **kwargs: 额外配置
            
        Returns:
            AuboI5Robot: 机器人配置对象
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
        xml_path = base_path / "aubo_i5" / f"{scene}.xml"
        
        if xml_path.exists():
            return xml_path
        
        # 尝试从impl/assets查找
        impl_path = Path(__file__).parent.parent.parent / "impl" / "assets" / "models" / "manipulators" / "Aubo_i5"
        if impl_path.exists():
            return impl_path / "Aubo_i5.xml"
        
        return None
    
    @classmethod
    def validate_qpos(cls, qpos: np.ndarray) -> bool:
        """
        验证关节位置是否在限位内
        
        Args:
            qpos: 关节位置数组 [dof]
            
        Returns:
            bool: 是否有效
        """
        if len(qpos) != cls.DOF:
            return False
        
        for i, q in enumerate(qpos):
            if q < cls.JOINT_LIMITS[i, 0] or q > cls.JOINT_LIMITS[i, 1]:
                return False
        
        return True
    
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
    
    @classmethod
    def get_mount_pos(cls, mount_name: str) -> list:
        """获取底座位置"""
        return cls.MOUNT_POS.get(mount_name, [0, 0, 0])
    
    @classmethod
    def get_info(cls) -> Dict[str, any]:
        """获取机器人信息"""
        return {
            "name": cls.NAME,
            "dof": cls.DOF,
            "manufacturer": "Aubo Robotics",
            "payload": 5.0,  # kg
            "reach": 0.920,  # meters
            "joint_limits": cls.JOINT_LIMITS.tolist(),
            "velocity_limits": cls.VELOCITY_LIMITS.tolist(),
            "torque_limits": cls.TORQUE_LIMITS.tolist(),
            "available_mounts": cls.AVAILABLE_MOUNTS,
            "available_grippers": cls.AVAILABLE_GRIPPERS,
        }
    
    @classmethod
    def list_mounts(cls) -> list:
        """列出支持的底座类型"""
        return cls.AVAILABLE_MOUNTS.copy()
    
    @classmethod
    def list_grippers(cls) -> list:
        """列出支持的夹爪类型"""
        return cls.AVAILABLE_GRIPPERS.copy()

