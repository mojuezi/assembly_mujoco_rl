"""
控制器模块

提供各种机器人控制器，包括：
- 关节空间控制：位置、速度、力矩、阻抗、自由驱动
- 任务空间控制：OSC、笛卡尔阻抗、笛卡尔IK、导纳控制

作者: Liu Gang
日期: 2025-12-21
"""

from typing import Dict, Type
from .base_controller import BaseController
from .joint_position import JointPositionController
from .joint_velocity import JointVelocityController
from .joint_torque import JointTorqueController
from .joint_impedance import JointImpedanceController
from .joint_freedrive import JointFreedriveController
from .operational_space import (
    OperationalSpaceController,
    OSCController,
    TaskSpaceController,
    opspace,  # 原始函数（向后兼容）
)
from .task_space_base import TaskSpaceController as TaskSpaceControllerBase
from .cartesian_impedance import CartesianImpedanceController
from .cartesian_ik import CartesianIKController
from .admittance import AdmittanceController


# 控制器注册表
CONTROLLER_REGISTRY: Dict[str, Type[BaseController]] = {
    # 关节空间控制器
    "joint_position": JointPositionController,
    "joint_velocity": JointVelocityController,
    "joint_torque": JointTorqueController,
    "joint_impedance": JointImpedanceController,
    "joint_freedrive": JointFreedriveController,
    
    # 任务空间控制器
    "operational_space": OperationalSpaceController,
    "cartesian_impedance": CartesianImpedanceController,
    "cartesian_ik": CartesianIKController,
    "admittance": AdmittanceController,
    
    # OSC别名
    "osc": OperationalSpaceController,
    "task_space": OperationalSpaceController,
    
    # 笛卡尔控制器别名
    "cart_imp": CartesianImpedanceController,
    "cart_ik": CartesianIKController,
    "cart_adm": AdmittanceController,
    
    # 简写别名
    "position": JointPositionController,
    "velocity": JointVelocityController,
    "torque": JointTorqueController,
    "impedance": JointImpedanceController,
    "freedrive": JointFreedriveController,
}


def get_controller(controller_type: str, **kwargs) -> BaseController:
    """
    获取控制器实例
    
    Args:
        controller_type: 控制器类型
        **kwargs: 控制器参数，必须包含:
            - model: MuJoCo模型
            - data: MuJoCo数据
            - dof: 自由度
            - control_freq: 控制频率（可选）
        
    Returns:
        controller: 控制器实例
        
    Raises:
        ValueError: 如果控制器类型不支持
    
    Example:
        >>> controller = get_controller(
        ...     'joint_position',
        ...     model=model,
        ...     data=data,
        ...     dof=7,
        ...     control_freq=20,
        ...     kp=np.array([200.0] * 7),
        ...     kd=np.array([20.0] * 7)
        ... )
    """
    if controller_type not in CONTROLLER_REGISTRY:
        available = list(CONTROLLER_REGISTRY.keys())
        raise ValueError(
            f"Unknown controller type: {controller_type}. "
            f"Available types: {available}"
        )
    
    controller_cls = CONTROLLER_REGISTRY[controller_type]
    return controller_cls(**kwargs)


def register_controller(name: str, controller_class: Type[BaseController]):
    """
    注册新的控制器类型
    
    Args:
        name: 控制器名称
        controller_class: 控制器类
    """
    CONTROLLER_REGISTRY[name] = controller_class


__all__ = [
    # 基类
    "BaseController",
    "TaskSpaceControllerBase",
    
    # 关节空间控制器
    "JointPositionController",
    "JointVelocityController",
    "JointTorqueController",
    "JointImpedanceController",
    "JointFreedriveController",
    
    # 任务空间控制器
    "OperationalSpaceController",
    "OSCController",
    "TaskSpaceController",
    "CartesianImpedanceController",
    "CartesianIKController",
    "AdmittanceController",
    
    # 向后兼容
    "opspace",  # 原始函数
    
    # 工具函数
    "get_controller",
    "register_controller",
    "CONTROLLER_REGISTRY",
]
