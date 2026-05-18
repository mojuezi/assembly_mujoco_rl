"""
关节自由驱动控制器

实现机器人的自由驱动模式，允许用户手动移动机器人。
这是一种特殊的阻抗控制器，刚度和阻尼都设置为零或很小的值。

应用场景：
- 手动示教
- 拖动示教
- 力反馈交互

作者: Liu Gang
日期: 2025-12-20
"""

import numpy as np
import mujoco
from typing import Optional, Dict
from mujoco_env.mujoco_env.controllers.joint_impedance import JointImpedanceController


class JointFreedriveController(JointImpedanceController):
    """
    关节自由驱动控制器
    
    允许机器人在关节空间自由移动，适用于手动示教和拖动编程。
    本质上是一个零刚度、低阻尼的阻抗控制器，仅进行重力补偿。
    
    与普通阻抗控制器的区别：
    - 刚度K = 0 (无弹性阻力)
    - 阻尼B = 很小值 (少量阻尼以避免振荡)
    - 目标位置设置为当前位置 (跟随运动)
    
    Attributes:
        damping: 阻尼系数，默认为较小值以提供少量阻尼
    """
    
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        dof: int,
        control_freq: float = 20.0,
        damping: float = 5.0,
        use_gravity_compensation: bool = True,
        **kwargs
    ):
        """
        初始化自由驱动控制器
        
        Args:
            model: MuJoCo模型
            data: MuJoCo数据
            dof: 自由度数量
            control_freq: 控制频率 (Hz)
            damping: 阻尼系数，提供少量阻尼以避免振荡，默认5.0
            use_gravity_compensation: 是否使用重力补偿，默认True
            **kwargs: 额外参数
        """
        # 初始化为零刚度、低阻尼的阻抗控制器
        super().__init__(
            model=model,
            data=data,
            dof=dof,
            control_freq=control_freq,
            kp=np.zeros(dof),  # 零刚度
            kd=damping * np.ones(dof),  # 小阻尼
            use_gravity_compensation=use_gravity_compensation,
            **kwargs
        )
        
        self.damping = damping
    
    def compute_control(
        self,
        target: Optional[np.ndarray] = None,
        current_state: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """
        计算自由驱动控制力矩
        
        在自由驱动模式下，目标位置总是当前位置，因此机器人不会主动移动，
        只进行重力补偿和少量阻尼。
        
        Args:
            target: 目标位置（在自由驱动模式下忽略）
            current_state: 当前状态，包含:
                - qpos: 当前关节位置
                - qvel: 当前关节速度
        
        Returns:
            tau: 关节力矩 (dof,)，主要是重力补偿和阻尼力矩
        """
        # 获取当前状态
        if current_state is None:
            current_state = self.get_state()
        
        q_cur = current_state["qpos"][:self.dof]
        qd_cur = current_state["qvel"][:self.dof]
        
        # 在自由驱动模式下，目标位置就是当前位置
        # 目标速度为0
        target_pos = q_cur
        target_vel = np.zeros(self.dof)
        
        # 计算质量矩阵
        M = np.zeros((self.dof, self.dof))
        mujoco.mj_fullM(self.model, M, self.data.qM)
        M = M[:self.dof, :self.dof]
        
        # 只应用阻尼力（K=0，所以没有弹性力）
        acc_des = self.B * (target_vel - qd_cur)  # 只有阻尼项
        
        # 计算力矩
        tau = M @ acc_des
        
        # 添加重力补偿（关键！）
        if self.use_gravity_compensation:
            tau += self.data.qfrc_bias[:self.dof]
        
        return tau
    
    def set_damping(self, damping: float):
        """
        设置阻尼系数
        
        Args:
            damping: 阻尼系数，较小的值使机器人更容易移动，
                    较大的值提供更多阻尼以减少振荡
        """
        self.damping = damping
        self.B = damping * np.ones(self.dof)
    
    def __repr__(self) -> str:
        return (
            f"JointFreedriveController("
            f"dof={self.dof}, "
            f"freq={self.control_freq}Hz, "
            f"damping={self.damping})"
        )

