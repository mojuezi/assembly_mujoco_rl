"""
关节阻抗控制器

实现关节空间的阻抗控制，通过设置虚拟弹簧-阻尼系统来实现柔顺控制。

阻抗控制方程：
τ = M(q)·(K·(q_d - q) + B·(q̇_d - q̇)) + C(q, q̇) + G(q)

其中：
- M(q): 质量矩阵
- K: 刚度矩阵
- B: 阻尼矩阵
- C(q, q̇): 科氏力和离心力
- G(q): 重力

作者: Liu Gang
日期: 2025-12-20
"""

import numpy as np
import mujoco
from typing import Optional, Dict
from mujoco_env.mujoco_env.controllers.base_controller import BaseController


class JointImpedanceController(BaseController):
    """
    关节空间阻抗控制器
    
    通过配置虚拟弹簧刚度(K)和阻尼(B)参数，实现机器人的柔顺控制。
    适用于需要力控制或接触任务的场景。
    
    Attributes:
        K: 刚度矩阵 (N·m/rad)
        B: 阻尼矩阵 (N·m·s/rad)
        use_gravity_compensation: 是否使用重力补偿
    """
    
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        dof: int,
        control_freq: float = 20.0,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
        use_gravity_compensation: bool = True,
        **kwargs
    ):
        """
        初始化关节阻抗控制器
        
        Args:
            model: MuJoCo模型
            data: MuJoCo数据
            dof: 自由度数量
            control_freq: 控制频率 (Hz)
            kp: 位置增益(刚度) (dof,)，默认40000.0
            kd: 速度增益(阻尼) (dof,)，默认282.8
            use_gravity_compensation: 是否使用重力补偿，默认True
            **kwargs: 额外参数
        """
        super().__init__(model, data, dof, control_freq, **kwargs)
        
        # 设置默认增益
        self.K = kp if kp is not None else 40000.0 * np.ones(dof)
        self.B = kd if kd is not None else 282.8 * np.ones(dof)
        
        self.use_gravity_compensation = use_gravity_compensation
        
        # 确保增益矩阵形状正确
        assert self.K.shape == (dof,), f"K shape must be ({dof},), got {self.K.shape}"
        assert self.B.shape == (dof,), f"B shape must be ({dof},), got {self.B.shape}"
    
    def compute_control(
        self,
        target: np.ndarray,
        current_state: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """
        计算阻抗控制力矩
        
        Args:
            target: 目标关节位置 (dof,)
            current_state: 当前状态，包含:
                - qpos: 当前关节位置 (dof,)
                - qvel: 当前关节速度 (dof,)
                - target_vel: 目标关节速度 (dof,)，可选，默认为0
        
        Returns:
            tau: 关节力矩 (dof,)
        """
        # 获取当前状态
        if current_state is None:
            current_state = self.get_state()
        
        q_cur = current_state["qpos"][:self.dof]
        qd_cur = current_state["qvel"][:self.dof]
        
        # 目标速度（默认为0）
        qd_des = current_state.get("target_vel", np.zeros(self.dof))
        
        # 计算质量矩阵
        M = np.zeros((self.dof, self.dof))
        mujoco.mj_fullM(self.model, M, self.data.qM)
        M = M[:self.dof, :self.dof]
        
        # 计算期望加速度
        acc_des = self.K * (target - q_cur) + self.B * (qd_des - qd_cur)
        
        # 计算力矩
        tau = M @ acc_des
        
        # 添加重力和科氏力补偿
        if self.use_gravity_compensation:
            tau += self.data.qfrc_bias[:self.dof]  # 包含重力和科氏力
        
        return tau
    
    def set_gains(self, kp: Optional[np.ndarray] = None, kd: Optional[np.ndarray] = None):
        """
        设置控制器增益
        
        Args:
            kp: 位置增益(刚度)
            kd: 速度增益(阻尼)
        """
        if kp is not None:
            assert kp.shape == (self.dof,), f"kp shape must be ({self.dof},), got {kp.shape}"
            self.K = kp
        if kd is not None:
            assert kd.shape == (self.dof,), f"kd shape must be ({self.dof},), got {kd.shape}"
            self.B = kd
    
    def __repr__(self) -> str:
        return (
            f"JointImpedanceController("
            f"dof={self.dof}, "
            f"freq={self.control_freq}Hz, "
            f"K={self.K[0]:.1f}, "
            f"B={self.B[0]:.1f})"
        )

