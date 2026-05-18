"""
关节位置控制器 (PD控制)

作者: Liu Gang
日期: 2025-12-20
"""

from typing import Dict, Optional
import numpy as np
from mujoco_env.mujoco_env.controllers.base_controller import BaseController


class JointPositionController(BaseController):
    """
    关节空间位置控制器（PD控制）
    
    实现公式: τ = Kp * (q_target - q) + Kd * (0 - dq)
    """
    
    def __init__(
        self,
        model,
        data,
        dof: int,
        control_freq: float = 20.0,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
        site_name: str = None, # 兼容性参数，可能会被传入但未使用
        use_target_as_ctrl: bool = False, # 新增参数：是否直接将目标作为控制信号
        **kwargs
    ):
        """
        初始化关节位置控制器
        
        Args:
            model: MuJoCo模型
            data: MuJoCo数据
            dof: 自由度
            control_freq: 控制频率
            kp: 比例增益（如果为None，使用默认值）
            kd: 微分增益（如果为None，使用默认值）
            site_name: 兼容性参数，未使用
            use_target_as_ctrl: 如果为True，直接返回target作为控制信号（用于Position Actuators）
            **kwargs: 其他参数
        """
        super().__init__(model, data, dof, control_freq, **kwargs)
        
        self.use_target_as_ctrl = use_target_as_ctrl
        
        # 默认PD增益
        if kp is None:
            self.kp = np.array([200.0] * dof)
        else:
            self.kp = np.array(kp)
        
        if kd is None:
            # 临界阻尼: Kd = 2 * sqrt(Kp)
            self.kd = 2.0 * np.sqrt(self.kp)
        else:
            self.kd = np.array(kd)
        
        # 验证增益维度
        assert len(self.kp) == dof, f"kp维度不匹配: {len(self.kp)} != {dof}"
        assert len(self.kd) == dof, f"kd维度不匹配: {len(self.kd)} != {dof}"
    
    def compute_control(
        self, 
        target: np.ndarray, 
        current_state: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """
        计算PD控制输出
        
        Args:
            target: 目标关节位置 [dof]
            current_state: 当前状态，包含qpos和qvel
            
        Returns:
            output: 关节控制信号 [dof] (力矩 或 位置)
        """
        # 如果配置为直接使用目标作为控制信号（针对 Position Actuators）
        if self.use_target_as_ctrl:
            return target

        # 获取当前状态
        if current_state is None:
            current_state = self.get_state()
        
        qpos = current_state["qpos"][:self.dof]
        qvel = current_state["qvel"][:self.dof]
        
        # PD控制
        position_error = target - qpos
        velocity_error = 0 - qvel
        
        torque = self.kp * position_error + self.kd * velocity_error
        
        return torque
    
    def set_gains(self, kp: Optional[np.ndarray] = None, kd: Optional[np.ndarray] = None):
        """
        设置PD增益
        
        Args:
            kp: 比例增益
            kd: 微分增益
        """
        if kp is not None:
            self.kp = np.array(kp)
        if kd is not None:
            self.kd = np.array(kd)

