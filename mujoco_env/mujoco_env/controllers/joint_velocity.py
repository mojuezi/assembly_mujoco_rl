"""
关节速度控制器

作者: Liu Gang
日期: 2025-12-20
"""

from typing import Dict, Optional
import numpy as np
from mujoco_env.mujoco_env.controllers.base_controller import BaseController


class JointVelocityController(BaseController):
    """
    关节速度控制器（P控制）
    
    实现公式: τ = Kv * (dq_target - dq)
    """
    
    def __init__(
        self,
        model,
        data,
        dof: int,
        control_freq: float = 20.0,
        kv: Optional[np.ndarray] = None,
        **kwargs
    ):
        """
        初始化关节速度控制器
        
        Args:
            model: MuJoCo模型
            data: MuJoCo数据
            dof: 自由度
            control_freq: 控制频率
            kv: 速度增益（如果为None，使用默认值）
            **kwargs: 其他参数
        """
        super().__init__(model, data, dof, control_freq, **kwargs)
        
        # 默认速度增益
        if kv is None:
            self.kv = np.array([50.0] * dof)
        else:
            self.kv = np.array(kv)
        
        assert len(self.kv) == dof, f"kv维度不匹配: {len(self.kv)} != {dof}"
    
    def compute_control(
        self, 
        target: np.ndarray, 
        current_state: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """
        计算速度控制输出
        
        Args:
            target: 目标关节速度 [dof]
            current_state: 当前状态
            
        Returns:
            torque: 关节力矩 [dof]
        """
        if current_state is None:
            current_state = self.get_state()
        
        qvel = current_state["qvel"][:self.dof]
        
        # P控制
        velocity_error = target - qvel
        torque = self.kv * velocity_error
        
        return torque
    
    def set_gains(self, kv: Optional[np.ndarray] = None):
        """设置速度增益"""
        if kv is not None:
            self.kv = np.array(kv)

