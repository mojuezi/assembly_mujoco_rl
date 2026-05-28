"""
关节力矩控制器（直接力矩控制）

作者: Liu Gang
日期: 2025-12-20
"""

from typing import Dict, Optional
import numpy as np
from mujoco_env.mujoco_env.controllers.base_controller import BaseController


class JointTorqueController(BaseController):
    """
    关节力矩控制器
    
    直接输出目标力矩（可选力矩限制）
    """
    
    def __init__(
        self,
        model,
        data,
        dof: int,
        control_freq: float = 20.0,
        torque_limits: Optional[np.ndarray] = None,
        **kwargs
    ):
        """
        初始化关节力矩控制器
        
        Args:
            model: MuJoCo模型
            data: MuJoCo数据
            dof: 自由度
            control_freq: 控制频率
            torque_limits: 力矩限制 [dof] 或 [dof, 2]
            **kwargs: 其他参数
        """
        super().__init__(model, data, dof, control_freq, **kwargs)
        
        # 设置力矩限制
        if torque_limits is not None:
            torque_limits = np.array(torque_limits)
            if torque_limits.ndim == 1:
                # 如果是一维数组，假设是对称限制
                self.torque_limits = np.stack([-torque_limits, torque_limits], axis=1)
            else:
                self.torque_limits = torque_limits
        else:
            # 没有限制
            self.torque_limits = None
    
    def compute_control(
        self, 
        target: np.ndarray, 
        current_state: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """
        计算力矩控制输出
        
        Args:
            target: 目标关节力矩 [dof]
            current_state: 当前状态（此控制器不需要）
            
        Returns:
            torque: 关节力矩 [dof]
        """
        torque = np.array(target)
        
        # 应用力矩限制
        if self.torque_limits is not None:
            for i in range(self.dof):
                torque[i] = np.clip(
                    torque[i],
                    self.torque_limits[i, 0],
                    self.torque_limits[i, 1]
                )
        
        return torque

