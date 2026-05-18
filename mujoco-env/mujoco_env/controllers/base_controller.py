"""
控制器基类

作者: Liu Gang
日期: 2025-12-20
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
import mujoco


class BaseController(ABC):
    """
    控制器基类
    
    所有控制器都应继承此类并实现compute_control方法
    """
    
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        dof: int,
        control_freq: float = 20.0,
        **kwargs
    ):
        """
        初始化控制器
        
        Args:
            model: MuJoCo模型
            data: MuJoCo数据
            dof: 自由度数量
            control_freq: 控制频率 (Hz)
            **kwargs: 额外参数
        """
        self.model = model
        self.data = data
        self.dof = dof
        self.control_freq = control_freq
        self.control_dt = 1.0 / control_freq
        
        # 控制器名称
        self.name = self.__class__.__name__
    
    @abstractmethod
    def compute_control(
        self, 
        target: np.ndarray, 
        current_state: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """
        计算控制输出
        
        Args:
            target: 目标值（可能是位置、速度、力矩等）
            current_state: 当前状态字典，包含:
                - qpos: 关节位置
                - qvel: 关节速度
                - qacc: 关节加速度（可选）
                
        Returns:
            control: 控制输出（通常是关节力矩）
        """
        raise NotImplementedError
    
    def reset(self):
        """重置控制器状态"""
        pass
    
    def get_state(self) -> Dict[str, np.ndarray]:
        """
        从MuJoCo数据中获取当前状态
        
        Returns:
            状态字典
        """
        return {
            "qpos": self.data.qpos[:self.dof].copy(),
            "qvel": self.data.qvel[:self.dof].copy(),
        }
    
    def set_gains(self, **kwargs):
        """
        设置控制器增益（子类可选实现）
        
        Args:
            **kwargs: 增益参数
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.name}(dof={self.dof}, freq={self.control_freq}Hz)"

