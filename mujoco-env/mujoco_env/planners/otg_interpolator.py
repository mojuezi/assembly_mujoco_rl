"""
关节空间轨迹插值器

使用Ruckig库实现关节空间的平滑轨迹插值。

支持：
- 关节位置插值
- 速度/加速度/加加速度约束
- 多自由度机械臂（默认7-DOF）

作者: Liu Gang
日期: 2025-12-20
"""

import numpy as np

try:
    from ruckig import InputParameter, OutputParameter, Result, Ruckig
except ImportError:
    raise ImportError(
        "ruckig is required but not installed. "
        "Please install it using: pip install ruckig"
    )


class OTG:
    """
    在线轨迹生成器（Online Trajectory Generation）
    
    使用Ruckig进行关节空间的实时轨迹生成，确保速度、加速度和
    加加速度约束下的平滑运动。
    
    Attributes:
        otg: Ruckig轨迹生成器实例
        inp: Ruckig输入参数
        out: Ruckig输出参数
        max_velocity: 最大关节速度 (rad/s)
        max_acceleration: 最大关节加速度 (rad/s²)
        max_jerk: 最大关节加加速度 (rad/s³)
    
    Note:
        默认配置为7-DOF机械臂（如Franka Panda）
    """
    
    def __init__(
            self,
            OTG_dim: int = 7,
            control_cycle: float = 0.001,
            max_velocity: float = 0.0,
            max_acceleration: float = 0.0,
            max_jerk: float = 0.0
    ):
        """
        初始化在线轨迹生成器
        
        Args:
            OTG_dim: 自由度数量，默认7（适用于7-DOF机械臂）
            control_cycle: 控制周期，单位秒，默认1ms
            max_velocity: 最大关节速度，rad/s。0表示无限制
            max_acceleration: 最大关节加速度，rad/s²。0表示无限制
            max_jerk: 最大关节加加速度，rad/s³。0表示无限制
        """
        self.dof = OTG_dim
        self.otg = Ruckig(OTG_dim, control_cycle)

        self.inp = InputParameter(OTG_dim)
        self.out = OutputParameter(OTG_dim)

        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.max_jerk = max_jerk

    def set_params(self, qpos: np.ndarray, qvel: np.ndarray):
        """
        设置当前关节状态参数
        
        Args:
            qpos: 当前关节位置 (n,)
            qvel: 当前关节速度 (n,)
        """
        self.inp.current_position = qpos
        self.inp.current_velocity = qvel
        self.inp.current_acceleration = np.zeros(self.dof)

        # 初始化目标状态为零
        self.inp.target_position = np.zeros(self.dof)
        self.inp.target_velocity = np.zeros(self.dof)
        self.inp.target_acceleration = np.zeros(self.dof)

        # 设置运动约束
        if self.max_velocity > 0:
            self.inp.max_velocity = self.max_velocity * np.ones(self.dof)
        if self.max_acceleration > 0:
            self.inp.max_acceleration = self.max_acceleration * np.ones(self.dof)
        if self.max_jerk > 0:
            self.inp.max_jerk = self.max_jerk * np.ones(self.dof)

    def update_target_position(self, action: np.ndarray):
        """
        更新目标关节位置
        
        Args:
            action: 目标关节位置 (n,)
        """
        self.inp.target_position = action

    def update_state(self):
        """
        更新插值状态，计算下一步轨迹点
        
        Returns:
            tuple: (目标位置, 目标速度)
                - q_target (n,): 插值后的关节位置
                - qd_target (n,): 插值后的关节速度
        """
        self.otg.update(self.inp, self.out)
        
        q_target = self.out.new_position
        qd_target = self.out.new_velocity
        
        # 将当前状态传递给下一次输入
        self.out.pass_to_input(self.inp)
        
        return q_target, qd_target
