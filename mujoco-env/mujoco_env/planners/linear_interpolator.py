"""
线性插值器

使用Ruckig库实现笛卡尔空间的平滑轨迹插值。

支持：
- 位置插值（3D位置）
- 姿态插值（四元数 → 欧拉角）
- 速度/加速度/加加速度约束

作者: Liu Gang
日期: 2025-12-20
"""

import numpy as np
import math
from mujoco_env.mujoco_env.utils import transform as T

try:
    from ruckig import InputParameter, OutputParameter, Result, Ruckig
except ImportError:
    raise ImportError(
        "ruckig is required but not installed. "
        "Please install it using: pip install ruckig"
    )

EPS = np.finfo(float).eps * 4.0


class LinearInterpolator:
    """
    笛卡尔空间线性插值器
    
    使用Ruckig进行实时轨迹生成，在笛卡尔空间中平滑插值。
    支持位置（xyz）和姿态（四元数）的同时插值。
    
    Attributes:
        control_cycle: 控制周期 (s)
        otg: Ruckig在线轨迹生成器（6维：3位置+3姿态）
        inp: Ruckig输入参数
        out: Ruckig输出参数
        max_velocity: 最大速度限制
        max_acceleration: 最大加速度限制
        max_jerk: 最大加加速度限制
    
    Note:
        姿态使用欧拉角进行插值，输入输出使用四元数表示
    """

    def __init__(
            self,
            control_cycle: float = 0.005,
            max_velocity: float = 0.2,
            max_acceleration: float = 0.2,
            max_jerk: float = 0.2
    ):
        """
        初始化线性插值器
        
        Args:
            control_cycle: 控制周期，单位秒，默认5ms
            max_velocity: 最大速度，默认0.2 m/s 或 rad/s
            max_acceleration: 最大加速度，默认0.2 m/s² 或 rad/s²
            max_jerk: 最大加加速度，默认0.2 m/s³ 或 rad/s³
        """
        self.control_cycle = control_cycle
        self.otg = Ruckig(6, control_cycle)  # 6-DOF: 3位置 + 3姿态（欧拉角）

        self.inp = InputParameter(6)
        self.out = OutputParameter(6)

        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.max_jerk = max_jerk

    def set_params(self, curr_pos: np.ndarray, curr_vel: np.ndarray):
        """
        设置当前状态参数
        
        Args:
            curr_pos: 当前位姿 (7,) - [x, y, z, qx, qy, qz, qw]
            curr_vel: 当前速度 (6,) - [vx, vy, vz, wx, wy, wz]
        """
        # 将四元数姿态转换为欧拉角进行插值
        self.inp.current_position = np.concatenate([
            curr_pos[:3],  # 位置
            T.quat_2_euler(curr_pos[3:])  # 四元数 → 欧拉角
        ])
        self.inp.current_velocity = curr_vel
        self.inp.current_acceleration = np.zeros(6)

        # 初始化目标为零（将在update_target_position中设置）
        self.inp.target_position = np.zeros(6)
        self.inp.target_velocity = np.zeros(6)
        self.inp.target_acceleration = np.zeros(6)

        # 设置运动约束
        self.inp.max_velocity = self.max_velocity * np.ones(6)
        self.inp.max_acceleration = self.max_acceleration * np.ones(6)
        self.inp.max_jerk = self.max_jerk * np.ones(6)

    def update_target_position(self, action: np.ndarray):
        """
        更新目标位姿
        
        Args:
            action: 目标位姿 (7,) - [x, y, z, qx, qy, qz, qw]
        """
        # 将四元数姿态转换为欧拉角
        self.inp.target_position = np.concatenate([
            action[:3],  # 目标位置
            T.quat_2_euler(action[3:])  # 四元数 → 欧拉角
        ])

    def update_state(self):
        """
        更新插值状态，计算下一步轨迹点
        
        Returns:
            tuple: (位姿, 速度, 加速度)
                - 位姿 (7,): [x, y, z, qx, qy, qz, qw]
                - 速度 (6,): [vx, vy, vz, wx, wy, wz]
                - 加速度 (6,): [ax, ay, az, alpha_x, alpha_y, alpha_z]
        """
        self.otg.update(self.inp, self.out)
        
        pos_out = self.out.new_position[:6]
        vel_out = self.out.new_velocity
        acc_out = self.out.new_acceleration
        
        # 将当前状态传递给下一次输入
        self.out.pass_to_input(self.inp)
        
        # 将欧拉角转换回四元数
        pose_out = np.concatenate([
            pos_out[:3],  # 位置
            T.euler_2_quat(pos_out[3:])  # 欧拉角 → 四元数
        ])
        
        return pose_out, vel_out, acc_out
