"""
笛卡尔空间阻抗控制器

在笛卡尔坐标系（任务空间）中实现阻抗控制，允许在末端执行器层面直接控制位置和姿态的柔顺性。

控制方程:
τ = J^T · (K_c · x_error - B_c · ẋ + M_d · (ẍ_d - J̇·q̇)) + C(q, q̇) + G(q)

其中:
- J: 雅可比矩阵
- K_c: 笛卡尔刚度矩阵 (6x6)
- B_c: 笛卡尔阻尼矩阵 (6x6)
- M_d: 笛卡尔空间质量矩阵
- x_error: 位姿误差 (位置3 + 姿态3)

作者: Liu Gang
日期: 2025-12-21
"""

import numpy as np
import mujoco
from typing import Optional, Dict
from mujoco_env.mujoco_env.controllers.task_space_base import (
    TaskSpaceController,
    orientation_error
)
from mujoco_env.mujoco_env.utils import transform as T


class CartesianImpedanceController(TaskSpaceController):
    """
    笛卡尔空间阻抗控制器
    
    在末端执行器的笛卡尔坐标系中实现阻抗控制。
    相比关节空间阻抗控制，笛卡尔阻抗控制更直观，
    适合需要在任务空间直接控制柔顺性的场景。
    
    Attributes:
        Kc: 笛卡尔刚度 (6,) - [Kx, Ky, Kz, Krx, Kry, Krz]
        Bc: 笛卡尔阻尼 (6,) - [Bx, By, Bz, Brx, Bry, Brz]
    """
    
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        dof: int,
        control_freq: float = 20.0,
        ee_site_name: str = "pinch",
        kp_pos: Optional[np.ndarray] = None,
        kp_ori: Optional[np.ndarray] = None,
        kd_pos: Optional[np.ndarray] = None,
        kd_ori: Optional[np.ndarray] = None,
        **kwargs
    ):
        """
        初始化笛卡尔阻抗控制器
        
        Args:
            model: MuJoCo模型
            data: MuJoCo数据
            dof: 自由度数量
            control_freq: 控制频率 (Hz)
            ee_site_name: 末端执行器site名称
            kp_pos: 位置刚度 (3,)，默认[100, 100, 100]
            kp_ori: 姿态刚度 (3,)，默认[200, 200, 200]
            kd_pos: 位置阻尼 (3,)，默认[200, 800, 800]
            kd_ori: 姿态阻尼 (3,)，默认[400, 400, 400]
            **kwargs: 额外参数
        """
        super().__init__(model, data, dof, control_freq, ee_site_name, **kwargs)
        
        # 设置默认增益
        self.kp_pos = kp_pos if kp_pos is not None else np.array([100.0, 100.0, 100.0])
        self.kp_ori = kp_ori if kp_ori is not None else np.array([200.0, 200.0, 200.0])
        self.kd_pos = kd_pos if kd_pos is not None else np.array([200.0, 800.0, 800.0])
        self.kd_ori = kd_ori if kd_ori is not None else np.array([400.0, 400.0, 400.0])
        
        # 组合为6维向量
        self.Kc = np.concatenate([self.kp_pos, self.kp_ori])
        self.Bc = np.concatenate([self.kd_pos, self.kd_ori])
    
    def compute_control(
        self,
        target: np.ndarray,
        current_state: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """
        计算笛卡尔阻抗控制力矩
        
        Args:
            target: 目标位姿 (7,) [x, y, z, qw, qx, qy, qz]
                   或 (6,) [x, y, z, rx, ry, rz] (欧拉角)
            current_state: 当前状态，包含:
                - qpos: 关节位置
                - qvel: 关节速度
        
        Returns:
            tau (dof,): 关节力矩
        """
        # 解析目标位姿
        if len(target) == 7:
            # 四元数表示
            target_pos = target[:3]
            target_quat = target[3:]
        elif len(target) == 6:
            # 欧拉角表示
            target_pos = target[:3]
            target_quat = T.euler_2_quat(target[3:])
        else:
            raise ValueError(f"Invalid target shape: {target.shape}. Expected (7,) or (6,)")
        
        # 获取当前状态
        if current_state is None:
            current_state = self.get_state()
        
        q_cur = current_state["qpos"]
        qd_cur = current_state["qvel"]
        
        # 正运动学
        current_pos, current_quat = self.forward_kinematics()
        current_mat = T.quat_2_mat(current_quat)
        target_mat = T.quat_2_mat(target_quat)
        
        # 计算雅可比
        J_pos, J_rot = self.compute_jacobian()
        J = np.vstack([J_pos, J_rot])  # (6, dof)
        
        # 计算质量矩阵
        M = self.compute_mass_matrix()
        
        # 笛卡尔空间质量矩阵 (可选)
        # Md = J_inv.T @ M @ J_inv
        # 为简化，这里直接使用关节空间质量矩阵
        
        # 位置误差
        pos_error = target_pos - current_pos
        
        # 姿态误差
        ori_error = orientation_error(target_mat, current_mat)
        
        # 6维误差
        x_error = np.concatenate([pos_error, ori_error])
        
        # 笛卡尔速度
        v_cart = J @ qd_cur
        
        # 笛卡尔空间力
        F_cart = self.Kc * x_error - self.Bc * v_cart
        
        # 雅可比转置映射到关节空间
        tau = J.T @ F_cart
        
        # 添加重力和科氏力补偿
        tau += self.data.qfrc_bias[:self.dof]
        
        return tau
    
    def set_gains(
        self,
        kp_pos: Optional[np.ndarray] = None,
        kp_ori: Optional[np.ndarray] = None,
        kd_pos: Optional[np.ndarray] = None,
        kd_ori: Optional[np.ndarray] = None
    ):
        """
        设置控制器增益
        
        Args:
            kp_pos: 位置刚度 (3,)
            kp_ori: 姿态刚度 (3,)
            kd_pos: 位置阻尼 (3,)
            kd_ori: 姿态阻尼 (3,)
        """
        if kp_pos is not None:
            self.kp_pos = kp_pos
        if kp_ori is not None:
            self.kp_ori = kp_ori
        if kd_pos is not None:
            self.kd_pos = kd_pos
        if kd_ori is not None:
            self.kd_ori = kd_ori
        
        # 更新组合向量
        self.Kc = np.concatenate([self.kp_pos, self.kp_ori])
        self.Bc = np.concatenate([self.kd_pos, self.kd_ori])
    
    def __repr__(self) -> str:
        return (
            f"CartesianImpedanceController("
            f"dof={self.dof}, "
            f"freq={self.control_freq}Hz, "
            f"Kp_pos={self.kp_pos[0]:.1f}, "
            f"Kp_ori={self.kp_ori[0]:.1f})"
        )

