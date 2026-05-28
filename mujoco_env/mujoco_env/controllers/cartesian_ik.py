"""
笛卡尔空间逆运动学控制器

通过逆运动学求解将笛卡尔空间的目标位姿转换为关节空间位置，
然后使用关节位置控制器实现跟踪。

核心功能:
- 数值逆运动学求解 (基于最小二乘优化)
- 支持位置和姿态约束
- 支持关节限位
- 支持正则化项（避免远离初始姿态）

作者: Liu Gang
日期: 2025-12-21
"""

import numpy as np
import mujoco
from mujoco import minimize
from typing import Optional, Dict, Tuple
from mujoco_env.mujoco_env.controllers.task_space_base import TaskSpaceController
from mujoco_env.mujoco_env.controllers.joint_position import JointPositionController
from mujoco_env.mujoco_env.utils import transform as T


class CartesianIKController(TaskSpaceController):
    """
    笛卡尔空间逆运动学控制器
    
    将笛卡尔空间的目标位姿通过IK求解转换为关节空间目标，
    然后使用关节位置控制器实现。
    
    特点:
    - 使用数值优化求解IK
    - 支持位置和姿态约束
    - 支持关节限位
    - 正则化避免奇异配置
    
    Attributes:
        joint_controller: 内部关节位置控制器
        ik_regularization: IK正则化系数
        ik_radius: 姿态误差缩放因子
    """
    
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        dof: int,
        control_freq: float = 20.0,
        ee_site_name: str = "pinch",
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
        ik_regularization: float = 0.01,
        ik_radius: float = 0.2,
        joint_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        use_target_as_ctrl: bool = False,
        **kwargs
    ):
        """
        初始化笛卡尔IK控制器
        
        Args:
            model: MuJoCo模型
            data: MuJoCo数据
            dof: 自由度数量
            control_freq: 控制频率 (Hz)
            ee_site_name: 末端执行器site名称
            kp: 关节位置增益 (dof,)
            kd: 关节速度增益 (dof,)
            ik_regularization: IK正则化系数，默认0.01
            ik_radius: 姿态误差缩放因子，默认0.2
            joint_limits: 关节限位 (lower, upper)，每个都是(dof,)数组
            use_target_as_ctrl: 是否直接使用目标作为控制信号
            **kwargs: 额外参数
        """
        super().__init__(model, data, dof, control_freq, ee_site_name, **kwargs)
        
        # 创建内部关节位置控制器
        self.joint_controller = JointPositionController(
            model=model,
            data=data,
            dof=dof,
            control_freq=control_freq,
            kp=kp,
            kd=kd,
            use_target_as_ctrl=use_target_as_ctrl
        )
        
        # IK参数
        self.ik_regularization = ik_regularization
        self.ik_radius = ik_radius
        
        # 关节限位
        if joint_limits is not None:
            lower, upper = joint_limits
        else:
            # 从模型中获取
            lower = model.jnt_range[:dof, 0].copy()  # lower
            upper = model.jnt_range[:dof, 1].copy()  # upper
            
            # 检查并修复无效的限位（lower >= upper 或包含 inf）
            for i in range(dof):
                if not np.isfinite(lower[i]) or not np.isfinite(upper[i]) or lower[i] >= upper[i]:
                    # 使用默认的大范围限位
                    lower[i] = -np.pi * 2
                    upper[i] = np.pi * 2
        
        # 确保 lower < upper
        self.joint_limits = (lower, upper)
        
        # 记录初始关节位置（用于正则化）
        self.q_initial = self.get_arm_qpos()
    
    def solve_ik(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        q_init: Optional[np.ndarray] = None,
        position_only: bool = False
    ) -> np.ndarray:
        """
        求解逆运动学
        
        使用最小二乘优化求解IK问题:
        min ||p_ee(q) - p_target||^2 + ||quat_ee(q) - quat_target||^2 + reg*||q - q_init||^2
        
        Args:
            target_pos: 目标位置 (3,)
            target_quat: 目标姿态四元数 (4,) [w, x, y, z]
            q_init: 初始猜测关节位置 (dof,)，None表示使用当前位置
            position_only: 是否只约束位置（忽略姿态），默认False
        
        Returns:
            q_solution (dof,): 求解的关节位置
        """
        # 初始猜测
        if q_init is None:
            x = self.get_arm_qpos()
        else:
            x = q_init.copy()
        
        # 使用当前位置作为正则化目标
        x_prev = self.get_arm_qpos()
        
        # 定义残差函数
        def ik_residual(q):
            return self._ik_residual(
                q, 
                target_pos=target_pos,
                target_quat=target_quat,
                reg_target=x_prev,
                position_only=position_only
            )
        
        # 定义雅可比函数
        def ik_jacobian(q, r):
            return self._ik_jacobian(
                q, r,
                target_pos=target_pos,
                target_quat=target_quat,
                position_only=position_only
            )
        
        # 求解IK
        try:
            # 确保 bounds 格式正确：检查 lower < upper
            lower, upper = self.joint_limits
            # 验证 bounds 有效性
            if np.any(lower >= upper):
                invalid_indices = np.where(lower >= upper)[0]
                print(f"警告: 关节限位无效 (lower >= upper) 在索引 {invalid_indices}, 使用默认限位")
                # 使用默认限位
                lower = np.full(self.dof, -np.pi * 2)
                upper = np.full(self.dof, np.pi * 2)
                bounds = (lower, upper)
            else:
                bounds = self.joint_limits
            
            q_solution, _ = minimize.least_squares(
                x, ik_residual,
                bounds=bounds,
                jacobian=ik_jacobian,
                eps=1e-6,
                verbose=0
            )
        except Exception as e:
            # 如果求解失败，返回当前位置以保持稳定
            print(f"IK求解失败: {e}, 保持当前位置")
            q_solution = self.get_arm_qpos()
        
        return q_solution
    
    def _ik_residual(
        self,
        q: np.ndarray,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        reg_target: np.ndarray,
        position_only: bool = False
    ) -> np.ndarray:
        """
        IK残差函数
        
        Args:
            q: 当前关节位置
            target_pos: 目标位置
            target_quat: 目标姿态
            reg_target: 正则化目标位置
            position_only: 是否只考虑位置
        
        Returns:
            residual: 残差向量
        """
        # 确保q是一维数组（minimize.least_squares可能传递(n,1)形状的数组）
        q = np.asarray(q).flatten()
        
        # 计算正运动学
        current_pos, current_quat = self.forward_kinematics(q)
        
        # 位置残差
        res_pos = current_pos - target_pos
        
        # 姿态残差（使用四元数差分）
        if not position_only:
            res_quat = np.empty(3)
            mujoco.mju_subQuat(res_quat, target_quat, current_quat)
            res_quat *= self.ik_radius
        else:
            res_quat = np.zeros(3)
        
        # 正则化残差
        res_reg = self.ik_regularization * (q - reg_target)
        
        return np.expand_dims(np.hstack((res_pos, res_quat, res_reg)),axis=1) # explicitly indicate the shape of column vector to avoid mismatching in broadcasting
    
    def _ik_jacobian(
        self,
        q: np.ndarray,
        residual: np.ndarray,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        position_only: bool = False
    ) -> np.ndarray:
        """
        IK残差的雅可比矩阵
        
        Args:
            q: 当前关节位置
            residual: 当前残差（未使用，但接口需要）
            target_pos: 目标位置
            target_quat: 目标姿态
            position_only: 是否只考虑位置
        
        Returns:
            jacobian: 雅可比矩阵
        """
        # 确保q是一维数组（minimize.least_squares可能传递(n,1)形状的数组）
        q = np.asarray(q).flatten()
        
        # 计算几何雅可比
        jac_pos, jac_rot = self.compute_jacobian(q)
        
        # 如果只考虑位置，姿态雅可比为零
        if position_only:
            jac_quat = np.zeros((3, self.dof))
        else:
            # 获取当前末端姿态
            _, current_quat = self.forward_kinematics(q)
            
            # 计算子四元数的雅可比 Deffector
            Deffector = np.empty((3, 3))
            mujoco.mjd_subQuat(target_quat, current_quat, None, Deffector)
            
            # 旋转到目标坐标系
            target_mat = T.quat_2_mat(target_quat)
            mat = self.ik_radius * Deffector.T @ target_mat.T
            jac_quat = mat @ jac_rot
        
        # 正则化雅可比
        jac_reg = self.ik_regularization * np.eye(self.dof)
        
        return np.vstack((jac_pos, jac_quat, jac_reg))
    
    def compute_control(
        self,
        target: np.ndarray,
        current_state: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """
        计算笛卡尔IK控制输出
        
        Args:
            target: 目标位姿 (7,) [x, y, z, qw, qx, qy, qz]
                   或 (3,) [x, y, z] (仅位置)
            current_state: 当前状态
        
        Returns:
            q_target (dof,): 目标关节位置（适配position actuators）
        """
        # 解析目标
        if len(target) == 3:
            # 仅位置
            target_pos = target
            _, current_quat = self.forward_kinematics()
            target_quat = current_quat  # 保持当前姿态
            position_only = True
        elif len(target) == 7:
            # 位置 + 姿态（四元数）
            target_pos = target[:3]
            target_quat = target[3:]
            position_only = False
        else:
            raise ValueError(f"Invalid target shape: {target.shape}. Expected (3,) or (7,)")
        
        # 求解IK得到目标关节位置
        q_target = self.solve_ik(target_pos, target_quat, position_only=position_only)
        
        # 使用关节位置控制器计算输出（根据use_target_as_ctrl决定返回位置还是力矩）
        return self.joint_controller.compute_control(q_target, current_state)
    
    def set_gains(self, kp: Optional[np.ndarray] = None, kd: Optional[np.ndarray] = None):
        """设置关节控制器增益"""
        self.joint_controller.set_gains(kp=kp, kd=kd)
    
    def reset(self):
        """重置控制器"""
        super().reset()
        self.joint_controller.reset()
        # 更新初始位置，避开非机器人关节的干扰
        self.q_initial = self.get_arm_qpos()
    
    def __repr__(self) -> str:
        return (
            f"CartesianIKController("
            f"dof={self.dof}, "
            f"freq={self.control_freq}Hz, "
            f"reg={self.ik_regularization})"
        )

