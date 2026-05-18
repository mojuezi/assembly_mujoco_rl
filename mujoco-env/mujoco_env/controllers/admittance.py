"""
导纳控制器 (Admittance Controller)

导纳控制是一种力控制策略，通过测量外部力/力矩来调整末端执行器的位置/姿态。
适用于需要力反馈的接触任务，如装配、打磨、插拔等。

控制原理:
根据力误差调整位置: M·ẍ + D·ẋ + K·x = F_external - F_desired

其中:
- M: 虚拟质量
- D: 虚拟阻尼
- K: 虚拟刚度
- F_external: 外部力/力矩
- F_desired: 期望力/力矩

重构说明:
- 继承 JointPositionController，设置 use_target_as_ctrl=True 直接输出关节位置
- 组合 CartesianIKController 复用IK求解功能，避免代码重复
- 导纳控制逻辑：力误差 → 笛卡尔位姿调整 → IK求解 → 关节位置输出

作者: Liu Gang
日期: 2025-12-21
重构: 2026-01-20
"""

import numpy as np
import mujoco
from typing import Optional, Dict
from mujoco_env.mujoco_env.controllers.joint_position import JointPositionController
from mujoco_env.mujoco_env.controllers.cartesian_ik import CartesianIKController
from mujoco_env.mujoco_env.utils import transform as T
from dm_robotics.transformations import transformations as tr
import pdb
import time

class AdmittanceController(JointPositionController):
    """
    导纳控制器
    
    继承 JointPositionController，添加笛卡尔空间的力反馈导纳控制。
    设置 use_target_as_ctrl=True，直接输出关节位置（用于position actuator）。
    根据外部力/力矩调整目标位姿，通过IK求解转换为关节位置。
    
    使用场景:
    - 接触任务（装配、插拔）
    - 力控制打磨/抛光
    - 拖动示教（力引导）
    - 柔顺操作
    
    Attributes:
        ik_controller: 笛卡尔IK控制器（组合，用于复用IK求解功能）
        M: 虚拟质量 (6,)
        D: 虚拟阻尼 (6,)
        K: 虚拟刚度 (6,)
        selection_vector: 选择向量 (6,)，1表示启用导纳控制，0表示位置控制
        force_offset: 力传感器偏置 (6,)
        desired_force: 期望力/力矩 (6,)
    """
    
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        dof: int,
        control_freq: float = 20.0,
        ee_site_name: str = "pinch",
        mass: Optional[np.ndarray] = None,
        damping: Optional[np.ndarray] = None,
        stiffness: Optional[np.ndarray] = None,
        selection_vector: Optional[np.ndarray] = None,
        ik_regularization: float = 0.01,
        ik_radius: float = 0.2,
        joint_limits: Optional[tuple] = None,
        mode=None, 
        robot_interface=None, 
        **kwargs
    ):
        """
        初始化导纳控制器
        
        Args:
            model: MuJoCo模型
            data: MuJoCo数据
            dof: 自由度数量
            control_freq: 控制频率 (Hz)
            ee_site_name: 末端执行器site名称
            mass: 虚拟质量 (6,)，默认[20, 20, 20, 20, 20, 20]
            damping: 虚拟阻尼 (6,)，默认[200, 200, 200, 200, 200, 200]
            stiffness: 虚拟刚度 (6,)，默认[0, 0, 0, 0, 0, 0]
            selection_vector: 选择向量 (6,)，默认全0（纯位置控制）
            ik_regularization: IK正则化系数，默认0.01
            ik_radius: 姿态误差缩放因子，默认0.2
            joint_limits: 关节限位 (lower, upper)，每个都是(dof,)数组
            **kwargs: 额外参数
        """
        self.mode = mode
        self.robot_interface = robot_interface
       
        # 初始化关节位置控制器（父类），设置 use_target_as_ctrl=True 直接输出位置
        if self.mode != "for_real": 
            super().__init__(
                model, data, dof, control_freq,
                kp = None,  
                kd = None,  
                use_target_as_ctrl = True, 
                **kwargs
            )
            
            # 组合 CartesianIKController 来复用IK求解功能
            self.ik_controller = CartesianIKController(
                model = model,
                data = data,
                dof = dof,
                control_freq = control_freq,
                ee_site_name = ee_site_name,
                ik_regularization = ik_regularization,
                ik_radius = ik_radius,
                joint_limits = joint_limits
            )
        
        self.ee_site_name = ee_site_name
        
        # 导纳参数
        self.M = mass if mass is not None else 10.0 * np.ones(6)
        # self.M[0], self.M[0] = 3.0, 3.0
        self.D = damping if damping is not None else 100.0 * np.ones(6)
        self.D[0], self.D[1] = 400.0, 350.0
        # self.K = stiffness if stiffness is not None else 0.0 * np.ones(6)
        self.K = stiffness if stiffness is not None else 90.0 * np.ones(6)
        self.K[0], self.K[1] = 4000.0, 3000.0

        # self.set_admittance_params(mass=np.array([2.0, 2.0, 1.0, 1.0, 1.0, 1.0]), 
        #                                 damping=np.array([70.0, 60.0, 50.0, 5.0, 5.0, 5.0]), 
        #                                 stiffness=np.array([400.0, 300.0, 150.0, 9.0, 9.0, 9.0]))
        
        # 选择向量：1表示启用导纳控制，0表示位置控制
        self.selection_vector = (
            selection_vector if selection_vector is not None 
            else np.zeros(6)
        )
        
        # 力传感器偏置和期望力
        self.force_offset = np.zeros(6)
        self.desired_force = np.zeros(6)
        
        # 导纳状态
        self.adm_pos = np.zeros(6)  # [x, y, z, rx, ry, rz] (欧拉角)
        self.adm_vel = np.zeros(6)
        self.adm_acc = np.zeros(6)
        
        # 初始化导纳位置为当前位置
        if self.mode == "for_real": 
            current_pose = self.robot_interface.get_tcp_pose()
            self.adm_pos[:3] = current_pose[:3]
            # self.adm_pos[3:] = T.quat_2_euler(current_pose[3:])
            self.adm_pos[3:] = tr.quat_to_euler(current_pose[3:], 'ZYX')
        else: 
            current_pos, current_quat = self.ik_controller.forward_kinematics()
            self.adm_pos[:3] = current_pos
            # self.adm_pos[3:] = T.quat_2_euler(current_quat)
            self.adm_pos[3:] = tr.quat_to_euler(current_quat, 'ZYX')
        
        self.cycle_time = 1.0 / control_freq
    
    def solve_ik(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        q_init: Optional[np.ndarray] = None,
        position_only: bool = False
    ) -> np.ndarray:
        """
        求解逆运动学（委托给ik_controller）
        
        Args:
            target_pos: 目标位置 (3,)
            target_quat: 目标姿态四元数 (4,) [w, x, y, z]
            q_init: 初始猜测关节位置 (dof,)，None表示使用当前位置
            position_only: 是否只约束位置（忽略姿态），默认False
        
        Returns:
            q_solution (dof,): 求解的关节位置
        """
        return self.ik_controller.solve_ik(target_pos, target_quat, q_init, position_only)
    
    def forward_kinematics(
        self,
        q: Optional[np.ndarray] = None
    ):
        """
        计算正运动学（委托给ik_controller）
        
        Args:
            q: 关节位置 (dof,)，None表示使用当前位置
        
        Returns:
            tuple: (position, quaternion)
        """
        return self.ik_controller.forward_kinematics(q)
    
    def update_admittance(
        self,
        ref_pos: np.ndarray,
        ref_vel: np.ndarray,
        ref_acc: np.ndarray,
        external_force: np.ndarray
    ) -> np.ndarray:
        """
        更新导纳控制状态
        
        根据外部力和参考轨迹，计算新的导纳位置。
        
        Args:
            ref_pos: 参考位姿 (7,) [x, y, z, qw, qx, qy, qz]
            ref_vel: 参考速度 (6,) [vx, vy, vz, wx, wy, wz]
            ref_acc: 参考加速度 (6,)
            external_force: 外部力/力矩 (6,) [fx, fy, fz, tx, ty, tz]
        
        Returns:
            adm_pose (7,): 新的导纳位姿 [x, y, z, qw, qx, qy, qz]
        """
        # 转换参考位姿到欧拉角
        ref_pos_xyz = ref_pos[:3]
        # ref_euler = T.quat_2_euler(ref_pos[3:])
        ref_euler = tr.quat_to_euler(ref_pos[3:], 'ZYX')
        ref_pos_euler = np.concatenate([ref_pos_xyz, ref_euler])
        
        # 计算误差
        pos_error = ref_pos_euler - self.adm_pos
        vel_error = ref_vel - self.adm_vel
        
        # 力误差（去除偏置）
        force_error = self.desired_force - (external_force - self.force_offset)
        
        # 导纳控制更新
        delta_pos = np.zeros(6)
        
        for i in range(6):
            if self.selection_vector[i] == 1:
                # 启用导纳控制
                # M·ẍ + D·ẋ + K·x = F_ext - F_des
                self.adm_acc[i] = ref_acc[i] + (
                    -force_error[i] + self.D[i] * vel_error[i] + self.K[i] * pos_error[i]
                ) / self.M[i]
                
                # 速度积分
                self.adm_vel[i] = ref_vel[i] - vel_error[i] + self.adm_acc[i] * self.cycle_time
                
                # 位置积分
                delta_pos[i] = self.adm_vel[i] * self.cycle_time
            else:
                # 纯位置控制
                self.adm_acc[i] = ref_acc[i]
                self.adm_vel[i] = ref_vel[i]
                delta_pos[i] = 0.0
        
        # 更新位置（笛卡尔分量直接加）
        # A limit for delta
        # delta_pos = np.clip(delta_pos, -0.1, 0.1) # For real machine 
        self.adm_pos[:3] += delta_pos[:3]
        
        # 更新姿态（欧拉角增量左乘）
        # delta_mat = T.euler_2_mat(delta_pos[3:])
        # current_mat = T.euler_2_mat(self.adm_pos[3:])
        delta_mat = tr.euler_to_rmat(delta_pos[3:], 'ZYX')
        current_mat = tr.euler_to_rmat(self.adm_pos[3:], 'ZYX')
        new_mat = delta_mat @ current_mat
        self.adm_pos[3:] = tr.rmat_to_euler(new_mat, 'ZYX')
        
        # 转换回四元数
        # adm_quat = T.euler_2_quat(self.adm_pos[3:])
        adm_quat = tr.euler_to_quat(self.adm_pos[3:], 'ZYX')
        return np.concatenate([self.adm_pos[:3], adm_quat])
    
    def compute_control(
        self,
        target: np.ndarray,
        current_state: Optional[Dict[str, np.ndarray]] = None,
        external_force: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        计算导纳控制输出（关节位置）
        
        Args:
            target: 目标位姿 (7,) [x, y, z, qw, qx, qy, qz]
                   或 (3,) [x, y, z] (仅位置)
            current_state: 当前状态（可选，未使用）
            external_force: 外部力/力矩 (6,)，如果为None则使用零力
        
        Returns:
            q_target (dof,): 目标关节位置（用于position actuator）
        """
        # print("adm_time1: ", time.time())
        # 如果没有提供外部力，使用零力
        if external_force is None:
            external_force = np.zeros(6)
        # 解析目标位姿
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
        
        # 参考轨迹（简化：假设参考速度和加速度为0）
        ref_vel = np.zeros(6)
        ref_acc = np.zeros(6)
        
        # 如果选择向量全为0，直接使用位置控制（不启用导纳）
        if np.all(self.selection_vector == 0):
            # 直接IK求解，然后通过父类返回（因为use_target_as_ctrl=True，会直接返回target）
            if self.mode == "for_real": 
                return np.concatenate([target_pos, target_quat])
            q_target = self.solve_ik(target_pos, target_quat, position_only = position_only)
            return super().compute_control(q_target, current_state)
       
        # print("adm_time2: ", time.time())
        # 更新导纳状态（根据力误差调整位姿）
        adm_pose = self.update_admittance(
            np.concatenate([target_pos, target_quat]),
            ref_vel,
            ref_acc,
            external_force
        )

        # print("adm_time3: ", time.time())

        if self.mode == "for_real": 
            return adm_pose
        
        # print("adm_time4: ", time.time())
        # IK求解：将导纳位姿转换为关节位置
        adm_pos = adm_pose[:3]
        adm_quat = adm_pose[3:]
        q_target = self.solve_ik(adm_pos, adm_quat, position_only = position_only)
        
        # 通过父类返回（因为use_target_as_ctrl=True，会直接返回q_target作为位置）
        return super().compute_control(q_target, current_state)
    
    def set_admittance_params(
        self,
        mass: Optional[np.ndarray] = None,
        damping: Optional[np.ndarray] = None,
        stiffness: Optional[np.ndarray] = None
    ):
        """
        设置导纳参数
        
        Args:
            mass: 虚拟质量 (6,)
            damping: 虚拟阻尼 (6,)
            stiffness: 虚拟刚度 (6,)
        """
        if mass is not None:
            self.M = mass
        if damping is not None:
            self.D = damping
        if stiffness is not None:
            self.K = stiffness
    
    def set_selection_vector(self, selection: np.ndarray):
        """
        设置选择向量
        
        Args:
            selection (6,): 1表示启用导纳控制，0表示位置控制
                           例如: [1, 1, 0, 0, 0, 0] 表示只在x,y方向启用力控制
        """
        assert len(selection) == 6, "Selection vector must be length 6"
        self.selection_vector = selection
    
    def set_desired_force(self, force: np.ndarray):
        """
        设置期望力/力矩
        
        Args:
            force (6,): [fx, fy, fz, tx, ty, tz]
        """
        assert len(force) == 6, "Force vector must be length 6"
        self.desired_force = force
    
    def calibrate_force_sensor(self, force_reading: np.ndarray):
        """
        标定力传感器（设置偏置）
        
        Args:
            force_reading (6,): 当前力/力矩读数
        """
        self.force_offset = force_reading.copy()
    
    def reset(self):
        """重置导纳控制器"""
        super().reset()

        if self.mode != "for_real": 
            self.ik_controller.reset()
        
        # 重置导纳状态
        self.adm_vel = np.zeros(6)
        self.adm_acc = np.zeros(6)
        
        # 重置导纳位置为当前位置
        if self.mode == "for_real": 
            current_pose = self.robot_interface.get_tcp_pose()
            self.adm_pos[:3] = current_pose[:3]
            # self.adm_pos[3:] = T.quat_2_euler(current_pose[3:])
            self.adm_pos[3:] = tr.quat_to_euler(current_pose[3:], 'ZYX')
        else: 
            current_pos, current_quat = self.ik_controller.forward_kinematics()
            self.adm_pos[:3] = current_pos
            # self.adm_pos[3:] = T.quat_2_euler(current_quat)
            self.adm_pos[3:] = tr.quat_to_euler(current_quat, 'ZYX')
    
    def __repr__(self) -> str:
        active_dims = np.sum(self.selection_vector)
        return (
            f"AdmittanceController("
            f"dof={self.dof}, "
            f"freq={self.control_freq}Hz, "
            f"active_dims={int(active_dims)})"
        )

