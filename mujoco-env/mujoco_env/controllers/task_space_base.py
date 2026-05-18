"""
任务空间控制器基类

提供任务空间控制所需的高级功能，包括：
- 正运动学
- 逆运动学
- 雅可比矩阵计算
- 质量矩阵计算
- 笛卡尔空间到关节空间的映射

作者: Liu Gang
日期: 2025-12-21
"""

import numpy as np
import mujoco
from typing import Optional, Dict, Tuple
from mujoco_env.mujoco_env.controllers.base_controller import BaseController
from mujoco_env.mujoco_env.utils import transform as T


class TaskSpaceController(BaseController):
    """
    任务空间控制器基类
    
    提供笛卡尔空间控制所需的通用功能。
    子类可以继承此类来实现具体的任务空间控制策略。
    
    Attributes:
        ee_site_name: 末端执行器site名称
        ee_site_id: 末端执行器site ID
    """
    
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        dof: int,
        control_freq: float = 20.0,
        ee_site_name: str = "pinch",
        **kwargs
    ):
        """
        初始化任务空间控制器基类
        
        Args:
            model: MuJoCo模型
            data: MuJoCo数据
            dof: 自由度数量
            control_freq: 控制频率 (Hz)
            ee_site_name: 末端执行器site名称，默认"pinch"
            **kwargs: 额外参数
        """
        super().__init__(model, data, dof, control_freq, **kwargs)
        
        # 末端执行器site
        self.ee_site_name = ee_site_name
        try:
            self.ee_site_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name
            )
        except:
            # 如果site不存在，尝试使用body
            try:
                self.ee_site_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, ee_site_name
                )
                self._use_body = True
            except:
                raise ValueError(f"Cannot find site or body named '{ee_site_name}'")
        else:
            self._use_body = False
        
        # 获取机械臂关节的 qpos 和 dof 起始地址
        # 默认使用第一个非 free 关节的地址（适用于大多数机器人配置）
        self.arm_qpos_adr = 0
        self.arm_dof_adr = 0
        for i in range(model.njnt):
            if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE:
                self.arm_qpos_adr = model.jnt_qposadr[i]
                self.arm_dof_adr = model.jnt_dofadr[i]
                break

    def get_arm_qpos(self) -> np.ndarray:
        """获取机械臂当前的关节位置"""
        return self.data.qpos[self.arm_qpos_adr : self.arm_qpos_adr + self.dof].copy()

    def get_arm_qvel(self) -> np.ndarray:
        """获取机械臂当前的关节速度"""
        return self.data.qvel[self.arm_dof_adr : self.arm_dof_adr + self.dof].copy()
    
    def forward_kinematics(
        self,
        q: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算正运动学
        
        Args:
            q: 关节位置 (dof,)，None表示使用当前位置
        
        Returns:
            tuple: (position, quaternion)
                - position (3,): 末端位置 [x, y, z]
                - quaternion (4,): 末端姿态 [w, x, y, z]
        """
        if q is not None:
            # 保存当前状态
            q_save = self.get_arm_qpos()
            qvel_save = self.get_arm_qvel()
            
            # 设置新的关节位置
            self.data.qpos[self.arm_qpos_adr : self.arm_qpos_adr + self.dof] = q
            self.data.qvel[self.arm_dof_adr : self.arm_dof_adr + self.dof] = 0
            
            # 计算正运动学
            mujoco.mj_forward(self.model, self.data)
        
        # 获取末端位置和姿态
        if self._use_body:
            pos = self.data.body(self.ee_site_name).xpos.copy()
            mat = self.data.body(self.ee_site_name).xmat.reshape(3, 3).copy()
        else:
            pos = self.data.site(self.ee_site_name).xpos.copy()
            mat = self.data.site(self.ee_site_name).xmat.reshape(3, 3).copy()
        
        quat = T.mat_2_quat(mat)
        
        if q is not None:
            # 恢复原来的状态
            self.data.qpos[self.arm_qpos_adr : self.arm_qpos_adr + self.dof] = q_save
            self.data.qvel[self.arm_dof_adr : self.arm_dof_adr + self.dof] = qvel_save
            mujoco.mj_forward(self.model, self.data)
        
        return pos, quat
    
    def compute_jacobian(
        self,
        q: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算雅可比矩阵
        
        Args:
            q: 关节位置 (dof,)，None表示使用当前位置
        
        Returns:
            tuple: (jac_pos, jac_rot)
                - jac_pos (3, dof): 位置雅可比
                - jac_rot (3, dof): 旋转雅可比
        """
        if q is not None:
            # 保存当前状态
            q_save = self.get_arm_qpos()
            self.data.qpos[self.arm_qpos_adr : self.arm_qpos_adr + self.dof] = q
            mujoco.mj_forward(self.model, self.data)
        
        # 分配雅可比矩阵
        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))
        
        # 计算雅可比
        if self._use_body:
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, self.ee_site_name
            )
            mujoco.mj_jacBody(self.model, self.data, jac_pos, jac_rot, body_id)
        else:
            mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_rot, self.ee_site_id)
        
        # 只返回机械臂对应的 dof 列
        jac_pos = jac_pos[:, self.arm_dof_adr : self.arm_dof_adr + self.dof]
        jac_rot = jac_rot[:, self.arm_dof_adr : self.arm_dof_adr + self.dof]
        
        if q is not None:
            # 恢复原来的状态
            self.data.qpos[self.arm_qpos_adr : self.arm_qpos_adr + self.dof] = q_save
            mujoco.mj_forward(self.model, self.data)
        
        return jac_pos, jac_rot
    
    def compute_mass_matrix(self) -> np.ndarray:
        """
        计算质量矩阵
        
        Returns:
            M (dof, dof): 质量矩阵
        """
        M = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M, self.data.qM)
        return M[:self.dof, :self.dof]
    
    def compute_cartesian_impedance_torque(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        kp_pos: np.ndarray = np.array([100.0, 100.0, 100.0]),
        kp_ori: np.ndarray = np.array([200.0, 200.0, 200.0]),
        kd_pos: np.ndarray = np.array([200.0, 800.0, 800.0]),
        kd_ori: np.ndarray = np.array([400.0, 400.0, 400.0])
    ) -> np.ndarray:
        """
        计算笛卡尔空间阻抗控制力矩
        
        Args:
            target_pos: 目标位置 (3,)
            target_quat: 目标姿态四元数 (4,) [w, x, y, z]
            kp_pos: 位置刚度 (3,)
            kp_ori: 姿态刚度 (3,)
            kd_pos: 位置阻尼 (3,)
            kd_ori: 姿态阻尼 (3,)
        
        Returns:
            tau (dof,): 关节力矩
        """
        # 获取当前状态
        current_state = self.get_state()
        q_cur = current_state["qpos"]
        qd_cur = current_state["qvel"]
        
        # 计算正运动学
        current_pos, current_quat = self.forward_kinematics()
        current_mat = T.quat_2_mat(current_quat)
        target_mat = T.quat_2_mat(target_quat)
        
        # 计算雅可比
        J_pos, J_rot = self.compute_jacobian()
        J = np.vstack([J_pos, J_rot])  # (6, dof)
        
        # 计算质量矩阵
        M = self.compute_mass_matrix()
        
        # 笛卡尔空间质量矩阵
        try:
            J_inv = np.linalg.pinv(J)
            Md = J_inv.T @ M @ J_inv
        except:
            Md = np.eye(6)
        
        # 位置误差
        pos_error = target_pos - current_pos
        
        # 姿态误差（orientation error）
        ori_error = orientation_error(target_mat, current_mat)
        
        # 组合误差
        x_error = np.concatenate([pos_error, ori_error])
        
        # 笛卡尔速度
        v_cart = J @ qd_cur
        
        # 组合增益
        Kc = np.concatenate([kp_pos, kp_ori])
        Bc = np.concatenate([kd_pos, kd_ori])
        
        # 笛卡尔空间控制力
        F_cart = Kc * x_error - Bc * v_cart
        
        # 映射到关节空间
        tau = J.T @ F_cart
        
        # 添加重力补偿
        tau += self.data.qfrc_bias[:self.dof]
        
        return tau
    
    def compute_control(
        self,
        target: np.ndarray,
        current_state: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """
        默认实现：需要子类override
        
        Args:
            target: 目标（位姿或其他）
            current_state: 当前状态
        
        Returns:
            control: 控制输出
        """
        raise NotImplementedError("Subclasses must implement compute_control")


def orientation_error(desired: np.ndarray, current: np.ndarray) -> np.ndarray:
    """
    计算姿态误差
    
    使用3D叉积方法计算两个旋转矩阵之间的姿态误差。
    
    Args:
        desired: 目标姿态矩阵 (3, 3)
        current: 当前姿态矩阵 (3, 3)
    
    Returns:
        error (3,): 姿态误差向量（欧拉角误差）
    """
    rc1 = current[:, 0]
    rc2 = current[:, 1]
    rc3 = current[:, 2]
    rd1 = desired[:, 0]
    rd2 = desired[:, 1]
    rd3 = desired[:, 2]
    
    w1, w2, w3 = 0.5, 0.5, 0.5
    error = w1 * np.cross(rc1, rd1) + w2 * np.cross(rc2, rd2) + w3 * np.cross(rc3, rd3)
    
    return error

