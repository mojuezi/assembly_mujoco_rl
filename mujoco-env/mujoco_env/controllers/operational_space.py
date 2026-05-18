"""
操作空间控制器 (Operational Space Control, OSC)

也称为任务空间控制器（Task Space Control）。
通过控制末端执行器在笛卡尔空间（位置和姿态）实现精确控制，
同时在零空间（nullspace）中执行次要任务（如保持关节配置）。

参考:
- Khatib, O. (1987). "A unified approach for motion and force control of robot manipulators"

作者: 原作者 (移植自impl/controllers/opspace.py)
日期: 2025-12-20
重构: Liu Gang
"""

from typing import Optional, Tuple, Union, Dict
import numpy as np
import mujoco
from dm_robotics.transformations import transformations as tr

from mujoco_env.mujoco_env.controllers.task_space_base import TaskSpaceController


# ============================================================================
# 辅助函数：PD控制
# ============================================================================

def pd_control(
    x: np.ndarray,
    x_des: np.ndarray,
    dx: np.ndarray,
    kp_kv: np.ndarray,
    ddx_max: float = 0.0,
) -> np.ndarray:
    """
    位置PD控制
    
    Args:
        x: 当前位置
        x_des: 期望位置
        dx: 当前速度
        kp_kv: PD增益 [kp, kv]
        ddx_max: 最大加速度限制
        
    Returns:
        控制加速度
    """
    # 计算误差
    x_err = x - x_des
    dx_err = dx

    # 应用增益
    x_err *= -kp_kv[:, 0]
    dx_err *= -kp_kv[:, 1]

    # 限制最大误差
    if ddx_max > 0.0:
        x_err_sq_norm = np.sum(x_err**2)
        ddx_max_sq = ddx_max**2
        if x_err_sq_norm > ddx_max_sq:
            x_err *= ddx_max / np.sqrt(x_err_sq_norm)

    return x_err + dx_err


def pd_control_orientation(
    quat: np.ndarray,
    quat_des: np.ndarray,
    w: np.ndarray,
    kp_kv: np.ndarray,
    dw_max: float = 0.0,
) -> np.ndarray:
    """
    姿态PD控制
    
    Args:
        quat: 当前四元数
        quat_des: 期望四元数
        w: 当前角速度
        kp_kv: PD增益 [kp, kv]
        dw_max: 最大角加速度限制
        
    Returns:
        控制角加速度
    """
    # 计算误差
    quat_err = tr.quat_diff_active(source_quat=quat_des, target_quat=quat)
    ori_err = tr.quat_to_axisangle(quat_err)
    w_err = w

    # 应用增益
    ori_err *= -kp_kv[:, 0]
    w_err *= -kp_kv[:, 1]

    # 限制最大误差
    if dw_max > 0.0:
        ori_err_sq_norm = np.sum(ori_err**2)
        dw_max_sq = dw_max**2
        if ori_err_sq_norm > dw_max_sq:
            ori_err *= dw_max / np.sqrt(ori_err_sq_norm)

    return ori_err + w_err


# ============================================================================
# 原始操作空间控制函数（兼容性）
# ============================================================================

def opspace(
    model,
    data,
    site_id,
    dof_ids: np.ndarray,
    qpos_ids: Optional[np.ndarray] = None,
    pos: Optional[np.ndarray] = None,
    ori: Optional[np.ndarray] = None,
    joint: Optional[np.ndarray] = None,
    pos_gains: Union[Tuple[float, float, float], np.ndarray] = (200.0, 200.0, 200.0),
    ori_gains: Union[Tuple[float, float, float], np.ndarray] = (200.0, 200.0, 200.0),
    damping_ratio: float = 1.0,
    nullspace_stiffness: float = 0.5,
    max_pos_acceleration: Optional[float] = None,
    max_ori_acceleration: Optional[float] = None,
    gravity_comp: bool = True,
) -> np.ndarray:
    """
    操作空间控制（原始函数，向后兼容）
    
    计算关节力矩以控制末端执行器的位置和姿态，
    同时在零空间中保持期望的关节配置。
    
    Args:
        model: MuJoCo模型
        data: MuJoCo数据
        site_id: 末端执行器site的ID
        dof_ids: 关节自由度ID数组 (用于 qvel, Jacobian, MassMatrix)
        qpos_ids: 关节位置ID数组 (用于 qpos)，如果为None则默认使用 dof_ids
        pos: 期望位置 (3,) [可选]
        ori: 期望姿态 (四元数或旋转矩阵) [可选]
        joint: 期望关节配置 (零空间) [可选]
        ...
    """
    # 确定 qpos 索引
    if qpos_ids is None:
        qpos_ids = dof_ids

    # 确定期望状态
    if pos is None:
        x_des = data.site_xpos[site_id]
    else:
        x_des = np.asarray(pos)
        
    if ori is None:
        xmat = data.site_xmat[site_id].reshape((3, 3))
        quat_des = tr.mat_to_quat(xmat.reshape((3, 3)))
    else:
        ori = np.asarray(ori)
        if ori.shape == (3, 3):
            quat_des = tr.mat_to_quat(ori)
        else:
            quat_des = ori
            
    if joint is None:
        q_des = data.qpos[qpos_ids]
    else:
        q_des = np.asarray(joint)

    # 计算PD增益
    kp = np.asarray(pos_gains)
    kd = damping_ratio * 2 * np.sqrt(kp)
    kp_kv_pos = np.stack([kp, kd], axis=-1)

    kp = np.asarray(ori_gains)
    kd = damping_ratio * 2 * np.sqrt(kp)
    kp_kv_ori = np.stack([kp, kd], axis=-1)

    kp_joint = np.full((len(dof_ids),), nullspace_stiffness)
    kd_joint = damping_ratio * 2 * np.sqrt(kp_joint)
    kp_kv_joint = np.stack([kp_joint, kd_joint], axis=-1)

    ddx_max = max_pos_acceleration if max_pos_acceleration is not None else 0.0
    dw_max = max_ori_acceleration if max_ori_acceleration is not None else 0.0

    # 获取当前状态
    q = data.qpos[qpos_ids]
    dq = data.qvel[dof_ids]

    # 计算末端执行器的雅可比矩阵（世界坐标系）
    J_v = np.zeros((3, model.nv), dtype=np.float64)
    J_w = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(
        model,
        data,
        J_v,
        J_w,
        site_id,
    )
    J_v = J_v[:, dof_ids]
    J_w = J_w[:, dof_ids]
    J = np.concatenate([J_v, J_w], axis=0)

    # 计算位置PD控制
    x = data.site_xpos[site_id]
    dx = J_v @ dq
    ddx = pd_control(
        x=x,
        x_des=x_des,
        dx=dx,
        kp_kv=kp_kv_pos,
        ddx_max=ddx_max,
    )

    # 计算姿态PD控制
    quat = tr.mat_to_quat(data.site_xmat[site_id].reshape((3, 3)))
    if quat @ quat_des < 0.0:
        quat *= -1.0
    w = J_w @ dq
    dw = pd_control_orientation(
        quat=quat,
        quat_des=quat_des,
        w=w,
        kp_kv=kp_kv_ori,
        dw_max=dw_max,
    )

    # 计算关节空间惯性矩阵
    M = np.zeros((model.nv, model.nv), dtype=np.float64)
    mujoco.mj_fullM(model, M, data.qM)
    M = M[dof_ids, :][:, dof_ids]

    # 计算任务空间惯性矩阵
    M_inv = np.linalg.inv(M)
    Mx_inv = J @ M_inv @ J.T
    if abs(np.linalg.det(Mx_inv)) >= 1e-2:
        Mx = np.linalg.inv(Mx_inv)
    else:
        Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)

    # 计算广义力
    ddx_dw = np.concatenate([ddx, dw], axis=0)
    tau = J.T @ Mx @ ddx_dw

    # 在零空间添加关节任务
    ddq = pd_control(
        x=q,
        x_des=q_des,
        dx=dq,
        kp_kv=kp_kv_joint,
        ddx_max=0.0,
    )
    Jnull = M_inv @ J.T @ Mx
    tau += (np.eye(len(q)) - J.T @ Jnull.T) @ ddq

    # 重力补偿
    if gravity_comp:
        tau += data.qfrc_bias[dof_ids]
        
    return tau


# ============================================================================
# 控制器类（面向对象封装）
# ============================================================================

class OperationalSpaceController(TaskSpaceController):
    """
    操作空间控制器（面向对象版本）
    ...
    """
    
    def __init__(
        self,
        model,
        data,
        dof: int,
        control_freq: int = 20,
        site_name: str = "pinch",
        pos_gains: Tuple[float, float, float] = (200.0, 200.0, 200.0),
        ori_gains: Tuple[float, float, float] = (200.0, 200.0, 200.0),
        damping_ratio: float = 1.0,
        nullspace_stiffness: float = 0.5,
        max_pos_acceleration: Optional[float] = None,
        max_ori_acceleration: Optional[float] = None,
        gravity_comp: bool = True,
        target_joint: Optional[np.ndarray] = None,
    ):
        """初始化操作空间控制器"""
        super().__init__(model, data, dof, control_freq, ee_site_name=site_name)
        
        # 参数
        self.site_name = site_name
        self.pos_gains = np.asarray(pos_gains)
        self.ori_gains = np.asarray(ori_gains)
        self.damping_ratio = damping_ratio
        self.nullspace_stiffness = nullspace_stiffness
        self.max_pos_acceleration = max_pos_acceleration
        self.max_ori_acceleration = max_ori_acceleration
        self.gravity_comp = gravity_comp
        
        # 缓存site ID
        self.site_id = model.site(site_name).id
        
        # 缓存 DOF 和 QPOS IDs (修复索引偏移)
        self.qpos_ids = np.arange(self.arm_qpos_adr, self.arm_qpos_adr + dof)
        self.dof_ids = np.arange(self.arm_dof_adr, self.arm_dof_adr + dof)
        
        # 零空间目标关节配置
        self.target_joint = target_joint.copy() if target_joint is not None else None
    
    def compute_control(
        self,
        target: Optional[np.ndarray] = None,
        current_state: Optional[Dict[str, np.ndarray]] = None,
        target_pos: Optional[np.ndarray] = None,
        target_ori: Optional[np.ndarray] = None,
        target_joint: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """计算控制力矩"""
        # 解析统一目标向量
        if target is not None:
            if len(target) >= 7:
                target_pos = target[:3]
                target_ori = target[3:7]
            elif len(target) == 3:
                target_pos = target
        
        # 如果未提供target_joint，使用初始化时设置的默认值
        if target_joint is None:
            target_joint = self.target_joint
        
        tau = opspace(
            model=self.model,
            data=self.data,
            site_id=self.site_id,
            dof_ids=self.dof_ids,
            qpos_ids=self.qpos_ids,
            pos=target_pos,
            ori=target_ori,
            joint=target_joint,
            pos_gains=self.pos_gains,
            ori_gains=self.ori_gains,
            damping_ratio=self.damping_ratio,
            nullspace_stiffness=self.nullspace_stiffness,
            max_pos_acceleration=self.max_pos_acceleration,
            max_ori_acceleration=self.max_ori_acceleration,
            gravity_comp=self.gravity_comp,
        )
        
        return tau
    
    def reset(self):
        """重置控制器"""
        # OSC控制器无状态，无需重置
        pass


# ============================================================================
# 别名（便于导入）
# ============================================================================

OSCController = OperationalSpaceController  # 简短别名
TaskSpaceController = OperationalSpaceController  # 别名

