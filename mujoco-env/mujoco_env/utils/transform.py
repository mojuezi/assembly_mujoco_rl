"""
3D坐标变换工具

提供各种3D旋转表示之间的转换：
- 欧拉角 (Euler angles)
- 四元数 (Quaternion)
- 旋转矩阵 (Rotation matrix)
- 旋转向量 (Rotation vector)

以及齐次变换矩阵、力变换等功能。

作者: Liu Gang (原作者)
日期: 2025-12-20
修订: 添加详细注释
"""

import numpy as np
from numpy import sin, cos
from typing import Tuple


# ============================================================================
# 欧拉角转换
# ============================================================================

def euler_2_quat(rpy: np.ndarray, degrees: bool = False) -> np.ndarray:
    """
    欧拉角转四元数 (Roll-Pitch-Yaw -> Quaternion)
    
    Args:
        rpy: 欧拉角 [roll, pitch, yaw]，形状为(3,)
        degrees: True表示角度值，False表示弧度值
        
    Returns:
        四元数 [w, x, y, z]，形状为(4,)
        
    示例:
        >>> euler = np.array([0, 0, np.pi/2])  # 绕Z轴旋转90度
        >>> quat = euler_2_quat(euler)
        >>> print(quat)  # [0.707, 0, 0, 0.707]
    """
    roll = rpy[0]
    pitch = rpy[1]
    yaw = rpy[2]
    
    # 如果是角度，转换为弧度
    if degrees:
        roll = np.deg2rad(roll)
        pitch = np.deg2rad(pitch)
        yaw = np.deg2rad(yaw)
    
    # 使用半角公式计算四元数
    cy = cos(yaw * 0.5)
    sy = sin(yaw * 0.5)
    cp = cos(pitch * 0.5)
    sp = sin(pitch * 0.5)
    cr = cos(roll * 0.5)
    sr = sin(roll * 0.5)
    
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr
    
    return np.array([w, x, y, z])


def euler_2_mat(euler: np.ndarray) -> np.ndarray:
    """
    欧拉角转旋转矩阵 (XYZ顺序)
    
    Args:
        euler: 欧拉角 [roll, pitch, yaw]，形状为(3,)
        
    Returns:
        旋转矩阵，形状为(3, 3)
        
    注意:
        使用XYZ欧拉角顺序（先绕X轴roll，再绕Y轴pitch，最后绕Z轴yaw）
    """
    roll = euler[0]
    pitch = euler[1]
    yaw = euler[2]
    
    cr = cos(roll)
    sr = sin(roll)
    cp = cos(pitch)
    sp = sin(pitch)
    cy = cos(yaw)
    sy = sin(yaw)
    
    # 构建旋转矩阵
    r11 = cy * cp
    r12 = cy * sp * sr - sy * cr
    r13 = cy * sp * cr + sy * sr
    r21 = sy * cp
    r22 = sy * sp * sr + cy * cr
    r23 = sy * sp * cr - cy * sr
    r31 = -sp
    r32 = cp * sr
    r33 = cp * cr
    
    rotation_matrix = np.array([[r11, r12, r13],
                                [r21, r22, r23],
                                [r31, r32, r33]])
    return rotation_matrix


# ============================================================================
# 四元数转换
# ============================================================================

def quat_2_mat(quaternion: np.ndarray) -> np.ndarray:
    """
    四元数转旋转矩阵
    
    Args:
        quaternion: 四元数 [w, x, y, z]，形状为(4,)
        
    Returns:
        旋转矩阵，形状为(3, 3)
        
    注意:
        输入的四元数会自动归一化
    """
    w, x, y, z = quaternion
    
    # 归一化四元数
    norm = np.sqrt(w ** 2 + x ** 2 + y ** 2 + z ** 2)
    w /= norm
    x /= norm
    y /= norm
    z /= norm
    
    # 计算旋转矩阵元素
    r11 = 1 - 2 * y ** 2 - 2 * z ** 2
    r12 = 2 * x * y - 2 * w * z
    r13 = 2 * x * z + 2 * w * y
    r21 = 2 * x * y + 2 * w * z
    r22 = 1 - 2 * x ** 2 - 2 * z ** 2
    r23 = 2 * y * z - 2 * w * x
    r31 = 2 * x * z - 2 * w * y
    r32 = 2 * y * z + 2 * w * x
    r33 = 1 - 2 * x ** 2 - 2 * y ** 2
    
    rotation_matrix = np.array([[r11, r12, r13],
                                [r21, r22, r23],
                                [r31, r32, r33]])
    
    return rotation_matrix


def quat_2_euler(quaternion: np.ndarray) -> np.ndarray:
    """
    四元数转欧拉角 (XYZ顺序)
    
    Args:
        quaternion: 四元数 [w, x, y, z]，形状为(4,)
        
    Returns:
        欧拉角 [roll, pitch, yaw]，形状为(3,)
    """
    w, x, y, z = quaternion
    
    # Roll (X轴旋转)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x ** 2 + y ** 2))
    
    # Pitch (Y轴旋转)
    input_value = 2 * (w * y - z * x)
    input_value = np.clip(input_value, -1, 1)  # 防止数值误差导致arcsin域错误
    pitch = np.arcsin(input_value)
    
    # Yaw (Z轴旋转)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))
    
    euler = np.array([roll, pitch, yaw])
    return euler


# ============================================================================
# 旋转矩阵转换
# ============================================================================

def mat_2_quat(mat: np.ndarray) -> np.ndarray:
    """
    旋转矩阵转四元数
    
    Args:
        mat: 旋转矩阵，形状为(3, 3)
        
    Returns:
        四元数 [w, x, y, z]，形状为(4,)
        
    注意:
        使用Shepperd算法，数值稳定性好
    """
    trace = np.trace(mat)
    
    if trace > 0:
        # w是最大的
        S = np.sqrt(trace + 1.0) * 2  # S = 4*qw
        qw = 0.25 * S
        qx = (mat[2, 1] - mat[1, 2]) / S
        qy = (mat[0, 2] - mat[2, 0]) / S
        qz = (mat[1, 0] - mat[0, 1]) / S
    elif mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
        # x是最大的
        S = np.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]) * 2  # S = 4*qx
        qw = (mat[2, 1] - mat[1, 2]) / S
        qx = 0.25 * S
        qy = (mat[0, 1] + mat[1, 0]) / S
        qz = (mat[0, 2] + mat[2, 0]) / S
    elif mat[1, 1] > mat[2, 2]:
        # y是最大的
        S = np.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2]) * 2  # S = 4*qy
        qw = (mat[0, 2] - mat[2, 0]) / S
        qx = (mat[0, 1] + mat[1, 0]) / S
        qy = 0.25 * S
        qz = (mat[1, 2] + mat[2, 1]) / S
    else:
        # z是最大的
        S = np.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1]) * 2  # S = 4*qz
        qw = (mat[1, 0] - mat[0, 1]) / S
        qx = (mat[0, 2] + mat[2, 0]) / S
        qy = (mat[1, 2] + mat[2, 1]) / S
        qz = 0.25 * S
    
    quaternion = np.array([qw, qx, qy, qz])  # w,x,y,z
    return quaternion


def mat_2_euler(mat: np.ndarray) -> np.ndarray:
    """
    旋转矩阵转欧拉角
    
    Args:
        mat: 旋转矩阵，形状为(3, 3)
        
    Returns:
        欧拉角 [roll, pitch, yaw]，形状为(3,)
    """
    euler = quat_2_euler(mat_2_quat(mat))
    return euler


def mat_2_vec(mat: np.ndarray) -> np.ndarray:
    """
    旋转矩阵转旋转向量 (轴角表示)
    
    Args:
        mat: 旋转矩阵，形状为(3, 3)
        
    Returns:
        旋转向量 [rx, ry, rz]，形状为(3,)
        向量方向为旋转轴，模长为旋转角度
    """
    theta = np.arccos((np.trace(mat) - 1) / 2)
    if theta == 0:
        return np.zeros(3)
    else:
        k = 1 / (2 * np.sin(theta))
        x = k * (mat[2, 1] - mat[1, 2]) * theta
        y = k * (mat[0, 2] - mat[2, 0]) * theta
        z = k * (mat[1, 0] - mat[0, 1]) * theta
        return np.array([x, y, z])


# ============================================================================
# 旋转向量转换
# ============================================================================

def vec2_mat(vec: np.ndarray) -> np.ndarray:
    """
    旋转向量转旋转矩阵 (轴角表示)
    
    Args:
        vec: 旋转向量 [rx, ry, rz]，形状为(3,)
             向量方向为旋转轴，模长为旋转角度
        
    Returns:
        旋转矩阵，形状为(3, 3)
        
    注意:
        使用罗德里格斯公式的四元数形式
    """
    x = vec[0]
    y = vec[1]
    z = vec[2]
    
    # 旋转角度
    theta = np.sqrt(x * x + y * y + z * z)
    
    # 旋转轴（单位向量）
    axis = np.array([x, y, z]) / theta
    
    # 转换为四元数
    a = np.cos(theta / 2)
    b, c, d = -axis * np.sin(theta / 2)
    bb, cc, dd = b ** 2, c ** 2, d ** 2
    bc, bd, cd = b * c, b * d, c * d
    
    # 构建旋转矩阵
    rotation_matrix = np.array([[a * a + bb - cc - dd, 2 * (bc + a * d), 2 * (bd - a * c)],
                                [2 * (bc - a * d), a * a + cc - bb - dd, 2 * (cd + a * b)],
                                [2 * (bd + a * c), 2 * (cd - a * b), a * a + dd - bb - cc]])
    return rotation_matrix


def vec_2_euler(vec: np.ndarray) -> np.ndarray:
    """
    旋转向量转欧拉角
    
    Args:
        vec: 旋转向量 [rx, ry, rz]，形状为(3,)
        
    Returns:
        欧拉角 [roll, pitch, yaw]，形状为(3,)
    """
    mat = vec2_mat(vec)
    euler = mat_2_euler(mat)
    return euler


# ============================================================================
# 齐次变换矩阵
# ============================================================================

def make_transform(pos: np.ndarray = None, rot: np.ndarray = None) -> np.ndarray:
    """
    构建4x4齐次变换矩阵
    
    将位置向量和旋转矩阵组合成齐次变换矩阵：
    T = [R  t]
        [0  1]
    
    Args:
        pos: 位置向量，形状为(3,)或(3,1)
        rot: 旋转矩阵，形状为(3,3)
        
    Returns:
        齐次变换矩阵，形状为(4,4)
        
    示例:
        >>> pos = np.array([1, 2, 3])
        >>> rot = np.eye(3)
        >>> T = make_transform(pos, rot)
        >>> print(T.shape)  # (4, 4)
    """
    pos = np.asarray(pos).reshape(3, 1)
    rot = np.asarray(rot)
    return np.vstack((np.hstack((rot, pos)),
                      np.array([0, 0, 0, 1])))


def pose_inv(pose: np.ndarray) -> np.ndarray:
    """
    计算齐次变换矩阵的逆
    
    如果pose表示B在A中的位姿，则逆表示A在B中的位姿。
    
    数学原理:
        [R t; 0 1]^-1 = [R^T -R^T*t; 0 1]
    
    Args:
        pose: 齐次变换矩阵，形状为(4,4)
        
    Returns:
        逆变换矩阵，形状为(4,4)
    """
    pose_inv = np.zeros((4, 4))
    pose_inv[:3, :3] = pose[:3, :3].T  # R^T
    pose_inv[:3, 3] = -pose_inv[:3, :3].dot(pose[:3, 3])  # -R^T*t
    pose_inv[3, 3] = 1.0
    return pose_inv


# ============================================================================
# 力和力矩变换
# ============================================================================

def force_in_A_to_force_in_B(
    force_A: np.ndarray,
    torque_A: np.ndarray,
    pose_A_in_B: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将坐标系A中的力和力矩转换到坐标系B
    
    Args:
        force_A: A坐标系中的力 [fx, fy, fz]，形状为(3,)
        torque_A: A坐标系中的力矩 [tx, ty, tz]，形状为(3,)
        pose_A_in_B: A在B中的位姿（4x4齐次变换矩阵）
        
    Returns:
        tuple: (force_B, torque_B)
            - force_B: B坐标系中的力，形状为(3,)
            - torque_B: B坐标系中的力矩，形状为(3,)
    """
    pos_A_in_B = pose_A_in_B[:3, 3]
    rot_A_in_B = pose_A_in_B[:3, :3]
    
    # 力的变换：F_B = R^T * F_A
    force_B = rot_A_in_B.T.dot(force_A)
    
    # 力矩的变换：T_B = R^T * T_A
    torque_B = rot_A_in_B.T.dot(torque_A)
    
    return force_B, torque_B


# ============================================================================
# 四元数工具
# ============================================================================

def convert_quat(q: np.ndarray, to: str = "xyzw") -> np.ndarray:
    """
    转换四元数的表示顺序
    
    Args:
        q: 四元数，形状为(4,)
        to: 目标格式，"xyzw" 或 "wxyz"
            - "xyzw": [x, y, z, w] (常用于某些库)
            - "wxyz": [w, x, y, z] (本模块默认格式)
        
    Returns:
        转换后的四元数，形状为(4,)
        
    Raises:
        Exception: 如果to参数不是"xyzw"或"wxyz"
    """
    if to == "xyzw":
        return q[[1, 2, 3, 0]]  # wxyz -> xyzw
    if to == "wxyz":
        return q[[3, 0, 1, 2]]  # xyzw -> wxyz
    raise Exception("convert_quat: choose a valid `to` argument (xyzw or wxyz)")


# ============================================================================
# 数据归一化
# ============================================================================

def normalize(
    data: np.ndarray,
    data_range: Tuple[float, float] = (-50.0, 50.0),
    normal_range: Tuple[float, float] = (-1.0, 1.0)
) -> np.ndarray:
    """
    数据归一化
    
    将数据从原始范围线性映射到目标范围。
    
    Args:
        data: 待归一化数据
        data_range: 原始数据范围 (min, max)
        normal_range: 归一化后的范围 (min, max)
        
    Returns:
        归一化后的数据
        
    示例:
        >>> data = np.array([0, 25, 50])
        >>> normalized = normalize(data, data_range=(0, 50), normal_range=(-1, 1))
        >>> print(normalized)  # [-1, 0, 1]
    """
    normalized_data = normal_range[0] + (data - data_range[0]) * \
                      (normal_range[1] - normal_range[0]) / (data_range[1] - data_range[0])
    return normalized_data


def denormalize(
    normalized_data: np.ndarray,
    data_range: Tuple[float, float] = (-50.0, 50.0),
    normal_range: Tuple[float, float] = (-1.0, 1.0)
) -> np.ndarray:
    """
    数据逆归一化
    
    将归一化数据还原到原始范围。
    
    Args:
        normalized_data: 待逆归一化的数据
        data_range: 原始数据范围 (min, max)
        normal_range: 归一化范围 (min, max)
        
    Returns:
        逆归一化后的数据
        
    示例:
        >>> normalized = np.array([-1, 0, 1])
        >>> data = denormalize(normalized, data_range=(0, 50), normal_range=(-1, 1))
        >>> print(data)  # [0, 25, 50]
    """
    denormalized_data = ((normalized_data - normal_range[0]) / (normal_range[1] - normal_range[0])) * \
                        (data_range[1] - data_range[0]) + data_range[0]
    return denormalized_data


# ============================================================================
# 模块导出
# ============================================================================

__all__ = [
    # 欧拉角转换
    'euler_2_quat',
    'euler_2_mat',
    # 四元数转换
    'quat_2_mat',
    'quat_2_euler',
    # 旋转矩阵转换
    'mat_2_quat',
    'mat_2_euler',
    'mat_2_vec',
    # 旋转向量转换
    'vec2_mat',
    'vec_2_euler',
    # 齐次变换
    'make_transform',
    'pose_inv',
    # 力变换
    'force_in_A_to_force_in_B',
    # 四元数工具
    'convert_quat',
    # 归一化
    'normalize',
    'denormalize',
]

