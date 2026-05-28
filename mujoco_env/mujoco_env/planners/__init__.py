"""
路径规划与轨迹插值模块

提供机器人运动规划和轨迹生成的工具。

子模块：
- rrt: RRT*路径规划算法
- interpolators: 关节空间轨迹插值器
- linear_interpolator: 笛卡尔空间线性插值器

规划器：
- RRT: 3D空间中的RRT*路径规划
- Node: RRT树节点
- AreaBounds: 规划区域边界

轨迹插值器：
- OTG: 关节空间在线轨迹生成
- LinearInterpolator: 笛卡尔空间线性插值

作者: Liu Gang
日期: 2026-01-03
"""

from typing import Dict, Type

# 路径规划
from .rrt import RRT, Node, AreaBounds

# 轨迹插值
from .otg_interpolator import OTG
from .linear_interpolator import LinearInterpolator


# 规划器注册表
PLANNER_REGISTRY: Dict[str, Type] = {
    "rrt": RRT,
    "rrt_star": RRT,  # 别名
}


# 插值器注册表
INTERPOLATOR_REGISTRY: Dict[str, Type] = {
    "otg": OTG,
    "joint": OTG,
    "joint_space": OTG,
    "linear": LinearInterpolator,
    "cartesian": LinearInterpolator,
    "task_space": LinearInterpolator,
}


def get_planner(planner_name: str) -> Type:
    """
    根据名称获取规划器类
    
    Args:
        planner_name: 规划器名称
        
    Returns:
        规划器类
        
    Raises:
        ValueError: 如果规划器不存在
    """
    if planner_name not in PLANNER_REGISTRY:
        available = list(PLANNER_REGISTRY.keys())
        raise ValueError(
            f"未知的规划器: '{planner_name}'.\n"
            f"可用的规划器: {available}"
        )
    return PLANNER_REGISTRY[planner_name]


def get_interpolator(interpolator_name: str) -> Type:
    """
    根据名称获取插值器类
    
    Args:
        interpolator_name: 插值器名称
        
    Returns:
        插值器类
        
    Raises:
        ValueError: 如果插值器不存在
    """
    if interpolator_name not in INTERPOLATOR_REGISTRY:
        available = list(INTERPOLATOR_REGISTRY.keys())
        raise ValueError(
            f"未知的插值器: '{interpolator_name}'.\n"
            f"可用的插值器: {available}"
        )
    return INTERPOLATOR_REGISTRY[interpolator_name]


def list_planners() -> list:
    """列出所有可用的规划器"""
    return sorted(PLANNER_REGISTRY.keys())


def list_interpolators() -> list:
    """列出所有可用的插值器"""
    return sorted(INTERPOLATOR_REGISTRY.keys())


__all__ = [
    # 路径规划
    "RRT",
    "Node",
    "AreaBounds",
    
    # 轨迹插值
    "OTG",
    "JointSpaceInterpolator",
    "LinearInterpolator",
    
    # 注册表
    "PLANNER_REGISTRY",
    "INTERPOLATOR_REGISTRY",
    
    # 工具函数
    "get_planner",
    "get_interpolator",
    "list_planners",
    "list_interpolators",
]

