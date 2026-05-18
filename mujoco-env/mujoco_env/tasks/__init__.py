"""
任务定义模块

提供各种机器人操作任务
"""

from typing import Dict, Type
from .base_task import (
    BaseTask,
    RobotConfig,
    ObservationConfig,
)
from .pcb_insertion.pcb_insertion import PCBInsertionTask
from .peg_insertion.peg_insertion import PegInsertionTask
from .pick_cube.pick_cube import PickCubeTask


# 任务注册表
TASK_REGISTRY: Dict[str, Type[BaseTask]] = {
    "pick_cube": PickCubeTask,
    "pick": PickCubeTask,  # 别名
    "cube": PickCubeTask,  # 别名
    "pcb_insertion": PCBInsertionTask,
    "pcb": PCBInsertionTask,  # 别名
    "peg_insertion": PegInsertionTask,
    "peg": PegInsertionTask,  # 别名
    "peg_in_hole": PegInsertionTask,  # 别名
}


def get_task(task_name: str, **kwargs) -> BaseTask:
    """
    获取任务实例
    
    Args:
        task_name: 任务名称
        **kwargs: 任务参数
        
    Returns:
        task: 任务实例
        
    Raises:
        ValueError: 如果任务不存在
    
    Example:
        >>> task = get_task("pcb_insertion", max_episode_steps=500)
        >>> goal = task.sample_goal()
        >>> reward = task.compute_reward(achieved, desired, {})
    """
    if task_name not in TASK_REGISTRY:
        available = list(TASK_REGISTRY.keys())
        raise ValueError(
            f"Unknown task: {task_name}. "
            f"Available tasks: {available}"
        )
    
    task_cls = TASK_REGISTRY[task_name]
    return task_cls(**kwargs)


def register_task(name: str, task_class: Type[BaseTask]):
    """
    注册新的任务类型
    
    Args:
        name: 任务名称
        task_class: 任务类
    """
    TASK_REGISTRY[name] = task_class


__all__ = [
    "BaseTask",
    "PickCubeTask",
    "PCBInsertionTask",
    "PegInsertionTask",
    "get_task",
    "register_task",
    "TASK_REGISTRY",
]
