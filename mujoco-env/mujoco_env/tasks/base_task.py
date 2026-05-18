"""
任务基类

作者: Liu Gang
日期: 2025-12-20
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import os
import numpy as np
from gymnasium import spaces
from ..robot_config.base import RobotConfig
from ..utils.robot_assembler import RobotAssembler




class ObservationConfig:
    """
    观测配置数据类
    
    定义任务需要的观测配置
    """
    
    def __init__(
        self,
        include_image: bool = False,
        image_size: Tuple[int, int] = (128, 128),
        include_depth: bool = False,
        include_proprioception: bool = True,
        include_goal: bool = False,
    ):
        """
        Args:
            include_image: 是否包含图像观测
            image_size: 图像尺寸 (height, width)
            include_depth: 是否包含深度图
            include_proprioception: 是否包含本体感知
            include_goal: 是否包含目标
        """
        self.include_image = include_image
        self.image_size = image_size
        self.include_depth = include_depth
        self.include_proprioception = include_proprioception
        self.include_goal = include_goal


class BaseTask(ABC):
    """
    任务基类
    """
    
    def __init__(
        self,
        name: str,
        robot_config: RobotConfig,
        scene_name: str = "default",
        include_image: bool = False,
        image_size: Tuple[int, int] = (128, 128),
        include_depth: bool = False,
        env: Optional[Any] = None,  # 环境引用，用于访问 site 位置
        **kwargs
    ):
        self.name = name
        self.robot_config = robot_config
        self.scene_name = scene_name
        self.env = env  # 环境引用
        
        # 观测配置属于任务
        self.obs_config = ObservationConfig(
            include_image=include_image,
            image_size=image_size,
            include_depth=include_depth,
            include_proprioception=True,
            include_goal=True
        )
        
        # max_episode_steps 不属于任务，而是强化学习参数
        
        # 构建模型文件
        assets_dir = self.get_assets_path()
        output_dir = assets_dir.parent / "tasks" / self.name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        assembler = RobotAssembler(assets_dir)
        self.model_path = assembler.build_robot_scene(self.scene_name, self.robot_config, output_dir)
        
        self.current_step = 0
        self.is_success = False
        self.np_random = np.random.RandomState()
        self.env = env  # 环境引用

    def set_env(self, env: Any):
        """设置环境引用"""
        self.env = env
    
    def get_assets_path(self) -> Path:
        """获取assets目录路径"""
        return Path(__file__).parent.parent / "assets"
        
    @abstractmethod
    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Dict[str, Any]
    ) -> float:
        """
        计算奖励
        
        Args:
            achieved_goal: 当前达到的目标
            desired_goal: 期望的目标
            info: 额外信息
            
        Returns:
            reward: 奖励值
        """
        raise NotImplementedError
    
    @abstractmethod
    def is_success_fn(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray
    ) -> bool:
        """
        判断任务是否成功
        
        Args:
            achieved_goal: 当前达到的目标
            desired_goal: 期望的目标
            
        Returns:
            success: 是否成功
        """
        raise NotImplementedError
    
    @abstractmethod
    def sample_goal(self) -> np.ndarray:
        """
        采样一个新的目标
        
        Returns:
            goal: 目标数组
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_achieved_goal(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        从观测中提取当前达到的目标
        
        Args:
            obs: 观测字典
            
        Returns:
            achieved_goal: 当前达到的目标
        """
        raise NotImplementedError
    
    def reset(self) -> Dict[str, Any]:
        """
        重置任务
    
        Returns:
            task_info: 任务信息
        """
        self.current_step = 0
        self.is_success = False
        
        # 采样新目标
        desired_goal = self.sample_goal()
        
        return {
            "desired_goal": desired_goal,
            "task_name": self.name
        }
    
    def step(self, obs: Dict[str, np.ndarray]) -> Tuple[float, bool, Dict[str, Any]]:
        """
        任务步进
        
        Args:
            obs: 当前观测
            
        Returns:
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        self.current_step += 1
        
        # 获取achieved和desired goal
        achieved_goal = self.get_achieved_goal(obs)
        desired_goal = obs.get("desired_goal", self.sample_goal())
        
        # 计算奖励
        reward = self.compute_reward(achieved_goal, desired_goal, {})
        
        # 判断成功
        self.is_success = self.is_success_fn(achieved_goal, desired_goal)
        
        # 判断是否结束
        done = self.is_success or self.current_step >= self.max_episode_steps
        
        info = {
            "is_success": self.is_success,
            "task_step": self.current_step,
            "achieved_goal": achieved_goal,
            "desired_goal": desired_goal
        }
        
        return reward, done, info
    
    def get_obs_space(self) -> spaces.Dict:
        """
        获取任务相关的观测空间（子类可选实现）
        
        观测空间由三部分组成：
        1. 机器人本体感知 (qpos, qvel, tcp_pos, tcp_quat)
        2. 任务相关观测 (从 task.get_obs_space() 获取)
        3. 传感器观测 (image, depth)
        
        Returns:
            obs_space: 任务相关的观测空间（如 object_pos, achieved_goal 等）
        """
        return spaces.Dict({})
    
    def get_sensor_config(self) -> Dict[str, Any]:
        """
        获取任务需要的传感器配置（子类可选实现）
        
        Returns:
            sensor_config: 传感器配置字典，包含：
                - include_image: bool - 是否需要图像
                - image_size: Tuple[int, int] - 图像尺寸
                - include_depth: bool - 是否需要深度图
                
        Examples:
            >>> def get_sensor_config(self):
            ...     return {
            ...         "include_image": True,
            ...         "image_size": (84, 84),
            ...         "include_depth": False,
            ...     }
        """
        return {
            "include_image": False,
            "image_size": (128, 128),
            "include_depth": False,
        }
    
    def get_info(self) -> Dict[str, Any]:
        """
        获取任务当前状态信息（子类可选实现）
        
        Returns:
            info: 任务信息字典，基本包含：
                - is_success: bool - 是否成功
                - task_step: int - 当前任务步数
        """
        return {
            "is_success": self.is_success,
            "task_step": self.current_step,
        }
    
    def __repr__(self) -> str:
        return f"{self.name}(max_steps={self.max_episode_steps})"

