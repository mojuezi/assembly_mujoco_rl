"""
单次运动任务 - 最简单的任务实现

这是一个最简单的任务，用于演示机器人的基本运动能力。
任务没有复杂的目标，只是让机器人在场景中运动。

作者: Liu Gang
日期: 2025-01-02
"""

from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
from gymnasium import spaces

from ..base_task import BaseTask, ObservationConfig
from ...robot_config.base import RobotConfig


class SingleMoveTask(BaseTask):
    """
    单次运动任务
    
    任务描述:
        这是一个最简单的任务，机器人只需要在场景中运动。
        没有复杂的目标或奖励函数，主要用于：
        1. 测试机器人配置
        2. 验证仿真环境
        3. 演示基本运动
    
    奖励函数:
        简单的存在奖励，每步给予小的正奖励
    
    成功条件:
        无特定成功条件，任务会在达到最大步数时结束
    
    参数:
        scene_name: 场景名称（默认 "empty"）
    """
    
    # 任务绑定的场景
    DEFAULT_SCENE = "default"
    
    def __init__(
        self,
        robot_config: RobotConfig,
        scene_name: str = "default",
        include_image: bool = False,
        image_size: tuple = (128, 128),
        include_depth: bool = False,
        **kwargs
    ):
        super().__init__(
            name="single_move",
            robot_config=robot_config,
            scene_name=scene_name,
            include_image=include_image,
            image_size=image_size,
            include_depth=include_depth,
            **kwargs
        )
        
        # 简单的目标：机器人的TCP位置
        self._target_pos = np.array([0.5, 0.0, 0.3])  # 默认目标位置
        
    def reset(self) -> Dict[str, Any]:
        """重置任务"""
        result = super().reset()
        
        # 随机化目标位置（在工作空间内）
        self._target_pos = np.array([
            0.3 + 0.4 * self.np_random.random(),  # x: 0.3 ~ 0.7
            -0.2 + 0.4 * self.np_random.random(), # y: -0.2 ~ 0.2  
            0.2 + 0.3 * self.np_random.random()   # z: 0.2 ~ 0.5
        ])
        
        result["desired_goal"] = self._target_pos.copy()
        return result
    
    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Dict[str, Any]
    ) -> float:
        """
        计算奖励
        
        简单的距离奖励：越接近目标位置奖励越高
        """
        # 计算TCP到目标的距离
        distance = np.linalg.norm(achieved_goal - desired_goal)
        
        # 距离奖励（指数衰减）
        distance_reward = np.exp(-5.0 * distance)
        
        # 存在奖励
        existence_reward = 0.1
        
        return distance_reward + existence_reward
    
    def is_success_fn(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray
    ) -> bool:
        """
        判断任务是否成功
        
        当TCP位置接近目标位置时认为成功
        """
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return distance < 0.05  # 5cm 阈值
    
    def sample_goal(self) -> np.ndarray:
        """
        采样一个新的目标
        
        在机器人工作空间内随机采样目标位置
        """
        return np.array([
            0.3 + 0.4 * self.np_random.random(),  # x: 0.3 ~ 0.7
            -0.2 + 0.4 * self.np_random.random(), # y: -0.2 ~ 0.2  
            0.2 + 0.3 * self.np_random.random()   # z: 0.2 ~ 0.5
        ])
    
    def get_achieved_goal(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        从观测中提取当前达到的目标
        
        对于单次运动任务，achieved_goal 就是当前的TCP位置
        """
        # 假设观测中包含 tcp_pos
        if "tcp_pos" in obs:
            return obs["tcp_pos"].copy()
        else:
            # 如果没有tcp_pos，返回默认位置
            return np.array([0.0, 0.0, 0.0])
    
    def get_obs_space(self) -> spaces.Dict:
        """
        获取任务相关的观测空间
        
        对于单次运动任务，只需要目标位置
        """
        return spaces.Dict({
            "desired_goal": spaces.Box(
                low=np.array([0.2, -0.3, 0.1]),
                high=np.array([0.8, 0.3, 0.6]),
                dtype=np.float32
            ),
            "achieved_goal": spaces.Box(
                low=np.array([0.2, -0.3, 0.1]),
                high=np.array([0.8, 0.3, 0.6]),
                dtype=np.float32
            )
        })
    
    def get_action_space(self) -> spaces.Box:
        """
        获取动作空间
        
        根据控制器类型返回相应的动作空间：
        - cartesian_ik: 7维 (位置+四元数)
        - joint_position: dof维 (关节位置)
        
        Returns:
            action_space: 动作空间 [-1, 1]
        """
        if self.robot_config.controller_type == "cartesian_ik":
            return spaces.Box(
                low=-3.0,
                high=3.0,
                shape=(7,),  # 笛卡尔位姿：位置 + 四元数
                dtype=np.float32
            )
        else:
            # 关节空间控制 (joint_position, joint_velocity, joint_torque)
            # 使用更大的范围以支持完整的关节运动（Aubo i5关节范围通常为 +/- 175度 或 +/- 2*pi）
            return spaces.Box(
                low=-2 * np.pi,
                high=2 * np.pi,
                shape=(self.robot_config.dof,),
                dtype=np.float32
            )
    
    def get_sensor_config(self) -> Dict[str, Any]:
        """
        获取传感器配置
        
        单次运动任务不需要复杂的传感器
        """
        return {
            "include_image": self.obs_config.include_image,
            "image_size": self.obs_config.image_size,
            "include_depth": self.obs_config.include_depth,
        }
    
    def get_info(self) -> Dict[str, Any]:
        """获取任务信息"""
        info = super().get_info()
        info.update({
            "target_pos": self._target_pos.copy(),
            "task_type": "single_move"
        })
        return info
