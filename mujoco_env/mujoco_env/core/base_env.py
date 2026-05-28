"""
基础机器人环境 - Gymnasium接口

作者: Liu Gang
日期: 2025-12-20
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class BaseRobotEnv(gym.Env, ABC):
    """
    基础机器人环境类 - 遵循Gymnasium接口
    
    所有机器人环境（仿真或真机）都应继承此类
    
    Attributes:
        observation_space: 观测空间
        action_space: 动作空间
        metadata: 环境元数据
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array", "depth"],
        "render_fps": 30,
    }
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        **kwargs
    ):
        """
        初始化环境
        
        Args:
            render_mode: 渲染模式 ('human', 'rgb_array', 'depth', None)
            **kwargs: 其他配置参数
        """
        super().__init__()
        
        self.render_mode = render_mode
        self._elapsed_steps = 0
        self._episode_count = 0
        
        # 子类需要实现这些空间的定义
        self.observation_space = None
        self.action_space = None
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        重置环境到初始状态
        
        Args:
            seed: 随机种子
            options: 额外选项
            
        Returns:
            observation: 初始观测
            info: 额外信息字典
        """
        # 调用父类设置随机种子
        super().reset(seed=seed)
        
        self._elapsed_steps = 0
        self._episode_count += 1
        
        # 子类实现具体重置逻辑
        # observation = self._get_obs()
        # info = self._get_info()
        
        # return observation, info
    
    @abstractmethod
    def step(
        self, 
        action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        执行一步动作
        
        Args:
            action: 动作数组
            
        Returns:
            observation: 新的观测
            reward: 奖励值
            terminated: 是否达到终止状态（成功或失败）
            truncated: 是否被截断（超时）
            info: 额外信息字典
        """
        raise NotImplementedError
    
    @abstractmethod
    def render(self):
        """
        渲染环境
        
        Returns:
            如果render_mode='rgb_array'，返回图像数组
            如果render_mode='depth'，返回深度图
            如果render_mode='human'，在窗口显示，
            否则返回None
        """
        raise NotImplementedError
    
    def close(self):
        """关闭环境，释放资源"""
        pass
    
    @abstractmethod
    def _get_obs(self) -> Dict[str, np.ndarray]:
        """
        获取当前观测
        
        Returns:
            observation: 观测字典
        """
        raise NotImplementedError
    
    def _get_info(self) -> Dict[str, Any]:
        """
        获取额外信息
        
        Returns:
            info: 信息字典，可包含:
                - is_success: 任务是否成功
                - elapsed_steps: 已执行步数
                - episode: episode数量
        """
        return {
            "elapsed_steps": self._elapsed_steps,
            "episode": self._episode_count,
        }
    
    @abstractmethod
    def _compute_reward(self) -> float:
        """
        计算奖励
        
        Returns:
            reward: 奖励值
        """
        raise NotImplementedError
    
    def _check_terminated(self) -> bool:
        """
        检查是否达到终止状态（成功或失败）
        
        Returns:
            terminated: 是否终止
        """
        return False
    
    def _check_truncated(self) -> bool:
        """
        检查是否被截断（如超时）
        
        Returns:
            truncated: 是否截断
        """
        return False

