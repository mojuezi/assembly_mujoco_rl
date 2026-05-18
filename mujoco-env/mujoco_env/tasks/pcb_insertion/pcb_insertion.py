"""
PCB插拔任务

作者: Liu Gang
日期: 2025-12-20
"""

from typing import Dict, Any
import numpy as np
from gymnasium import spaces
from mujoco_env.mujoco_env.tasks.base_task import BaseTask


class PCBInsertionTask(BaseTask):
    """
    PCB插拔任务
    
    目标：将PCB板精确插入到插槽中
    成功条件：PCB位置与目标位置的距离小于阈值，且姿态对齐
    """
    
    def __init__(
        self,
        max_episode_steps: int = 500,
        success_threshold: float = 0.005,  # 5mm
        position_tolerance: float = 0.005,
        orientation_tolerance: float = 0.1,  # 约5.7度
        insertion_depth: float = 0.05,  # 插入深度 50mm
        **kwargs
    ):
        """
        初始化PCB插拔任务
        
        Args:
            max_episode_steps: 最大步数
            success_threshold: 成功阈值
            position_tolerance: 位置容差
            orientation_tolerance: 姿态容差（弧度）
            insertion_depth: 插入深度
            **kwargs: 额外参数
        """
        super().__init__(
            name="PCBInsertion",
            max_episode_steps=max_episode_steps,
            success_threshold=success_threshold,
            **kwargs
        )
        
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        self.insertion_depth = insertion_depth
        
        # PCB插槽位置范围
        self.slot_position_range = np.array([
            [0.3, 0.5],   # x: 300-500mm
            [-0.2, 0.2],  # y: -200-200mm
            [0.1, 0.3],   # z: 100-300mm
        ])
        
    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Dict[str, Any]
    ) -> float:
        """
        计算奖励
        
        奖励组成：
        - 距离奖励：与目标位置的距离
        - 姿态奖励：姿态对齐程度
        - 插入奖励：成功插入给予额外奖励
        
        Args:
            achieved_goal: 当前PCB位姿 [x, y, z, qw, qx, qy, qz]
            desired_goal: 目标位姿 [x, y, z, qw, qx, qy, qz]
            info: 额外信息
            
        Returns:
            reward: 奖励值
        """
        # 位置误差
        pos_achieved = achieved_goal[:3]
        pos_desired = desired_goal[:3]
        pos_error = np.linalg.norm(pos_achieved - pos_desired)
        
        # 姿态误差（简化：使用四元数点积）
        quat_achieved = achieved_goal[3:7]
        quat_desired = desired_goal[3:7]
        quat_dot = np.abs(np.dot(quat_achieved, quat_desired))
        quat_error = 1.0 - quat_dot
        
        # 距离奖励（越近越好）
        distance_reward = -pos_error * 10.0
        
        # 姿态奖励
        orientation_reward = -quat_error * 5.0
        
        # 成功插入额外奖励
        success_bonus = 0.0
        if pos_error < self.position_tolerance and quat_error < self.orientation_tolerance:
            success_bonus = 10.0
        
        reward = distance_reward + orientation_reward + success_bonus
        
        return reward
    
    def is_success_fn(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray
    ) -> bool:
        """
        判断任务是否成功
        
        Args:
            achieved_goal: 当前PCB位姿
            desired_goal: 目标位姿
            
        Returns:
            success: 是否成功
        """
        # 位置误差
        pos_error = np.linalg.norm(achieved_goal[:3] - desired_goal[:3])
        
        # 姿态误差
        quat_achieved = achieved_goal[3:7]
        quat_desired = desired_goal[3:7]
        quat_dot = np.abs(np.dot(quat_achieved, quat_desired))
        quat_error = 1.0 - quat_dot
        
        # 同时满足位置和姿态要求
        position_ok = pos_error < self.position_tolerance
        orientation_ok = quat_error < self.orientation_tolerance
        
        return position_ok and orientation_ok
    
    def sample_goal(self) -> np.ndarray:
        """
        采样一个新的PCB插槽目标位置
        
        Returns:
            goal: 目标位姿 [x, y, z, qw, qx, qy, qz]
        """
        # 随机采样插槽位置
        position = np.array([
            np.random.uniform(self.slot_position_range[0, 0], self.slot_position_range[0, 1]),
            np.random.uniform(self.slot_position_range[1, 0], self.slot_position_range[1, 1]),
            np.random.uniform(self.slot_position_range[2, 0], self.slot_position_range[2, 1]),
        ])
        
        # 固定姿态（垂直向下）
        orientation = np.array([0.0, 1.0, 0.0, 0.0])  # [qw, qx, qy, qz] - 90度绕x轴
        
        return np.concatenate([position, orientation])
    
    def get_achieved_goal(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        从观测中提取当前PCB位姿
        
        Args:
            obs: 观测字典，应包含"pcb_pos"和"pcb_quat"
            
        Returns:
            achieved_goal: 当前PCB位姿 [x, y, z, qw, qx, qy, qz]
        """
        if "pcb_pos" in obs and "pcb_quat" in obs:
            return np.concatenate([obs["pcb_pos"], obs["pcb_quat"]])
        
        # 如果没有PCB观测，使用机械臂末端位置作为替代
        if "tcp_pos" in obs and "tcp_quat" in obs:
            return np.concatenate([obs["tcp_pos"], obs["tcp_quat"]])
        
        # 默认返回零向量
        return np.zeros(7)
    
    def get_obs_space(self) -> spaces.Dict:
        """
        获取任务相关的观测空间
        
        Returns:
            obs_space: 观测空间
        """
        return spaces.Dict({
            "pcb_pos": spaces.Box(
                low=np.array([0.0, -0.5, 0.0]),
                high=np.array([1.0, 0.5, 0.5]),
                dtype=np.float32
            ),
            "pcb_quat": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(4,),
                dtype=np.float32
            ),
            "desired_goal": spaces.Box(
                low=np.array([0.0, -0.5, 0.0, -1, -1, -1, -1]),
                high=np.array([1.0, 0.5, 0.5, 1, 1, 1, 1]),
                dtype=np.float32
            ),
        })

