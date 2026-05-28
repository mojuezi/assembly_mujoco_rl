"""
抓取方块任务 - 主任务文件

描述具体的任务、观测空间和奖励函数设置，并绑定场景。
每个任务可以由不同的机器人来实现。

作者: Liu Gang
日期: 2025-12-25
"""

from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
from gymnasium import spaces

from ..base_task import BaseTask, ObservationConfig
from ...robot_config.base import RobotConfig


class PickCubeTask(BaseTask):
    """
    抓取方块任务
    
    任务描述:
        机器人需要：
        1. 定位桌面上的方块
        2. 移动末端执行器靠近方块
        3. 闭合夹爪抓取方块
        4. 举起方块到指定高度
    
    奖励函数:
        reward = 0.3 * r_close + 0.7 * r_lift
        - r_close: 末端执行器靠近方块的奖励（指数衰减）
        - r_lift: 方块举起高度的奖励（归一化到[0, 1]）
    
    成功条件:
        方块被举起到 z_init + lift_height 以上
    
    参数:
        cube_size: 方块半尺寸（默认0.02m）
        lift_height: 成功所需的举起高度（默认0.2m）
        sampling_bounds: 方块初始位置的采样范围
        distance_threshold: 距离阈值，用于判断接近程度
        scene_name: 场景名称（默认 "grasping"）
    """
    
    # 任务绑定的场景
    DEFAULT_SCENE = "grasping"
    
    def __init__(
        self,
        robot_config: RobotConfig,
        scene_name: str = "grasping",
        include_image: bool = False,
        image_size: tuple = (128, 128),
        include_depth: bool = False,
        cube_size: float = 0.02,
        lift_height: float = 0.2,
        sampling_bounds: np.ndarray = None,
        distance_threshold: float = 0.05,
        **kwargs
    ):
        super().__init__(
            name="pick_cube",
            robot_config=robot_config,
            scene_name=scene_name,
            include_image=include_image,
            image_size=image_size,
            include_depth=include_depth,
            **kwargs
        )
        
        self.cube_size = cube_size
        self.lift_height = lift_height
        self.distance_threshold = distance_threshold
        
        if sampling_bounds is None:
            sampling_bounds = np.array([[0.25, -0.25], [0.55, 0.25]])
        self.sampling_bounds = sampling_bounds
        
        self._z_init = None
        self._z_success = None
        self._block_initial_pos = None

    def reset(self) -> Dict[str, Any]:
        """
        重置任务
        
        Returns:
            info: 包含任务初始信息的字典
                - desired_goal: 期望目标位置 (x, y, z)
                - block_pos: 方块初始位置
        """
        super().reset()
        
        # 随机采样方块位置（XY平面）
        block_xy = self.np_random.uniform(
            self.sampling_bounds[0],
            self.sampling_bounds[1]
        )
        
        # 方块初始位置（Z=cube_size，放在桌面上）
        block_pos = np.array([block_xy[0], block_xy[1], self.cube_size])
        self._block_initial_pos = block_pos.copy()
        
        # 初始化高度记录
        self._z_init = self.cube_size
        self._z_success = self._z_init + self.lift_height
        
        # 期望目标：将方块举起到目标高度
        desired_goal = np.array([block_xy[0], block_xy[1], self._z_success])
        
        return {
            "desired_goal": desired_goal,
            "block_pos": block_pos,
        }
    
    def step(self, observation: Dict[str, np.ndarray]) -> tuple:
        """
        任务步进
        
        Args:
            observation: 环境观测
        
        Returns:
            reward: 奖励值
            terminated: 是否终止（成功）
            info: 额外信息
        """
        # 计算奖励
        achieved_goal = self.get_achieved_goal(observation)
        desired_goal = observation.get("desired_goal")
        
        # 构建info字典
        info = {
            "tcp_pos": observation.get("tcp_pos", np.zeros(3)),
            "block_pos": observation.get("state", {}).get("block_pos", achieved_goal),
        }
    
        reward = self.compute_reward(achieved_goal, desired_goal, info)
        
        # 判断是否成功
        terminated = self.is_success_fn(achieved_goal, desired_goal)
        
        # 更新步数
        self.current_step += 1
        
        info["is_success"] = terminated
        
        return reward, terminated, info
    
    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Dict[str, Any]
    ) -> float:
        """
        计算稠密奖励
        
        奖励由两部分组成：
        1. 接近奖励 (r_close): 末端执行器靠近方块，使用指数衰减
        2. 举起奖励 (r_lift): 方块举起的高度，归一化到[0, 1]
        
        Args:
            achieved_goal: 当前达到的目标（方块位置）
            desired_goal: 期望目标（方块目标位置）
            info: 额外信息（包含tcp_pos, block_pos）
        
        Returns:
            reward: 标量奖励值
        """
        # 获取方块和末端执行器位置
        block_pos = info.get("block_pos", achieved_goal)
        tcp_pos = info.get("tcp_pos", np.zeros(3))
        
        # 1. 接近奖励：末端执行器靠近方块（指数衰减）
        dist = np.linalg.norm(block_pos - tcp_pos)
        r_close = np.exp(-20 * dist)
        
        # 2. 举起奖励：方块举起高度（归一化）
        if self._z_init is not None:
            z_current = block_pos[2]
            r_lift = (z_current - self._z_init) / self.lift_height
            r_lift = np.clip(r_lift, 0.0, 1.0)
        else:
            r_lift = 0.0
        
        # 组合奖励（权重：30% 接近，70% 举起）
        reward = 0.3 * r_close + 0.7 * r_lift
        
        return float(reward)
    
    def is_success_fn(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray
    ) -> bool:
        """
        判断任务是否成功
        
        成功条件：方块Z坐标达到或超过目标高度
        
        Args:
            achieved_goal: 当前达到的目标（方块位置）
            desired_goal: 期望目标（方块目标位置）
        
        Returns:
            success: 是否成功
        """
        if self._z_success is None:
            return False
        
        # 判断方块是否举起到目标高度
        block_z = achieved_goal[2]
        success = block_z >= self._z_success
        
        return bool(success)
    
    def get_achieved_goal(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """
        从观测中提取achieved_goal
        
        对于抓取任务，achieved_goal就是方块的当前位置
        
        Args:
            observation: 环境观测
        
        Returns:
            achieved_goal: 方块当前位置 (x, y, z)
        """
        # 尝试从observation中获取方块位置
        if "state" in observation:
            state = observation["state"]
            if "block_pos" in state:
                return state["block_pos"].copy()
        
        # 如果没有方块位置信息，返回初始位置（fallback）
        if self._block_initial_pos is not None:
            return self._block_initial_pos.copy()
        
        # 默认返回零向量
        return np.zeros(3, dtype=np.float32)

    def get_ft_sensor_data(self, data: Any) -> np.ndarray:
        """
        安全地获取力传感器数据 [Fx, Fy, Fz, Tx, Ty, Tz]
        
        Args:
            data: MuJoCo data 对象
            
        Returns:
            wrench: 6维力/力矩数组，如果未配置或读取失败则返回全 0 数组
        """
        wrench = np.zeros(6, dtype=np.float32)
        if self.robot_config.use_ft_sensor:
            try:
                # 优先使用固定的传感器名称 force_ee 和 torque_ee
                wrench[:3] = data.sensor("force_ee").data
                wrench[3:] = data.sensor("torque_ee").data
            except Exception:
                try:
                    # 备选名称
                    wrench[:3] = data.sensor("end_force_sensor").data
                    wrench[3:] = data.sensor("end_torque_sensor").data
                except Exception:
                    pass
        return wrench
    
    def sample_goal(self) -> np.ndarray:
        """
        采样一个目标
        
        对于抓取任务，目标是将方块举起到指定高度
        
        Returns:
            goal: 目标位置 (x, y, z_target)
        """
        # 随机采样XY位置
        block_xy = self.np_random.uniform(
            self.sampling_bounds[0],
            self.sampling_bounds[1]
        )
        
        # Z坐标为目标高度
        z_target = self.cube_size + self.lift_height
        
        goal = np.array([block_xy[0], block_xy[1], z_target], dtype=np.float32)
        
        return goal
    
    def get_obs_space(self) -> spaces.Dict:
        """
        获取任务相关的观测空间
        
        Returns:
            obs_space: 任务相关的观测空间字典
        """
        obs_dict = {}
        
        # 方块位置
        obs_dict["block_pos"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3,),
            dtype=np.float32
        )
        
        # 目标位置
        obs_dict["achieved_goal"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3,),
            dtype=np.float32
        )
        
        obs_dict["desired_goal"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3,),
            dtype=np.float32
        )
        
        return spaces.Dict(obs_dict)
    
    def get_action_space(self) -> spaces.Box:
        """
        获取动作空间
        
        根据控制器类型返回相应的动作空间：
        - cartesian_ik, operational_space, cartesian_impedance: 7维位姿 + 1维夹爪 = 8维
        - joint_position, joint_velocity, joint_torque: dof维关节 + 1维夹爪
        
        Returns:
            action_space: 动作空间 [-6.28, 16.28]
        """

        # 根据控制器类型确定动作维度
        controller_type = self.robot_config.controller_type
        if controller_type in ["cartesian_ik", "operational_space", "cartesian_impedance", "osc", "task_space"]:
            # 任务空间控制：7维位姿（位置+四元数）+ 1维夹爪
            dim = 7
            if self.robot_config.gripper_name:
                dim += 1
        else:
            # 关节空间控制：dof维关节 + 1维夹爪
            dim = self.robot_config.dof
            if self.robot_config.gripper_name:
                dim += 1
            
        return spaces.Box(
            low=-6.28,
            high=6.28,
            shape=(dim,), 
            dtype=np.float32
        )
    
    def get_info(self) -> Dict[str, Any]:
        """
        获取任务信息（包含场景信息）
        
        Returns:
            info: 任务信息字典
        """
        info = super().get_info()
        info.update({
            "cube_size": self.cube_size,
            "lift_height": self.lift_height,
            "z_init": self._z_init,
            "z_success": self._z_success,
            "scene_name": self.scene_name,
            "scene_path": str(self.model_path),
        })
        return info
    
    def __repr__(self) -> str:
        return f"{self.name}(robot={self.robot_config.name}, scene={self.scene_name})"



# 便捷函数：创建任务实例
# 移除创建函数，推荐用户直接创建配置和任务对象

