"""
水果采摘任务 - 主任务文件

描述机器人采摘水果的任务逻辑、观测空间和奖励函数。

日期: 2026-01-15
"""

from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from gymnasium import spaces

from ..base_task import BaseTask, ObservationConfig
from ...robot_config.base import RobotConfig


class FruitPickTask(BaseTask):
    """
    水果采摘任务
    
    任务描述:
        机器人需要：
        1. 定位水果 (通过 fruit_site 获取位置)
        2. 移动末端执行器靠近水果
        3. 闭合夹爪采摘水果
    
    奖励函数:
        reward = r_close + bonus
        - r_close: 末端执行器靠近水果的奖励（指数衰减）
        - bonus: 成功采摘的额外奖励
    
    成功条件:
        末端执行器距离抓取目标点（pick_site）小于 distance_threshold
    
    参数:
        distance_threshold: 成功采摘的距离阈值（默认0.01m）
        pick_site_name: 抓取目标位置site名称（默认 "pick_site"）
        fruit_site_name: 水果中心位置site名称（默认 "fruit_site"）
    """
    
    # 任务绑定的场景
    DEFAULT_SCENE = "fruitpick"
    
    def __init__(
        self,
        robot_config: RobotConfig,
        scene_name: str = "fruitpick",
        include_image: bool = False,
        image_size: tuple = (128, 128),
        include_depth: bool = False,
        distance_threshold: float = 0.01,
        pick_site_name: str = "pick_site",
        fruit_site_name: str = "fruit_site",
        **kwargs
    ):
        super().__init__(
            name="fruit_pick",
            robot_config=robot_config,
            scene_name=scene_name,
            include_image=include_image,
            image_size=image_size,
            include_depth=include_depth,
            **kwargs
        )
        
        self.distance_threshold = distance_threshold
        self.pick_site_name = pick_site_name
        self.fruit_site_name = fruit_site_name
        
        # 苹果与果柄的连接约束名称
        self.stem_connect_name = "apple_stem_connect"
        
        self._fruit_initial_pos = None
        self._desired_goal = None
        self._is_cut = False  # 苹果是否已被剪断
    
    def get_scene_path(self) -> Path:
        """获取场景XML文件路径"""
        return self.get_assets_path() / "scenes" / "fruitpick.xml"

    def reset(self) -> Dict[str, Any]:
        """
        重置任务
        
        Returns:
            info: 包含任务初始信息的字典
                - desired_goal: 期望目标位置 (pick_site位置)
                - fruit_pos: 水果中心位置
        """
        super().reset()
        
        # 从场景中解析位置
        fruit_pos, pick_pos = self._get_positions_from_scene()
        
        self._fruit_initial_pos = fruit_pos.copy()
        self._desired_goal = pick_pos.copy()
        self._is_cut = False  # 重置剪断状态

        return {
            "desired_goal": self._desired_goal.copy(),
            "fruit_pos": fruit_pos.copy(),
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
        
        # 优先使用 _desired_goal（通过 set_fruit_position 设置）
        # 然后才从 observation 获取，避免使用零向量
        if self._desired_goal is not None and np.linalg.norm(self._desired_goal) > 0.01:
            desired_goal = self._desired_goal
        else:
            desired_goal = observation.get("desired_goal", np.zeros(3, dtype=np.float32))
        
        # 构建info字典
        info = {
            "tcp_pos": observation.get("tcp_pos", np.zeros(3)),
            "fruit_pos": desired_goal,
        }
        
        reward = self.compute_reward(achieved_goal, desired_goal, info)
        
        # 判断是否成功
        terminated = self.is_success_fn(achieved_goal, desired_goal)
        
        # 如果任务成功且尚未剪断，自动触发剪断
        if terminated and not self._is_cut:
            self._is_cut = True
            info["apple_cut"] = True
        else:
            info["apple_cut"] = False
        
        # 更新步数
        self.current_step += 1
        
        info["is_success"] = terminated
        info["is_cut"] = self._is_cut
        
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
        1. 接近奖励 (r_close): 末端执行器靠近水果，使用指数衰减
        2. 成功奖励 (bonus): 成功采摘时的额外奖励
        
        Args:
            achieved_goal: 当前达到的目标（TCP位置）
            desired_goal: 期望目标（水果位置）
            info: 额外信息
        
        Returns:
            reward: 标量奖励值
        """
        # 获取末端执行器和水果位置
        tcp_pos = info.get("tcp_pos", achieved_goal)
        fruit_pos = desired_goal
        
        # 1. 接近奖励：末端执行器靠近水果（指数衰减）
        dist = np.linalg.norm(fruit_pos - tcp_pos)
        r_close = np.exp(-10 * dist)
        
        # 2. 成功奖励
        bonus = 10.0 if dist < self.distance_threshold else 0.0
        
        reward = r_close + bonus
        
        return float(reward)
    
    def is_success_fn(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray
    ) -> bool:
        """
        判断任务是否成功
        
        成功条件：末端执行器距离水果小于阈值
        
        Args:
            achieved_goal: 当前达到的目标（TCP位置）
            desired_goal: 期望目标（水果位置）
        
        Returns:
            success: 是否成功
        """
        dist = np.linalg.norm(achieved_goal - desired_goal)
        return bool(dist < self.distance_threshold)
    
    @property
    def is_cut(self) -> bool:
        """苹果是否已被剪断"""
        return self._is_cut
    
    def cut_apple(self, model, data) -> bool:
        """
        剪断苹果（禁用 connect 约束，使苹果掉落）
        
        Args:
            model: MuJoCo model 对象
            data: MuJoCo data 对象
        
        Returns:
            success: 是否成功剪断
        
        使用示例:
            if env.task.cut_apple(env.model, env.data):
                print("苹果已剪断!")
        
        后续扩展:
            可以结合剪刀开合动作判定，例如:
            if distance < threshold and scissors_closing:
                env.task.cut_apple(env.model, env.data)
        """
        if self._is_cut:
            return False  # 已经剪断，无需重复操作
        
        try:
            # 获取约束 ID
            eq_id = model.equality(self.stem_connect_name).id
            # 禁用约束 (eq_active 在 data 中，不是 model)
            data.eq_active[eq_id] = 0
            self._is_cut = True
            return True
        except Exception as e:
            print(f"Warning: 无法剪断苹果: {e}")
            return False
    
    def restore_apple(self, model, data) -> bool:
        """
        恢复苹果连接（重新启用约束）
        
        Args:
            model: MuJoCo model 对象
            data: MuJoCo data 对象
        
        Returns:
            success: 是否成功恢复
        """
        try:
            # 获取约束 ID
            eq_id = model.equality(self.stem_connect_name).id
            # 启用约束 (eq_active 在 data 中)
            data.eq_active[eq_id] = 1
            self._is_cut = False
            return True
        except Exception as e:
            print(f"Warning: 无法恢复苹果: {e}")
            return False
    
    def get_achieved_goal(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """
        从观测中提取achieved_goal
        
        对于采摘任务，achieved_goal是末端执行器的当前位置
        
        Args:
            observation: 环境观测
        
        Returns:
            achieved_goal: TCP当前位置 (x, y, z)
        """
        # 尝试从observation中获取TCP位置
        if "tcp_pos" in observation:
            return observation["tcp_pos"].copy()
        
        # 默认返回零向量
        return np.zeros(3, dtype=np.float32)
    
    def sample_goal(self) -> np.ndarray:
        """
        采样一个目标
        
        对于采摘任务，目标就是水果的位置（从场景中获取）
        
        Returns:
            goal: 目标位置 (x, y, z)
        """
        if self._desired_goal is not None:
            return self._desired_goal.copy()

        # 默认位置（场景中抓取点的大致位置）
        return np.array([0.65, 0.15, 0.85], dtype=np.float32)
    
    def set_pick_position(self, pos: np.ndarray):
        """
        设置抓取目标位置（由环境调用）
        
        Args:
            pos: 抓取目标位置
        """
        self._desired_goal = pos.copy()
    
    def set_fruit_position(self, fruit_pos: np.ndarray, pick_pos: np.ndarray = None):
        """
        设置水果位置和抓取目标位置
        
        Args:
            fruit_pos: 水果中心位置
            pick_pos: 抓取目标位置（可选，默认使用 fruit_pos）
        """
        if self._fruit_initial_pos is None:
            self._fruit_initial_pos = fruit_pos.copy()
        if pick_pos is not None:
            self._desired_goal = pick_pos.copy()
        else:
            self._desired_goal = fruit_pos.copy()

    def _get_positions_from_scene(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        从场景XML中解析水果中心和抓取目标位置
        
        Returns:
            (fruit_pos, pick_pos): 水果中心位置和抓取目标位置
        """
        scene_path = self.get_scene_path()
        default_fruit = np.array([0.65, 0.15, 0.8], dtype=np.float32)
        default_pick = np.array([0.65, 0.15, 0.85], dtype=np.float32)
        
        if not scene_path.exists():
            return default_fruit, default_pick

        try:
            tree = ET.parse(scene_path)
            root = tree.getroot()
        except Exception:
            return default_fruit, default_pick

        def _parse_vec(attr_val: Optional[str]) -> np.ndarray:
            if not attr_val:
                return np.zeros(3, dtype=np.float32)
            return np.array([float(x) for x in attr_val.split()], dtype=np.float32)

        # 查找苹果 body
        apple_body = root.find(".//body[@name='apple_body']")
        if apple_body is None:
            for body in root.findall(".//body"):
                if body.find(f".//site[@name='{self.fruit_site_name}']") is not None:
                    apple_body = body
                    break
        if apple_body is None:
            return default_fruit, default_pick

        body_pos = _parse_vec(apple_body.attrib.get("pos"))
        
        # 获取 fruit_site 位置
        fruit_site = apple_body.find(f".//site[@name='{self.fruit_site_name}']")
        if fruit_site is not None:
            fruit_pos = body_pos + _parse_vec(fruit_site.attrib.get("pos"))
        else:
            fruit_pos = body_pos.copy()
        
        # 获取 pick_site 位置
        pick_site = apple_body.find(f".//site[@name='{self.pick_site_name}']")
        if pick_site is not None:
            pick_pos = body_pos + _parse_vec(pick_site.attrib.get("pos"))
        else:
            # 没有 pick_site，使用 fruit_pos + 默认偏移
            pick_pos = fruit_pos.copy()
            pick_pos[2] += 0.05
        
        return fruit_pos, pick_pos
    
    def get_obs_space(self) -> spaces.Dict:
        """
        获取任务相关的观测空间
        
        Returns:
            obs_space: 任务相关的观测空间字典
        """
        obs_dict = {}
        
        # 水果位置
        obs_dict["fruit_pos"] = spaces.Box(
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
        
        Returns:
            action_space: 动作空间
        """
        controller_type = self.robot_config.controller_type
        if controller_type in ["cartesian_ik", "operational_space", "cartesian_impedance", "osc", "task_space"]:
            dim = 7  # 位姿（位置+四元数）
            if self.robot_config.gripper_name:
                dim += 1
        else:
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
        获取任务信息
        
        Returns:
            info: 任务信息字典
        """
        info = super().get_info()
        info.update({
            "distance_threshold": self.distance_threshold,
            "fruit_site_name": self.fruit_site_name,
            "hover_height": self.hover_height,
            "fruit_pos": self._fruit_initial_pos,
            "scene_name": self.scene_name,
            "scene_path": str(self.get_scene_path()),
        })
        return info
    
    def __repr__(self) -> str:
        return f"{self.name}(robot={self.robot_config.name}, scene={self.scene_name})"
