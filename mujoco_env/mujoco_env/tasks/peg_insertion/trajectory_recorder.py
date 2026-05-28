"""
轨迹记录器

在无头模式训练时记录机械臂关节角度，用于后续本地回放调试。

日期: 2026-01-12
"""

import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import numpy as np


class TrajectoryRecorder:
    """
    轨迹记录器
    
    在强化学习训练过程中记录机械臂的关节位置数据，
    支持保存为 .npz 或 .json 格式，便于后续本地可视化回放。
    
    使用示例:
        >>> recorder = TrajectoryRecorder(robot_name="aubo_i5", dof=6)
        >>> recorder.new_episode()
        >>> for step in range(100):
        ...     obs, reward, _, _, _ = env.step(action)
        ...     recorder.record(qpos=obs["qpos"])
        >>> recorder.save()
    
    Attributes:
        robot_name: 机器人名称
        dof: 自由度
        save_dir: 保存目录
        control_freq: 控制频率 (Hz)
    """
    
    def __init__(
        self,
        robot_name: str = "aubo_i5",
        dof: int = 6,
        save_dir: str = "./trajectories",
        control_freq: float = 20.0,
        auto_save_on_episode_end: bool = True
    ):
        """
        初始化轨迹记录器
        
        Args:
            robot_name: 机器人名称，用于元数据
            dof: 机器人自由度
            save_dir: 轨迹文件保存目录
            control_freq: 控制频率 (Hz)，用于计算时间戳
            auto_save_on_episode_end: 在 new_episode 时自动保存上一个 episode
        """
        self.robot_name = robot_name
        self.dof = dof
        self.save_dir = Path(save_dir)
        self.control_freq = control_freq
        self.auto_save_on_episode_end = auto_save_on_episode_end
        
        # 创建保存目录
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 当前 episode 数据缓冲区
        self._current_episode_id: int = 0
        self._step_count: int = 0
        self._episode_start_time: float = 0.0
        
        # 环境信息（用于回放时重建环境）
        self._env_info: Dict[str, Any] = {}
        
        # 数据存储列表（仅关节位置）
        self._joint_positions: List[np.ndarray] = []
        self._timestamps: List[float] = []
        
        # 已保存的 episode 计数
        self._saved_episode_count: int = 0
    
    def record(self, qpos: np.ndarray):
        """
        记录一帧关节位置数据
        
        Args:
            qpos: 关节位置, shape=(dof,)
        """
        # 验证 qpos
        qpos = np.asarray(qpos, dtype=np.float32)
        if qpos.shape[0] != self.dof:
            raise ValueError(f"Expected qpos of length {self.dof}, got {qpos.shape[0]}")
        
        self._joint_positions.append(qpos.copy())
        
        # 时间戳
        elapsed = self._step_count / self.control_freq
        self._timestamps.append(elapsed)
        
        self._step_count += 1
    
    def set_env_info(
        self,
        xml_path: str = None,
        scene_name: str = None,
        task_name: str = None,
        initial_qpos: np.ndarray = None,
        gripper_name: str = None,
        joint_names: list = None,
        **kwargs
    ):
        """
        设置环境信息，用于回放时重建相同环境
        
        Args:
            xml_path: MuJoCo XML 文件路径
            scene_name: 场景名称
            task_name: 任务名称
            initial_qpos: 初始关节位置
            gripper_name: 夹爪名称
            joint_names: 关节名称列表（用于回放时正确索引）
            **kwargs: 其他环境参数
        """
        if xml_path:
            self._env_info["xml_path"] = str(xml_path)
        if scene_name:
            self._env_info["scene_name"] = scene_name
        if task_name:
            self._env_info["task_name"] = task_name
        if initial_qpos is not None:
            # 确保 initial_qpos 是 numpy 数组，然后转换为 list
            if hasattr(initial_qpos, 'tolist'):
                self._env_info["initial_qpos"] = initial_qpos.tolist()
            else:
                self._env_info["initial_qpos"] = list(initial_qpos)
        if gripper_name:
            self._env_info["gripper_name"] = gripper_name
        if joint_names:
            self._env_info["joint_names"] = joint_names
        self._env_info.update(kwargs)
    
    def new_episode(self, episode_id: Optional[int] = None):
        """
        开始新的 episode
        
        如果当前有未保存的数据且 auto_save_on_episode_end=True，会自动保存。
        
        Args:
            episode_id: 指定 episode ID，如果为 None 则自动递增
        """
        # 自动保存上一个 episode
        if self.auto_save_on_episode_end and len(self._joint_positions) > 0:
            self.save()
        
        # 清空缓冲区
        self._clear_buffers()
        
        # 更新 episode ID
        if episode_id is not None:
            self._current_episode_id = episode_id
        else:
            self._current_episode_id = self._saved_episode_count
        
        self._episode_start_time = time.time()
    
    def save(
        self,
        filename: Optional[str] = None,
        format: str = "npz"
    ) -> Path:
        """
        保存当前 episode 的轨迹数据
        
        Args:
            filename: 文件名（不含扩展名），如果为 None 则自动命名并避免覆盖
            format: 保存格式，"npz" 或 "json"
            
        Returns:
            保存的文件路径
        """
        if len(self._joint_positions) == 0:
            raise ValueError("No data to save. Call record() first.")
        
        # 构建文件名，自动避免覆盖
        if filename is None:
            # 查找下一个可用的 episode ID
            episode_id = 0
            while True:
                candidate_path = self.save_dir / f"episode_{episode_id:04d}.{format}"
                if not candidate_path.exists():
                    break
                episode_id += 1
            filename = f"episode_{episode_id:04d}"
            self._current_episode_id = episode_id
        else:
            # 用户指定了文件名，检查是否存在
            candidate_path = self.save_dir / f"{filename}.{format}"
            if candidate_path.exists():
                print(f"[TrajectoryRecorder] Warning: 文件 {candidate_path} 将被覆盖")
        
        # 构建元数据
        metadata = {
            "robot_name": self.robot_name,
            "dof": self.dof,
            "control_freq": self.control_freq,
            "total_steps": len(self._joint_positions),
            "episode_id": self._current_episode_id,
            "created_at": datetime.now().isoformat(),
            "env_info": self._env_info,
        }
        
        # 转换为 numpy 数组
        data = {
            "joint_positions": np.array(self._joint_positions, dtype=np.float32),
            "timestamps": np.array(self._timestamps, dtype=np.float64),
        }
        
        # 保存
        if format == "npz":
            filepath = self.save_dir / f"{filename}.npz"
            np.savez_compressed(
                filepath,
                metadata=json.dumps(metadata),
                **data
            )
        elif format == "json":
            filepath = self.save_dir / f"{filename}.json"
            json_data = {
                "metadata": metadata,
                **{k: v.tolist() for k, v in data.items()}
            }
            with open(filepath, "w") as f:
                json.dump(json_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'npz' or 'json'.")
        
        self._saved_episode_count += 1
        print(f"[TrajectoryRecorder] Saved {len(self._joint_positions)} frames to {filepath}")
        
        return filepath
    
    def _clear_buffers(self):
        """清空所有数据缓冲区"""
        self._joint_positions.clear()
        self._timestamps.clear()
        self._step_count = 0
    
    def get_episode_count(self) -> int:
        """获取已保存的 episode 数量"""
        return self._saved_episode_count
    
    def get_current_frame_count(self) -> int:
        """获取当前 episode 已记录的帧数"""
        return len(self._joint_positions)
    
    @property
    def current_episode_id(self) -> int:
        """当前 episode ID"""
        return self._current_episode_id


# 便捷函数
def create_recorder(
    robot_name: str = "aubo_i5",
    save_dir: str = "./trajectories"
) -> TrajectoryRecorder:
    """
    创建轨迹记录器的便捷函数
    
    Args:
        robot_name: 机器人名称
        save_dir: 保存目录
        
    Returns:
        TrajectoryRecorder 实例
    """
    # 根据机器人名称自动获取 DOF
    dof_map = {
        "aubo_i5": 6,
        "franka_panda": 7,
        "franka_fr3": 7,
        "ur5e": 6,
        "diana_med": 7,
    }
    dof = dof_map.get(robot_name, 6)
    
    return TrajectoryRecorder(
        robot_name=robot_name,
        dof=dof,
        save_dir=save_dir
    )
