"""
测试基础环境类

运行: pytest mujoco_env/tests/test_base_env.py -v
或在conda serl环境下: conda run -n serl pytest tests/test_base_env.py -v
"""

import pytest
import numpy as np
from gymnasium import spaces
# 基础环境类从 core.base_env 导入
from mujoco_env.mujoco_env.core.base_env import BaseRobotEnv
# 配置类从 tasks.base_task 导入
from mujoco_env.mujoco_env.tasks.base_task import RobotConfig, ObservationConfig


class DummyEnv(BaseRobotEnv):
    """
    用于测试的虚拟环境
    
    实现 BaseRobotEnv 的抽象方法，用于测试基础环境类的功能
    不依赖 MuJoCo，可以快速验证环境接口的正确性
    """
    
    def __init__(self):
        """初始化虚拟环境，设置观测空间和动作空间"""
        super().__init__()
        # 定义观测空间：4维状态向量
        self.observation_space = spaces.Dict({
            "state": spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        })
        # 定义动作空间：2维动作向量
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    
    def step(self, action):
        """
        执行一步动作（实现抽象方法）
        
        Args:
            action: 动作数组（虽然不使用，但保持接口一致性）
            
        Returns:
            obs, reward, terminated, truncated, info: Gymnasium标准返回值
        """
        self._elapsed_steps += 1
        obs = {"state": np.zeros(4, dtype=np.float32)}
        reward = 0.0
        terminated = False
        truncated = False
        info = self._get_info()
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self):
        """获取当前观测（实现抽象方法）"""
        return {"state": np.zeros(4, dtype=np.float32)}
    
    def _compute_reward(self):
        """计算奖励（实现抽象方法）"""
        return 0.0
    
    def render(self):
        """渲染环境（实现抽象方法）"""
        return None


def test_base_env_creation():
    """
    测试环境创建
    
    验证：
    - 环境实例可以正常创建
    - 观测空间和动作空间正确设置
    """
    env = DummyEnv()
    assert env is not None
    assert isinstance(env.observation_space, spaces.Dict)
    assert isinstance(env.action_space, spaces.Box)
    print("✓ 环境创建测试通过")


def test_base_env_reset():
    """
    测试环境重置
    
    验证：
    - reset() 返回正确的观测格式
    - info 字典包含必要的元信息
    - 重置后步数归零
    """
    env = DummyEnv()
    obs, info = env.reset()
    
    assert "state" in obs
    assert obs["state"].shape == (4,)
    assert isinstance(info, dict)
    assert "elapsed_steps" in info
    assert info["elapsed_steps"] == 0
    print("✓ 环境重置测试通过")


def test_base_env_step():
    """
    测试环境步进
    
    验证：
    - step() 返回格式符合 Gymnasium 标准
    - 观测在观测空间内
    - 返回值类型正确（reward为float，terminated/truncated为bool）
    """
    env = DummyEnv()
    env.reset()
    
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    assert obs in env.observation_space
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    print("✓ 环境步进测试通过")


def test_robot_config():
    """
    测试机器人配置类
    
    验证：
    - RobotConfig 可以正确创建
    - 配置属性可以正确访问
    """
    config = RobotConfig(
        name="test_robot",
        robot_type="franka",
        dof=7,
        control_freq=20,
    )
    
    assert config.name == "test_robot"
    assert config.dof == 7
    assert config.control_freq == 20
    print("✓ 机器人配置测试通过")


def test_observation_config():
    """
    测试观测配置类
    
    验证：
    - ObservationConfig 可以正确创建
    - 观测相关配置属性可以正确访问
    """
    config = ObservationConfig(
        include_image=True,
        image_size=(128, 128),
        include_proprioception=True,
    )
    
    assert config.include_image is True
    assert config.image_size == (128, 128)
    assert config.include_proprioception is True
    print("✓ 观测配置测试通过")


def test_episode_counting():
    """
    测试Episode计数功能
    
    验证：
    - 每次调用 reset() 后 episode 计数递增
    - info 字典中包含正确的 episode 编号
    """
    env = DummyEnv()
    
    # 第一个episode
    obs, info = env.reset()
    assert info["episode"] == 1
    
    # 第二个episode
    obs, info = env.reset()
    assert info["episode"] == 2
    
    print("✓ Episode计数测试通过")


def test_step_counting():
    """
    测试步数计数功能
    
    验证：
    - 每执行一步，elapsed_steps 正确递增
    - 重置后步数归零
    - info 字典中包含正确的步数信息
    """
    env = DummyEnv()
    env.reset()
    
    # 执行几步，验证步数递增
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert info["elapsed_steps"] == i + 1
    
    # 重置后步数归零
    obs, info = env.reset()
    assert info["elapsed_steps"] == 0
    
    print("✓ 步数计数测试通过")


if __name__ == "__main__":
    """
    直接运行测试脚本时的入口点
    
    按顺序执行所有测试函数，用于快速验证基础环境类的功能
    也可以使用 pytest 运行以获得更详细的测试报告
    """
    # 运行所有测试
    print("=" * 60)
    print("运行BaseRobotEnv单元测试")
    print("=" * 60)
    
    test_base_env_creation()
    test_base_env_reset()
    test_base_env_step()
    test_robot_config()
    test_observation_config()
    test_episode_counting()
    test_step_counting()
    
    print("=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)

