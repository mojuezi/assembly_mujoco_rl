"""
MuJoCo环境模块

提供基于Gymnasium的机器人仿真环境和真机环境
"""

import sys
import types

# 创建 mujoco_env.mujoco_env 命名空间包，指向当前模块
# 这样代码中的 mujoco_env.mujoco_env.xxx 导入可以正常工作
if 'mujoco_env.mujoco_env' not in sys.modules:
    mujoco_env_module = types.ModuleType('mujoco_env.mujoco_env')
    mujoco_env_module.__path__ = [__path__[0]]  # 使用当前包的路径
    sys.modules['mujoco_env.mujoco_env'] = mujoco_env_module

from .core.base_env import BaseRobotEnv
from .core.sim_env import SimulationRobotEnv
from .core.real_env import RealRobotEnv
from .tasks.base_task import (
    BaseTask,
    ObservationConfig,
)
from .robot_config.base import RobotConfig
from .envs.env_factory import make_sim_env, make_real_env

__version__ = "0.0.1"

__all__ = [
    "BaseRobotEnv",
    "SimulationRobotEnv",
    "RealRobotEnv",
    "BaseTask",
    "RobotConfig",
    "ObservationConfig",
    "make_sim_env",
    "make_real_env",
]

