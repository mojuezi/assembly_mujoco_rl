"""
Core module for mujoco_env

Provides base environment classes following Gymnasium interface.
"""

from .base_env import BaseRobotEnv
from .sim_env import SimulationRobotEnv
from .real_env import RealRobotEnv

__all__ = [
    "BaseRobotEnv",
    "SimulationRobotEnv",
    "RealRobotEnv",
]

