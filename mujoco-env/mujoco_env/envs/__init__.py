"""
环境模块

提供统一的环境创建接口

作者: Liu Gang
日期: 2025-12-20
"""

from .env_factory import (
    make_env,
    make_sim_env,
    make_real_env,
)
from .env_instance import register_envs


__all__ = [
    # 工厂函数（推荐使用）
    "make_env",
    "make_sim_env",
    "make_real_env",
    "register_envs",
]

