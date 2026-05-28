"""
Aubo i5机器人采摘任务配置

日期: 2026-01-15
"""

import numpy as np
from typing import Optional
from ...robot_config.aubo_i5 import AuboI5Robot


class AuboI5FruitPickConfig(AuboI5Robot):
    """Aubo i5机器人采摘任务配置类"""
    
    # 支持的夹爪（添加 umi_scissors）
    AVAILABLE_GRIPPERS = ["robotiq_2f85", "panda_hand", "robotiq_gripper", "umi_scissors"]
    
    # 覆盖底座位置：将 pedestal 放在原点
    MOUNT_POS = {
        "pedestal": [0, 0, 0.092],  # 原点位置，Z轴微调
        "top_point2": [0, 0, 0],
    }
    
    # 覆盖机器人安装偏移
    ROBOT_OFFSET = {
        "pedestal": [0, 0, 0.392],  # pedestal 高度偏移
        "top_point2": [0, 0, 0],
    }
    
    # 采摘任务初始关节位置
    INIT_QPOS = np.array([
        0.0,
        -15.0 / 180.0 * np.pi,
        100.0 / 180.0 * np.pi,
        25.0 / 180.0 * np.pi,
        90.0 / 180.0 * np.pi,   
        0.0
    ])
    
    @classmethod
    def get_config(cls) -> "AuboI5FruitPickConfig":
        """
        获取 fruit_pick 任务专用的 Aubo i5 机器人配置
        
        这是一个固定配置，专门为水果采摘任务设计：
        - 使用 pedestal 底座（有可视化模型）
        - 使用 umi_scissors 剪刀夹爪
        - 笛卡尔 IK 控制模式
        - 20Hz 控制频率
        """
        return cls(
            controller_type="cartesian_ik",
            control_freq=20,
            mount_name="pedestal",  # 使用 pedestal 底座（有视觉模型）
            gripper_name="umi_scissors",
            use_ft_sensor=False,
        )
