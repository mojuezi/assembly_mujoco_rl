"""
Franka FR3机器人抓取方块任务配置

作者: Liu Gang
日期: 2026-01-03
"""

from typing import Optional
from ...robot_config.franka_panda import FrankaPandaRobot


class FrankaPandaPickCubeConfig(FrankaPandaRobot):
    """Franka Panda机器人抓取方块任务配置类"""
    
    @classmethod
    def get_config(cls) -> "FrankaPandaPickCubeConfig":
        """
        获取 pick_cube 任务专用的 Franka Panda 机器人配置

        这是一个固定配置，专门为抓取方块任务设计：
        - 使用 franka_panda 机器人 (7自由度)
        - 使用 panda_hand 夹爪
        - 关节位置控制模式
        - 20Hz 控制频率
        - 底座：由于 grasping 场景中已经定义了桌子，这里使用默认位置 (0,0,0) 或根据桌子调整
          注意：grasping.xml 中 table 的位置是 (1.0, 0.3, 0.0)
          机器人通常安装在 (0,0,0) 或桌子上。
          如果使用 pedestal，它可能会与桌子冲突。
          我们这里使用 "default" 底座 (无几何体)，假设机器人安装在地板上
        """
        return cls(
            controller_type="operational_space",  # 使用操作空间控制
            control_freq=20,                   # 控制频率
            mount_name="pedestal",              # 使用默认底座 (无)
            gripper_name="panda_hand",         # 使用 panda_hand 夹爪
            use_ft_sensor=False,               # 不需要力传感器
        )

