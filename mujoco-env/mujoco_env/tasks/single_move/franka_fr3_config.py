"""
Franka FR3机器人单次运动任务配置

作者: Liu Gang
日期: 2026-01-03
"""

from typing import Optional
from ...robot_config.franka_fr3 import FrankaFR3Robot


class FrankaFR3Config(FrankaFR3Robot):
    """Franka FR3机器人单次运动任务配置类"""
    
    @classmethod
    def get_config(cls) -> "FrankaFR3Config":
        """
        获取 single_move 任务专用的 Franka FR3 机器人配置

        这是一个固定配置，专门为单次运动任务设计：
        - 使用 default 底座
        - 使用 franka_hand 夹爪
        - 关节位置控制模式
        - 20Hz 控制频率 (与Aubo示例一致，或者可以更高)
        - 不使用六维力传感器
        """
        return cls(
            controller_type="cartesian_ik",  # 默认使用笛卡尔IK控制，也可以在demo脚本中覆盖
            control_freq=20,                 # 控制频率
            mount_name="default",            # 使用default底座
            gripper_name="panda_hand",       # 使用panda_hand夹爪
            use_ft_sensor=False,             # 单次运动任务不需要力传感器
        )

