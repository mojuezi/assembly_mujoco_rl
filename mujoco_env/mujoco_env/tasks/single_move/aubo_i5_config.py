"""
Aubo i5机器人单次运动任务配置

作者: Liu Gang
日期: 2025-01-02
"""

from typing import Optional
from ...robot_config.aubo_i5 import AuboI5Robot


class AuboI5Config(AuboI5Robot):
    """Aubo i5机器人单次运动任务配置类"""
    
    @classmethod
    def get_config(cls) -> "AuboI5Config":
        """
        获取 single_move 任务专用的 Aubo i5 机器人配置

        这里的 cls 指的是调用该类方法的类本身。
        在本例中，cls 就是 AuboI5Config 类，它继承自 AuboI5Robot。
        通过 cls(...) 可以灵活地创建当前类的实例，即使被子类继承时也能返回子类实例。

        这是一个固定配置，专门为单次运动任务设计：
        - 使用 top_point2 底座（提供稳定的基础）
        - 不使用夹爪（单次运动不需要）
        - 关节位置控制模式
        - 20Hz 控制频率
        - 不使用六维力传感器
        """
        return cls(
            controller_type="cartesian_ik",  # 单次运动使用位置控制
            control_freq=20,                   # 标准控制频率
            mount_name="pedestal",             # 使用pedestal底座
            gripper_name="assemble_axle",      # 使用assemble_axle夹爪
            use_ft_sensor=False,               # 单次运动任务不需要力传感器
        )
    

