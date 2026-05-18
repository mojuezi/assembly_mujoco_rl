"""
Aubo i5机器人装配任务配置

作者: Liu Gang
日期: 2026-01-07
"""

from typing import Optional
from ...robot_config.aubo_i5 import AuboI5Robot


class AuboI5Config(AuboI5Robot):
    """Aubo i5机器人装配任务配置类"""

    @classmethod
    def get_config(cls) -> "AuboI5Config":
        """
        获取 peg_insertion 任务专用的 Aubo i5 机器人配置

        这是一个固定配置，专门为装配任务设计：
        - 使用 pedestal 底座（适合装配任务的高度）
        - 使用 robotiq_gripper 夹爪（适合精密操作）
        - 笛卡尔IK控制模式
        - 20Hz 控制频率
        - 使用六维力传感器
        """
        return cls(
            controller_type="admittance",    # 装配任务使用笛卡尔IK控制
            control_freq=20,                     # 标准控制频率
            mount_name="pedestal_new",              # 适合装配任务的台座高度
            gripper_name="assemble_axle",       # 使用装配轴（与simple_move脚本一致）
            use_ft_sensor="KWR75B",                # 装配任务需要力传感器
        )



