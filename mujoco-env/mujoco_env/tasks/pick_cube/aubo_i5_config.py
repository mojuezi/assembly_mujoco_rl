"""
Aubo i5机器人抓取任务配置

作者: Liu Gang
日期: 2025-12-20
"""

from typing import Optional
from ...robot_config.aubo_i5 import AuboI5Robot


class AuboI5Config(AuboI5Robot):
    """Aubo i5机器人抓取任务配置类"""
    
    @classmethod
    def get_config(cls) -> "AuboI5Config":
        """
        获取 pick_cube 任务专用的 Aubo i5 机器人配置
        
        这是一个固定配置，专门为抓取方块任务设计：
        - 使用 pedestal 底座（适合抓取任务的高度）
        - 使用 robotiq_gripper 夹爪（适合抓取小物体）
        - 关节位置控制模式
        - 20Hz 控制频率
        - 不使用六维力传感器
        """
        return cls(
            controller_type="cartesian_ik",    # 抓取任务使用笛卡尔IK控制
            control_freq=20,                     # 标准控制频率
            mount_name="pedestal",              # 适合抓取任务的台座高度
            gripper_name="robotiq_gripper",     # 适合小物体抓取的夹爪
            use_ft_sensor="KWR75B",                # 抓取任务暂时不需要力传感器，需要时设为字符串如 "KWR75B"
        )
    

