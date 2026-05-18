from typing import Optional, List
from pathlib import Path

class RobotConfig:
    """机器人配置数据类"""
    
    def __init__(
        self,
        name: str,
        robot_type: str,
        dof: int,
        control_freq: int = 20,
        controller_type: str = "joint_position",
        mount_name: Optional[str] = None,
        mount_pos: Optional[list] = None,
        robot_offset: Optional[list] = None,  # 新增：机器人相对于底座的偏移
        gripper_name: Optional[str] = None,
        use_ft_sensor: Optional[str] = None,
        joint_names: Optional[List[str]] = None,  # 新增：关节名称列表
    ):
        self.name = name
        self.robot_type = robot_type
        self.dof = dof
        self.control_freq = control_freq
        self.controller_type = controller_type
        self.mount_name = mount_name
        self.mount_pos = mount_pos or [0, 0, 0]
        self.robot_offset = robot_offset or [0, 0, 0]
        self.gripper_name = gripper_name
        self.use_ft_sensor = use_ft_sensor
        self.joint_names = joint_names  # 用于 _get_obs 中获取正确的关节索引

    @property
    def action_dim(self) -> int:
        """获取动作空间维度 (包含夹爪)"""
        dim = self.dof
        if self.gripper_name:
            # 假设夹爪动作是1维（0-1 或 -1-1）
            # 对于 Robotiq 2F-85 这种简单夹爪，通常是1维控制
            dim += 1
        return dim
