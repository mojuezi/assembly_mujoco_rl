"""
Aubo i5 AssembleHole 环境实例

从 demo_aubo_i5_assemble_hole.py 提取的环境创建函数
"""

from ..env_factory import make_env
from mujoco_env.mujoco_env.tasks.peg_insertion.aubo_i5_config import AuboI5Config
from mujoco_env.mujoco_env.tasks.peg_insertion.peg_insertion import PegInsertionTask
import pdb

def make_aubo_i5_assemble_hole_env(**kwargs):
    """
    创建 Aubo i5 AssembleHole 环境

    Args:
        **kwargs: 传递给 make_sim_env 的参数，包括：
            - scene_name, include_image, image_size, include_depth
            - control_dt, physics_dt, max_episode_steps, render_mode
            - record_trajectory: 是否启用轨迹录制
    Returns:
        env: SimulationRobotEnv 实例
    """
    # 1. 创建机器人配置
    robot_config = AuboI5Config.get_config()

    # 2. 确定末端执行器site名称
    ee_site_name = kwargs.pop("ee_site_name", "grip_site" if robot_config.gripper_name else "attachment_site")
    mode = kwargs.pop("mode", "sim")

    # 3. 创建任务配置
    task = PegInsertionTask(
        robot_config=robot_config,
        scene_name=kwargs.pop("scene_name", "assemble_hole"),
        include_image=kwargs.pop("include_image", False),
        image_size=kwargs.pop("image_size", (128, 128)),
        include_depth=kwargs.pop("include_depth", False),
        ee_site_name=ee_site_name,
        mode=mode, 
        hole_position_real=kwargs.pop("hole_position_real", None), 
        workspace_low_real = kwargs.pop("workspace_low_real", None), 
        workspace_high_real = kwargs.pop("workspace_high_real", None), 
    )

    # 3. 确定末端执行器site名称
    ee_site_name = kwargs.pop("ee_site_name", "grip_site" if robot_config.gripper_name else "attachment_site")

    # 4. 设置默认参数
    if "control_dt" not in kwargs:
        kwargs["control_dt"] = 0.02
    if "physics_dt" not in kwargs:
        kwargs["physics_dt"] = 0.002
    if "max_episode_steps" not in kwargs:
        kwargs["max_episode_steps"] = 500
    if "render_mode" not in kwargs:
        kwargs["render_mode"] = "rgb_array"
    # if "use_target_as_ctrl" not in kwargs:
    #     kwargs["use_target_as_ctrl"] = True
    if "ik_regularization" not in kwargs:
        kwargs["ik_regularization"] = 0.001
    if "ik_radius" not in kwargs:
        kwargs["ik_radius"] = 0.1

    # 5. 创建环境
    return make_env(
        task=task,
        mode=mode, 
        robot_ip=kwargs.pop("robot_ip", None), 
        render_mode=kwargs.pop("render_mode", "rgb_array"),
        ee_site_name=ee_site_name,
        **kwargs
    )


