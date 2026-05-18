"""
Franka FR3 SimpleMove 环境实例

从 demo_franka_fr3_simple_move.py 提取的环境创建函数
"""

from ..env_factory import make_sim_env
from mujoco_env.mujoco_env.tasks.single_move.franka_fr3_config import FrankaFR3Config
from mujoco_env.mujoco_env.tasks.single_move.single_move import SingleMoveTask


def make_franka_fr3_simple_move_env(mode='moveL', **kwargs):
    """
    创建 Franka FR3 SimpleMove 环境
    
    Args:
        mode: 运动模式 ('moveL' 或 'moveJ')
        **kwargs: 传递给 make_sim_env 的参数，包括：
            - max_episode_steps: 最大episode步数 (默认: 10000)
            - render_mode: 渲染模式 (默认: "rgb_array")
            - control_dt: 控制周期 (根据mode自动设置)
            - ee_site_name: 末端执行器site名称 (默认: "grip_site")
            - use_target_as_ctrl: 是否使用目标作为控制 (默认: True)
            - ik_regularization: IK正则化系数 (根据mode自动设置)
            - ik_radius: IK半径 (根据mode自动设置)
    
    Returns:
        env: SimulationRobotEnv 实例
    """
    # 1. 根据模式设置控制器类型和频率
    if mode == 'moveL':
        controller_type = "cartesian_ik"
        control_freq = kwargs.pop("control_freq", 50)
    else:  # moveJ
        controller_type = "joint_position"
        control_freq = kwargs.pop("control_freq", 50)
    
    # 2. 创建机器人配置
    robot_config = FrankaFR3Config(
        controller_type=controller_type,
        control_freq=control_freq,
        mount_name=kwargs.pop("mount_name", "default"),
        gripper_name=kwargs.pop("gripper_name", "panda_hand"),
        use_ft_sensor=kwargs.pop("use_ft_sensor", False),
    )
    
    # 3. 创建任务配置
    task = SingleMoveTask(
        robot_config=robot_config,
        scene_name=kwargs.pop("scene_name", "default"),
        include_image=kwargs.pop("include_image", False),
        image_size=kwargs.pop("image_size", (128, 128)),
        include_depth=kwargs.pop("include_depth", False),
    )
    
    # 4. 确定末端执行器site名称
    ee_site_name = kwargs.pop("ee_site_name", "grip_site" if robot_config.gripper_name else "attachment_site")
    
    # 5. 设置默认参数
    if "control_dt" not in kwargs:
        kwargs["control_dt"] = 1.0 / control_freq
    if "max_episode_steps" not in kwargs:
        kwargs["max_episode_steps"] = 10000
    if "render_mode" not in kwargs:
        kwargs["render_mode"] = "rgb_array"
    if "use_target_as_ctrl" not in kwargs:
        kwargs["use_target_as_ctrl"] = True
    if "ik_regularization" not in kwargs:
        kwargs["ik_regularization"] = 1e-4 if mode == 'moveL' else 0.01
    if "ik_radius" not in kwargs:
        kwargs["ik_radius"] = 0.05 if mode == 'moveL' else 0.2
    
    # 6. 创建环境
    return make_sim_env(
        task=task,
        ee_site_name=ee_site_name,
        **kwargs
    )

