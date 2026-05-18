"""
Aubo i5 PickCube 环境实例

从 demo_aubo_i5_pick_cube.py 提取的环境创建函数
"""

from ..env_factory import make_sim_env
from mujoco_env.mujoco_env.tasks.pick_cube.aubo_i5_config import AuboI5Config
from mujoco_env.mujoco_env.tasks.pick_cube.pick_cube import PickCubeTask


def make_aubo_i5_pick_cube_env(**kwargs):
    """
    创建 Aubo i5 PickCube 环境
    
    Args:
        **kwargs: 传递给 make_sim_env 的参数，包括：
            - max_episode_steps: 最大episode步数 (默认: 500)
            - render_mode: 渲染模式 (默认: "rgb_array")
            - control_dt: 控制周期 (默认: 0.02)
            - physics_dt: 物理仿真周期 (默认: 0.002)
            - ee_site_name: 末端执行器site名称 (默认: "grip_site")
            - use_target_as_ctrl: 是否使用目标作为控制 (默认: True)
            - ik_regularization: IK正则化系数 (默认: 0.001)
            - ik_radius: IK半径 (默认: 0.1)
    
    Returns:
        env: SimulationRobotEnv 实例
    """
    # 1. 创建机器人配置
    robot_config = AuboI5Config.get_config()
    
    # 2. 创建任务配置
    task = PickCubeTask(
        robot_config=robot_config,
        scene_name=kwargs.pop("scene_name", "cubes"),  # 使用抓取场景
        include_image=kwargs.pop("include_image", False),
        image_size=kwargs.pop("image_size", (128, 128)),
        include_depth=kwargs.pop("include_depth", False),
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
    if "use_target_as_ctrl" not in kwargs:
        kwargs["use_target_as_ctrl"] = True
    if "ik_regularization" not in kwargs:
        kwargs["ik_regularization"] = 0.001
    if "ik_radius" not in kwargs:
        kwargs["ik_radius"] = 0.1
    
    # 5. 创建环境
    return make_sim_env(
        task=task,
        render_mode=kwargs.pop("render_mode", "rgb_array"),
        ee_site_name=ee_site_name,
        **kwargs
    )

