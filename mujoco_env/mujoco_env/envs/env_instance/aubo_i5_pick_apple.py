"""
Aubo i5 FruitPick 环境实例

提供创建 Aubo i5 水果采摘仿真环境的工厂函数
"""

from ..env_factory import make_sim_env
from mujoco_env.mujoco_env.tasks.fruit_pick.aubo_i5_config import AuboI5FruitPickConfig
from mujoco_env.mujoco_env.tasks.fruit_pick.fruit_pick import FruitPickTask


def make_aubo_i5_fruit_pick_env(**kwargs):
    """
    创建 Aubo i5 FruitPick 环境
    
    Args:
        **kwargs: 传递给 make_sim_env 的参数，包括：
            - max_episode_steps: 最大episode步数 (默认: 1000)
            - render_mode: 渲染模式 (默认: "rgb_array")
            - control_dt: 控制周期 (默认: 0.02)
            - physics_dt: 物理仿真周期 (默认: 0.002)
            - ee_site_name: 末端执行器site名称 (默认: umi_scissors -> "action_point")
            - use_target_as_ctrl: 是否使用目标作为控制 (默认: True)
            - ik_regularization: IK正则化系数 (默认: 0.001)
            - ik_radius: IK半径 (默认: 0.1)
            - distance_threshold: 成功距离阈值 (默认: task 内部值)
    
    Returns:
        env: SimulationRobotEnv 实例
    """
    # 1. 创建机器人配置
    robot_config = AuboI5FruitPickConfig.get_config()
    
    # 2. 创建任务配置
    task_kwargs = {}
    distance_threshold = kwargs.pop("distance_threshold", None)
    if distance_threshold is not None:
        task_kwargs["distance_threshold"] = distance_threshold

    task = FruitPickTask(
        robot_config=robot_config,
        scene_name=kwargs.pop("scene_name", "fruitpick"),
        include_image=kwargs.pop("include_image", False),
        image_size=kwargs.pop("image_size", (128, 128)),
        include_depth=kwargs.pop("include_depth", False),
        **task_kwargs,
    )
    
    # 3. 确定末端执行器site名称
    # 采摘默认使用 action_point，其他夹爪回退到 pinch
    if robot_config.gripper_name == "umi_scissors":
        default_ee_site = "pinch"
    elif robot_config.gripper_name == "robotiq_2f85":
        default_ee_site = "pinch"
    else:
        default_ee_site = "attachment_site"
    ee_site_name = kwargs.pop("ee_site_name", default_ee_site)
    
    # 4. 设置默认参数
    if "control_dt" not in kwargs:
        kwargs["control_dt"] = 0.02
    if "physics_dt" not in kwargs:
        kwargs["physics_dt"] = 0.002
    if "max_episode_steps" not in kwargs:
        kwargs["max_episode_steps"] = 1000
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
