"""
Franka Panda PickCube 环境实例

从 demo_franka_panda_pick_cube.py 提取的环境创建函数
"""

from ..env_factory import make_sim_env
from mujoco_env.mujoco_env.tasks.pick_cube.franka_panda_config import FrankaPandaPickCubeConfig
from mujoco_env.mujoco_env.tasks.pick_cube.pick_cube import PickCubeTask


def make_franka_panda_pick_cube_env(**kwargs):
    """
    创建 Franka Panda PickCube 环境
    
    Args:
        **kwargs: 传递给 make_sim_env 的参数，包括：
            - max_episode_steps: 最大episode步数 (默认: 500)
            - render_mode: 渲染模式 (默认: None)
            - control_dt: 控制周期 (默认: 0.02)
            - site_name: 末端执行器site名称 (默认: "grip_site")
            - target_joint: 零空间目标关节配置
            - 其他控制器参数
    
    Returns:
        env: SimulationRobotEnv 实例
    """
    # 1. 创建机器人配置
    robot_config = FrankaPandaPickCubeConfig.get_config()
    
    # 2. 创建任务配置
    task = PickCubeTask(
        robot_config=robot_config,
        scene_name=kwargs.pop("scene_name", "cubes"),  # 使用抓取场景
        include_image=kwargs.pop("include_image", False),
        image_size=kwargs.pop("image_size", (128, 128)),
        include_depth=kwargs.pop("include_depth", False),
    )
    
    # 3. 确定末端执行器site名称
    ee_site_name = kwargs.pop("site_name", "grip_site" if robot_config.gripper_name else "attachment_site")
    
    # 4. 设置默认参数
    if "control_dt" not in kwargs:
        kwargs["control_dt"] = 0.02
    if "max_episode_steps" not in kwargs:
        kwargs["max_episode_steps"] = 500
    
    # 5. 创建环境
    return make_sim_env(
        task=task,
        render_mode=kwargs.pop("render_mode", None),
        site_name=ee_site_name,
        **kwargs
    )

