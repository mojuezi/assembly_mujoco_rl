"""
环境实例模块

提供预定义的环境创建函数，用于快速创建常用的机器人环境。
这些函数从示例文件中提取，可以直接用于强化学习训练。
"""

__all__ = [
    
    "register_envs",
]

# 软导入可选/待修复环境
try:
    from .aubo_i5_pick_apple import make_aubo_i5_fruit_pick_env
    __all__.append("make_aubo_i5_fruit_pick_env")
except ImportError:
    pass

try:
    from .franka_panda_pick_cube import make_franka_panda_pick_cube_env
    __all__.append("make_franka_panda_pick_cube_env")
except ImportError:
    pass

try:
    from .franka_fr3_simple_move import make_franka_fr3_simple_move_env
    __all__.append("make_franka_fr3_simple_move_env")
except ImportError:
    pass

try:
    from .aubo_i5_simple_move import make_aubo_i5_simple_move_env
    __all__.append("make_aubo_i5_simple_move_env")
except ImportError:
    pass

try:
    from .aubo_i5_pick_cube import make_aubo_i5_pick_cube_env
    __all__.append("make_aubo_i5_pick_cube_env")
except ImportError:
    pass

try:
    from .aubo_i5_assemble_hole import make_aubo_i5_assemble_hole_env
    __all__.append("make_aubo_i5_assemble_hole_env")
except ImportError:
    pass


def register_envs():
    """
    将环境注册到 Gymnasium
    
    注册后可以使用 gymnasium.make() 创建环境
    
    Examples:
        >>> import gymnasium as gym
        >>> from mujoco_env.mujoco_env.envs.env_instance import register_envs
        >>> 
        >>> register_envs()
        >>> env = gym.make("FrankaPandaPickCube-v0")
    """
    try:
        import gymnasium as gym
    except ImportError:
        print("Warning: gymnasium not installed, skipping environment registration")
        return
    
    # 定义要注册的环境
    env_configs = [
        # Franka Panda PickCube 环境
        {
            "id": "FrankaPandaPickCube-v0",
            "entry_point": "mujoco_env.mujoco_env.envs.env_instance:make_franka_panda_pick_cube_env",
        },
        # PandaPickCube-v0 作为 FrankaPandaPickCube-v0 的别名
        {
            "id": "PandaPickCube-v0",
            "entry_point": "mujoco_env.mujoco_env.envs.env_instance:make_franka_panda_pick_cube_env",
        },
        # Franka Panda PegInsertion 环境 (使用通用工厂函数)
        {
            "id": "FrankaPandaPegInsertion-v0",
            "entry_point": "mujoco_env.mujoco_env.envs.env_factory:_make_gym_env",
            "kwargs": {
                "robot_name": "franka_panda",
                "task_name": "peg_insertion",
            }
        },
        # Aubo i5 PickCube 环境
        {
            "id": "AuboI5PickCube-v0",
            "entry_point": "mujoco_env.mujoco_env.envs.env_instance:make_aubo_i5_pick_cube_env",
        },
    ]
    
    # 注册环境
    for config in env_configs:
        try:
            gym.register(**config)
        except gym.error.Error:
            # 环境已经注册过
            pass


