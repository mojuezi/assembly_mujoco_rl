# 环境实例模块使用指南

本目录用于存放预定义的环境创建函数，这些函数封装了常用的机器人环境配置，可以直接用于强化学习训练和演示。

## 目录结构

```
env_instance/
├── README.md                    # 本文档
├── __init__.py                  # 模块初始化，包含 register_envs 函数
├── franka_panda_pick_cube.py    # Franka Panda 抓取方块环境
├── franka_fr3_simple_move.py    # Franka FR3 简单运动环境
├── aubo_i5_simple_move.py       # Aubo i5 简单运动环境
└── aubo_i5_pick_cube.py         # Aubo i5 抓取方块环境
```

## 如何添加新的环境实例

### 步骤 1: 创建环境实例文件

在 `env_instance/` 目录下创建一个新的 Python 文件，例如 `my_robot_my_task.py`。

**文件模板：**

```python
"""
[机器人名称] [任务名称] 环境实例

从 demo_xxx.py 提取的环境创建函数
"""

from ..env_factory import make_sim_env
from mujoco_env.mujoco_env.tasks.[task_name].[robot_name]_config import [RobotConfigClass]
from mujoco_env.mujoco_env.tasks.[task_name].[task_name] import [TaskClass]


def make_[robot_name]_[task_name]_env(**kwargs):
    """
    创建 [机器人名称] [任务名称] 环境
    
    Args:
        **kwargs: 传递给 make_sim_env 的参数，包括：
            - max_episode_steps: 最大episode步数 (默认: 500)
            - render_mode: 渲染模式 (默认: None)
            - control_dt: 控制周期 (默认: 0.02)
            - scene_name: 场景名称
            - include_image: 是否包含图像观测
            - image_size: 图像尺寸
            - include_depth: 是否包含深度图
            - 其他控制器参数
    
    Returns:
        env: SimulationRobotEnv 实例
    """
    # 1. 创建机器人配置
    robot_config = [RobotConfigClass].get_config()
    # 或者使用自定义配置：
    # robot_config = [RobotConfigClass](
    #     controller_type="operational_space",
    #     control_freq=20,
    #     mount_name="pedestal",
    #     gripper_name="panda_hand",
    #     use_ft_sensor=False,
    # )
    
    # 2. 创建任务配置
    task = [TaskClass](
        robot_config=robot_config,
        scene_name=kwargs.pop("scene_name", "default"),
        include_image=kwargs.pop("include_image", False),
        image_size=kwargs.pop("image_size", (128, 128)),
        include_depth=kwargs.pop("include_depth", False),
    )
    
    # 3. 确定末端执行器site名称
    ee_site_name = kwargs.pop("ee_site_name", "grip_site" if robot_config.gripper_name else "attachment_site")
    
    # 4. 设置默认参数
    if "control_dt" not in kwargs:
        kwargs["control_dt"] = 0.02
    if "max_episode_steps" not in kwargs:
        kwargs["max_episode_steps"] = 500
    
    # 5. 创建环境
    return make_sim_env(
        task=task,
        render_mode=kwargs.pop("render_mode", None),
        ee_site_name=ee_site_name,
        **kwargs
    )
```

### 步骤 2: 在 `__init__.py` 中导入新函数

编辑 `env_instance/__init__.py`，添加导入语句：

```python
from .my_robot_my_task import make_my_robot_my_task_env

__all__ = [
    # ... 其他函数
    "make_my_robot_my_task_env",
    "register_envs",
]
```

### 步骤 3: 在 `register_envs` 函数中注册环境

编辑 `env_instance/__init__.py` 中的 `register_envs` 函数，添加环境注册配置：

```python
def register_envs():
    # ...
    env_configs = [
        # ... 其他环境配置
        {
            "id": "MyRobotMyTask-v0",  # 环境ID，遵循 Gymnasium 命名规范
            "entry_point": "mujoco_env.mujoco_env.envs.env_instance:make_my_robot_my_task_env",
            # 可选：如果需要固定参数，可以添加 kwargs
            # "kwargs": {
            #     "scene_name": "default",
            #     "max_episode_steps": 1000,
            # }
        },
    ]
    # ...
```

**环境ID命名规范：**
- 格式：`[RobotName][TaskName]-v[版本号]`
- 示例：`FrankaPandaPickCube-v0`, `AuboI5SimpleMove-v0`
- 版本号从 `v0` 开始，如果环境有重大变更，可以增加版本号

## 使用方式

### 方式 1: 直接调用环境创建函数（推荐）

```python
from mujoco_env.mujoco_env.envs.env_instance import make_franka_panda_pick_cube_env

# 创建环境
env = make_franka_panda_pick_cube_env(
    scene_name="cubes",
    render_mode="human",
    max_episode_steps=1000,
    control_dt=0.02,
)

# 使用环境
obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

### 方式 2: 通过 Gymnasium 注册使用

```python
import gymnasium as gym
from mujoco_env.mujoco_env.envs.env_instance import register_envs

# 注册环境
register_envs()

# 使用 gym.make() 创建环境
env = gym.make("PandaPickCube-v0")

# 或者使用完整名称
env = gym.make("FrankaPandaPickCube-v0")

# 使用环境
obs, info = env.reset()
# ... 后续代码
```

### 方式 3: 在训练脚本中使用

```python
# 在训练脚本中（如 async_sac_state_sim.py）
from mujoco_env.mujoco_env.envs.env_instance import register_envs
import gymnasium as gym

# 注册环境
register_envs()

# 创建环境
env = gym.make("PandaPickCube-v0")
```

## 完整示例

### 示例 1: 创建新的 PickCube 环境

假设我们要为 `UR5e` 机器人创建一个 `PickCube` 环境：

**1. 创建文件 `ur5e_pick_cube.py`：**

```python
"""
UR5e PickCube 环境实例
"""

from ..env_factory import make_sim_env
from mujoco_env.mujoco_env.tasks.pick_cube.ur5e_config import UR5ePickCubeConfig
from mujoco_env.mujoco_env.tasks.pick_cube.pick_cube import PickCubeTask


def make_ur5e_pick_cube_env(**kwargs):
    """
    创建 UR5e PickCube 环境
    """
    robot_config = UR5ePickCubeConfig.get_config()
    
    task = PickCubeTask(
        robot_config=robot_config,
        scene_name=kwargs.pop("scene_name", "cubes"),
        include_image=kwargs.pop("include_image", False),
        image_size=kwargs.pop("image_size", (128, 128)),
        include_depth=kwargs.pop("include_depth", False),
    )
    
    ee_site_name = kwargs.pop("ee_site_name", "grip_site" if robot_config.gripper_name else "attachment_site")
    
    if "control_dt" not in kwargs:
        kwargs["control_dt"] = 0.02
    if "max_episode_steps" not in kwargs:
        kwargs["max_episode_steps"] = 500
    
    return make_sim_env(
        task=task,
        render_mode=kwargs.pop("render_mode", None),
        ee_site_name=ee_site_name,
        **kwargs
    )
```

**2. 更新 `__init__.py`：**

```python
from .ur5e_pick_cube import make_ur5e_pick_cube_env

__all__ = [
    # ... 其他函数
    "make_ur5e_pick_cube_env",
    "register_envs",
]

def register_envs():
    # ...
    env_configs = [
        # ... 其他配置
        {
            "id": "UR5ePickCube-v0",
            "entry_point": "mujoco_env.mujoco_env.envs.env_instance:make_ur5e_pick_cube_env",
        },
    ]
    # ...
```

**3. 使用新环境：**

```python
from mujoco_env.mujoco_env.envs.env_instance import make_ur5e_pick_cube_env

env = make_ur5e_pick_cube_env(render_mode="human")
```

或者：

```python
import gymnasium as gym
from mujoco_env.mujoco_env.envs.env_instance import register_envs

register_envs()
env = gym.make("UR5ePickCube-v0")
```

### 示例 2: 创建带参数的环境

如果环境需要特定参数，可以在注册时指定：

```python
{
    "id": "MyRobotMyTask-v0",
    "entry_point": "mujoco_env.mujoco_env.envs.env_instance:make_my_robot_my_task_env",
    "kwargs": {
        "scene_name": "custom_scene",
        "max_episode_steps": 1000,
        "control_dt": 0.01,
    }
}
```

## 参数说明

### 通用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_episode_steps` | int | 500 | 最大episode步数 |
| `render_mode` | str | None | 渲染模式：`"human"`, `"rgb_array"`, `"depth"`, `None` |
| `control_dt` | float | 0.02 | 控制周期（秒） |
| `scene_name` | str | 根据任务 | 场景名称 |
| `include_image` | bool | False | 是否包含图像观测 |
| `image_size` | tuple | (128, 128) | 图像尺寸 |
| `include_depth` | bool | False | 是否包含深度图 |
| `ee_site_name` | str | 自动检测 | 末端执行器site名称 |

### 控制器特定参数

根据使用的控制器类型，可能需要额外的参数：

- **OperationalSpaceController**: `site_name`, `target_joint`, `pos_gains`, `ori_gains`, `damping_ratio`, `nullspace_stiffness`
- **CartesianIKController**: `ee_site_name`, `use_target_as_ctrl`, `ik_regularization`, `ik_radius`
- **JointPositionController**: `use_target_as_ctrl`

## 最佳实践

1. **命名规范**：
   - 文件名使用小写字母和下划线：`robot_name_task_name.py`
   - 函数名使用小写字母和下划线：`make_robot_name_task_name_env`
   - 环境ID使用驼峰命名：`RobotNameTaskName-v0`

2. **参数处理**：
   - 使用 `kwargs.pop()` 来提取任务特定参数，避免传递给 `make_sim_env`
   - 为常用参数设置合理的默认值
   - 在文档字符串中说明所有可用参数

3. **错误处理**：
   - 确保导入的配置类和任务类存在
   - 处理可能的配置错误

4. **文档**：
   - 在文件头部说明环境用途
   - 在函数文档字符串中详细说明参数和返回值
   - 提供使用示例

## 常见问题

### Q: 如何查看已注册的环境？

```python
import gymnasium as gym
from mujoco_env.mujoco_env.envs.env_instance import register_envs

register_envs()
print(gym.envs.registry.all())
```

### Q: 环境注册失败怎么办？

- 检查 `entry_point` 路径是否正确
- 确保函数名与导入路径匹配
- 检查是否有循环导入问题

### Q: 如何为同一环境创建多个变体？

可以注册多个环境ID，指向同一个函数但使用不同的 `kwargs`：

```python
{
    "id": "MyRobotMyTask-v0",
    "entry_point": "mujoco_env.mujoco_env.envs.env_instance:make_my_robot_my_task_env",
    "kwargs": {"scene_name": "scene1"}
},
{
    "id": "MyRobotMyTask-v1",
    "entry_point": "mujoco_env.mujoco_env.envs.env_instance:make_my_robot_my_task_env",
    "kwargs": {"scene_name": "scene2"}
},
```

### Q: 如何修改现有环境？

1. 修改对应的环境实例文件
2. 如果接口有重大变更，增加版本号（如 `-v1`）
3. 保持旧版本注册以保持向后兼容

## 参考文件

- `franka_panda_pick_cube.py` - 简单的 PickCube 环境示例
- `franka_fr3_simple_move.py` - 带模式参数的环境示例
- `aubo_i5_simple_move.py` - 复杂配置的环境示例

## 贡献指南

添加新环境时，请确保：

1. ✅ 遵循命名规范
2. ✅ 添加完整的文档字符串
3. ✅ 在 `__init__.py` 中正确导入和导出
4. ✅ 在 `register_envs` 中注册环境
5. ✅ 测试环境可以正常创建和使用
6. ✅ 更新本文档（如需要）

