# MuJoCo_Env: 机器人强化学习训练环境

**Robot Reinforcement Learning Training Environment**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![MuJoCo 3.1.5](https://img.shields.io/badge/MuJoCo-3.1.5-green.svg)](https://mujoco.org/)


## 🎯 核心目标

**MuJoCo-Env** 是一个专为**机器人强化学习训练**设计的环境框架，提供统一的API接口用于：

- 🔬 **仿真训练**: 基于MuJoCo物理引擎的高保真机器人仿真
- 🤖 **真机部署**: 无缝迁移到真实机器人（Sim-to-Real）
- 🎓 **标准接口**: 完全兼容Gymnasium标准
- 🚀 **高效训练**: 优化的观测空间和奖励函数设计

### 主要支持的机器人

| 机器人 | DOF | 工作空间 | 仿真 | 真机 | 说明 |
|--------|-----|----------|------|------|------|
| **Franka Panda** | 7 | 855mm | ✅ | ✅ | 高精度协作机器人 |
| **Franka FR3** | 7 | 855mm | ✅ | 🔄 | Panda升级版 |
| **Aubo i5** | 6 | 920mm | ✅ | ✅ | 国产协作机器人 |
| **UR5e** | 6 | 850mm | ✅ | 🔄 | UR协作机器人 |
| **DianaMed** | 7 | Custom | ✅ | - | 医疗机器人 |

✅ 完全支持 | 🔄 开发中 | - 仅仿真

---

## ⚡ 快速开始

### 安装

```bash
# 创建conda环境
conda create -n serl python=3.10
conda activate serl

# 克隆并安装
git clone <repository-url>
cd mujoco-env
pip install -r requirements.txt
```

### 5分钟上手：强化学习训练

```python
from mujoco_env.mujoco_env.envs import RobotManipulationEnv
import gymnasium as gym

# 1. 创建Gymnasium环境
env = RobotManipulationEnv(
    robot_name="franka_panda",      # 或 "aubo_i5"
    task_name="pick_cube",           # 抓取任务
    controller_type="operational_space",
    render_mode="human",             # 可视化
    control_freq=20                  # 20Hz控制
)

# 2. 标准RL训练循环
obs, info = env.reset()

for episode in range(1000):
    action = env.action_space.sample()  # 替换为你的策略
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### 使用真实机器人

```python
# 仿真到真机，只需更改一个参数
env = RobotManipulationEnv(
    robot_name="franka_panda",
    task_name="pick_cube",
    use_real_robot=True,            # 🔥 使用真机
    robot_ip="192.168.1.1"          # 机器人IP
)

# API完全一致！
obs, info = env.reset()
action = policy.get_action(obs)
obs, reward, terminated, truncated, info = env.step(action)
```

---

## 🎮 特性

### 核心功能

- ✅ **Gymnasium兼容**: 标准的RL环境接口
- ✅ **模块化设计**: Robot + Task + Controller 灵活组合
- ✅ **仿真与真机统一**: 相同的API，无缝迁移
- ✅ **丰富的控制器**: 9种控制器（关节/笛卡尔空间）
- ✅ **预定义任务**: 抓取、装配、到达等
- ✅ **完整的传感器**: RGB-D相机、力传感器
- ✅ **夹爪控制**: 集成到任务或控制器中

### 强化学习优势

| 特性 | 说明 |
|------|------|
| **目标条件化** | 支持HER（Hindsight Experience Replay） |
| **稀疏+密集奖励** | 灵活的奖励函数设计 |
| **域随机化** | 内置域随机化支持Sim-to-Real |
| **高效采样** | 优化的观测空间设计 |
| **可扩展任务** | 简单继承即可定义新任务 |

---

## 📁 项目结构

```
mujoco_env/
├── mujoco_env/                    # 核心模块
│   ├── core/                      # 环境基类
│   │   ├── base_env.py           # BaseRobotEnv (Gymnasium)
│   │   ├── sim_env.py            # 仿真环境
│   │   └── real_env.py           # 真机环境
│   │
│   ├── robots/                    # 机器人配置
│   │   ├── franka_panda.py       # Franka Panda (7 DOF)
│   │   ├── franka_fr3.py         # Franka FR3 (7 DOF)
│   │   ├── aubo_i5.py            # Aubo i5 (6 DOF)
│   │   ├── ur5e.py               # UR5e (6 DOF)
│   │   └── diana_med.py          # DianaMed (7 DOF)
│   │
│   ├── controllers/               # 控制器（9种）
│   │   ├── base_controller.py    # 控制器基类
│   │   ├── joint_*.py            # 关节空间控制器 (5种)
│   │   ├── operational_space.py  # OSC控制器
│   │   ├── cartesian_*.py        # 笛卡尔控制器 (3种)
│   │   └── admittance.py         # 导纳控制器
│   │
│   ├── tasks/                     # RL任务
│   │   ├── base_task.py          # 任务基类
│   │   ├── pick_cube.py          # 抓取任务
│   │   ├── pcb_insertion.py      # PCB插拔
│   │   └── peg_insertion.py      # 轴孔装配
│   │
│   ├── envs/                      # 环境集成
│   │   ├── robot_env.py          # 机器人操作环境
│   │   └── benchmarks/           # 预定义环境
│   │       └── pick_cube.py      # Panda抓取环境
│   │
│   ├── planners/                  # 路径规划
│   │   ├── rrt.py                # RRT/RRT*
│   │   ├── interpolators.py      # 关节空间插值
│   │   └── linear_interpolator.py # 笛卡尔插值
│   │
│   ├── assets/                    # MuJoCo资源
│   │   ├── models/               # 机器人模型
│   │   │   ├── manipulators/    # 机械臂
│   │   │   └── grippers/        # 夹爪
│   │   ├── scenes/               # 预定义场景
│   │   └── objects/              # 场景物体
│   │
│   └── utils/                     # 工具函数
│       ├── transform.py          # 坐标变换
│       ├── keyboard.py           # 键盘控制
│       └── xml_splice.py         # XML处理
│
├── examples/                      # 示例
│   ├── demo_panda_pick_cube.py   # Panda抓取示例
│   ├── demo_controllers.py       # 控制器演示
│   ├── demo_planners.py          # 规划器演示
│   └── demo_task_controllers.py  # 任务空间控制
│
├── tests/                         # 单元测试
│   ├── test_base_env.py
│   ├── test_robots.py
│   └── test_controllers.py
│
├── docs/                          # 文档
│   └── refactor_doc/             # 重构文档
│
├── requirements.txt               # 依赖
├── setup.py                       # 安装脚本
└── README.md                      # 本文档
```

---

## 🎓 强化学习任务

### 预定义任务

| 任务 | 说明 | 观测维度 | 动作维度 | 奖励类型 |
|------|------|----------|----------|----------|
| **PickCube** | 抓取方块并举起 | 17 | 7 | 密集+稀疏 |
| **PCBInsertion** | PCB插拔装配 | 20 | 7 | 密集+稀疏 |
| **PegInsertion** | 轴孔装配 | 18 | 7 | 密集+稀疏 |

### 创建自定义任务

```python
from mujoco_env.mujoco_env.tasks import BaseTask
import numpy as np

class MyTask(BaseTask):
    """自定义强化学习任务"""
    
    def __init__(self, max_episode_steps=500, **kwargs):
        super().__init__("my_task", max_episode_steps, **kwargs)
    
    def reset(self, model, data):
        """重置任务（如随机化物体位置）"""
        super().reset(model, data)
        # 你的重置逻辑
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        """计算奖励 - RL训练的核心"""
        distance = np.linalg.norm(achieved_goal - desired_goal)
        
        # 密集奖励（引导学习）
        dense_reward = -distance * 10.0
        
        # 稀疏奖励（任务完成）
        sparse_reward = 10.0 if distance < 0.01 else 0.0
        
        return dense_reward + sparse_reward
    
    def _check_success(self, achieved_goal, desired_goal):
        """判断任务是否成功"""
        return np.linalg.norm(achieved_goal - desired_goal) < 0.01

# 使用自定义任务
env = RobotManipulationEnv(
    robot_name="franka_panda",
    task_name="my_task",  # 先注册到TASK_REGISTRY
    controller_type="operational_space"
)
```

---

## 🎮 控制器

框架提供9种控制器，支持关节空间和笛卡尔空间控制：

### 关节空间控制器（5种）

| 控制器 | 输入 | 适用场景 |
|--------|------|----------|
| `joint_position` | 关节角度 | 点到点运动 |
| `joint_velocity` | 关节速度 | 速度控制 |
| `joint_torque` | 关节力矩 | 低级控制 |
| `joint_impedance` | 目标位置+刚度 | 柔顺操作 |
| `joint_freedrive` | 无 | 拖动示教 |

### 任务空间控制器（4种）

| 控制器 | 输入 | 适用场景 |
|--------|------|----------|
| `operational_space` | 笛卡尔位姿 | 末端位置控制 |
| `cartesian_ik` | 笛卡尔位姿 | 精确定位 |
| `cartesian_impedance` | 位姿+刚度 | 接触任务 |
| `admittance` | 期望力 | 力控制 |

### 使用示例

```python
from mujoco_env.mujoco_env.controllers import get_controller

# 方式1: 通过环境自动创建
env = RobotManipulationEnv(
    robot_name="franka_panda",
    task_name="pick_cube",
    controller_type="cartesian_impedance"  # 选择控制器
)

# 方式2: 手动创建控制器
controller = get_controller(
    "operational_space",
    model=model,
    data=data,
    dof=7,
    control_freq=20
)
tau = controller.compute_control(target_pose)
```

---

## 🦾 夹爪控制

夹爪控制可以集成到任务或控制器中，有三种方法：

### 方法1: 在Task中处理（推荐）

```python
class PickCubeTask(BaseTask):
    """带夹爪控制的抓取任务"""
    
    def __init__(self, **kwargs):
        super().__init__("pick_cube", **kwargs)
        self.gripper_open_pos = 0.08
        self.gripper_close_pos = 0.0
    
    def apply_gripper_action(self, data, gripper_action):
        """
        应用夹爪动作
        Args:
            gripper_action: [-1, 1], -1=关闭, 1=打开
        """
        gripper_pos = (self.gripper_close_pos + 
                       (gripper_action + 1) / 2 * 
                       (self.gripper_open_pos - self.gripper_close_pos))
        data.ctrl[self.gripper_actuator_id] = gripper_pos

# 在环境的step中使用
def step(self, action):
    robot_action = action[:6]  # 机器人动作
    gripper_action = action[6]  # 夹爪动作
    
    # 执行机器人控制
    tau = self.controller.compute_control(robot_action)
    self.data.ctrl[:self.robot.dof] = tau
    
    # 执行夹爪控制
    self.task.apply_gripper_action(self.data, gripper_action)
    
    # 步进仿真
    mujoco.mj_step(self.model, self.data)
```

### 方法2: 在Controller中处理

```python
class OperationalSpaceWithGripper(OperationalSpaceController):
    """带夹爪的OSC控制器"""
    
    def compute_control(self, target, gripper_action=0.0):
        # 计算机器人力矩
        tau_robot = super().compute_control(target)
        
        # 计算夹爪控制
        gripper_pos = self._map_gripper_action(gripper_action)
        
        # 合并输出
        return np.concatenate([tau_robot, [gripper_pos]])
```

### 方法3: 混合控制器

适用于需要不同控制策略的场景，详见示例代码。

---

## 📦 资源管理

### MuJoCo资源

框架提供完整的资源管理系统：

```python
from mujoco_env.mujoco_env.assets import (
    get_model_path,
    list_models,
    get_asset_info
)

# 获取模型路径
panda_path = get_model_path("manipulator", "panda")
gripper_path = get_model_path("gripper", "robotiq_2f85")

# 列出所有可用模型
manipulators = list_models("manipulator")
print(f"可用机械臂: {manipulators}")

# 获取资源统计
info = get_asset_info()
print(f"资源统计: {info}")
```

### 资源统计

| 类型 | 数量 | 说明 |
|------|------|------|
| 机械臂 | 5 | Panda, FR3, Aubo, UR5e, DianaMed |
| 夹爪 | 7 | 各种夹爪和工具 |
| 安装座 | 6 | 不同安装方式 |
| 场景 | 5 | 预定义场景 |
| 物体 | 12+ | 各种场景物体 |
| 纹理 | 13+ | 材质纹理 |

---

## 🤖 真机集成

### Franka机器人

```python
from mujoco_env.mujoco_env.envs import RobotManipulationEnv

# 真机环境
env = RobotManipulationEnv(
    robot_name="franka_panda",
    task_name="pick_cube",
    use_real_robot=True,
    robot_ip="192.168.1.1",
    control_freq=20
)

# API与仿真完全一致
obs, info = env.reset()
action = policy.get_action(obs)
obs, reward, terminated, truncated, info = env.step(action)
```

### Aubo机器人

```python
# Aubo真机
env = RobotManipulationEnv(
    robot_name="aubo_i5",
    task_name="pcb_insertion",
    use_real_robot=True,
    robot_ip="192.168.1.6",
    control_freq=20
)
```

### 安全注意事项

⚠️ **使用真实机器人前，请确保：**

1. 工作空间内无人员和障碍物
2. 已配置紧急停止按钮
3. 机器人运动范围已正确设置
4. 启用碰撞检测和力限制
5. 首次运行使用低速模式

---

## 🔬 Sim-to-Real迁移

### 域随机化

```python
# 在仿真中启用域随机化
env = RobotManipulationEnv(
    robot_name="franka_panda",
    task_name="pick_cube",
    domain_randomization=True,  # 启用域随机化
    randomize_params={
        "object_mass": (0.01, 0.1),      # 物体质量范围
        "object_friction": (0.5, 2.0),   # 摩擦系数范围
        "camera_noise": 0.01,            # 相机噪声
        "joint_damping": (0.8, 1.2),     # 关节阻尼
    }
)
```

### 迁移验证

```python
# 1. 仿真训练
sim_env = RobotManipulationEnv(
    robot_name="franka_panda",
    task_name="pick_cube",
    use_real_robot=False
)
policy = train_policy(sim_env)

# 2. 真机测试
real_env = RobotManipulationEnv(
    robot_name="franka_panda",
    task_name="pick_cube",
    use_real_robot=True,
    robot_ip="192.168.1.1"
)
success_rate = evaluate_policy(policy, real_env)
print(f"真机成功率: {success_rate:.1%}")
```

---

## 📚 示例

### 示例1: Panda抓取方块

```bash
cd examples
python demo_panda_pick_cube.py
```

完整的Panda抓取方块强化学习环境，包含：
- 目标条件化（HER兼容）
- 密集+稀疏奖励
- 夹爪控制
- 实时可视化

### 示例2: 控制器演示

```bash
python demo_controllers.py
python demo_task_controllers.py
```

演示9种控制器的使用方法。

### 示例3: 路径规划

```bash
python demo_planners.py
```

演示RRT路径规划和轨迹插值。

---

## 🧪 测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_robot_env.py -v

# 真机集成测试（需要真实机器人）
export REAL_FRANKA_IP=192.168.1.1
pytest tests/test_real_robot_integration.py -v
```

---

## 🛠️ 开发

### 系统要求

- **操作系统**: Linux (Ubuntu 22.04推荐)
- **Python**: 3.10
- **MuJoCo**: 3.1.5
- **Gymnasium**: 1.1.1

### 依赖安装

```bash
pip install -r requirements.txt
```

主要依赖：
- `mujoco >= 3.1.5`
- `gymnasium >= 1.1.1`
- `numpy >= 1.21.0`
- `ruckig >= 0.9.0` (轨迹规划)

---

## 📖 文档

详细文档位于 `docs/` 目录：

- **架构设计**: `docs/refactor_doc/Robots目录架构分析与整合方案.md`
- **夹爪控制**: `docs/refactor_doc/夹爪控制集成指南.md`
- **Assets管理**: `mujoco_env/assets/README.md`
- **双臂支持**: `docs/refactor_doc/双臂机器人支持设计文档.md`

---

## 🤝 贡献

欢迎贡献！请遵循：

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

- [MuJoCo](https://mujoco.org/) - 物理仿真引擎
- [Gymnasium](https://gymnasium.farama.org/) - 强化学习标准接口
- [Ruckig](https://github.com/pantor/ruckig) - 在线轨迹生成
- [SERL](https://github.com/rail-berkeley/serl) - 样本高效强化学习框架

---

## 📧 联系方式

- **问题反馈**: [Issue Tracker](http://git.aubo-robotics.cn:8001/ailab/mujoco-env/-/issues)
- **邮件**: liug@aubo-robotics.cn
- **作者**: Liu Gang

---


**注意**: 本项目专注于机器人强化学习训练环境，主要支持 **Franka Panda** 和 **Aubo i5** 机器人。真机部署功能已验证可用，请严格遵守安全规范。

**Note**: This project focuses on robot reinforcement learning training environments, with primary support for **Franka Panda** and **Aubo i5** robots. Real robot deployment has been validated, please strictly follow safety regulations.
