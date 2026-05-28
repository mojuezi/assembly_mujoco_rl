# 真实机器人集成指南 (Real Robot Interface)

| 修订日期   | 修订版本 | 修订内容   | 修订人 |
| ---------- | -------- | ---------- | ------ |
| 2025.12.20 | V0.1     | 初始化文档 | 刘刚   |

本模块提供统一的机器人接口，支持 Franka Panda 和 Aubo i5 等真实机器人，确保仿真和真机环境使用一致的API。

---

## 📋 目录

- [设计理念](#设计理念)
- [支持的机器人](#支持的机器人)
- [安装依赖](#安装依赖)
- [快速开始](#快速开始)
- [API 文档](#api-文档)
- [示例代码](#示例代码)
- [故障排除](#故障排除)

---

## 🎯 设计理念

### 统一接口

所有机器人接口继承自 `RobotInterface` 基类，提供一致的API：

```python
class RobotInterface(ABC):
    def connect() -> bool
    def disconnect()
    def get_joint_positions() -> np.ndarray
    def get_tcp_pose() -> np.ndarray
    def move_to_joint_positions(positions, velocity, acceleration, blocking)
    def move_tcp_pose(pose, velocity, acceleration, blocking)
    def get_robot_state() -> Dict
    def stop_motion()
    def clear_errors()
```

### 仿真与真机无缝切换

```python
# 仿真环境
env = RobotManipulationEnv(
    robot_name="franka_panda",
    task_name="pick_cube",
    use_real_robot=False  # 使用仿真
)

# 真机环境（只需改变一个参数）
env = RobotManipulationEnv(
    robot_name="franka_panda",
    task_name="pick_cube",
    use_real_robot=True,  # 使用真机
    robot_ip="192.168.1.1"
)
```

---

## 🤖 支持的机器人

### 1. Franka Panda

| 属性 | 值 |
|------|------|
| 自由度 | 7 DOF |
| 负载 | 3 kg |
| 工作半径 | 855 mm |
| 通信方式 | Flask Server + ROS + libfranka |
| 接口类 | `FrankaInterface` |

### 2. Aubo i5

| 属性 | 值 |
|------|------|
| 自由度 | 6 DOF |
| 负载 | 5 kg |
| 工作半径 | 920 mm |
| 通信方式 | Aubo Python SDK |
| 接口类 | `AuboInterface` |

---

## 📦 安装依赖

### Franka Panda

```bash
# 1. 安装 ROS (推荐 ROS Noetic)
# 参考: http://wiki.ros.org/noetic/Installation

# 2. 安装 libfranka
sudo apt install ros-noetic-libfranka ros-noetic-franka-ros

# 3. 安装 Python 依赖
pip install requests scipy numpy
```

**启动 Franka 服务器**:
```bash
# 在 serl_robot_infra/robot_servers/ 目录下
python franka_server.py --robot_ip=192.168.1.1 --server_port=5000
```

### Aubo i5

```bash
# 1. 安装 Aubo Python SDK
pip install pyaubo_sdk

# 2. 安装其他依赖
pip install scipy numpy
```

---

## 🚀 快速开始

### Franka Panda

```python
from mujoco_env.mujoco_env.real import FrankaInterface
import numpy as np

# 创建接口（确保服务器已启动）
robot = FrankaInterface(
    robot_ip="192.168.1.1",
    server_ip="127.0.0.1",
    server_port=5000
)

# 方式1: 手动连接
if robot.connect():
    # 获取状态
    state = robot.get_robot_state()
    print(f"TCP位姿: {state['tcp_pose']}")
    
    # 移动到目标位姿
    target_pose = state['tcp_pose'].copy()
    target_pose[2] += 0.05  # Z轴上移5cm
    robot.move_tcp_pose(target_pose, velocity=0.1)
    
    robot.disconnect()

# 方式2: 使用上下文管理器（推荐）
with robot:
    state = robot.get_robot_state()
    print(f"当前位姿: {state['tcp_pose']}")
```

### Aubo i5

```python
from mujoco_env.mujoco_env.real import AuboInterface
import numpy as np

# 创建接口
robot = AuboInterface(robot_ip="192.168.1.6")

# 使用上下文管理器
with robot:
    # 获取状态
    state = robot.get_robot_state()
    print(f"关节位置: {state['joint_positions']}")
    
    # 关节空间移动
    target_joints = np.array([0., -0.262, 1.745, 0.436, 1.571, 0.])
    robot.move_to_joint_positions(target_joints, velocity=0.3, blocking=True)
    
    # 笛卡尔空间移动
    current_pose = robot.get_tcp_pose()
    target_pose = current_pose.copy()
    target_pose[0] += 0.1  # X轴移动10cm
    robot.move_tcp_pose(target_pose, velocity=0.1, blocking=True)
```

---

## 📖 API 文档

### 核心方法

#### `connect() -> bool`
连接到机器人

**Returns:**
- `bool`: 连接成功返回True

#### `disconnect()`
断开机器人连接

#### `get_joint_positions() -> np.ndarray`
获取当前关节位置

**Returns:**
- `np.ndarray`: 关节位置 (弧度), shape=(dof,)

#### `get_tcp_pose() -> np.ndarray`
获取TCP位姿

**Returns:**
- `np.ndarray`: [x, y, z, qw, qx, qy, qz], shape=(7,)

#### `move_tcp_pose(pose, velocity, acceleration, blocking)`
移动到目标TCP位姿

**Args:**
- `pose` (np.ndarray): 目标位姿 [x, y, z, qw, qx, qy, qz]
- `velocity` (float): 速度 (m/s)
- `acceleration` (float): 加速度 (m/s^2)
- `blocking` (bool): 是否阻塞等待完成

#### `get_robot_state() -> Dict`
获取完整机器人状态

**Returns:**
```python
{
    "joint_positions": np.ndarray,   # 关节位置
    "joint_velocities": np.ndarray,  # 关节速度
    "tcp_pose": np.ndarray,          # TCP位姿
    "tcp_velocity": np.ndarray,      # TCP速度
    "tcp_force": np.ndarray,         # TCP力
    "tcp_torque": np.ndarray,        # TCP力矩
    "errors": List[str],             # 错误列表
}
```

### 可选方法

#### `set_freedrive_mode(enable: bool)`
设置示教模式（拖动示教）

**Note**: Aubo 支持，Franka 需要服务器端实现

#### `set_compliance_mode(stiffness: float, damping: float)`
设置柔顺模式参数

**Note**: 两者都支持，但参数范围不同

---

## 💡 示例代码

### 示例1: 读取机器人状态

```python
from mujoco_env.mujoco_env.real import FrankaInterface, AuboInterface

def read_robot_state(robot):
    """读取并打印机器人状态"""
    with robot:
        state = robot.get_robot_state()
        
        print("\n=== 机器人状态 ===")
        print(f"关节位置 (rad): {state['joint_positions']}")
        print(f"关节速度 (rad/s): {state['joint_velocities']}")
        print(f"TCP 位置 (m): {state['tcp_pose'][:3]}")
        print(f"TCP 姿态 (quat): {state['tcp_pose'][3:]}")
        print(f"TCP 力 (N): {state['tcp_force']}")
        print(f"TCP 力矩 (Nm): {state['tcp_torque']}")

# Franka
robot_franka = FrankaInterface(robot_ip="192.168.1.1")
read_robot_state(robot_franka)

# Aubo
robot_aubo = AuboInterface(robot_ip="192.168.1.6")
read_robot_state(robot_aubo)
```

### 示例2: 直线运动

```python
from mujoco_env.mujoco_env.real import AuboInterface
import numpy as np
import time

robot = AuboInterface(robot_ip="192.168.1.6")

with robot:
    # 获取起始位姿
    start_pose = robot.get_tcp_pose()
    print(f"起始位姿: {start_pose}")
    
    # 目标位姿（X轴移动20cm）
    end_pose = start_pose.copy()
    end_pose[0] += 0.2
    
    # 执行移动
    robot.move_tcp_pose(end_pose, velocity=0.1, blocking=True)
    print("移动完成！")
    
    # 等待1秒
    time.sleep(1.0)
    
    # 返回起始位姿
    robot.move_tcp_pose(start_pose, velocity=0.1, blocking=True)
    print("返回起始位姿！")
```

### 示例3: 柔顺控制

```python
from mujoco_env.mujoco_env.real import FrankaInterface
import time

robot = FrankaInterface(robot_ip="192.168.1.1")

with robot:
    # 设置为柔顺模式（低刚度）
    robot.set_compliance_mode(stiffness=200.0, damping=20.0)
    print("已切换到柔顺模式，可以手动拖动机器人")
    time.sleep(10.0)
    
    # 设置为精确模式（高刚度）
    robot.set_compliance_mode(stiffness=600.0, damping=60.0)
    print("已切换到精确模式")
```

### 示例4: 示教模式

```python
from mujoco_env.mujoco_env.real import AuboInterface
import time

robot = AuboInterface(robot_ip="192.168.1.6")

with robot:
    # 启用示教模式
    robot.set_freedrive_mode(True)
    print("示教模式已启用，请拖动机器人到目标位置")
    time.sleep(30.0)
    
    # 记录位姿
    target_pose = robot.get_tcp_pose()
    print(f"记录的目标位姿: {target_pose}")
    
    # 退出示教模式
    robot.set_freedrive_mode(False)
    print("示教模式已退出")
```

---

## 🔧 故障排除

### Franka

**问题1**: `ConnectionError: Cannot connect to Franka server`

**解决方案**:
1. 确认 Franka 服务器已启动
2. 检查服务器 IP 和端口是否正确
3. 测试网络连通性: `ping <server_ip>`

**问题2**: `RuntimeError: Request failed with status 500`

**解决方案**:
1. 检查服务器日志
2. 确认机器人无错误（调用 `clear_errors()`）
3. 检查机器人是否在安全工作空间内

### Aubo

**问题1**: `ImportError: No module named 'pyaubo_sdk'`

**解决方案**:
```bash
pip install pyaubo_sdk
```

**问题2**: `RuntimeError: Robot not connected`

**解决方案**:
1. 检查机器人 IP 地址是否正确
2. 确认机器人已开机并完成初始化
3. 测试网络连通性: `ping <robot_ip>`

**问题3**: 移动命令无响应

**解决方案**:
1. 调用 `clear_errors()` 清除错误
2. 检查机器人是否在手动模式
3. 确认目标位姿在工作空间内

**问题4**: 碰撞误触发

**解决方案**:
1. 调整碰撞阈值
2. 检查负载配置
3. 校准力传感器
4. 降低加速度

**问题5**: 运动不平滑

**解决方案**:
1. 降低控制频率
2. 增加轨迹插补点
3. 调整PD增益
4. 检查网络延迟
---

## 📊 接口对比

| 功能 | Franka | Aubo | 说明 |
|------|--------|------|------|
| 关节位置控制 | ❌ | ✅ | Franka 需服务器扩展 |
| TCP位姿控制 | ✅ | ✅ | 两者都支持 |
| 力传感器 | ✅ | ✅ | 两者都支持 |
| 示教模式 | ⚠️ | ✅ | Franka 需服务器扩展 |
| 柔顺控制 | ✅ | ✅ | 两者都支持 |
| 雅可比矩阵 | ✅ | ❌ | Franka 服务器提供 |
| 阻塞模式 | ❌ | ✅ | Franka 为非阻塞 |

✅ 完全支持 | ⚠️ 部分支持 | ❌ 不支持

---

## 工作空间安全
DO:
✅ 确保工作空间无障碍物
✅ 设置安全围栏
✅ 使用软限位
✅ 准备紧急停止按钮

DON'T:
❌ 在机器人运动范围内工作
❌ 忽略警告信息
❌ 禁用安全功能

## 🔗 相关文档

- [Franka Control Interface](https://frankaemika.github.io/libfranka/)
- [Aubo SDK 文档](https://www.aubo-robotics.cn/)
- [机器人接口基类](./robot_interface.py)
- [真机集成指南](../../docs/AUBO_INTEGRATION_GUIDE.md)

---

## 📧 支持

如有问题，请联系：
- 邮箱: liug@aubo-robotics.cn
- Issue Tracker: [GitHub Issues](http://git.aubo-robotics.cn:8001/ailab/mujoco-env/-/issues)

---

