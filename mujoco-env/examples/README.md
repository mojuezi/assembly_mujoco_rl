# MuJoCo-Env 示例程序

本目录包含MuJoCo-Env框架的各类示例程序，演示不同机器人和控制模式。

**作者**: Liu Gang  
**日期**: 2025-12-20

---

## 📋 示例列表

### 基础架构示例（Day 1-2）

| 文件 | 说明 | 机器人 |
|------|------|--------|
| `demo_base_env.py` | BaseRobotEnv演示 | - |

### 机器人和控制器示例（Day 3-4）

| 文件 | 说明 | 机器人 |
|------|------|--------|
| `demo_robots.py` | 机器人定义和注册表演示 | Franka, Aubo |
| `demo_controllers.py` | 控制器功能演示 | - |

### 任务定义示例（Day 5-7）

| 文件 | 说明 | 任务 |
|------|------|------|
| `demo_tasks.py` | PCB插拔和轴孔装配任务演示 | PCB, Peg |

### 高级应用示例

#### 1. 导纳控制

**文件**: `demo_admittance_control.py`  
**机器人**: Aubo i5  
**控制模式**: 笛卡尔导纳控制 (CARTADM)

**功能**:
- 力控制模式
- 实时力反馈
- 动态力/力矩可视化（可选）

**运行**:
```bash
python examples/demo_admittance_control.py
```

**特点**:
- 质量-阻尼-刚度模型
- 柔顺运动控制
- 力偏置补偿

---

#### 2. Franka FR3关节运动

**文件**: `demo_franka_movej.py`  
**机器人**: Franka FR3 (7 DOF)  
**控制模式**: 关节位置控制 (JNTPOS)

**功能**:
- 关节空间运动
- Ruckig在线轨迹生成（OTG）
- 平滑运动轨迹

**运行**:
```bash
python examples/demo_franka_movej.py
```

**特点**:
- 满足速度、加速度、加加速度约束
- 循环运动演示
- 图形界面可视化

---

#### 3. Aubo i5关节运动

**文件**: `demo_aubo_movej.py`  
**机器人**: Aubo i5 (6 DOF)  
**控制模式**: 关节位置控制 (JNTPOS)

**功能**:
- 关节空间运动
- Ruckig OTG
- 循环运动

**运行**:
```bash
python examples/demo_aubo_movej.py
```

**轨迹**:
- home -> +90° -> -90° -> home (循环5次)

---

#### 4. Aubo i5笛卡尔直线运动

**文件**: `demo_aubo_movel.py`  
**机器人**: Aubo i5  
**控制模式**: 笛卡尔逆运动学 (CARTIK)

**功能**:
- 笛卡尔空间直线运动（MoveL）
- 轨迹可视化
- 矩形路径演示

**运行**:
```bash
python examples/demo_aubo_movel.py
```

**轨迹**:
- 上(0.2m) -> 前(0.1m) -> 下(0.2m) -> 左(0.2m)

---

#### 5. 键盘遥操作

**文件**: `demo_keyboard_teleoperation.py`  
**机器人**: Aubo i5  
**控制模式**: 笛卡尔逆运动学 (CARTIK)

**功能**:
- 实时键盘控制
- 位置和姿态控制
- 笛卡尔空间遥操作

**运行**:
```bash
python examples/demo_keyboard_teleoperation.py
```

**键位**:
- W/S: X轴前进/后退
- A/D: Y轴左移/右移
- Q/E: Z轴上升/下降
- I/K: 绕X轴旋转
- J/L: 绕Y轴旋转
- U/O: 绕Z轴旋转
- Space: 复位

---

#### 6. Panda抓取方块

**文件**: `demo_panda_pick_cube.py`  
**机器人**: Franka Panda  
**接口**: Gymnasium

**功能**:
- 抓取任务演示
- 随机动作采样
- 交互式可视化
- 键盘复位

**运行**:
```bash
python examples/demo_panda_pick_cube.py
```

**交互**:
- Space: 复位环境
- 鼠标: 旋转/平移/缩放视角
- ESC: 退出

---

## 🚀 快速开始

### 安装依赖

```bash
# 进入mujoco_env目录
cd mujoco_env

# 安装依赖
conda activate serl  # 或你的环境名
pip install -e .
```

### 运行示例

```bash
# 基础示例
python examples/demo_robots.py
python examples/demo_controllers.py
python examples/demo_tasks.py

# 高级示例
python examples/demo_franka_movej.py
python examples/demo_aubo_movel.py
python examples/demo_panda_pick_cube.py
```

---

## 📖 示例说明

### 控制模式

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| JNTPOS | 关节位置控制 | 关节空间运动 |
| CARTIK | 笛卡尔逆运动学 | 笛卡尔空间运动 |
| CARTADM | 笛卡尔导纳控制 | 力控制、柔顺操作 |

### 机器人型号

| 机器人 | DOF | 特点 |
|--------|-----|------|
| Franka Panda | 7 | 协作机器人，力矩传感器 |
| Franka FR3 | 7 | Panda升级版 |
| Aubo i5 | 6 | 国产协作机器人 |

### 轨迹生成

- **Ruckig OTG**: 在线轨迹生成，满足运动学约束
- **线性插补**: 笛卡尔空间线性插值
- **关节插补**: 关节空间插值

---

## 🔧 自定义示例

### 创建新示例

1. 复制现有示例作为模板
2. 修改机器人和控制器配置
3. 定义运动轨迹
4. 添加详细注释
5. 测试运行

### 示例模板

```python
#!/usr/bin/env python
"""
示例名称

功能描述

作者: Your Name
日期: YYYY-MM-DD
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from impl.robots.your_robot import YourRobot
from impl.envs import RobotEnv

def main():
    """主函数"""
    # 初始化环境
    env = RobotEnv(
        robot=YourRobot,
        render_mode='human',
        control_freq=200,
        controller='JNTPOS',
    )
    
    env.reset()
    
    # 你的代码
    
    env.close()

if __name__ == "__main__":
    main()
```

---

## 🐛 故障排除

### 常见问题

**问题1**: ImportError: No module named 'impl'

**解决**:
```bash
cd mujoco_env
pip install -e .
```

**问题2**: 渲染窗口不显示

**解决**:
- 检查 `render_mode='human'`
- 确保安装了OpenGL
- 尝试运行 `python -m mujoco.viewer`

**问题3**: 机器人运动不平滑

**解决**:
- 降低最大速度 `max_velocity`
- 增加控制频率 `control_freq`
- 启用轨迹插补 `is_interpolate=True`

---

## 📚 相关文档

- [框架重构方案](../docs/框架重构方案.md)
- [开发清单](../docs/开发清单与实施指南.md)
- [Assets资源](../mujoco_env/assets/README.md)
- [任务扩展指南](../mujoco_env/tasks/扩展任务库方法.md)
- [真机集成指南](../真实机器人集成指南.md)

---

## 🙏 贡献

欢迎贡献新的示例！请遵循以下规范：

1. 添加详细的文档字符串
2. 使用清晰的变量名
3. 添加充分的注释
4. 包含运行说明
5. 测试代码功能

---

**最后更新**: 2025-12-20  
**维护者**: Liu Gang

