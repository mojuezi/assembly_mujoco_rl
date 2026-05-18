# 扩展任务库指南

| 修订日期   | 修订版本 | 修订内容   | 修订人 |
| ---------- | -------- | ---------- | ------ |
| 2025.12.20 | V0.1     | 初始化文档 | 刘刚   |


本文档详细说明如何为MuJoCo-Env添加新的强化学习任务。

---

## 📋 目录

1. [任务设计原则](#任务设计原则)
2. [任务基类说明](#任务基类说明)
3. [创建新任务步骤](#创建新任务步骤)
4. [任务示例](#任务示例)
5. [最佳实践](#最佳实践)
6. [常见任务类型](#常见任务类型)
7. [调试和测试](#调试和测试)

---

## 🎯 任务定义

### 1. 任务基类 (BaseTask)

**文件**: `mujoco_env/tasks/base_task.py`

**主要功能**:
- 定义标准任务接口
- 奖励计算抽象方法
- 成功判断抽象方法
- 目标采样和观测提取
- 任务状态管理

**核心方法**:
```python
- compute_reward(achieved_goal, desired_goal, info) -> float
- is_success_fn(achieved_goal, desired_goal) -> bool
- sample_goal() -> np.ndarray
- get_achieved_goal(obs) -> np.ndarray
- reset() -> Dict
- step(obs) -> Tuple[reward, done, info]
```

### 2. PCB插拔任务 (PCBInsertionTask)

**文件**: `mujoco_env/tasks/pcb_insertion.py`

**任务描述**:
- 目标：将PCB板精确插入到插槽中
- 成功条件：位置误差 < 5mm，姿态对齐 < 5.7度

**关键参数**:
- `position_tolerance`: 0.005m (5mm)
- `orientation_tolerance`: 0.1 rad (约5.7度)
- `insertion_depth`: 0.05m (50mm)

**奖励组成**:
- 距离奖励：与目标位置的距离（-10×error）
- 姿态奖励：姿态对齐程度（-5×error）
- 成功奖励：完全成功时给予10分

**观测空间**:
```python
{
    "pcb_pos": Box(3),        # PCB位置
    "pcb_quat": Box(4),       # PCB姿态（四元数）
    "desired_goal": Box(7),   # 目标位姿
}
```

### 3. 轴孔装配任务 (PegInsertionTask)

**文件**: `mujoco_env/tasks/peg_insertion.py`

**任务描述**:
- 目标：将圆柱形的轴精确插入到孔中
- 成功条件：位置误差 < 2mm，姿态对齐 < 2.9度，完全插入

**关键参数**:
- `position_tolerance`: 0.002m (2mm)
- `orientation_tolerance`: 0.05 rad (约2.9度)
- `peg_radius`: 0.01m (10mm)
- `hole_radius`: 0.011m (11mm)
- `clearance`: 1mm

**奖励组成**:
- 距离奖励：非线性，越近奖励增长越快
- 姿态奖励：对齐很重要（-15×error）
- 插入深度奖励：（-10×depth_error）
- 接近奖励：小于10mm时+5分
- 成功奖励：完全成功时+20分

**观测空间**:
```python
{
    "peg_pos": Box(3),          # 轴位置
    "peg_quat": Box(4),         # 轴姿态
    "insertion_depth": Box(1),  # 插入深度
    "desired_goal": Box(8),     # 目标位姿+深度
}
```

## 🎯 任务设计原则

### 1. 目标条件化（Goal-Conditioned）

所有任务都应该遵循目标条件化强化学习范式：

- **观测包含目标**: `achieved_goal` 和 `desired_goal`
- **奖励基于目标距离**: 越接近目标奖励越高
- **支持HER**: 便于使用Hindsight Experience Replay

### 2. 奖励函数设计

**密集奖励 + 稀疏奖励**:
- 密集奖励：基于距离/误差的连续奖励
- 稀疏奖励：成功时的大额奖励

**多组件奖励**:
- 位置奖励
- 姿态奖励
- 速度奖励（可选）
- 接触奖励（可选）

**非线性奖励**:
- 越接近目标，奖励增长越快
- 使用平方或指数函数

### 3. 成功判断

**明确的成功条件**:
- 位置误差阈值
- 姿态误差阈值
- 时间约束（可选）

**容差设置**:
- 根据任务难度设置合理容差
- PCB插拔: 5mm
- 轴孔装配: 2mm
- 精密装配: 0.5mm

---

## 📚 任务基类说明

### BaseTask

所有任务都应继承 `BaseTask` 类：

```python
from mujoco_env.mujoco_env.tasks.base_task import BaseTask

class MyTask(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(
            name="MyTask",
            max_episode_steps=500,
            success_threshold=0.01,
            **kwargs
        )
```

### 必须实现的抽象方法

#### 1. compute_reward()

计算奖励函数。

**签名**:
```python
def compute_reward(
    self,
    achieved_goal: np.ndarray,
    desired_goal: np.ndarray,
    info: Dict[str, Any]
) -> float:
    """
    计算奖励
    
    Args:
        achieved_goal: 当前达到的目标
        desired_goal: 期望的目标
        info: 额外信息（如力/力矩）
        
    Returns:
        reward: 奖励值
    """
```

**设计要点**:
- 返回值应该是有界的
- 成功时给予大额奖励
- 失败时给予惩罚（可选）
- 考虑多个奖励组件

#### 2. is_success_fn()

判断任务是否成功。

**签名**:
```python
def is_success_fn(
    self,
    achieved_goal: np.ndarray,
    desired_goal: np.ndarray
) -> bool:
    """
    判断是否成功
    
    Args:
        achieved_goal: 当前达到的目标
        desired_goal: 期望的目标
        
    Returns:
        success: 是否成功
    """
```

**设计要点**:
- 返回布尔值
- 条件应该明确
- 与奖励函数一致

#### 3. sample_goal()

采样新的目标。

**签名**:
```python
def sample_goal(self) -> np.ndarray:
    """
    采样目标
    
    Returns:
        goal: 目标数组
    """
```

**设计要点**:
- 目标应该是可达的
- 考虑目标的多样性
- 可以使用随机采样或预定义目标

#### 4. get_achieved_goal()

从观测中提取当前达到的目标。

**签名**:
```python
def get_achieved_goal(
    self, 
    obs: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    提取achieved goal
    
    Args:
        obs: 观测字典
        
    Returns:
        achieved_goal: 当前达到的目标
    """
```

**设计要点**:
- 从观测中提取相关信息
- 与desired_goal维度一致
- 处理缺失的观测

---

## 🔨 创建新任务步骤

### 步骤1: 创建任务文件

在 `mujoco_env/tasks/` 目录下创建新文件，如 `my_task.py`：

```python
"""
我的任务

作者: Your Name
日期: 2025-12-20
"""

from typing import Dict, Any
import numpy as np
from gymnasium import spaces
from mujoco_env.mujoco_env.tasks.base_task import BaseTask


class MyTask(BaseTask):
    """
    任务描述
    
    目标：...
    成功条件：...
    """
    
    def __init__(
        self,
        max_episode_steps: int = 500,
        success_threshold: float = 0.01,
        **kwargs
    ):
        super().__init__(
            name="MyTask",
            max_episode_steps=max_episode_steps,
            success_threshold=success_threshold,
            **kwargs
        )
        
        # 任务特定参数
        self.param1 = kwargs.get('param1', default_value)
        self.param2 = kwargs.get('param2', default_value)
    
    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Dict[str, Any]
    ) -> float:
        # 实现奖励计算
        pass
    
    def is_success_fn(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray
    ) -> bool:
        # 实现成功判断
        pass
    
    def sample_goal(self) -> np.ndarray:
        # 实现目标采样
        pass
    
    def get_achieved_goal(
        self, 
        obs: Dict[str, np.ndarray]
    ) -> np.ndarray:
        # 实现achieved goal提取
        pass
```

### 步骤2: 注册任务

在 `mujoco_env/tasks/__init__.py` 中注册新任务：

```python
from mujoco_env.mujoco_env.tasks.my_task import MyTask

TASK_REGISTRY = {
    # ... 现有任务 ...
    "my_task": MyTask,
    "my": MyTask,  # 别名
}
```

### 步骤3: 编写测试

在 `tests/test_tasks.py` 中添加测试：

```python
class TestMyTask:
    """测试我的任务"""
    
    def test_creation(self):
        """测试任务创建"""
        task = MyTask(max_episode_steps=500)
        assert task.name == "MyTask"
        assert task.max_episode_steps == 500
    
    def test_sample_goal(self):
        """测试目标采样"""
        task = MyTask()
        goal = task.sample_goal()
        assert goal.shape == (expected_dim,)
    
    def test_compute_reward(self):
        """测试奖励计算"""
        task = MyTask()
        achieved = np.array([...])
        desired = np.array([...])
        reward = task.compute_reward(achieved, desired, {})
        assert isinstance(reward, float)
    
    def test_is_success(self):
        """测试成功判断"""
        task = MyTask()
        # 测试成功情况
        # 测试失败情况
```

### 步骤4: 创建示例

在 `examples/` 中创建演示程序：

```python
"""
我的任务演示

作者: Your Name
日期: 2025-12-20
"""

from mujoco_env.mujoco_env.tasks import get_task

def demo_my_task():
    # 创建任务
    task = get_task("my_task", max_episode_steps=500)
    
    # 采样目标
    goal = task.sample_goal()
    print(f"目标: {goal}")
    
    # 模拟执行
    # ...

if __name__ == "__main__":
    demo_my_task()
```

---

## 📖 任务示例

### 示例1: 简单到达任务

```python
class ReachTask(BaseTask):
    """
    到达任务：机械臂末端到达目标位置
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            name="Reach",
            max_episode_steps=200,
            success_threshold=0.02,  # 2cm
            **kwargs
        )
        
        # 工作空间范围
        self.workspace = np.array([
            [0.3, 0.7],   # x
            [-0.3, 0.3],  # y
            [0.1, 0.5],   # z
        ])
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        # 距离误差
        distance = np.linalg.norm(achieved_goal - desired_goal)
        
        # 距离奖励（非线性）
        distance_reward = -distance * 10.0 - distance**2 * 50.0
        
        # 成功奖励
        success_bonus = 10.0 if distance < self.success_threshold else 0.0
        
        return distance_reward + success_bonus
    
    def is_success_fn(self, achieved_goal, desired_goal):
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return distance < self.success_threshold
    
    def sample_goal(self):
        # 在工作空间内随机采样
        goal = np.array([
            np.random.uniform(self.workspace[0, 0], self.workspace[0, 1]),
            np.random.uniform(self.workspace[1, 0], self.workspace[1, 1]),
            np.random.uniform(self.workspace[2, 0], self.workspace[2, 1]),
        ])
        return goal
    
    def get_achieved_goal(self, obs):
        # 使用末端位置作为achieved goal
        return obs.get("tcp_pos", np.zeros(3))
```

### 示例2: 抓取任务

```python
class PickTask(BaseTask):
    """
    抓取任务：抓取物体并举起
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            name="Pick",
            max_episode_steps=300,
            success_threshold=0.05,  # 5cm
            **kwargs
        )
        
        self.lift_height = 0.2  # 举起高度
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        # achieved_goal: [tcp_pos(3), object_pos(3), gripper_state(1)]
        # desired_goal: [target_pos(3), lift_height(1)]
        
        tcp_pos = achieved_goal[:3]
        object_pos = achieved_goal[3:6]
        gripper_state = achieved_goal[6]
        
        target_pos = desired_goal[:3]
        target_height = desired_goal[3]
        
        # 阶段1: 接近物体
        approach_dist = np.linalg.norm(tcp_pos - object_pos)
        approach_reward = -approach_dist * 5.0
        
        # 阶段2: 抓取物体
        grasp_reward = 0.0
        if approach_dist < 0.05 and gripper_state < 0.5:  # 夹爪闭合
            grasp_reward = 5.0
        
        # 阶段3: 举起物体
        lift_reward = 0.0
        if gripper_state < 0.5:  # 已抓取
            lift_dist = object_pos[2] - target_height
            lift_reward = -abs(lift_dist) * 10.0
            if abs(lift_dist) < 0.02:
                lift_reward += 10.0  # 成功举起
        
        return approach_reward + grasp_reward + lift_reward
    
    def is_success_fn(self, achieved_goal, desired_goal):
        object_pos = achieved_goal[3:6]
        gripper_state = achieved_goal[6]
        target_height = desired_goal[3]
        
        # 物体被抓取且达到目标高度
        is_grasped = gripper_state < 0.5
        is_lifted = abs(object_pos[2] - target_height) < 0.02
        
        return is_grasped and is_lifted
    
    def sample_goal(self):
        # 物体初始位置 + 目标高度
        object_pos = np.array([0.5, 0.0, 0.02])
        target_height = object_pos[2] + self.lift_height
        return np.concatenate([object_pos, [target_height]])
    
    def get_achieved_goal(self, obs):
        tcp_pos = obs.get("tcp_pos", np.zeros(3))
        object_pos = obs.get("object_pos", np.zeros(3))
        gripper_state = obs.get("gripper_state", np.array([1.0]))
        return np.concatenate([tcp_pos, object_pos, gripper_state])
```

### 示例3: 放置任务

```python
class PlaceTask(BaseTask):
    """
    放置任务：将物体放置到目标位置
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            name="Place",
            max_episode_steps=400,
            success_threshold=0.03,  # 3cm
            **kwargs
        )
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        # achieved_goal: [object_pos(3), object_quat(4)]
        # desired_goal: [target_pos(3), target_quat(4)]
        
        object_pos = achieved_goal[:3]
        object_quat = achieved_goal[3:7]
        target_pos = desired_goal[:3]
        target_quat = desired_goal[3:7]
        
        # 位置奖励
        pos_error = np.linalg.norm(object_pos - target_pos)
        pos_reward = -pos_error * 20.0 - pos_error**2 * 100.0
        
        # 姿态奖励
        quat_dot = np.abs(np.dot(object_quat, target_quat))
        quat_error = 1.0 - quat_dot
        quat_reward = -quat_error * 10.0
        
        # 成功奖励
        success_bonus = 0.0
        if pos_error < self.success_threshold and quat_error < 0.1:
            success_bonus = 20.0
        
        return pos_reward + quat_reward + success_bonus
    
    def is_success_fn(self, achieved_goal, desired_goal):
        object_pos = achieved_goal[:3]
        object_quat = achieved_goal[3:7]
        target_pos = desired_goal[:3]
        target_quat = desired_goal[3:7]
        
        pos_error = np.linalg.norm(object_pos - target_pos)
        quat_dot = np.abs(np.dot(object_quat, target_quat))
        quat_error = 1.0 - quat_dot
        
        return pos_error < self.success_threshold and quat_error < 0.1
    
    def sample_goal(self):
        # 随机目标位置和姿态
        target_pos = np.array([
            np.random.uniform(0.3, 0.7),
            np.random.uniform(-0.2, 0.2),
            0.02  # 桌面高度
        ])
        target_quat = np.array([1.0, 0.0, 0.0, 0.0])  # 垂直放置
        return np.concatenate([target_pos, target_quat])
    
    def get_achieved_goal(self, obs):
        object_pos = obs.get("object_pos", np.zeros(3))
        object_quat = obs.get("object_quat", np.array([1, 0, 0, 0]))
        return np.concatenate([object_pos, object_quat])
```

---

## 💡 最佳实践

### 1. 奖励函数设计

**DO**:
- ✅ 使用密集奖励引导学习
- ✅ 成功时给予大额奖励
- ✅ 使用非线性奖励（越近越好）
- ✅ 平衡不同组件的权重
- ✅ 奖励应该有界

**DON'T**:
- ❌ 纯稀疏奖励（难以学习）
- ❌ 奖励过于复杂
- ❌ 奖励不一致
- ❌ 奖励无界

### 2. 目标采样

**DO**:
- ✅ 目标应该可达
- ✅ 考虑目标多样性
- ✅ 避免过于简单的目标
- ✅ 可以使用课程学习

**DON'T**:
- ❌ 目标超出工作空间
- ❌ 目标过于集中
- ❌ 目标过于困难

### 3. 成功判断

**DO**:
- ✅ 明确的成功条件
- ✅ 与奖励函数一致
- ✅ 合理的容差
- ✅ 考虑多个条件

**DON'T**:
- ❌ 条件过于宽松
- ❌ 条件过于严格
- ❌ 与奖励不一致

---

## 📂 常见任务类型

### 1. 操作任务

- **到达 (Reach)**: 末端到达目标位置
- **抓取 (Pick)**: 抓取物体
- **放置 (Place)**: 放置物体到目标
- **推动 (Push)**: 推动物体到目标
- **滑动 (Slide)**: 滑动物体

### 2. 装配任务

- **PCB插拔**: 电路板插入插槽
- **轴孔装配**: 圆柱插入孔
- **螺丝拧紧**: 拧螺丝
- **卡扣装配**: 卡扣连接

### 3. 灵巧操作

- **开门**: 转动门把手开门
- **拧瓶盖**: 拧开瓶盖
- **翻转**: 翻转物体
- **旋转**: 旋转物体到目标姿态

### 4. 双臂协作

- **协同抓取**: 两个机械臂协同抓取大物体
- **传递**: 物体在两臂间传递
- **协同装配**: 两臂协同完成装配

---

## 🐛 调试和测试

### 调试技巧

1. **可视化目标**:
```python
# 在MuJoCo中可视化目标位置
viewer.add_marker(
    pos=desired_goal[:3],
    size=[0.02, 0.02, 0.02],
    rgba=[1, 0, 0, 0.5],
    type=mujoco.mjtGeom.mjGEOM_SPHERE
)
```

2. **打印调试信息**:
```python
def compute_reward(self, achieved_goal, desired_goal, info):
    distance = np.linalg.norm(achieved_goal - desired_goal)
    print(f"Distance: {distance:.4f}")
    # ...
```

3. **分析奖励组件**:
```python
reward_components = {
    "distance": distance_reward,
    "orientation": orientation_reward,
    "success": success_bonus
}
print(f"Reward components: {reward_components}")
```

### 单元测试

```python
def test_reward_range():
    """测试奖励范围"""
    task = MyTask()
    
    # 测试最好情况
    achieved = desired = np.array([0.5, 0.0, 0.2])
    reward_best = task.compute_reward(achieved, desired, {})
    
    # 测试最坏情况
    achieved_worst = np.array([1.0, 1.0, 1.0])
    reward_worst = task.compute_reward(achieved_worst, desired, {})
    
    assert reward_best > reward_worst
    assert reward_best < 100  # 有界
    assert reward_worst > -100  # 有界
```

### 性能测试

```python
import time

def test_performance():
    """测试性能"""
    task = MyTask()
    
    start = time.time()
    for _ in range(1000):
        achieved = np.random.rand(7)
        desired = np.random.rand(7)
        reward = task.compute_reward(achieved, desired, {})
    elapsed = time.time() - start
    
    print(f"1000次奖励计算耗时: {elapsed:.3f}秒")
    assert elapsed < 1.0  # 应该很快
```

---

## 📝 文档模板

为新任务创建文档：

```markdown
# 任务名称

## 描述

简要描述任务目标和场景。

## 目标

详细说明任务目标。

## 成功条件

列出成功的具体条件。

## 观测空间

描述观测空间的结构。

## 动作空间

描述动作空间的结构。

## 奖励函数

说明奖励函数的设计。

## 难度

- 简单 / 中等 / 困难

## 使用示例

```python
from mujoco_env.mujoco_env.tasks import get_task

task = get_task("task_name", max_episode_steps=500)
```

## 参数

列出任务特定参数。

## 注意事项

特殊注意事项。
```

---

## 🔗 相关资源

- [BaseTask API文档](base_task.py)
- [PCB插拔任务示例](pcb_insertion.py)
- [轴孔装配任务示例](peg_insertion.py)
- [任务测试示例](../tests/test_tasks.py)

---

## 📞 技术支持

如有问题，请：
1. 查看现有任务实现
2. 运行单元测试
3. 查阅MuJoCo文档
4. 联系维护者

---

