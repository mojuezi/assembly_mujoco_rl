# 控制器介绍

| 修订日期   | 修订版本 | 修订内容   | 修订人 |
| ---------- | -------- | ---------- | ------ |
| 2025.12.20 | V0.1     | 初始化文档 | 刘刚   |

---


## 🎯 任务空间基类

为了解决任务空间控制器的复杂依赖问题，创建了 `TaskSpaceController` 基类：

### 提供的核心功能

1. **正运动学**
```python
def forward_kinematics(q: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """计算末端执行器位姿"""
    return pos, quat
```

2. **雅可比矩阵**
```python
def compute_jacobian(q: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """计算几何雅可比"""
    return jac_pos, jac_rot
```

3. **质量矩阵**
```python
def compute_mass_matrix() -> np.ndarray:
    """计算关节空间质量矩阵"""
    return M
```

4. **笛卡尔阻抗控制**
```python
def compute_cartesian_impedance_torque(
    target_pos, target_quat, kp_pos, kp_ori, kd_pos, kd_ori
) -> np.ndarray:
    """计算笛卡尔阻抗力矩"""
    return tau
```

5. **姿态误差计算**
```python
def orientation_error(desired: np.ndarray, current: np.ndarray) -> np.ndarray:
    """使用3D叉积计算姿态误差"""
    return error
```

### 设计优势

- ✅ **独立于robot对象**：只依赖model和data
- ✅ **可复用**：所有任务空间控制器的共同基础
- ✅ **易于测试**：清晰的接口，无复杂依赖
- ✅ **可扩展**：子类可以override或添加功能

---

## 🎨 控制器详解

### 1. 笛卡尔阻抗控制器 (CartesianImpedanceController)

**文件**: `cartesian_impedance.py` (200行)

**功能**: 在笛卡尔空间实现阻抗控制

**控制方程**:
```
τ = J^T · (K_c · x_error - B_c · ẋ) + gravity_compensation
```

**参数**:
- `kp_pos`: 位置刚度 (3,)，默认[100, 100, 100]
- `kp_ori`: 姿态刚度 (3,)，默认[200, 200, 200]
- `kd_pos`: 位置阻尼 (3,)，默认[200, 800, 800]
- `kd_ori`: 姿态阻尼 (3,)，默认[400, 400, 400]

**使用场景**:
- 任务空间柔顺操作
- 接触力控制
- 装配任务

**示例**:
```python
controller = CartesianImpedanceController(
    model=model,
    data=data,
    dof=7,
    ee_site_name="pinch",
    kp_pos=np.array([100.0, 100.0, 100.0]),
    kp_ori=np.array([200.0, 200.0, 200.0])
)

# 目标位姿 [x, y, z, qw, qx, qy, qz]
target = np.array([0.5, 0.0, 0.3, 1, 0, 0, 0])
tau = controller.compute_control(target)
```

---

### 2. 笛卡尔IK控制器 (CartesianIKController)

**文件**: `cartesian_ik.py` (350行)

**功能**: 通过逆运动学将笛卡尔目标转换为关节目标

**核心方法**:
```python
def solve_ik(
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    q_init: Optional[np.ndarray] = None,
    position_only: bool = False
) -> np.ndarray:
    """数值IK求解（最小二乘优化）"""
    return q_solution
```

**特点**:
- 使用MuJoCo的 `minimize.least_squares` 优化器
- 支持位置和姿态约束
- 支持关节限位
- 正则化避免奇异配置

**参数**:
- `ik_regularization`: 正则化系数，默认0.01
- `ik_radius`: 姿态误差缩放因子，默认0.2
- `joint_limits`: 关节限位 (lower, upper)

**使用场景**:
- 笛卡尔空间轨迹跟踪
- 末端位姿精确控制
- 避障路径跟踪

**示例**:
```python
controller = CartesianIKController(
    model=model,
    data=data,
    dof=7,
    ee_site_name="pinch",
    ik_regularization=0.01
)

# 仅位置控制
target_pos = np.array([0.5, 0.0, 0.3])
q_solution = controller.solve_ik(target_pos, current_quat, position_only=True)

# 完整位姿控制
target = np.concatenate([target_pos, target_quat])
tau = controller.compute_control(target)
```

---

### 3. 导纳控制器 (AdmittanceController)

**文件**: `admittance.py` (330行)

**功能**: 力控制策略，根据外部力调整位姿

**控制原理**:
```
M·ẍ + D·ẋ + K·x = F_external - F_desired
```

**参数**:
- `mass`: 虚拟质量 (6,)，默认[20, 20, 20, 20, 20, 20]
- `damping`: 虚拟阻尼 (6,)，默认[200, 200, 200, 200, 200, 200]
- `stiffness`: 虚拟刚度 (6,)，默认[0, 0, 0, 0, 0, 0]
- `selection_vector`: 选择向量 (6,)，1启用导纳，0位置控制

**核心方法**:
```python
def update_admittance(
    ref_pos: np.ndarray,
    ref_vel: np.ndarray,
    ref_acc: np.ndarray,
    external_force: np.ndarray
) -> np.ndarray:
    """根据外部力更新导纳位姿"""
    return adm_pose
```

**使用场景**:
- 力控制装配
- 接触任务
- 拖动示教
- 打磨/抛光

**示例**:
```python
controller = AdmittanceController(
    model=model,
    data=data,
    dof=7,
    ee_site_name="pinch",
    control_freq=50.0  # 力控制需要更高频率
)

# 设置X方向力控制
controller.set_selection_vector(np.array([1, 0, 0, 0, 0, 0]))
controller.set_desired_force(np.array([5.0, 0, 0, 0, 0, 0]))

# 提供外部力反馈
external_force = np.array([3.0, 0, 0, 0, 0, 0])  # 从力传感器读取
tau = controller.compute_control(target, external_force=external_force)
```

### 4. 关节阻抗控制器 (JointImpedanceController)

**功能**: 通过配置虚拟弹簧-阻尼系统实现柔顺控制

**控制方程**:
```
τ = M(q)·(K·(q_d - q) + B·(q̇_d - q̇)) + C(q, q̇) + G(q)
```

**参数**:
- `kp`: 刚度矩阵 K (N·m/rad)，默认40000.0
- `kd`: 阻尼矩阵 B (N·m·s/rad)，默认282.8
- `use_gravity_compensation`: 是否补偿重力，默认True

**使用场景**:
- 接触任务
- 力控制
- 柔顺操作

**示例**:
```python
controller = JointImpedanceController(
    model=model,
    data=data,
    dof=7,
    kp=40000.0 * np.ones(7),
    kd=282.8 * np.ones(7)
)

tau = controller.compute_control(target_qpos)
```

### 5. 关节自由驱动控制器 (JointFreedriveController)

**功能**: 实现机器人自由移动，用于手动示教

**特点**:
- 零刚度 (K = 0)
- 低阻尼 (B = 5.0，可调)
- 重力补偿
- 目标位置自动跟随当前位置

**使用场景**:
- 手动示教
- 拖动编程
- 力反馈交互

**示例**:
```python
controller = JointFreedriveController(
    model=model,
    data=data,
    dof=7,
    damping=5.0
)

# 不需要目标位置
tau = controller.compute_control()
```

---


### 使用方法

```python
from mujoco_env.mujoco_env.controllers import get_controller

# 完整名称
controller = get_controller('cartesian_impedance', model=model, data=data, dof=7)

# 使用别名
controller = get_controller('cart_imp', model=model, data=data, dof=7)
```

---

### 新架构 (mujoco_env/controllers)

```python
class CartesianImpedanceController(TaskSpaceController):
    def __init__(self, model, data, dof, ...):
        super().__init__(model, data, dof, ...)
        
    def compute_control(self, target, current_state):
        # 使用基类提供的方法
        J_pos, J_rot = self.compute_jacobian()
        M = self.compute_mass_matrix()
        pos, quat = self.forward_kinematics()
```

**优势**:
- ✅ 松耦合，只依赖model和data
- ✅ 基类提供通用功能
- ✅ 易于测试和复用
- ✅ 清晰的接口定义

---

## 🧪 测试验证

### 测试文件

**文件**: `examples/demo_task_controllers.py` (250行)

**演示内容**:
1. 笛卡尔阻抗控制
   - 创建控制器
   - 设置目标位姿
   - 计算控制力矩
   
2. 笛卡尔IK控制
   - IK求解
   - 验证IK解
   - 力矩计算
   
3. 导纳控制
   - 纯位置控制模式
   - 力控制模式
   - 选择向量设置
   
4. 注册表使用
   - 列出控制器
   - 动态创建

### 测试结果

```
✓ 所有导入测试通过
✓ 19个控制器全部可用
✓ demo_task_controllers.py运行成功
  - 演示1: ✅ 笛卡尔阻抗控制正常
  - 演示2: ✅ 笛卡尔IK控制正常
  - 演示3: ✅ 导纳控制正常
  - 演示4: ✅ 注册表正常
✓ 无linter错误
```

---

## 📚 使用指南

### 笛卡尔阻抗控制

```python
from mujoco_env.mujoco_env.controllers import CartesianImpedanceController

controller = CartesianImpedanceController(
    model=model,
    data=data,
    dof=7,
    ee_site_name="pinch",
    kp_pos=np.array([100, 100, 100]),  # 位置刚度
    kp_ori=np.array([200, 200, 200]),  # 姿态刚度
    kd_pos=np.array([200, 800, 800]),  # 位置阻尼
    kd_ori=np.array([400, 400, 400])   # 姿态阻尼
)

# 设置目标 [x, y, z, qw, qx, qy, qz]
target = np.array([0.5, 0.0, 0.3, 1, 0, 0, 0])
tau = controller.compute_control(target)

# 调整增益
controller.set_gains(
    kp_pos=np.array([50, 50, 50]),
    kd_pos=np.array([100, 400, 400])
)
```

### 笛卡尔IK控制

```python
from mujoco_env.mujoco_env.controllers import CartesianIKController

controller = CartesianIKController(
    model=model,
    data=data,
    dof=7,
    ee_site_name="pinch",
    ik_regularization=0.01,  # 正则化系数
    ik_radius=0.2            # 姿态误差缩放
)

# 方式1: 直接求解IK
target_pos = np.array([0.5, 0.0, 0.3])
target_quat = np.array([1, 0, 0, 0])
q_solution = controller.solve_ik(target_pos, target_quat)

# 方式2: 使用控制器（内部调用IK）
target = np.concatenate([target_pos, target_quat])
tau = controller.compute_control(target)
```

### 导纳控制

```python
from mujoco_env.mujoco_env.controllers import AdmittanceController

controller = AdmittanceController(
    model=model,
    data=data,
    dof=7,
    ee_site_name="pinch",
    control_freq=50.0,  # 建议更高频率
    mass=20.0 * np.ones(6),
    damping=200.0 * np.ones(6),
    stiffness=0.0 * np.ones(6)
)

# 配置力控制
controller.set_selection_vector(np.array([1, 1, 0, 0, 0, 0]))  # X,Y力控制
controller.set_desired_force(np.array([5.0, 5.0, 0, 0, 0, 0]))  # 期望力

# 标定力传感器
initial_reading = get_force_sensor_reading()  # 从传感器获取
controller.calibrate_force_sensor(initial_reading)

# 控制循环
while True:
    external_force = get_force_sensor_reading()
    tau = controller.compute_control(target, external_force=external_force)
    apply_torque(tau)
```

---

## 💡 最佳实践

### 1. 选择合适的控制器

| 任务需求 | 推荐控制器 | 原因 |
|---------|-----------|------|
| 笛卡尔轨迹跟踪 | CartesianIKController | 精确位姿控制 |
| 柔顺操作 | CartesianImpedanceController | 可调柔顺性 |
| 力控制装配 | AdmittanceController | 力反馈 |
| 快速定位 | CartesianIKController | 直接IK求解 |
| 接触任务 | CartesianImpedanceController | 阻抗特性 |
| 拖动示教 | AdmittanceController | 力引导 |

### 2. 参数调优

**笛卡尔阻抗**:
- 高刚度 (K=100-200) + 高阻尼 (B=400-800): 刚性控制
- 中刚度 (K=50-100) + 中阻尼 (B=200-400): 柔顺操作
- 低刚度 (K=10-50) + 低阻尼 (B=50-200): 极度柔顺

**导纳控制**:
- 低质量 (M=5-10): 快速响应，可能不稳定
- 中质量 (M=20-50): 平衡，推荐
- 高质量 (M=100+): 慢响应，稳定

### 3. 控制频率

| 控制器 | 建议频率 | 最低频率 |
|--------|---------|---------|
| CartesianImpedanceController | 20-50 Hz | 10 Hz |
| CartesianIKController | 20-100 Hz | 20 Hz |
| AdmittanceController | 50-200 Hz | 50 Hz |

**注意**: 力控制（导纳）通常需要更高的控制频率以保证稳定性。

---