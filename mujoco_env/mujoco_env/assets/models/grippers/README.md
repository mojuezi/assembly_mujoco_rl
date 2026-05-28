# 夹爪控制集成指南

| 修订日期   | 修订版本 | 修订内容   | 修订人 |
| ---------- | -------- | ---------- | ------ |
| 2025.12.20 | V0.1     | 初始化文档 | 刘刚   |

## 📊 方法对比

| 方法 | 优点 | 缺点 | 推荐场景 |
|------|------|------|---------|
| **方法1: Task中处理** | 任务逻辑清晰<br>易于理解<br>灵活 | 需要在环境中集成 | 夹爪控制与任务逻辑紧密相关 |
| **方法2: Controller中处理** | 控制逻辑统一<br>易于复用 | 增加控制器复杂度 | 夹爪需要复杂的控制策略 |
| **方法3: 混合控制器** | 灵活性最高<br>可组合不同策略 | 实现复杂 | 需要不同的控制策略组合 |

---

## 💡 最佳实践

### 1. 夹爪动作范围标准化

建议统一使用 `[-1, 1]` 范围：
- `1`: 完全打开
- `-1`: 完全关闭
- `0`: 中间位置

```python
def normalize_gripper_action(action: float, min_pos: float, max_pos: float) -> float:
    """将[-1, 1]映射到[min_pos, max_pos]"""
    return min_pos + (action + 1) / 2 * (max_pos - min_pos)
```

### 2. 夹爪状态观测

在观测空间中包含夹爪状态：

```python
gripper_obs = np.array([
    gripper_position,     # 当前位置
    gripper_velocity,     # 当前速度
    gripper_force,        # 当前力（如果有力传感器）
    gripper_target,       # 目标位置（可选）
])
```

### 3. 力控制

如果夹爪支持力控制（如Robotiq夹爪）：

```python
class ForceControlledGripper:
    """力控制夹爪"""
    
    def __init__(self, model, data, actuator_name, force_sensor_name):
        self.actuator_id = model.actuator(actuator_name).id
        self.force_sensor_id = model.sensor(force_sensor_name).id
        
        # 力控制参数
        self.target_force = 10.0  # N
        self.force_threshold = 2.0  # N
    
    def apply_force_control(self, data, target_force: float):
        """基于目标力的控制"""
        current_force = data.sensordata[self.force_sensor_id]
        force_error = target_force - current_force
        
        # 简单的比例控制
        kp = 0.1
        gripper_velocity = kp * force_error
        
        # 限制速度
        gripper_velocity = np.clip(gripper_velocity, -1.0, 1.0)
        
        data.ctrl[self.actuator_id] = gripper_velocity
```

### 4. 碰撞检测

检测夹爪是否接触到物体：

```python
def check_gripper_contact(model, data, gripper_geom_names: List[str]) -> bool:
    """检查夹爪是否与物体接触"""
    gripper_geom_ids = [model.geom(name).id for name in gripper_geom_names]
    
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1 = contact.geom1
        geom2 = contact.geom2
        
        if geom1 in gripper_geom_ids or geom2 in gripper_geom_ids:
            return True
    
    return False
```

---

## 📝 完整示例

### 使用方法1的完整环境

```python
class RobotPickPlaceEnv(RobotManipulationEnv):
    """完整的抓取放置环境，包含夹爪控制"""
    
    def __init__(
        self,
        robot_name: str = "franka_panda",
        render_mode: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            robot_name=robot_name,
            task_name="pick_place",
            controller_type="operational_space",
            render_mode=render_mode,
            **kwargs
        )
        
        # 重新定义动作空间：6维机器人 + 1维夹爪
        self.action_space = spaces.Box(
            low=np.array([-1.0] * 7),
            high=np.array([1.0] * 7),
            dtype=np.float32
        )
        
        # 夹爪配置
        self.gripper_actuator_id = self.model.actuator("gripper_actuator").id
        self.gripper_open_pos = 0.08
        self.gripper_close_pos = 0.0
    
    def reset(self, seed=None, options=None):
        """重置环境，打开夹爪"""
        obs, info = super().reset(seed=seed, options=options)
        
        # 打开夹爪
        self.data.ctrl[self.gripper_actuator_id] = self.gripper_open_pos
        mujoco.mj_forward(self.model, self.data)
        
        return obs, info
    
    def step(self, action):
        """执行一步"""
        # 分离动作
        robot_action = action[:6]  # [dx, dy, dz, droll, dpitch, dyaw]
        gripper_action = action[6]  # [-1, 1]
        
        # 机器人控制
        current_ee_pos = self.data.site("ee_site").xpos
        current_ee_quat = self.data.site("ee_site").xmat.reshape(9)[:4]  # 简化
        
        target_ee_pos = current_ee_pos + robot_action[:3] * 0.01  # 缩放
        target_ee_quat = current_ee_quat  # 简化：不改变姿态
        
        target = np.concatenate([target_ee_pos, target_ee_quat])
        tau = self.controller.compute_control(target)
        self.data.ctrl[:self.robot.dof] = tau
        
        # 夹爪控制
        gripper_pos = self._compute_gripper_position(gripper_action)
        self.data.ctrl[self.gripper_actuator_id] = gripper_pos
        
        # 步进仿真
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        
        # 获取结果
        obs = self._get_obs()
        reward = self.task.compute_reward(
            obs["achieved_goal"],
            obs["desired_goal"],
            {}
        )
        terminated = self.task._check_success(
            obs["achieved_goal"],
            obs["desired_goal"]
        )
        truncated = self.task.time_limit_exceeded()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _compute_gripper_position(self, action: float) -> float:
        """计算夹爪位置"""
        return self.gripper_close_pos + (action + 1) / 2 * (
            self.gripper_open_pos - self.gripper_close_pos
        )
    
    def _get_obs(self):
        """获取观测，包含夹爪状态"""
        obs = super()._get_obs()
        
        # 添加夹爪状态
        gripper_joint_id = self.model.joint("gripper_joint").id
        gripper_state = np.array([
            self.data.qpos[gripper_joint_id],
            self.data.qvel[gripper_joint_id],
        ])
        
        obs["observation"] = np.concatenate([
            obs["observation"],
            gripper_state,
        ])
        
        return obs


# 使用环境
if __name__ == "__main__":
    env = RobotPickPlaceEnv(render_mode="human")
    
    obs, info = env.reset()
    
    for _ in range(1000):
        # 随机动作
        action = env.action_space.sample()
        
        # 示例：先移动，再关闭夹爪
        if _ < 50:
            action[6] = 1.0  # 保持打开
        elif _ < 100:
            action[6] = -1.0  # 关闭夹爪
        else:
            action[6] = 1.0  # 打开夹爪
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
```

---

## 🎯 推荐方案

根据任务复杂度选择：

1. **简单抓取任务**: 使用方法1（Task中处理）
   - 代码最简单
   - 易于理解和维护

2. **复杂抓取任务**: 使用方法2（Controller中处理）
   - 需要精细的力控制
   - 需要实时调整夹爪参数

3. **研究和实验**: 使用方法3（混合控制器）
   - 最高灵活性
   - 可以尝试不同的控制策略

---

## 📚 参考资料

1. **夹爪硬件**:
   - Robotiq 2F-85: https://robotiq.com/products/2f85-140-adaptive-robot-gripper
   - Franka Hand: https://frankaemika.github.io/docs/

2. **抓取算法**:
   - Levine et al., "Learning Hand-Eye Coordination for Robotic Grasping with Deep Learning"
   - Mahler et al., "Dex-Net 2.0: Deep Learning to Plan Robust Grasps"

3. **力控制**:
   - Siciliano & Khatib, "Springer Handbook of Robotics"

---

