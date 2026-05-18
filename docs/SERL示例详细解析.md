# SERL 示例详细解析

## 📋 目录
1. [Panda机器人抓取示例](#1-panda机器人抓取示例)
2. [SERL异步训练示例](#2-serl异步训练示例)
3. [代码架构分析](#3-代码架构分析)

---

## 1. Panda机器人抓取示例

### 1.1 示例代码结构

**文件**: `mujoco_env/example/PandaPickCube/test_gym_env_human.py`

```python
from impl.robots.panda_pick_gym_env import PandaPickCubeGymEnv

# 1. 创建环境
env = PandaPickCubeGymEnv(action_scale=(0.1, 1))
action_spec = env.action_space

# 2. 获取MuJoCo模型和数据
m = env.model  # MuJoCo 物理模型
d = env.data   # 当前状态数据

# 3. 设置键盘回调（空格键重置环境）
def key_callback(keycode):
    if keycode == KEY_SPACE:
        global reset
        reset = True

# 4. 主循环
env.reset()
with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
    while viewer.is_running():
        if reset:
            env.reset()
            reset = False
        else:
            step_start = time.time()
            env.step(sample())  # 执行随机动作
            viewer.sync()       # 同步渲染
            # 控制频率
            time_until_next_step = env.control_dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
```

### 1.2 关键组件

#### PandaPickCubeGymEnv 环境

**位置**: `mujoco_env/impl/robots/panda_pick_gym_env.py`

**功能**:
- 继承自 `MujocoGymEnv`
- 实现了 Panda 机器人抓取立方体的仿真环境
- 提供标准的 Gym 接口：`reset()`, `step()`, `render()`

**关键参数**:
```python
env = PandaPickCubeGymEnv(
    action_scale=(0.1, 1),  # (位置缩放, 夹爪缩放)
    # 动作空间: [dx, dy, dz, gripper]
    # dx, dy, dz: 末端执行器的位置增量
    # gripper: 夹爪开合 (0=关闭, 1=打开)
)
```

**观测空间**:
- 机器人关节位置和速度
- 末端执行器位置
- 目标物体位置
- 形状: `(10,)` 状态向量

**动作空间**:
- 形状: `(4,)`
- 范围: `[-1, 1]`
- 含义: `[delta_x, delta_y, delta_z, gripper_action]`

### 1.3 控制流程

```
┌─────────────┐
│  初始化环境  │
│ env.reset() │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  启动MuJoCo     │
│  Viewer窗口     │
└──────┬──────────┘
       │
       ▼
    ┌─────────────────────┐
    │   主循环 (60Hz)     │
    │                     │
    │  1. 检查键盘输入    │
    │  2. 采样随机动作    │
    │  3. env.step()      │
    │  4. 更新物理状态    │
    │  5. 渲染画面        │
    │  6. 时间同步        │
    └─────────┬───────────┘
              │
              ▼
         直到窗口关闭
```

### 1.4 物理仿真细节

**场景配置** (`assets/scenes/PandaPickCube/arena.xml`):
- Panda 7自由度机械臂
- 2指夹爪
- 可抓取的立方体
- 桌面工作空间

**控制频率**:
- `control_dt = 0.02` (50Hz)
- 每个控制步进行多次物理仿真步

**控制器**:
- 使用操作空间控制 (Operational Space Control)
- 末端执行器的笛卡尔空间控制
- 内部使用逆运动学求解关节角度

---

## 2. SERL异步训练示例

### 2.1 架构概览

SERL 使用**异步Actor-Learner架构**：

```
┌─────────────────────────────────────────────────────────────┐
│                        训练系统                               │
│                                                               │
│  ┌──────────────┐                      ┌──────────────┐     │
│  │   Actor节点   │◄────网络参数────────│  Learner节点  │     │
│  │  (与环境交互)  │                      │  (训练策略)   │     │
│  └───────┬──────┘                      └──────▲───────┘     │
│          │                                     │              │
│          │ 经验数据 (s,a,r,s',done)           │ 从buffer采样  │
│          │                                     │              │
│          └─────────►  QueuedDataStore  ───────┘              │
│                      (共享数据队列)                            │
│                                                               │
│  ┌──────────────┐                                            │
│  │ PandaPickCube │◄── Actor与环境交互                        │
│  │  仿真环境      │                                           │
│  └──────────────┘                                            │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Learner节点详解

**文件**: `examples/async_sac_state_sim/async_sac_state_sim.py` (learner函数)

**主要职责**:
1. 初始化SAC算法
2. 创建Replay Buffer
3. 等待收集初始数据
4. 循环训练并更新策略
5. 定期同步参数到Actor

**代码流程**:

```python
def learner(rng, agent):
    # 1. 创建数据存储和服务器
    data_store = QueuedDataStore(50000)  # 容量50000条经验
    server = TrainerServer(make_trainer_config(), data_store)
    
    # 2. 创建Replay Buffer
    replay_buffer = make_replay_buffer(
        env,
        capacity=FLAGS.replay_buffer_capacity,  # 1,000,000
    )
    
    # 3. 等待填充初始数据 (training_starts = 1000步)
    pbar = tqdm.tqdm(total=FLAGS.training_starts)
    while replay_buffer.size < FLAGS.training_starts:
        batch = data_store.recv()  # 从Actor接收数据
        replay_buffer.insert(batch)
        pbar.update(batch["observations"].shape[0])
    
    # 4. 训练循环
    for step in tqdm.tqdm(range(FLAGS.max_steps)):
        # 从Replay Buffer采样
        batch = replay_buffer.sample(FLAGS.batch_size * FLAGS.utd_ratio)
        
        # 更新SAC算法
        agent, update_info = agent.update(batch, FLAGS.utd_ratio)
        
        # 定期同步参数到Actor
        if step % FLAGS.steps_per_update == 0:
            server.publish_network(agent.state.params)
        
        # 记录训练指标
        if step % FLAGS.log_period == 0:
            wandb.log(update_info, step=step)
```

**关键参数**:
- `batch_size`: 256 - 每次训练使用的样本数
- `utd_ratio`: 1.0 - Update-to-Data比率
- `critic_actor_ratio`: 8 - Critic更新8次，Actor更新1次
- `training_starts`: 1000 - 收集1000步后开始训练

### 2.3 Actor节点详解

**主要职责**:
1. 与环境交互
2. 使用当前策略采样动作
3. 收集经验数据
4. 发送数据到Learner
5. 接收更新的策略参数

**代码流程**:

```python
def actor(agent, data_store, env, sampling_rng):
    # 1. 创建客户端，连接到Learner
    client = TrainerClient(
        "actor_env",
        FLAGS.ip,  # Learner的IP地址
        make_trainer_config(),
        data_store,
    )
    
    # 2. 设置参数更新回调
    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))
    
    client.recv_network_callback(update_params)
    
    # 3. 主循环
    obs, _ = env.reset()
    for step in range(FLAGS.max_steps):
        # 采样动作
        if step < FLAGS.random_steps:  # 前1000步随机探索
            actions = env.action_space.sample()
        else:
            actions = agent.sample_actions(obs, deterministic=False)
        
        # 执行动作
        next_obs, reward, done, truncated, info = env.step(actions)
        
        # 存储经验
        data_store.insert({
            'observations': obs,
            'actions': actions,
            'next_observations': next_obs,
            'rewards': reward,
            'masks': 1.0 - done,
            'dones': done or truncated,
        })
        
        # 重置环境（如果episode结束）
        obs = next_obs
        if done or truncated:
            obs, _ = env.reset()
```

**探索策略**:
1. **前1000步**: 完全随机探索（`random_steps`）
2. **之后**: 使用SAC策略采样动作（包含探索噪声）

### 2.4 SAC算法详解

**SAC (Soft Actor-Critic)** 是一种基于最大熵的强化学习算法。

**关键特点**:
1. **Off-policy**: 可以使用Replay Buffer中的历史数据
2. **最大熵**: 鼓励策略保持随机性，提高探索
3. **双Critic**: 使用两个Q网络减少过估计

**网络结构**:
```python
Actor: 
  观测 → MLP(256, 256) → 均值和标准差 → 高斯分布 → 动作

Critic (×2):
  [观测, 动作] → MLP(256, 256) → Q值
```

**损失函数**:

1. **Critic损失**:
```
L_Q = E[(Q(s,a) - (r + γ * min(Q'(s',a')) - α*log π(a'|s')))²]
```

2. **Actor损失**:
```
L_π = E[α*log π(a|s) - Q(s,a)]
```

3. **温度参数α损失** (自动调节探索程度):
```
L_α = E[-α * (log π(a|s) + H_target)]
```

### 2.5 数据流详解

```
时间线:
t=0    Actor启动，开始随机探索
│      └─► 收集经验 → QueuedDataStore
│
t=1000 Replay Buffer填充到1000条
│      Learner开始训练
│      └─► 从Buffer采样 → 更新SAC → 发送新参数
│
t=1030 Actor收到新参数
│      └─► 使用新策略采样动作
│
...    循环训练
│
t=N    训练完成
```

**数据同步机制**:
- **Actor → Learner**: 通过`QueuedDataStore`异步发送经验
- **Learner → Actor**: 每30步(`steps_per_update`)同步一次参数
- **无阻塞**: Actor和Learner完全异步运行

---

## 3. 代码架构分析

### 3.1 核心依赖

```
serl/
├── serl_launcher/           # 强化学习算法库
│   ├── agents/              # SAC, DrQ等算法实现
│   ├── networks/            # 神经网络架构
│   ├── data/                # Replay Buffer
│   └── utils/               # 工具函数
│
├── mujoco_env/              # 机器人仿真环境
│   ├── impl/
│   │   ├── robots/          # 机器人定义
│   │   ├── controllers/     # 控制器
│   │   └── envs/            # Gym环境
│   └── example/             # 示例代码
│
└── examples/                # 训练脚本
    ├── async_sac_state_sim/ # 从状态训练
    └── async_drq_sim/       # 从图像训练
```

### 3.2 关键库

**1. AgentLace** (`agentlace`):
- 提供Actor-Learner通信框架
- `TrainerServer`: Learner端服务器
- `TrainerClient`: Actor端客户端
- `QueuedDataStore`: 共享数据队列

**2. JAX & Flax**:
- JAX: 自动微分和加速计算
- Flax: 神经网络框架
- 支持GPU/TPU加速

**3. MuJoCo**:
- 物理仿真引擎
- 精确的接触力和摩擦力仿真
- 高效的多体动力学计算

### 3.3 环境注册机制

```python
# mujoco_env/impl/robots/panda_pick_gym_env.py
gym.register(
    id="PandaPickCube-v0",
    entry_point=PandaPickCubeGymEnv,
    max_episode_steps=100,
)
```

**导入触发注册**:
```python
from mujoco_env.impl import commons  # 触发环境注册
env = gym.make("PandaPickCube-v0")   # 创建环境
```

### 3.4 训练流程总结

```
1. 初始化
   ├─ 创建环境
   ├─ 创建SAC Agent
   └─ 创建Replay Buffer

2. 数据收集阶段 (0-1000步)
   ├─ Actor: 随机探索
   └─ Learner: 等待数据

3. 训练阶段 (1000步后)
   ├─ Actor: 使用策略收集数据
   └─ Learner: 训练并更新策略
       ├─ 从Buffer采样
       ├─ 计算损失
       ├─ 梯度下降
       └─ 同步参数到Actor

4. 评估阶段 (每2000步)
   └─ 运行确定性策略评估性能
```

### 3.5 关键超参数说明

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| `batch_size` | 256 | 训练批次大小 |
| `replay_buffer_capacity` | 1,000,000 | Replay Buffer容量 |
| `random_steps` | 1000 | 随机探索步数 |
| `training_starts` | 1000 | 开始训练的步数 |
| `critic_actor_ratio` | 8 | Critic/Actor更新比率 |
| `utd_ratio` | 1.0 | 每步环境交互的梯度更新次数 |
| `steps_per_update` | 30 | 参数同步频率 |
| `max_traj_length` | 100 | 最大episode长度 |

---

## 4. 运行与调试

### 4.1 查看训练进度

**Learner输出**:
```
Filling up replay buffer: 100%|████████| 1000/1000
starting learner loop
Training: 1000/1000000 [00:10<2:45:30, 100.0it/s]
```

**Actor输出**:
```
starting actor loop
0%|          | 0/1000000 [00:00<?, ?it/s]
Episode return: 42.5
```

### 4.2 常见问题

**Q: Actor等待很久才开始？**
A: Actor在等待Learner启动。确保Learner先启动。

**Q: 训练不稳定？**
A: 尝试调整：
- 减小学习率
- 增加`batch_size`
- 增加`random_steps`

**Q: 内存不足？**
A: 减小`replay_buffer_capacity`或`batch_size`

### 4.3 修改训练参数

编辑 `run_learner.sh`:
```bash
python async_sac_state_sim.py \
    --learner \
    --batch_size 128 \          # 改小批次
    --training_starts 500 \     # 更早开始训练
    --critic_actor_ratio 4      # 减少更新比率
```

---

## 5. 扩展阅读

- **SAC论文**: [Soft Actor-Critic](https://arxiv.org/abs/1801.01290)
- **MuJoCo文档**: [mujoco.readthedocs.io](https://mujoco.readthedocs.io)
- **SERL项目**: [serl-robot.github.io](https://serl-robot.github.io)









