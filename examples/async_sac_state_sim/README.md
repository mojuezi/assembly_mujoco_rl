# 基于SAC的仿真训练完整流程

这是一个完整的基于Soft Actor-Critic (SAC)算法的仿真训练流程，使用异步的actor-learner架构进行强化学习训练。

## 目录结构

```
examples/async_sac_state_sim/
├── async_sac_state_sim.py    # 主训练脚本
├── run_actor.sh              # Actor节点启动脚本
├── run_learner.sh            # Learner节点启动脚本
├── tmux_launch.sh            # 一键启动脚本（使用tmux）
└── README.md                 # 本文档
```

## 系统要求

1. **Python环境**: Python 3.8+
2. **依赖库**:
   - JAX
   - Flax
   - Gymnasium
   - MuJoCo
   - agentlace
   - franka_sim

3. **硬件要求**:
   - 支持CUDA的GPU（推荐，用于加速训练）
   - 或CPU（训练速度较慢）

## 安装步骤

### 1. 安装franka_sim仿真环境

```bash
cd franka_sim
pip install -e .
pip install -r requirements.txt
```

### 2. 验证安装

测试仿真环境是否正常工作：

```bash
python franka_sim/franka_sim/test/test_gym_env_human.py
```

### 3. 安装serl_launcher

```bash
cd serl_launcher
pip install -e .
```

## 训练流程

SAC训练采用**异步actor-learner架构**：
- **Actor**: 负责与环境交互，收集经验数据
- **Learner**: 负责从经验回放缓冲区中采样并更新网络参数

### 方法1: 使用tmux一键启动（推荐）

```bash
cd examples/async_sac_state_sim
bash tmux_launch.sh
```

这将自动启动两个进程：
- 一个learner节点（训练网络）
- 一个actor节点（与环境交互）

停止训练：
```bash
tmux kill-session -t serl_session
```

### 方法2: 手动启动（两个终端）

#### 终端1: 启动Learner节点

```bash
cd examples/async_sac_state_sim
bash run_learner.sh
```

或者手动运行：

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.05
python async_sac_state_sim.py \
    --learner \
    --env PandaPickCube-v0 \
    --exp_name=serl_dev_sim_test \
    --seed 0 \
    --training_starts 1000 \
    --critic_actor_ratio 8 \
    --batch_size 256 \
    --utd_ratio 1
```

#### 终端2: 启动Actor节点

```bash
cd examples/async_sac_state_sim
bash run_actor.sh
```

或者手动运行：

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.05
python async_sac_state_sim.py \
    --actor \
    --render \
    --env PandaPickCube-v0 \
    --exp_name=serl_dev_sim_test \
    --seed 0 \
    --random_steps 1000
```

## 参数说明

### 主要训练参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--env` | 环境名称 | `PandaPickCube-v0` |
| `--seed` | 随机种子 | `0` |
| `--batch_size` | 批次大小 | `256` |
| `--utd_ratio` | Update-to-data比率 | `1` |
| `--critic_actor_ratio` | Critic与Actor的更新比例 | `8` |
| `--max_steps` | 最大训练步数 | `1000000` |
| `--replay_buffer_capacity` | 经验回放缓冲区容量 | `1000000` |
| `--training_starts` | 开始训练的步数（等待缓冲区填充） | `1000` |
| `--random_steps` | 随机探索步数 | `1000` |

### 评估参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--eval_period` | 评估周期（步数） | `2000` |
| `--eval_n_trajs` | 每次评估的轨迹数 | `5` |

### 日志和检查点

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--exp_name` | 实验名称（用于wandb） | `None` |
| `--debug` | 调试模式（禁用wandb） | `False` |
| `--checkpoint_period` | 保存检查点的周期 | `0`（不保存） |
| `--checkpoint_path` | 检查点保存路径 | `None` |
| `--log_rlds_path` | RLDS日志保存路径 | `None` |
| `--preload_rlds_path` | 预加载RLDS数据路径 | `None` |

### 网络参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--ip` | Learner节点的IP地址 | `localhost` |
| `--steps_per_update` | Actor更新网络的步数间隔 | `30` |

## 训练流程详解

### 1. 初始化阶段

- 创建SAC agent（包含Actor、Critic和温度参数网络）
- 初始化经验回放缓冲区
- 建立actor-learner之间的网络连接

### 2. 数据收集阶段（前N步）

- Actor使用随机策略与环境交互
- 收集的经验数据存入经验回放缓冲区
- 等待缓冲区达到`training_starts`大小

### 3. 训练阶段

**Learner端**:
1. 从经验回放缓冲区采样批次数据
2. 使用`update_high_utd`方法更新网络：
   - 对Critic进行多次更新（根据`utd_ratio`）
   - 对Actor和温度参数进行一次更新
3. 将更新后的网络参数发送给Actor

**Actor端**:
1. 使用当前策略采样动作
2. 与环境交互，收集新的经验
3. 将经验数据发送给Learner
4. 定期接收Learner发送的更新后的网络参数
5. 定期进行评估（使用确定性策略）

### 4. 评估阶段

- 每`eval_period`步进行一次评估
- 使用确定性策略（argmax）运行`eval_n_trajs`个轨迹
- 记录平均回报等指标

## 使用其他环境

### 使用Gym标准环境

```bash
python async_sac_state_sim.py \
    --learner \
    --env HalfCheetah-v4 \
    --exp_name=halfcheetah_sac
```

### 使用自定义环境

确保你的环境遵循Gym接口，然后在脚本中注册：

```python
import gym
gym.register(
    id='YourEnv-v0',
    entry_point='your_module:YourEnv',
)
```

## 保存和加载数据

### 保存训练数据（RLDS格式）

```bash
python async_sac_state_sim.py \
    --learner \
    --log_rlds_path /path/to/save/rlds_data \
    ...
```

### 从预训练数据开始

```bash
python async_sac_state_sim.py \
    --learner \
    --preload_rlds_path /path/to/pretrained/rlds_data \
    ...
```

## 保存和加载模型检查点

### 保存检查点

```bash
python async_sac_state_sim.py \
    --learner \
    --checkpoint_period 10000 \
    --checkpoint_path /path/to/checkpoints \
    ...
```

### 加载检查点

在代码中添加检查点加载逻辑：

```python
from flax.training import checkpoints

# 在learner函数中
if FLAGS.checkpoint_path:
    agent_state = checkpoints.restore_checkpoint(
        FLAGS.checkpoint_path, 
        target=agent.state
    )
    agent = agent.replace(state=agent_state)
```

## 监控训练

### 使用WandB（推荐）

移除`--debug`标志，训练日志会自动上传到WandB：

```bash
python async_sac_state_sim.py \
    --learner \
    --exp_name=my_sac_experiment \
    # 移除 --debug
```

### 查看本地日志

在调试模式下，日志会打印到控制台，包括：
- Critic损失
- Actor损失
- 温度参数
- 评估回报
- 训练时间统计

## 常见问题

### 1. 内存不足

减少批次大小：

```bash
--batch_size 128  # 或更小
```

### 2. 训练速度慢

- 确保使用GPU：`jax.devices()`应该显示GPU设备
- 减少`critic_actor_ratio`（但可能影响性能）
- 减少`utd_ratio`

### 3. Actor和Learner连接失败

- 检查防火墙设置
- 如果在不同机器上运行，确保指定正确的IP地址：
  ```bash
  # Learner端
  --ip 0.0.0.0  # 监听所有接口
  
  # Actor端
  --ip <learner_ip_address>
  ```

### 4. 环境渲染问题

对于无头服务器（无显示器），设置：

```bash
export MUJOCO_GL=egl
# 并移除 --render 标志
```

## 性能调优建议

1. **批次大小**: 根据GPU内存调整，通常256-512效果较好
2. **UTD比率**: 对于简单任务，`utd_ratio=1`足够；复杂任务可以尝试2-4
3. **Critic-Actor比率**: 通常8-16，Critic需要更多更新
4. **经验回放缓冲区**: 根据任务复杂度调整，通常100万-1000万

## 训练示例

### 完整训练命令（PandaPickCube环境）

**Learner**:
```bash
python async_sac_state_sim.py \
    --learner \
    --env PandaPickCube-v0 \
    --exp_name=panda_pick_sac \
    --seed 42 \
    --batch_size 256 \
    --utd_ratio 1 \
    --critic_actor_ratio 8 \
    --training_starts 1000 \
    --max_steps 1000000 \
    --replay_buffer_capacity 1000000 \
    --eval_period 2000 \
    --eval_n_trajs 5 \
    --checkpoint_period 10000 \
    --checkpoint_path ./checkpoints/panda_sac
```

**Actor**:
```bash
python async_sac_state_sim.py \
    --actor \
    --env PandaPickCube-v0 \
    --exp_name=panda_pick_sac \
    --seed 42 \
    --random_steps 1000 \
    --render
```

## 下一步

- 尝试不同的超参数组合
- 在更复杂的环境中测试
- 使用图像观察（参考`async_drq_sim`示例）
- 添加演示数据（参考`async_drq_sim`的RLPD示例）

## 参考资料

- [SAC论文](https://arxiv.org/abs/1801.01290)
- [SERL文档](../docs/sim_quick_start.md)
- [Franka Sim文档](../../franka_sim/README.md)

