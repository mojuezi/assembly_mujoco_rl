# SAC (Soft Actor-Critic) 算法详解

## 📖 文件概述

**文件**: `serl_launcher/agents/continuous/sac.py`  
**总行数**: 597 行  
**核心类**: `SACAgent`

这个文件实现了 **SAC (Soft Actor-Critic)** 算法及其变体，是 SERL 框架的核心算法实现。

---

## 🎯 SAC 算法简介

### 什么是 SAC？

SAC (Soft Actor-Critic) 是一种**基于最大熵的强化学习算法**，核心思想是在最大化期望回报的同时，最大化策略的熵（随机性）。

**目标函数**:
```
J(π) = E[ Σ γ^t (r_t + α H(π(·|s_t))) ]
       │          │       └─ 策略熵（鼓励探索）
       │          └─ 即时奖励
       └─ 期望回报
```

**三大组件**:
1. **Actor (策略网络)**: 学习策略 π(a|s)
2. **Critic (价值网络)**: 评估 Q(s, a)
3. **Temperature (温度参数)**: 自动调节探索度 α

---

## 📂 文件结构

```python
sac.py (597 行)
├── 导入部分 (1-18 行)
│   ├── JAX 相关: jax, jax.numpy, flax
│   ├── 分布: distrax
│   └── SERL 组件: networks, optimizers, typing
│
├── SACAgent 类 (21-596 行)
│   ├── 前向传播方法 (33-116 行)
│   │   ├── forward_critic()
│   │   ├── forward_target_critic()
│   │   ├── forward_policy()
│   │   └── forward_temperature()
│   │
│   ├── 损失函数 (118-241 行)
│   │   ├── critic_loss_fn()      # Critic 损失
│   │   ├── policy_loss_fn()      # Actor 损失
│   │   └── temperature_loss_fn() # Temperature 损失
│   │
│   ├── 更新方法 (243-299 行)
│   │   └── update()              # 标准更新
│   │
│   ├── 采样方法 (301-320 行)
│   │   └── sample_actions()      # 采样动作
│   │
│   ├── 创建方法 (322-542 行)
│   │   ├── create()              # 通用创建
│   │   ├── create_pixels()       # 图像观测
│   │   └── create_states()       # 状态观测
│   │
│   └── 高 UTD 更新 (544-596 行)
│       └── update_high_utd()     # 高频更新
```

---

## 🔍 核心组件详解

### 1. SACAgent 类定义 (21-31 行)

```python
class SACAgent(flax.struct.PyTreeNode):
    """
    支持多种算法配置:
     - SAC (默认)
     - TD3 (固定标准差)
     - REDQ (大 ensemble)
     - SAC-ensemble (多 Critic)
    """
    state: JaxRLTrainState      # 训练状态（参数、优化器等）
    config: dict = nonpytree_field()  # 配置（不参与梯度计算）
```

**关键特点**:
- 继承自 `flax.struct.PyTreeNode`，支持 JAX 的函数式编程
- `state`: 包含所有网络参数和优化器状态
- `config`: 算法超参数（折扣因子、目标熵等）

---

## 📐 前向传播方法

### 1.1 forward_critic() - Critic 前向传播 (33-55 行)

```python
def forward_critic(
    self,
    observations: Data,        # 观测
    actions: jax.Array,        # 动作
    rng: PRNGKey,              # 随机数生成器
    *,
    grad_params: Optional[Params] = None,  # 可选参数
    train: bool = True,        # 训练模式
) -> jax.Array:
    """
    前向传播 Critic 网络
    
    输入: (观测, 动作)
    输出: Q 值估计 shape=(ensemble_size, batch_size)
    """
    if train:
        assert rng is not None, "训练时必须提供 rng"
    
    return self.state.apply_fn(
        {"params": grad_params or self.state.params},
        observations,
        actions,
        name="critic",
        rngs={"dropout": rng} if train else {},  # 训练时使用 dropout
        train=train,
    )
```

**用途**:
- 评估给定状态-动作对的 Q 值
- 支持 ensemble（多个 Critic 网络）
- 训练时使用 dropout 提高泛化能力

**示例**:
```python
# 假设 batch_size=256, critic_ensemble_size=2
obs = jnp.array([...])  # shape=(256, obs_dim)
actions = jnp.array([...])  # shape=(256, act_dim)
rng = jax.random.PRNGKey(0)

q_values = agent.forward_critic(obs, actions, rng)
# q_values.shape = (2, 256)  # 2 个 Critic，每个输出 256 个 Q 值
```

### 1.2 forward_target_critic() - Target Critic (57-69 行)

```python
def forward_target_critic(
    self,
    observations: Data,
    actions: jax.Array,
    rng: PRNGKey,
) -> jax.Array:
    """
    使用 Target 网络计算 Q 值
    Target 网络参数更新较慢，提高训练稳定性
    """
    return self.forward_critic(
        observations, 
        actions, 
        rng=rng, 
        grad_params=self.state.target_params  # 使用 target 参数
    )
```

**Target 网络的作用**:
```
问题: 直接用同一个网络计算 target 会导致不稳定
  target_q = r + γ * Q(s', a')
  ↑                    ↑
  用这个更新        用这个计算
  
解决: 使用慢更新的 Target 网络
  target_q = r + γ * Q_target(s', a')
  
  Q_target ← τ * Q + (1-τ) * Q_target  # τ=0.005 (软更新)
```

### 1.3 forward_policy() - Policy 前向传播 (71-91 行)

```python
def forward_policy(
    self,
    observations: Data,
    rng: Optional[PRNGKey] = None,
    *,
    grad_params: Optional[Params] = None,
    train: bool = True,
) -> distrax.Distribution:
    """
    前向传播 Policy 网络
    
    输入: 观测
    输出: 动作分布 (高斯分布)
    """
    return self.state.apply_fn(
        {"params": grad_params or self.state.params},
        observations,
        name="actor",
        rngs={"dropout": rng} if train else {},
        train=train,
    )
```

**返回的分布**:
```python
# 高斯分布 N(μ, σ)
distribution = agent.forward_policy(obs, rng)

# 可以进行的操作:
action = distribution.sample(seed=rng)     # 采样动作
log_prob = distribution.log_prob(action)   # 计算对数概率
mode = distribution.mode()                  # 获取均值（确定性动作）
```

### 1.4 forward_temperature() - Temperature (93-102 行)

```python
def forward_temperature(
    self, 
    *, 
    grad_params: Optional[Params] = None
) -> float:
    """
    获取当前温度参数 α
    
    温度控制探索程度:
    - α 大 → 更随机 (更多探索)
    - α 小 → 更确定 (更多利用)
    """
    return self.state.apply_fn(
        {"params": grad_params or self.state.params}, 
        name="temperature"
    )
```

**温度的作用**:
```python
# Actor 损失中的温度项
actor_loss = -E[Q(s,a) - α * log π(a|s)]
                        ↑
                    温度参数
                    
# α 的自动调整
# 目标: 保持策略熵 ≈ target_entropy
α_loss = α * (H(π) - target_entropy)
```

---

## 🎯 损失函数详解

### 2.1 critic_loss_fn() - Critic 损失 (134-191 行)

这是 SAC 算法的核心！

```python
def critic_loss_fn(self, batch, params: Params, rng: PRNGKey):
    """
    计算 Critic (Q 网络) 的损失
    
    目标: 最小化 TD 误差
    """
    batch_size = batch["rewards"].shape[0]
    
    # ===== 步骤 1: 计算下一个状态的动作 =====
    rng, next_action_sample_key = jax.random.split(rng)
    next_actions, next_actions_log_probs = self._compute_next_actions(
        batch, next_action_sample_key
    )
    # next_actions.shape = (batch_size, action_dim)
    # next_actions_log_probs.shape = (batch_size,)
    
    # ===== 步骤 2: 计算 Target Q 值 =====
    # 使用 Target 网络评估下一个状态
    target_next_qs = self.forward_target_critic(
        batch["next_observations"],
        next_actions,
        rng=rng,
    )  # shape=(critic_ensemble_size, batch_size)
    
    # ===== 步骤 3: Ensemble 子采样 (REDQ) =====
    if self.config["critic_subsample_size"] is not None:
        # REDQ: 从 N 个 Critic 中随机选 M 个
        rng, subsample_key = jax.random.split(rng)
        subsample_idcs = jax.random.randint(
            subsample_key,
            (self.config["critic_subsample_size"],),
            0,
            self.config["critic_ensemble_size"],
        )
        target_next_qs = target_next_qs[subsample_idcs]
    
    # ===== 步骤 4: 取最小 Q 值 (减少过估计) =====
    target_next_min_q = target_next_qs.min(axis=0)
    # shape=(batch_size,)
    
    # ===== 步骤 5: 计算 TD Target =====
    target_q = (
        batch["rewards"]                              # r_t
        + self.config["discount"]                     # γ
        * batch["masks"]                              # 1 - done
        * target_next_min_q                           # min Q(s', a')
    )
    
    # ===== 步骤 6: 添加熵项 (可选) =====
    if self.config["backup_entropy"]:
        temperature = self.forward_temperature()
        target_q = target_q - temperature * next_actions_log_probs
        #                     └─ 熵奖励
    
    # ===== 步骤 7: 计算当前 Q 值 =====
    predicted_qs = self.forward_critic(
        batch["observations"], 
        batch["actions"], 
        rng=rng, 
        grad_params=params
    )
    # shape=(critic_ensemble_size, batch_size)
    
    # ===== 步骤 8: 计算 MSE 损失 =====
    target_qs = target_q[None].repeat(
        self.config["critic_ensemble_size"], axis=0
    )
    
    critic_loss = jnp.mean((predicted_qs - target_qs) ** 2)
    #                      └─────────┬─────────┘
    #                           TD Error
    
    # ===== 返回损失和信息 =====
    info = {
        "critic_loss": critic_loss,
        "predicted_qs": jnp.mean(predicted_qs),  # 平均预测 Q 值
        "target_qs": jnp.mean(target_qs),        # 平均目标 Q 值
    }
    
    return critic_loss, info
```

**数学原理**:

```
Bellman 方程:
Q(s, a) = r + γ * E[Q(s', a')]

SAC 的 Bellman 方程:
Q(s, a) = r + γ * E[min_i Q_i(s', a') - α log π(a'|s')]
                    └────┬────┘         └──────┬──────┘
                    双 Q 网络            熵项（可选）
                    (减少过估计)

损失函数:
L_critic = E[(Q(s,a) - target)²]
where target = r + γ * [min Q'(s',a') - α log π(a'|s')]
```

**关键技巧**:

1. **双 Q 网络** (Double Q-Learning)
   ```python
   target_next_min_q = target_next_qs.min(axis=0)
   # 取最小值，减少 Q 值过估计
   ```

2. **Target 网络**
   ```python
   target_next_qs = self.forward_target_critic(...)
   # 使用慢更新的 target 参数，提高稳定性
   ```

3. **REDQ 子采样** (可选)
   ```python
   # 从 10 个 Critic 中随机选 2 个
   # 提高样本效率和多样性
   ```

### 2.2 policy_loss_fn() - Actor 损失 (193-221 行)

```python
def policy_loss_fn(self, batch, params: Params, rng: PRNGKey):
    """
    计算 Actor (策略网络) 的损失
    
    目标: 最大化期望回报 - 温度 * 熵
    """
    batch_size = batch["rewards"].shape[0]
    
    # ===== 步骤 1: 获取当前温度 =====
    temperature = self.forward_temperature()
    
    # ===== 步骤 2: 采样动作 =====
    rng, policy_rng, sample_rng, critic_rng = jax.random.split(rng, 4)
    
    action_distributions = self.forward_policy(
        batch["observations"], 
        rng=policy_rng, 
        grad_params=params
    )
    
    # 重参数化技巧 (reparameterization trick)
    actions, log_probs = action_distributions.sample_and_log_prob(
        seed=sample_rng
    )
    # actions.shape = (batch_size, action_dim)
    # log_probs.shape = (batch_size,)
    
    # ===== 步骤 3: 计算 Q 值 =====
    predicted_qs = self.forward_critic(
        batch["observations"],
        actions,
        rng=critic_rng,
    )  # shape=(ensemble_size, batch_size)
    
    # 取所有 Critic 的平均
    predicted_q = predicted_qs.mean(axis=0)
    
    # ===== 步骤 4: 计算 Actor 目标 =====
    actor_objective = predicted_q - temperature * log_probs
    #                 └─────┬─────┘   └────────┬──────────┘
    #                   期望回报          熵惩罚
    #                                  (鼓励探索)
    
    # ===== 步骤 5: 最大化目标 = 最小化负目标 =====
    actor_loss = -jnp.mean(actor_objective)
    
    # ===== 返回损失和信息 =====
    info = {
        "actor_loss": actor_loss,
        "temperature": temperature,
        "entropy": -log_probs.mean(),  # 策略熵
    }
    
    return actor_loss, info
```

**数学原理**:

```
SAC 的策略优化目标:
J(π) = E[Q(s, a) - α log π(a|s)]
       ↑            ↑
    价值最大化    熵最大化
    
梯度:
∇_θ J(π) = ∇_θ E[Q(s, a_θ(s)) - α log π_θ(a_θ(s)|s)]

其中 a_θ(s) 是通过重参数化采样的动作
```

**重参数化技巧**:

```python
# ❌ 不可微分的采样
a ~ N(μ(s), σ(s))
# 梯度无法反向传播

# ✅ 重参数化 (可微分)
ε ~ N(0, 1)
a = μ(s) + σ(s) * ε
# 梯度可以通过 μ 和 σ 反向传播
```

### 2.3 temperature_loss_fn() - Temperature 损失 (223-234 行)

```python
def temperature_loss_fn(self, batch, params: Params, rng: PRNGKey):
    """
    计算 Temperature (温度参数) 的损失
    
    目标: 自动调整 α，使策略熵接近目标熵
    """
    # ===== 步骤 1: 采样下一个动作 =====
    rng, next_action_sample_key = jax.random.split(rng)
    next_actions, next_actions_log_probs = self._compute_next_actions(
        batch, next_action_sample_key
    )
    
    # ===== 步骤 2: 计算熵 =====
    entropy = -next_actions_log_probs.mean()
    
    # ===== 步骤 3: 计算 Lagrange 惩罚 =====
    temperature_loss = self.temperature_lagrange_penalty(
        entropy,
        grad_params=params,
    )
    # 等价于:
    # α * (entropy - target_entropy)
    
    return temperature_loss, {"temperature_loss": temperature_loss}
```

**数学原理**:

```
约束优化问题:
max J(π) = E[Q(s,a)]
s.t. H(π) ≥ H_target

转化为无约束问题 (Lagrange 乘子法):
L = E[Q(s,a)] + α(H(π) - H_target)
    └────┬────┘   └─────────┬─────────┘
     原始目标        约束

α 的更新:
α ← α + η * (H(π) - H_target)

如果 H(π) < H_target → α 增大 → 更多探索
如果 H(π) > H_target → α 减小 → 更多利用
```

---

## 🔄 更新方法

### 3.1 update() - 标准更新 (243-299 行)

```python
@partial(jax.jit, static_argnames=("pmap_axis", "networks_to_update"))
def update(
    self,
    batch: Batch,
    *,
    pmap_axis: str = None,
    networks_to_update: FrozenSet[str] = frozenset(
        {"actor", "critic", "temperature"}
    ),
) -> Tuple["SACAgent", dict]:
    """
    对指定网络进行一次梯度更新
    
    参数:
        batch: 数据批次
            - observations: (batch_size, obs_dim)
            - actions: (batch_size, act_dim)
            - next_observations: (batch_size, obs_dim)
            - rewards: (batch_size,)
            - masks: (batch_size,)  # 1 - done
        
        pmap_axis: 多设备训练轴
        networks_to_update: 要更新的网络
            - {"critic"}: 只更新 Critic
            - {"actor", "temperature"}: 只更新 Actor 和 Temperature
            - {"actor", "critic", "temperature"}: 全部更新
    
    返回:
        (new_agent, info_dict)
    """
    batch_size = batch["rewards"].shape[0]
    chex.assert_tree_shape_prefix(batch, (batch_size,))
    
    # ===== 步骤 1: 获取损失函数 =====
    loss_fns = self.loss_fns(batch)
    # {
    #     "critic": critic_loss_fn,
    #     "actor": policy_loss_fn,
    #     "temperature": temperature_loss_fn,
    # }
    
    # ===== 步骤 2: 只计算指定网络的梯度 =====
    for key in loss_fns.keys() - networks_to_update:
        loss_fns[key] = lambda params, rng: (0.0, {})
    # 未指定的网络损失设为 0，不会更新
    
    # ===== 步骤 3: 应用损失函数并更新参数 =====
    new_state, info = self.state.apply_loss_fns(
        loss_fns, 
        pmap_axis=pmap_axis, 
        has_aux=True
    )
    # 内部执行:
    # 1. 计算梯度: grads = jax.grad(loss_fn)(params)
    # 2. 应用优化器: params = optimizer.update(grads, params)
    
    # ===== 步骤 4: 更新 Target 网络 =====
    if "critic" in networks_to_update:
        new_state = new_state.target_update(
            self.config["soft_target_update_rate"]
        )
        # 软更新:
        # target_params = τ * params + (1-τ) * target_params
        # τ = 0.005 (默认)
    
    # ===== 步骤 5: 更新 RNG =====
    rng, _ = jax.random.split(self.state.rng)
    new_state = new_state.replace(rng=rng)
    
    # ===== 步骤 6: 记录学习率 =====
    for name, opt_state in new_state.opt_states.items():
        if hasattr(opt_state, "hyperparams"):
            if "learning_rate" in opt_state.hyperparams.keys():
                info[f"{name}_lr"] = opt_state.hyperparams["learning_rate"]
    
    return self.replace(state=new_state), info
```

**使用示例**:

```python
# 示例 1: 标准更新（全部网络）
agent, info = agent.update(batch)

# 示例 2: 只更新 Critic (高 UTD)
agent, info = agent.update(
    batch, 
    networks_to_update=frozenset({"critic"})
)

# 示例 3: 只更新 Actor 和 Temperature
agent, info = agent.update(
    batch,
    networks_to_update=frozenset({"actor", "temperature"})
)
```

### 3.2 update_high_utd() - 高 UTD 更新 (544-596 行)

这是 SERL 的核心优化！

```python
@partial(jax.jit, static_argnames=("utd_ratio", "pmap_axis"))
def update_high_utd(
    self,
    batch: Batch,
    *,
    utd_ratio: int,
    pmap_axis: Optional[str] = None,
) -> Tuple["SACAgent", dict]:
    """
    高 UTD (Update-to-Data) 比率的快速更新
    
    UTD = 每步环境交互进行多少次梯度更新
    
    例如 utd_ratio=4:
    - 收集 1 步环境数据
    - 进行 4 次 Critic 更新
    - 进行 1 次 Actor/Temperature 更新
    
    这大大提高了样本效率！
    """
    batch_size = batch["rewards"].shape[0]
    assert batch_size % utd_ratio == 0, \
        f"Batch size {batch_size} 必须能被 UTD ratio {utd_ratio} 整除"
    
    minibatch_size = batch_size // utd_ratio
    
    # ===== 步骤 1: 将 batch 分成 minibatches =====
    def make_minibatch(data: jnp.ndarray):
        return jnp.reshape(
            data, 
            (utd_ratio, minibatch_size) + data.shape[1:]
        )
    
    minibatches = jax.tree_map(make_minibatch, batch)
    # 例如: batch_size=256, utd_ratio=4
    # minibatches["rewards"].shape = (4, 64)
    
    # ===== 步骤 2: 多次更新 Critic =====
    def scan_body(carry: Tuple[SACAgent], data: Tuple[Batch]):
        (agent,) = carry
        (minibatch,) = data
        
        # 只更新 Critic
        agent, info = agent.update(
            minibatch, 
            pmap_axis=pmap_axis, 
            networks_to_update=frozenset({"critic"})
        )
        return (agent,), info
    
    # 使用 jax.lax.scan 高效循环
    (agent,), critic_infos = jax.lax.scan(
        scan_body, 
        (self,), 
        (minibatches,)
    )
    # 执行 utd_ratio 次 Critic 更新
    
    # ===== 步骤 3: 平均 Critic 信息 =====
    critic_infos = jax.tree_map(
        lambda x: jnp.mean(x, axis=0), 
        critic_infos
    )
    del critic_infos["actor"]
    del critic_infos["temperature"]
    
    # ===== 步骤 4: 一次更新 Actor 和 Temperature =====
    agent, actor_temp_infos = agent.update(
        batch,  # 使用完整 batch
        pmap_axis=pmap_axis,
        networks_to_update=frozenset({"actor", "temperature"}),
    )
    del actor_temp_infos["critic"]
    
    # ===== 步骤 5: 合并信息 =====
    infos = {**critic_infos, **actor_temp_infos}
    
    return agent, infos
```

**为什么高 UTD 有效？**

```
传统 RL (UTD=1):
  收集数据 → 更新 1 次 → 收集数据 → 更新 1 次
  样本效率低

高 UTD (UTD=4):
  收集数据 → 更新 4 次 Critic + 1 次 Actor → 收集数据
  样本效率高！

关键洞察:
• Critic 更新便宜 (只需前向传播)
• Actor 更新昂贵 (需要通过 Critic)
• 多次更新 Critic，少次更新 Actor
```

**可视化流程**:

```
输入: batch_size=256, utd_ratio=4

步骤 1: 分割 batch
┌────────────────────────────────┐
│   batch (256 samples)          │
└────────────────────────────────┘
         ↓ split
┌────────┬────────┬────────┬────────┐
│mini 1  │mini 2  │mini 3  │mini 4  │
│ (64)   │ (64)   │ (64)   │ (64)   │
└────────┴────────┴────────┴────────┘

步骤 2: 循环更新 Critic
mini 1 → 更新 Critic → agent_1
mini 2 → 更新 Critic → agent_2
mini 3 → 更新 Critic → agent_3
mini 4 → 更新 Critic → agent_4

步骤 3: 一次更新 Actor
整个 batch (256) → 更新 Actor & Temperature → final_agent
```

---

## 🎨 创建方法

### 4.1 create_states() - 状态观测 Agent (486-542 行)

```python
@classmethod
def create_states(
    cls,
    rng: PRNGKey,
    observations: Data,
    actions: jnp.ndarray,
    # 网络架构
    critic_network_kwargs: dict = {
        "hidden_dims": [256, 256],
    },
    critic_ensemble_size: int = 2,
    policy_network_kwargs: dict = {
        "hidden_dims": [256, 256],
    },
    policy_kwargs: dict = {
        "tanh_squash_distribution": True,
        "std_parameterization": "uniform",
    },
    temperature_init: float = 1.0,
    **kwargs,
):
    """
    创建基于状态观测的 SAC Agent
    
    网络结构:
    Actor:  obs → MLP(256, 256) → (μ, σ) → 动作分布
    Critic: [obs, act] → MLP(256, 256) → Q 值
    """
    
    # ===== 步骤 1: 定义网络 =====
    policy_network_kwargs["activate_final"] = True
    critic_network_kwargs["activate_final"] = True
    
    # Actor (策略网络)
    policy_def = Policy(
        encoder=None,  # 无编码器（直接处理状态）
        network=MLP(**policy_network_kwargs),
        action_dim=actions.shape[-1],
        **policy_kwargs,
        name="actor",
    )
    
    # Critic (价值网络 ensemble)
    critic_cls = partial(
        Critic, 
        encoder=None, 
        network=MLP(**critic_network_kwargs)
    )
    critic_def = ensemblize(critic_cls, critic_ensemble_size)(
        name="critic"
    )
    # ensemblize: 创建多个独立的 Critic 网络
    
    # Temperature (温度参数)
    temperature_def = GeqLagrangeMultiplier(
        init_value=temperature_init,
        constraint_shape=(),
        constraint_type="geq",  # ≥ 约束
        name="temperature",
    )
    
    # ===== 步骤 2: 调用通用 create =====
    return cls.create(
        rng,
        observations,
        actions,
        actor_def=policy_def,
        critic_def=critic_def,
        temperature_def=temperature_def,
        critic_ensemble_size=critic_ensemble_size,
        **kwargs,
    )
```

**使用示例**:

```python
import jax
import jax.numpy as jnp
from serl_launcher.agents.continuous.sac import SACAgent

# 创建 Agent
rng = jax.random.PRNGKey(42)
obs_dim = 10
act_dim = 4

agent = SACAgent.create_states(
    rng=rng,
    observations=jnp.zeros((1, obs_dim)),
    actions=jnp.zeros((1, act_dim)),
    critic_network_kwargs={"hidden_dims": [256, 256]},
    policy_network_kwargs={"hidden_dims": [256, 256]},
    critic_ensemble_size=2,
    discount=0.99,
    soft_target_update_rate=0.005,
)

print("Agent 创建成功！")
print(f"Actor 参数形状: {jax.tree_map(lambda x: x.shape, agent.state.params['actor'])}")
```

### 4.2 create_pixels() - 图像观测 Agent (402-484 行)

```python
@classmethod
def create_pixels(
    cls,
    rng: PRNGKey,
    observations: Data,
    actions: jnp.ndarray,
    # 视觉编码器
    encoder_def: nn.Module,
    shared_encoder: bool = True,  # Actor 和 Critic 共享编码器
    use_proprio: bool = False,    # 是否使用本体感知信息
    **kwargs,
):
    """
    创建基于图像观测的 SAC Agent
    
    网络结构:
    Encoder: RGB image → ResNet/MobileNet → features
    Actor:   features → MLP → 动作分布
    Critic:  features + action → MLP → Q 值
    """
    
    # ===== 步骤 1: 包装编码器 =====
    encoder_def = EncodingWrapper(
        encoder=encoder_def,
        use_proprio=use_proprio,  # 是否拼接状态信息
        stop_gradient=False,       # 编码器参与梯度更新
        enable_stacking=True,      # 支持帧堆叠
    )
    
    # ===== 步骤 2: 共享或独立编码器 =====
    if shared_encoder:
        # Actor 和 Critic 共享编码器（节省参数）
        encoders = {
            "actor": encoder_def,
            "critic": encoder_def,
        }
    else:
        # Actor 和 Critic 独立编码器（更灵活）
        encoders = {
            "actor": encoder_def,
            "critic": copy.deepcopy(encoder_def),
        }
    
    # ===== 步骤 3: 定义网络 =====
    policy_def = Policy(
        encoder=encoders["actor"],
        network=MLP(**policy_network_kwargs),
        action_dim=actions.shape[-1],
        **policy_kwargs,
        name="actor",
    )
    
    critic_backbone = partial(MLP, **critic_network_kwargs)
    critic_backbone = ensemblize(
        critic_backbone, 
        critic_ensemble_size
    )(name="critic_ensemble")
    
    critic_def = partial(
        Critic, 
        encoder=encoders["critic"], 
        network=critic_backbone
    )(name="critic")
    
    # ... (其余同 create_states)
```

**使用示例**:

```python
from serl_launcher.vision.resnet_v1 import ResNetV1
from serl_launcher.agents.continuous.sac import SACAgent

# 创建视觉 Agent
rng = jax.random.PRNGKey(42)
image_shape = (128, 128, 3)
act_dim = 4

# 定义编码器
encoder = ResNetV1(
    stage_sizes=(2, 2, 2, 2),
    features=(16, 32, 64, 128),
)

agent = SACAgent.create_pixels(
    rng=rng,
    observations={"image": jnp.zeros((1,) + image_shape)},
    actions=jnp.zeros((1, act_dim)),
    encoder_def=encoder,
    shared_encoder=True,
    use_proprio=False,
)

print("视觉 Agent 创建成功！")
```

---

## 📊 训练流程示例

### 完整训练循环

```python
import jax
import jax.numpy as jnp
from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.data.replay_buffer import ReplayBuffer

# ===== 1. 创建 Agent =====
rng = jax.random.PRNGKey(42)
agent = SACAgent.create_states(
    rng=rng,
    observations=jnp.zeros((1, 10)),
    actions=jnp.zeros((1, 4)),
)

# ===== 2. 创建 Replay Buffer =====
buffer = ReplayBuffer(capacity=1000000)

# ===== 3. 训练循环 =====
for step in range(max_steps):
    # 3.1 采样动作
    rng, sample_key = jax.random.split(rng)
    action = agent.sample_actions(obs, seed=sample_key)
    
    # 3.2 执行动作
    next_obs, reward, done, truncated, info = env.step(action)
    
    # 3.3 存储经验
    buffer.insert({
        'observations': obs,
        'actions': action,
        'next_observations': next_obs,
        'rewards': reward,
        'masks': 1.0 - done,
    })
    
    # 3.4 训练 (UTD=4)
    if step >= training_starts:
        batch = buffer.sample(batch_size=256 * 4)
        agent, info = agent.update_high_utd(batch, utd_ratio=4)
        
        if step % log_period == 0:
            print(f"Step {step}")
            print(f"  Critic Loss: {info['critic_loss']:.4f}")
            print(f"  Actor Loss: {info['actor_loss']:.4f}")
            print(f"  Q Value: {info['predicted_qs']:.2f}")
            print(f"  Entropy: {info['entropy']:.2f}")
```

---

## 🎓 关键概念总结

### 1. 最大熵强化学习

```
传统 RL 目标:
max E[Σ γ^t r_t]

最大熵 RL 目标:
max E[Σ γ^t (r_t + α H(π(·|s_t)))]
                    └─ 熵奖励

优势:
• 自动探索
• 学习多模态策略
• 提高鲁棒性
```

### 2. Actor-Critic 架构

```
Actor (策略):
• 学习策略 π(a|s)
• 最大化 E[Q(s,a) - α log π(a|s)]

Critic (价值):
• 学习 Q 函数 Q(s,a)
• 最小化 TD 误差 (Q - target)²

协同工作:
• Critic 告诉 Actor 哪些动作好
• Actor 提供动作让 Critic 评估
```

### 3. 双 Q 网络

```
问题: 单 Q 网络容易过估计

解决: 使用两个 Q 网络，取最小值
Q_target = min(Q_1(s',a'), Q_2(s',a'))

效果: 减少过估计偏差
```

### 4. 软目标更新

```
硬更新 (DQN):
每 N 步: Q_target ← Q

软更新 (SAC):
每步: Q_target ← τ * Q + (1-τ) * Q_target
      (τ = 0.005)

优势: 更稳定的训练
```

### 5. 高 UTD

```
标准训练:
收集 1 步 → 更新 1 次

高 UTD (utd_ratio=4):
收集 1 步 → 更新 4 次 Critic + 1 次 Actor

优势: 大幅提高样本效率！
```

---

## 🔧 常用配置

### 推荐配置

```python
# 标准 SAC
agent = SACAgent.create_states(
    rng=rng,
    observations=obs,
    actions=actions,
    critic_network_kwargs={"hidden_dims": [256, 256]},
    policy_network_kwargs={"hidden_dims": [256, 256]},
    critic_ensemble_size=2,
    discount=0.99,
    soft_target_update_rate=0.005,
    target_entropy=-action_dim / 2,
    backup_entropy=True,
)

# REDQ (高样本效率)
agent = SACAgent.create_states(
    ...
    critic_ensemble_size=10,
    critic_subsample_size=2,
)

# TD3 (确定性策略)
agent = SACAgent.create_states(
    ...
    policy_kwargs={
        "std_parameterization": "fixed",
        "fixed_std": 0.1,
    },
)
```

---

## 📚 参考资料

### 论文

1. **SAC**: [Soft Actor-Critic](https://arxiv.org/abs/1801.01290)
2. **SAC Applications**: [SAC for Real-World Robots](https://arxiv.org/abs/1812.05905)
3. **REDQ**: [Randomized Ensembled Double Q-Learning](https://arxiv.org/abs/2101.05982)

### 代码资源

- SERL GitHub: https://github.com/rail-berkeley/serl
- JAX 文档: https://jax.readthedocs.io
- Flax 文档: https://flax.readthedocs.io

---

## 🎉 总结

`sac.py` 实现了完整的 SAC 算法及其变体，包括：

✅ **三大组件**: Actor, Critic, Temperature  
✅ **核心机制**: 最大熵、双 Q 网络、软目标更新  
✅ **高级特性**: 高 UTD、REDQ、Ensemble  
✅ **两种模式**: 状态观测、图像观测

**关键优势**:
- 自动探索（最大熵）
- 样本高效（高 UTD）
- 训练稳定（双 Q + 软更新）
- 易于扩展（模块化设计）

这是 SERL 框架的核心算法，理解它对掌握整个项目至关重要！



