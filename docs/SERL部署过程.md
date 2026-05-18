

# SERL

| 修订日期   | 修订版本 | 修订内容   | 修订人 |
| ---------- | -------- | ---------- | ------ |
| 2025.12.15 | V0.1     | 初始化文档 | 高振宇 |

SERL 的三进程并行设计是其适配真实世界机器人训练的关键，各进程功能与协作逻辑如下：

| 进程名称                        | 核心功能                                                     | 与其他进程的交互                                             |
| ------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Actor（执行者）                 | 根据当前学习到的策略，从机器人环境的观测（如图像、传感器数据）中选择动作（如机械臂关节角度、 gripper 开合指令）。 | 接收机器人环境的实时观测 → 输出动作 → 发送给机器人环境执行。 |
| Learner Node（学习器节点）      | 运行强化学习训练算法（如 SAC、PPO），从机器人环境反馈的数据中更新策略模型（ Actor 所依赖的决策模型）。 | 接收机器人环境收集的 “状态 - 动作 - 奖励 - 下一状态” 数据 → 训练并优化策略 → 将更新后的策略同步给 Actor。 |
| Robot Environment（机器人环境） | 物理世界中的机器人本体及周边环境，执行 Actor 发送的动作，记录交互数据并计算奖励。 | 接收 Actor 的动作 → 执行动作并采集新状态（如物体位置变化、图像帧） → 生成奖励 → 将完整数据（状态、动作、奖励等）发送给 Learner。 |



## 部署过程问题

### Preperation

Ubuntu：22.04

Conda：>=4.5.11

### Installtion

1. 创建并激活serl虚拟环境

   ```
   conda create -n serl python=3.10
   conda activate serl
   ```
   
   > 1. 如果出现以下错误：
   >
   >    An HTTP error occurred when trying to retrieve this URL.
   >    HTTP errors are often intermittent, and a simple retry will get you on your way.
   >
   >    CondaHTTPError: HTTP 404 NOT FOUND for url <https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/sqlite-3.45.3-h5eee18b_0.tar.bz2>
   >    Elapsed: 00:01.436482
   >
   >    解决方法：使用官方源 `conda create -n serl python=3.10 --channel defaults`
   
2. 安装Jax库

   ```
   pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "jax[cpu]==0.4.33" "jaxlib==0.4.33"
   ```
   
   **note: jax版本非常重要！**

3. 安装serl_launcher模块

   ```
   cd serl_launcher
   pip install -e .
   或者加清华源（推荐）
   pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

   > 1. 可能打印以下错误：
   >    error: subprocess-exited-with-error
   >
   >    × git clone --filter=blob:none --quiet https://github.com/youliangtan/agentlace.git /tmp/pip-install-xheuishl/agentlace_88f91ec298bb45c881e9b860477b56de did not run successfully.
   >    │ exit code: 128
   >    ╰─> See above for output.
   >
   >    note: This error originates from a subprocess, and is likely not a problem with pip.
   >    error: subprocess-exited-with-error
   >
   >    × git clone --filter=blob:none --quiet https://github.com/youliangtan/agentlace.git /tmp/pip-install-xheuishl/agentlace_88f91ec298bb45c881e9b860477b56de did not run successfully.
   >    │ exit code: 128
   >    ╰─> See above for output.
   >
   >    解决方法：
   >
   >    终端执行 
   >
   >    ```
   >    nano ~/.gitconfig
   >    ```
   >
   >    #[url "https://ghproxy.com/https://github.com"]
   >
   >    #insteadOf = https://github.com
   >
   >    将上述两行注释，保存退出。重新执行 `pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple`

   ```
   pip install -r requirements.txt
   或者加清华源（推荐）
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple  
   ```

   > 1. 如果出现以下错误：
   >
   >    error: resolution-too-deep
   >
   >    × Dependency resolution exceeded maximum depth
   >    ╰─> Pip cannot resolve the current dependencies as the dependency graph is too complex for pip to solve efficiently.
   >
   >    hint: Try adding lower bounds to constrain your dependencies, for example: 'package>=2.0.0' instead of just 'package'. 
   >
   >    解决方法：升级pip
   >
   >    ```
   >    pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
   >    ```
   >
   >    重新执行 `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`  
   >
   >    如果仍出现 pip依赖解析器深度超限 的错误，需要单独安装：执行 pip list，对照 requirements.txt单独安装未安装成功的依赖包，单独安装时最好指定requirements.txt中要求的版本，否则后续很容易遇到版本不兼容问题
   >
   > 2. 如果出现错误，可先忽略
   >
   >    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
   >    opencv-python 4.12.0.88 requires numpy<2.3.0,>=2; python_version >= "3.9", but you have numpy 1.26.4 which is incompatible.

### Quick Start with SERL in Sim

1. 安装 `mujoco_env` 仿真环境（SERL 仓库自带子模块）

   ```
   cd ../mujoco_env
   pip install -e . 
   ```

   > - 只需在新的 Python 虚拟环境里首次运行一次 `pip install -e .`，后续 demo 直接复用已安装的 `mujoco_env` 包即可。

2. 验证仿真环境是否正常工作

   ```
   python example/PandaPickCube/test_gym_env_human.py
   ```

   > - 如果遇到 `AssertionError: The width: None and height: None cannot be 'None' when the render_mode is not 'human'.`
   >   可编辑 `example/PandaPickCube/test_gym_env_human.py`，将环境构造函数改为：
   >   ```
   >   env = envs.PandaPickCubeGymEnv(
   >       action_scale=(0.1, 1),
   >       render_mode="human",
   >   )
   >   ```
   >   再次运行即可打开渲染窗口。

#### Training from state observation example

1. 运行learner节点

   ```
   cd ../examples/async_sac_state_sim
   bash run_learner.sh 
   ```

   > 1. 如果出现以下错误：
   >    AttributeError: module 'jax.experimental.layout' has no attribute 'Format'
   >
   >    问题原因： JAX/Jaxlib 与Orbax版本不匹配
   >
   >    解决方法：保持旧版 JAX，降低 Orbax 版本，运行 `pip install "orbax-checkpoint==0.5.5" -i https://pypi.tuna.tsinghua.edu.cn/simple`

2. 运行actor节点

   另起终端，还是在async_sac_state_sim路径下，执行

   ```
   bash run_actor.sh
   ```

#### Training from image observation example

```
cd examples/async_drq_sim
wget https://github.com/rail-berkeley/serl/releases/download/resnet10/resnet10_params.pkl
bash run_learner.sh
```

另起终端，还是在async_drq_sim路径下，执行

```
bash run_actor.sh
```

#### Training from image observation with 20 demo trajectories example

```
cd examples/async_drq_sim
wget https://github.com/rail-berkeley/serl/releases/download/resnet10/resnet10_params.pkl

wget \
https://github.com/rail-berkeley/serl/releases/download/franka_sim_lift_cube_demos/franka_lift_cube_image_20_trajs.pkl
bash run_learner.sh --demo_path franka_lift_cube_image_20_trajs.pkl
```

另起终端，还是在async_drq_sim路径下，执行

```
bash run_actor.sh
```

