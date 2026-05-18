# 在真实 Franka 机械臂上运行

我们展示了如何通过 4 个不同的任务在真实机器人机械臂上使用 SERL。它们分别是：插钉任务、PCB 元件插入、电缆布线和物体搬运。我们提供了详细的说明来重现插钉任务，作为整个 SERL 包的设置测试。

在真实机器人上运行时，需要一个单独的 gym 环境。在我们的示例中，我们将 gym 环境隔离为机器人服务器的客户端。机器人服务器是一个 Flask 服务器，通过 ROS 向机器人发送命令。gym 环境通过 post 请求与机器人服务器通信。

![](./images/robot_infra_interfaces.png)


### 安装 `serl_robot_infra`

按照 `serl_robot_infra` 中的 [README](../serl_robot_infra/README.md) 进行安装和基本机器人操作说明。其中包含了安装基于阻抗的 [serl_franka_controllers](https://github.com/rail-berkeley/serl_franka_controllers) 的说明。

安装完成后，您应该能够运行机器人服务器，与 gym `franka_env`（硬件）进行交互。

> 注意：以下示例代码不会直接运行，因为它需要自定义数据、检查点和机器人环境。我们提供代码作为如何在真实机器人上使用 SERL 的参考。按增量顺序学习本节，从第一个任务（插钉）到最后一个任务（垃圾箱搬运）。根据您的需求修改代码。

## 1. 插钉任务 📍

![](./images/peg.png)

> 示例位于 [examples/async_peg_insert_drq/](../examples/async_peg_insert_drq/)

> 环境和默认配置位于 `serl_robot_infra/franka_env/envs/peg_env/`

> `franka_env.envs.wrappers.SpacemouseIntervention` gym 包装器提供了使用空间鼠标干预机器人的能力。这对于演示收集、测试机器人以及确保训练 Gym 环境按预期工作非常有用。

插钉任务是在真实机器人上开始运行 SERL 的最佳选择。在最简单的情况下，策略应该在单个 GPU 上 30 分钟内收敛并达到 100% 的成功率，因此该任务非常适合快速排查设置问题。以下步骤假设您有一个配备 Robotiq Hand-E 夹爪和 2 个 RealSense D405 相机的 Franka 机械臂。

### Procedure
1. 3D-print (1) **Assembly Object** of choice and (1) corresponding **Assembly Board** from the **Single-Object Manipulation Objects** section of [FMB](https://functional-manipulation-benchmark.github.io/files/index.html). Fix the board to the workspace and grasp the peg with the gripper.
2. 3D-print (2) wrist camera mounts for the RealSense D405 and install onto the threads on the Robotiq Gripper. Create your own config from [peg_env/config.py](../serl_robot_infra/franka_env/envs/peg_env/config.py), and update the camera serial numbers in `REALSENSE_CAMERAS`.
3. Adjust for the weight of the wrist camera by editing `Desk > Settings > End-effector > Mechnical Data > Mass`.
4. Unlock the robot and activate FCI in Desk. Then, start the franka_server by running:
    ```bash
    python serl_robo_infra/robot_servers/franka_server.py --gripper_type=<Robotiq|Franka|None> --robot_ip=<robot_IP> --gripper_ip=<[Optional] Robotiq_gripper_IP>
    ```
    This should start the impedance controller and a Flask server ready to recieve requests.
5. The reward in this task is given by checking whether the end-effector pose matches a fixed target pose. Grasp the desired peg with  `curl -X POST http://127.0.0.1:5000/close_gripper` and manually move the arm into a pose where the peg is inserted into the board. Print the current pose with `curl -X POST http://127.0.0.1:5000/getpos_euler` and update the `TARGET_POSE` in [peg_env/config.py](../serl_robot_infra/franka_env/envs/peg_env/config.py) with the measured end-effector pose.

    **Note: make sure the wrist joint is centered (away from joint limits) and z-axis euler angle is positive at the target pose to avoid discontinuities.

6. Set `RANDOM_RESET` to `False` inside the config file to speedup training. Note the policy would only generalize to any board pose when this is set to `True`, but only try this after the basic task works.
7. Record 20 demo trajectories with the spacemouse.
    ```bash
    cd examples/async_peg_insert_drq
    python record_demo.py
    ```
    The trajectories are saved in `examples/async_peg_insert_drq/peg_insertion_20_trajs_{UUID}.pkl`.
8. Edit `demo_path` and `checkpoint_path` in `run_learner.sh` and `run_actor.sh`. Train the RL agent with the collected demos by running both learner and actor nodes.
    ```bash
    bash run_learner.sh
    bash run_actor.sh
    ```
9. If nothing went wrong, the policy should converge with 100% success rate within 30 minutes without `RANDOM_RESET` and 60 minutes with `RANDOM_RESET`.
10. The checkpoints are automatically saved and can be evaluated by setting the `--eval_checkpoint_step=CHECKPOINT_NUMBER_TO_EVAL` and `--eval_n_trajs=N_TIMES_TO_EVAL` flags in `run_actor.sh`. Then run:
    ```bash
    bash run_actor.sh
    ```
    If the policy is trained with `RANDOM_RESET`, it should be able to insert the peg even when you move the board at test time.


Let's take the peg insertion task as an example. We wrapped the env as such. The composability of the gym wrappers allows us to easily add or remove functionalities to the gym env. ([code](../examples/async_peg_insert_drq/async_drq_randomized.py))

```python
env = gym.make('FrankaPegInsert-Vision-v0')  # create the gym env
env = GripperCloseEnv(env)         # always keep the gripper close for peg insertion
env = SpacemouseIntervention(env)  # utilize spacemouse to intervene the robot
env = RelativeFrame(env)           # transform the TCP abs frame of ref to relative frame
env = Quat2EulerWrapper(env)       # convert rotation from quaternion to euler
env = SERLObsWrapper(env)          # convert observation to SERL format
env = ChunkingWrapper(env)         # chunking the observation
env = RecordEpisodeStatistics(env) # record episode statistics
```


### 2. PCB Component Insertion 🖥️

![](./images/pcb.png)

> Example is located in [examples/async_pcb_insert_drq/](../examples/async_pcb_insert_drq/)

> Env and default config are located in `serl_robot_infra/franka_env/envs/pcb_env/`

Similar to peg insertion task, we define the reward in this task is given by checking whether the end-effector pose matches a fixed target pose. Update the `TARGET_POSE` in [peg_env/config.py](../serl_robot_infra/franka_env/envs/peg_env/config.py) with the measured end-effector pose.

Here we record demo trajectories with the robot, then run the learner and actor nodes.
```bash
# record demo trajectories
python record_demo.py

# run learner and actor nodes
bash run_learner.sh
bash run_actor.sh
```

A baseline of using BC as policy is also provided. To train BC, simply run the following command:
```bash
python3 examples/bc_policy.py ....TODO_ADD_ARGS.....
```

To run the BC policy, simply run the following command:
```bash
bash run_bc.sh
```

### 3. Cable Routing 🔌

![](./images/cable.png)

> Example is located in [examples/async_cable_routing_drq/](../examples/async_cable_routing_drq/)

> Env and default config are located in `serl_robot_infra/franka_env/envs/cable_env/`

In this cable routing task, we provided an example of an image-based reward classifier. This replaced the hardcoded reward classifier which depends on the known `TARGET_POSE` defined in the `config.py`. This image-based reward classifier is pretrained ResNet10, then trained to classify whether the cable is routed successfully or not. The reward classifier is trained with demo trajectories of successful and failed samples.

```bash
# NOTE: populate the custom paths to train a reward classifier
python train_reward_classifier.py \
    --classifier_ckpt_path CHECKPOINT_OUTPUT_DIR \
    --positive_demo_paths PATH_TO_POSITIVE_DEMO1.pkl \
    --positive_demo_paths PATH_TO_POSITIVE_DEMO2.pkl \
    --negative_demo_paths PATH_TO_NEGATIVE_DEMO1.pkl \
```

The reward classifier is used as a gym wrapper `franka_env.envs.wrapper.BinaryRewardClassifier`. The wrapper classifies the current observation and returns a reward of 1 if the observation is classified as successful, and 0 otherwise.

The reward classifier is then used in the BC policy and DRQ policy for the actor node, the path is provided as `--reward_classifier_ckpt_path` argument in `run_bc.sh` and `run_actor.sh`


### 4. Object Relocation 🗑️

![](./images/forward.png)

![](./images/backward.png)

> Example is located in [examples/async_bin_relocation_fwbw_drq/](../examples/async_bin_relocation_fwbw_drq/)

> Env and default config are located in `serl_robot_infra/franka_env/envs/bin_env/`

This bin relocation example demonstrates the usage of forward and backward policies. Dual-task formulation is helpful for RL tasks, helping the robot to "reset". In this case, the robot is moving an object from one bin to another. The forward policy is used to move the object from the right bin to the left bin, and the backward policy is used to move the object from the left bin to the right bin.

1. Record demo trajectories

Multiple utility scripts have been provided to record demo trajectories. (e.g. `record_demo.py`: for RLPD, `record_transitions.py` for training the reward classifier, `reward_bc_demos.py`: for bc policy). Note that both forward and backward trajectories require different demo trajectories.

2. Reward Classifier

Similar to the cable routing example, we need to train two reward classifiers for both forward and backward policies. Since the observations has both wrist camera and front camera, we use a `FrontCameraWrapper(env)` to only provide the front camera image to the reward classifier.

```bash
# NOTE: populate the custom paths to train reward classifiers for both forward and backward policies
python train_reward_classifier.py \
    --classifier_ckpt_path CHECKPOINT_OUTPUT_DIR \
    --positive_demo_paths PATH_TO_POSITIVE_DEMO1.pkl \
    --positive_demo_paths PATH_TO_POSITIVE_DEMO2.pkl \
    --negative_demo_paths PATH_TO_NEGATIVE_DEMO1.pkl \
```

The reward classifiers are then used in the BC and DRQ policy for the actor node, checkpoint path is provided as `--fw_reward_classifier_ckpt_path` and `--bw_reward_classifier_ckpt_path` argument in `run_actor.sh`. To compare with BC as baseline, provide the classifier as `--reward_classifier_ckpt_path` for the `run_bc.sh` script.

3. Run 2 learners and 1 actor with 2 policies

Finally, 2 learner nodes will learn both forward and backward policies respectively. The actor node will switch between running the forward and backward policies with their respective reward classifiers during the RL training process.

```bash
bash run_actor.sh

# run 2 learners
bash run_fw_learner.sh
bash run_bw_learner.sh
```
