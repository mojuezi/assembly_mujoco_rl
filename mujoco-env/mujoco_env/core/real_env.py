"""
真机环境基类

作者: Liu Gang
日期: 2025-12-20
更新: 2025-12-24 - Phase 2: 完整实现真机环境
"""

from typing import Dict, Optional, Tuple, Any
import numpy as np
from gymnasium import spaces
import time
from dm_robotics.transformations import transformations as tr
import pdb

from mujoco_env.mujoco_env.core.base_env import BaseRobotEnv
from mujoco_env.mujoco_env.tasks.base_task import BaseTask
from mujoco_env.mujoco_env.real.robot_interface import RobotInterface
from ..controllers import get_controller, BaseController

import threading


class RealRobotEnv(BaseRobotEnv):
    """
    真机环境基类
    
    提供与真实机器人交互的完整Gymnasium环境接口。
    
    设计思路:
    - robot_interface: 负责与真实机器人的底层通信
    - task: 定义任务目标、奖励函数、成功判断
    - 环境本身: 实现Gymnasium标准接口
    
    使用示例:
        ```python
        from mujoco_env.mujoco_env.real import get_robot_interface
        from mujoco_env.mujoco_env.tasks import PickCubeTask
        from mujoco_env.mujoco_env.tasks.base_task import RobotConfig
        
        # 创建机器人接口
        robot_interface = get_robot_interface(
            robot_type="franka",
            robot_ip="192.168.1.1"
        )
        
        # 创建任务
        robot_config = RobotConfig(name="franka", dof=7, ...)
        task = PickCubeTask(robot_config=robot_config)
        
        # 创建环境
        env = RealRobotEnv(
            robot_interface=robot_interface,
            task=task
        )
        
        # 使用环境
        obs, info = env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
        env.close()
        ```
    """
    
    def __init__(
        self,
        robot_interface: RobotInterface,
        task: BaseTask,
        control_freq: float = 20.0,
        max_episode_steps=500,
        **controller_kwargs
    ):
        """
        初始化真机环境
        
        Args:
            robot_interface: 机器人接口实例（需要实现 RobotInterface）
            task: 任务实例（需要实现 BaseTask）
            control_freq: 控制频率 (Hz)
            render_mode: 渲染模式（真机环境通常不需要渲染）
        """
        self.robot_interface = robot_interface
        self.task = task
        self.control_freq = control_freq
        self.control_dt = 1.0 / control_freq
        
        # 从 task 中获取配置
        self.robot_config = task.robot_config
        self.obs_config = task.obs_config
        self.max_episode_steps = max_episode_steps
        
        # Episode 状态
        self.current_step = 0
        self._desired_goal = None

        if self.robot_interface.connect(): 
            self.robot_interface.power_on()

        # 连接机器人
        if not self.robot_interface.is_connected:
            if not self.robot_interface.connect():
                raise ConnectionError(
                    f"Failed to connect to robot at {self.robot_interface.robot_ip}"
                )
        
        self.robot_interface.robot_config.setTcpOffset([0.0, 0.0, 0.1, 0.0, 0.0, 0.0])
        self.robot_interface.robot_config.setCollisionLevel(1)

        # 初始化父类
        super().__init__(render_mode=None)
        
        # 设置观测和动作空间（必须在 task 创建之后）
        self._setup_spaces()

        # self.init_qpos = None
        controller_kwargs = controller_kwargs.copy()
        self.controller = get_controller(
            self.robot_config.controller_type,
            model=None,
            data=None,
            dof=self.robot_config.dof,
            control_freq=1.0 / self.control_dt,
            ee_site_name="for_real", 
            mode="for_real", 
            robot_interface=robot_interface, 
            **controller_kwargs
        )
        # self.robot_config.controller_type = "position"

        self.wrench_offset = np.array(self.robot_interface.get_robot_state()["tcp_wrench"]).astype(np.float32)
        # self.init_qpos = np.array([167.0 / 180.0 * np.pi,  -2.40 / 180.0 * np.pi, 120.0 / 180.0 * np.pi, 31.0 / 180.0 * np.pi, 90.7 / 180.0 * np.pi, -3.0 / 180.0 * np.pi])
        self.init_qpos = np.array([177.0 / 180.0 * np.pi,  4.0 / 180.0 * np.pi, 95.0 / 180.0 * np.pi, 1.7 / 180.0 * np.pi, 89 / 180.0 * np.pi, 20.0 / 180.0 * np.pi]) # The VLA arm
        self.init_tcppose = np.array([-0.3944, 0.21332, 0.15174, 0.00247, -0.01343,  0.76602, -0.64267])


    
    def _setup_spaces(self):
        """
        设置观测空间和动作空间
        
        观测空间由三部分组成：
        1. 机器人本体感知（qpos, qvel, tcp_pose, tcp_vel）
        2. 任务相关观测（从 task.get_obs_space() 获取）
        3. 传感器观测（force, torque 等）
        """
        # ========== Action Space ==========
        # 优先使用任务定义的动作空间，否则使用机器人DOF
        if hasattr(self.task, 'get_action_space'):
            self.action_space = self.task.get_action_space()
        else:
            # 默认使用机器人DOF（向后兼容）
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.robot_config.dof,),
                dtype=np.float32
            )
        
        # ========== Observation Space ==========
        obs_dict = {}
        
        # 1. 机器人本体感知（始终包含）
        # obs_dict["qpos"] = spaces.Box(
        #     low=-np.inf,
        #     high=np.inf,
        #     shape=(self.robot_config.dof,),
        #     dtype=np.float32
        # )
        # obs_dict["qvel"] = spaces.Box(
        #     low=-np.inf,
        #     high=np.inf,
        #     shape=(self.robot_config.dof,),
        #     dtype=np.float32
        # )
        # obs_dict["tcp_pose"] = spaces.Box(
        #     low=-np.inf,
        #     high=np.inf,
        #     shape=(7,),  # [x, y, z, qw, qx, qy, qz]
        #     dtype=np.float32
        # )
        # obs_dict["tcp_vel"] = spaces.Box(
        #     low=-np.inf,
        #     high=np.inf,
        #     shape=(6,),  # [vx, vy, vz, wx, wy, wz]
        #     dtype=np.float32
        # )

        obs_dict["tcp_pos"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3,),
            dtype=np.float32
        )

        obs_dict["tcp_quat"] = spaces.Box(
            low=-1,
            high=1,
            shape=(4,),
            dtype=np.float32
        )
        
        # 2. 任务相关观测（从 task.get_obs_space() 获取）
        if self.task is not None:
            task_obs_space = self.task.get_obs_space()
            if isinstance(task_obs_space, spaces.Dict):
                obs_dict.update(task_obs_space.spaces)
        
        # 3. 传感器观测（力/力矩）
        # if self.obs_config.include_proprioception:
        #     obs_dict["tcp_force"] = spaces.Box(
        #         low=-np.inf,
        #         high=np.inf,
        #         shape=(3,),
        #         dtype=np.float32
        #     )
        #     obs_dict["tcp_torque"] = spaces.Box(
        #         low=-np.inf,
        #         high=np.inf,
        #         shape=(3,),
        #         dtype=np.float32
        #     )
        
        self.observation_space = spaces.Dict(obs_dict)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        重置环境
        
        Args:
            seed: 随机种子
            options: 额外选项
        
        Returns:
            observation: 初始观测
            info: 额外信息
        """
        super().reset(seed=seed)
        
        # 重置步数
        self.current_step = 0
        
        # 清除机器人错误
        # self.robot_interface.clear_errors()

        # TODO: First raise the tcp upwards
        # self.robot_interface.move_to_joint_positions(self.init_qpos, 
        #                                              velocity=0.5, 
        #                                              acceleration=0.5, 
        #                                              blocking=True)
        #===========TODO thread: move to init
        # self.controller.set_self_target(self.robot_interface.robot_algorithm.forwardKinematics(self.init_qpos))
        # pdb.set_trace()

        
        # 重置任务
        if self.task is not None:
            task_info = self.task.reset()
            
            # 获取期望目标
            self._desired_goal = self.task.sample_goal()
        

        # if hasattr(self.controller, 'reset'):
        #     self.controller.reset()

        # if hasattr(self.controller, 'set_selection_vector'): 
        #     self.controller.set_selection_vector(np.ones(6))
        #     self.controller.set_selection_vector(np.zeros(6))
        #     self.controller.set_selection_vector(np.array([1, 1, 1, 0, 0, 0]))
            # self.controller.set_selection_vector(np.array([0, 0, 0, 1, 1, 1]))
        
        # 获取初始观测
        obs, obs_norm = self._get_obs()
        
        # 构建info
        info = {
            "is_success": False,
            "episode_step": self.current_step, 
            "euler": np.array(self.robot_interface.robot_state.getTcpPose()[3:]).astype(np.float32)
        }
        if self.task is not None:
            info.update(self.task.get_info())
        
        info["obs"] = obs
        
        return obs, info
    
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        执行一步动作
        
        Args:
            action: 动作，shape=(dof,)，范围 [-1, 1]
        
        Returns:
            observation: 新的观测
            reward: 奖励
            terminated: 是否达成任务目标
            truncated: 是否超时
            info: 额外信息
        """
        # 记录开始时间（用于控制频率）
        # start_time = time.time()

        # 1. 发送动作到机器人
        # self._set_joint_action(action)
        # print("real_env_time1: ", time.time())
        # self._set_tcp_action(action)
        # self.action_for_set_action = action.copy()
        # print("real_env_time2: ", time.time())
        
        # 2. 等待一个控制周期（保持控制频率）
        # elapsed = time.time() - start_time
        # if elapsed < self.control_dt:
        #     time.sleep(self.control_dt - elapsed)
        
        # 3. 获取新的观测
        obs, obs_norm = self._get_obs()
        self.current_step += 1
        
        # 4. 计算奖励
        reward = self._compute_reward()
        
        # 5. 检查是否终止
        terminated = self._check_terminated(obs)
        if terminated: reward += 1000
        
        # 6. 检查是否超时
        truncated = False
        if self.current_step >= self.max_episode_steps: 
            truncated = True
        # truncated = truncated or self.robot_interface.isCollisionOccurred() # TODO
        if hasattr(self.task, 'is_out_of_workspace'): 
            truncated = truncated or self.task.is_out_of_workspace(obs) 
            if self.task.is_out_of_workspace(obs): reward -= 100000 
        
        # 7. 构建info
        info = {
            "is_success": terminated,
            "episode_step": self.current_step,
        }
        if self.task is not None:
            info.update(self.task.get_info())
        
        info["obs"] = obs
        info["qpos"] = self.robot_interface.get_joint_positions()
        info["euler"] = np.array(self.robot_interface.robot_state.getTcpPose()[3:]).astype(np.float32)

        if self.task.obs_norm: obs = obs_norm
        
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self) -> Dict[str, np.ndarray]:
        """
        获取当前观测
        
        从机器人接口获取实时状态，并组合任务相关观测
        
        Returns:
            observation: 观测字典
        """
        obs = {}
        obs_norm = {}
        
        # 1. 从机器人接口获取状态
        robot_state = self.robot_interface.get_robot_state()
        
        # 机器人本体感知
        # obs["qpos"] = np.array(robot_state["joint_positions"]).astype(np.float32)
        # obs["qvel"] = np.array(robot_state["joint_velocities"]).astype(np.float32)
        obs["tcp_pos"] = np.array(robot_state["tcp_pose"][:3]).astype(np.float32)
        euler = np.array(robot_state["tcp_pose"][3:]).astype(np.float32)
        obs["tcp_quat"] = tr.euler_to_quat(euler, 'ZYX')
        # obs["tcp_vel"] = np.array(robot_state["tcp_velocity"]).astype(np.float32)
        
        # 力/力矩传感器
        if self.obs_config.include_proprioception:
            # obs["tcp_force"] = np.array(robot_state["tcp_force"]).astype(np.float32)
            # obs["tcp_torque"] = np.array(robot_state["tcp_torque"]).astype(np.float32)
            wrench_tmp = np.array(robot_state["tcp_wrench"]).astype(np.float32) - self.wrench_offset
            for i in range(wrench_tmp.shape[0]): 
                if wrench_tmp[i] > -0.5 and wrench_tmp[i] < 0.5: wrench_tmp[i] = 0
            obs["wrench"] = wrench_tmp
        
        # 2. 任务相关观测
        if self.task is not None:
            # 获取 achieved_goal 和 desired_goal
            # obs["achieved_goal"] = self.task.get_achieved_goal(obs).astype(np.float32)
            
            if self._desired_goal is not None:
                obs["desired_goal"] = self._desired_goal.astype(np.float32)
            else:
                # 如果 desired_goal 未设置，采样一个
                obs["desired_goal"] = self.task.sample_goal().astype(np.float32)
        
        if self.task.obs_norm: 
            obs_norm["tcp_pos"] = (obs["tcp_pos"] - self.task.center) / self.task.workspace_high
            obs_norm["tcp_quat"] = obs["tcp_quat"]
            if obs["desired_goal"] is None:
                obs_norm["desired_goal"] = np.zeros(3)
            else: 
                obs_norm["desired_goal"] = (obs["desired_goal"] - self.task.center) / self.task.workspace_high
            wrench_norm = np.zeros(6)
            wrench_norm[:3] = obs["wrench"][:3] / 50.0 # TODO
            wrench_norm[3:] = obs["wrench"][3:] / 10.0 # TODO
            obs_norm["wrench"] = wrench_norm
        
        for key in obs.keys(): 
            obs[key] = np.around(obs[key], 5)
        for key in obs_norm.keys(): 
            obs_norm[key] = np.around(obs_norm[key], 5)
        
        return obs, obs_norm
    
    def _set_joint_action(self, action: np.ndarray):
        """
        发送动作到机器人
        
        将归一化的动作转换为实际控制指令并发送到机器人
        
        Args:
            action: 归一化动作，shape=(dof,)，范围 [-1, 1]
        """
        # 根据 controller_type 处理动作
        controller_type = self.robot_config.controller_type
        
        if controller_type in ["joint_position", "position"]:
            # 关节位置控制
            # 将 [-1, 1] 映射到关节限位
            current_qpos = self.robot_interface.get_joint_positions()
            
            # 简单的增量控制: delta_qpos = action * max_delta
            max_delta = 0.1  # 最大增量（弧度）
            target_qpos = current_qpos + action * max_delta
            
            # 发送目标位置（非阻塞）
            self.robot_interface.move_to_joint_positions(
                positions=target_qpos,
                velocity=0.5,
                acceleration=0.5,
                blocking=False
            )
        
        elif controller_type in ["joint_velocity", "velocity"]:
            # 关节速度控制
            # action 直接作为速度指令
            # 注意：需要机器人接口支持速度控制
            raise NotImplementedError(
                "Joint velocity control not yet implemented for RealRobotEnv"
            )
        
        elif controller_type in ["joint_torque", "torque"]:
            # 关节力矩控制
            # action 直接作为力矩指令
            raise NotImplementedError(
                "Joint torque control not yet implemented for RealRobotEnv"
            )
        
        else:
            raise ValueError(
                f"Unknown controller type: {controller_type}. "
                f"Supported types: joint_position, joint_velocity, joint_torque"
            )
    
    def _set_tcp_action(self, action: np.ndarray): # TODO
        """
        发送动作到机器人
        
        将归一化的动作转换为实际控制指令并发送到机器人
        
        Args:
            action: 归一化动作，shape=(dof,)，范围 [-1, 1]
        """
        # 根据 controller_type 处理动作
        controller_type = self.robot_config.controller_type
        dof = self.robot_config.dof

        if len(action) >= 7:
            robot_action = action[:7]
        else:
            robot_action = action
        
        # print("time1: ", time.time())
        current_state = {
            "qpos": self.robot_interface.get_joint_positions(), 
            "qvel": self.robot_interface.get_joint_velocities(), 
            "wrench": self.robot_interface.get_tcp_wrench() - self.wrench_offset
        }

        print("wrench: ", current_state['wrench'])
        print("set_tcp_time1: ", time.time())
        ctrl_signal = self.controller.compute_control(
            target=robot_action,
            current_state=current_state, 
            external_force= current_state["wrench"], 
        )
        print("set_tcp_time2: ", time.time())
        # print(f'ctrl_signal={ctrl_signal}')

        # if controller_type in ["joint_position", "position"]:
        #     # 关节位置控制
        #     # 将 [-1, 1] 映射到关节限位
        #     current_qpos = self.robot_interface.get_joint_positions()
            
        #     # 简单的增量控制: delta_qpos = action * max_delta
        #     max_delta = 0.1  # 最大增量（弧度）
        #     target_qpos = current_qpos + action * max_delta
    
        # elif controller_type in ["joint_velocity", "velocity"]:
        #     # 关节速度控制
        #     # action 直接作为速度指令
        #     # 注意：需要机器人接口支持速度控制
        #     raise NotImplementedError(
        #         "Joint velocity control not yet implemented for RealRobotEnv"
        #     )
        
        # elif controller_type in ["joint_torque", "torque"]:
        #     # 关节力矩控制
        #     # action 直接作为力矩指令
        #     raise NotImplementedError(
        #         "Joint torque control not yet implemented for RealRobotEnv"
        #     )
        
        # else:
        #     raise ValueError(
        #         f"Unknown controller type: {controller_type}. "
        #         f"Supported types: joint_position, joint_velocity, joint_torque"
        #     )

        # 发送目标位置
        ###========================================================================================
        
        # self.robot_interface.move_tcp_pose(
        #     ctrl_signal,
        #     velocity=0.5,
        #     acceleration=5,
        #     blocking=True,
        # )
        
        ###========================================================================================

        ###========================================================================================
        # pose = np.zeros(6)
        # ret = self.robot_interface.enable_servo_mode(True)
        # if ret:
        #     qnear = self.robot_interface.robot_state.getJointPositions()
        #     pose[:3] = ctrl_signal[:3]
        #     pose[3:] = tr.quat_to_euler(ctrl_signal[3:], 'ZYX')
        #     q_pos, flag = self.robot_interface.robot_algorithm.inverseKinematics(qnear, pose)
        #     print("flag: ", flag)
        #     if flag != 0:
        #         q_pos = qnear
        #     q_pos = np.array(q_pos)
        #     ret = self.robot_interface.servo_to_joint_positions(q_pos)
        #     print("ret: ", ret)
        # self.robot_interface.enable_servo_mode(False)
        ###========================================================================================
        # if not self.robot_interface.robot_motion.isServoModeEnabled(): 
        ret = self.robot_interface.enable_servo_mode(True)
        if ret:
            ret = self.robot_interface.servo_to_tcp_positions(ctrl_signal)
            print("ret: ", ret)
        # # self.robot_interface.enable_servo_mode(False)
    
    #=====================================================================
        print("set_tcp_time3: ", time.time())

    
    def _compute_reward(self, obs: Dict = None, xy_limit=False) -> float:
        """
        计算奖励
        
        委托给 task 进行奖励计算
        
        Returns:
            reward: 当前步的奖励
        """
        if self.task is not None:
            obs, obs_norm = self._get_obs()
            achieved_goal = self.task.get_achieved_goal(obs)
            desired_goal = obs.get("desired_goal", self._desired_goal)
            return self.task.compute_reward(obs, desired_goal, {})
        return 0.0
    
    def _check_terminated(self, obs: Dict = None) -> bool:
        """
        检查任务是否完成
        
        委托给 task 进行成功判断
        
        Returns:
            terminated: 是否完成任务
        """
        if self.task is None:
            return False
        if obs is None:
            obs, obs_norm = self._get_obs()

        achieved_goal = self.task.get_achieved_goal(obs)
        desired_goal = obs.get("desired_goal", self._desired_goal)
        return self.task.is_success_fn(obs, desired_goal)

    
    def close(self):
        """关闭环境并断开机器人连接"""
        if self.robot_interface.is_connected:
            # 停止机器人运动
            try:
                self.robot_interface.stop_motion()
            except Exception as e:
                print(f"Warning: Failed to stop robot motion: {e}")
            
            # 断开连接
            self.robot_interface.disconnect()
    
    def render(self):
        """
        渲染环境
        
        真机环境通常不需要渲染，可以返回相机图像（如果有）
        """
        # TODO: 如果有外部相机，可以在这里返回图像
        return None
    

    def get_obs(self): 
        return self._get_obs()
        # obs, obs_norm = self._get_obs()
        # return obs, obs_norm

    def get_reward(self): 
        return self._compute_reward()
    
    def get_check_terminated(self, obs): 
        return self._check_terminated(obs)
    
    def get_check_chunked(self, obs, reward): 
        truncated = False
        if self.current_step >= self.max_episode_steps: 
            truncated = True
        # truncated = truncated or self.robot_interface.isCollisionOccurred() # TODO
        if hasattr(self.task, 'is_out_of_workspace'): 
            truncated = truncated or self.task.is_out_of_workspace(obs) 
            if self.task.is_out_of_workspace(obs): reward -= 100000 
        return truncated, reward
    
    def get_info(self, obs): 
        info = {
            # "is_success": terminated,
            "episode_step": self.current_step,
        }
        if self.task is not None:
            info.update(self.task.get_info())
        
        info["obs"] = obs
        info["qpos"] = self.robot_interface.get_joint_positions()
        info["euler"] = np.array(self.robot_interface.robot_state.getTcpPose()[3:]).astype(np.float32)
    
    def step_fake_for_multiprocess(
        self,
        # action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        执行一步动作
        
        Args:
            action: 动作，shape=(dof,)，范围 [-1, 1]
        
        Returns:
            observation: 新的观测
            reward: 奖励
            terminated: 是否达成任务目标
            truncated: 是否超时
            info: 额外信息
        """
        # 记录开始时间（用于控制频率）
        # start_time = time.time()

        # 1. 发送动作到机器人
        # self._set_joint_action(action)
        # print("real_env_time1: ", time.time())
        # self._set_tcp_action(action)
        # self.action_for_set_action = action.copy()
        # print("real_env_time2: ", time.time())
        
        # 2. 等待一个控制周期（保持控制频率）
        # elapsed = time.time() - start_time
        # if elapsed < self.control_dt:
        #     time.sleep(self.control_dt - elapsed)
        
        # 3. 获取新的观测
        obs, obs_norm = self._get_obs()
        self.current_step += 1
        
        # 4. 计算奖励
        reward = self._compute_reward()
        
        # 5. 检查是否终止
        terminated = self._check_terminated(obs)
        if terminated: reward += 1000
        
        # 6. 检查是否超时
        truncated = False
        if self.current_step >= self.max_episode_steps: 
            truncated = True
        # truncated = truncated or self.robot_interface.isCollisionOccurred() # TODO
        if hasattr(self.task, 'is_out_of_workspace'): 
            truncated = truncated or self.task.is_out_of_workspace(obs) 
            if self.task.is_out_of_workspace(obs): reward -= 100000 
        
        # 7. 构建info
        info = {
            "is_success": terminated,
            "episode_step": self.current_step,
        }
        if self.task is not None:
            info.update(self.task.get_info())
        
        info["obs"] = obs
        info["qpos"] = self.robot_interface.get_joint_positions()
        info["euler"] = np.array(self.robot_interface.robot_state.getTcpPose()[3:]).astype(np.float32)

        # obs, reward, terminated, truncated, info = self.step(np.zeros(7))
        
        return obs, reward, terminated, truncated, info