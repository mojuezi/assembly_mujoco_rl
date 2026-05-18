"""
MuJoCo仿真环境

作者: Liu Gang
日期: 2025-12-20
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import numpy as np
import mujoco
from mujoco import viewer as mj_viewer
from gymnasium import spaces
from dm_robotics.transformations import transformations as tr
import time
import cv2

from .base_env import BaseRobotEnv
from ..controllers import get_controller, BaseController
from ..tasks.base_task import BaseTask
from ..utils.sensor_utils import ForceSensorMonitor
from mujoco_env.mujoco_env.robot_config.aubo_i5 import AuboI5Robot

import pdb
import threading

class SimulationRobotEnv(BaseRobotEnv):
    """
    基于MuJoCo的仿真机器人环境
    
    提供基础的仿真功能，包括:
    - MuJoCo模型加载和管理
    - 仿真步进
    - 渲染
    - 状态获取
    Group classification in Mujoco: group 0 is for background, table and hole. Group 1 is for gripper. 
    Group 2 is for manimulator visual while group 3 is for manipulator collision
    """
    
    def __init__(
        self,
        xml_path: str,
        task: BaseTask,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        render_dt: float = 0.02, 
        render_mode: str = None,
        max_episode_steps: int = 500,
        ee_site_name: str = None,
        **controller_kwargs
    ):
        """
        初始化仿真环境

        Args:
            xml_path: MuJoCo 场景文件路径
            task: 任务实例（包含 robot_config 和 obs_config）
            control_dt: 控制时间步长（秒）
            physics_dt: 物理仿真时间步长（秒）
            render_mode: 渲染模式（必需），可选值: "rgb_array", "depth", "human"
            max_episode_steps: 最大episode步数（强化学习参数）
            ee_site_name: 末端执行器site名称，用于TCP位置读取
            **controller_kwargs: 传递给控制器的额外参数
        """
        self.xml_path = Path(xml_path)
        self.task = task
        # 设置 task 的环境引用
        if hasattr(self.task, 'set_env'):
            self.task.set_env(self)
        
        # 从 task 中获取配置
        self.robot_config = task.robot_config
        self.obs_config = task.obs_config
        self.arm_joint_names = ["shoulder_joint", "upperArm_joint", "foreArm_joint", "wrist1_joint", "wrist2_joint", "wrist3_joint"]
        # self.init_qpos = AuboI5Robot.INIT_QPOS
        self.init_qpos = np.array([177.0 / 180.0 * np.pi,  4.0 / 180.0 * np.pi, 95.0 / 180.0 * np.pi, 
                                    1.7 / 180.0 * np.pi, 89 / 180.0 * np.pi, 0.0 / 180.0 * np.pi]) # The VLA arm

        # 强化学习环境参数
        self.max_episode_steps = max_episode_steps
        self.ee_site_name = ee_site_name or "grip_site"  # 默认使用 grip_site
        
        self.control_dt = control_dt
        self.physics_dt = physics_dt
        self.render_dt = render_dt
        
        # 加载MuJoCo模型
        if not self.xml_path.exists():
            raise FileNotFoundError(f"XML file not found: {xml_path}")
        
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        # Set force sensor
        # pdb.set_trace()
        self.sensor_monitor = None
        if self.robot_config.use_ft_sensor is not None:
            self.sensor_monitor = ForceSensorMonitor(self.model, self.data)
        
        # 设置时间步长
        self.model.opt.timestep = physics_dt
        self._n_substeps = int(control_dt / physics_dt)
        
        # 创建控制器（必须在 MuJoCo 模型加载之后）
        # 控制器类型从 robot_config 中获取
        controller_kwargs = controller_kwargs.copy()
        controller_kwargs['ee_site_name'] = self.ee_site_name  # 强制设置，不使用setdefault

        self.controller = get_controller(
            self.robot_config.controller_type,
            model=self.model,
            data=self.data,
            dof=self.robot_config.dof,
            control_freq=1.0 / control_dt,
            **controller_kwargs
        )
        
        # 任务相关状态
        self._desired_goal = None
        
        # 渲染器
        self._viewer = None
        self._render_width = 640
        self._render_height = 480
        
        # 初始化父类
        super().__init__(render_mode=render_mode)
        
        # 设置观测和动作空间（必须在 task 和 controller 创建之后）
        self._setup_spaces()
    
    def _setup_spaces(self):
        """
        设置观测空间和动作空间
        
        观测空间由三部分组成：
        1. 机器人本体感知（qpos, qvel, tcp_pos, tcp_quat）
        2. 任务相关观测（从 task.get_obs_space() 获取）
        3. 传感器观测（image, depth 等）
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
        obs_dict["tcp_pos"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3,),
            dtype=np.float32
        )
        # obs_dict["tcp_quat"] = spaces.Box(
        #     low=-1,
        #     high=1,
        #     shape=(4,),
        #     dtype=np.float32
        # )
        
        # 2. 任务相关观测（从 task.get_obs_space() 获取）
        task_obs_space = self.task.get_obs_space()
        if isinstance(task_obs_space, spaces.Dict):
            # 合并 task 建议的观测空间
            obs_dict.update(task_obs_space.spaces)
        
        # 3. 传感器观测（从 obs_config 决定）
        if self.obs_config.include_image or self.render_mode == 'rgb_array':
            h, w = self.obs_config.image_size
            obs_dict["image"] = spaces.Box(
                low=0,
                high=255,
                shape=(h, w, 3),
                dtype=np.uint8
            )
        
        if self.obs_config.include_depth:
            h, w = self.obs_config.image_size
            obs_dict["depth"] = spaces.Box(
                low=0,
                high=np.inf,
                shape=(h, w),
                dtype=np.float32
            )
        
        # 最终的观测空间
        self.observation_space = spaces.Dict(obs_dict)
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None,
        random = False,
    ) -> Tuple[Dict, Dict]:
        """重置仿真环境"""
        # 调用父类reset
        super().reset(seed=seed, options=options)
        
        # 重置MuJoCo仿真
        mujoco.mj_resetData(self.model, self.data)
        
        # 设置初始状态
        self._set_initial_state()
        
        # 重置 task（如果存在）
        if self.task is not None:
            task_info = self.task.reset()
            self._desired_goal = task_info.get("desired_goal", None)
        else:
            self._desired_goal = None
        

        if random or self.task.random: 
            init_qpos_rand = self.task.random_reset()
            # init_qpos_rand = self.init_qpos + np.random.normal(loc=0, scale=0.1, size=self.init_qpos.shape)
            # while init_qpos_rand[2] < 1.0: 
            #     init_qpos_rand = self.init_qpos + np.random.normal(loc=0, scale=0.1, size=self.init_qpos.shape)
        else:
            init_qpos_rand = self.init_qpos
        
        for i, name in enumerate(self.arm_joint_names):
            try:
                joint_id = self.model.joint(name).id
                qpos_adr = self.model.jnt_qposadr[joint_id]
                self.data.qpos[qpos_adr] = init_qpos_rand[i]
                # 同时重置速度
                qvel_adr = self.model.jnt_dofadr[joint_id]
                self.data.qvel[qvel_adr] = 0
            except Exception as e:
                print(f"Warning: Could not set joint {name}: {e}")
        
        self.data.ctrl[:] = self.data.qpos.copy()
        self.data.qacc[:] = 0

        mujoco.mj_forward(self.model, self.data)
        # 执行几步仿真使其稳定
        for _ in range(10*self._n_substeps):
            mujoco.mj_step(self.model, self.data)
        # mujoco.mj_forward(self.model, self.data)

        if self.render_mode is not None:
            self.render()
        
        self.sensor_monitor.set_offset(self.sensor_monitor.get_wrench_data_original())
        obs, obs_norm = self._get_obs()
        info = {}
        info["obs"] = obs
        self.target_mv_tcp = obs["tcp_pos"]
        self.target_mv_quat = obs["tcp_quat"]
        # pdb.set_trace()


        # 重置 controller
        if hasattr(self.controller, 'reset'):
            self.controller.reset()
        
        # self.controller.set_admittance_params(mass=np.array([2.0, 2.0, 1.0, 1.0, 1.0, 1.0]), 
        #                                 damping=np.array([100.0, 140.0, 50.0, 5.0, 5.0, 5.0]), 
        #                                 stiffness=np.array([4000.0, 4000.0, 2000.0, 9.0, 9.0, 9.0]))
        
        # self.controller.set_desired_force([0, 0, -3000, 0, 0, 0])

        if hasattr(self.controller, 'set_selection_vector'): 
            self.controller.set_selection_vector(np.ones(6))
            self.controller.set_selection_vector(np.zeros(6))
            # self.controller.set_selection_vector(np.array([1, 1, 1, 0, 0, 0]))
            # self.controller.set_selection_vector(np.array([0, 0, 0, 1, 1, 1]))
        

        if self.task.obs_norm: obs = obs_norm
        return obs, info
    
    def step(self, action: np.ndarray, action_clip=False) -> Tuple[Dict, float, bool, bool, Dict]:
        """执行一步仿真"""
        # 限制动作范围
        if action_clip:
            action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # self.target_mv_tcp[0] = self._desired_goal[0]
        # self.target_mv_tcp[1] = self._desired_goal[1]
        # action = np.array([-0.00, 0.00, -0.001])
        print("action: ", action)
        self.target_mv_tcp += action[:3]

        if self.task is not None:
            self.target_mv_tcp = np.clip(
                self.target_mv_tcp,
                self.task.workspace_low,
                self.task.workspace_high,
            )
        print("goal_tcp: ", self.target_mv_tcp)
        
        # 执行多步物理仿真
        for _ in range(self._n_substeps):
            # 设置动作
            self._set_action(np.concatenate([self.target_mv_tcp, self.target_mv_quat]))
            mujoco.mj_step(self.model, self.data)

            # ee_site_name = getattr(self.controller, "ee_site_name", None)
            # site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
            # tcp_pos = self.data.site_xpos[site_id].copy()
            # # print("force: ", self.sensor_monitor.get_force_data())
            # print("tcp_pos: ", tcp_pos)

            if self.render_mode is not None:
                self.render()
                
        self._elapsed_steps += 1

        
        
        # 获取观测（只调用一次，避免重复渲染）
        observation, obs_norm = self._get_obs()
        print("observation: ", observation["tcp_pos"])
        # print("final_goal: ", observation["desired_goal"])
        
        # 计算奖励和终止状态（复用观测，不再重复调用 _get_obs）
        xy_distance, z_distance, reward = self._compute_reward(observation)
        # reward -= 200 * np.linalg.norm(action[:2])

        done = self._check_terminated(observation)
        if done: reward += 100


        truncated = False
        # Check max episode steps
        if self._elapsed_steps >= self.max_episode_steps: 
            truncated = True
        
        # Check if collision
        collided = self.is_contact(in_group=True, group1=1, group2=3) or \
            self.is_contact(in_group=True, group1=0, group2=3)
        
        if not (xy_distance < 0.045 and z_distance < 0.1): 
            collided = collided or self.is_contact(in_group=True, group1=0, group2=1)

        
        # Check if out of space
        out_of_workspace = self.task.is_out_of_workspace(observation)

        if collided:
            print("Collision detected - resetting")
            done = True
        elif out_of_workspace: 
            print("Out of space")
            done = True
        else:
            done = self._check_terminated(observation)

        # truncated = truncated or (self.is_contact(in_group=True, group1=0, group2=1) and (abs(observation["wrench"][0]) >= 45 or abs(observation["wrench"][1]) >= 45 or abs(observation["wrench"][2]) >= 45))
        # if abs(observation["wrench"][0]) >= 45 or abs(observation["wrench"][1]) >= 45 or abs(observation["wrench"][2]) >= 45: reward -= 1000
        
        
        print(f"Step {self._elapsed_steps} | XY distance: {xy_distance:.5f} | Z distance: {z_distance:.5f} | Reward: {reward:.2f}")
        # self.cnt += 1
        # print("cnt: ", self.cnt)
        
        info = {}
        info["obs"] = observation
        info["qpos"] = self.data.qpos[:6].astype(np.float32)

        if self.task.obs_norm: observation = obs_norm

        # print("obs_norm: ", obs_norm)
        
        return observation, reward, done, truncated, info
    
    def render(self):
        """渲染当前状态"""
        if self.render_mode is None:
            return None
        
        if self._viewer is None:
            if self.render_mode == "human" or self.render_mode == "rgb_array": 
                self._viewer = mj_viewer.launch_passive(
                    self.model, 
                    self.data, 
                    show_left_ui=False, 
                    show_right_ui=False, 
                )
                self.last_render_time = time.time()

            if self.render_mode == "rgb_array": 
                self._cam_viewer = mujoco.Renderer(
                    self.model,
                    height=self.obs_config.image_size[0], 
                    width=self.obs_config.image_size[1], 
                )
                self.cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "ee_cam")

        
        if self.render_mode == "depth":
            # 启用深度渲染
            self._viewer.enable_depth_rendering()
            return self._viewer.render()
        
        if self.render_mode == "human" or self.render_mode == "rgb_array":
            # 显示在窗口（需要额外的viewer）
            # return self._viewer.render()
            now = time.time()
            if now - self.last_render_time > self.render_dt: 
                self._viewer.sync()
                # time.sleep(self.control_dt*10)
                self.last_render_time = now
        
        return None
    
    def close(self):
        """关闭环境"""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
    
    def save_state(self) -> np.ndarray:
        """保存MuJoCo仿真状态
        
        Returns:
            state: 保存的状态数组
        """
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        size = mujoco.mj_stateSize(self.model, spec)
        state = np.empty(size, np.float64)
        mujoco.mj_getState(self.model, self.data, state, spec)
        return state.copy()
    
    def load_state(self, state: np.ndarray):
        """加载MuJoCo仿真状态
        
        Args:
            state: 之前保存的状态数组
        """
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.array(state.flatten(), np.float64)
        mujoco.mj_setState(self.model, self.data, state, spec)
    
    def get_state(self) -> np.ndarray:
        """获取当前状态的副本（不改变内部状态）
        
        Returns:
            state: 当前状态的副本
        """
        return self.save_state()
    
    def is_contact(self, geom1=None, geom2=None, in_group: bool=False, group1=None, group2=None, verbose: bool = False) -> bool:
        """检查两个几何体是否接触
        
        Args:
            geom1: 几何体1的名称（字符串）或几何体ID列表
            geom2: 几何体2的名称（字符串）或几何体ID列表
            verbose: 是否打印详细接触信息
            
        Returns:
            bool: 如果存在接触返回True，否则返回False
        """
        from typing import List, Union
        
        # 转换为ID列表
        def to_geom_ids(geom) -> List[int]:
            if isinstance(geom, str):
                # 单个几何体名称
                geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom)
                return [geom_id] if geom_id >= 0 else []
            elif isinstance(geom, list):
                # 几何体名称列表
                ids = []
                for g in geom:
                    if isinstance(g, str):
                        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, g)
                        if geom_id >= 0:
                            ids.append(geom_id)
                    else:
                        ids.append(g)  # 假设已经是ID
                return ids
            else:
                # 假设是单个ID
                return [geom]
        
        if not in_group:
            geom1_ids = to_geom_ids(geom1)
            geom2_ids = to_geom_ids(geom2)
        
        # 检查所有接触
        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            if not in_group:
                geom_id1 = contact.geom1
                geom_id2 = contact.geom2

                # 检查是否匹配（任意顺序）
                if (geom_id1 in geom1_ids and geom_id2 in geom2_ids) or \
                (geom_id1 in geom2_ids and geom_id2 in geom1_ids):
                    if verbose:
                        name1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id1)
                        name2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id2)
                        dist = contact.dist
                        print(f"Contact detected between: {name1} and {name2}")
                        print(f"Contact distance: {dist:.6f}")
                    return True
        
            else:  
                g1 = self.model.geom_group[contact.geom1]
                g2 = self.model.geom_group[contact.geom2]
                if (g1 == group1 and g2 == group2) or (g2 == group1 and g1 == group2): 
                    if verbose:
                        name1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
                        name2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
                        dist = contact.dist
                        print(f"Contact detected between: {name1} and {name2}")
                        print(f"Contact distance: {dist:.6f}")
                    return True
        return False
    
    def get_body_id(self, name: str) -> int:
        """根据body名称获取ID"""
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
    
    def get_site_id(self, name: str) -> int:
        """根据site名称获取ID"""
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
    
    def get_geom_id(self, name: str) -> int:
        """根据geom名称获取ID"""
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
    
    def get_body_pos(self, name: str) -> np.ndarray:
        """获取body的位置"""
        return self.data.body(name).xpos.copy()
    
    def get_body_quat(self, name: str) -> np.ndarray:
        """获取body的四元数"""
        return self.data.body(name).xquat.copy()
    
    def get_body_rotm(self, name: str) -> np.ndarray:
        """获取body的旋转矩阵"""
        return self.data.body(name).xmat.copy().reshape(3, 3)
    
    def get_site_pos(self, name: str) -> np.ndarray:
        """获取site的位置"""
        return self.data.site(name).xpos.copy()
    
    def get_site_quat(self, name: str) -> np.ndarray:
        """获取site的四元数"""
        return self.data.site(name).xquat.copy()
    
    def get_site_rotm(self, name: str) -> np.ndarray:
        """获取site的旋转矩阵"""
        return self.data.site(name).xmat.copy().reshape(3, 3)
    
    def get_body_jacp(self, name: str) -> np.ndarray:
        """获取body的位置雅可比"""
        bid = self.get_body_id(name)
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, None, bid)
        return jacp
    
    def get_body_jacr(self, name: str) -> np.ndarray:
        """获取body的旋转雅可比"""
        bid = self.get_body_id(name)
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, None, jacr, bid)
        return jacr
    
    def get_site_jacp(self, name: str) -> np.ndarray:
        """获取site的位置雅可比"""
        sid = self.get_site_id(name)
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, None, sid)
        return jacp
    
    def get_site_jacr(self, name: str) -> np.ndarray:
        """获取site的旋转雅可比"""
        sid = self.get_site_id(name)
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, None, jacr, sid)
        return jacr
    
    def get_body_xvelp(self, name: str) -> np.ndarray:
        """获取body的平移速度"""
        jacp = self.get_body_jacp(name)
        return np.dot(jacp, self.data.qvel)
    
    def get_body_xvelr(self, name: str) -> np.ndarray:
        """获取body的旋转速度"""
        jacr = self.get_body_jacr(name)
        return np.dot(jacr, self.data.qvel)
    
    def get_site_xvelp(self, name: str) -> np.ndarray:
        """获取site的平移速度"""
        jacp = self.get_site_jacp(name)
        return np.dot(jacp, self.data.qvel)
    
    def get_site_xvelr(self, name: str) -> np.ndarray:
        """获取site的旋转速度"""
        jacr = self.get_site_jacr(name)
        return np.dot(jacr, self.data.qvel)
    
    def set_joint_qpos(self, qpos: np.ndarray, joint_names: Optional[list] = None):
        """设置关节位置
        
        Args:
            qpos: 关节位置数组
            joint_names: 关节名称列表（如果为None，使用前N个关节）
        """
        if joint_names is None:
            self.data.qpos[:len(qpos)] = qpos
        else:
            for i, name in enumerate(joint_names):
                self.data.joint(name).qpos = qpos[i]
    
    def set_actuator_ctrl(self, ctrl: np.ndarray, actuator_names: Optional[list] = None):
        """设置执行器控制信号
        
        Args:
            ctrl: 控制信号数组
            actuator_names: 执行器名称列表（如果为None，使用前N个执行器）
        """
        if actuator_names is None:
            self.data.ctrl[:len(ctrl)] = ctrl
        else:
            for i, name in enumerate(actuator_names):
                self.data.actuator(name).ctrl = ctrl[i]
    
    def _get_obs(self) -> Dict[str, np.ndarray]:
        """
        获取观测
        
        Returns:
            obs: 观测字典，包含：
                - qpos: 关节位置
                - qvel: 关节速度
                - tcp_pos: TCP位置
                - tcp_quat: TCP四元数
                - achieved_goal: 当前目标（如果有task）
                - desired_goal: 期望目标（如果有task）
                - image: 图像（如果配置）
                - depth: 深度图（如果配置）
        """
        obs = {}
        obs_norm = {}
        
        # 1. 机器人本体感知（始终包含）
        # 需要使用正确的关节索引，因为场景中可能有其他物体的 freejoint
        dof = self.robot_config.dof
        
        # 从 robot_config 获取关节名称（如果未定义则使用前 dof 个 qpos）
        joint_names = self.robot_config.joint_names
        
        qpos = np.zeros(dof, dtype=np.float32)
        qvel = np.zeros(dof, dtype=np.float32)
        
        if joint_names is not None and len(joint_names) >= dof:
            # 使用配置中的关节名称查找正确的索引
            for i, name in enumerate(joint_names[:dof]):
                try:
                    joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                    if joint_id >= 0:
                        qpos_adr = self.model.jnt_qposadr[joint_id]
                        qvel_adr = self.model.jnt_dofadr[joint_id]
                        qpos[i] = self.data.qpos[qpos_adr]
                        qvel[i] = self.data.qvel[qvel_adr]
                except Exception:
                    pass
        else:
            # 回退：直接使用前 dof 个 qpos/qvel（适用于简单场景）
            qpos = self.data.qpos[:dof].astype(np.float32)
            qvel = self.data.qvel[:dof].astype(np.float32)
        
        # obs["qpos"] = qpos
        # obs["qvel"] = qvel
        
        # TCP 位姿（优先使用控制器指定的末端 site）
        try:
            tcp_pos = None
            tcp_quat = None
            ee_site_name = getattr(self.controller, "ee_site_name", None)
            if ee_site_name:
                site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
                if site_id >= 0:
                    tcp_pos = self.data.site_xpos[site_id].copy()
                    if hasattr(self.data, "site_xquat"):
                        tcp_quat = self.data.site_xquat[site_id].copy()
                    else:
                        quat = np.zeros(4, dtype=np.float64)
                        mujoco.mju_mat2Quat(quat, self.data.site_xmat[site_id])
                        tcp_quat = quat
                else:
                    body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, ee_site_name)
                    if body_id >= 0:
                        body = self.data.body(ee_site_name)
                        tcp_pos = body.xpos.copy()
                        if hasattr(body, "xquat"):
                            tcp_quat = body.xquat.copy()
                        else:
                            quat = np.zeros(4, dtype=np.float64)
                            mujoco.mju_mat2Quat(quat, body.xmat)
                            tcp_quat = quat

            if tcp_pos is None:
                if self.model.nsite > 0:
                    tcp_pos = self.data.site_xpos[-1].copy()
                    if hasattr(self.data, "site_xquat"):
                        tcp_quat = self.data.site_xquat[-1].copy()
                    else:
                        quat = np.zeros(4, dtype=np.float64)
                        mujoco.mju_mat2Quat(quat, self.data.site_xmat[-1])
                        tcp_quat = quat
                else:
                    tcp_pos = self.data.xpos[-1].copy()
                    if hasattr(self.data, "xquat"):
                        tcp_quat = self.data.xquat[-1].copy()
                    else:
                        quat = np.zeros(4, dtype=np.float64)
                        mujoco.mju_mat2Quat(quat, self.data.xmat[-1])
                        tcp_quat = quat

            obs["tcp_pos"] = tcp_pos.astype(np.float32)
            obs["tcp_quat"] = tcp_quat.astype(np.float32)
        except Exception:
            # 如果失败，使用零
            obs["tcp_pos"] = np.zeros(3, dtype=np.float32)
            obs["tcp_quat"] = np.array([1, 0, 0, 0], dtype=np.float32)
        
        # 2. 任务相关观测
        if self.task is not None:
            pass
            # 任务可以从当前观测中提取 achieved_goal
            # achieved_goal = self.task.get_achieved_goal(obs)
            # obs["achieved_goal"] = achieved_goal.astype(np.float32)
            
            # 期望目标
            if self._desired_goal is not None:
                obs["desired_goal"] = self._desired_goal.astype(np.float32)
            else:
                # 如果没有，创建一个零向量
                obs["desired_goal"] = None
        
        # 3. 传感器观测
        if self.render_mode == "rgb_array":
            self._cam_viewer.update_scene(self.data, camera=self.cam_id)
            img = self._cam_viewer.render()
            obs["image"] = img
            cv2.imshow('image', img[:, :, ::-1])
            cv2.waitKey(1)
            # print("img", img)
            # pdb.set_trace()
        
        if self.obs_config.include_image:
            image = self.render()
            if image is not None:
                obs["image"] = image
        
        if self.obs_config.include_depth:
            # TODO: 实现深度图渲染
            h, w = self.obs_config.image_size
            obs["depth"] = np.zeros((h, w), dtype=np.float32)
        
        if self.robot_config.use_ft_sensor is not None: 
            sensor_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "end_force_sensor_site")
            sensor_mat = self.data.site_xmat[sensor_site_id].reshape(3, 3).copy()
            # sensor_force = sensor_mat @ self.sensor_monitor.get_force_data()
            # sensor_torque = sensor_mat @ self.sensor_monitor.get_torque_data()
            # obs["wrench"] = np.concatenate([sensor_force, sensor_torque], -1)
            obs["wrench"] = np.concatenate([self.sensor_monitor.get_force_data(), self.sensor_monitor.get_torque_data()], -1)
            # pdb.set_trace()
        
        if self.task.obs_norm: 
            obs_norm["tcp_pos"] = (obs["tcp_pos"] - self.task.center) / self.task.workspace_high
            # obs_norm["tcp_quat"] = obs["tcp_quat"]
            # if obs["desired_goal"] is None:
            #     obs_norm["desired_goal"] = np.zeros(3)
            # else: 
            #     obs_norm["desired_goal"] = (obs["desired_goal"] - self.task.center) / self.task.workspace_high
            # wrench_norm = np.zeros(6)
            # wrench_norm[:3] = obs["wrench"][:3] / 50.0
            # wrench_norm[3:] = obs["wrench"][3:] / 10.0
            # obs_norm["wrench"] = wrench_norm
            if "image" in obs.keys(): 
                obs_norm["image"] = obs["image"]
            
        
        return obs, obs_norm
    
    def _set_action(self, action: np.ndarray):
        """
        设置动作
        
        使用 controller 将用户动作转换为 MuJoCo 控制信号
        如果配置了夹爪，处理夹爪动作
        
        Args:
            action: 用户动作
        """
        dof = self.robot_config.dof
        
        current_state = {
            "qpos": self.data.qpos[:dof].copy(),
            "qvel": self.data.qvel[:dof].copy(),
        }

        if self.sensor_monitor is not None: 
            sensor_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "end_force_sensor_site")
            sensor_mat = self.data.site_xmat[sensor_site_id].reshape(3, 3).copy()
            # sensor_force = sensor_mat @ self.sensor_monitor.get_force_data()
            # sensor_torque = sensor_mat @ self.sensor_monitor.get_torque_data()
            # current_state["wrench"] = np.concatenate([sensor_force, sensor_torque], -1)
            current_state["wrench"] = np.concatenate([self.sensor_monitor.get_force_data(), self.sensor_monitor.get_torque_data()], -1)
            # print(current_state["wrench"])
            # pdb.set_trace()
            # current_state["wrench"] = np.zeros(6)
        
        # 根据控制器类型处理动作
        if self.robot_config.controller_type in ["cartesian_ik", "operational_space", "admittance"]:
            # 笛卡尔控制器需要完整的笛卡尔位姿 (位置+四元数 = 7)
            # 注意：如果 action 包含夹爪（例如8维），我们需要提取前7维作为机器人目标
            if len(action) >= 7:
                robot_action = action[:7]
            else:
                robot_action = action
        else:
            # 其他控制器使用关节动作
            robot_action = action[:dof]
        
        # 使用 controller 计算机器人控制信号
        ctrl_signal = self.controller.compute_control(
            target=robot_action,
            current_state=current_state, 
            external_force= current_state["wrench"]
        )
        # print("robot_action: ", robot_action)
        # print("external_force: ", current_state["wrench"])
        # print("ctrl_signal: ", ctrl_signal)
        # pdb.set_trace()
        
        # 应用机器人控制信号到 MuJoCo
        self.data.ctrl[:dof] = ctrl_signal
        
        # 2. 夹爪动作 (如果有)
        # 根据控制器类型确定机械臂动作占据的长度
        is_task_space = self.robot_config.controller_type in ["cartesian_ik", "operational_space"]
        robot_action_dim = 7 if is_task_space else dof
        # pdb.set_trace()

        if self.robot_config.gripper_name and len(action) > robot_action_dim:
            # 夹爪动作通常位于机器人动作之后
            gripper_action = action[robot_action_dim]

            if self.robot_config.gripper_name == "umi_scissors":
                left_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_actuator")
                right_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_actuator")
                if left_id != -1 and right_id != -1:
                    def _ctrl_limits(act_id, fallback=(-0.5, 0.5)):
                        ctrl_range = self.model.actuator_ctrlrange[act_id]
                        if ctrl_range[1] > ctrl_range[0]:
                            return float(ctrl_range[0]), float(ctrl_range[1])
                        return fallback

                    left_min, left_max = _ctrl_limits(left_id)
                    right_min, right_max = _ctrl_limits(right_id)
                    # 双驱动：闭合时两侧对向运动（与实际开合方向对齐）
                    if gripper_action > 0.5:
                        left_ctrl = left_min
                        right_ctrl = right_max
                    else:
                        left_ctrl = left_max
                        right_ctrl = right_min
                    self.data.ctrl[left_id] = left_ctrl
                    self.data.ctrl[right_id] = right_ctrl
                    return
            
            # 尝试找到夹爪的 actuator
            # Robotiq 2F-85 的 actuator 名在不同场景中可能不同，依次尝试常用名称
            gripper_actuator_id = -1
            for name in ["robotiq_2f_85", "fingers_actuator", "gripper_actuator"]:
                id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                if id != -1:
                    gripper_actuator_id = id
                    break
            
            if gripper_actuator_id == -1:
                # 如果通过名称找不到，使用第 dof 个 actuator (假设前 dof 个是机器人关节)
                if self.model.nu > dof:
                    gripper_actuator_id = dof
            
            # === 简化控制：仅支持开/闭 ===
            # action > 0.5 -> Close (max_ctrl)
            # action <= 0.5 -> Open (min_ctrl)
            
            if gripper_actuator_id != -1 and gripper_actuator_id < self.model.nu:
                ctrl_range = self.model.actuator_ctrlrange[gripper_actuator_id]
                
                # 检查是否定义了有效的控制范围
                if ctrl_range[1] > ctrl_range[0]:
                    min_ctrl, max_ctrl = ctrl_range
                    gripper_ctrl = max_ctrl if gripper_action > 0.5 else min_ctrl
                else:
                    # 默认 Robotiq 2F-85: 0=Open, 255=Close (如果 xml 中没有定义 ctrlrange)
                    gripper_ctrl = 255 if gripper_action > 0.5 else 0
                
                # 应用控制
                self.data.ctrl[gripper_actuator_id] = gripper_ctrl
                
    
    def _set_initial_state(self):
        """设置初始状态（子类可重写）"""
        # 默认：使用keyframe或零状态
        pass
    
    def _compute_reward(self, obs: Dict = None, xy_limit=False) -> float:
        """
        计算奖励
        
        如果存在 task，委托给 task.compute_reward()
        否则返回 0.0
        
        Args:
            obs: 可选的观测字典，如果提供则复用，避免重复调用 _get_obs()
        """
        if self.task is None:
            return 0.0
        
        if obs is None:
            obs, obs_norm = self._get_obs()
        # achieved_goal = obs.get("achieved_goal", np.zeros(7))
        desired_goal = obs.get("desired_goal", np.zeros(7))

        return self.task.compute_reward(obs, desired_goal, {"xy_limit": xy_limit})
    
    def _check_terminated(self, obs: Dict = None) -> bool:
        """
        检查是否达到终止状态
        
        如果存在 task，委托给 task.is_success_fn()
        否则返回 False
        
        Args:
            obs: 可选的观测字典，如果提供则复用，避免重复调用 _get_obs()
        """
        if self.task is None:
            return False
        
        if obs is None:
            obs, obs_norm = self._get_obs()
        # achieved_goal = obs.get("achieved_goal", np.zeros(7))
        desired_goal = obs.get("desired_goal", np.zeros(7))
        
        return self.task.is_success_fn(obs, desired_goal)
    
    def _get_info(self) -> Dict[str, Any]:
        """获取额外信息"""
        info = super()._get_info()
        # 可添加仿真特定信息
        return info
