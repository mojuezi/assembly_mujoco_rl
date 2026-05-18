"""
轴孔装配任务（Peg-in-Hole）

作者: Liu Gang
日期: 2025-12-20
"""

from typing import Dict, Any, Tuple
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from gymnasium import spaces
from mujoco_env.mujoco_env.tasks.base_task import BaseTask
import mujoco
from dm_robotics.transformations import transformations as tr

import pdb

class PegInsertionTask(BaseTask):
    """
    轴孔装配任务（Peg-in-Hole）
    
    目标：将圆柱形的轴（peg）精确插入到孔（hole）中
    成功条件：轴完全插入孔中，位置和姿态都在容差范围内
    
    这是一个经典的机器人装配任务，需要高精度的位置和力控制。
    """
    
    def __init__(
        self,
        max_episode_steps: int = 500,
        success_threshold: float = 0.002,  # 2mm
        position_tolerance: float = 0.02,
        orientation_tolerance: float = 0.05,  # 约2.9度
        peg_radius: float = 0.015,  # 轴半径 15mm（根据实际装配需求调整）
        hole_radius: float = 0.016,  # 孔半径 16mm（与STL检测结果匹配）
        insertion_depth: float = 0.05,  # 插入深度 50mm
        ee_site_name: str = "grip_site",  # 末端执行器 site 名称
        mode: str = "sim", 
        hole_position_real = None, 
        workspace_low_real = None, 
        workspace_high_real = None, 
        **kwargs
    ):
        """
        初始化轴孔装配任务
        
        Args:
            max_episode_steps: 最大步数
            success_threshold: 成功阈值
            position_tolerance: 位置容差
            orientation_tolerance: 姿态容差（弧度）
            peg_radius: 轴的半径
            hole_radius: 孔的半径
            insertion_depth: 插入深度
            **kwargs: 额外参数
        """
        super().__init__(
            name="PegInsertion",
            max_episode_steps=max_episode_steps,
            success_threshold=success_threshold,
            **kwargs
        )
        
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        self.peg_radius = peg_radius
        self.hole_radius = hole_radius
        self.insertion_depth = insertion_depth
        self.clearance = hole_radius - peg_radius
        self.ee_site_name = ee_site_name
        self.obs_norm = True
        self.random = False

        self.mode = mode 
        self.hole_position_real = hole_position_real 
        
        # 孔的位置范围（用于动态场景）
        self.hole_position_range = np.array([
            [0.3, 0.6],   # x: 300-600mm
            [-0.3, 0.3],  # y: -300-300mm
            [0.05, 0.2],  # z: 50-200mm
        ])

        # 从场景文件中读取孔信息
        self.hole_position = None
        self.hole_orientation = None
        if hasattr(self, 'scene_name') and self.scene_name:
            self._load_hole_info_from_scene()

        # default workspace bounds (world coordinates) used for action/violation checks
        # If hole position was loaded from scene, set workspace to a cube centered at hole_pos
        # with diameter 0.5m (radius 0.25m); otherwise fall back to a reasonable default.
        # if self.hole_position is not None and len(self.hole_position) >= 3:
        #     self.center = np.asarray(self.hole_position[:3], dtype=np.float32)
        #     radius = 0.5  # half of diameter 0.5m
        #     self.workspace_low = (self.center - radius).astype(np.float32)
        #     self.workspace_high = (self.center + radius).astype(np.float32)
        #     self.workspace_high[1] -= 0.42
        #     self.workspace_high[2] -= 0.2
        #     # pdb.set_trace()
        # else:
        # fallback defaults
        self.center = np.asarray(self.hole_position[:3], dtype=np.float32)
        self.workspace_low = np.array([-0.67,-0.23,0.01], dtype=np.float32)
        self.workspace_high = np.array([-0.2,0.41,0.4], dtype=np.float32)
        # Reset target for Step 0: keyboard_offset and TCP position to restore after violation.
        # By default we don't hardcode TCP; we'll prefer env.initial_tcp/initial_qpos if available.
        self.reset_keyboard_offset = np.zeros(3, dtype=np.float32)
        self.reset_tcp = None

        if mode == "real": 
            self.hole_position = hole_position_real
            self.center = np.asarray(self.hole_position[:3], dtype=np.float32)
            self.workspace_low = workspace_low_real
            self.workspace_high = workspace_high_real

    def _load_hole_info_from_scene(self):
        """
        从场景XML文件中读取孔的位置和方向信息
        """
        try:
            # 构建场景文件路径
            scene_file = Path(__file__).parent.parent.parent / "assets" / "scenes" / f"{self.scene_name}.xml"

            if not scene_file.exists():
                print(f"Warning: Scene file {scene_file} not found, using default hole info")
                return

            # 解析XML文件
            tree = ET.parse(scene_file)
            root = tree.getroot()

            # 查找孔body
            hole_body = None
            for body in root.findall(".//body"):
                if body.get('name') == 'hole':
                    hole_body = body
                    break

            if hole_body is None:
                print("Warning: Hole body not found in scene file, using default hole info")
                return

            # 读取孔位置
            pos_str = hole_body.get('pos')
            if pos_str:
                pos_values = [float(x) for x in pos_str.split()]
                # self.hole_position = np.array(pos_values)
                # self.hole_position[0] += 0.002
                # self.hole_position[1] += 0.042
                # self.hole_position[2] -= 0.06
                self.hole_position = np.array(pos_values, dtype=np.float32)
                self.hole_position += np.array([0.002, 0.042, -0.06], dtype=np.float32)
                print(f"Loaded hole position from scene: {self.hole_position}")

            # 读取孔方向（euler角度）
            euler_str = hole_body.get('euler')
            if euler_str:
                euler_values = [float(x) for x in euler_str.split()]
                # 将euler角度转换为四元数
                # 这里使用简化的转换，实际应该使用更精确的方法
                if len(euler_values) >= 3:
                    # 绕x轴旋转theta的四元数: [cos(theta/2), sin(theta/2), 0, 0]
                    theta = euler_values[0]  # x轴旋转
                    qw = np.cos(theta / 2)
                    qx = np.sin(theta / 2)
                    qy = 0.0
                    qz = 0.0
                    self.hole_orientation = np.array([qw, qx, qy, qz])
                    print(f"Loaded hole orientation from scene: {self.hole_orientation} (from euler: {euler_values})")

        except Exception as e:
            print(f"Warning: Failed to load hole info from scene file: {e}")
            self.hole_position = None
            self.hole_orientation = None
    
    # def compute_simple_reward(
    #     self,
    #     obs: np.ndarray,
    #     desired_goal: np.ndarray,
    #     info: Dict[str, Any]
    # ) -> float:

    def compute_reward(
        self,
        obs: np.ndarray,
        desired_goal: np.ndarray,
        info: Dict[str, Any]
    ) -> float:
        """
        计算奖励
        
        奖励组成：
        - 距离奖励：轴尖端与孔中心的距离
        - 姿态奖励：轴的方向与孔的方向对齐程度
        - 接触奖励：轴与孔接触时的奖励
        - 插入奖励：成功插入给予大额奖励
        
        Args:
            achieved_goal: 当前轴位姿 [x, y, z, qw, qx, qy, qz, depth]
            desired_goal: 目标位姿 [x, y, z, qw, qx, qy, qz, depth]
            info: 额外信息（可能包含力信息）
            
        Returns:
            reward: 奖励值
        """
        # 对于轴孔装配，我们考虑：
        # - 在XY平面上的对齐（优先鼓励XY位置对齐）
        # - 插入深度（未插入时为负，插入时为正且增大）
        # - 当插入深度>0且XY方向有力反馈时给予插入奖励
        # achieved_goal：可为 [x,y,z]（轴尖位置）或 [x,y,z,depth]（若包含深度）
        # desired_goal：目标孔中心位置 [x,y,z]

        # Extract positions
        # pdb.set_trace()
        position_cur = obs["tcp_pos"][:3].copy()
        position_goal = desired_goal[:3].copy()

        # XY alignment error and reward
        xy_distance = np.linalg.norm(position_cur[:2] - position_goal[:2])
        
        # 力反馈占位（后续可由环境或传感器填入真实值）
        force_reward = 0.0
        force = obs["wrench"]

        # 深度奖励：鼓励更深的插入
        z_distance = abs(position_cur[-1] - position_goal[-1])

        reward = 10 * (-2 * xy_distance - 0.5 * z_distance)


        return xy_distance, z_distance, reward
    
    def is_success_fn(
        self,
        obs: np.ndarray,
        desired_goal: np.ndarray
    ) -> bool:
        """
        判断任务是否成功
        
        Args:
            achieved_goal: 当前轴位姿
            desired_goal: 目标位姿
            
        Returns:
            success: 是否成功
        """
        # pdb.set_trace()
        # If TCP is out of workspace, treat as terminal (violation) and perform soft reset.

        # if self.is_out_of_workspace(obs):
        #     # perform soft reset to INIT_QPOS and end episode
        #     # self._perform_soft_reset()
        #     return True


        # 成功定义：
        # - XY 在 position_tolerance 内对齐
        # - 插入深度为正（已插入）
        pos_error = np.linalg.norm(obs["tcp_pos"][:3] - desired_goal[:3])
        if pos_error <= self.position_tolerance:
            return True
        
        return False

        # # 插入深度检查
        # if len(achieved_goal) > 3:
        #     depth_achieved = achieved_goal[3]
        # else:
        #     depth_achieved = desired_goal[2] - achieved_goal[2]

        # # 当插入深度超过小阈值时认为已插入
        # return depth_achieved > 0.002  # 2mm 插入阈值
    
    def sample_goal(self) -> np.ndarray:
        """
        采样一个新的孔位置和方向

        如果从场景文件中成功读取了孔信息，则使用实际的孔位置和方向。
        否则，使用默认的随机采样。

        Returns:
            goal: 目标位姿 [x, y, z, qw, qx, qy, qz, depth]
        """
        # 返回目标：仅包含位置 [x,y,z]（假设为孔中心）
        if self.hole_position is not None:
            return self.hole_position[:3].copy()
        return np.array([
            np.random.uniform(self.hole_position_range[0, 0], self.hole_position_range[0, 1]),
            np.random.uniform(self.hole_position_range[1, 0], self.hole_position_range[1, 1]),
            np.random.uniform(self.hole_position_range[2, 0], self.hole_position_range[2, 1]),
        ])

    def is_out_of_workspace(self, obs: Dict[str, np.ndarray]) -> bool:
        """
        Check if the TCP position is outside the defined workspace bounds.
        """
        pos = None
        try:
            if obs is not None and "tcp_pos" in obs:
                pos = np.asarray(obs["tcp_pos"], dtype=np.float32)
            elif self.env is not None:
                # try to query env for site position
                pos = self.env.get_site_pos(self.ee_site_name)
        except Exception:
            pos = None

        if pos is None:
            return False

        return np.any(pos < self.workspace_low) or np.any(pos > self.workspace_high)

    def _perform_soft_reset(self):
        """
        Reset joints, velocities and controller to initial state without tearing down the env.
        """
        if self.env is None:
            return
        try:
            # Always reset directly to the robot-configured INIT_QPOS (no IK, no env snapshots).
            q0 = getattr(self.robot_config, "INIT_QPOS", None)
            if q0 is not None:
                reset_q = np.asarray(q0, dtype=np.float32).copy()
                try:
                    if hasattr(self.env, "set_joint_qpos"):
                        try:
                            self.env.set_joint_qpos(reset_q)
                        except Exception:
                            self.env.data.qpos[: len(reset_q)] = reset_q
                    else:
                        self.env.data.qpos[: len(reset_q)] = reset_q
                except Exception:
                    pass

            # zero velocities
            self.env.data.qvel[: self.robot_config.dof] = 0.0

            # forward to update derived quantities
            mujoco.mj_forward(self.env.model, self.env.data)

            # reset controller state if available
            if hasattr(self.env, "controller") and hasattr(self.env.controller, "reset"):
                self.env.controller.reset()

            # record the actual TCP after reset for debugging (best-effort)
            try:
                if hasattr(self.env, "get_site_pos"):
                    actual_tcp = self.env.get_site_pos(self.ee_site_name)
                elif hasattr(self.env, "data") and self.env.model.nsite > 0:
                    actual_tcp = self.env.data.site_xpos[-1].copy()
                else:
                    actual_tcp = None
                self._last_reset_tcp = np.asarray(actual_tcp) if actual_tcp is not None else None
            except Exception:
                self._last_reset_tcp = None
        except Exception:
            pass

    # def step(self, obs: Dict[str, np.ndarray]) -> Tuple[float, bool, Dict[str, Any]]:
    #     """
    #     Override BaseTask.step to include workspace violation handling:
    #     if TCP goes outside workspace, perform a soft reset to initial pose and end episode.
    #     """
    #     # # If out of workspace, immediately soft-reset and end episode
    #     # if self.is_out_of_workspace(obs):
    #     #     # soft reset robot to initial pose
    #     #     self._perform_soft_reset()
    #     #     # reset internal counters
    #     #     self.current_step = 0
    #     #     self.is_success = False
    #     #     info = {"out_of_workspace": True}
    #     #     # reward 0 and mark done (episode will be restarted by env loop)
    #     #     return 0.0, True, info

    #     # otherwise fall back to default behavior
    #     return super().step(obs)

    def get_action_space(self) -> spaces.Box:
        """
        获取动作空间

        根据控制器类型返回相应的动作空间：
        - cartesian_ik, operational_space, cartesian_impedance: 7维位姿 + 1维夹爪 = 8维
        - joint_position, joint_velocity, joint_torque: dof维关节 + 1维夹爪

        Returns:
            action_space: 动作空间 [-6.28, 16.28]
        """

        # 根据控制器类型确定动作维度
        controller_type = self.robot_config.controller_type
        if controller_type in ["cartesian_ik", "operational_space", "cartesian_impedance", "osc", "task_space"]:
            # 任务空间控制：7维位姿（位置+四元数）+ 1维夹爪
            dim = 3
            # if self.robot_config.gripper_name:
            #     dim += 1

            # 设置动作范围
            low = np.full(dim, -3.0, dtype=np.float32)
            high = np.full(dim, 3.0, dtype=np.float32)

            # 位置限制放宽到 [-10, 10]
            # Set Cartesian position bounds to the task workspace bounds (world coordinates)

            # low[:3] = self.workspace_low
            # high[:3] = self.workspace_high

            # low[:2] = self.workspace_low[:2]
            # high[:2] = self.workspace_high[:2]

            # 夹爪动作范围 [0, 1] 或 [-1, 1]
            # if self.robot_config.gripper_name:
            #     low[-1] = -1.0
            #     high[-1] = 1.0
        elif controller_type in ["admittance"]: 
            dim = 3
            low = np.full(dim, -0.001, dtype=np.float32)
            high = np.full(dim, 0.001, dtype=np.float32)
            # low[-1] = -30
            # high[-1] = 30

        else:
            # 关节空间控制：关节角度 + 夹爪
            # Use robot's joint limits for joint-space bounds.
            joint_limits = getattr(self.robot_config, "JOINT_LIMITS", None)
            if joint_limits is not None:
                # joint_limits shape: (dof, 2) with [low, high] per joint
                low = joint_limits[:, 0].astype(np.float32).copy()
                high = joint_limits[:, 1].astype(np.float32).copy()
            else:
                # fallback to wide angle bounds if JOINT_LIMITS not available
                low = np.full(self.robot_config.dof, -6.28, dtype=np.float32)
                high = np.full(self.robot_config.dof, 6.28, dtype=np.float32)

            dim = len(low)
            if self.robot_config.gripper_name:
                # extend bounds for gripper command (normalized -1..1)
                low = np.concatenate([low, np.array([-1.0], dtype=np.float32)])
                high = np.concatenate([high, np.array([1.0], dtype=np.float32)])

        return spaces.Box(low=low, high=high, dtype=np.float32)

    def get_achieved_goal(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        从观测中提取当前轴的位姿

        Args:
            obs: 观测字典，应包含"peg_pos"、"peg_quat"和"insertion_depth"

        Returns:
            achieved_goal: 当前轴位姿 [x, y, z, qw, qx, qy, qz, depth]
        """
        # 优先使用正确的 site 位置，而不是依赖可能错误的 tcp_pos
        if self.env is not None and hasattr(self.env, 'data') and hasattr(self.env, 'model'):
            try:
                import mujoco
                ee_site_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_SITE, self.ee_site_name)
                if ee_site_id != -1:
                    return self.env.data.site_xpos[ee_site_id].copy().astype(np.float32)
            except:
                pass

        # 回退到观测中的位置
        if "peg_pos" in obs:
            return obs["peg_pos"].copy()
        if "tcp_pos" in obs and not np.allclose(obs["tcp_pos"], 0.0):  # 只在非零时使用
            return obs["tcp_pos"].copy()
        return np.zeros(3, dtype=np.float32)
    
    def get_obs_space(self) -> spaces.Dict:
        """
        获取任务相关的观测空间
        
        Returns:
            obs_space: 观测空间
        """
        # 装配任务的观测：
        # - force: 力/力矩占位（3维）（fx, fy, fz）或仅使用 XY 平面的 fx,fy
        # - peg_pos: 轴端（末端）位置 (x, y, z)
        # - peg_vel: 轴端速度 (x, y, z)
        # - desired_goal: 孔心位置 (x, y, z)
        return spaces.Dict({
            "tcp_pos": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(3,),
                dtype=np.float32
            ),
            # "tcp_vel": spaces.Box(
            #     low=-np.inf,
            #     high=np.inf,
            #     shape=(3,),
            #     dtype=np.float32
            # ),
            # "tcp_quat": spaces.Box(
            #     low=-np.inf,
            #     high=np.inf,
            #     shape=(4,),
            #     dtype=np.float32
            # ),
            # "wrench": spaces.Box(
            #     low=-np.inf,
            #     high=np.inf,
            #     shape=(6,),
            #     dtype=np.float32
            # ),
            # "desired_goal": spaces.Box(
            #     low=np.array([0.0, -0.5, 0.0]),
            #     high=np.array([1.0, 0.5, 0.5]),
            #     shape=(3,),
            #     dtype=np.float32
            # ),
        })
    
    def random_reset(self): 

        init_poses= np.array([[1.5698031e+00 ,-1.6738579e-02  ,2.0485368e+00  ,4.7045410e-01 ,1.5708055e+00 ,-1.9727128e-04], 
                            [1.5697385e+00  ,5.1062841e-02  ,2.1137140e+00  ,4.6878573e-01 ,1.5707788e+00 ,-2.6215918e-04], 
                            [1.6033599, 0.06084549, 2.122788 ,  0.46809027, 1.5711976,  0.03279676], 
                            [1.8406367, 0.09820616, 2.1561744,  0.46496674, 1.5740643,  0.270612  ], 
                            [1.8638191, 0.09863637, 2.1565764,  0.46501848, 1.5743443,  0.29324108], 
                            [2.0198193, 0.08573906, 2.1448982,  0.46750414, 1.5759734,  0.44444507], 
                            [2.1791549, 0.03733731, 2.100417 ,  0.47137704, 1.5778319,  0.60911745], 
                            [2.1791546, 0.05942347, 2.0677977,  0.41593295, 1.5778322,  0.60911745], 
                            [2.1686568, 0.10909947, 1.9795997,  0.2784583 , 1.5777378,  0.5990742 ], 
                            [2.0844638, 0.13926835, 2.0012214,  0.26906383, 1.5769491,  0.5179765 ], 
                            [1.6770656, 0.1485466 , 2.0077503,  0.26563597, 1.5721201,  0.10705632], 
                            [1.630425 , 0.13815147, 1.9989423,  0.26729584, 1.5715692,  0.06098213], 
                            [1.3608285,   0.0318845 ,  1.9039623,  0.27803588, 1.5682591, -0.20915672], 
                            [1.3608286,   0.02695363,  1.9173881,  0.2966864 , 1.5682585, -0.20915674], 
                            [1.3607233,  -0.02550597,  2.016799 ,  0.44970533, 1.568247 , -0.2092584 ], 
                            [1.3608285,  -0.15660436,  2.1378865,  0.7005147 , 1.5682606, -0.20915677], 
                            [1.3608279,  -0.11701277,  2.1096916,  0.63162893, 1.5682622, -0.20915678], 
                            [1.3509862,   0.01746576,  1.9225519,  0.31106317, 1.5681573, -0.2185009 ], 
                            [1.2790363,  -0.02066364,  1.8713063,  0.2985891 , 1.5673715, -0.2881041 ], 
                            [1.0639966,  -0.19525626,  1.6780525,  0.27919805, 1.5648568, -0.5059716 ], 
                            [1.0816427,  -0.23091939,  1.6331633,  0.27049258, 1.5650225, -0.48868895], 
                            [1.1604524,  -0.44392428,  1.3413365,  0.18913618, 1.5659187, -0.40952033], 
                            [1.1604524,  -0.4520753 ,  1.3696444,  0.2257895 , 1.5659189, -0.40952033], 
                            [1.1604525,  -0.5375221 ,  1.5183172,  0.46090615, 1.5659188, -0.4095204 ], 
                            [1.1680702,  -0.54903436,  1.5446223,  0.4972916 , 1.5659944, -0.40223438], 
                            [1.2314025,  -0.500824  ,  1.6237984,  0.5279968 , 1.5666349, -0.34112144], 
                            [1.3756872,  -0.41302854,  1.7583348,  0.575476  , 1.5684376, -0.19429918], 
                            [1.3756871,  -0.39198503,  1.7403394,  0.5360194 , 1.5684373, -0.19429916], 
                            [1.3756862,  -0.28772104,  1.5826961,  0.27302918, 1.5684385, -0.19429915], 
                            [1.3756871,  -0.2681899 ,  1.5229806,  0.19515185, 1.5684375, -0.19429916], 
                            [1.3756871,  -0.26625624,  1.5148345,  0.1849828 , 1.5684378, -0.19429916], 
                            [1.3756871,  -0.25974277,  1.4840249,  0.14769602, 1.5684378, -0.19429916], 
                            [1.3476348,  -0.2753906 ,  1.4591614,  0.13880906, 1.5681133, -0.22196417], 
                            [1.3014842,  -0.30696115,  1.4188324,  0.12939698, 1.5675384, -0.26886985], 
                            [1.3434923,  -0.2796741 ,  1.4550804,  0.137892  , 1.5679992, -0.22798052], 
                            [1.5263199,  -0.18951668,  1.5659467,  0.1596652 , 1.570272 , -0.04367781], 
                            [1.5263199,  -0.19447897,  1.5894511,  0.18835977, 1.5702718, -0.04367781], 
                            [1.5262988,  -0.25538722,  1.7418973,  0.4028244 , 1.5702628, -0.0436982 ], 
                            [1.5263199,  -0.3842341 ,  1.8669227,  0.65565026, 1.570273 , -0.04367783], 
                            [1.5263199,  -0.36062387,  1.8507308,  0.61487377, 1.5702732, -0.04367783], 
                            [1.5263196,  -0.21114416,  1.6444533,  0.25897756, 1.5702733, -0.04367781], 
                            [1.5586139,  -0.19137627,  1.6292914,  0.2247653 , 1.5706537, -0.0118098 ], 
                            [1.8132563,  -0.154401  ,  1.6720942,  0.2313602 , 1.5737818,  0.24322985], 
                            [1.8132647,  -0.1542271 ,  1.6713483,  0.2303638 , 1.5737644,  0.24324568], 
                            [1.8132647,  -0.14472212,  1.6341274,  0.18328534, 1.5737633,  0.24324569], 
                            [1.8132647,  -0.13271026,  1.566697 ,  0.10421424, 1.5737653,  0.24324568], 
                            [1.8132648,  -0.1337465 ,  1.5752207,  0.11395054, 1.573765 ,  0.24324566], 
                            [1.8132647,  -0.15300618,  1.6706007,  0.22998402, 1.5737636,  0.24324566], 
                            [1.822858 ,  -0.2687007 ,  1.8673131,  0.5413669 , 1.5738655,  0.25241622], 
                            [1.8746656,  -0.27618667,  1.867055 ,  0.54862237, 1.5744096,  0.30272058], 
                            [2.0382695,  -0.31471533,  1.8146358,  0.53559375, 1.5763506,  0.46842486], 
                            [2.0222092,  -0.35770613,  1.7540606,  0.5177528 , 1.576177 ,  0.45244342], 
                            [1.9776908,  -0.4975506 ,  1.5458605,  0.44749135, 1.5756751,  0.40766165], 
                            [1.9776908,  -0.48799804,  1.5341809,  0.42589232, 1.5756751,  0.40766165], 
                            [1.9776907,  -0.44882447,  1.47334  ,  0.3253606 , 1.5756757,  0.40766165], 
                            [1.9623061,  -0.43351445,  1.4611224,  0.2984657 , 1.5755117,  0.39261675], 
                            [1.849712 ,  -0.40846846,  1.4974864,  0.30898255, 1.5743368,  0.2835987 ], 
                            [1.6390686,  -0.41398042,  1.4883919,  0.3055557 , 1.5716553,  0.06906229], 
                            [1.6390686,  -0.4227169 ,  1.5064008,  0.33262077, 1.5716553,  0.06906229], 
                            [1.639071 ,  -0.49198475,  1.5996634,  0.4961046 , 1.5716566,  0.06906538], 
                            [1.6390686,  -0.5204426 ,  1.6230742,  0.54685336, 1.5716554,  0.06906228], 
                            [1.6210505,  -0.5240791 ,  1.6184026,  0.54582155, 1.5714473,  0.05141565], 
                            [1.5116965,  -0.55116004,  1.5729194,  0.52783203, 1.570223 , -0.05463841], 
                            [1.497357 ,  -0.55067253,  1.5574354,  0.5107571 , 1.5699171, -0.07263842], 
                            [1.4973568,  -0.5073411 ,  1.512772 ,  0.42179453, 1.5699161, -0.07263841], 
                            [1.4906087,  -0.43485495,  1.358586 ,  0.1963688 , 1.5698473, -0.07902652], 
                            [1.4535037,  -0.44965765,  1.3287525,  0.18184446, 1.5694371, -0.11530518], 
                            [1.1358144,  -0.72963476,  0.8721428,  0.0050169 , 1.565728 , -0.4289851 ], 
                            [1.1304827,  -0.7409976 ,  0.8725534,  0.01598695, 1.5655838, -0.43948856], 
                            [1.1304827,  -0.7464875 ,  0.9368700,  0.0864031 , 1.565586 , -0.4394886 ], 
                            [1.1335509,  -0.8080556 ,  1.0625517,  0.27373636, 1.5656135, -0.43655023], 
                            [1.1465135,  -0.86745435,  0.9640649,  0.23383011, 1.5657972, -0.42348754], 
                            [1.1465522,  -0.86394644,  0.9551429,  0.22079746, 1.5657628, -0.42341977], 
                            [1.1465522,  -0.8476296 ,  0.9148429,  0.16412948, 1.5657636, -0.42341977], 
                            [1.1635242,  -0.821321  ,  0.9479134,  0.17076811, 1.5659447, -0.40672517], 
                            [1.2853627,  -0.69159013,  1.1753244,  0.26815483, 1.5672253, -0.2883671 ], 
                            [1.3690476,  -0.6592233 ,  1.2272968,  0.28925583, 1.5683554, -0.20100364], 
                            [1.372618 ,  -0.6910726 ,  1.1745224,  0.2675071 , 1.5684004, -0.19736813], 
                            [1.3927479,  -0.6781777 ,  1.1968617,  0.27675247, 1.5686324, -0.17755745], 
                            [1.564285 ,  -0.5990221 ,  1.3282213,  0.32946777, 1.5707649, -0.00573385], 
                            [1.5643066,  -0.5943094 ,  1.319883 ,  0.3161565 , 1.5707382, -0.00569395], 
                            [1.5643067,  -0.5592637 ,  1.2343365,  0.19432282, 1.5707338, -0.00569391], 
                            [1.5643066,  -0.53562534,  0.9957343, -0.06649935, 1.5707392, -0.00569395], 
                            [1.5643066,  -0.53592926,  0.9903241, -0.07158826, 1.5707382, -0.00569395], 
                            [1.5681486,  -0.5347563 ,  0.9922737, -0.0709481 , 1.5707731, -0.00218995], 
                            [1.5988274,  -0.5247488 ,  1.0077598, -0.06548732, 1.571126 ,  0.02810217], 
                            [1.7685045,  -0.50234914,  1.0415132, -0.0535639 , 1.5732291,  0.19848849], 
                            [1.7685045,  -0.50234914,  1.0415132, -0.0535639 , 1.573229 ,  0.19848849], 
                            [1.7685045,  -0.50234914,  1.0415132, -0.0535639 , 1.573229 ,  0.19848849]])
        
        idx = np.random.randint(init_poses.shape[0])
        q_pose = init_poses[idx]
        return q_pose
