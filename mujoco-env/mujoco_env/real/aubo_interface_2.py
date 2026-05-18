"""
Aubo 机器人接口实现

基于 Aubo Python SDK 实现的真机接口
"""

from this import s
import numpy as np
from typing import Dict, Optional, List, Union
from scipy.spatial.transform import Rotation
import logging
import time

from mujoco_env.mujoco_env.real.robot_interface import RobotInterface

import pdb

class AuboInterface(RobotInterface):
    """
    Aubo 机器人接口
    
    基于 Aubo Python SDK 实现，支持 Aubo i5 等系列机器人。
    
    Requirements:
        安装 Aubo SDK: pip install pyaubo_sdk
    
    Examples:
        >>> robot = AuboInterface(robot_ip="192.168.1.6")
        >>> if robot.connect():
        ...     pose = robot.get_tcp_pose()
        ...     print(f"Current TCP: {pose}")
        ...     robot.disconnect()
    """
    
    def __init__(
        self,
        robot_ip: str = "192.168.1.6",
        **kwargs
    ):
        """
        初始化 Aubo 机器人接口
        
        Args:
            robot_ip: Aubo 机器人 IP 地址
            **kwargs: 其他配置参数
        """
        super().__init__(robot_ip, **kwargs)
        
        self.robot = None
        self.dof = 6  # Aubo i5 是 6 自由度
        
        # 尝试导入 Aubo SDK
        try:
            import pyaubo_sdk
            self.pyaubo_sdk = pyaubo_sdk
            self.logger.info("Aubo SDK imported successfully")

        except ImportError:
            self.logger.error(
                "Failed to import pyaubo_sdk. "
                "Please install: pip install pyaubo_sdk"
            )
            self.pyaubo_sdk = None
    
    def connect(self) -> bool:
        """
        连接到 Aubo 机器人
        
        Returns:
            bool: 连接成功返回True
        """
        if self.pyaubo_sdk is None:
            self.logger.error("Aubo SDK not available")
            return False
        
        try:
            # 初始化 SDK
            self.robot = self.pyaubo_sdk.RpcClient()

            # 连接到机器人
            self.robot.connect(self.robot_ip, 30004)
            if self.robot.hasConnected():
                self.robot.login("aubo", "123456")

                if self.robot.hasLogined(): 
                    self.is_connected = True
                    self.logger.info(f"Successfully connected to Aubo robot at {self.robot_ip}")

                    # 获取机器人接口/状态/管理/运动控制等句柄
                    self.robot_name = self.robot.getRobotNames()[0]
                    self.run_time_machine = self.robot.getRuntimeMachine()
                    self.robot_interface = self.robot.getRobotInterface(self.robot_name)
                    self.robot_config = self.robot_interface.getRobotConfig()
                    self.robot_state = self.robot_interface.getRobotState()
                    self.robot_manage = self.robot_interface.getRobotManage()
                    self.robot_motion = self.robot_interface.getMotionControl()
                    self.robot_force_control = self.robot_interface.getForceControl()
                    self.robot_algorithm = self.robot_interface.getRobotAlgorithm()
                    self.robot_motion.setSpeedFraction(0.25)

                    # 设置请求超时时间
                    self.robot.setRequestTimeout(1000)
                    return True
                else:
                    self.logger.error(f"Log In failed with error code: {self.robot.hasLogined()}")
                    return False
            else:
                self.logger.error(f"Connection failed with error code: {self.robot.hasConnected()}")
                return False
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            return False
    
    def disconnect(self):
        """断开连接"""
        if self.robot and self.is_connected:
            try:
                self.robot.disconnect()
                self.logger.info("Disconnected from Aubo robot")
            except Exception as e:
                self.logger.error(f"Disconnection error: {e}")
            finally:
                self.is_connected = False
    
    def power_on(self) -> bool:
        """
        机器人上电
        
        Returns:
            bool: 上电成功返回True，否则返回False
        """
        if not self.is_connected:
            self.logger.error("Failed to power on: Robot not connected. Please call connect() first before powering on.")
            return False
        
        try:
            if self.robot_state.getRobotModeType() == self.pyaubo_sdk.RobotModeType.Running:
                self.logger.info("The robot is already running!")
                return True
            else:
                if 0 == self.robot_manage.poweron():
                    self.logger.info("The robot is requesting power-on!")
                    if 0 == self.robot_manage.startup():
                        self.logger.info("The robot is starting up!")
                        while 1:
                            robot_mode = self.robot_state.getRobotModeType()  
                            self.logger.info(f"Robot current mode: {robot_mode.name}")

                            if robot_mode == self.pyaubo_sdk.RobotModeType.Running:
                                break
                            time.sleep(1)
                        self.is_power_on = True
                        return True
                    else:
                       self.logger.error(f"Failed to start up: {self.robot_manage.start_up()}")
                       return False
                else:
                    self.logger.error(f"Failed to power on: {self.robot_manage.powerOn()}")
                    return False
        except Exception as e:
            self.logger.error(f"Failed to power on: {e}")
            return False
    
    def power_off(self) -> bool:
        """
        机器人下电
        
        Returns:
            bool: 下电成功返回True，否则返回False
        """
        if not self.is_connected:
            raise RuntimeError("Robot not connected")
        
        try:
            if 0 == self.robot_manage.poweroff():
                self.logger.info("The robot is requesting power-off!")
                if 0 == self.robot_manage.shutDown():
                    self.logger.info("The robot is shutting down!")
                    while 1:
                        robot_mode = self.robot_state.getRobotModeType()
                        self.logger.info(f"Robot current mode: {robot_mode.name}")
                        if robot_mode == self.pyaubo_sdk.RobotModeType.PowerOff:
                            break
                        time.sleep(1)
                    self.is_power_on = False
                    self.is_connected = False
                    return True
                else:
                    self.logger.error(f"Failed to shut down: {self.robot_manage.shutDown()}")
                    return False
            else:
                self.logger.error(f"Failed to power off: {self.robot_manage.powerOff()}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to power off: {e}")
            return False
    
    def get_joint_positions(self) -> np.ndarray:
        """
        获取当前关节位置
        
        Returns:
            np.ndarray: 关节位置 (弧度), shape=(6,)
        """
        if not self.is_power_on:
            raise RuntimeError("Robot not powered on. Please call power_on() first before getting joint positions.")
        
        try:
            joint_pos = self.robot_state.getJointPositions()
            return np.array(joint_pos, dtype=np.float64)
        except Exception as e:
            self.logger.error(f"Failed to get joint positions: {e}")
            return np.zeros(self.dof)
    
    def get_joint_velocities(self) -> np.ndarray:
        """
        获取当前关节速度
        
        Returns:
            np.ndarray: 关节速度 (rad/s), shape=(6,)
        """
        if not self.is_power_on:
            raise RuntimeError("Robot not powered on. Please call power_on() first before getting joint velocities.")
        
        try:
            joint_vel = self.robot_state.getJointSpeeds()
            return np.array(joint_vel, dtype=np.float64)
        except Exception as e:
            self.logger.error(f"Failed to get joint velocities: {e}")
            return np.zeros(self.dof)
    
    def get_tcp_pose(self) -> np.ndarray:
        """
        获取 TCP 位姿
        
        Returns:
            np.ndarray: TCP位姿 [x, y, z, qw, qx, qy, qz], shape=(7,)
        """
        if not self.is_power_on:
            raise RuntimeError("Robot not powered on. Please call power_on() first before getting TCP pose.")
        
        try:
            # 获取位姿（Aubo SDK 通常返回 [x, y, z, rx, ry, rz]）
            pose = self.robot_state.getTcpPose()
            
            # 提取位置和欧拉角
            position = np.array(pose[:3], dtype=np.float64)
            rotation_euler = np.array(pose[3:], dtype=np.float64)  # 欧拉角 (rx, ry, rz)
            
            # 转换欧拉角为四元数 [qw, qx, qy, qz]
            rotation_quat_scipy = Rotation.from_euler('xyz', rotation_euler).as_quat()  # [qx, qy, qz, qw]
            quat = np.array([
                rotation_quat_scipy[3],  # qw
                rotation_quat_scipy[0],  # qx
                rotation_quat_scipy[1],  # qy
                rotation_quat_scipy[2]   # qz
            ])
            
            return np.concatenate([position, quat])
        except Exception as e:
            self.logger.error(f"Failed to get TCP pose: {e}")
            return np.zeros(7)
    
    def get_tcp_velocity(self) -> np.ndarray:
        """
        获取 TCP 速度
        
        Returns:
            np.ndarray: TCP速度 [vx, vy, vz, wx, wy, wz], shape=(6,)
        """
        if not self.is_power_on:
            raise RuntimeError("Robot not powered on. Please call power_on() first before getting TCP velocity.")
        
        try:
            vel = self.robot_state.getTcpSpeed()
            return np.array(vel, dtype=np.float64)
        except Exception as e:
            self.logger.error(f"Failed to get TCP velocity: {e}")
            return np.zeros(6)
    
    def get_tcp_force(self) -> np.ndarray:
        """
        获取 TCP 力传感器读数
        
        Returns:
            np.ndarray: TCP力 [fx, fy, fz], shape=(3,)
        """
        if not self.is_power_on:
            raise RuntimeError("Robot not powered on. Please call power_on() first before getting TCP force.")
        
        try:
            # 从力传感器读取（可能返回 [fx, fy, fz, tx, ty, tz]）
            force_torque = self.robot_state.getTcpForce()
            return np.array(force_torque[:3], dtype=np.float64)
        except Exception as e:
            self.logger.error(f"Failed to get TCP force: {e}")
            return np.zeros(3)
    
    def get_tcp_torque(self) -> np.ndarray:
        """
        获取 TCP 力矩传感器读数
        
        Returns:
            np.ndarray: TCP力矩 [tx, ty, tz], shape=(3,)
        """
        if not self.is_power_on:
            raise RuntimeError("Robot not powered on. Please call power_on() first before getting TCP torque.")
        
        try:
            force_torque = self.robot_state.getTcpForce()
            return np.array(force_torque[3:], dtype=np.float64)
        except Exception as e:
            self.logger.error(f"Failed to get TCP torque: {e}")
            return np.zeros(3)
    
    def get_tcp_wrench(self) -> np.ndarray:
        """
        获取 TCP 力矩传感器读数
        
        Returns:
            np.ndarray: TCP力矩 [tx, ty, tz], shape=(3,)
        """
        if not self.is_power_on:
            raise RuntimeError("Robot not powered on. Please call power_on() first before getting TCP wrench.")
        
        try:
            force_torque = self.robot_state.getTcpForce()
            return np.array(force_torque, dtype=np.float64)
        except Exception as e:
            self.logger.error(f"Failed to get TCP wrench: {e}")
            return np.zeros(3)
    
    def set_servo_mode(self, mode: int = 0) -> bool:
        """
        设置伺服运动模式
        
        Args:
            mode: 伺服模式 0: 关闭伺服模式，1: （截断式）规划伺服模式，7: 规划伺服模式，可叠加力控
        Returns:
            bool: 使能或关闭伺服运动模式成功返回True，否则返回False
        
        """
        if not self.is_power_on:
            raise RuntimeError("Robot not powered on. Please call power_on() first before enabling servo mode.")
        
        try:
            if mode == 0:
                # 等待伺服运动完成
                start_time = time.perf_counter()
                # 第一步：等待运动剩余时间为0（循环检查getMotionLeftTime(0)）
                while self.robot_motion.getMotionLeftTime(0) != 0:
                    current_time = time.perf_counter()
                    elapsed_ms = (current_time - start_time) * 1000
                    if elapsed_ms >= 100000:
                        print(f"Warning: WaitServoJointComplete timeout after {elapsed_ms:.0f} ms")
                    time.sleep(0.005)

                # 第二步：等待机械臂进入稳定状态（isSteady()为True）
                while not self.robot_state.isSteady():
                    time.sleep(0.005)
            
            print(f"Setting servo mode to {mode}")
            self.robot_motion.setServoModeSelect(mode)
            time.sleep(0.1)     # 等待设置伺服模式完成
            if mode == self.robot_motion.getServoModeSelect():
                return True
            else:
                self.logger.error(f"Failed to set servo mode: {mode}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to enable servo mode: {e}")
            return False
            
    def servo_to_joint_positions(
        self,
        positions: np.ndarray
    ) -> int:
        """
        伺服到目标关节位置
        
        Args:
            positions: 目标关节位置 (弧度), shape=(dof,)
        Returns:
            int: 返回码。AUBO_QUEUE_FULL 表示队列已满，需要等待。其他错误码见SDK文档。
        """
        if not self.is_power_on:
            raise RuntimeError("Robot not powered on. Please call power_on() first before servoing to joint positions.")
        
        if len(positions) != self.dof:
            raise ValueError(f"Expected {self.dof} joint positions, got {len(positions)}")
        
        try:
            acceleration = 0.2
            velocity = 0.1
            run_time = 0.02       # 运行时间: t 值越大, 机器臂运动越慢, 反之运动越快; 该参数最优值为连续调用 servoJoint 接口的间隔时间
            lookahead_time = 0.1 
            gain = 200            

            ret = self.robot_motion.servoJoint(
                positions.tolist(),
                acceleration,
                velocity,
                run_time,
                lookahead_time,
                gain
            )
            while ret == 2:
                time.sleep(0.005)
                print(f"Queue is full, waiting for 5ms")
                ret = self.robot_motion.servoJoint(
                    positions.tolist(),
                    acceleration,
                    velocity,
                    run_time,
                    lookahead_time,
                    gain
                )
            return int(ret)   
        except Exception as e:
            self.logger.error(f"Failed to servo to joint positions: {e}")
            return -1  
    
    def wait_arrival(self) -> bool:
        """
        等待机械臂到达目标位置
        
        Returns:
            bool: 到达目标位置返回True，否则返回False
        """
        max_retry_count = 5 
        retry_count = 0
        exec_id = self.robot_motion.getExecId()
        while exec_id == -1:
            if retry_count >= max_retry_count:
                return False
            time.sleep(0.05)
            retry_count += 1
            exec_id = self.robot_motion.getExecId()
        while self.robot_motion.getExecId() != -1:
            time.sleep(0.05)
        return True
    
    def move_to_joint_positions(
        self,
        positions: np.ndarray,
        velocity: float = 0.5,
        acceleration: float = 0.5,
        blocking: bool = False
    ):
        """
        移动到目标关节位置
        
        Args:
            positions: 目标关节位置 (弧度), shape=(6,)
            velocity: 速度因子 [0, 1]
            acceleration: 加速度因子 [0, 1]
            blocking: 是否阻塞等待完成
        """
        if not self.is_power_on:
            raise RuntimeError("Robot not powered on. Please call power_on() first before moving.")
        
        if len(positions) != self.dof:
            raise ValueError(f"Expected {self.dof} joint positions, got {len(positions)}")
        
        try:
            # TODO: 速度因子和加速度因子需要转换为实际值

            # 设置机械臂的速度比率
            # self.robot_motion.setSpeedFraction(0.75)

            # 运动到路点
            self.robot_motion.moveJoint(
                positions.tolist(), 
                acceleration, 
                velocity, 
                0.0,          # blend_radius
                0.0,          # duration
            )

            # 阻塞
            if blocking:
                if (self.wait_arrival()):
                    self.logger.info("Move to joint positions successfully")
                else:
                    self.logger.error("Move to joint positions failed")
        except Exception as e:
            self.logger.error(f"Failed to move to joint positions: {e}")
            raise
    
    def move_tcp_pose(
        self,
        pose: np.ndarray,
        velocity: float = 0.1,
        acceleration: float = 0.1,
        blocking: bool = False
    ):
        """
        移动到目标 TCP 位姿
        
        Args:
            pose: 目标位姿 [x, y, z, qw, qx, qy, qz], shape=(7,)
            velocity: 速度 (m/s)
            acceleration: 加速度 (m/s^2)
            blocking: 是否阻塞等待完成
        """
        if not self.is_power_on:
            raise RuntimeError("Robot not powered on. Please call power_on() first before moving.")
        
        if len(pose) != 7:
            raise ValueError(f"Expected 7-element pose, got {len(pose)}")
        
        try:
            # 提取位置和四元数
            position = pose[:3]
            quat = pose[3:]  # [qw, qx, qy, qz]
            
            # 转换四元数为欧拉角（Aubo SDK 需要欧拉角）
            quat_scipy = [quat[1], quat[2], quat[3], quat[0]]  # [qx, qy, qz, qw]
            euler = Rotation.from_quat(quat_scipy).as_euler('xyz')
            
            # 合并为目标位姿 [x, y, z, rx, ry, rz]
            target_pose = np.concatenate([position, euler])
            
            # 设置机械臂的速度比率
            # self.robot_motion.setSpeedFraction(0.75)

            # 运动到路点
            self.robot_motion.moveLine(
                target_pose.tolist(),
                acceleration, 
                velocity, 
                0.0,          # blend_radius
                0.0,          # duration
            )

            # 阻塞
            if blocking:
                if (self.wait_arrival()):
                    self.logger.info("Move to TCP pose successfully")
                else:
                    self.logger.error("Move to TCP pose failed")
        except Exception as e:
            self.logger.error(f"Failed to move to TCP pose: {e}")
            raise
    
    def get_robot_state(self) -> Dict:
        """
        获取完整机器人状态
        
        Returns:
            Dict: 机器人状态字典
        """
        if not self.is_power_on:
            raise RuntimeError("Robot not powered on. Please call power_on() first before getting robot state.")
        
        try:
            state = {
                "joint_positions": self.robot_state.getJointPositions(),
                "joint_velocities": self.robot_state.getJointSpeeds(),
                "tcp_pose": self.robot_state.getTcpPose(),
                "tcp_velocity": self.robot_state.getTcpSpeed(),
                "tcp_force": self.robot_state.getTcpForce()[:3],
                "tcp_torque": self.robot_state.getTcpForce()[3:],
                "robot_mode": None,  # 如果 SDK 支持，可添加
                "errors": [],  # 如果 SDK 支持，可添加
            }
            
            # 尝试获取机器人模式
            try:
                state["robot_mode"] = self.robot_state.getRobotModeType()
            except:
                pass
            
            # TODO:尝试获取错误信息
            try:
                if hasattr(self.robot, 'get_errors'):
                    state["errors"] = self.robot.get_errors()
            except:
                pass
            
            return state
        except Exception as e:
            self.logger.error(f"Failed to get robot state: {e}")
            raise
    
    def stop_motion(
        self,
        max_acceleration: float = 30.0
        ):
        """立即停止机器人运动"""
        if not self.is_power_on:
            raise RuntimeError("Robot not powered on. Please call power_on() first before stopping motion.")
        
        try:
            # 终止机器人运行
            self.run_time_machine.abort()
            
            # 立即停止运动
            self.robot_motion.stopJoint(max_acceleration)
           
            # 等待运动结束
            is_steady = self.robot_state.isSteady()
            while is_steady is False:
                is_steady = self.robot_state.isSteady()
                time.sleep(0.005)

            self.logger.info("Robot motion stopped")
        except Exception as e:
            self.logger.error(f"Failed to stop motion: {e}")
            raise
    
    def clear_errors(self):
        """清除机器人错误"""
        if not self.is_power_on:
            raise RuntimeError("Robot not powered on. Please call power_on() first before clearing errors.")
        
        try:
            self.robot_manage.clearErrors()
            self.logger.info("Robot errors cleared")
        except Exception as e:
            self.logger.error(f"Failed to clear errors: {e}")
            raise
    
    def set_freedrive_mode(self, enable: bool):
        """
        设置示教模式（拖动示教）
        
        Args:
            enable: True启用示教模式，False退出示教模式
        """
        if not self.is_power_on:
            raise RuntimeError("Robot not powered on. Please call power_on() first before setting freedrive mode.")
        
        try:
            if enable:
                self.run_time_machine.start()

                # 发起机器人自由驱动请求
                self.robot_manage.freedrive(True)
                self.logger.info("Freedrive mode enabled")
            else:
                self.robot_manage.freedrive(False)
                self.logger.info("Freedrive mode disabled")
        except Exception as e:
            self.logger.error(f"Failed to set freedrive mode: {e}")
            raise
    
    def set_compliance_mode(self, stiffness: float, damping: float):
        """
        设置柔顺模式参数
        
        Args:
            stiffness: 刚度参数
            damping: 阻尼参数
        """
        if not self.is_connected:
            raise RuntimeError("Robot not connected")
        
        try:
            # 根据实际 Aubo SDK API 调整
            if hasattr(self.robot, 'set_compliance_param'):
                self.robot.set_compliance_param(stiffness=stiffness, damping=damping)
                self.logger.info(f"Compliance mode set: stiffness={stiffness}, damping={damping}")
            else:
                super().set_compliance_mode(stiffness, damping)
        except Exception as e:
            self.logger.error(f"Failed to set compliance mode: {e}")
            raise

    def set_force_control_mdk(self, mass: np.ndarray, damping: np.ndarray, stiffness: np.ndarray):
        """
        设置力控模式参数
        
        Args:
            mass: 虚拟质量参数, shape=(n,)，通常n=6对应[x, y, z, rx, ry, rz]
            damping: 阻尼参数, shape=(n,)，通常n=6对应[x, y, z, rx, ry, rz]
            stiffness: 刚度参数, shape=(n,)，通常n=6对应[x, y, z, rx, ry, rz]
        """
        if not self.is_power_on:
            raise RuntimeError("Robot not powered on. Please call power_on() first before setting force control mdk.")
        
        try:
            self.robot_force_control.setDynamicModel(mass, damping, stiffness)
            self.logger.info(f"Force control mdk set: mass={mass}, damping={damping}, stiffness={stiffness}")
        except Exception as e:
            self.logger.error(f"Failed to set force control mdk: {e}")
            raise

    def set_force_control_target(
        self, 
        feature: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
        target_wrench: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
        compliance: List[bool] = [False, False, False, False, False, False], 
        speed_limits: List[float] = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0], 
        task_frame: int = 0):
        """
        设置力控目标
        
        Args:
            feature: 特征参数, List[float]，通常6个元素对应[x, y, z, rx, ry, rz]
            target_wrench: 目标力矩, List[float]，通常6个元素对应[x, y, z, rx, ry, rz]
            compliance: 力控方向选择参数, List[bool]，通常6个元素对应[x, y, z, rx, ry, rz]，True表示启用，False表示禁用
            speed_limits: 速度限制, List[float]，通常6个元素对应[x, y, z, rx, ry, rz]
            task_frame: 任务坐标系类型
        """
        if not self.is_power_on:
            raise RuntimeError("Robot not powered on. Please call power_on() first before setting force control target.")
        
        try:
            # 确保参数类型正确：将列表元素显式转换为正确的类型
            feature_list = [float(f) for f in feature]
            compliance_list = [bool(c) for c in compliance]  # 确保是 bool 类型
            target_wrench_list = [float(t) for t in target_wrench]
            speed_limits_list = [float(s) for s in speed_limits]
            
            task_frame_type = self.pyaubo_sdk.TaskFrameType.NONE
            if task_frame == 1:
                task_frame_type = self.pyaubo_sdk.TaskFrameType.POINT_FORCE
            elif task_frame == 2:
                task_frame_type = self.pyaubo_sdk.TaskFrameType.FRAME_FORCE
            elif task_frame == 3:
                task_frame_type = self.pyaubo_sdk.TaskFrameType.MOTION_FORCE
            elif task_frame == 4:
                task_frame_type = self.pyaubo_sdk.TaskFrameType.TOOL_FORCE
            
            self.robot_force_control.setTargetForce(feature_list, compliance_list, target_wrench_list, speed_limits_list, task_frame_type)
            self.logger.info(f"Force control target set: feature={feature}, target_wrench={target_wrench}, compliance={compliance}, speed_limits={speed_limits}, task_frame={task_frame}")
        except Exception as e:
            self.logger.error(f"Failed to set force control target: {e}")
            raise

    def enable_force_control(self, enable: bool = True)-> bool:
        """
        使能或关闭力控模式
        
        Args:
            enable: True 使能力控模式，False 关闭力控模式
        Returns:
            bool: 使能或关闭力控模式成功返回True，否则返回False
        """
        if not self.is_power_on:
            raise RuntimeError("Robot not powered on. Please call power_on() first before enabling force control.")
        
        try:
            if enable:
                if self.robot_force_control.isFcEnabled():
                    self.logger.info("The robot has already been force control mode.")
                    return True
                else:
                    self.robot_force_control.fcEnable()
                    self.logger.info("Enter force control mode")
                    return True
            else:
                if not self.robot_force_control.isFcEnabled():
                    self.logger.info("The robot has already quit force control mode.")
                    return True
                else:
                    self.robot_force_control.fcDisable()
                    self.logger.info("Quit force control mode")
                    return True
        except Exception as e:
            self.logger.error(f"Failed to enable/disable force control: {e}")
            return False

    def set_tcp_offset(self, offset: np.ndarray = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])):
        """
        设置 TCP 相对于机器人法兰盘的偏移量
        
        Args:
            offset: TCP 相对于机器人法兰盘的偏移量, shape=(6,)，通常n=6对应[x, y, z, Rx, Ry, Rz]
        """
        if not self.is_power_on:
            raise RuntimeError("Robot not powered on. Please call power_on() first before setting TCP offset.")
        
        try:
            self.robot_config.setTcpOffset(offset)
            self.logger.info(f"TCP offset set: {offset}")
        except Exception as e:
            self.logger.error(f"Failed to set TCP offset: {e}")
            raise

    def set_sensor_offset(
        self, 
        sensor_name: str = "kw_ftsensor",
        offset: np.ndarray = np.array([0.0, 0.0, 0.047, 0.0, 0.0, 0.0]) # 默认坤维传感器偏移量
    ): 
        """
        设置传感器相对于机器人法兰盘的偏移量(设置前连带选择传感器类型)

        Note:
            不是所有机器人都支持此功能
        Args:
            sensor_name: 传感器名称
            offset: 传感器相对于机器人法兰盘的偏移量, shape=(6,)，通常n=6对应[x, y, z, Rx, Ry, Rz]
        """
        if not self.is_power_on:
            raise RuntimeError("Robot not powered on. Please call power_on() first before setting sensor offset.")
        
        try:
            # 选择传感器类型
            if sensor_name == "embedded":
                self.robot_config.selectTcpForceSensor("embedded")
            elif sensor_name == "kw_ftsensor":
                self.robot_config.selectTcpForceSensor("kw_ftsensor")
            else:
                raise ValueError(f"Unsupported sensor name: {sensor_name}") 

            # 判断传感器是否选择成功
            force_sensors = self.robot_state.getTcpForceSensors()
            if np.allclose(force_sensors, 0.0, atol=1e-6):
                self.is_connected_sensor = False
                raise RuntimeError(f"Failed to select sensor: {sensor_name}, the sensor is not connected.")
            else:
                self.is_connected_sensor = True
                self.logger.info(f"Sensor {sensor_name} selected successfully")

            # 设置传感器偏移量
            self.robot_config.setTcpForceSensorPose(offset)
            self.logger.info(f"Sensor offset set: {offset}")

        except Exception as e:
            self.logger.error(f"Failed to set sensor offset: {e}")
            raise

    def payload_identify(
        self,
        mode: int = 1   # 1: 三点标定，2: 静态辨识
        ):
        """
        基于传感器的负载辨识并设置以下参数:负载质量、质心和传感器偏置

        Note:
            不是所有机器人都支持此功能
        Args:
            mode: 负载辨识模式
        """
        if not self.is_power_on:
            raise RuntimeError("Robot not powered on. Please call power_on() first before payload identification.")
        
        if not self.is_connected_sensor:
            raise RuntimeError("Sensor not connected. Please call set_sensor_offset() first before payload identification.")

        try:
            if mode == 1:
                self.logger.info("Starting payload identification in three-point calibration mode")

                # 负载辨识参考点（关节角度单位：rad）
                joint1 = np.array([-0.261799, 0.261799, 1.309,    1.0472,   1.39626, 0.0])
                joint2 = np.array([-0.628319, 0.471239, 1.65806, -0.471239, 0.0,     0.0])
                joint3 = np.array([-0.628319, 0.366519, 1.74533, -0.10472,  1.5708,  0.0])
                
                self.move_to_joint_positions(joint1, blocking=True)
                self.logger.info("Move to calibration point 1 successfully")
                time.sleep(1.0)
                tcp_force1 = self.robot_state.getTcpForceSensors()
                tcp_pose1 = self.robot_algorithm.forwardKinematics(joint1)
                self.logger.info(f"TCP force at calibration point 1: {tcp_force1}, TCP pose at calibration point 1: {tcp_pose1}")

                self.move_to_joint_positions(joint2, blocking=True)
                self.logger.info("Move to calibration point 2 successfully")
                time.sleep(1.0)
                tcp_force2 = self.robot_state.getTcpForceSensors()
                tcp_pose2 = self.robot_algorithm.forwardKinematics(joint2)
                self.logger.info(f"TCP force at calibration point 2: {tcp_force2}, TCP pose at calibration point 2: {tcp_pose2}")

                self.move_to_joint_positions(joint3, blocking=True)
                self.logger.info("Move to calibration point 3 successfully")
                time.sleep(1.0)
                tcp_force3 = self.robot_state.getTcpForceSensors()
                tcp_pose3 = self.robot_algorithm.forwardKinematics(joint3)
                self.logger.info(f"TCP force at calibration point 3: {tcp_force3}, TCP pose at calibration point 3: {tcp_pose3}")

                # 三点标定接口(传入的calib_poses为TCP位姿, 算法接口要求为法兰位姿, SDK内部转换为了法兰盘位姿)
                calib_forces = [list(tcp_force1), list(tcp_force2), list(tcp_force3)]
                calib_poses = [list(tcp_pose1[0]), list(tcp_pose2[0]), list(tcp_pose3[0])]
                calib_result = self.robot_algorithm.calibrateTcpForceSensor(calib_forces, calib_poses)   
                sensor_offset = calib_result[0]
                com = calib_result[1]
                mass = calib_result[2]
                self.logger.info(f"Payload identification: mass={mass}, com={com}, sensor_offset={sensor_offset}")

                # 根据标定结果设置负载参数 + 传感器偏置
                self.robot_config.setPayload(mass, com, [0.0], [0.0])
                self.robot_config.setTcpForceOffset(sensor_offset)
                self.logger.info(f"Payload set: mass={mass}, com={com}, sensor_offset={sensor_offset}")
            elif mode == 2:
                self.logger.info("Starting payload identification in static mode")

                # 获取当前传感器数据作为传感器偏置
                tcp_force = self.robot_state.getTcpForceSensors()
                sensor_offset = tcp_force
                self.robot_config.setPayload(0.0, [0.0, 0.0, 0.0], [0.0], [0.0])
                self.robot_config.setTcpForceOffset(sensor_offset)
                self.logger.info(f"Payload set: mass=0.0, com=[0.0, 0.0, 0.0], sensor_offset={sensor_offset}")
            else:
                raise ValueError(f"Unsupported payload identify mode: {mode}")

        except Exception as e:
            self.logger.error(f"Failed to payload identification: {e}")
            raise

# 测试代码
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建机器人接口
    robot = AuboInterface(robot_ip="192.168.1.100")
    
    # 使用上下文管理器
    try:
        with robot:
            is_connected = robot.connect()  
            if  is_connected:
                print("Robot connect suceessfully")
                # 连接成功后上电
                if robot.power_on():
                    print("Robot power on successfully")
                    print("\n=== 机器人状态 ===")
                    try:
                        joint_pos = robot.get_joint_positions()
                        print("关节位置 (rad):", joint_pos)
                    except Exception as e:
                        print("get_joint_positions 调用失败:", e)

                    try:
                        tcp_pose = robot.get_tcp_pose()
                        print("TCP 位姿 [x, y, z, qw, qx, qy, qz]:", tcp_pose)
                    except Exception as e:
                        print("get_tcp_pose 调用失败:", e)

                    try:
                        tcp_vel = robot.get_tcp_velocity()
                        print("TCP 速度 [vx, vy, vz, wx, wy, wz]:", tcp_vel)
                    except Exception as e:
                        print("get_tcp_velocity 调用失败:", e)

                    try:
                        tcp_force = robot.get_tcp_force()
                        print("TCP 力 [fx, fy, fz]:", tcp_force)
                    except Exception as e:
                        print("get_tcp_force 调用失败:", e)

                    try:
                        tcp_torque = robot.get_tcp_torque()
                        print("TCP 力矩 [tx, ty, tz]:", tcp_torque)
                    except Exception as e:
                        print("get_tcp_torque 调用失败:", e)

                    try:
                        state = robot.get_robot_state()
                        print("\n完整状态字典 get_robot_state():")
                        for k, v in state.items():
                            print(f"{k}: {v}")
                    except Exception as e:
                        print("get_robot_state 调用失败:", e)

                    # 如果只想简单验证接口，可以在此提前退出
                    # exit()
                else:
                    print("Robot power on failed")
                
            # 测试 move_to_joint_positions（小心！！！务必确认周围无障碍物）
            response = input("\n是否测试小幅关节移动 move_to_joint_positions? (y/n): ")
            response_stop = input("\n是否测试小幅关节移动过程中停止 stop_motion? (y/n): ")

            if response.lower() == 'y':
                try:
                    current_joints = robot.get_joint_positions()
                    target_joints = current_joints.copy()
                    # 关节1 小幅 +0.1 rad，避免大幅运动
                    target_joints[0] += 0.1
                    
                    print("当前关节位置:", current_joints)
                    print("目标关节位置:", target_joints)
                    
                    # 根据 response_stop 的值确定 blocking 参数
                    # 如果要测试停止功能，需要设置为 False（非阻塞），这样才能在运动过程中调用 stop_motion()
                    blocking = response_stop.lower() != 'y'
                    
                    robot.move_to_joint_positions(
                        target_joints,
                        velocity=0.5,
                        acceleration=0.5,
                        blocking=blocking,
                    )
                    
                    # 如果用户想测试停止功能，在运动开始后稍等片刻然后停止
                    if response_stop.lower() == 'y':
                        print("运动已开始，等待 1 秒后停止...")
                        time.sleep(1.0)
                        print("调用 stop_motion() 停止运动...")
                        robot.stop_motion()
                        print("stop_motion() 调用完成！")
                    else:
                        print("move_to_joint_positions 测试完成！")
                except Exception as e:
                    print("move_to_joint_positions 测试失败:", e)

            # 测试 move_tcp_pose（小心！！！务必确认周围无障碍物）
            response = input("\n是否测试小幅 TCP 直线移动 move_tcp_pose? (y/n): ")
            response_stop = input("\n是否测试小幅 TCP 移动过程中停止 stop_motion? (y/n): ")
            
            if response.lower() == 'y':
                try:
                    current_tcp = robot.get_tcp_pose()
                    target_tcp = current_tcp.copy()
                    # Z 轴方向小幅上移 5 cm，避免大幅运动
                    target_tcp[2] += 0.05

                    print("当前 TCP 位姿:", current_tcp)
                    print("目标 TCP 位姿:", target_tcp)

                    # 根据 response_stop 的值确定 blocking 参数
                    # 如果要测试停止功能，需要设置为 False（非阻塞），这样才能在运动过程中调用 stop_motion()
                    blocking = response_stop.lower() != 'y'
                    
                    robot.move_tcp_pose(
                        target_tcp,
                        velocity=0.05,
                        acceleration=0.05,
                        blocking=blocking,
                    )
                    
                    # 如果用户想测试停止功能，在运动开始后稍等片刻然后停止
                    if response_stop.lower() == 'y':
                        print("运动已开始，等待 1 秒后停止...")
                        time.sleep(1.0)
                        print("调用 stop_motion() 停止运动...")
                        robot.stop_motion()
                        print("stop_motion() 调用完成！")
                    else:
                        print("move_tcp_pose 测试完成！")
                except Exception as e:
                    print("move_tcp_pose 测试失败:", e)

            # 测试伺服到目标关节位置（小心！！！务必确认周围无障碍物）                 
            response = input("\n是否测试伺服到目标关节位置? (y/n): ")
            response_stop = input("\n是否测试伺服过程中停止 stop_motion? (y/n): ")
            
            if response.lower() == 'y':
                try:
                    current_joints = robot.get_joint_positions()
                    print("当前关节位置:", current_joints)
                    target_joints = current_joints.copy()
                    ret = robot.set_servo_mode(1)
                    if ret:
                        # 如果用户想测试停止功能，在循环进行到一半时停止
                        stop_at_iteration = 50 if response_stop.lower() == 'y' else None
                        
                        for i in range(100):
                            target_joints += 0.001  # 每个关节都增加0.001 
                            print("目标关节位置:", target_joints)
                            ret = robot.servo_to_joint_positions(target_joints)
                            if ret == 0:
                                print(f"伺服到目标关节位置 测试第{i+1}次完成！")
                            else:
                                print(f"伺服到目标关节位置 测试第{i+1}次失败: {ret}")
                                break
                            
                            # 如果用户想测试停止功能，在指定迭代次数后停止
                            if stop_at_iteration is not None and i + 1 == stop_at_iteration:
                                print(f"\n已执行 {stop_at_iteration} 次伺服，调用 stop_motion() 停止运动...")
                                robot.stop_motion()
                                print("stop_motion() 调用完成！")
                                break
                        
                        robot.enable_servo_mode(False)
                except Exception as e:
                    print(f"伺服到目标关节位置 测试失败: {e}")

            # 测试设置示教模式（拖动示教）
            response = input("\n是否测试设置示教模式? (y/n): ")
            if response.lower() == 'y':
                try:
                    robot.set_freedrive_mode(True)
                    print("示教模式已启用，请拖动机器人到目标位置")
                    time.sleep(30.0)
                    robot.set_freedrive_mode(False)
                    print("示教模式已退出")
                except Exception as e:
                    print(f"设置示教模式 测试失败: {e}")

            # 测试负载辨识
            response = input("\n是否测试负载辨识? (y/n): ")
            if response.lower() == 'y':
                try:
                    tcp_offset = np.array([0.0, 0.0, 0.25, 0.0, 0.0, 0.0])
                    robot.set_tcp_offset(tcp_offset)
                    robot.set_sensor_offset()
                    robot.payload_identify(mode=1)
                    print("负载辨识测试完成！")
                except Exception as e:
                    print(f"负载辨识测试失败: {e}")

            # 测试力控
            response = input("\n是否测试力控? (y/n): ")
            if response.lower() == 'y':
                try:
                    tcp_offset = np.array([0.0, 0.0, 0.25, 0.0, 0.0, 0.0])
                    robot.set_tcp_offset(tcp_offset)
                    robot.set_sensor_offset()
                    robot.payload_identify(mode=1)
                    print("负载辨识测试完成！")
                    time.sleep(1.0)
                    robot.move_to_joint_positions(np.array([0.0, 0.0, 1.57, 0.0, 1.57, 0.0]), blocking=True)
                    print("移动到初始位置完成！")
                    time.sleep(1.0)

                    mass = np.array([10.0, 10.0, 10.0, 2.0, 2.0, 2.0])
                    damp = np.array([200.0, 200.0, 200.0, 20.0, 20.0, 20.0])    
                    stiff = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    robot.set_force_control_mdk(mass=mass, damping=damp, stiffness=stiff)
                    
                    feature = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    compliance = [False, False, True, False, False, False]
                    target_wrench = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    speed_limits = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
                    robot.set_force_control_target(
                        feature = feature,
                        compliance = compliance, 
                        target_wrench = target_wrench, 
                        speed_limits = speed_limits,
                        task_frame = 0
                    )

                    robot.enable_force_control(True)
                    print("力控已开始，等待 15 秒后停止...")
                    time.sleep(15.0)
                    print("力控测试完成！")
                    robot.enable_force_control(False)
                except Exception as e:
                    print(f"力控测试失败: {e}")
            
            # 测试伺服模式下的力控功能
            response = input("\n是否测试伺服模式下的力控功能? (y/n): ")
            if response.lower() == 'y':
                try:
                    # 读取轨迹文件并加载轨迹点
                    # file = open('/home/aubo/mujoco_env/mujoco_env/real/plan_movej_trajectory.csv')
                    # traj = []
                    # for line in file:
                    #     str_list = line.split(",")
                    #     float_list = []
                    #     for strs in str_list:
                    #         float_list.append(float(strs))
                    #     traj.append(float_list)
                    
                    # traj_sz = len(traj)
                    # if traj_sz == 0:
                    #     print("没有轨迹点")
                    # else:
                    #     print("加载的轨迹点数量为: ", traj_sz)

                    # # 负载辨识
                    # tcp_offset = np.array([0.0, 0.0, 0.25, 0.0, 0.0, 0.0])
                    # robot.set_tcp_offset(tcp_offset)
                    # robot.set_sensor_offset()
                    # robot.payload_identify(mode=1)
                    # print("负载辨识完成！")

                    # # 关节运动到第一个点，当前位置要与轨迹中的第一个点一致
                    # robot.move_to_joint_positions(
                    #     np.array(traj[0]),
                    #     velocity = 0.5,
                    #     acceleration = 0.5,
                    #     blocking = True,
                    # )
                    
                    mass = np.array([10.0, 10.0, 10.0, 2.0, 2.0, 2.0])
                    damp = np.array([200.0, 200.0, 200.0, 20.0, 20.0, 20.0])    
                    stiff = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    robot.set_force_control_mdk(mass=mass, damping=damp, stiffness=stiff)
                    
                    feature = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    compliance = [False, True, False, False, False, False]
                    target_wrench = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    speed_limits = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
                    robot.set_force_control_target(
                        feature = feature,
                        compliance = compliance, 
                        target_wrench = target_wrench, 
                        speed_limits = speed_limits,
                        task_frame = 0
                    )

                    robot.enable_force_control(True)
                    print("开启力控模式...")
                    robot.set_servo_mode(7)
                    print("设置伺服模式为7...")
                    for q in traj:
                        robot.servo_to_joint_positions(np.array(q))

                    print("伺服模式下的力控功能测试完成！")
                    robot.set_servo_mode(0)
                    robot.enable_force_control(False)
                except Exception as e:
                    print(f"伺服模式下的力控功能测试失败: {e}")
    except Exception as e:
        print(f"错误: {e}")
