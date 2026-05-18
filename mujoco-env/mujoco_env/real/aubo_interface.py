"""
Aubo 机器人接口实现

基于 Aubo Python SDK 实现的真机接口
"""

import numpy as np
from typing import Dict, Optional
# from scipy.spatial.transform import Rotation
import logging
import time
from dm_robotics.transformations import transformations as tr

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
        self.is_power_on = False
        
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
                    self.robot_interface = self.robot.getRobotInterface(self.robot_name)
                    self.robot_state = self.robot_interface.getRobotState()
                    self.robot_manage = self.robot_interface.getRobotManage()
                    self.robot_motion = self.robot_interface.getMotionControl()
                    self.robot_motion.setSpeedFraction(0.25)
                    self.robot_config = self.robot_interface.getRobotConfig()
                    self.robot_algorithm = self.robot_interface.getRobotAlgorithm()

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
                self.is_power_on = True
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
        # TODO
        if not self.is_power_on:
            raise RuntimeError("Robot not powered on. Please call power_on() first before getting TCP pose.")
        
        try:
            # 获取位姿（Aubo SDK 通常返回 [x, y, z, rx, ry, rz]）
            pose = self.robot_state.getTcpPose()
            
            # 提取位置和欧拉角
            position = np.array(pose[:3], dtype=np.float64)
            rotation_euler = np.array(pose[3:], dtype=np.float64)  # 欧拉角 (rx, ry, rz)
            
            # 转换欧拉角为四元数 [qw, qx, qy, qz]
            # rotation_quat_scipy = Rotation.from_euler('xyz', rotation_euler).as_quat()  # [qx, qy, qz, qw]
            # quat = np.array([
            #     rotation_quat_scipy[3],  # qw
            #     rotation_quat_scipy[0],  # qx
            #     rotation_quat_scipy[1],  # qy
            #     rotation_quat_scipy[2]   # qz
            # ])

            quat = tr.euler_to_quat(rotation_euler, 'ZYX')
            
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
    
    def enable_servo_mode(self, enable: bool = True) -> bool:
        """
        使能或关闭伺服运动模式
        
        Args:
            enable: True 使能伺服模式，False 关闭伺服模式
        Returns:
            bool: 使能或关闭伺服运动模式成功返回True，否则返回False
        
        """
        if not self.is_power_on:
            raise RuntimeError("Robot not powered on. Please call power_on() first before enabling servo mode.")
        
        try:
            if enable:
                self.robot_motion.setServoModeSelect(1)
                time.sleep(0.01)     # 等待伺服模式使能完成
                if 1 == self.robot_motion.getServoModeSelect():
                    return True
                else:
                    self.logger.error(f"Failed to enable servo mode: {self.robot_motion.getServoModeSelect()}")
                    return False
                # self.robot_motion.setServoMode(True)
                # i = 0
                # while not self.robot_motion.isServoModeEnabled():
                #     i = i + 1
                #     if i > 5:
                #         print("开启Servo模式失败！当前的Servo模式是： ", self.robot_motion.isServoModeEnabled())
                #         return -1
                #     time.sleep(0.005)
            else:
                self.robot_motion.setServoMode(0)
                time.sleep(0.1)     # 等待伺服模式关闭完成
                if 0 == self.robot_motion.getServoModeSelect():
                    return True
                else:
                    self.logger.error(f"Failed to disable servo mode: {self.robot_motion.getServoModeSelect()}")
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
    
    def servo_to_tcp_positions(
        self,
        pose: np.ndarray
    ) -> int:
        """
        伺服到目标tcp位置
        
        Args:
            pose: 目标tcp位置 (弧度), shape=(dof,)
        Returns:
            int: 返回码。AUBO_QUEUE_FULL 表示队列已满，需要等待。其他错误码见SDK文档。
        """
        if not self.is_power_on:
            raise RuntimeError("Robot not powered on. Please call power_on() first before servoing to joint positions.")
        
        if len(pose) != 7:
            raise ValueError(f"Expected 7-element pose, got {len(pose)}")
        
        try:
            acceleration = 0.0
            velocity = 0.0
            run_time = 0.01       # 运行时间: t 值越大, 机器臂运动越慢, 反之运动越快; 该参数最优值为连续调用 servoJoint 接口的间隔时间
            lookahead_time = 0.0 
            gain = 0            

            position = pose[:3]
            quat = pose[3:]  # [qw, qx, qy, qz]
            euler = tr.quat_to_euler(quat, 'ZYX')
            target_pose = np.concatenate([position, euler])

            ret = self.robot_motion.servoCartesian(
                target_pose.tolist(),
                acceleration,
                velocity,
                run_time,
                lookahead_time,
                gain
            )

            time.sleep(run_time)
            # while ret == 2:
            #     time.sleep(0.005)
            #     print(f"Queue is full, waiting for 5ms")
            #     ret = self.robot_motion.servoCartesian(
            #         target_pose.tolist(),
            #         acceleration,
            #         velocity,
            #         run_time,
            #         lookahead_time,
            #         gain
            #     )
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
            # quat_scipy = [quat[1], quat[2], quat[3], quat[0]]  # [qx, qy, qz, qw]
            # euler = Rotation.from_quat(quat_scipy).as_euler('xyz')

            if quat.shape[0] == 4: 
                euler = tr.quat_to_euler(quat, 'ZYX')
            else: 
                euler = quat
            
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
                "tcp_wrench": self.robot_state.getTcpForce(),
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
    
    def stop_motion(self):
        """立即停止机器人运动"""
        if not self.is_power_on:
            raise RuntimeError("Robot not powered on. Please call power_on() first before stopping motion.")
        
        try:
            self.robot_motion.stop()
            self.logger.info("Robot motion stopped")
        except Exception as e:
            self.logger.error(f"Failed to stop motion: {e}")
            raise
    
    def clear_errors(self):
        """清除机器人错误"""
        if not self.is_power_on:
            raise RuntimeError("Robot not powered on. Please call power_on() first before clearing errors.")
        
        try:
            pass
            # self.robot_manage.clearErrors()
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
        if not self.is_connected:
            raise RuntimeError("Robot not connected")
        
        try:
            if enable:
                self.robot.enable_freedrive_mode()
                self.logger.info("Freedrive mode enabled")
            else:
                self.robot.disable_freedrive_mode()
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
            if response.lower() == 'y':
                try:
                    current_joints = robot.get_joint_positions()
                    target_joints = current_joints.copy()
                    # 关节1 小幅 +0.1 rad，避免大幅运动
                    target_joints[0] += 0.1
                    
                    print("当前关节位置:", current_joints)
                    print("目标关节位置:", target_joints)
                    
                    robot.move_to_joint_positions(
                        target_joints,
                        velocity=0.5,
                        acceleration=0.5,
                        blocking=True,
                    )
                    print("move_to_joint_positions 测试完成！")
                except Exception as e:
                    print("move_to_joint_positions 测试失败:", e)

            # 测试 move_tcp_pose（小心！！！务必确认周围无障碍物）
            response = input("\n是否测试小幅 TCP 直线移动 move_tcp_pose? (y/n): ")
            if response.lower() == 'y':
                try:
                    current_tcp = robot.get_tcp_pose()
                    target_tcp = current_tcp.copy()
                    # Z 轴方向小幅上移 5 cm，避免大幅运动
                    target_tcp[2] += 0.05

                    print("当前 TCP 位姿:", current_tcp)
                    print("目标 TCP 位姿:", target_tcp)

                    robot.move_tcp_pose(
                        target_tcp,
                        velocity=0.05,
                        acceleration=0.05,
                        blocking=True,
                    )
                    print("move_tcp_pose 测试完成！")
                except Exception as e:
                    print("move_tcp_pose 测试失败:", e)

            # 测试伺服到目标关节位置（小心！！！务必确认周围无障碍物）                 
            response = input("\n是否测试伺服到目标关节位置? (y/n): ")
            if response.lower() == 'y':
                try:
                    current_joints = robot.get_joint_positions()
                    print("当前关节位置:", current_joints)
                    target_joints = current_joints.copy()
                    ret = robot.enable_servo_mode(True)
                    if ret:
                        for i in range(100):
                            target_joints += 0.001  # 每个关节都增加0.001 
                            print("目标关节位置:", target_joints)
                            ret = robot.servo_to_joint_positions(target_joints)
                            if ret == 0:
                                print(f"伺服到目标关节位置 测试第{i+1}次完成！")
                            else:
                                print(f"伺服到目标关节位置 测试第{i+1}次失败: {ret}")
                                break
                        robot.enable_servo_mode(False)
                except Exception as e:
                    print(f"伺服到目标关节位置 测试失败: {e}")

    except Exception as e:
        print(f"错误: {e}")
