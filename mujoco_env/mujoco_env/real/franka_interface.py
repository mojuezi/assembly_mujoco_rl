"""
Franka 机器人接口实现

基于 Flask 服务器和 ROS 通信实现的真机接口（从 serl_robot_infra 移植）
"""

import numpy as np
from typing import Dict, Optional
import requests
import logging
from scipy.spatial.transform import Rotation

from mujoco_env.mujoco_env.real.robot_interface import RobotInterface


class FrankaInterface(RobotInterface):
    """
    Franka Panda 机器人接口
    
    通过 Flask 服务器与 Franka 机器人通信，服务器端使用 ROS 和 libfranka。
    
    Requirements:
        1. 启动 Franka 服务器: python franka_server.py
        2. 服务器默认端口: 5000
    
    Server Endpoints:
        - POST /getstate: 获取机器人状态
        - POST /pose: 设置目标位姿
        - POST /getpos_euler: 获取当前位姿(欧拉角)
        - POST /clearerr: 清除错误
        - POST /jointreset: 关节位置重置
        - POST /update_param: 更新控制参数
    
    Examples:
        >>> robot = FrankaInterface(
        ...     robot_ip="192.168.1.1",
        ...     server_ip="127.0.0.1",
        ...     server_port=5000
        ... )
        >>> if robot.connect():
        ...     pose = robot.get_tcp_pose()
        ...     print(f"Current TCP: {pose}")
        ...     robot.disconnect()
    """
    
    def __init__(
        self,
        robot_ip: str = "192.168.1.1",
        server_ip: str = "127.0.0.1",
        server_port: int = 5000,
        timeout: float = 5.0,
        **kwargs
    ):
        """
        初始化 Franka 机器人接口
        
        Args:
            robot_ip: Franka 机器人 IP 地址（用于标识，实际通过服务器连接）
            server_ip: Flask 服务器 IP 地址
            server_port: Flask 服务器端口
            timeout: 请求超时时间（秒）
            **kwargs: 其他配置参数
        """
        super().__init__(robot_ip, **kwargs)
        
        self.server_ip = server_ip
        self.server_port = server_port
        self.timeout = timeout
        self.base_url = f"http://{server_ip}:{server_port}"
        self.dof = 7  # Franka Panda 是 7 自由度
        
        # 缓存状态
        self._last_state = None
    
    def connect(self) -> bool:
        """
        连接到 Franka 机器人服务器
        
        Returns:
            bool: 连接成功返回True
        """
        try:
            # 测试连接 - 尝试获取状态
            response = requests.post(
                f"{self.base_url}/getstate",
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                self.logger.info(
                    f"Successfully connected to Franka server at {self.base_url}"
                )
                self.is_connected = True
                return True
            else:
                self.logger.error(
                    f"Connection failed with status code: {response.status_code}"
                )
                return False
        except requests.exceptions.ConnectionError:
            self.logger.error(
                f"Cannot connect to Franka server at {self.base_url}. "
                "Make sure the server is running."
            )
            return False
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            return False
    
    def disconnect(self):
        """断开连接（Flask 服务器无需显式断开）"""
        if self.is_connected:
            self.logger.info("Disconnected from Franka server")
            self.is_connected = False
    
    def _send_request(self, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """
        发送请求到服务器
        
        Args:
            endpoint: API 端点（如 "/getstate"）
            data: 请求数据
        
        Returns:
            Dict: 响应数据
        """
        if not self.is_connected:
            raise RuntimeError("Robot not connected")
        
        try:
            url = f"{self.base_url}{endpoint}"
            if data is not None:
                response = requests.post(url, json=data, timeout=self.timeout)
            else:
                response = requests.post(url, timeout=self.timeout)
            
            if response.status_code == 200:
                return response.json()
            else:
                raise RuntimeError(
                    f"Request failed with status {response.status_code}: {response.text}"
                )
        except requests.exceptions.Timeout:
            self.logger.error(f"Request to {endpoint} timed out")
            raise
        except Exception as e:
            self.logger.error(f"Request error: {e}")
            raise
    
    def get_joint_positions(self) -> np.ndarray:
        """
        获取当前关节位置
        
        Returns:
            np.ndarray: 关节位置 (弧度), shape=(7,)
        """
        state = self._send_request("/getstate")
        self._last_state = state
        return np.array(state["q"], dtype=np.float64)
    
    def get_joint_velocities(self) -> np.ndarray:
        """
        获取当前关节速度
        
        Returns:
            np.ndarray: 关节速度 (rad/s), shape=(7,)
        """
        state = self._send_request("/getstate")
        self._last_state = state
        return np.array(state["dq"], dtype=np.float64)
    
    def get_tcp_pose(self) -> np.ndarray:
        """
        获取 TCP 位姿
        
        Returns:
            np.ndarray: TCP位姿 [x, y, z, qw, qx, qy, qz], shape=(7,)
        """
        state = self._send_request("/getstate")
        self._last_state = state
        return np.array(state["pose"], dtype=np.float64)
    
    def get_tcp_velocity(self) -> np.ndarray:
        """
        获取 TCP 速度
        
        Returns:
            np.ndarray: TCP速度 [vx, vy, vz, wx, wy, wz], shape=(6,)
        """
        state = self._send_request("/getstate")
        self._last_state = state
        return np.array(state["vel"], dtype=np.float64)
    
    def get_tcp_force(self) -> np.ndarray:
        """
        获取 TCP 力传感器读数
        
        Returns:
            np.ndarray: TCP力 [fx, fy, fz], shape=(3,)
        """
        state = self._send_request("/getstate")
        self._last_state = state
        return np.array(state["force"], dtype=np.float64)
    
    def get_tcp_torque(self) -> np.ndarray:
        """
        获取 TCP 力矩传感器读数
        
        Returns:
            np.ndarray: TCP力矩 [tx, ty, tz], shape=(3,)
        """
        state = self._send_request("/getstate")
        self._last_state = state
        return np.array(state["torque"], dtype=np.float64)
    
    def move_to_joint_positions(
        self,
        positions: np.ndarray,
        velocity: float = 0.5,
        acceleration: float = 0.5,
        blocking: bool = False
    ):
        """
        移动到目标关节位置
        
        Note:
            Franka 服务器当前不支持关节空间运动，
            请使用 move_tcp_pose 或扩展服务器功能
        """
        self.logger.warning(
            "Franka server does not support joint space motion. "
            "Use move_tcp_pose instead or extend the server."
        )
        raise NotImplementedError("Joint space motion not implemented in Franka server")
    
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
            velocity: 速度 (m/s) - 注意：当前服务器可能不使用此参数
            acceleration: 加速度 (m/s^2) - 注意：当前服务器可能不使用此参数
            blocking: 是否阻塞等待完成 - 注意：当前服务器不支持阻塞
        """
        if len(pose) != 7:
            raise ValueError(f"Expected 7-element pose, got {len(pose)}")
        
        # 发送位姿命令
        data = {"arr": pose.astype(np.float32).tolist()}
        self._send_request("/pose", data)
        
        if blocking:
            self.logger.warning("Franka server does not support blocking mode")
    
    def get_robot_state(self) -> Dict:
        """
        获取完整机器人状态
        
        Returns:
            Dict: 机器人状态字典
        """
        state = self._send_request("/getstate")
        self._last_state = state
        
        return {
            "joint_positions": np.array(state["q"]),
            "joint_velocities": np.array(state["dq"]),
            "tcp_pose": np.array(state["pose"]),
            "tcp_velocity": np.array(state["vel"]),
            "tcp_force": np.array(state["force"]),
            "tcp_torque": np.array(state["torque"]),
            "jacobian": np.array(state.get("jacobian", [])),
            "gripper_pos": state.get("gripper_pos", 0.0),
            "errors": state.get("errors", []),
        }
    
    def stop_motion(self):
        """
        立即停止机器人运动
        
        Note:
            当前 Franka 服务器可能不支持此功能，
            需要在服务器端添加相应的 ROS 服务调用
        """
        self.logger.warning("Stop motion not implemented in Franka server")
        # 可以尝试发送当前位姿作为目标来停止运动
        try:
            current_pose = self.get_tcp_pose()
            self.move_tcp_pose(current_pose)
            self.logger.info("Sent current pose as target to stop motion")
        except Exception as e:
            self.logger.error(f"Failed to stop motion: {e}")
            raise
    
    def clear_errors(self):
        """清除机器人错误"""
        self._send_request("/clearerr")
        self.logger.info("Robot errors cleared")
    
    def joint_reset(self):
        """
        关节位置重置（返回到 Home 位置）
        
        这是 Franka 服务器特有的功能
        """
        self._send_request("/jointreset")
        self.logger.info("Joint reset commanded")
    
    def update_control_param(self, stiffness: float, damping: float):
        """
        更新控制参数（阻抗控制）
        
        Args:
            stiffness: 刚度参数
            damping: 阻尼参数
        """
        data = {
            "stiffness": stiffness,
            "damping": damping,
        }
        self._send_request("/update_param", data)
        self.logger.info(f"Control params updated: stiffness={stiffness}, damping={damping}")
    
    def set_compliance_mode(self, stiffness: float, damping: float):
        """
        设置柔顺模式参数（调用 update_control_param）
        
        Args:
            stiffness: 刚度参数
            damping: 阻尼参数
        """
        self.update_control_param(stiffness, damping)
    
    def get_tcp_pose_euler(self) -> np.ndarray:
        """
        获取 TCP 位姿（欧拉角格式）
        
        Returns:
            np.ndarray: TCP位姿 [x, y, z, roll, pitch, yaw], shape=(6,)
        """
        result = self._send_request("/getpos_euler")
        return np.array(result["pose"], dtype=np.float64)
    
    def get_jacobian(self) -> Optional[np.ndarray]:
        """
        获取雅可比矩阵
        
        Returns:
            np.ndarray: 雅可比矩阵 shape=(6, 7)
        """
        if self._last_state is None:
            state = self._send_request("/getstate")
            self._last_state = state
        else:
            state = self._last_state
        
        jacobian = state.get("jacobian", None)
        if jacobian:
            return np.array(jacobian).reshape(6, self.dof)
        return None


# 测试代码
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建机器人接口
    robot = FrankaInterface(
        robot_ip="192.168.1.1",
        server_ip="127.0.0.1",
        server_port=5000
    )
    
    # 使用上下文管理器
    try:
        with robot:
            print("\n=== 机器人状态 ===")
            state = robot.get_robot_state()
            print(f"关节位置: {state['joint_positions']}")
            print(f"TCP 位姿: {state['tcp_pose']}")
            print(f"TCP 速度: {state['tcp_velocity']}")
            print(f"TCP 力: {state['tcp_force']}")
            print(f"雅可比矩阵形状: {state['jacobian'].shape if len(state['jacobian']) > 0 else 'N/A'}")
            
            # 测试移动（小心！）
            response = input("\n是否测试小幅移动? (y/n): ")
            if response.lower() == 'y':
                current_tcp = robot.get_tcp_pose()
                target_tcp = current_tcp.copy()
                target_tcp[2] += 0.05  # Z轴上移5cm
                
                print(f"移动到: {target_tcp}")
                robot.move_tcp_pose(target_tcp, velocity=0.05)
                print("移动命令已发送（非阻塞）")
    except Exception as e:
        print(f"错误: {e}")

