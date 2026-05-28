"""
键盘遥操作工具

提供通过键盘控制机器人末端执行器的功能。

快捷键说明：
- <ARROW> : 沿 x/y 轴移动末端执行器
- <CTRL + ARROW> : 沿 z 轴移动末端执行器
- <SHIFT + ARROW> : 绕 x/y 轴旋转末端执行器
- <CTRL + SHIFT + ARROW> : 绕 z 轴旋转末端执行器
- <CAPSLOCK> : 打开/关闭夹爪
- <ALT> : 切换机械臂（双臂情况下）
- <R> : 重置环境
- <ESC> : 退出

作者: Liu Gang
日期: 2025-12-20
"""

import time
import logging
import numpy as np
import jax
if jax.default_backend() == "cpu": 
    try:
        from pynput import keyboard
    except ImportError:
        raise ImportError(
            "pynput is required but not installed. "
            "Please install it using: pip install pynput"
        )

# 导入transform工具
from mujoco_env.mujoco_env.utils import transform as T


class KeyboardIO:
    """
    键盘输入/输出控制类
    
    通过键盘控制机器人末端执行器的位置和姿态。
    支持双臂机器人的切换控制。
    
    Attributes:
        _pos_step: 位置步长 (m)
        _rot_step: 旋转步长 (rad)
        _end_pos_offset: 末端位置偏移
        _end_rot_offset: 末端旋转偏移（旋转矩阵）
        _reset_flag: 重置标志
        _exit_flag: 退出标志
        _gripper_flag: 夹爪状态标志 (0=打开, 1=关闭)
        _agent_id: 当前控制的机械臂ID（双臂情况下）
    """
    
    def __init__(self) -> None:
        """初始化键盘控制器"""
        # 控制步长
        self._pos_step = 0.0001  # 位置步长：1mm
        self._rot_step = 0.001  # 旋转步长：约0.057度
        
        # 按键状态
        self._is_ctrl_l_pressed = False
        self._is_shift_pressed = False
        
        # 末端执行器偏移
        self._end_pos_offset = np.array([0.0, 0.0, 0.0])
        self._end_rot_offset = np.eye(3)
        
        # 控制标志
        self._reset_flag = False
        self._exit_flag = False
        self._gripper_flag = 0
        self._agent_id = 0  # 0 or 1 for dual-arm
        
        # 键盘监听器
        self.listener = None
    
    def start(self):
        """
        启动键盘监听器
        
        开始监听键盘输入，并显示控制说明。
        """
        self.command_introduction()
        
        if keyboard is None:
            logging.warning("Keyboard control unavailable (headless/no pynput).")
            return

        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.listener.start()

    def stop(self):
        """停止键盘监听器"""
        if self.listener is not None:
            self.listener.stop()
            self.listener = None
    
    def command_introduction(self):
        """打印键盘控制说明"""
        logging.info("=" * 60)
        logging.info("键盘遥操作控制说明:")
        logging.info("-" * 60)
        logging.info("位置控制:")
        logging.info("  <ARROW>              : 沿 x/y 轴移动末端执行器")
        logging.info("  <CTRL + ARROW>       : 沿 z 轴移动末端执行器")
        logging.info("")
        logging.info("姿态控制:")
        logging.info("  <SHIFT + ARROW>      : 绕 x/y 轴旋转末端执行器")
        logging.info("  <CTRL+SHIFT+ARROW>   : 绕 z 轴旋转末端执行器")
        logging.info("")
        logging.info("其他控制:")
        logging.info("  <CAPSLOCK>           : 切换夹爪状态")
        logging.info("  <O> / <P>            : 打开/关闭夹爪")
        logging.info("  <ALT>                : 切换机械臂（双臂）")
        logging.info("  <R>                  : 重置环境")
        logging.info("  <ESC>                : 退出程序")
        logging.info("=" * 60)
    
    def on_press(self, key):
        """
        按键按下事件处理
        
        Args:
            key: 按下的键
        """
        try:
            if key == keyboard.Key.up:
                if self._is_ctrl_l_pressed: 
                    if self._is_shift_pressed:
                        # Ctrl + Shift + Up: 绕 z 轴正向旋转
                        self._end_rot_offset = self._end_rot_offset.dot(
                            T.euler_2_mat(self._rot_step * np.array([0, 0, 1]))
                        )
                    else:
                        # Ctrl + Up: 沿 z 轴正向移动
                        self._end_pos_offset[2] += self._pos_step
                elif self._is_shift_pressed:
                    # Shift + Up: 绕 y 轴负向旋转
                    self._end_rot_offset = self._end_rot_offset.dot(
                        T.euler_2_mat(self._rot_step * np.array([0, -1, 0]))
                    )
                else:
                    # Up: 沿 x 轴正向移动
                    self._end_pos_offset[0] += self._pos_step
            
            elif key == keyboard.Key.down:
                if self._is_ctrl_l_pressed:
                    if self._is_shift_pressed:
                        # Ctrl + Shift + Down: 绕 z 轴负向旋转
                        self._end_rot_offset = self._end_rot_offset.dot(
                            T.euler_2_mat(self._rot_step * np.array([0, 0, -1]))
                        )
                    else:
                        # Ctrl + Down: 沿 z 轴负向移动
                        self._end_pos_offset[2] -= self._pos_step
                elif self._is_shift_pressed:
                    # Shift + Down: 绕 y 轴正向旋转
                    self._end_rot_offset = self._end_rot_offset.dot(
                        T.euler_2_mat(self._rot_step * np.array([0, 1, 0]))
                    )
                else:
                    # Down: 沿 x 轴负向移动
                    self._end_pos_offset[0] -= self._pos_step
            
            elif key == keyboard.Key.left:
                if self._is_shift_pressed:
                    # Shift + Left: 绕 x 轴正向旋转
                    self._end_rot_offset = self._end_rot_offset.dot(
                        T.euler_2_mat(self._rot_step * np.array([1, 0, 0]))
                    )
                else:
                    # Left: 沿 y 轴正向移动
                    self._end_pos_offset[1] += self._pos_step
            
            elif key == keyboard.Key.right:
                if self._is_shift_pressed:
                    # Shift + Right: 绕 x 轴负向旋转
                    self._end_rot_offset = self._end_rot_offset.dot(
                        T.euler_2_mat(self._rot_step * np.array([-1, 0, 0]))
                    )
                else:
                    # Right: 沿 y 轴负向移动
                    self._end_pos_offset[1] -= self._pos_step
            
            elif key == keyboard.Key.ctrl_l:
                self._is_ctrl_l_pressed = True
            
            elif key == keyboard.Key.shift:
                self._is_shift_pressed = True
            
            elif key == keyboard.Key.caps_lock:
                # 切换夹爪状态
                self._gripper_flag = not self._gripper_flag
                logging.info(f"夹爪状态: {'关闭' if self._gripper_flag else '打开'}")
            
            # 添加 O/P 键控制夹爪
            elif hasattr(key, 'char'):
                if key.char == 'o':
                    self._gripper_flag = 0
                    logging.info("夹爪: 打开")
                elif key.char == 'p':
                    self._gripper_flag = 1
                    logging.info("夹爪: 关闭")
        
        except AttributeError:
            pass
    
    def on_release(self, key):
        """
        按键释放事件处理
        
        Args:
            key: 释放的键
            
        Returns:
            False if ESC is pressed (stop listener), None otherwise
        """
        try:
            # 重置末端偏移（只在按键时生效）
            self._end_pos_offset = np.zeros(3)
            self._end_rot_offset = np.eye(3)
            
            if key == keyboard.Key.ctrl_l:
                self._is_ctrl_l_pressed = False
            
            elif key == keyboard.Key.shift:
                self._is_shift_pressed = False
            
            elif key == keyboard.Key.esc:
                # 停止监听器
                logging.info("接收到退出信号 (ESC)")
                self._exit_flag = True
                return False
            
            elif key.char == "r":
                # 重置标志
                logging.info("接收到重置信号 (R)")
                self._reset_flag = True
            
            elif key == keyboard.Key.alt:
                # 切换机械臂ID（用于双臂机器人）
                self._agent_id = 0 if self._agent_id else 1
                logging.info(f"切换到机械臂: arm{self._agent_id}")
        
        except AttributeError:
            pass
    
    def get_end_pos_offset(self) -> np.ndarray:
        """
        获取末端位置偏移
        
        Returns:
            位置偏移向量 (3,)，限制在 [-0.05, 0.05] 范围内
        """
        return np.clip(self._end_pos_offset, -0.05, 0.05)
    
    def get_end_rot_offset(self) -> np.ndarray:
        """
        获取末端旋转偏移
        
        Returns:
            旋转矩阵 (3, 3)
        """
        return self._end_rot_offset
    
    @property
    def reset_flag(self) -> bool:
        """重置标志"""
        return self._reset_flag
    
    @reset_flag.setter
    def reset_flag(self, value: bool):
        self._reset_flag = value
    
    @property
    def exit_flag(self) -> bool:
        """退出标志"""
        return self._exit_flag
    
    @property
    def gripper_flag(self) -> int:
        """夹爪状态标志 (0=打开, 1=关闭)"""
        return self._gripper_flag
    
    @property
    def agent_id(self) -> int:
        """当前控制的机械臂ID"""
        return self._agent_id

