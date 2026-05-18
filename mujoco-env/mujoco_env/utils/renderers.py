"""
MuJoCo渲染器

提供MuJoCo仿真的可视化功能

主要功能：
- 实时渲染（human模式）
- 离线渲染（rgb_array、depth模式）
- 相机视图显示
- 轨迹可视化
- 键盘交互

作者: Liu Gang (原作者)
日期: 2025-12-20
修订: 添加详细注释，更新导入路径
"""

import logging
from queue import Queue
from collections import deque
from typing import Union, List, Optional

import numpy as np
import mujoco
from mujoco import viewer

from . import cv_utils as cv

logging.basicConfig(level=logging.INFO)


# ============================================================================
# MuJoCo渲染器
# ============================================================================

class MjRenderer:
    """
    MuJoCo渲染器
    
    支持多种渲染模式：
    - human: 实时交互式渲染
    - rgb_array: 离线RGB图像渲染
    - depth: 离线深度图渲染
    - None: 无渲染
    
    键盘快捷键：
    - Space: 暂停/继续渲染
    - Esc: 退出
    - Enter: 保存当前相机图像（需要enable_camera_view=True）
    
    使用示例：
        >>> model = mujoco.MjModel.from_xml_path("scene.xml")
        >>> data = mujoco.MjData(model)
        >>> renderer = MjRenderer(model, data, render_mode='human')
        >>> 
        >>> for _ in range(1000):
        ...     mujoco.mj_step(model, data)
        ...     renderer.render()
        >>> 
        >>> renderer.close()
    """
    
    def __init__(
        self,
        mj_model: mujoco.MjModel,
        mj_data: mujoco.MjData,
        render_mode: Union[str, None] = 'human',
        enable_camera_view: bool = False,
        camera_name: str = '0_cam'
    ):
        """
        初始化渲染器
        
        Args:
            mj_model: MuJoCo模型
            mj_data: MuJoCo数据
            render_mode: 渲染模式
                - 'human': 实时交互式渲染
                - 'rgb_array': 离线RGB图像
                - 'depth': 离线深度图
                - None: 无渲染
            enable_camera_view: 是否启用相机视图显示（需要OpenCV）
            camera_name: 相机名称
        """
        self.mj_model = mj_model
        self.mj_data = mj_data
        
        self.render_mode = render_mode
        self.enable_camera_view = enable_camera_view if cv.CV_FLAG else False
        self.camera_name = camera_name
        
        # 键盘控制标志
        self.enable_viewer_keyboard = True  # 启用键盘控制
        self.render_paused = True           # 渲染暂停状态
        self.exit_flag = False              # 退出标志
        
        # 初始化查看器
        self.viewer: Optional[viewer.Handle] = None
        if self.render_mode in ["human", "rgb_array", "depth"]:
            self._init_renderer(mj_model, mj_data)
        elif self.render_mode is None:
            pass
        else:
            raise ValueError(f'{self.render_mode} is not a valid mode.')
        
        # 图像渲染器（用于相机视图）
        self.image_renderer = mujoco.Renderer(self.mj_model)
        self._image = None
        self.image_queue = Queue(3)
        
        # 轨迹可视化
        self.traj = deque(maxlen=200)  # 最多显示200个点
    
    def key_callback(self, keycode: int):
        """
        键盘回调函数
        
        Args:
            keycode: 键盘码
                - 32: Space（暂停/继续）
                - 256: Esc（退出）
                - 257: Enter（保存图像）
        """
        if self.enable_viewer_keyboard:
            if keycode == 32:  # Space
                self.render_paused = not self.render_paused
            elif keycode == 256:  # Esc
                self.exit_flag = True
            elif keycode == 257 and self.enable_camera_view:  # Enter
                image = self.image_queue.get()
                cv.save_image(image)
                logging.info(f"图像已保存到 {cv.CV_CACHE_DIR}")
    
    def _init_renderer(self, mj_model: mujoco.MjModel, mj_data: mujoco.MjData):
        """
        初始化渲染器
        
        Args:
            mj_model: MuJoCo模型
            mj_data: MuJoCo数据
        """
        # 刷新数据
        self.mj_model = mj_model
        self.mj_data = mj_data
        
        # 设置渲染器
        if self.render_mode == "unity":
            # TODO: 支持Unity渲染器
            raise ValueError("Unity渲染器暂不支持")
        elif self.render_mode in ["human", "rgb_array", "depth"]:
            # 关闭旧的查看器
            if isinstance(self.viewer, viewer.Handle):
                self.viewer.close()
            
            # 启动被动查看器（非阻塞）
            self.viewer = viewer.launch_passive(
                mj_model,
                mj_data,
                key_callback=self.key_callback,
                show_left_ui=False,
                show_right_ui=True
            )
            self.set_renderer_config()
            
            # 初始化OpenCV窗口
            if self.enable_camera_view:
                cv.init_cv_window()
        else:
            raise ValueError('Invalid renderer name.')
    
    def render(self) -> Optional[np.ndarray]:
        """
        渲染一帧
        
        Returns:
            如果render_mode为'rgb_array'或'depth'，返回图像数组
            否则返回None
        """
        if self.render_paused and self.render_mode in ["human", "rgb_array", "depth"]:
            # 同步查看器
            if isinstance(self.viewer, viewer.Handle):
                if self.viewer.is_running():
                    self.viewer.sync()
                else:
                    self.close()
            
            # 渲染相机视图
            if self.enable_camera_view:
                enable_depth = True if self.render_mode == 'depth' else False
                image = self.render_pixels_from_camera(self.camera_name, enable_depth=enable_depth)
                self.image_queue.put(image)
                if self.image_queue.full():
                    self.image_queue.get()
                cv.show_image(image)
            
            # 返回图像（如果需要）
            if self.render_mode in ["rgb_array", "depth"]:
                return image
        return None
    
    def close(self):
        """关闭渲染器"""
        if self.enable_camera_view:
            cv.close_cv_window()
        if isinstance(self.viewer, viewer.Handle) and self.viewer.is_running():
            self.viewer.close()
            del self.viewer
            logging.info("查看器已关闭！")
    
    def close_render_window(self):
        """关闭渲染窗口"""
        if self.viewer is not None:
            self.viewer.close()
    
    def set_renderer_config(self):
        """
        设置渲染器配置
        
        配置相机位置、可视化选项等。
        """
        self.viewer.cam.lookat = np.array([0.4, 0, 0.5])
        self.viewer.cam.azimuth += 0.005
        with self.viewer.lock():
            # 根据时间切换接触点显示
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(self.mj_data.time % 2)
    
    def add_visual_point(self, pos: Union[np.ndarray, List[np.ndarray]]):
        """
        添加可视化点（用于轨迹显示）
        
        Args:
            pos: 位置或位置列表，形状为(3,)或[(3,), (3,), ...]
        """
        assert self.render_mode in ["human", "rgb_array", "depth"]
        
        if isinstance(pos, np.ndarray):
            self.traj.append(pos.copy())
            self.viewer.user_scn.ngeom = len(self.traj)
        else:
            for p in pos:
                self.traj.append(p.copy())
            self.viewer.user_scn.ngeom = len(pos)
        
        # 渲染轨迹点
        for i, point in enumerate(self.traj):
            mujoco.mjv_initGeom(
                self.viewer.user_scn.geoms[i],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.008, 0, 0],
                pos=point,
                mat=np.eye(3).flatten(),
                rgba=np.concatenate([np.random.uniform(0, 1, 3), np.array([1])], axis=0)
            )
    
    def visualize_site_frame(self):
        """
        可视化站点坐标系和标签
        
        显示MuJoCo模型中所有site的坐标系和名称。
        """
        assert self.render_mode in ["human", "rgb_array", "depth"]
        self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        self.viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    
    def render_pixels_from_camera(self, cam: str = '0_cam', enable_depth: bool = True) -> np.ndarray:
        """
        从指定相机渲染图像
        
        Args:
            cam: 相机名称或ID
            enable_depth: 是否渲染深度图
            
        Returns:
            图像数组
            - 如果enable_depth=True: 深度图，形状为(H, W)
            - 如果enable_depth=False: RGB图像，形状为(H, W, 3)
        """
        self.image_renderer.update_scene(self.mj_data, camera=cam)
        
        if enable_depth is True:
            # 渲染深度图
            self.image_renderer.enable_depth_rendering()
            org = self.image_renderer.render()
            image = org[:, :]
        else:
            # 渲染RGB图像（BGR -> RGB）
            org = self.image_renderer.render()
            image = org[:, :, ::-1]
        
        self._image = image
        return image


# ============================================================================
# 模块导出
# ============================================================================

__all__ = [
    'MjRenderer',
]

