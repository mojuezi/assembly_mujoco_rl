"""
计算机视觉工具

提供图像处理和相机相关的实用函数

主要功能：
- OpenCV窗口管理
- 图像显示和保存
- 相机内参计算

作者: Liu Gang (原作者)
日期: 2025-12-20
修订: 添加详细注释
"""

import logging
import os
import numpy as np
from typing import Tuple

# 尝试导入OpenCV
try:
    import cv2
    CV_FLAG = True
except ImportError:
    logging.warning('无法导入cv2，请安装OpenCV以启用相机查看器: pip install opencv-python')
    CV_FLAG = False

# 图像缓存目录
CV_CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cv_cache')


# ============================================================================
# OpenCV窗口管理
# ============================================================================

def init_cv_window(window_name: str = 'RGB Image'):
    """
    初始化OpenCV窗口
    
    Args:
        window_name: 窗口名称
    """
    if not CV_FLAG:
        logging.error("OpenCV未安装，无法创建窗口")
        return
    
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)


def close_cv_window():
    """关闭所有OpenCV窗口"""
    if not CV_FLAG:
        return
    
    cv2.destroyAllWindows()


def show_image(image: np.ndarray, window_name: str = 'RGB Image'):
    """
    显示图像
    
    Args:
        image: 图像数组，形状为(H, W, 3)或(H, W)
        window_name: 窗口名称
    """
    if not CV_FLAG:
        logging.error("OpenCV未安装，无法显示图像")
        return
    
    cv2.imshow(window_name, image)
    cv2.waitKey(1)  # 等待1ms以刷新窗口


def save_image(image: np.ndarray, filename: str = None):
    """
    保存图像到文件
    
    Args:
        image: 图像数组
        filename: 文件名（可选），如果不提供则自动生成
        
    Returns:
        保存的文件路径
    """
    if not CV_FLAG:
        logging.error("OpenCV未安装，无法保存图像")
        return None
    
    # 创建缓存目录
    if not os.path.exists(CV_CACHE_DIR):
        os.makedirs(CV_CACHE_DIR)
    
    # 自动生成文件名
    if filename is None:
        i = 0
        while os.path.exists(os.path.join(CV_CACHE_DIR, f'cv_cache_image_{i}.png')):
            i += 1
        filename = f'cv_cache_image_{i}.png'
    
    image_path = os.path.join(CV_CACHE_DIR, filename)
    cv2.imwrite(image_path, image)
    logging.info(f"图像已保存到: {image_path}")
    return image_path


# ============================================================================
# 相机内参计算
# ============================================================================

def get_cam_intrinsic(
    fovy: float = 45.0,
    width: int = 320,
    height: int = 240
) -> np.ndarray:
    """
    计算相机内参矩阵
    
    根据视场角和图像尺寸计算相机内参矩阵K。
    
    Args:
        fovy: 垂直视场角（度）
        width: 图像宽度（像素）
        height: 图像高度（像素）
        
    Returns:
        相机内参矩阵K，形状为(3, 3)
        K = [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]
        
    说明:
        - fx, fy: 焦距（像素单位）
        - cx, cy: 主点坐标（通常是图像中心）
        
    示例:
        >>> K = get_cam_intrinsic(fovy=45.0, width=640, height=480)
        >>> print(K)
        [[554.256  0.    320.   ]
         [  0.    554.256 240.   ]
         [  0.      0.      1.   ]]
    """
    # 计算宽高比
    aspect = width * 1.0 / height
    
    # 根据垂直视场角计算水平视场角
    fovx = np.degrees(2 * np.arctan(aspect * np.tan(np.radians(fovy / 2))))
    
    # 主点坐标（图像中心）
    cx = 0.5 * width
    cy = 0.5 * height
    
    # 焦距（像素单位）
    fx = cx / np.tan(fovx * np.pi / 180 * 0.5)
    fy = cy / np.tan(fovy * np.pi / 180 * 0.5)
    
    # 构建内参矩阵
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.float32)
    return K


# ============================================================================
# 模块导出
# ============================================================================

__all__ = [
    'init_cv_window',
    'close_cv_window',
    'show_image',
    'save_image',
    'get_cam_intrinsic',
    'CV_FLAG',
    'CV_CACHE_DIR',
]

