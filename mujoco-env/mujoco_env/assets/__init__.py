"""
Assets module for robot models and scenes

提供机器人模型、夹爪、场景等资源的访问接口

作者: Liu Gang
日期: 2025-12-20
"""

from pathlib import Path
from typing import Optional, Dict, List
import os


# 资源根目录
ASSETS_ROOT = Path(__file__).parent


# 资源子目录
MODELS_DIR = ASSETS_ROOT / "models"
MANIPULATORS_DIR = MODELS_DIR / "manipulators"
GRIPPERS_DIR = MODELS_DIR / "grippers"
MOUNTS_DIR = MODELS_DIR / "mounts"
OBJECTS_DIR = ASSETS_ROOT / "objects"
SCENES_DIR = ASSETS_ROOT / "scenes"
TEXTURES_DIR = ASSETS_ROOT / "textures"


# 机器人模型注册表
MANIPULATOR_MODELS = {
    "panda": MANIPULATORS_DIR / "Panda" / "Panda.xml",
    "franka_panda": MANIPULATORS_DIR / "Panda" / "Panda.xml",
    "panda_hand": MANIPULATORS_DIR / "Panda" / "Panda_hand.xml",
    "fr3": MANIPULATORS_DIR / "franka_fr3" / "franka_fr3.xml",
    "franka_fr3": MANIPULATORS_DIR / "franka_fr3" / "franka_fr3.xml",
    "aubo_i5": MANIPULATORS_DIR / "Aubo_i5" / "Aubo_i5.xml",
    "aubo": MANIPULATORS_DIR / "Aubo_i5" / "Aubo_i5.xml",
    "ur5e": MANIPULATORS_DIR / "UR5e" / "UR5e.xml",
    "diana_med": MANIPULATORS_DIR / "DianaMed" / "DianaMed.xml",
}


# 夹爪模型注册表
GRIPPER_MODELS = {
    "panda_hand": GRIPPERS_DIR / "panda_hand" / "panda_hand.xml",
    "robotiq_2f85": GRIPPERS_DIR / "robotiq_2f85" / "2f85.xml",
    "robotiq": GRIPPERS_DIR / "robotiq_gripper" / "robotiq_gripper.xml",
    "rethink": GRIPPERS_DIR / "rethink_gripper" / "rethink_gripper.xml",
    "realsense": GRIPPERS_DIR / "realsense" / "realsense.xml",
    "realsense_usb": GRIPPERS_DIR / "realsense_usb" / "realsense_usb.xml",
    "assemble_axle": GRIPPERS_DIR / "assemble_axle" / "assemble_axle.xml",
}


# 安装座模型注册表
MOUNT_MODELS = {
    "floor_left": MOUNTS_DIR / "floor_left" / "floor_left.xml",
    "floor_right": MOUNTS_DIR / "floor_right" / "floor_right.xml",
    "cylinder": MOUNTS_DIR / "cylinder" / "cylinder.xml",
    "cylinder2": MOUNTS_DIR / "cylinder2" / "cylinder2.xml",
    "top_point": MOUNTS_DIR / "top_point" / "top_point.xml",
    "top_point2": MOUNTS_DIR / "top_point2" / "top_point2.xml",
}


# 场景模型注册表
SCENE_MODELS = {
    "default": SCENES_DIR / "default.xml",
    "grasping": SCENES_DIR / "grasping.xml",
    "assemble": SCENES_DIR / "assemble.xml",
    "assemble_usb": SCENES_DIR / "assemble_usb.xml",
    "panda_pick_cube": SCENES_DIR / "PandaPickCube" / "arena.xml",
}


# 物体模型注册表
OBJECT_MODELS = {
    "table": OBJECTS_DIR / "table" / "table.xml",
    "cube_red": OBJECTS_DIR / "cube" / "red_cube.xml",
    "cube_green": OBJECTS_DIR / "cube" / "green_cube.xml",
    "cube_blue": OBJECTS_DIR / "cube" / "blue_cube.xml",
    "pedestal": OBJECTS_DIR / "pedestal" / "pedestal.xml",
    "cabinet": OBJECTS_DIR / "cabinet" / "cabinet.xml",
    "cupboard": OBJECTS_DIR / "cupboard" / "cupboard.xml",
    "motor": OBJECTS_DIR / "motor" / "motor.xml",
    "carton": OBJECTS_DIR / "carton" / "carton.xml",
    "conveyor_belt": OBJECTS_DIR / "conveyor belt" / "conveyor belt.xml",
    "aruco": OBJECTS_DIR / "aruco" / "aruco.xml",
    "realsense_d435": OBJECTS_DIR / "realsense_d435" / "realsense.xml",
}


def get_model_path(model_type: str, model_name: str) -> Optional[Path]:
    """
    获取模型文件路径
    
    Args:
        model_type: 模型类型 ("manipulator", "gripper", "mount", "scene", "object")
        model_name: 模型名称
        
    Returns:
        Path: 模型文件路径，如果不存在返回None
        
    Example:
        >>> path = get_model_path("manipulator", "panda")
        >>> path = get_model_path("gripper", "robotiq_2f85")
    """
    registries = {
        "manipulator": MANIPULATOR_MODELS,
        "gripper": GRIPPER_MODELS,
        "mount": MOUNT_MODELS,
        "scene": SCENE_MODELS,
        "object": OBJECT_MODELS,
    }
    
    if model_type not in registries:
        return None
    
    registry = registries[model_type]
    path = registry.get(model_name)
    
    if path and path.exists():
        return path
    
    return None


def list_models(model_type: str) -> List[str]:
    """
    列出指定类型的所有可用模型
    
    Args:
        model_type: 模型类型
        
    Returns:
        List[str]: 模型名称列表
        
    Example:
        >>> manipulators = list_models("manipulator")
        >>> print(manipulators)
        ['panda', 'aubo_i5', 'ur5e', ...]
    """
    registries = {
        "manipulator": MANIPULATOR_MODELS,
        "gripper": GRIPPER_MODELS,
        "mount": MOUNT_MODELS,
        "scene": SCENE_MODELS,
        "object": OBJECT_MODELS,
    }
    
    if model_type not in registries:
        return []
    
    return list(registries[model_type].keys())


def get_texture_path(texture_name: str) -> Optional[Path]:
    """
    获取纹理文件路径
    
    Args:
        texture_name: 纹理文件名（带扩展名）
        
    Returns:
        Path: 纹理文件路径，如果不存在返回None
    """
    texture_path = TEXTURES_DIR / texture_name
    if texture_path.exists():
        return texture_path
    return None


def get_asset_info() -> Dict[str, int]:
    """
    获取资源统计信息
    
    Returns:
        Dict: 包含各类资源数量的字典
    """
    return {
        "manipulators": len(MANIPULATOR_MODELS),
        "grippers": len(GRIPPER_MODELS),
        "mounts": len(MOUNT_MODELS),
        "scenes": len(SCENE_MODELS),
        "objects": len(OBJECT_MODELS),
        "textures": len(list(TEXTURES_DIR.glob("*.png"))) if TEXTURES_DIR.exists() else 0,
    }


__all__ = [
    "ASSETS_ROOT",
    "MODELS_DIR",
    "MANIPULATORS_DIR",
    "GRIPPERS_DIR",
    "MOUNTS_DIR",
    "OBJECTS_DIR",
    "SCENES_DIR",
    "TEXTURES_DIR",
    "MANIPULATOR_MODELS",
    "GRIPPER_MODELS",
    "MOUNT_MODELS",
    "SCENE_MODELS",
    "OBJECT_MODELS",
    "get_model_path",
    "list_models",
    "get_texture_path",
    "get_asset_info",
]
