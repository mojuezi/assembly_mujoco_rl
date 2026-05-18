"""
XML拼接工具 - 新版

用于动态组合MuJoCo场景、机械臂、夹爪等XML文件。
替代旧版的 XMLSplicer，支持更灵活的XML合并和路径修复。

作者: Liu Gang
日期: 2025-12-26
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..robot_config.base import RobotConfig


class RobotAssembler:
    """机器人组装器，负责组装XML文件"""
    
    def __init__(self, assets_dir: Path):
        self.assets_dir = assets_dir
    
    def build_robot_scene(
        self,
        scene_name: str,
        robot_config: 'RobotConfig',
        output_dir: Path
    ) -> Path:
        """组装机器人场景XML"""
        
        output_path = output_dir / f"{robot_config.name}_{scene_name}.xml"
        
        def get_rel_path(p: Path) -> str:
            try:
                return os.path.relpath(p, output_dir)
            except:
                return str(p)

        root = ET.Element("mujoco", model=f"{robot_config.name}_{scene_name}")
        ET.SubElement(root, "compiler", angle="radian", meshdir=get_rel_path(self.assets_dir / "models"))
        
        # 1. 场景
        scene_path = self.assets_dir / "scenes" / f"{scene_name}.xml"
        if scene_path.exists():
            ET.SubElement(root, "include", file=get_rel_path(scene_path))
        
        # 2. 底座
        if robot_config.mount_name:
            mount_path = self.assets_dir / "models" / "mounts" / robot_config.mount_name / f"{robot_config.mount_name}.xml"
            if mount_path.exists():
                print(f"DEBUG: Processing mount {robot_config.mount_name} with pos {robot_config.mount_pos}")
                
                # 特殊处理：如果有位置偏移，我们需要解析XML并手动合并
                if any(x != 0 for x in robot_config.mount_pos):
                    try:
                        mount_tree = ET.parse(mount_path)
                        mount_root = mount_tree.getroot()
                        
                        # 辅助函数：修正路径
                        def fix_paths(element, base_path, target_path):
                            for key in element.attrib:
                                if key in ["file", "texture"]:
                                    val = element.attrib[key]
                                    if val and not os.path.isabs(val):
                                        # 计算原文件的绝对路径
                                        abs_path = (base_path.parent / val).resolve()
                                        # 计算相对于新文件的相对路径
                                        try:
                                            rel_path = os.path.relpath(abs_path, target_path)
                                            element.attrib[key] = rel_path
                                        except:
                                            pass  # 保持原样
                            
                            for child in element:
                                fix_paths(child, base_path, target_path)

                        # 1. 合并 asset
                        mount_asset = mount_root.find("asset")
                        if mount_asset is not None:
                            # 查找主XML的asset节点
                            main_asset = root.find("asset")
                            if main_asset is None:
                                main_asset = ET.SubElement(root, "asset")
                            
                            # 复制并修正asset内容
                            for child in mount_asset:
                                fix_paths(child, mount_path, output_dir)
                                main_asset.append(child)
                        
                        # 2. 创建包含body
                        # 注意：body必须在worldbody内。检查是否已有worldbody，没有则创建
                        main_worldbody = root.find("worldbody")
                        if main_worldbody is None:
                            main_worldbody = ET.SubElement(root, "worldbody")
                        
                        pos_str = f"{robot_config.mount_pos[0]} {robot_config.mount_pos[1]} {robot_config.mount_pos[2]}"
                        mount_container = ET.SubElement(main_worldbody, "body", name="mount_base", pos=pos_str)
                        
                        # 3. 复制worldbody下的内容
                        mount_worldbody = mount_root.find("worldbody")
                        if mount_worldbody is not None:
                            for child in mount_worldbody:
                                fix_paths(child, mount_path, output_dir)
                                mount_container.append(child)
                        else:
                            # 如果没有worldbody，假设根目录下就是body内容（除了asset等）
                            for child in mount_root:
                                if child.tag not in ["asset", "compiler", "option", "visual", "default", "statistic"]:
                                    mount_container.append(child)
                                    
                    except Exception as e:
                        print(f"WARNING: Failed to parse mount XML for dynamic positioning: {e}")
                        print("Falling back to direct include (position may be incorrect)")
                        ET.SubElement(root, "include", file=get_rel_path(mount_path))
                else:
                    # 如果位置为0，直接include（保持兼容性）
                    ET.SubElement(root, "include", file=get_rel_path(mount_path))
        
        # 3. 机器人本体
        robot_path = self.assets_dir / "models" / "manipulators" / robot_config.robot_type / f"{robot_config.robot_type}.xml"
        if robot_path.exists():
            # 计算机器人总位置：底座位置 + 偏移
            total_pos = [
                robot_config.mount_pos[0] + robot_config.robot_offset[0],
                robot_config.mount_pos[1] + robot_config.robot_offset[1],
                robot_config.mount_pos[2] + robot_config.robot_offset[2]
            ]
            
            # 只有当有偏移时才进行复杂合并，否则直接 include (保持简单)
            # 修改：如果需要挂载夹爪，也必须进行合并，以便找到attachment body
            should_merge = any(x != 0 for x in total_pos) or robot_config.gripper_name
            
            if should_merge:
                print(f"DEBUG: Processing robot {robot_config.robot_type} with pos {total_pos}")
                try:
                    robot_tree = ET.parse(robot_path)
                    robot_root = robot_tree.getroot()
                    
                    # 辅助函数：修正路径 (复用之前的逻辑，或者重新定义)
                    def fix_paths(element, base_path, target_path):
                        for key in element.attrib:
                            if key in ["file", "texture"]:
                                val = element.attrib[key]
                                if val and not os.path.isabs(val):
                                    # 此时 base_path 是源 XML 文件路径
                                    abs_path = (base_path.parent / val).resolve()
                                    try:
                                        rel_path = os.path.relpath(abs_path, target_path)
                                        element.attrib[key] = rel_path
                                    except:
                                        pass
                        for child in element:
                            fix_paths(child, base_path, target_path)

                    # 通用合并函数
                    def merge_section(source_root, target_root, tag_name, fix_path_func=None, source_path=None, target_dir=None):
                        section = source_root.find(tag_name)
                        if section is not None:
                            target_section = target_root.find(tag_name)
                            if target_section is None:
                                target_section = ET.SubElement(target_root, tag_name)
                            for child in section:
                                if fix_path_func and source_path and target_dir:
                                    fix_path_func(child, source_path, target_dir)
                                target_section.append(child)

                    # 合并各个部分
                    merge_section(robot_root, root, "asset", fix_paths, robot_path, output_dir)
                    merge_section(robot_root, root, "default")
                    merge_section(robot_root, root, "actuator")
                    merge_section(robot_root, root, "sensor")
                    merge_section(robot_root, root, "contact")
                    merge_section(robot_root, root, "tendon")
                    merge_section(robot_root, root, "equality")

                    # 5. 创建包含body并复制worldbody内容
                    main_worldbody = root.find("worldbody")
                    if main_worldbody is None:
                        main_worldbody = ET.SubElement(root, "worldbody")
                    
                    pos_str = f"{total_pos[0]} {total_pos[1]} {total_pos[2]}"
                    robot_container = ET.SubElement(main_worldbody, "body", name="robot_base", pos=pos_str)
                    
                    robot_worldbody = robot_root.find("worldbody")
                    if robot_worldbody is not None:
                        for child in robot_worldbody:
                            fix_paths(child, robot_path, output_dir)
                            robot_container.append(child)
                            
                except Exception as e:
                    print(f"WARNING: Failed to parse robot XML: {e}")
                    ET.SubElement(root, "include", file=get_rel_path(robot_path))
            else:
                ET.SubElement(root, "include", file=get_rel_path(robot_path))
        
        # 3.5 Force Sensor (Must be processed BEFORE gripper to establish chain: Robot -> Sensor -> Gripper)
        target_attachment_name = "attachment"
        
        if robot_config.use_ft_sensor and isinstance(robot_config.use_ft_sensor, str):
            sensor_name = robot_config.use_ft_sensor
            # Try to find sensor XML in ftsensor directory first
            sensor_xml_path = self.assets_dir / "models" / "sensors" / "ftsensor" / f"{sensor_name}.xml"
            if not sensor_xml_path.exists():
                # Fallback to general sensors dir
                sensor_xml_path = self.assets_dir / "models" / "sensors" / f"{sensor_name}.xml"
            
            if sensor_xml_path.exists():
                print(f"DEBUG: Processing force sensor {sensor_name} from {sensor_xml_path}")
                
                # Find flange (attachment)
                flange_body = None
                for body in root.iter("body"):
                    if body.get("name") == "attachment":
                        flange_body = body
                        break
                
                if flange_body is not None:
                    try:
                        sensor_tree = ET.parse(sensor_xml_path)
                        sensor_root = sensor_tree.getroot()
                        
                        # Reuse fix_paths helper
                        def fix_paths(element, base_path, target_path):
                            for key in element.attrib:
                                if key in ["file", "texture"]:
                                    val = element.attrib[key]
                                    if val and not os.path.isabs(val):
                                        abs_path = (base_path.parent / val).resolve()
                                        try:
                                            rel_path = os.path.relpath(abs_path, target_path)
                                            element.attrib[key] = rel_path
                                        except:
                                            pass
                            for child in element:
                                fix_paths(child, base_path, target_path)
                        
                        # Helper for merging sections
                        def merge_section(source_root, target_root, tag_name, fix_path_func=None, source_path=None, target_dir=None):
                            section = source_root.find(tag_name)
                            if section is not None:
                                target_section = target_root.find(tag_name)
                                if target_section is None:
                                    target_section = ET.SubElement(target_root, tag_name)
                                for child in section:
                                    if fix_path_func and source_path and target_dir:
                                        fix_path_func(child, source_path, target_dir)
                                    target_section.append(child)

                        # Merge assets, sensors, etc.
                        merge_section(sensor_root, root, "asset", fix_paths, sensor_xml_path, output_dir)
                        merge_section(sensor_root, root, "default")
                        merge_section(sensor_root, root, "actuator")
                        merge_section(sensor_root, root, "sensor")
                        merge_section(sensor_root, root, "contact")
                        
                        # Add sensor body to flange
                        sensor_worldbody = sensor_root.find("worldbody")
                        if sensor_worldbody is not None:
                            for child in sensor_worldbody:
                                fix_paths(child, sensor_xml_path, output_dir)
                                flange_body.append(child)
                                
                                # Look for a mount point for the gripper inside the added sensor body
                                # We look for a body with "mount" in its name
                                if child.tag == "body":
                                    # Check the body itself or its children
                                    if "mount" in (child.get("name") or ""):
                                        target_attachment_name = child.get("name")
                                    else:
                                        for subchild in child.iter("body"):
                                            if "mount" in (subchild.get("name") or ""):
                                                target_attachment_name = subchild.get("name")
                        
                        print(f"DEBUG: Sensor attached. New target attachment body: {target_attachment_name}")
                        
                    except Exception as e:
                        print(f"WARNING: Failed to parse sensor XML: {e}")
                else:
                    print("WARNING: Could not find 'attachment' body for sensor")

        # 4. 夹爪
        if robot_config.gripper_name:
            gripper_path = self.assets_dir / "models" / "grippers" / robot_config.gripper_name / f"{robot_config.gripper_name}.xml"
            attached_path = gripper_path.parent / f"{robot_config.gripper_name}_attached.xml"
            final_path = attached_path if attached_path.exists() else gripper_path
            
            print(f"DEBUG: Checking gripper path: {final_path}, exists: {final_path.exists()}")
            
            if final_path.exists():
                # 尝试找到机器人的 attachment body
                # 使用 target_attachment_name (默认为 "attachment" 或 传感器末端)
                attachment_body = None
                for body in root.iter("body"):
                    if body.get("name") == target_attachment_name:
                        attachment_body = body
                        break
                
                if attachment_body is not None:
                    print(f"DEBUG: Attaching gripper to body '{attachment_body.get('name')}'")
                    try:
                        gripper_tree = ET.parse(final_path)
                        gripper_root = gripper_tree.getroot()
                        
                        # 复用 fix_paths 函数
                        def fix_paths(element, base_path, target_path):
                            for key in element.attrib:
                                if key in ["file", "texture"]:
                                    val = element.attrib[key]
                                    if val and not os.path.isabs(val):
                                        abs_path = (base_path.parent / val).resolve()
                                        try:
                                            rel_path = os.path.relpath(abs_path, target_path)
                                            element.attrib[key] = rel_path
                                        except:
                                            pass
                            for child in element:
                                fix_paths(child, base_path, target_path)

                        # 通用合并函数 (需要在gripper作用域内也定义或移到外部)
                        def merge_section(source_root, target_root, tag_name, fix_path_func=None, source_path=None, target_dir=None):
                            section = source_root.find(tag_name)
                            if section is not None:
                                target_section = target_root.find(tag_name)
                                if target_section is None:
                                    target_section = ET.SubElement(target_root, tag_name)
                                for child in section:
                                    # 特殊处理：跳过 equality 中的 connect 约束
                                    # 因为 connect 约束使用全局坐标，当夹爪被移动时会失效导致散架
                                    if tag_name == "equality" and child.tag == "connect":
                                        print("WARNING: Skipping 'connect' equality constraint to avoid gripper breakage")
                                        continue
                                    
                                    if fix_path_func and source_path and target_dir:
                                        fix_path_func(child, source_path, target_dir)
                                    target_section.append(child)

                        # 合并各个部分
                        merge_section(gripper_root, root, "asset", fix_paths, final_path, output_dir)
                        merge_section(gripper_root, root, "default")
                        merge_section(gripper_root, root, "actuator")
                        merge_section(gripper_root, root, "sensor")
                        merge_section(gripper_root, root, "contact")
                        merge_section(gripper_root, root, "tendon")
                        merge_section(gripper_root, root, "equality")

                        # 5. 将 gripper body 内容添加到 attachment body
                        gripper_worldbody = gripper_root.find("worldbody")
                        if gripper_worldbody is not None:
                            for child in gripper_worldbody:
                                fix_paths(child, final_path, output_dir)
                                attachment_body.append(child)
                        else:
                            # 尝试直接添加根目录下的非配置元素
                            for child in gripper_root:
                                if child.tag not in ["asset", "compiler", "option", "visual", "default", "statistic", "actuator", "sensor", "mujoco"]:
                                    fix_paths(child, final_path, output_dir)
                                    attachment_body.append(child)
                                    
                    except Exception as e:
                        print(f"WARNING: Failed to parse gripper XML: {e}")
                        ET.SubElement(attachment_body, "include", file=get_rel_path(final_path))
                else:
                    print("WARNING: Could not find 'attachment' body, adding gripper to worldbody")
                    ET.SubElement(root, "include", file=get_rel_path(final_path))
        
        # 5. 力传感器
        if robot_config.use_ft_sensor:
            ft_path = self.assets_dir / "models" / "sensors" / "ft_sensor.xml"
            if ft_path.exists():
                ET.SubElement(root, "include", file=get_rel_path(ft_path))
        
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(output_path, encoding="utf-8", xml_declaration=True)
        return output_path
