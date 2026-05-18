#!/usr/bin/env python3
"""
演示场景的可视化 - 支持选择所有可用场景

运行方式:
    cd /home/lg/workspace/serl
    python mujoco_env/mujoco_env/assets/scenes/scene_view.py

或:
    python -m mujoco_env.mujoco_env.assets.scenes.scene_view
"""

import sys
from pathlib import Path

# 添加项目路径（如果需要导入其他模块）
project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import mujoco
from mujoco import viewer as mj_viewer


def get_scenes_dir():
    """获取场景目录路径"""
    # 当前文件就在 scenes 目录下，直接返回当前目录
    scenes_dir = Path(__file__).parent
    
    if not scenes_dir.exists() or not scenes_dir.is_dir():
        raise FileNotFoundError(f"场景目录不存在: {scenes_dir}")
    
    return scenes_dir


def list_scenes(scenes_dir: Path):
    """列出所有可用的场景文件"""
    scene_files = sorted(scenes_dir.glob("*.xml"))
    return scene_files


def select_scene(scene_files):
    """让用户选择要加载的场景"""
    if not scene_files:
        print("❌ 未找到任何场景文件")
        return None
    
    print("\n" + "="*60)
    print("可用的场景列表:")
    print("="*60)
    
    for idx, scene_file in enumerate(scene_files, start=1):
        print(f"  [{idx}] {scene_file.name}")
    
    print("="*60)
    
    while True:
        try:
            choice = input(f"\n请选择要加载的场景 (1-{len(scene_files)}, 或 'q' 退出): ").strip()
            
            if choice.lower() in ['q', 'quit', 'exit']:
                print("退出程序")
                return None
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(scene_files):
                selected = scene_files[choice_num - 1]
                print(f"\n✅ 已选择: {selected.name}")
                return selected
            else:
                print(f"❌ 无效选择，请输入 1-{len(scene_files)} 之间的数字")
        except ValueError:
            print("❌ 无效输入，请输入数字或 'q' 退出")
        except (EOFError, KeyboardInterrupt):
            print("\n\n退出程序")
            return None


def main():
    """主函数：加载并可视化场景"""
    print("="*60)
    print("MuJoCo 场景可视化演示")
    print("="*60)
    
    try:
        # 获取场景目录
        scenes_dir = get_scenes_dir()
        print(f"\n✅ 场景目录: {scenes_dir}")
        
        # 列出所有场景
        scene_files = list_scenes(scenes_dir)
        if not scene_files:
            print("❌ 场景目录中没有找到任何 .xml 文件")
            return 1
        
        print(f"\n📁 找到 {len(scene_files)} 个场景文件")
        
        # 让用户选择场景
        selected_scene = select_scene(scene_files)
        if selected_scene is None:
            return 0
        
        xml_path = str(selected_scene.absolute())
        print(f"\n✅ 加载场景: {xml_path}")
        
        # 加载模型
        # MuJoCo 解析相对路径时基于 XML 文件所在目录
        # 如果 XML 中有 include，需要确保路径正确
        # 解决方案：切换到 assets 目录，使用相对路径加载
        import os
        assets_dir = selected_scene.parent  # scenes -> assets (当前文件在 scenes 目录下)
        original_cwd = os.getcwd()
        
        try:
            # 切换到 assets 目录
            os.chdir(str(assets_dir))
            # 使用相对于 assets 目录的路径
            relative_path = selected_scene.relative_to(assets_dir)
            print(f"   工作目录: {assets_dir}")
            print(f"   相对路径: {relative_path}")
            
            model = mujoco.MjModel.from_xml_path(str(relative_path))
        except Exception as e:
            # 如果失败，尝试使用绝对路径
            os.chdir(original_cwd)
            print(f"\n⚠️  使用相对路径加载失败，尝试绝对路径...")
            print(f"   错误: {e}")
            try:
                model = mujoco.MjModel.from_xml_path(xml_path)
            except Exception as e2:
                print(f"\n❌ 加载失败: {e2}")
                raise
        finally:
            # 恢复原始工作目录
            os.chdir(original_cwd)
        
        data = mujoco.MjData(model)
        
        print(f"✅ 模型加载成功")
        print(f"   - 模型名称: {model.names.decode('utf-8') if hasattr(model, 'names') else 'N/A'}")
        print(f"   - 自由度: {model.nv}")
        print(f"   - 物体数量: {model.nbody}")
        print(f"   - 几何体数量: {model.ngeom}")
        print(f"   - 相机数量: {model.ncam}")
        
        # 列出相机
        print(f"\n📷 场景中的相机:")
        for i in range(model.ncam):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            if name:
                cam_pos = model.cam_pos[i]
                print(f"   - {name}: 位置 [{cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f}]")
        
        # 检查关键物体（只显示找到的物体）
        print(f"\n🔍 场景中的关键物体:")
        key_objects = ["table", "usb", "pedestal", "floor"]
        found_objects = []
        for obj_name in key_objects:
            try:
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
                if body_id >= 0:
                    pos = data.xpos[body_id]
                    found_objects.append((obj_name, pos))
            except:
                pass  # 忽略查找失败的物体
        
        if found_objects:
            for obj_name, pos in found_objects:
                print(f"   ✅ {obj_name}: 位置 [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        else:
            print("   (未找到任何关键物体)")
        
        # 重置仿真
        mujoco.mj_resetData(model, data)
        print(f"\n✅ 仿真已重置")
        print(f"   - 时间步长: {model.opt.timestep:.6f} 秒")
        print(f"   - 重力: {model.opt.gravity}")
        
        # 启动交互式查看器
        print(f"\n" + "="*60)
        print("启动交互式查看器...")
        print("="*60)
        print("\n💡 操作提示:")
        print("   - 鼠标左键拖拽: 旋转视角")
        print("   - 鼠标右键拖拽: 平移视角")
        print("   - 滚轮: 缩放")
        print("   - ESC: 退出查看器")
        print("\n正在启动...")
        
        with mj_viewer.launch_passive(model, data) as viewer:
            step_count = 0
            max_steps = 1000000  # 最大步数限制
            
            while viewer.is_running() and step_count < max_steps:
                # 步进仿真
                mujoco.mj_step(model, data)
                
                # 同步查看器（每10步同步一次以提高性能）
                if step_count % 10 == 0:
                    viewer.sync()
                
                step_count += 1
            
            if step_count >= max_steps:
                print(f"\n⚠️  达到最大步数限制 ({max_steps})，退出")
        
        print(f"\n✅ 可视化完成")
        print(f"   - 总仿真步数: {step_count}")
        print(f"   - 仿真时间: {data.time:.2f} 秒")
        
    except FileNotFoundError as e:
        print(f"\n❌ 错误: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

