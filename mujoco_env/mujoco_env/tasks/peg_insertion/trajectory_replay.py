#!/usr/bin/env python3
"""
轨迹回放工具

从 .npz 文件读取录制的关节角度数据，在 MuJoCo 中可视化回放。

使用方法:
    python examples/trajectory_replay.py <trajectory_file.npz>
    python examples/trajectory_replay.py ./trajectories/episode_0000.npz --speed 2.0

日期: 2026-01-12
"""

import sys
import time
import json
import argparse
from pathlib import Path

import numpy as np
import mujoco
from mujoco import viewer as mj_viewer
import pdb


def load_trajectory(filepath: str) -> dict:
    """
    加载轨迹文件
    
    Args:
        filepath: 轨迹文件路径 (.npz 或 .json)
        
    Returns:
        包含 metadata 和 joint_positions 的字典
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"轨迹文件不存在: {filepath}")
    
    if filepath.suffix == ".npz":
        data = np.load(filepath, allow_pickle=True)
        metadata = json.loads(str(data["metadata"]))
        return {
            "metadata": metadata,
            "joint_positions": data["joint_positions"],
            "timestamps": data.get("timestamps", None),
        }
    
    elif filepath.suffix == ".json":
        with open(filepath, "r") as f:
            data = json.load(f)
        return {
            "metadata": data["metadata"],
            "joint_positions": np.array(data["joint_positions"], dtype=np.float32),
            "timestamps": np.array(data.get("timestamps", []), dtype=np.float64),
        }
    
    else:
        raise ValueError(f"不支持的文件格式: {filepath.suffix}")


def replay_trajectory(model, data, trajectory_data: dict, speed: float = 1.0, loop: bool = False):
    """
    回放轨迹
    """
    metadata = trajectory_data["metadata"]
    joint_positions = trajectory_data["joint_positions"]
    
    dof = metadata["dof"]
    control_freq = metadata.get("control_freq", 20.0)
    total_frames = len(joint_positions)
    
    env_info = metadata.get("env_info", {})
    
    print(f"\n📼 轨迹信息:")
    print(f"   - 机器人: {metadata['robot_name']}")
    print(f"   - 自由度: {dof}")
    print(f"   - 总帧数: {total_frames}")
    print(f"   - 控制频率: {control_freq} Hz")
    print(f"   - 时长: {total_frames / control_freq:.2f} 秒")
    print(f"   - 播放速度: {speed}x")
    if env_info:
        print(f"   - 场景: {env_info.get('scene_name', 'N/A')}")
        print(f"   - 夹爪: {env_info.get('gripper_name', 'N/A')}")
    
    frame_delay = 1.0 / (control_freq * speed)
    
    # 从 metadata 获取关节名称（如果未保存，则回退到直接使用前 dof 个 qpos）
    env_joint_names = env_info.get("joint_names", None)
    
    joint_qpos_addrs = []
    if env_joint_names is not None and len(env_joint_names) >= dof:
        # 使用录制时保存的关节名称
        for name in env_joint_names[:dof]:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id >= 0:
                joint_qpos_addrs.append(model.jnt_qposadr[joint_id])
            else:
                joint_qpos_addrs.append(-1)
    else:
        # 回退：直接使用前 dof 个关节
        print(f"   ⚠️  轨迹文件未包含 joint_names，使用前 {dof} 个关节")
        for i in range(dof):
            if i < model.njnt:
                joint_qpos_addrs.append(model.jnt_qposadr[i])
            else:
                joint_qpos_addrs.append(-1)
    
    print(f"\n🎬 开始回放... (关闭窗口退出)")
    
    # 设置初始位姿
    initial_qpos = env_info.get("initial_qpos", None)
    if initial_qpos is not None:
        initial_qpos = np.array(initial_qpos)
        for i, addr in enumerate(joint_qpos_addrs):
            if addr >= 0 and i < len(initial_qpos):
                data.qpos[addr] = initial_qpos[i]
        mujoco.mj_forward(model, data)
    
    with mj_viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for frame_idx, qpos in enumerate(joint_positions):
                if not viewer.is_running():
                    break
                
                for i, addr in enumerate(joint_qpos_addrs):
                    if addr >= 0 and i < len(qpos):
                        data.qpos[addr] = qpos[i]
                
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(frame_delay)
                
                if frame_idx % 50 == 0:
                    print(f"   帧 {frame_idx}/{total_frames} ({100*frame_idx/total_frames:.1f}%)")
            
            print(f"\n✅ 回放完成!")
            
            if not loop:
                print("   关闭窗口退出...")
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.05)
                break


def main():
    parser = argparse.ArgumentParser(description="轨迹回放工具")
    parser.add_argument("trajectory_file", type=str, help="轨迹文件路径 (.npz 或 .json)")
    parser.add_argument("--speed", type=float, default=1.0, help="播放速度倍率 (默认: 1.0)")
    parser.add_argument("--loop", action="store_true", help="循环播放")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("轨迹回放工具")
    print("=" * 60)
    
    print(f"\n📂 加载轨迹文件: {args.trajectory_file}")
    trajectory_data = load_trajectory(args.trajectory_file)
    
    metadata = trajectory_data["metadata"]
    env_info = metadata.get("env_info", {})
    xml_path = env_info.get("xml_path", None)
    xml_path='/home/z/serl/mujoco_env/mujoco_env/tasks/PegInsertion/aubo_i5_assemble_hole.xml'
    
    if xml_path and Path(xml_path).exists():
        print(f"📄 使用录制时的环境: {xml_path}")
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
    else:
        print(f"⚠️  未找到原始环境 XML: {xml_path}")
        print("   请确保轨迹文件包含有效的 xml_path")
        return 1
    
    try:
        replay_trajectory(model, data, trajectory_data, speed=args.speed, loop=args.loop)
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
