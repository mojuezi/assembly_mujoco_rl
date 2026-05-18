#!/usr/bin/env python3
"""
Aubo i5 抓取方块演示 (RRT/纯仿真版)

基于 PickCubeTask 任务和 AuboI5Config 配置。
演示机器人在仿真环境中保持初始位姿，去除了键盘控制和示波器。

作者: Liu Gang
日期: 2026-01-03
"""

import sys
import time
import argparse
import numpy as np
import mujoco
from pathlib import Path
from mujoco import viewer as mj_viewer
from dm_robotics.transformations import transformations as tr

# 将项目根目录添加到 sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 使用新的模块结构
from mujoco_env.mujoco_env.envs.env_instance import make_aubo_i5_pick_cube_env
from mujoco_env.mujoco_env.robot_config.aubo_i5 import AuboI5Robot
from mujoco_env.mujoco_env.planners.rrt import RRT
import mujoco_env.mujoco_env.utils.transform as T


class SimWrapper:
    """包装环境以适配 RRT 规划器"""
    def __init__(self, env):
        self.env = env
        self.mj_model = env.model
        # RRT 需要 controller.ik(pos, quat)
        class ControllerWrapper:
            def __init__(self, controller):
                self.controller = controller
            def ik(self, pos, quat):
                return self.controller.solve_ik(pos, quat)
        self.controller = ControllerWrapper(env.controller)

    def set_joint_qpos(self, qpos):
        self.env.set_joint_qpos(qpos)

    def forward(self):
        mujoco.mj_forward(self.env.model, self.env.data)

    def render(self):
        pass

    def get_geom_id(self, names):
        if isinstance(names, list):
            return [self.env.get_geom_id(name) for name in names]
        return self.env.get_geom_id(names)

    def is_contact(self, g1, g2, verbose=0):
        return self.env.is_contact(g1, g2, verbose=bool(verbose))

    def save_state(self):
        return self.env.save_state()

    def load_state(self, state):
        self.env.load_state(state)


def sample_action(action_space, target_pose=None, gripper_action=0.0):
    """
    返回固定的目标动作
    """
    if target_pose is None:
        return np.zeros(action_space.shape)
        
    # 构造动作：[pos(3), quat(4), gripper(1)]
    action = np.concatenate([target_pose, [gripper_action]])
    return action


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Aubo i5 抓取方块演示')
    parser.add_argument('--no-render', action='store_true', help='禁用MuJoCo渲染窗口')
    parser.add_argument('--steps', type=int, default=1000, help='每个 episode 的运行步数')
    parser.add_argument('--episodes', type=int, default=5, help='演示的 episode 数量')
    args = parser.parse_args()

    render = not args.no_render

    print("="*60)
    print("Aubo i5 抓取方块演示 (Cartesian IK + 初始位姿保持)")
    print(f"模式: {'渲染' if render else '无渲染'}, Episode数: {args.episodes}, 每Episode步数: {args.steps}")
    print("="*60)
    
    try:
        # === 获取初始关节配置 ===
        init_qpos = AuboI5Robot.INIT_QPOS
        print(f"初始关节配置 (INIT): {init_qpos}")
        
        # === 创建仿真环境 ===
        print("\n🌍 创建仿真环境...")
        
        env = make_aubo_i5_pick_cube_env(
            scene_name="cubes",  # 使用抓取场景
            include_image=False,    # 可视化不需要图像观测
            image_size=(128, 128),
            include_depth=False,
            use_ft_sensor=False,
            control_dt=0.02,
            physics_dt=0.002,
            max_episode_steps=args.steps,
            render_mode="human" if render else "rgb_array",
        )
        
        # 获取机器人配置信息用于显示
        robot_config = env.task.robot_config
        print(f"✅ 仿真环境创建成功")
        print(f"   - 机器人: {robot_config.name}")
        print(f"   - 夹爪: {robot_config.gripper_name}")
        print(f"   - 控制器: {robot_config.controller_type}")
        print(f"   - 任务: {env.task.name}")
        print(f"   - 场景: {env.task.scene_name}")
        
        # 确定末端执行器site名称
        ee_site_name = "grip_site" if robot_config.gripper_name else "attachment_site"
        print(f"末端执行器Site: {ee_site_name}")
        print(f"   - 动作空间: {env.action_space}")
        
        # === 4. 运行 Episodes ===
        # 获取机器人关节名
        arm_joint_names = ["shoulder_joint", "upperArm_joint", "foreArm_joint", "wrist1_joint", "wrist2_joint", "wrist3_joint"]
        
        for episode in range(args.episodes):
            print(f"\n🚀 开始 Episode {episode + 1}/{args.episodes}...")
            
            # 1. 环境重置
            obs, info = env.reset()
            
            # 2. 强制设置机器人到初始关节配置 (确保稳定性)
            for i, name in enumerate(arm_joint_names):
                try:
                    joint_id = env.model.joint(name).id
                    qpos_adr = env.model.jnt_qposadr[joint_id]
                    env.data.qpos[qpos_adr] = init_qpos[i]
                    qvel_adr = env.model.jnt_dofadr[joint_id]
                    env.data.qvel[qvel_adr] = 0
                except Exception as e:
                    print(f"Warning: Could not set joint {name}: {e}")
            
            # 3. 推进仿真以更新运动学状态
            mujoco.mj_forward(env.model, env.data)
            
            # 4. 重置控制器状态
            env.controller.reset()
            
            # 5. 获取初始末端位姿，作为 Cartesian IK 的目标
            site_id = env.model.site(ee_site_name).id
            initial_pos = env.data.site_xpos[site_id].copy()
            initial_mat = env.data.site_xmat[site_id].reshape(3, 3).copy()
            initial_quat = tr.mat_to_quat(initial_mat)  # [w, x, y, z]
            
            current_target_pose = np.concatenate([initial_pos, initial_quat])
            
            print(f"   Episode {episode+1} 重置完成. 初始 TCP: {initial_pos}")
            
            # --- RRT 任务逻辑 ---
            sim_wrapper = SimWrapper(env)
            cube_names = [
                "cube_red_1", "cube_red_2", "cube_red_3",
                "cube_green_1", "cube", "cube_green_3",
                "cube_blue_1", "cube_blue_2", "cube_blue_3"
            ]
            drop_pos_center = np.array([0.3, 0.3, 0.55])
            safe_height = 0.6
            approach_height = 0.55
            
            def execute_path(path, gripper_val, steps_per_segment=12):
                """执行路径轨迹"""                
                if path is None:
                    return
                # RRT 返回的路径是 [goal, ..., start]，需要反转
                reversed_path = list(reversed(path))
                for i in range(len(reversed_path) - 1):
                    start_wp = np.array(reversed_path[i])
                    end_wp = np.array(reversed_path[i + 1])
                    # 在两个 waypoint 之间线性插值
                    for s in range(steps_per_segment):
                        frac = (s + 1) / steps_per_segment
                        interp_pos = start_wp + frac * (end_wp - start_wp)
                        target_pose = np.concatenate([interp_pos, initial_quat])
                        action = sample_action(env.action_space, target_pose=target_pose, gripper_action=gripper_val)
                        env.step(action)
                        if render:
                            viewer.sync()
                            time.sleep(env.control_dt)
        

            def move_to(pos, gripper_val, steps=20):
                """直线移动到目标位置"""
                curr_pos = env.data.site_xpos[site_id].copy()
                for s in range(steps):
                    frac = (s + 1) / steps
                    interp_pos = curr_pos + frac * (pos - curr_pos)
                    target_pose = np.concatenate([interp_pos, initial_quat])
                    action = sample_action(env.action_space, target_pose=target_pose, gripper_action=gripper_val)
                    env.step(action)
                    if render:
                        viewer.sync()
                        time.sleep(env.control_dt)

            if render:
                with mj_viewer.launch_passive(env.model, env.data) as viewer:
                    for idx, cube_name in enumerate(cube_names):
                        print(f"   正在搬运: {cube_name}")
                        
                        # 计算当前方块的错开放置点 (3x3 网格在篮子内)
                        row = idx // 3
                        col = idx % 3
                        offset_x = (row - 1) * 0.06  # 间距 5cm
                        offset_y = (col - 1) * 0.06
                        current_drop_pos = drop_pos_center + np.array([offset_x, offset_y, 0])
                        
                        # 1. 获取方块位置
                        try:
                            cube_pos = env.data.body(cube_name).xpos.copy()
                        except Exception:
                            print(f"   Warning: Could not find cube {cube_name}, skipping.")
                            continue

                        # 2. 规划到方块上方
                        start_pos = env.data.site_xpos[site_id].copy()
                        goal_pos = cube_pos.copy()
                        goal_pos[2] = approach_height
                        
                        print(f"      规划 RRT 路径到 {cube_name} 上方...")
                        rrt = RRT(
                            start=start_pos,
                            goal=goal_pos,
                            play_area=[0.1, 0.8, -0.5, 0.5, 0.4, 0.8],
                            sim=sim_wrapper,
                            expand_dis=0.03,
                            goal_sample_rate=20,
                            max_iter=500
                        )
                        # 重定义 is_collide 以适应 aubo_i5
                        def aubo_is_collide(sim, node):
                            if sim is None or node is None: return False
                            # 简化碰撞检测：主要检查是否与桌面碰撞，以及是否超过工作空间
                            for x, y, z in zip(node.path_x, node.path_y, node.path_z):
                                if z < 0.42: return True # 桌面高度约 0.411
                            return False
                        rrt.is_collide = aubo_is_collide
                        
                        path = rrt.planning(animation=False)
                        if path:
                            execute_path(path, 0.0)
                        else:
                            print("      无法找到 RRT 路径，尝试直线移动")
                            move_to(goal_pos, 0.0)

                        # 3. 下降并抓取
                        print("      下降抓取...")
                        pick_pos = cube_pos.copy()
                        pick_pos[2] = cube_pos[2] - 0.01 # 稍微深入一点确保抓稳
                        move_to(pick_pos, 0.0)

                        # 闭合夹爪
                        for _ in range(10):
                            action = sample_action(env.action_space, target_pose=np.concatenate([pick_pos, initial_quat]), gripper_action=1.0)
                            env.step(action)
                            if render: viewer.sync(); time.sleep(env.control_dt)
                        
                        # 4. 提升
                        print("      提升...")
                        up_pos = pick_pos.copy()
                        up_pos[2] = approach_height
                        move_to(up_pos, 1.0)

                        # 5. 规划到目标位置上方
                        print(f"      规划 RRT 路径到目标位置上方...")
                        start_pos = env.data.site_xpos[site_id].copy()
                        goal_drop_above = current_drop_pos.copy()
                        goal_drop_above[2] = approach_height
                        
                        rrt = RRT(
                            start=start_pos,
                            goal=goal_drop_above,
                            play_area=[0.1, 0.8, -0.5, 0.5, 0.5, 0.7],
                            sim=sim_wrapper,
                            expand_dis=0.03,
                            goal_sample_rate=20,
                            max_iter=500
                        )
                        rrt.is_collide = aubo_is_collide
                        path = rrt.planning(animation=False)
                        if path:
                            execute_path(path, 1.0)
                        else:
                            move_to(goal_drop_above, 1.0)

                        # 6. 下降并放下
                        print("      放下...")
                        move_to(current_drop_pos, 1.0)
                        
                        # 松开夹爪
                        for _ in range(10):
                            action = sample_action(env.action_space, target_pose=np.concatenate([current_drop_pos, initial_quat]), gripper_action=0.0)
                            env.step(action)
                            if render: viewer.sync(); time.sleep(env.control_dt)
                        
                        # 7. 回到安全高度
                        print("      回到安全高度...")
                        final_up = current_drop_pos.copy()
                        final_up[2] = safe_height
                        move_to(final_up, 0.0)
                        
                        # 8. 重置夹爪关节角度到完全打开状态
                        print("      重置夹爪...")
                        gripper_joint_names = [
                            "robotiq_2f_85_right_driver_joint",
                            "robotiq_2f_85_left_driver_joint",
                            "robotiq_2f_85_right_follower_joint",
                            "robotiq_2f_85_left_follower_joint",
                            "robotiq_2f_85_right_spring_link_joint",
                            "robotiq_2f_85_left_spring_link_joint"
                        ]
                        for joint_name in gripper_joint_names:
                            try:
                                joint_id = env.model.joint(joint_name).id
                                qpos_adr = env.model.jnt_qposadr[joint_id]
                                env.data.qpos[qpos_adr] = 0.0  # 0 = 完全打开
                                qvel_adr = env.model.jnt_dofadr[joint_id]
                                env.data.qvel[qvel_adr] = 0.0
                            except Exception:
                                pass
                        mujoco.mj_forward(env.model, env.data)

            else:
                # 无渲染模式逻辑（类似 render 模式，但无 viewer 相关操作）
                # 为了简洁，这里暂不实现无渲染模式的完整逻辑，或直接报错
                print("本演示目前主要支持渲染模式以观察 RRT 效果。")
            
            print(f"✅ Episode {episode+1} 演示结束")
        
        print("\n🎉 所有演示任务完成")
        env.close()
        
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
