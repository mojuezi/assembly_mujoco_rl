"""
机器人环境测试（基于新API）

作者: Liu Gang
日期: 2025-12-20
"""

import tempfile
from pathlib import Path

import numpy as np

from mujoco_env.mujoco_env.core.sim_env import SimulationRobotEnv
from mujoco_env.mujoco_env.envs.env_factory import make_env
from mujoco_env.mujoco_env.tasks.base_task import RobotConfig
from mujoco_env.mujoco_env.tasks import PickCubeTask, PCBInsertionTask, PegInsertionTask


# ============================================================================
# Helper Functions
# ============================================================================

def create_simple_test_xml():
    """创建临时测试 XML 文件"""
    xml_content = """
    <mujoco model="test">
      <option timestep="0.002"/>

      <worldbody>
        <light pos="0 0 2" dir="0 0 -1"/>
        <geom type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>

        <body name="base" pos="0 0 0">
          <body name="link1" pos="0 0 0.1">
            <joint name="joint1" type="hinge" axis="0 0 1"/>
            <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.1"/>
            <body name="link2" pos="0 0 0.1">
              <joint name="joint2" type="hinge" axis="0 0 1"/>
              <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.1"/>
              <body name="link3" pos="0 0 0.1">
                <joint name="joint3" type="hinge" axis="0 0 1"/>
                <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.1"/>
                <body name="link4" pos="0 0 0.1">
                  <joint name="joint4" type="hinge" axis="0 0 1"/>
                  <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.1"/>
                  <body name="link5" pos="0 0 0.1">
                    <joint name="joint5" type="hinge" axis="0 0 1"/>
                    <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.1"/>
                    <body name="link6" pos="0 0 0.1">
                      <joint name="joint6" type="hinge" axis="0 0 1"/>
                      <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.1"/>
                      <site name="tcp" pos="0 0 0.05" size="0.01"/>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </worldbody>

      <actuator>
        <motor name="motor1" joint="joint1" gear="1"/>
        <motor name="motor2" joint="joint2" gear="1"/>
        <motor name="motor3" joint="joint3" gear="1"/>
        <motor name="motor4" joint="joint4" gear="1"/>
        <motor name="motor5" joint="joint5" gear="1"/>
        <motor name="motor6" joint="joint6" gear="1"/>
      </actuator>
    </mujoco>
    """

    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False)
    temp_file.write(xml_content)
    temp_file.close()
    return temp_file.name


def build_robot_config(controller_type="joint_position"):
    """创建用于测试的机器人配置"""
    return RobotConfig(
        name="test_robot",
        robot_type="test",
        dof=6,
        control_freq=20,
        controller_type=controller_type,
        gripper_name=None,
    )


def build_task(task_cls, xml_path, controller_type="joint_position", **kwargs):
    """创建任务并绑定测试XML"""
    robot_config = build_robot_config(controller_type=controller_type)
    task = task_cls(robot_config=robot_config, **kwargs)
    task.model_path = Path(xml_path)
    return task


# ============================================================================
# Tests
# ============================================================================

class TestEnvLifecycle:
    """测试环境创建、重置、步进等"""

    def test_creation(self):
        """测试环境创建"""
        xml_path = create_simple_test_xml()
        task = build_task(PickCubeTask, xml_path)
        env = make_env(task=task, mode="sim")

        assert isinstance(env, SimulationRobotEnv)
        assert env.robot_config.controller_type == "joint_position"
        assert env.task.name == "pick_cube"

        env.close()

    def test_reset(self):
        """测试环境重置"""
        xml_path = create_simple_test_xml()
        task = build_task(PickCubeTask, xml_path)
        env = make_env(task=task, mode="sim")

        obs, info = env.reset()

        assert "qpos" in obs
        assert "qvel" in obs
        assert "tcp_pos" in obs
        assert "tcp_quat" in obs
        assert "achieved_goal" in obs
        assert "desired_goal" in obs
        assert "elapsed_steps" in info

        env.close()

    def test_step(self):
        """测试环境步进"""
        xml_path = create_simple_test_xml()
        task = build_task(PickCubeTask, xml_path)
        env = make_env(task=task, mode="sim")

        env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(obs, dict)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert np.isfinite(reward)

        env.close()

    def test_action_space(self):
        """测试动作空间"""
        xml_path = create_simple_test_xml()

        env_pos = make_env(
            task=build_task(PickCubeTask, xml_path, controller_type="joint_position"),
            mode="sim",
        )
        assert env_pos.action_space.shape == (6,)
        env_pos.close()

        env_vel = make_env(
            task=build_task(PickCubeTask, xml_path, controller_type="joint_velocity"),
            mode="sim",
        )
        assert env_vel.action_space.shape == (6,)
        env_vel.close()

    def test_observation_space(self):
        """测试观测空间"""
        xml_path = create_simple_test_xml()

        env_no_img = make_env(
            task=build_task(PickCubeTask, xml_path, include_image=False),
            mode="sim",
        )
        assert "image" not in env_no_img.observation_space.spaces
        env_no_img.close()

        env_with_img = make_env(
            task=build_task(PickCubeTask, xml_path, include_image=True, image_size=(84, 84)),
            mode="sim",
        )
        assert "image" in env_with_img.observation_space.spaces
        assert env_with_img.observation_space.spaces["image"].shape == (84, 84, 3)
        env_with_img.close()

    def test_episode(self):
        """测试完整的episode"""
        xml_path = create_simple_test_xml()
        task = build_task(PickCubeTask, xml_path)
        env = make_env(task=task, mode="sim", max_episode_steps=10)

        env.reset()
        steps = 0

        for _ in range(15):
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            steps += 1

            if terminated or truncated:
                break

        assert steps <= 10
        env.close()

    def test_different_tasks(self):
        """测试不同任务"""
        xml_path = create_simple_test_xml()

        env_pcb = make_env(
            task=build_task(PCBInsertionTask, xml_path),
            mode="sim",
        )
        assert env_pcb.task.name == "PCBInsertion"
        env_pcb.close()

        env_peg = make_env(
            task=build_task(PegInsertionTask, xml_path),
            mode="sim",
        )
        assert env_peg.task.name == "PegInsertion"
        env_peg.close()


if __name__ == "__main__":
    print("=" * 60)
    print("运行机器人环境单元测试")
    print("=" * 60)

    test = TestEnvLifecycle()
    test.test_creation()
    test.test_reset()
    test.test_step()
    test.test_action_space()
    test.test_observation_space()
    test.test_episode()
    test.test_different_tasks()

    print("=" * 60)
    print("\u2705 所有环境测试通过！")
    print("=" * 60)
