"""
环境工厂函数测试

作者: Liu Gang
日期: 2025-12-24
"""

import importlib
import tempfile
from pathlib import Path

import pytest

from mujoco_env.mujoco_env.envs.env_factory import (
    make_env,
    make_sim_env,
    make_real_env,
    _get_scene_xml_path,
)
from mujoco_env.mujoco_env.tasks.base_task import RobotConfig
from mujoco_env.mujoco_env.tasks import PickCubeTask, PCBInsertionTask, PegInsertionTask
from mujoco_env.mujoco_env.core.sim_env import SimulationRobotEnv

env_factory = importlib.import_module("mujoco_env.mujoco_env.envs.env_factory")


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


def build_robot_config(controller_type="joint_position", name="test_robot"):
    """创建用于测试的机器人配置"""
    return RobotConfig(
        name=name,
        robot_type="test",
        dof=6,
        control_freq=20,
        controller_type=controller_type,
        gripper_name=None,
    )


def build_pick_cube_task(
    xml_path,
    controller_type="joint_position",
    include_image=False,
    include_depth=False,
    image_size=(128, 128),
    name="test_robot",
):
    """创建PickCubeTask并绑定测试XML"""
    robot_config = build_robot_config(controller_type=controller_type, name=name)
    task = PickCubeTask(
        robot_config=robot_config,
        include_image=include_image,
        include_depth=include_depth,
        image_size=image_size,
    )
    task.model_path = Path(xml_path)
    return task


# ============================================================================
# 测试辅助函数
# ============================================================================

class TestHelperFunctions:
    """测试辅助函数"""

    def test_get_scene_xml_path_default(self):
        """测试获取默认场景 XML 路径"""
        xml_path = _get_scene_xml_path("aubo_i5")
        assert xml_path.endswith("default.xml")
        assert Path(xml_path).exists()

    def test_get_scene_xml_path_custom(self):
        """测试获取自定义场景 XML 路径"""
        xml_path = _get_scene_xml_path("aubo_i5", scene_name="panda_pick_cube")
        assert "panda_pick_cube.xml" in xml_path
        assert Path(xml_path).exists()

    def test_get_scene_xml_path_without_extension(self):
        """测试场景名称不带 .xml 扩展名"""
        xml_path = _get_scene_xml_path("aubo_i5", scene_name="default")
        assert xml_path.endswith("default.xml")
        assert Path(xml_path).exists()

    def test_get_scene_xml_path_invalid(self):
        """测试无效的场景名称"""
        with pytest.raises(FileNotFoundError):
            _get_scene_xml_path("aubo_i5", scene_name="nonexistent_scene")


# ============================================================================
# 测试 make_sim_env
# ============================================================================

class TestMakeSimEnv:
    """测试 make_sim_env 函数"""

    def test_make_sim_env_basic(self):
        """测试基本的仿真环境创建"""
        xml_path = create_simple_test_xml()
        task = build_pick_cube_task(xml_path)

        env = make_sim_env(task=task)

        assert isinstance(env, SimulationRobotEnv)
        assert env.task is task
        assert env.controller is not None
        assert env.action_space is not None
        assert env.observation_space is not None

        env.close()

    def test_make_sim_env_with_params(self):
        """测试带参数的仿真环境创建"""
        xml_path = create_simple_test_xml()
        task = build_pick_cube_task(xml_path, controller_type="joint_velocity")

        env = make_sim_env(
            task=task,
            max_episode_steps=300,
            control_dt=0.05,
        )

        assert isinstance(env, SimulationRobotEnv)
        assert env.robot_config.controller_type == "joint_velocity"
        assert env.max_episode_steps == 300
        assert env.control_dt == 0.05

        env.close()

    def test_make_sim_env_with_image(self):
        """测试带图像观测的环境创建"""
        xml_path = create_simple_test_xml()
        task = build_pick_cube_task(
            xml_path,
            include_image=True,
            image_size=(84, 84),
        )

        env = make_sim_env(task=task)

        assert isinstance(env, SimulationRobotEnv)
        assert env.obs_config.include_image is True
        assert env.observation_space.spaces["image"].shape == (84, 84, 3)

        env.close()

    def test_make_sim_env_with_depth(self):
        """测试带深度图观测的环境创建"""
        xml_path = create_simple_test_xml()
        task = build_pick_cube_task(
            xml_path,
            include_depth=True,
            image_size=(64, 64),
        )

        env = make_sim_env(task=task)

        assert isinstance(env, SimulationRobotEnv)
        assert env.obs_config.include_depth is True
        assert env.observation_space.spaces["depth"].shape == (64, 64)

        env.close()

    def test_make_sim_env_different_controllers(self):
        """测试不同的控制器类型"""
        xml_path = create_simple_test_xml()
        controller_types = ["joint_position", "joint_velocity", "joint_torque"]

        for ctrl_type in controller_types:
            task = build_pick_cube_task(xml_path, controller_type=ctrl_type)
            env = make_sim_env(task=task)
            assert env.robot_config.controller_type == ctrl_type
            env.close()

    def test_make_sim_env_different_tasks(self):
        """测试不同的任务类型"""
        xml_path = create_simple_test_xml()
        robot_config = build_robot_config()

        task_cases = [
            (PickCubeTask, "pick_cube"),
            (PCBInsertionTask, "PCBInsertion"),
            (PegInsertionTask, "PegInsertion"),
        ]

        for task_cls, expected_name in task_cases:
            task = task_cls(robot_config=robot_config)
            task.model_path = Path(xml_path)
            env = make_sim_env(task=task)
            assert env.task.name == expected_name
            env.close()

    def test_make_sim_env_reset_step(self):
        """测试环境可以正常 reset 和 step"""
        xml_path = create_simple_test_xml()
        task = build_pick_cube_task(xml_path)
        env = make_sim_env(task=task)

        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert isinstance(info, dict)

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(obs, dict)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

        env.close()


# ============================================================================
# 测试 make_real_env（使用 Mock）
# ============================================================================

class DummyRealEnv:
    """用于测试 make_real_env 的简化 env"""

    def __init__(self, robot_interface, task, max_episode_steps=500, **kwargs):
        self.robot_interface = robot_interface
        self.task = task
        self.max_episode_steps = max_episode_steps


class TestMakeRealEnv:
    """测试 make_real_env 函数"""

    def test_make_real_env_basic(self, monkeypatch):
        """测试基本的真机环境创建"""
        xml_path = create_simple_test_xml()
        task = build_pick_cube_task(xml_path, name="aubo")
        dummy_interface = object()

        def fake_get_robot_interface(robot_type, robot_ip, **kwargs):
            assert robot_type == "aubo"
            assert robot_ip == "127.0.0.1"
            return dummy_interface

        monkeypatch.setattr(env_factory, "get_robot_interface", fake_get_robot_interface)
        monkeypatch.setattr(env_factory, "RealRobotEnv", DummyRealEnv)

        env = make_real_env(
            task=task,
            robot_ip="127.0.0.1",
            max_episode_steps=250,
        )

        assert isinstance(env, DummyRealEnv)
        assert env.robot_interface is dummy_interface
        assert env.task is task
        assert env.max_episode_steps == 250


# ============================================================================
# 测试 make_env（通用接口）
# ============================================================================

class TestMakeEnv:
    """测试 make_env 通用接口"""

    def test_make_env_sim_mode(self):
        """测试 mode='sim'"""
        xml_path = create_simple_test_xml()
        task = build_pick_cube_task(xml_path)
        env = make_env(task=task, mode="sim")
        assert isinstance(env, SimulationRobotEnv)
        env.close()

    def test_make_env_simulation_mode(self):
        """测试 mode='simulation'"""
        xml_path = create_simple_test_xml()
        task = build_pick_cube_task(xml_path)
        env = make_env(task=task, mode="simulation")
        assert isinstance(env, SimulationRobotEnv)
        env.close()

    def test_make_env_real_mode(self, monkeypatch):
        """测试 mode='real'"""
        xml_path = create_simple_test_xml()
        task = build_pick_cube_task(xml_path)
        sentinel = object()

        def fake_make_real_env(*args, **kwargs):
            return sentinel

        monkeypatch.setattr(env_factory, "make_real_env", fake_make_real_env)

        env = make_env(
            task=task,
            mode="real",
            robot_ip="127.0.0.1",
        )
        assert env is sentinel

    def test_make_env_robot_mode(self, monkeypatch):
        """测试 mode='robot'"""
        xml_path = create_simple_test_xml()
        task = build_pick_cube_task(xml_path)
        sentinel = object()

        def fake_make_real_env(*args, **kwargs):
            return sentinel

        monkeypatch.setattr(env_factory, "make_real_env", fake_make_real_env)

        env = make_env(
            task=task,
            mode="robot",
            robot_ip="127.0.0.1",
        )
        assert env is sentinel

    def test_make_env_invalid_mode(self):
        """测试无效的 mode"""
        xml_path = create_simple_test_xml()
        task = build_pick_cube_task(xml_path)

        with pytest.raises(ValueError, match="Unknown mode"):
            make_env(
                task=task,
                mode="invalid_mode",
            )

    def test_make_env_real_without_ip(self):
        """测试真机模式缺少 robot_ip"""
        xml_path = create_simple_test_xml()
        task = build_pick_cube_task(xml_path)

        with pytest.raises(ValueError, match="robot_ip is required"):
            make_env(
                task=task,
                mode="real",
            )

    def test_make_env_with_kwargs(self):
        """测试传递额外参数"""
        xml_path = create_simple_test_xml()
        task = build_pick_cube_task(xml_path, controller_type="joint_velocity")

        env = make_env(
            task=task,
            mode="sim",
            max_episode_steps=200,
        )

        assert isinstance(env, SimulationRobotEnv)
        assert env.robot_config.controller_type == "joint_velocity"
        assert env.max_episode_steps == 200

        env.close()

    def test_make_env_render_mode(self):
        """测试渲染模式参数"""
        xml_path = create_simple_test_xml()
        task = build_pick_cube_task(xml_path)

        render_modes = [None, "rgb_array", "human"]
        for render_mode in render_modes:
            env = make_env(
                task=task,
                mode="sim",
                render_mode=render_mode,
            )
            assert isinstance(env, SimulationRobotEnv)
            env.close()
