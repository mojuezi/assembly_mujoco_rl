"""
测试 SimulationRobotEnv 与 Controller 和 Task 的集成

作者: Liu Gang
日期: 2025-12-24
"""

import pytest
import numpy as np

from mujoco_env.mujoco_env.core.sim_env import SimulationRobotEnv
from mujoco_env.mujoco_env.tasks.base_task import RobotConfig
from mujoco_env.mujoco_env.tasks import PickCubeTask
from mujoco_env.mujoco_env.robot_config import get_robot


def get_test_xml_path():
    """获取测试用的XML文件路径"""
    # 总是创建一个简单的测试XML（确保与 robot_config 匹配）
    return create_simple_test_xml()


def create_simple_test_xml():
    """创建一个简单的测试XML"""
    import tempfile
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
    
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
    temp_file.write(xml_content)
    temp_file.close()
    return temp_file.name


@pytest.fixture
def robot_config():
    """创建测试用的机器人配置"""
    return RobotConfig(
        name="test_robot",
        robot_type="test",
        dof=6,
        control_freq=20,
        controller_type="joint_position",
        gripper_name=None
    )


@pytest.fixture
def task(robot_config):
    """创建测试用的任务"""
    return PickCubeTask(
        robot_config=robot_config
    )


class TestSimulationRobotEnv:
    """测试 SimulationRobotEnv"""
    
    def test_env_initialization(self, task):
        """测试环境初始化"""
        xml_path = get_test_xml_path()
        
        env = SimulationRobotEnv(
            xml_path=xml_path,
            task=task,
            control_dt=0.02,
            physics_dt=0.002
        )
        
        # 检查环境是否正确初始化
        assert env.model is not None
        assert env.data is not None
        assert env.controller is not None
        assert env.task is not None
        assert env.observation_space is not None
        assert env.action_space is not None
        
        env.close()
    
    def test_different_controllers(self, robot_config):
        """测试不同的控制器类型"""
        xml_path = get_test_xml_path()
        
        controller_types = [
            "joint_position",
            "joint_velocity",
            "joint_torque",
        ]
        
        for controller_type in controller_types:
            # 为每个控制器创建一个 task
            local_robot_config = RobotConfig(
                name=robot_config.name,
                robot_type=robot_config.robot_type,
                dof=robot_config.dof,
                control_freq=robot_config.control_freq,
                controller_type=controller_type,
                gripper_name=None
            )
            task = PickCubeTask(
                robot_config=local_robot_config
            )
            
            env = SimulationRobotEnv(
                xml_path=xml_path,
                task=task
            )
            
            # 检查控制器是否正确创建
            assert env.controller is not None
            assert env.robot_config.controller_type == controller_type
            
            # 测试 reset
            obs, info = env.reset()
            assert isinstance(obs, dict)
            assert "qpos" in obs
            assert "qvel" in obs
            
            # 测试 step
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            assert isinstance(obs, dict)
            assert isinstance(reward, (int, float))
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)
            
            env.close()
    
    def test_task_integration(self, task):
        """测试 Task 集成"""
        xml_path = get_test_xml_path()
        
        env = SimulationRobotEnv(
            xml_path=xml_path,
            task=task
        )
        
        # Reset 应该调用 task.reset()
        obs, info = env.reset()
        
        # 检查观测包含任务相关的内容
        assert "achieved_goal" in obs
        assert "desired_goal" in obs
        
        # Step 应该使用 task 计算奖励
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 奖励应该由 task 计算（不是 0.0）
        # 注意：可能是负数或零，取决于任务
        assert isinstance(reward, (int, float))
        
        env.close()
    
    def test_observation_space_with_task(self, task):
        """测试有 Task 时的观测空间"""
        xml_path = get_test_xml_path()
        
        env = SimulationRobotEnv(
            xml_path=xml_path,
            task=task
        )
        
        # 观测空间应该包含 task 建议的观测
        assert "qpos" in env.observation_space.spaces
        assert "qvel" in env.observation_space.spaces
        assert "tcp_pos" in env.observation_space.spaces
        assert "tcp_quat" in env.observation_space.spaces
        
        # Task 建议的观测也应该在里面
        task_obs_space = task.get_obs_space()
        for key in task_obs_space.spaces.keys():
            assert key in env.observation_space.spaces, f"Missing task observation: {key}"
        
        env.close()
    
    def test_controller_action_conversion(self, robot_config):
        """测试 Controller 的动作转换"""
        xml_path = get_test_xml_path()
        
        task = PickCubeTask(
            robot_config=robot_config
        )
        
        env = SimulationRobotEnv(
            xml_path=xml_path,
            task=task
        )
        
        env.reset()
        
        # 记录初始状态
        initial_qpos = env.data.qpos[:robot_config.dof].copy()
        
        # 应用一个动作
        action = np.ones(robot_config.dof) * 0.1
        
        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step(action)
        
        # 关节位置应该改变
        final_qpos = env.data.qpos[:robot_config.dof].copy()
        assert not np.allclose(initial_qpos, final_qpos), "Joint positions should change"
        
        env.close()
    
    def test_task_with_obs_config(self, robot_config):
        """测试 Task 的观测配置"""
        xml_path = get_test_xml_path()
        
        task = PickCubeTask(
            robot_config=robot_config,
            include_image=False,
            include_depth=False
        )
        
        env = SimulationRobotEnv(
            xml_path=xml_path,
            task=task
        )
        
        obs, info = env.reset()
        
        # PickCubeTask 是 goal-conditioned，会包含 goal 观测
        assert "achieved_goal" in obs
        assert "desired_goal" in obs
        
        # 但不应该包含图像观测
        assert "image" not in obs
        assert "depth" not in obs
        
        # Step 应该计算奖励
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # PickCubeTask 会计算奖励
        assert isinstance(reward, (int, float))
        
        env.close()
    
    def test_controller_reset(self, robot_config):
        """测试 Controller reset"""
        xml_path = get_test_xml_path()
        
        task = PickCubeTask(
            robot_config=robot_config
        )
        
        env = SimulationRobotEnv(
            xml_path=xml_path,
            task=task
        )
        
        # 第一次 reset
        obs1, info1 = env.reset()
        
        # 执行一些步骤
        for _ in range(10):
            action = env.action_space.sample()
            env.step(action)
        
        # 第二次 reset
        obs2, info2 = env.reset()
        
        # Controller 应该被重置
        # （具体行为取决于 controller 实现）
        
        env.close()


class TestControllerIntegration:
    """测试不同控制器的集成"""
    
    @pytest.mark.parametrize("controller_type", [
        "joint_position",
        "joint_velocity",
        "joint_torque",
    ])
    def test_controller_types(self, controller_type):
        """参数化测试不同控制器类型"""
        xml_path = get_test_xml_path()
        
        robot_config = RobotConfig(
            name="test_robot",
            robot_type="test",
            dof=6,
            control_freq=20,
            controller_type=controller_type,
            gripper_name=None
        )
        
        task = PickCubeTask(
            robot_config=robot_config
        )
        
        env = SimulationRobotEnv(
            xml_path=xml_path,
            task=task
        )
        
        # 测试环境可以正常运行
        obs, info = env.reset()
        
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
        
        env.close()


def test_env_with_real_robot_config():
    """测试使用机器人类的配置工厂方法"""
    xml_path = get_test_xml_path()  # 6 DOF 测试 XML
    
    # 使用 Franka Panda 类获取配置，但修改 DOF 以匹配测试 XML
    robot_class = get_robot("franka_panda")
    franka_config = robot_class.get_config(
        controller_type="joint_position",
        control_freq=20
    )
    
    # 创建一个与测试 XML 匹配的配置（6 DOF）
    robot_config = RobotConfig(
        name=franka_config.name,
        robot_type=franka_config.robot_type,
        dof=6,  # 匹配测试 XML
        control_freq=20,
        controller_type="joint_position",
        gripper_name=None
    )
    
    task = PickCubeTask(
        robot_config=robot_config
    )
    
    env = SimulationRobotEnv(
        xml_path=xml_path,
        task=task
    )
    
    obs, info = env.reset()
    assert "qpos" in obs
    assert len(obs["qpos"]) == robot_config.dof  # 应该是 6
    
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    env.close()


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "-s"])
