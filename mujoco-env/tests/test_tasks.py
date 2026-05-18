"""
任务测试

作者: Liu Gang
日期: 2025-12-20
"""

import pytest
import numpy as np

from mujoco_env.mujoco_env.tasks import (
    PCBInsertionTask,
    PegInsertionTask,
    get_task,
    TASK_REGISTRY,
)
from mujoco_env.mujoco_env.tasks.base_task import RobotConfig


@pytest.fixture
def robot_config():
    """创建用于测试的机器人配置"""
    return RobotConfig(
        name="test_robot",
        robot_type="test",
        dof=6,
        control_freq=20,
        controller_type="joint_position",
        gripper_name=None,
    )


class TestPCBInsertionTask:
    """测试PCB插拔任务"""

    def test_creation(self, robot_config):
        """测试任务创建"""
        print("\n✓ PCB插拔任务创建测试通过")
        task = PCBInsertionTask(robot_config=robot_config, max_episode_steps=500)
        assert task.name == "PCBInsertion"
        assert task.position_tolerance == 0.005

    def test_sample_goal(self, robot_config):
        """测试目标采样"""
        print("✓ PCB目标采样测试通过")
        task = PCBInsertionTask(robot_config=robot_config)
        goal = task.sample_goal()

        # 检查维度
        assert goal.shape == (7,), f"Goal shape should be (7,), got {goal.shape}"

        # 检查位置在范围内
        assert 0.3 <= goal[0] <= 0.5, "x position out of range"
        assert -0.2 <= goal[1] <= 0.2, "y position out of range"
        assert 0.1 <= goal[2] <= 0.3, "z position out of range"

        # 检查四元数归一化
        quat = goal[3:7]
        quat_norm = np.linalg.norm(quat)
        assert abs(quat_norm - 1.0) < 1e-6, "Quaternion not normalized"

    def test_compute_reward(self, robot_config):
        """测试奖励计算"""
        print("✓ PCB奖励计算测试通过")
        task = PCBInsertionTask(robot_config=robot_config)

        # 完全对齐的情况
        achieved = np.array([0.4, 0.0, 0.2, 0.0, 1.0, 0.0, 0.0])
        desired = np.array([0.4, 0.0, 0.2, 0.0, 1.0, 0.0, 0.0])
        reward = task.compute_reward(achieved, desired, {})

        # 完全对齐应该有很高的奖励（包含success bonus）
        assert reward > 5.0, f"Perfect alignment should give high reward, got {reward}"

        # 有误差的情况
        achieved_with_error = np.array([0.45, 0.05, 0.2, 0.0, 1.0, 0.0, 0.0])
        reward_with_error = task.compute_reward(achieved_with_error, desired, {})

        # 有误差的奖励应该更低
        assert reward_with_error < reward, "Reward should be lower with error"

    def test_is_success(self, robot_config):
        """测试成功判断"""
        print("✓ PCB成功判断测试通过")
        task = PCBInsertionTask(robot_config=robot_config)

        # 成功的情况
        achieved = np.array([0.4, 0.0, 0.2, 0.0, 1.0, 0.0, 0.0])
        desired = np.array([0.4001, 0.0001, 0.2001, 0.0, 1.0, 0.0, 0.0])
        assert task.is_success_fn(achieved, desired), "Should be successful"

        # 失败的情况（位置误差大）
        achieved_fail = np.array([0.5, 0.1, 0.2, 0.0, 1.0, 0.0, 0.0])
        assert not task.is_success_fn(achieved_fail, desired), "Should fail due to position error"

    def test_get_achieved_goal(self, robot_config):
        """测试获取achieved goal"""
        print("✓ PCB获取achieved goal测试通过")
        task = PCBInsertionTask(robot_config=robot_config)

        # 使用PCB观测
        obs = {
            "pcb_pos": np.array([0.4, 0.0, 0.2]),
            "pcb_quat": np.array([0.0, 1.0, 0.0, 0.0]),
        }
        achieved = task.get_achieved_goal(obs)
        assert achieved.shape == (7,)
        np.testing.assert_array_equal(achieved[:3], obs["pcb_pos"])

        # 使用TCP观测
        obs_tcp = {
            "tcp_pos": np.array([0.5, 0.1, 0.3]),
            "tcp_quat": np.array([1.0, 0.0, 0.0, 0.0]),
        }
        achieved_tcp = task.get_achieved_goal(obs_tcp)
        assert achieved_tcp.shape == (7,)


class TestPegInsertionTask:
    """测试轴孔装配任务"""

    def test_creation(self, robot_config):
        """测试任务创建"""
        print("\n✓ 轴孔装配任务创建测试通过")
        task = PegInsertionTask(robot_config=robot_config, max_episode_steps=500)
        assert task.name == "PegInsertion"
        assert task.peg_radius == 0.01
        assert task.hole_radius == 0.011
        assert abs(task.clearance - 0.001) < 1e-6  # 浮点数比较

    def test_sample_goal(self, robot_config):
        """测试目标采样"""
        print("✓ 轴孔目标采样测试通过")
        task = PegInsertionTask(robot_config=robot_config)
        goal = task.sample_goal()

        # 检查维度（包含深度）
        assert goal.shape == (8,), f"Goal shape should be (8,), got {goal.shape}"

        # 检查位置在范围内
        assert 0.3 <= goal[0] <= 0.6, "x position out of range"
        assert -0.3 <= goal[1] <= 0.3, "y position out of range"
        assert 0.05 <= goal[2] <= 0.2, "z position out of range"

        # 检查插入深度
        assert goal[7] == task.insertion_depth

    def test_compute_reward(self, robot_config):
        """测试奖励计算"""
        print("✓ 轴孔奖励计算测试通过")
        task = PegInsertionTask(robot_config=robot_config)

        # 完全对齐且插入的情况
        achieved = np.array([0.4, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0, 0.05])
        desired = np.array([0.4, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0, 0.05])
        reward = task.compute_reward(achieved, desired, {})

        # 完全成功应该有很高的奖励
        assert reward > 10.0, f"Perfect insertion should give high reward, got {reward}"

        # 未插入的情况
        achieved_not_inserted = np.array([0.4, 0.0, 0.15, 1.0, 0.0, 0.0, 0.0, 0.0])
        reward_not_inserted = task.compute_reward(achieved_not_inserted, desired, {})

        # 未插入的奖励应该更低
        assert reward_not_inserted < reward, "Reward should be lower without insertion"

    def test_is_success(self, robot_config):
        """测试成功判断"""
        print("✓ 轴孔成功判断测试通过")
        task = PegInsertionTask(robot_config=robot_config)

        # 成功的情况
        achieved = np.array([0.4, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0, 0.05])
        desired = np.array([0.4001, 0.0001, 0.1001, 1.0, 0.0, 0.0, 0.0, 0.05])
        assert task.is_success_fn(achieved, desired), "Should be successful"

        # 失败的情况（未插入）
        achieved_fail = np.array([0.4, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0, 0.0])
        assert not task.is_success_fn(achieved_fail, desired), "Should fail without insertion"


class TestTaskRegistry:
    """测试任务注册表"""

    def test_registry_contents(self):
        """测试注册表内容"""
        print("\n✓ 任务注册表测试通过")
        assert "pick_cube" in TASK_REGISTRY
        assert "pcb_insertion" in TASK_REGISTRY
        assert "pcb" in TASK_REGISTRY
        assert "peg_insertion" in TASK_REGISTRY
        assert "peg" in TASK_REGISTRY
        assert "peg_in_hole" in TASK_REGISTRY

    def test_get_task(self, robot_config):
        """测试获取任务"""
        print("✓ 获取任务测试通过")

        # 获取PCB任务
        pcb_task = get_task("pcb_insertion", robot_config=robot_config)
        assert isinstance(pcb_task, PCBInsertionTask)

        # 使用别名
        pcb_task_alias = get_task("pcb", robot_config=robot_config)
        assert isinstance(pcb_task_alias, PCBInsertionTask)

        # 获取轴孔任务
        peg_task = get_task("peg_insertion", robot_config=robot_config, max_episode_steps=1000)
        assert isinstance(peg_task, PegInsertionTask)

    def test_get_invalid_task(self):
        """测试获取不存在的任务"""
        print("✓ 无效任务测试通过")
        with pytest.raises(ValueError):
            get_task("invalid_task")


if __name__ == "__main__":
    print("============================================================")
    print("运行任务单元测试")
    print("============================================================")

    robot_config_value = RobotConfig(
        name="test_robot",
        robot_type="test",
        dof=6,
        control_freq=20,
        controller_type="joint_position",
        gripper_name=None,
    )

    pcb_test = TestPCBInsertionTask()
    pcb_test.test_creation(robot_config_value)
    pcb_test.test_sample_goal(robot_config_value)
    pcb_test.test_compute_reward(robot_config_value)
    pcb_test.test_is_success(robot_config_value)
    pcb_test.test_get_achieved_goal(robot_config_value)

    peg_test = TestPegInsertionTask()
    peg_test.test_creation(robot_config_value)
    peg_test.test_sample_goal(robot_config_value)
    peg_test.test_compute_reward(robot_config_value)
    peg_test.test_is_success(robot_config_value)

    registry_test = TestTaskRegistry()
    registry_test.test_registry_contents()
    registry_test.test_get_task(robot_config_value)
    registry_test.test_get_invalid_task()

    print("============================================================")
    print("✅ 所有任务测试通过！")
    print("============================================================")
