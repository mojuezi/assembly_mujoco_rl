"""
机器人定义测试

作者: Liu Gang
日期: 2025-12-20
"""

import pytest
import numpy as np
from mujoco_env.mujoco_env.robot_config import (
    FrankaPandaRobot,
    AuboI5Robot,
    get_robot,
    ROBOT_REGISTRY
)


class TestFrankaPandaRobot:
    """测试Franka Panda机器人定义"""
    
    def test_robot_parameters(self):
        """测试机器人基本参数"""
        print("\n✓ Franka Panda机器人参数测试通过")
        assert FrankaPandaRobot.NAME == "franka_panda"
        assert FrankaPandaRobot.DOF == 7
        assert FrankaPandaRobot.JOINT_LIMITS.shape == (7, 2)
        assert FrankaPandaRobot.VELOCITY_LIMITS.shape == (7,)
        assert FrankaPandaRobot.TORQUE_LIMITS.shape == (7,)
        assert FrankaPandaRobot.HOME_QPOS.shape == (7,)
    
    def test_get_config(self):
        """测试获取机器人配置"""
        print("✓ Franka Panda配置获取测试通过")
        config = FrankaPandaRobot.get_config(
            controller_type="joint_position",
            control_freq=20,
            gripper_name="panda_hand"
        )
        assert config.name == "franka_panda"
        assert config.dof == 7
        assert config.controller_type == "joint_position"
        assert config.gripper_name == "panda_hand"
    
    def test_validate_qpos(self):
        """测试关节位置验证"""
        print("✓ Franka Panda关节位置验证测试通过")
        # 有效的home位置
        assert FrankaPandaRobot.validate_qpos(FrankaPandaRobot.HOME_QPOS)
        
        # 超出限位
        invalid_qpos = np.array([10.0, 0, 0, 0, 0, 0, 0])
        assert not FrankaPandaRobot.validate_qpos(invalid_qpos)
        
        # 错误的维度
        invalid_qpos = np.array([0, 0, 0])
        assert not FrankaPandaRobot.validate_qpos(invalid_qpos)
    
    def test_clip_qpos(self):
        """测试关节位置裁剪"""
        print("✓ Franka Panda关节位置裁剪测试通过")
        # 超出上限
        qpos = np.array([10.0, 0, 0, 0, 0, 0, 0])
        clipped = FrankaPandaRobot.clip_qpos(qpos)
        assert clipped[0] == FrankaPandaRobot.JOINT_LIMITS[0, 1]
        
        # 超出下限
        qpos = np.array([-10.0, 0, 0, 0, 0, 0, 0])
        clipped = FrankaPandaRobot.clip_qpos(qpos)
        assert clipped[0] == FrankaPandaRobot.JOINT_LIMITS[0, 0]
    
    def test_get_info(self):
        """测试获取机器人信息"""
        print("✓ Franka Panda信息获取测试通过")
        info = FrankaPandaRobot.get_info()
        assert info["name"] == "franka_panda"
        assert info["dof"] == 7
        assert "manufacturer" in info
        assert "payload" in info
        assert "reach" in info


class TestAuboI5Robot:
    """测试Aubo i5机器人定义"""
    
    def test_robot_parameters(self):
        """测试机器人基本参数"""
        print("\n✓ Aubo i5机器人参数测试通过")
        assert AuboI5Robot.NAME == "aubo_i5"
        assert AuboI5Robot.DOF == 6
        assert AuboI5Robot.JOINT_LIMITS.shape == (6, 2)
        assert AuboI5Robot.VELOCITY_LIMITS.shape == (6,)
        assert AuboI5Robot.TORQUE_LIMITS.shape == (6,)
        assert AuboI5Robot.HOME_QPOS.shape == (6,)
    
    def test_get_config(self):
        """测试获取机器人配置"""
        print("✓ Aubo i5配置获取测试通过")
        config = AuboI5Robot.get_config(
            controller_type="joint_position",
            control_freq=20,
            gripper_name="robotiq_2f85"
        )
        assert config.name == "aubo_i5"
        assert config.dof == 6
    
    def test_validate_qpos(self):
        """测试关节位置验证"""
        print("✓ Aubo i5关节位置验证测试通过")
        # 有效的home位置
        assert AuboI5Robot.validate_qpos(AuboI5Robot.HOME_QPOS)
        
        # 超出限位
        invalid_qpos = np.array([10.0, 0, 0, 0, 0, 0])
        assert not AuboI5Robot.validate_qpos(invalid_qpos)
    
    def test_get_info(self):
        """测试获取机器人信息"""
        print("✓ Aubo i5信息获取测试通过")
        info = AuboI5Robot.get_info()
        assert info["name"] == "aubo_i5"
        assert info["dof"] == 6


class TestRobotRegistry:
    """测试机器人注册表"""
    
    def test_registry_contents(self):
        """测试注册表内容"""
        print("\n✓ 机器人注册表测试通过")
        assert "franka_panda" in ROBOT_REGISTRY
        assert "panda" in ROBOT_REGISTRY
        assert "aubo_i5" in ROBOT_REGISTRY
        assert "aubo" in ROBOT_REGISTRY
    
    def test_get_robot(self):
        """测试获取机器人"""
        print("✓ 获取机器人测试通过")
        franka = get_robot("franka_panda")
        assert franka == FrankaPandaRobot
        
        franka_alias = get_robot("panda")
        assert franka_alias == FrankaPandaRobot
        
        aubo = get_robot("aubo_i5")
        assert aubo == AuboI5Robot
    
    def test_get_invalid_robot(self):
        """测试获取不存在的机器人"""
        print("✓ 无效机器人测试通过")
        with pytest.raises(ValueError):
            get_robot("invalid_robot")


if __name__ == "__main__":
    print("============================================================")
    print("运行机器人定义单元测试")
    print("============================================================")
    
    # Franka Panda测试
    franka_test = TestFrankaPandaRobot()
    franka_test.test_robot_parameters()
    franka_test.test_get_config()
    franka_test.test_validate_qpos()
    franka_test.test_clip_qpos()
    franka_test.test_get_info()
    
    # Aubo i5测试
    aubo_test = TestAuboI5Robot()
    aubo_test.test_robot_parameters()
    aubo_test.test_get_config()
    aubo_test.test_validate_qpos()
    aubo_test.test_get_info()
    
    # 注册表测试
    registry_test = TestRobotRegistry()
    registry_test.test_registry_contents()
    registry_test.test_get_robot()
    registry_test.test_get_invalid_robot()
    
    print("============================================================")
    print("✅ 所有机器人测试通过！")
    print("============================================================")
