"""
控制器测试

测试所有控制器的基本功能，包括：
- 关节空间控制器：位置、速度、力矩、阻抗、自由驱动
- 任务空间控制器：OSC、笛卡尔阻抗、笛卡尔IK、导纳控制
- 控制器工厂函数和注册表

运行: pytest mujoco_env/tests/test_controllers.py -v
或在conda serl环境下: conda run -n serl pytest tests/test_controllers.py -v

作者: Liu Gang
日期: 2025-12-26
"""

import pytest
import numpy as np
import mujoco
from pathlib import Path
from mujoco_env.mujoco_env.controllers import (
    BaseController,
    JointPositionController,
    JointVelocityController,
    JointTorqueController,
    JointImpedanceController,
    JointFreedriveController,
    OperationalSpaceController,
    CartesianImpedanceController,
    CartesianIKController,
    AdmittanceController,
    get_controller,
    register_controller,
    CONTROLLER_REGISTRY
)


# ============================================================================
# 测试辅助函数
# ============================================================================

def create_test_model(dof: int = 7, add_site: bool = False):
    """
    创建一个简单的MuJoCo测试模型
    
    Args:
        dof: 自由度数量
        add_site: 是否添加末端执行器site（用于任务空间控制器）
    
    Returns:
        model, data: MuJoCo模型和数据
    """
    # 创建分层结构避免单个body超过6个DOF的限制
    bodies_xml = ""
    actuators_xml = ""
    
    for i in range(dof):
        indent = "  " * (i + 1)
        bodies_xml += f'{indent}<body name="link{i}">\n'
        bodies_xml += f'{indent}  <joint name="joint{i}" type="hinge"/>\n'
        bodies_xml += f'{indent}  <geom type="box" size="0.05 0.05 0.05" pos="0 0 {0.1 * (i+1)}"/>\n'
        actuators_xml += f'    <motor name="motor{i}" joint="joint{i}"/>\n'
    
    # 在最后一个link上添加site（用于任务空间控制器）
    if add_site:
        indent = "  " * (dof + 1)
        bodies_xml += f'{indent}<site name="pinch" pos="0 0 0.1" size="0.01"/>\n'
    
    # 闭合所有body标签
    for i in range(dof):
        indent = "  " * (dof - i)
        bodies_xml += f'{indent}</body>\n'
    
    xml_string = f"""
    <mujoco>
        <worldbody>
{bodies_xml}        </worldbody>
        <actuator>
{actuators_xml}        </actuator>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)
    return model, data


# ============================================================================
# 关节空间控制器测试
# ============================================================================

class TestJointPositionController:
    """测试关节位置控制器（PD控制）"""
    
    def test_creation(self):
        """测试控制器创建和默认参数"""
        model, data = create_test_model(dof=7)
        controller = JointPositionController(
            model=model,
            data=data,
            dof=7,
            control_freq=20.0
        )
        assert controller.dof == 7
        assert controller.control_freq == 20.0
        assert len(controller.kp) == 7
        assert len(controller.kd) == 7
        # 验证默认增益（临界阻尼）
        assert np.allclose(controller.kd, 2.0 * np.sqrt(controller.kp))
        print("\n✓ 关节位置控制器创建测试通过")
    
    def test_custom_gains(self):
        """测试自定义PD增益"""
        model, data = create_test_model(dof=7)
        kp = np.array([100.0] * 7)
        kd = np.array([10.0] * 7)
        controller = JointPositionController(
            model=model,
            data=data,
            dof=7,
            kp=kp,
            kd=kd
        )
        np.testing.assert_array_equal(controller.kp, kp)
        np.testing.assert_array_equal(controller.kd, kd)
        print("✓ 自定义增益测试通过")
    
    def test_compute_control(self):
        """测试PD控制计算"""
        model, data = create_test_model(dof=7)
        controller = JointPositionController(
            model=model,
            data=data,
            dof=7,
            kp=np.array([200.0] * 7),
            kd=np.array([20.0] * 7)
        )
        
        # 设置当前状态
        current_state = {
            "qpos": np.zeros(7),
            "qvel": np.zeros(7)
        }
        
        # 目标位置
        target = np.array([0.1] * 7)
        
        # 计算控制输出
        torque = controller.compute_control(target, current_state)
        
        # 验证输出
        assert torque.shape == (7,)
        # 期望力矩 = kp * (target - qpos) = 200 * 0.1 = 20
        expected_torque = 200.0 * 0.1
        np.testing.assert_array_almost_equal(torque, np.array([expected_torque] * 7))
        print("✓ PD控制计算测试通过")
    
    def test_set_gains(self):
        """测试动态设置增益"""
        model, data = create_test_model(dof=7)
        controller = JointPositionController(model=model, data=data, dof=7)
        
        new_kp = np.array([300.0] * 7)
        new_kd = np.array([30.0] * 7)
        controller.set_gains(kp=new_kp, kd=new_kd)
        
        np.testing.assert_array_equal(controller.kp, new_kp)
        np.testing.assert_array_equal(controller.kd, new_kd)
        print("✓ 设置增益测试通过")


class TestJointVelocityController:
    """测试关节速度控制器"""
    
    def test_creation(self):
        """测试控制器创建"""
        model, data = create_test_model(dof=7)
        controller = JointVelocityController(
            model=model,
            data=data,
            dof=7,
            control_freq=20.0
        )
        assert controller.dof == 7
        assert len(controller.kv) == 7
        print("\n✓ 关节速度控制器创建测试通过")
    
    def test_compute_control(self):
        """测试速度控制计算"""
        model, data = create_test_model(dof=7)
        controller = JointVelocityController(
            model=model,
            data=data,
            dof=7,
            kv=np.array([50.0] * 7)
        )
        
        current_state = {
            "qpos": np.zeros(7),
            "qvel": np.zeros(7)
        }
        
        target = np.array([0.5] * 7)  # 目标速度
        torque = controller.compute_control(target, current_state)
        
        assert torque.shape == (7,)
        # 期望力矩 = kv * target_vel = 50 * 0.5 = 25
        expected_torque = 50.0 * 0.5
        np.testing.assert_array_almost_equal(torque, np.array([expected_torque] * 7))
        print("✓ 速度控制计算测试通过")


class TestJointTorqueController:
    """测试关节力矩控制器"""
    
    def test_creation(self):
        """测试控制器创建"""
        model, data = create_test_model(dof=7)
        controller = JointTorqueController(
            model=model,
            data=data,
            dof=7
        )
        assert controller.dof == 7
        assert controller.torque_limits is None
        print("\n✓ 关节力矩控制器创建测试通过")
    
    def test_compute_control(self):
        """测试力矩控制计算（直接输出目标力矩）"""
        model, data = create_test_model(dof=7)
        controller = JointTorqueController(model=model, data=data, dof=7)
        
        target = np.array([10.0] * 7)
        torque = controller.compute_control(target)
        
        np.testing.assert_array_equal(torque, target)
        print("✓ 力矩控制计算测试通过")
    
    def test_torque_limits(self):
        """测试力矩限制功能"""
        model, data = create_test_model(dof=7)
        torque_limits = np.array([50.0] * 7)
        controller = JointTorqueController(
            model=model,
            data=data,
            dof=7,
            torque_limits=torque_limits
        )
        
        # 超出限制的目标力矩
        target = np.array([100.0] * 7)
        torque = controller.compute_control(target)
        
        # 应该被限制到50.0
        np.testing.assert_array_equal(torque, torque_limits)
        print("✓ 力矩限制测试通过")


class TestJointImpedanceController:
    """测试关节阻抗控制器"""
    
    def test_creation(self):
        """测试控制器创建"""
        model, data = create_test_model(dof=7)
        controller = JointImpedanceController(
            model=model,
            data=data,
            dof=7,
            control_freq=20.0
        )
        assert controller.dof == 7
        assert len(controller.K) == 7
        assert len(controller.B) == 7
        assert controller.use_gravity_compensation is True
        print("\n✓ 关节阻抗控制器创建测试通过")
    
    def test_compute_control(self):
        """测试阻抗控制计算"""
        model, data = create_test_model(dof=7)
        controller = JointImpedanceController(
            model=model,
            data=data,
            dof=7,
            kp=np.array([1000.0] * 7),
            kd=np.array([100.0] * 7),
            use_gravity_compensation=False  # 简化测试
        )
        
        # 设置初始状态
        data.qpos[:7] = np.zeros(7)
        data.qvel[:7] = np.zeros(7)
        mujoco.mj_forward(model, data)
        
        current_state = {
            "qpos": np.zeros(7),
            "qvel": np.zeros(7)
        }
        
        target = np.array([0.1] * 7)
        torque = controller.compute_control(target, current_state)
        
        assert torque.shape == (7,)
        # 阻抗控制包含质量矩阵，所以力矩会更大
        assert np.all(np.abs(torque) > 0)
        print("✓ 阻抗控制计算测试通过")
    
    def test_set_gains(self):
        """测试设置阻抗增益"""
        model, data = create_test_model(dof=7)
        controller = JointImpedanceController(model=model, data=data, dof=7)
        
        new_kp = np.array([5000.0] * 7)
        new_kd = np.array([500.0] * 7)
        controller.set_gains(kp=new_kp, kd=new_kd)
        
        np.testing.assert_array_equal(controller.K, new_kp)
        np.testing.assert_array_equal(controller.B, new_kd)
        print("✓ 设置阻抗增益测试通过")


class TestJointFreedriveController:
    """测试关节自由驱动控制器"""
    
    def test_creation(self):
        """测试控制器创建"""
        model, data = create_test_model(dof=7)
        controller = JointFreedriveController(
            model=model,
            data=data,
            dof=7,
            control_freq=20.0
        )
        assert controller.dof == 7
        # 自由驱动模式下，刚度应该为0
        assert np.allclose(controller.K, 0.0)
        # 阻尼应该很小
        assert np.all(controller.B > 0)
        print("\n✓ 关节自由驱动控制器创建测试通过")
    
    def test_compute_control(self):
        """测试自由驱动控制（主要是重力补偿）"""
        model, data = create_test_model(dof=7)
        controller = JointFreedriveController(
            model=model,
            data=data,
            dof=7,
            use_gravity_compensation=True
        )
        
        # 设置初始状态
        data.qpos[:7] = np.array([0.1] * 7)
        data.qvel[:7] = np.zeros(7)
        mujoco.mj_forward(model, data)
        
        current_state = {
            "qpos": np.array([0.1] * 7),
            "qvel": np.zeros(7)
        }
        
        # 在自由驱动模式下，目标位置被忽略
        torque = controller.compute_control(None, current_state)
        
        assert torque.shape == (7,)
        # 应该包含重力补偿
        print("✓ 自由驱动控制计算测试通过")
    
    def test_set_damping(self):
        """测试设置阻尼"""
        model, data = create_test_model(dof=7)
        controller = JointFreedriveController(model=model, data=data, dof=7)
        
        new_damping = 10.0
        controller.set_damping(new_damping)
        
        assert controller.damping == new_damping
        np.testing.assert_array_equal(controller.B, np.array([new_damping] * 7))
        print("✓ 设置阻尼测试通过")


# ============================================================================
# 任务空间控制器测试
# ============================================================================

class TestOperationalSpaceController:
    """测试操作空间控制器（OSC）"""
    
    def test_creation(self):
        """测试控制器创建"""
        model, data = create_test_model(dof=7, add_site=True)
        controller = OperationalSpaceController(
            model=model,
            data=data,
            dof=7,
            site_name="pinch"  # OSC使用site_name而不是ee_site_name
        )
        assert controller.dof == 7
        assert controller.site_name == "pinch"
        print("\n✓ 操作空间控制器创建测试通过")
    
    def test_compute_control(self):
        """
        测试OSC控制计算（基本功能）
        
        注意：OSC的compute_control接受分离的pos和ori参数，不是合并的数组
        """
        model, data = create_test_model(dof=7, add_site=True)
        controller = OperationalSpaceController(
            model=model,
            data=data,
            dof=7,
            site_name="pinch"  # OSC使用site_name而不是ee_site_name
        )
        
        # 设置初始状态
        data.qpos[:7] = np.zeros(7)
        data.qvel[:7] = np.zeros(7)
        mujoco.mj_forward(model, data)
        
        # 目标位姿（分离的位置和姿态）
        target_pos = np.array([0.3, 0.0, 0.5])
        target_quat = np.array([1, 0, 0, 0])  # 单位四元数
        
        # OSC的compute_control接受分离的参数
        torque = controller.compute_control(
            target_pos=target_pos,
            target_ori=target_quat
        )
        
        assert torque.shape == (7,)
        print("✓ OSC控制计算测试通过")


class TestCartesianImpedanceController:
    """测试笛卡尔阻抗控制器"""
    
    def test_creation(self):
        """测试控制器创建"""
        model, data = create_test_model(dof=7, add_site=True)
        controller = CartesianImpedanceController(
            model=model,
            data=data,
            dof=7,
            ee_site_name="pinch"
        )
        assert controller.dof == 7
        assert len(controller.Kc) == 6  # 3个位置 + 3个姿态
        assert len(controller.Bc) == 6
        print("\n✓ 笛卡尔阻抗控制器创建测试通过")
    
    def test_compute_control(self):
        """测试笛卡尔阻抗控制计算"""
        model, data = create_test_model(dof=7, add_site=True)
        controller = CartesianImpedanceController(
            model=model,
            data=data,
            dof=7,
            ee_site_name="pinch"
        )
        
        # 设置初始状态
        data.qpos[:7] = np.zeros(7)
        data.qvel[:7] = np.zeros(7)
        mujoco.mj_forward(model, data)
        
        # 目标位姿
        target_pos = np.array([0.3, 0.0, 0.5])
        target_quat = np.array([1, 0, 0, 0])
        target = np.concatenate([target_pos, target_quat])
        
        current_state = controller.get_state()
        torque = controller.compute_control(target, current_state)
        
        assert torque.shape == (7,)
        print("✓ 笛卡尔阻抗控制计算测试通过")


class TestCartesianIKController:
    """测试笛卡尔IK控制器"""
    
    def test_creation(self):
        """测试控制器创建"""
        model, data = create_test_model(dof=7, add_site=True)
        controller = CartesianIKController(
            model=model,
            data=data,
            dof=7,
            ee_site_name="pinch"
        )
        assert controller.dof == 7
        assert controller.joint_controller is not None
        print("\n✓ 笛卡尔IK控制器创建测试通过")
    
    def test_compute_control(self):
        """测试笛卡尔IK控制计算"""
        model, data = create_test_model(dof=7, add_site=True)
        controller = CartesianIKController(
            model=model,
            data=data,
            dof=7,
            ee_site_name="pinch"
        )
        
        # 设置初始状态
        data.qpos[:7] = np.zeros(7)
        data.qvel[:7] = np.zeros(7)
        mujoco.mj_forward(model, data)
        
        # 目标位姿
        target_pos = np.array([0.3, 0.0, 0.5])
        target_quat = np.array([1, 0, 0, 0])
        target = np.concatenate([target_pos, target_quat])
        
        current_state = controller.get_state()
        torque = controller.compute_control(target, current_state)
        
        assert torque.shape == (7,)
        print("✓ 笛卡尔IK控制计算测试通过")


class TestAdmittanceController:
    """测试导纳控制器"""
    
    def test_creation(self):
        """测试控制器创建"""
        model, data = create_test_model(dof=7, add_site=True)
        controller = AdmittanceController(
            model=model,
            data=data,
            dof=7,
            ee_site_name="pinch"
        )
        assert controller.dof == 7
        assert len(controller.M) == 6
        assert len(controller.D) == 6
        assert len(controller.K) == 6
        print("\n✓ 导纳控制器创建测试通过")
    
    def test_compute_control(self):
        """测试导纳控制计算"""
        model, data = create_test_model(dof=7, add_site=True)
        controller = AdmittanceController(
            model=model,
            data=data,
            dof=7,
            ee_site_name="pinch"
        )
        
        # 设置初始状态
        data.qpos[:7] = np.zeros(7)
        data.qvel[:7] = np.zeros(7)
        mujoco.mj_forward(model, data)
        
        # 目标位姿
        target_pos = np.array([0.3, 0.0, 0.5])
        target_quat = np.array([1, 0, 0, 0])
        target = np.concatenate([target_pos, target_quat])
        
        current_state = controller.get_state()
        torque = controller.compute_control(target, current_state)
        
        assert torque.shape == (7,)
        print("✓ 导纳控制计算测试通过")


# ============================================================================
# 控制器工厂和注册表测试
# ============================================================================

class TestControllerFactory:
    """测试控制器工厂函数"""
    
    def test_registry_contents(self):
        """测试注册表包含所有控制器"""
        # 关节空间控制器
        assert "joint_position" in CONTROLLER_REGISTRY
        assert "joint_velocity" in CONTROLLER_REGISTRY
        assert "joint_torque" in CONTROLLER_REGISTRY
        assert "joint_impedance" in CONTROLLER_REGISTRY
        assert "joint_freedrive" in CONTROLLER_REGISTRY
        
        # 任务空间控制器
        assert "operational_space" in CONTROLLER_REGISTRY
        assert "cartesian_impedance" in CONTROLLER_REGISTRY
        assert "cartesian_ik" in CONTROLLER_REGISTRY
        assert "admittance" in CONTROLLER_REGISTRY
        
        # 别名
        assert "position" in CONTROLLER_REGISTRY
        assert "osc" in CONTROLLER_REGISTRY
        assert "task_space" in CONTROLLER_REGISTRY
        print("\n✓ 控制器注册表测试通过")
    
    def test_get_controller_joint_space(self):
        """测试获取关节空间控制器"""
        model, data = create_test_model(dof=7)
        
        # 测试位置控制器
        controller = get_controller(
            "joint_position",
            model=model,
            data=data,
            dof=7,
            control_freq=20.0
        )
        assert isinstance(controller, JointPositionController)
        assert controller.dof == 7
        
        # 测试速度控制器
        controller = get_controller(
            "joint_velocity",
            model=model,
            data=data,
            dof=7
        )
        assert isinstance(controller, JointVelocityController)
        
        # 测试力矩控制器
        controller = get_controller(
            "joint_torque",
            model=model,
            data=data,
            dof=7
        )
        assert isinstance(controller, JointTorqueController)
        
        print("✓ 获取关节空间控制器测试通过")
    
    def test_get_controller_task_space(self):
        """测试获取任务空间控制器"""
        model, data = create_test_model(dof=7, add_site=True)
        
        # 测试OSC控制器（使用site_name参数）
        controller = get_controller(
            "operational_space",
            model=model,
            data=data,
            dof=7,
            site_name="pinch"  # OSC使用site_name而不是ee_site_name
        )
        assert isinstance(controller, OperationalSpaceController)
        
        # 测试笛卡尔阻抗控制器
        controller = get_controller(
            "cartesian_impedance",
            model=model,
            data=data,
            dof=7,
            ee_site_name="pinch"
        )
        assert isinstance(controller, CartesianImpedanceController)
        
        print("✓ 获取任务空间控制器测试通过")
    
    def test_get_controller_aliases(self):
        """测试控制器别名"""
        model, data = create_test_model(dof=7)
        
        # 测试别名
        controller1 = get_controller("position", model=model, data=data, dof=7)
        controller2 = get_controller("joint_position", model=model, data=data, dof=7)
        assert type(controller1) == type(controller2)
        
        model, data = create_test_model(dof=7, add_site=True)  # OSC需要site
        controller1 = get_controller("osc", model=model, data=data, dof=7, site_name="pinch")
        controller2 = get_controller("operational_space", model=model, data=data, dof=7, site_name="pinch")
        assert type(controller1) == type(controller2)
        
        print("✓ 控制器别名测试通过")
    
    def test_get_invalid_controller(self):
        """测试获取不存在的控制器（应该抛出异常）"""
        model, data = create_test_model(dof=7)
        
        with pytest.raises(ValueError):
            get_controller(
                "invalid_controller",
                model=model,
                data=data,
                dof=7
            )
        print("✓ 无效控制器测试通过")
    
    def test_register_controller(self):
        """测试注册自定义控制器"""
        model, data = create_test_model(dof=7)
        
        # 创建一个简单的自定义控制器
        class CustomController(BaseController):
            def compute_control(self, target, current_state=None):
                return np.zeros(self.dof)
        
        # 注册控制器
        register_controller("custom", CustomController)
        assert "custom" in CONTROLLER_REGISTRY
        
        # 使用注册的控制器
        controller = get_controller("custom", model=model, data=data, dof=7)
        assert isinstance(controller, CustomController)
        
        # 清理：从注册表中移除（可选）
        del CONTROLLER_REGISTRY["custom"]
        print("✓ 注册自定义控制器测试通过")


# ============================================================================
# 主测试运行
# ============================================================================

if __name__ == "__main__":
    """
    直接运行测试脚本时的入口点
    
    按顺序执行所有测试类，用于快速验证控制器的功能
    也可以使用 pytest 运行以获得更详细的测试报告
    """
    print("=" * 60)
    print("运行控制器单元测试")
    print("=" * 60)
    
    # 关节空间控制器测试
    pos_test = TestJointPositionController()
    pos_test.test_creation()
    pos_test.test_custom_gains()
    pos_test.test_compute_control()
    pos_test.test_set_gains()
    
    vel_test = TestJointVelocityController()
    vel_test.test_creation()
    vel_test.test_compute_control()
    
    torque_test = TestJointTorqueController()
    torque_test.test_creation()
    torque_test.test_compute_control()
    torque_test.test_torque_limits()
    
    impedance_test = TestJointImpedanceController()
    impedance_test.test_creation()
    impedance_test.test_compute_control()
    impedance_test.test_set_gains()
    
    freedrive_test = TestJointFreedriveController()
    freedrive_test.test_creation()
    freedrive_test.test_compute_control()
    freedrive_test.test_set_damping()
    
    # 任务空间控制器测试
    osc_test = TestOperationalSpaceController()
    osc_test.test_creation()
    osc_test.test_compute_control()
    
    cart_imp_test = TestCartesianImpedanceController()
    cart_imp_test.test_creation()
    cart_imp_test.test_compute_control()
    
    cart_ik_test = TestCartesianIKController()
    cart_ik_test.test_creation()
    cart_ik_test.test_compute_control()
    
    admittance_test = TestAdmittanceController()
    admittance_test.test_creation()
    admittance_test.test_compute_control()
    
    # 工厂函数测试
    factory_test = TestControllerFactory()
    factory_test.test_registry_contents()
    factory_test.test_get_controller_joint_space()
    factory_test.test_get_controller_task_space()
    factory_test.test_get_controller_aliases()
    factory_test.test_get_invalid_controller()
    factory_test.test_register_controller()
    
    print("=" * 60)
    print("✅ 所有控制器测试通过！")
    print("=" * 60)
