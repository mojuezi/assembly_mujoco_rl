# Assets 资源目录

| 修订日期   | 修订版本 | 修订内容   | 修订人 |
| ---------- | -------- | ---------- | ------ |
| 2025.12.20 | V0.1     | 初始化文档 | 刘刚   |

本目录包含MuJoCo仿真所需的所有3D模型、物体、纹理资源 和 场景。

---

## 📁 目录结构

```shell
assets/
├── models/                  # 模型文件
│   ├── manipulators/        # 机械臂模型
│   │   ├── Panda/           # Franka Panda (7 DOF)
│   │   ├── franka_fr3/      # Franka FR3 (7 DOF)
│   │   ├── Aubo_i5/         # Aubo i5 (6 DOF)
│   │   ├── UR5e/            # Universal Robots UR5e (6 DOF)
│   │   └── DianaMed/        # DianaMed (7 DOF)
│   │
│   ├── grippers/            # 夹爪/末端执行器
│   │   ├── panda_hand/      # Panda Hand
│   │   ├── robotiq_2f85/    # Robotiq 2F-85
│   │   ├── robotiq_gripper/ # Robotiq通用夹爪
│   │   ├── rethink_gripper/ # Rethink夹爪
│   │   ├── realsense/       # RealSense相机
│   │   ├── realsense_usb/   # RealSense相机（带USB）
│   │   └── assemble_axle/   # 装配轴工具
│   │
│   └── mounts/              # 安装座
│       ├── floor_left/      # 左侧地面安装
│       ├── floor_right/     # 右侧地面安装
│       ├── cylinder/        # 圆柱安装座
│       ├── cylinder2/       # 圆柱安装座2
│       ├── top_point/       # 顶部点安装
│       └── top_point2/      # 顶部点安装2
│
├── objects/                 # 场景物体
│   ├── table/               # 桌子
│   ├── cube/                # 方块（红/绿/蓝）
│   ├── pedestal/            # 基座
│   ├── cabinet/             # 柜子
│   ├── cupboard/            # 橱柜
│   ├── motor/               # 电机
│   ├── carton/              # 纸箱
│   ├── conveyor belt/       # 传送带
│   ├── aruco/               # ArUco标记
│   ├── fruits/              # 水果
│   ├── usb/                 # USB接口
│   └── realsense_d435/      # RealSense D435相机
│
├── scenes/                  # 预定义场景
│   ├── default.xml          # 默认场景
│   ├── grasping.xml         # 抓取场景
│   ├── assemble.xml         # 装配场景
│   ├── assemble_usb.xml     # USB装配场景
│   └── PandaPickCube/       # Panda抓取方块场景
│
└── textures/                # 纹理贴图
    ├── aruco.png            # ArUco标记纹理
    ├── block.png            # 方块纹理
    ├── chessboard.png       # 棋盘纹理
    ├── concrete.png         # 混凝土纹理
    ├── wood.png             # 木纹理
    └── ...                  # 其他纹理
```

---

## 🔗 目录关系说明

### Assets目录内部关系

`assets/` 目录采用分层组织结构，各子目录之间存在引用和依赖关系：

```
┌─────────────────────────────────────────────────────┐
│                    scenes/                          │
│           (预定义完整场景XML)                         │
│  ┌─────────────────────────────────────────┐        │
│  │  • default.xml                          │        │
│  │  • grasping.xml                         │        │
│  │  • assemble.xml                         │        │
│  └─────────────────────────────────────────┘        │
│         │              │              │             │
│         ▼              ▼              ▼             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐        │
│  │ models/  │   │ objects/ │   │textures/ │        │
│  │          │   │          │   │          │        │
│  │ 引用机械臂 │   │ 引用物体  │   │ 引用纹理  │        │
│  │ 引用夹爪  │   │ 引用环境  │   │          │        │
│  └──────────┘   └────┬─────┘   └─────▲────┘        │
│       │                │               │             │
│       └────────────────┴───────────────┘             │
│            models和objects也可能引用textures          │
└─────────────────────────────────────────────────────┘
```

#### 1. **models/** - 基础模型资源

包含机器人系统的核心组件，采用模块化设计：

- **manipulators/** - 机械臂本体模型（Panda, Aubo, UR5e等）
- **grippers/** - 末端执行器模型（夹爪、相机等）
- **mounts/** - 安装座模型（地面、顶部、圆柱等）

特点：
- 独立的XML文件，可单独加载或组合使用
- 包含网格文件（STL/OBJ/DAE）和材质定义
- 可能引用 `textures/` 中的纹理文件

#### 2. **objects/** - 场景物体资源

包含与机器人交互的环境物体：

- 家具：table（桌子）、cabinet（柜子）、cupboard（橱柜）
- 操作对象：cube（方块）、motor（电机）、usb（USB接口）
- 环境设施：conveyor belt（传送带）、pedestal（基座）
- 标记/传感器：aruco（标记）、realsense_d435（相机）

特点：
- 独立的XML文件定义
- 包含物理属性（质量、摩擦力等）
- 可能引用 `textures/` 中的纹理

#### 3. **textures/** - 纹理资源库

包含所有材质纹理图片：
- 环境纹理：concrete.png、wood.png、carpet-black.png
- 功能纹理：chessboard.png（标定）、aruco.png（标记）
- 装饰纹理：light_wood.png、white-marble.png

特点：
- PNG格式图片文件
- 被 `models/` 和 `objects/` 中的XML引用
- 通过MuJoCo的 `<texture>` 标签加载

#### 4. **scenes/** - 预定义完整场景

集成上述所有资源，创建可直接使用的完整场景：

```xml
<!-- scenes/grasping.xml 示例 -->
<mujoco model="grasping_scene">
  <asset>
    <!-- 引用纹理 -->
    <texture name="wood" file="../textures/wood.png"/>
    <!-- 引用物体网格 -->
    <mesh name="table" file="../objects/table/table.STL"/>
  </asset>
  
  <worldbody>
    <!-- 引用物体 -->
    <body name="table">
      <include file="../objects/table/table.xml"/>
    </body>
  </worldbody>
</mujoco>
```

特点：
- 通过 `<include>` 标签引用其他目录的资源
- 定义完整的场景配置（光照、相机、地面等）
- 可以选择性组合 `models/` 中的机械臂和夹爪
- 添加 `objects/` 中的交互物体

### Assets与Robots目录的关系

`assets/` 和 `robots/` 目录在架构上是**数据层**和**逻辑层**的分离：

```
┌──────────────────────────────────────────────────────┐
│                   应用层 (User Code)                  │
│  import mujoco                                        │
│  from mujoco_env.robots import get_robot             │
└──────────────┬───────────────────────┬────────────────┘
               │                       │
               ▼                       ▼
┌──────────────────────────┐  ┌──────────────────────┐
│    robots/ (逻辑层)       │  │  assets/ (数据层)     │
│  Python类定义             │  │  MuJoCo XML/Mesh     │
│  ┌────────────────────┐  │  │  ┌────────────────┐  │
│  │ FrankaPandaRobot  │──┼──┼─▶│ Panda.xml      │  │
│  │ AuboI5Robot       │──┼──┼─▶│ Aubo_i5.xml    │  │
│  │ UR5eRobot         │──┼──┼─▶│ UR5e.xml       │  │
│  └────────────────────┘  │  │  └────────────────┘  │
│                          │  │                      │
│  提供：                   │  │  提供：               │
│  • 关节限位              │  │  • 3D模型             │
│  • DH参数                │  │  • 网格文件           │
│  • 控制接口              │  │  • 碰撞体             │
│  • Home位置              │  │  • 视觉外观           │
│  • 运动学参数            │  │  • 场景配置           │
│  • XML路径获取           │  │                      │
└──────────────────────────┘  └──────────────────────┘
```

#### 分工详解

**`assets/` 目录（数据层）**：
- 存储MuJoCo仿真所需的**物理模型数据**
- XML文件定义几何形状、惯性、关节等物理属性
- 网格文件（STL/OBJ）提供视觉和碰撞网格
- 纹理文件提供视觉外观
- **独立于Python，可被任何MuJoCo应用加载**

**`robots/` 目录（逻辑层）**：
- 提供机器人的**Python编程接口**
- 定义机器人参数（关节限位、速度限制、DH参数等）
- 提供配置生成、路径验证等工具函数
- 关联对应的 `assets/` 中的XML文件
- **为Python应用提供高层次API**

#### 典型使用流程

```python
# 1. 通过robots模块获取机器人配置
from mujoco_env.robots import AuboI5Robot

# 2. 获取机器人参数
config = AuboI5Robot.get_config(
    controller_type="joint_position",
    has_gripper=True
)

# 3. 获取对应的XML文件路径（指向assets/）
xml_path = AuboI5Robot.get_xml_path(scene="default")
# 返回: assets/models/manipulators/Aubo_i5/Aubo_i5.xml

# 4. 加载MuJoCo模型
import mujoco
model = mujoco.MjModel.from_xml_path(str(xml_path))

# 5. 使用机器人类的工具函数
qpos = [0, -0.5, 1.5, 0, 1.57, 0]
is_valid = AuboI5Robot.validate_qpos(qpos)  # 验证关节位置
qpos_clipped = AuboI5Robot.clip_qpos(qpos)  # 限制在安全范围
```

#### 优势

这种分离设计的优势：

1. **模块化**：物理模型和控制逻辑独立，易于维护
2. **复用性**：同一个XML可被多个Python类使用（如不同控制策略）
3. **灵活性**：可以单独加载XML进行仿真，无需Python类
4. **扩展性**：添加新机器人只需同时添加XML和Python类
5. **分工明确**：
   - 建模人员专注于 `assets/` 中的物理模型
   - 算法工程师专注于 `robots/` 中的控制逻辑

#### 注册机制

两个目录都使用注册表模式：

```python
# assets/__init__.py
MANIPULATOR_MODELS = {
    "panda": "models/manipulators/Panda/Panda.xml",
    "aubo_i5": "models/manipulators/Aubo_i5/Aubo_i5.xml",
    # ...
}

# robots/__init__.py
ROBOT_REGISTRY = {
    "franka_panda": FrankaPandaRobot,
    "aubo_i5": AuboI5Robot,
    # ...
}
```

这确保了名称一致性和统一的访问接口。

---

## 🤖 机械臂模型

### 1. Franka Panda

**文件**: `models/manipulators/Panda/Panda.xml`

**规格**:
- 自由度: 7 DOF
- 负载: 3 kg
- 工作半径: 855 mm
- 重复精度: ±0.1 mm

**特点**:
- 高精度协作机器人
- 力矩传感器
- 碰撞检测
- 完整的视觉和碰撞网格

**使用**:
```python
from mujoco_env.mujoco_env.assets import get_model_path

panda_path = get_model_path("manipulator", "panda")
```

### 2. Aubo i5

**文件**: `models/manipulators/Aubo_i5/Aubo_i5.xml`

**规格**:
- 自由度: 6 DOF
- 负载: 5 kg
- 工作半径: 920 mm
- 重复精度: ±0.05 mm

**特点**:
- 国产协作机器人
- 高性价比
- 适合工业应用

### 3. UR5e

**文件**: `models/manipulators/UR5e/UR5e.xml`

**规格**:
- 自由度: 6 DOF
- 负载: 5 kg
- 工作半径: 850 mm

**特点**:
- Universal Robots经典型号
- 广泛应用
- 丰富的生态系统

### 4. Franka FR3

**文件**: `models/manipulators/franka_fr3/franka_fr3.xml`

**规格**:
- 自由度: 7 DOF
- Panda的升级版本

### 5. DianaMed

**文件**: `models/manipulators/DianaMed/DianaMed.xml`

**规格**:
- 自由度: 7 DOF
- 医疗机器人

---

## 🦾 夹爪/末端执行器

### 1. Panda Hand

**文件**: `models/grippers/panda_hand/panda_hand.xml`

**特点**:
- Franka原装夹爪
- 平行夹爪
- 最大开口: 80 mm

### 2. Robotiq 2F-85

**文件**: `models/grippers/robotiq_2f85/2f85.xml`

**特点**:
- 自适应夹爪
- 最大开口: 85 mm
- 广泛应用

### 3. RealSense相机

**文件**: `models/grippers/realsense/realsense.xml`

**特点**:
- Intel RealSense D435
- RGB-D相机
- 用于视觉感知

### 4. 装配轴工具

**文件**: `models/grippers/assemble_axle/assemble_axle.xml`

**特点**:
- 专用装配工具
- 用于轴孔装配任务

---

## 🏗️ 背景物体

### 桌子

**文件**: `objects/table/table.xml`

标准工作台，用于大多数操作任务。

### 方块

**文件**: 
- `objects/cube/red_cube.xml`
- `objects/cube/green_cube.xml`
- `objects/cube/blue_cube.xml`

不同颜色的方块，用于抓取和放置任务。

### 电机

**文件**: `objects/motor/motor.xml`

复杂的电机模型，用于装配任务。包含63个凸分解网格。

### 传送带

**文件**: `objects/conveyor belt/conveyor_belt.xml`

传送带模型，用于工业场景模拟。

---

## 🎨 纹理

纹理文件位于 `textures/` 目录，包括：

- **aruco.png**: ArUco标记
- **chessboard.png**: 棋盘格（用于相机标定）
- **concrete.png**: 混凝土
- **wood.png**: 木纹
- **carpet-black.png**: 黑色地毯
- **light_wood.png**: 浅色木纹
- **white-marble.png**: 白色大理石

---

## 🎬 预定义场景

### 1. 默认场景

**文件**: `scenes/default.xml`

基础场景，包含地面和光照。

### 2. 抓取场景

**文件**: `scenes/grasping.xml`

包含桌子和物体的抓取场景。

### 3. 装配场景

**文件**: `scenes/assemble.xml`

用于装配任务的场景。

### 4. Panda抓取方块

**文件**: `scenes/PandaPickCube/arena.xml`

完整的Panda抓取方块场景，包括：
- Panda机器人
- 工作台
- 方块
- 相机
- 光照

---


## 💻 使用方法

### 基础使用

```python
from mujoco_env.mujoco_env.assets import (
    get_model_path,
    list_models,
    get_asset_info
)

# 获取模型路径
panda_path = get_model_path("manipulator", "panda")
gripper_path = get_model_path("gripper", "robotiq_2f85")
scene_path = get_model_path("scene", "grasping")

# 列出所有可用模型
manipulators = list_models("manipulator")
print(f"可用机械臂: {manipulators}")

# 获取资源统计
info = get_asset_info()
print(f"资源统计: {info}")
```

### 在环境中使用

```python
import mujoco
from mujoco_env.mujoco_env.assets import get_model_path

# 加载Panda模型
model_path = get_model_path("manipulator", "panda")
model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)
```

### 组合模型

```python
# 创建自定义场景
xml_string = f"""
<mujoco>
    <include file="{get_model_path('manipulator', 'panda')}"/>
    <include file="{get_model_path('gripper', 'robotiq_2f85')}"/>
    <include file="{get_model_path('object', 'table')}"/>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml_string)
```

---

## 📊 资源统计

| 类型 | 数量 | 说明 |
|------|------|------|
| 机械臂 | 5 | Panda, FR3, Aubo, UR5e, DianaMed |
| 夹爪 | 7 | 各种夹爪和工具 |
| 安装座 | 6 | 不同的安装方式 |
| 场景 | 5 | 预定义场景 |
| 物体 | 12+ | 各种场景物体 |
| 纹理 | 13+ | 各种材质纹理 |

---

## 🔧 文件格式

### XML文件

MuJoCo场景描述文件，包含：
- 模型定义
- 关节配置
- 碰撞体
- 视觉网格
- 材质和纹理

### 网格文件

- **STL**: 碰撞网格（简化）
- **OBJ**: 视觉网格（高精度）
- **DAE**: Collada格式（包含材质）

### 纹理文件

- **PNG**: 纹理图片

---

## 📝 添加新资源

### 1. 添加新机械臂

1. 在 `models/manipulators/` 创建新目录
2. 添加XML文件和网格文件
3. 在 `__init__.py` 的 `MANIPULATOR_MODELS` 中注册

### 2. 添加新夹爪

1. 在 `models/grippers/` 创建新目录
2. 添加XML文件和网格文件
3. 在 `__init__.py` 的 `GRIPPER_MODELS` 中注册

### 3. 添加新场景

1. 在 `scenes/` 创建XML文件
2. 在 `__init__.py` 的 `SCENE_MODELS` 中注册

---

## ⚠️ 注意事项

### 文件路径

- 所有路径使用相对路径
- 网格文件相对于XML文件
- 纹理文件相对于XML文件

### 网格优化

- 碰撞网格应尽量简化
- 视觉网格可以保持高精度
- 大型网格考虑使用凸分解

### 材质和纹理

- 使用合适的材质参数
- 纹理分辨率不要过大
- 考虑使用程序化材质

---

## 📚 参考资源

- [MuJoCo文档](https://mujoco.readthedocs.io/)
- [Franka Panda规格](https://www.franka.de/)
- [Aubo机器人](https://www.aubo-robotics.cn/)
- [Universal Robots](https://www.universal-robots.com/)

---

## 🙏 致谢

感谢以下开源项目提供的模型：
- Franka Emika官方模型
- MuJoCo Menagerie
- 各机器人厂商的官方资源

---

