# RealSense 深度相机集成

| 修订日期   | 修订版本 | 修订内容   | 修订人 |
| ---------- | -------- | ---------- | ------ |
| 2025.12.05 | V0.1     | 初始化文档 | 徐天   |

## 概述
本文档记录Intel RealSense D435I深度相机的完整开发过程.

## 环境配置
- **操作系统**: Ubuntu 22.04 LTS
- **ROS版本**: ROS2 Humble  
- **相机型号**: Intel RealSense D435i/D435if

## 安装步骤

### 1. RealSense SDK 安装
这里采用预编译包安装的方法,也可以采用源码安装的方法
```bash
# 注册公钥
sudo mkdir -p /etc/apt/keyrings
curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null

# 安装HTTPS支持
sudo apt-get install apt-transport-https

# 添加软件源
echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo `lsb_release -cs` main" | \
sudo tee /etc/apt/sources.list.d/librealsense.list
sudo apt-get update

# 安装核心包
sudo apt-get install librealsense2-dkms
sudo apt-get install librealsense2-utils
sudo apt-get install librealsense2-dev

# 验证安装(重新连接RealSense深度相机,这里需要注意连接的是USB 3.0端口,否则会导致部分模块无法使用或者连接传输异常)
realsense-viewer
```
### 2. RealSense-Ros2安装
首先确保安装好了ROS2,并配置好了环境变量,这里以humble版本为例
```bash
# 源码安装realsense ros2
mkdir -p ./ros2_ws/src
cd ./ros2_ws/src
git clone https://github.com/IntelRealSense/realsense-ros.git -b ros2-master
cd ../
# 编译
sudo rosdep init
rosdep update
rosdep install -i --from-path src --rosdistro $ROS_DISTRO --skip-keys=librealsense2 -y
colcon build
# source environment
ROS_DISTRO=humble
# 写入以下内容到 ~/.bashrc(记得改第二个路径)
source /opt/ros/humble/setup.bash
source /path/to/ros2_ws/install/setup.bash
```
这期间如果遇到colcon的报错,安装缺失的依赖即可
### 3. 结果验证

#### 3.1 单独使用SDK验证
  使用`rs-`系列命令进行设备功能验证：
  - `rs-enumerate-devices` - 查看设备型号信息
  - `rs-depth` - 查看深度信息
  - `rs-pointcloud` - 查看点云信息
#### 3.2 配合ROS2使用验证
先运行launch文件(记得source)`ros2 launch realsense2_camera rs_launch.py`,然后查看当前发布话题`ros2 topic list`.根据当前发布的话题,可以实现画面获取,数据读取以及查看URDF描述文件.如:
- 1.打开可视化界面`rqt`(或者`rviz2`),然后选择`/camera/camera/color/image_raw/compressed`可以看到原始压缩图片画面,选择`/camera/camera/depth/image_rect_raw/`里的话题可以看到压缩深度图片等等;
- 2.利用`ros2 topic echo /camera/camera/accel/imu_info`或者`ros2 topic echo /camera/camera/accel/sample`,`ros2 topic echo /camera/camera/gyro/sample`可以获取imu数据;
- 3.在工作空间下,利用`ros2 run xacro xacro ~/ros2_ws/install/realsense2_description/share/realsense2_description/urdf/test_d435i_camera.urdf.xacro > d435i.urdf`可以直接保存D435i的`urdf`文件,或者可以前往`urdf`目录里查看现有的所有`xacro`文件,然后选择需要的进行`urdf`的生成并查看,或者保存.
### 注:
- 由于D435i与D435if只存在外表以及是否带红外滤光片的区别,因此在软件层面会统一识别成`D435I`.
- sdk支持多相机输入,可自行探索
- 如果需要对底层代码进行修改,需要先下载源码,再重新编译.如果只是在应用层进行修改,只需修改`ros2`库中`src`里的代码即可