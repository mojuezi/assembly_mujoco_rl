# impl/commons 移植完成总结



## 📊 核心模块功能

### transform.py - 坐标变换
- 欧拉角 ↔ 四元数 ↔ 旋转矩阵 ↔ 旋转向量
- 齐次变换矩阵构建和求逆
- 力和力矩的坐标系变换
- 数据归一化/逆归一化

### renderers.py - MuJoCo渲染器
- 实时交互式渲染（human模式）
- 离线图像渲染（rgb_array、depth模式）
- 相机视图显示
- 轨迹可视化
- 键盘交互（Space暂停、Esc退出、Enter保存）

### cv_utils.py - 计算机视觉
- OpenCV窗口管理
- 图像显示和保存
- 相机内参计算

### plot_utils.py - 实时绘图
- 多子图支持
- 多进程绘图（不阻塞主程序）
- 滑动窗口显示

### xml_splice.py - XML拼接
- 机器人模型组装
- 场景构建
- 资源管理

---

## 🔧 技术细节

### 坐标变换模块 (`transform.py`)

**功能**:
- 欧拉角 ↔ 四元数 ↔ 旋转矩阵 ↔ 旋转向量
- 齐次变换矩阵构建和求逆
- 力和力矩的坐标系变换
- 数据归一化/逆归一化

**核心函数**:
```python
euler_2_quat()    # 欧拉角 -> 四元数
quat_2_mat()      # 四元数 -> 旋转矩阵
mat_2_euler()     # 旋转矩阵 -> 欧拉角
make_transform()  # 构建齐次变换矩阵
pose_inv()        # 变换矩阵求逆
```

### 渲染器模块 (`renderers.py`)

**功能**:
- 实时交互式渲染（human模式）
- 离线图像渲染（rgb_array、depth模式）
- 相机视图显示（需要OpenCV）
- 轨迹可视化
- 键盘交互

**键盘快捷键**:
- Space: 暂停/继续
- Esc: 退出
- Enter: 保存图像

### 计算机视觉模块 (`cv_utils.py`)

**功能**:
- OpenCV窗口管理
- 图像显示和保存
- 相机内参计算

### 绘图模块 (`plot_utils.py`)

**功能**:
- 实时数据绘图
- 多子图支持
- 多进程绘图（避免阻塞）

---

## 🎯 重要提示

1. **环境要求**: 必须在 `conda serl` 环境中运行
2. **Gymnasium版本**: 需要 `gymnasium==1.0.1`
3. **向后兼容**: 旧代码仍可工作，但推荐使用新导入路径
4. **依赖安装**: 运行 `pip install -r requirements.txt`

---

