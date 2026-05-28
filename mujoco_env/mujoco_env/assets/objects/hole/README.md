轴孔（hole）模型处理说明
=====================

目录
- 本说明位于：`mujoco_env/assets/objects/hole/README.md`

背景
----
本项目中的轴孔场景原始几何为 `zhoukong.STL`（位于 `mujoco_env/assets/objects/hole/zhoukong.STL`），MuJoCo 对复杂凹面网格直接作为单个 mesh 的碰撞表示支持有限，且可能导致“不可穿透”的问题。为保证仿真中碰撞表现良好，我们将原始 STL 转为 OBJ，并使用 `obj2mjcf` 工具把凹面几何分解为 MuJoCo 可识别的凸面碰撞块集合（collision parts）。

处理流程（概览）
----------------
1. 将 `zhoukong.STL` 转为 `hole_fixed.obj`（确保为 ASCII 编码的 OBJ 文件）。  
   - 使用你偏好的 3D 工具（Blender / MeshLab / MeshlabServer / 等）导入 `zhoukong.STL`，然后另存为 `.obj`（ASCII 编码）。

2. 将 OBJ 文件命名为 `hole_fixed.obj` 并放入目录：  
   `mujoco_env/assets/objects/hole/hole_fixed/hole_fixed.obj`

3. 安装并使用 obj2mjcf 工具包将 OBJ 分解为多个凸面碰撞部件：  
   - 仓库与文档： `https://github.com/kevinzakka/obj2mjcf/blob/main/README.md`  
   - 示例命令（在 Windows 命令行中运行）：  
     ```
     obj2mjcf --obj-dir "D:\mujoco-env\mujoco_env\assets\objects\hole" --save-mjcf --decompose --add-free-joint --obj-filter "hole_fixed"（注意保持mtl文件名和obj文件中的引用名一致）
     ```
   - 该命令会在 `hole_fixed` 子目录下生成多个 `hole_fixed_collision_*.obj` 文件和一个 MJCF 模板文件。

4. 将分解出的 collision meshes 声明内联到场景 XML：  
   - 在 `mujoco_env/assets/scenes/assemble_hole.xml` 的 `<asset>` 中声明 `hole_fixed_collision_*.obj` 为 `<mesh file="..."/>`。  
   - 在 `body name="hole"` 中添加对应的 `<geom type="mesh" mesh="hole_fixed_collision_X" contype="1" conaffinity="1" condim="4"/>` 作为碰撞几何。

关于缩放调整
------------
- 由于 `obj2mjcf` 默认的分解精度导致孔洞略小，我们在场景中把 collision meshes 放大了 1.2 倍（即 `scale="1.2 1.2 1.2"`），以让分解后的孔尺寸更接近原始模型。  
- 若需更精确的孔尺寸，建议提高 `obj2mjcf` 的分解精度或在 3D 工具中进一步清理/重网格化后重试分解。



附：obj2mjcf 参考
-----------------
- 官方仓库：`https://github.com/kevinzakka/obj2mjcf/blob/main/README.md`  
- 该工具会把复杂网格分解为多个凸面几何（便于 MuJoCo 的碰撞处理），并可以生成带 freejoint 的 MJCF 结构以便调试与定位。

。


