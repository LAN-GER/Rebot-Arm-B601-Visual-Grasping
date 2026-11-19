# 模块与坐标约定

`cli.py` 解析参数并分派到 `pipeline/commands.py`。`config.py` 加载应用配置和后端 YAML，`bootstrap.py` 根据配置创建相机、检测器、抓取后端和 B601 会话。示例目录负责交互与启动；相机、标定、运动控制和抓取估计均可在应用包中独立导入。

## 一次抓取的数据流

1. 相机产生对齐的 `RGBDFrame`，含 BGR、米制深度、内参、时间和帧 ID。保留上游接口的相机流驱动位于 `cameras/streams/`，其 `get_frame()` 深度单位为毫米。
2. YOLO 提供与原图对齐的实例 mask，目标选择器选出一个实例。AnyGrasp 与过滤版 OBB 共用点云预处理和质量规则。过滤版 OBB 始终从完整 mask 计算二维几何。
3. 后端返回 `GraspCandidate`，统一为相机系的指尖中点和接近轴。AnyGrasp 在独立工作进程加载 gsnet，用 `create_detector` 和 `get_grasp` 推理。后端将 SDK 返回的深度和配置的额外下探各应用一次。
4. 规划层按宽度、分数和基坐标工作范围筛选；有机器人时再检查 IK。实时会话以当前帧的计划调用 `B601Session`。显示层只绘制图像、点云和诊断结果。
5. `robots/actions.py` 定义进入准备位、张爪、预抓取、下降、夹持和返回。`robots/gripper_driver.py` 将夹爪动作交给 B601 控制循环，AnyGrasp 在预抓取前确认张爪到位。

## 坐标与采样

矩阵 `T_A_B` 将 B 系中的点转换到 A 系。手眼标定得到 `T_tcp_camera`。相机光学轴 X 向右、Y 向下、Z 向前，工具 +X 指向接近方向，+Y 沿夹爪开合方向。矩阵平移单位为米。

```text
T_base_camera = T_base_tcp_at_capture @ T_tcp_camera
T_base_grasp  = T_base_camera @ T_camera_grasp
T_base_tcp    = T_base_grasp @ T_grasp_tcp
```

现场配置使用单位工具换算和零手眼平移补偿。AnyGrasp 真机采集对比拍摄前后的 TCP 读数以拒绝观察到运动的帧；过滤版 OBB 沿用准备位静止时取一次位姿的方式。稳定采样标定按编码器时间戳配对图像和关节位置。RGB-D 回放保留帧 ID 和已有的采集变换。

## 扩展后端

实现 `core/interfaces.py` 的后端接口，在 `grasping/backends/` 封装 SDK 及其坐标约定，在 `config/backends/` 添加参数，并在 `config.py` 和 `bootstrap.py` 注册。使用已有 `frame.npz` 可以验证后端到规划层的整条路径。新示例放入独立的 `examples/<任务>/`，复用这些接口。
