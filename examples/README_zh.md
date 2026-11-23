# 示例索引

[English](README.md)

以下命令从工程根目录执行。每个目录都提供独立的 `run.sh`；追加 `--help` 查看参数。普通 OBB、抓放和 GraspNet 真机示例保留上游启动行为，进入应用即会移动到准备位。

旧示例命令的兼容入口位于 `scripts/legacy/`。

| 示例 | 运行方式 | 对应的兼容命令或 CLI |
| --- | --- | --- |
| [AnyGrasp 实时抓取](anygrasp/README_zh.md) | 相机预览，显式执行时连接机械臂 | `scripts/legacy/anygrasp.sh` |
| [过滤版 AnyGrasp 对照](anygrasp_filtered/README_zh.md) | 相机预览、离线回放或显式执行 | `rebot-grasp anygrasp-filtered` |
| [过滤版 OBB 对照](obb_filtered/README_zh.md) | 相机预览、离线回放或显式执行 | `scripts/legacy/obb_filtered.sh` |
| [手眼标定](hand_eye/README_zh.md) | 标记检查只用相机，自动轨迹会移动机械臂 | `scripts/legacy/collect_handeye_eih.py` |
| [稳定采样手眼标定](stationary_calibration/README_zh.md) | 支持已存反馈离线预演，显式执行时跑轨迹 | `scripts/legacy/stationary_calibration.py` |
| [分割与点云质量检查](camera_check/README_zh.md) | 采集一帧或回放录制，无机械臂动作 | `scripts/legacy/camera_check.sh` |
| [7 号电机张爪诊断](gripper_check/README_zh.md) | 仅连接并驱动 B601-RS 的 7 号夹爪电机 | `scripts/legacy/gripper_check.py` |
| [合成 OBB](offline_demo/README_zh.md) | 纯 CPU，无设备 | `rebot-grasp demo` |
| [RGB-D 单帧采集](rgbd_capture/README_zh.md) | 只用相机 | `rebot-grasp record` |
| [录制帧抓取回放](replay/README_zh.md) | 离线推理 | `rebot-grasp preview` |
| [普通 OBB 抓取](obb_grasp/README_zh.md) | 原版交互流程，启动会使能并移动到准备位 | `scripts/legacy/main.py` |
| [OBB 相机预览](obb_preview/README_zh.md) | 只用相机，或离线回放 | `scripts/legacy/ordinary_grasp_pipeline.py` |
| [目标检测与深度取样](object_detection/README_zh.md) | 只用相机 | `scripts/legacy/object_detection.py` |
| [香蕉放入盒子](pick_and_place/README_zh.md) | 原版交互流程，启动会使能并移动到准备位 | `scripts/legacy/set.py` |
| [GraspNet 相机预览](graspnet_camera/README_zh.md) | 相机和 GPU | `scripts/legacy/graspnet_camera_demo.py` |
| [GraspNet 真机抓取](graspnet_grasp/README_zh.md) | 相机、GPU、机械臂，启动会使能并进入准备位 | `scripts/legacy/grasp.py` |
