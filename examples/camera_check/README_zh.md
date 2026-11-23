# 分割与点云质量检查

[English](README.md)

采集一帧或回放录制，无机械臂动作。从仓库根目录运行。

```bash
bash examples/camera_check/run.sh --target-class "yellow banana"
```

显示分割与点云，保存 `inspection.json` 和过滤前后的像素、点云。可加 `--input /path/to/frame.npz` 回放。无图形桌面可直接使用 `bash scripts/run_python.sh -m rebot_grasp camera-check --input /path/to/frame.npz --output outputs/check`。[质量指标](../../docs/pointcloud_preprocessing_zh.md)
