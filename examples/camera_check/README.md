# Segmentation and point cloud quality check

[简体中文](README_zh.md)

Capture one frame or replay a recording without moving the robot. Run from the repository root.

```bash
bash examples/camera_check/run.sh --target-class "yellow banana"
```

The example displays segmentation and point clouds and saves `inspection.json`, the pixel masks, and the point clouds before and after filtering. Append `--input /path/to/frame.npz` to replay a frame. On a headless machine, use `bash scripts/run_python.sh -m rebot_grasp camera-check --input /path/to/frame.npz --output outputs/check`. [Quality metrics](../../docs/pointcloud_preprocessing.md)
