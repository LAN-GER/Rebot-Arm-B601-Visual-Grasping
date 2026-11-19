# RGB-D 单帧采集

只用相机。从仓库根目录运行。

```bash
bash examples/rgbd_capture/run.sh --output outputs/capture/frame.npz
```

使用配置指定的相机采集并对齐彩色图和深度。深度以米保存。可用 `--camera-type`、`--width`、`--height`、`--fps` 覆盖默认档位。
