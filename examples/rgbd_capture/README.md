# Single-frame RGB-D capture

[简体中文](README_zh.md)

This example uses only the camera. Run from the repository root.

```bash
bash examples/rgbd_capture/run.sh --output outputs/capture/frame.npz
```

The configured camera captures and aligns a color image and depth frame. Depth is saved in meters. Use `--camera-type`, `--width`, `--height`, and `--fps` to override the defaults.
