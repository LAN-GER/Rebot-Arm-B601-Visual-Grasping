# Synthetic OBB demo

[简体中文](README_zh.md)

This demo runs on the CPU and requires no devices. Run from the repository root.

```bash
bash examples/offline_demo/run.sh --output outputs/demo
```

The demo generates synthetic RGB-D data and a target mask, runs OBB grasp estimation and planning, and saves `frame.npz`, `preview.png`, and `result.json`.
