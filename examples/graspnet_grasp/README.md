# GraspNet robot grasping

[简体中文](README_zh.md)

This example uses a camera, GPU, and robot. It enables the arm and moves it to the ready pose on startup. Run from the repository root.

```bash
bash examples/graspnet_grasp/run.sh --help
```

Set up GraspNet-Baseline, model weights, and the hand–eye matrix before running. Press **G** to infer and execute a candidate. The arm still moves to the ready pose with `--dry-run`. To inspect candidates with the camera only, use the [GraspNet camera preview](../graspnet_camera/README.md).
