# Standard OBB grasping

[简体中文](README_zh.md)

The original interactive workflow enables the arm and moves it to the ready pose on startup. Run from the repository root.

```bash
bash examples/obb_grasp/run.sh --help
```

Press **G** to execute a contour-based OBB grasp. Startup connects to the arm and enters the ready pose. `--dry-run` skips the grasp itself. To inspect the estimate with the camera only, use the [OBB camera preview](../obb_preview/README.md).
