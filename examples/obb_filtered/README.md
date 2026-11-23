# Filtered OBB comparison

[简体中文](README_zh.md)

Run from the repository root. The first command opens the camera preview. The second moves the arm to the ready pose. Press **G** in the window to capture a snapshot. Execution mode selects a reachable candidate from the filtered results.

```bash
bash examples/obb_filtered/run.sh
bash examples/obb_filtered/run.sh --execute
```

If the grip is too shallow, append `--insertion-depth-mm <millimeters>` to the execution command. This overrides the extra travel along the tool's approach axis for that run. The default is 0 mm. [Backend configuration](../../config/backends/obb_filtered.yaml)

The main window displays the raw and filtered results for the same frame side by side. The default output is `outputs/obb-filtered/run-*/result/`. [Windows, algorithm, and output](../../docs/obb_filtered.md)
