# AnyGrasp point cloud filtering comparison

[简体中文](README_zh.md)

The current tabletop vocabulary is `pen`, `orange`, and `yellow banana`. The window shows the class and confidence for each detected instance. The green box marks the current target, and blue boxes mark other detections. The default target is the banana. Use `--target-class pen` or `--target-class orange` to switch it. The preview runs AnyGrasp for each instance using its mask. White lines show current target candidates, yellow lines show other candidates, and Z gives the depth of the candidate center in camera coordinates. The window labels instances rejected by point cloud quality checks or without a valid candidate. Press **1** for pen, **2** for orange, or **3** for banana in the image window. The keys follow the configured vocabulary order. Press **G** to capture a new frame. Execution mode grasps the selected class and skips the action if the target is absent.

Both panels use the same RGB-D frame, the same selected instance, and the same AnyGrasp model. **RAW REFERENCE** uses the raw points and target mask within the depth range. **FILTERED** applies the gripper mask, camera workspace, outlier removal, and quality checks. Both sides share model, collision detection, width and score limits, and extra insertion depth. The standard AnyGrasp example uses its own configuration.

A white line connects the projected fingertips and a red dot marks the grasp center. While waiting for **G**, each side shows its highest-scoring valid candidate. Each detected instance typically takes two inference calls. Frame age is shown at the bottom. The auxiliary windows show the filtered cloud and segmentation/depth, with excluded gripper pixels in purple.

## Run the example

Set up the [AnyGrasp license and detection weights](../../README.md#anygrasp-license-and-detection-weights) before starting. Run from the repository root. The first command starts the camera preview. The second moves the arm to the ready pose. Press **G** to capture a new frame and select a right-panel candidate that passes workspace and IK checks. If quality checks fail or no candidate is reachable, the example skips the grasp. The preview resumes automatically after execution. Press **Q** or **Esc** to end the session.

```bash
bash examples/anygrasp_filtered/run.sh
bash examples/anygrasp_filtered/run.sh --execute --target-class "yellow banana"
```

If the grip is too shallow, append `--insertion-depth-mm <millimeters>` to the execution command. This overrides extra travel along the tool's approach axis for that run. The default is 5 mm. [Backend configuration](../../config/backends/anygrasp_filtered.yaml)

Use `--single-window` to show only the side-by-side comparison. The current detection confidence threshold is 0.30. [Runtime configuration](../../config/anygrasp_filtered.yaml)

## Replay and results

```bash
bash examples/anygrasp_filtered/run.sh --input /path/to/snapshot-001/frame.npz
```

Offline replay redetects targets in recorded RGB-D data and runs the same comparison on the GPU. Output goes to `outputs/anygrasp-filtered/run-*/result/` by default, and the terminal prints the run directory. Use `--output` to specify a new directory. Each press of **G** in a live session saves a `snapshot-NNN/`.

- `comparison.png` and `comparison.json` show both grasps, the change in center in camera coordinates, the width difference, and the angle between the undirected opening axes. The two inference calls may select different grasps, so candidate ranking can contribute to the single-frame difference.
- Review segmentation, gripper exclusion, and outliers using `diagnostics.png`, `pointcloud_raw.npz`, `pointcloud.npz`, and the masks. An empty comparison is saved if the target is absent.
- `result.json` records the filtered candidates, plan, and quality result. During execution the right panel marks the selected reachable candidate, and `execution.json` records the action.
