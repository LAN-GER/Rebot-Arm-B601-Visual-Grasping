# OBB point cloud filtering comparison

[简体中文](obb_filtered_zh.md)

This example computes two OBB estimates for a banana in the same frame so you can inspect how gripper exclusion and outlier removal affect the executed pose. It uses YOLO, OpenCV, and the point cloud processing modules.

## Run the example

Use this command to preview with the camera only.

```bash
bash examples/obb_filtered/run.sh
```

Use this command for a robot trial. It enables the B601 and moves it to the ready pose on startup.

```bash
bash examples/obb_filtered/run.sh --execute
```

- **G** pauses background inference and captures a fresh frame for the comparison. With `--execute`, it runs the filtered candidate and resumes the live preview afterwards.
- **R** resumes the camera preview.
- **Q** or **Esc** ends the session. In execution mode the B601 releases the gripper and returns to zero using the original procedure.

Three windows open by default.

1. The OBB comparison shows the raw reference on the left and the filtered execution candidate on the right.
2. The point cloud window shows the filtered scene, selected target in green, and candidate gripper in orange. Use the Open3D mouse controls to pan, rotate, or zoom.
3. The segmentation and depth diagnostics show retained target pixels in green, excluded target pixels in yellow, and gripper exclusions in purple. The right panel displays depth in meters, with invalid depth in black, and shows point counts and quality metrics.

Live preview continuously runs YOLO and filters the point cloud. The main window shows the banana box, class, confidence, processing rate, and frame age. All three windows use the same processed frame. While processing one frame, the background worker drops intermediate frames to avoid a backlog. The refresh rate depends on processing time.

Use **G**, **R**, and **Q** in the image window. Closing the point cloud window closes that view only. **G** uses a new capture. Preview computes estimates without reading robot state, solving robot IK, moving the arm, or saving every frame. When no target is selected, the cloud window clears stale points. During execution the windows show the execution frame, then resume live preview when the action returns.

Use `--single-window` to display only the OBB comparison. Offline `--input` mode saves diagnostic images without opening windows. Use `--input` and `--execute` in separate runs.

```bash
bash examples/obb_filtered/run.sh \
  --input /path/to/frame.npz
```

## How filtering affects OBB

The project passes the aligned target mask to the original B601 `estimate_grasp` routine. It retains the minimum-area rectangle, short-axis cross-section refinement, depth quantile, and TCP axis conversion.

The same RGB-D frame and selected detection feed both estimates.

1. The left **RAW REFERENCE** panel uses the raw mask and depth. It saves the reference estimate for comparison.
2. The right **FILTERED** panel applies gripper exclusion, workspace limits, outlier removal, and quality checks. It projects retained points back to the source pixels. **The full 2D segmentation contour supplies the OBB rectangle, center, short-axis direction, and pixel width. Filtering selects the target depths used for the depth quantile and the quality check.** Blue boxes mark rectangles, red dots mark grasp centers, and white lines mark the opening axis. Green indicates points used for depth estimation. Purple indicates gripper exclusions.

The algorithm identifier `obb_full_mask_filtered_depth_v2` appears in each `comparison.json`. Both panels should agree on the 2D center and direction. Depth can still change the 3D center and metric width. The default extra insertion depth is 0 mm.

When the fingers reach only the top of the object, test a small insertion adjustment.

```bash
bash examples/obb_filtered/run.sh --execute --insertion-depth-mm 5
```

This example adds 5 mm along tool +X. The pipeline applies the offset once, keeps the pre-grasp 80 mm back from the final candidate, and still checks the workspace and IK. The setting affects insertion distance. B601 motion timing and gripping force use their own settings. Omit the flag to use the configured value, which defaults to 0 mm.

The side-by-side comparison still uses the surface estimates. The right panel marks extra insertion. In `comparison.json`, `candidate_position_m` gives the execution position and `insertion_offset_camera_m` gives the offset. The 3D gripper uses the adjusted pose. `session.json` records the active parameters. The fitted tabletop clearance is an estimate, and the pipeline never uses it to increase insertion automatically.

Depth holes remain unfilled. B601's fallback for finding depth near the center only uses retained target depths. The execution path rejects failed quality checks, empty targets, and invalid OBBs.

The left reference uses B601 geometry with the same detection as the right panel. This entry point defaults to confidence 0.30 and target `yellow banana`. The original entry point uses its own settings, which previously used 0.50 and selected the class with highest confidence. Compare estimates from the same frame.

## Configuration and control

- `config/obb_filtered.yaml` configures the camera, YOLO, robot, calibration path, and planning workspace.
- `config/backends/obb_filtered.yaml` configures the OBB depth quantile of 0.5, filtering, and quality thresholds. Initial preprocessing values match AnyGrasp and can be tuned independently.
- `grasping/backends/obb_filtered.py` computes the estimates from inputs.
- `pipeline/obb_filtered.py` saves results and calls the shared B601 session.
- `pipeline/live_session.py` provides the shared live workflow. Existing AnyGrasp entry points retain compatibility exports.
- `visualization/obb_comparison.py` exports images and point clouds.

`B601Session` calls `move_ready` and `execute_grasp` in `robots/actions.py`. The sequence uses a 2-second pre-grasp and a 1.5-second descent, then closes the gripper and returns to ready. The filtered backend's insertion setting defaults to 0 mm and overrides placement independently of the standard OBB backend's 15 mm offset.

Execution checks width, base-frame workspace, and IK. The example captures the image and reads the TCP once while the robot waits at the ready pose. This procedure uses the stationary pose assumption. Hardware timestamp synchronization and full motion-path collision planning are outside the scope of this example.

## Saved results

Each run creates `outputs/obb-filtered/run-XXXXXXXX/` and saves terminal output to `console.log`. The effective configuration, overrides, and calibration transform are saved in `result/session.json`.

Each press of **G** in live mode creates `result/snapshot-NNN/`. Offline mode saves directly to `result/`.

| File | Contents |
| --- | --- |
| `frame.npz`, `color.png` | RGB-D, target mask, and camera-to-base transform when available |
| `comparison.png` | Raw and filtered OBBs from the same frame |
| `diagnostics.png` | Segmentation, depth, and quality metrics shown in the diagnostics window |
| `comparison.json` | Centers, widths, depths, TCP poses, and their differences |
| `target_mask.png`, `eligible_target.png` | Target pixels before and after filtering |
| `gripper_exclusion.png` | Pixels excluded by the gripper mask |
| `pointcloud_raw.npz`, `pointcloud.npz` | Point clouds before and after filtering |
| `result.json` | Final candidate, plan, quality report, and reason execution was skipped |
| `execution.json` | Written when a grasp is invoked. Records actions and gripper feedback. The field outcome starts as unknown. |

If the target is absent, the example saves the new frame and an empty comparison. `object_held: false` is controller feedback. The software has returned it when the fingers held the object. `field_success` remains `null` pending field confirmation.

The example was tested with offline inference on recorded camera frames and simulated execution. In field trials, the operator observed better grips after adding 5 mm along tool +X.
