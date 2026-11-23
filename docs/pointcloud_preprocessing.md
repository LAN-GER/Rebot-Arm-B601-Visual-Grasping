# Point cloud preprocessing and quality checks

[简体中文](pointcloud_preprocessing_zh.md)

`camera-check` and `AnyGraspBackend` share `perception/preprocessing.py`. It takes an RGB-D frame, target mask, and parameters. The robot uses the B601 actions in `robots/actions.py`.

## Module responsibilities

| Module | Responsibility |
| --- | --- |
| `perception/pointcloud.py` | RGB-D projection, statistical outlier removal, and alignment of points, colors, and masks |
| `perception/spatial_filters.py` | Camera-frame workspace and fixed-view gripper mask |
| `perception/cloud_quality.py` | Quality metrics, threshold validation, and pass or rejection reasons |
| `perception/preprocessing.py` | Processing sequence, per-stage counts, processed data, and report |
| `visualization/segmentation.py` | Inspection images, point cloud files, and windows |
| `grasping/backends/anygrasp.py` | SDK inference after quality checks, candidate transforms, and camera workspace validation |

## Processing order

1. Filter by depth and project to camera optical coordinates. X points right, Y down, and Z forward. Distances are in meters.
2. Exclude identified gripper points that fall inside both the fixed-view polygon and its depth range.
3. Limit target points with a camera-frame 3D box. **Scene points outside the workspace remain available for collision detection.** The box limits target `region_steering`. Full-scene mode also applies the workspace mask.
4. Remove statistical outliers and keep the colors and target mask aligned with the remaining points.
5. Check quality. A failed check returns its reason and skips SDK inference. Empty targets or workspaces also stop this stage.
6. After inference, check whether the common fingertip reference lies within the camera workspace. The planner checks the robot workspace in base coordinates and solves IK.

The camera workspace bounds visual processing. Robot reachability and motion-path collisions require separate checks.

## Field settings

The parameters live in `config/backends/anygrasp.yaml`.

- `workspace_camera` sets XYZ bounds from `[-0.25, -0.20, 0.10]` to `[0.25, 0.20, 0.60]` meters.
- `gripper_exclusion` uses a conservative interior outline from the fixed Gemini 336 view in `camera-check-F9b4OCbE`. It excludes points between 0.05 and 0.19 m using polygons in normalized image coordinates.
- `outlier_filter` uses 20 neighbors and a standard deviation multiplier of 2.0.
- `pointcloud_quality` requires at least 1,000 scene points, 300 target points, 60% valid depth in the original target mask, and 50% retention of original valid target points. Outlier removal may discard up to 20% of the target points entering that step.

Treat these values as tunable starting points. The gripper mask excludes a conservative part of the visible gripper. Review it after moving the mount or changing the image crop, resolution, or gripper shape. A resolution mismatch raises an error. A mounting change at the same resolution still needs a visual check. Object points outside the mask's depth range remain in the cloud.

Set any of the three optional filters to `null` to disable it. `pointcloud_quality: null` disables the extra count and ratio thresholds. The pipeline still rejects empty targets and scenes. Confirm that a region belongs to the gripper before adding it to the mask so objects and obstacles remain visible.

## Run and inspect results

```bash
bash examples/camera_check/run.sh
```

This example captures and segments an RGB-D frame and checks its point cloud. It needs no AnyGrasp license and does not move the arm. Use `examples/camera_check/run.sh --input /absolute/path/frame.npz` to replay one frame without a camera.

Each run creates a new `outputs/camera-check/run-*/result` directory.

| File | Contents |
| --- | --- |
| `frame.npz` / `color.png` | Original RGB-D frame and color image |
| `segmentation.png` / `mask-*.png` | Original detection instances for comparison |
| `preprocessing.png` | Filtered target in green, excluded gripper points in purple, and out-of-workspace pixels or missing depth in gray. Quality appears at the top left. |
| `gripper_exclusion.png` | Pixels that intersect the gripper polygon and depth range in this frame |
| `eligible_target.png` | Final pixels eligible for target steering |
| `pointcloud_raw.npz` / `pointcloud.npz` | Clouds before and after processing, with aligned points, colors, and target masks |
| `inspection.json` | Per-stage counts, parameters, thresholds, metrics, and rejection reasons |

The inspection window displays preprocessing results. Press **Q** or **Esc** to open the cloud view. Exit code `0` means the quality checks passed, `2` means detection or quality checks failed, and `1` means a configuration or runtime error. Use actual AnyGrasp inference and a separate robot trial to verify grasping.

In `inspection.json`, `preprocessing.counts` tracks removals by the gripper mask, workspace filter, and outlier step. The top-level `removed_selected_point_count` sums all removals. `valid_depth_ratio` divides by the original target-mask pixel count. `retained_ratio` divides by the original count of valid target depth points. `outlier_ratio` divides by the target point count immediately before outlier removal. Full-scene mode uses all image pixels as the original region.

Inference saves the same `pointcloud_report` in `result.json`. A quality failure produces `pointcloud_rejected` and stops grasp planning and execution. A missing requested target produces `target_not_found`.

## Rejection reasons

| Code | Meaning |
| --- | --- |
| `target_not_found` | The requested target was absent |
| `insufficient_scene_points` | Too few scene points |
| `insufficient_target_points` | Too few points in the target or grasp region |
| `insufficient_valid_depth` | Too little valid depth in the original target mask |
| `excessive_region_removal` | Too few target points remain after filtering |
| `excessive_outliers` | Outlier removal discarded too much of the target |

Depth holes remain unfilled. Visual inspection can reveal local edge errors and occlusion even when quality checks pass.
