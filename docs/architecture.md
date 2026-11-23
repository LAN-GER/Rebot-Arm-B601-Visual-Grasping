# Modules and coordinate conventions

[简体中文](architecture_zh.md)

`cli.py` parses arguments and dispatches to `pipeline/commands.py`. `config.py` loads the application and backend YAML files. `bootstrap.py` builds the camera, detector, grasp backend, and B601 session from configuration. Example directories provide interaction and launch commands. The application package exposes camera, calibration, motion control, and grasp estimation independently.

## Grasp data flow

1. The camera produces an aligned `RGBDFrame` with a BGR image, depth in meters, intrinsics, timestamp, and frame ID. The stream drivers in `cameras/streams/` retain the upstream interface and return depth in millimeters from `get_frame()`.
2. YOLO supplies instance masks aligned to the image. The target selector chooses one instance. AnyGrasp and filtered OBB share point cloud preprocessing and quality rules. Filtered OBB derives 2D geometry from the full instance mask.
3. Backends return a `GraspCandidate` expressed as the fingertip midpoint and approach axis in camera coordinates. An isolated AnyGrasp worker loads gsnet and uses `create_detector` and `get_grasp`. The backend applies the SDK depth and the configured extra insertion depth once each.
4. Planning filters by width, score, and workspace in base coordinates. It also checks IK when a robot is available. The live session passes the current frame's plan to `B601Session`. Visualization draws the image, point cloud, and diagnostics.
5. `robots/actions.py` defines the ready pose, gripper opening, pre-grasp, descent, closing, and return. `robots/gripper_driver.py` sends gripper actions through the B601 control loop. AnyGrasp confirms that the gripper has opened before pre-grasp.

## Coordinates and sampling

`T_A_B` transforms points from B coordinates to A coordinates. Hand–eye calibration estimates `T_tcp_camera`. Camera optical +X points right, +Y down, and +Z forward. Tool +X points along the approach direction and +Y along the gripper opening axis. Transform translations are in meters.

```text
T_base_camera = T_base_tcp_at_capture @ T_tcp_camera
T_base_grasp  = T_base_camera @ T_camera_grasp
T_base_tcp    = T_base_grasp @ T_grasp_tcp
```

The field configuration uses an identity tool transform and zero hand–eye translation compensation. Live AnyGrasp compares TCP readings before and after capture and rejects frames with observed motion. Filtered OBB reads one pose while the arm waits at the ready position. Calibration with stationary samples pairs images and joint positions using encoder timestamps. RGB-D replay preserves the frame ID and any capture transform.

## Add a backend

Implement the interface in `core/interfaces.py`, wrap the SDK and its coordinate conventions in `grasping/backends/`, add parameters under `config/backends/`, and register the backend in `config.py` and `bootstrap.py`. An existing `frame.npz` can exercise the full path from backend to planning. Add new examples under `examples/<task>/` and reuse the same interfaces.
