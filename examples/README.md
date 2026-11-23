# Example index

[简体中文](README_zh.md)

Run these examples from the repository root. Each directory has its own `run.sh`. Append `--help` to see its options. The original OBB, pick-and-place, and GraspNet robot examples move the arm to its ready pose on startup.

Older example launchers now live in `scripts/legacy/`.

| Example | Run mode | Compatibility entry point or CLI |
| --- | --- | --- |
| [Live AnyGrasp](anygrasp/README.md) | Camera preview or explicit robot execution | `scripts/legacy/anygrasp.sh` |
| [Filtered AnyGrasp comparison](anygrasp_filtered/README.md) | Camera preview, offline replay, or explicit robot execution | `rebot-grasp anygrasp-filtered` |
| [Filtered OBB comparison](obb_filtered/README.md) | Camera preview, offline replay, or explicit robot execution | `scripts/legacy/obb_filtered.sh` |
| [Hand–eye calibration](hand_eye/README.md) | Camera-only marker check or automatic calibration with arm motion | `scripts/legacy/collect_handeye_eih.py` |
| [Hand–eye calibration with stationary samples](stationary_calibration/README.md) | Offline trajectory preview from recorded feedback or explicit robot execution | `scripts/legacy/stationary_calibration.py` |
| [Segmentation and point cloud quality](camera_check/README.md) | Single-frame capture or offline replay without arm motion | `scripts/legacy/camera_check.sh` |
| [Motor 7 gripper-opening diagnostic](gripper_check/README.md) | Drive only gripper motor 7 on the B601-RS | `scripts/legacy/gripper_check.py` |
| [Synthetic OBB demo](offline_demo/README.md) | CPU only, no devices required | `rebot-grasp demo` |
| [Single-frame RGB-D capture](rgbd_capture/README.md) | Camera only | `rebot-grasp record` |
| [Grasp preview from a recording](replay/README.md) | Offline inference | `rebot-grasp preview` |
| [Standard OBB grasping](obb_grasp/README.md) | Original interactive workflow with arm motion on startup | `scripts/legacy/main.py` |
| [OBB camera preview](obb_preview/README.md) | Camera or offline replay | `scripts/legacy/ordinary_grasp_pipeline.py` |
| [Object detection and depth sampling](object_detection/README.md) | Camera only | `scripts/legacy/object_detection.py` |
| [Place a banana in a box](pick_and_place/README.md) | Original interactive workflow with arm motion on startup | `scripts/legacy/set.py` |
| [GraspNet camera preview](graspnet_camera/README.md) | Camera and GPU | `scripts/legacy/graspnet_camera_demo.py` |
| [GraspNet robot grasping](graspnet_grasp/README.md) | Camera, GPU, and arm motion on startup | `scripts/legacy/grasp.py` |
