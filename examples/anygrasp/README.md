# Live AnyGrasp

[简体中文](README_zh.md)

The current tabletop vocabulary is `pen`, `orange`, and `yellow banana`. The window shows the class and confidence for each detected instance. The green box marks the current target, and blue boxes mark other detections. The default target is the banana. Use `--target-class pen` or `--target-class orange` to switch it. The preview runs AnyGrasp for each instance using its mask. White lines show current target candidates, yellow lines show other candidates, and Z gives the depth of the candidate center in camera coordinates. The window labels instances rejected by point cloud quality checks or without a valid candidate. Press **1** for pen, **2** for orange, or **3** for banana in the image window. The keys follow the configured vocabulary order. Press **G** to capture a new frame. Execution mode grasps the selected class and skips the action if the target is absent.

Set up the [AnyGrasp license and detection weights](../../README.md#anygrasp-license-and-detection-weights) before starting. Run the camera preview from the repository root. Robot execution is opt-in.

```bash
bash examples/anygrasp/run.sh --output "outputs/anygrasp-live/run-$(date +%Y%m%d-%H%M%S-%N)"
```

While waiting for **G**, the preview periodically runs AnyGrasp on new RGB-D frames. A **white line** connects the projected fingertips of the highest-scoring valid candidate, and a **red dot** marks its center. Target mode also shows a green instance box with YOLO confidence. Full-scene mode searches the scene for candidates. The window gives a reason when no candidates are available. Refresh speed depends on detection and grasp inference time. Frame age appears at the bottom. Press **G** to capture a new frame and run inference. In execution mode the planner checks the robot workspace and IK to choose a reachable candidate. Press **R** to resume live preview. With `--execute`, the robot moves to the ready pose on startup and executes a grasp when you press **G**. Results are saved by snapshot. Press **Q** or **Esc** to end the session.
