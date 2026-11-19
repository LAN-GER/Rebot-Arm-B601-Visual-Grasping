# reBot Arm B601 Visual Grasping

[简体中文](README_zh.md)

Use an RGB-D camera to identify objects, estimate grasp poses, and pick objects with a reBot Arm B601. The project includes OBB, filtered OBB, AnyGrasp, and GraspNet examples. This guide starts with a fresh Ubuntu environment and uses the B601-RS and Orbbec Gemini 336 as the reference setup. AnyGrasp also requires an NVIDIA GPU, a compatible CUDA environment, and an official license.

## 1. Create the environment and run the offline demo

Install Git and Conda, then clone the repository and create a Python 3.10 environment named `rebot`. Run the remaining commands from the repository root with `rebot` active.

```bash
git clone https://github.com/LAN-GER/Rebot-Arm-B601-Visual-Grasping.git
cd Rebot-Arm-B601-Visual-Grasping
conda create -n rebot python=3.10 pip
conda activate rebot
python -m pip install -e '.[obb]'
```

The synthetic demo runs on a CPU. The first command generates an RGB-D frame and an OBB grasp candidate. The second replays the same frame. Check `result.json` and the preview image in the output directory.

```bash
run_dir="outputs/setup-$(date +%s%N)"
rebot-grasp demo --output "$run_dir/demo"
rebot-grasp preview --input "$run_dir/demo/frame.npz" --output "$run_dir/replay"
```

To create an environment that includes kinematics dependencies, use `conda env create -f environment.yml` in place of `conda create`. Then activate `rebot` and install the project as shown above. [Environment definition](environment.yml)

## 2. Set up the camera, detection model, and SDKs

| Feature | Requirements |
| --- | --- |
| RGB-D capture and ArUco checks | Camera Python SDK, USB permissions, and OpenCV with GUI support |
| Live OBB preview | Camera, PyTorch, YOLOE segmentation weights, and a text encoder. The filtered example also needs Open3D |
| Hand–eye calibration and robot execution | Camera, reBotArm control SDK, Pinocchio, MotorBridge, and a working CAN interface |
| AnyGrasp preview and execution | Live OBB dependencies, plus an NVIDIA GPU, CUDA-enabled PyTorch, the AnyGrasp SDK, weights, and a license for the machine |
| GraspNet preview and execution | Live OBB dependencies, plus an NVIDIA GPU, CUDA-enabled PyTorch, GraspNet-Baseline, its compiled operators, and a pretrained checkpoint |

**Camera SDK.** Run the command for your camera. Set `camera.type` in the default configuration. The Gemini 336 preset uses 640×480 at 30 FPS. Set `camera.serial` to select a device when several cameras of the same model are connected. [Default configuration](config/default.yaml)

```bash
# Orbbec Gemini 336
python -m pip install 'pyorbbecsdk2==2.0.18'

# RealSense
python -m pip install -e '.[realsense]'
```

**PyTorch and YOLOE.** CPU-only PyTorch is sufficient for live OBB. For AnyGrasp, use the PyTorch and CUDA versions required by your SDK in place of the CPU install command below. Install YOLOE afterward. Save the segmentation weights in `models/` and the text encoder at the repository root. [PyTorch installation commands](https://pytorch.org/get-started/previous-versions/) · [Ultralytics YOLOE documentation](https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/yoloe.md)

```bash
python -m pip install -r requirements-pytorch.txt --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[yolo,visualization]'
python -m pip install 'git+https://github.com/ultralytics/CLIP.git@7ffa84b3bfa40c42ecc1c77147a855e69cb2dd40'
mkdir -p models
curl -fL -o models/yoloe-26s-seg.pt \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26s-seg.pt
curl -fL -o mobileclip2_b.ts \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/mobileclip2_b.ts
```

**Robot and AnyGrasp SDKs.** Check the robot variant (RS or DM), CAN channel, and URDF before running an example that moves the arm. Use the target machine's feature ID to request an AnyGrasp license. Place the license and detection checkpoint in the SDK directories listed in the setup guide. [SDK setup and checks (Chinese)](docs/sdk_setup.md)

## 3. Run on hardware

### 3.1 Check the camera and preview grasps

Record a frame to check RGB-to-depth alignment, then open the filtered OBB preview. The preview uses only the camera. Press **G** in the window to save the frame and its grasp candidates. The output shows the raw and filtered grasp estimates side by side. [Filtered OBB guide (Chinese)](docs/obb_filtered.md)

```bash
rebot-grasp record --output "outputs/rgbd/frame-$(date +%s%N).npz"
bash examples/obb_filtered/run.sh
```

### 3.2 Run hand–eye calibration

Check the ArUco marker before starting calibration. The default board uses a **100 mm marker with ID 0**. Measure the black marker edge after printing and enter its length in meters under `calibration.aruco.marker_length_m`. The first command opens the camera for a marker check. Clear the arm's travel area before running the second command, which moves it through the calibration trajectory. [Calibration instructions and printable boards (Chinese)](examples/hand_eye/README.md)

```bash
bash examples/hand_eye/run.sh --check-marker
bash examples/hand_eye/run.sh
```

For the Gemini 336, a successful calibration writes `config/calibration/orbbec_gemini336/hand_eye.npz`. Recalibrate if you change the camera or its mounting position. Point `calibration.hand_eye_path` to the matrix for the current setup.

### 3.3 Execute an OBB or AnyGrasp pick

**Filtered OBB.** Once the preview and calibration are correct, run with `--execute`. The arm first moves to its ready pose. Press **G** to capture a fresh frame and attempt a pick. Press **Q** or **Esc** to end the session.

```bash
bash examples/obb_filtered/run.sh --execute
```

**Filtered AnyGrasp.** The first command opens the camera and loads the model. The left pane displays grasp candidates from the raw point cloud and the right pane displays candidates from the filtered cloud. The second command moves the arm to the ready pose. Press **G** to attempt a pick on the selected class using a candidate from the right pane that passes workspace and inverse kinematics checks.

```bash
bash examples/anygrasp_filtered/run.sh
bash examples/anygrasp_filtered/run.sh --execute --target-class "yellow banana"
```

In the image window, **1** selects the pen, **2** selects the orange, and **3** selects the banana. The order follows `yolo.custom_classes` in the configuration. A green box and white grasp line mark the selected target. Blue boxes and yellow lines mark other targets. The red dot is the grasp center between the jaws, and `Z` is its depth in camera coordinates. The arm skips the pick if the target is missing, the depth check fails, or no candidate is reachable. Press **Q** or **Esc** to end the session. [Filtered AnyGrasp guide (Chinese)](examples/anygrasp_filtered/README.md)

**If the grasp is too shallow**, append `--insertion-depth-mm <mm>` to either filtered execution command and replace `<mm>` with the desired extra travel along the tool's approach direction. This sets the total extra insertion for the run. The current backend defaults are 0 mm for filtered OBB and 5 mm for filtered AnyGrasp. [Filtered OBB settings](config/backends/obb_filtered.yaml) · [Filtered AnyGrasp settings](config/backends/anygrasp_filtered.yaml)

Standard AnyGrasp, single-frame capture, and playback have separate launch commands. [Example index and options (Chinese)](examples/README.md)

### 3.4 Use GraspNet-Baseline

Start with a working NVIDIA driver and Conda. Run the checker first. If it reports missing components, run the setup script. It creates or completes the `rebot` environment, installs the required CUDA toolkit and Python packages, builds missing PointNet2 and kNN operators, and downloads the default models. It finishes by running the checker again.

```bash
bash scripts/check_graspnet_env.sh
bash scripts/setup_graspnet_env.sh
conda activate rebot
```

The default checkpoint goes to `sdk/graspnet-baseline/checkpoints/checkpoint-rs.tar`. To use a different checkpoint, set `checkpoint` in the backend configuration. Relative paths are resolved from that YAML file. [Official pretrained weights](https://github.com/graspnet/graspnet-baseline#training-and-testing) · [GraspNet configuration](config/backends/graspnet.yaml)

If Google Drive limits the download, place the checkpoint at the configured path using one of the official download options, or set `GRASPNET_CHECKPOINT_URL` to a trusted direct download. Run setup again to finish the check.

For robot execution, run setup with `--robot` to install the B601 software as well. Run the checker with `--robot` after configuring the robot and completing hand–eye calibration. [Robot SDK and CAN setup (Chinese)](docs/sdk_setup.md)

**Camera preview.** Use the camera and YOLOE setup from section 2. Run the preview and press **G** or **Space** to generate grasp candidates and inspect them in the point cloud window. Press **R** to resume live viewing or **Q** or **Esc** to exit. [GraspNet camera example (Chinese)](examples/graspnet_camera/README.md)

```bash
bash examples/graspnet_camera/run.sh
```

**Robot execution.** First calibrate the selected camera and check the candidates in the camera preview. The execution example reads `config/calibration/<camera_type>/hand_eye.npz`. It moves the arm to its ready pose on startup, including with `--dry-run`. Press **G** or **Space** to infer and execute a reachable grasp, or **Q** or **Esc** to end the session. Pass a class from the YOLOE vocabulary with `--target-class`. [GraspNet robot example (Chinese)](examples/graspnet_grasp/README.md)

```bash
bash examples/graspnet_grasp/run.sh --target-class "yellow banana"
```

## 4. Adjust settings and inspect results

| Setting | Location |
| --- | --- |
| Camera type, resolution, calibration, ready pose, and gripper force | [config/default.yaml](config/default.yaml) |
| Initial target, vocabulary, detection confidence, robot workspace, and TCP transform | [AnyGrasp](config/anygrasp.yaml) · [Filtered AnyGrasp](config/anygrasp_filtered.yaml) · [Filtered OBB](config/obb_filtered.yaml) |
| Additional insertion depth, gripper mask, camera workspace, and point cloud quality thresholds | [Backend configuration](config/backends/). Set `insertion_depth_m` or use `--insertion-depth-mm` to override it for one filtered run |

In live examples, pressing **G** saves results to a `snapshot-NNN/` subdirectory for the current run under `outputs/`. The launcher prints the run directory. If you pass `--output`, use a path that does not yet exist.

- `frame.npz` contains the RGB-D frame for offline playback.
- `result.json` records the target, point cloud quality, candidates, and plans. Filtered examples also save `comparison.png` and `comparison.json`.
- `execution.json` records the result of an attempted robot pick. The controller reports whether it detected a held object in `object_held`.

## 5. Project structure

```text
Rebot-Arm-B601-Visual-Grasping/
├── environment.yml               # rebot Conda environment
├── pyproject.toml                # Python package, optional dependencies, and CLI
├── config/
│   ├── default.yaml              # Shared camera and robot settings
│   ├── backends/                 # OBB, AnyGrasp, and GraspNet settings
│   └── calibration/              # Per-camera calibration files
├── src/rebot_grasp/
│   ├── cli.py                    # Argument parsing and command dispatch
│   ├── bootstrap.py              # Camera, backend, and robot setup
│   ├── config.py                 # Configuration loading and validation
│   ├── core/                     # Frames, targets, grasps, plans, and interfaces
│   ├── cameras/                  # Orbbec, RealSense, and recorded frames
│   ├── perception/               # YOLO segmentation, target selection, and point clouds
│   ├── grasping/                 # Grasp backends, filtering, and ranking
│   ├── geometry/                 # Coordinate and TCP transforms
│   ├── calibration/              # ArUco detection and hand–eye calibration
│   ├── robots/                   # B601, gripper, and motion control
│   ├── pipeline/                 # Capture, planning, execution, and live sessions
│   └── visualization/            # Overlays, point cloud views, and diagnostics
├── examples/                     # Per-example launch script and instructions
├── scripts/                      # Python launcher and legacy entry points
├── tests/                        # Offline tests
├── docs/                         # SDK setup, coordinate conventions, and module guides
├── sdk/                          # Local third-party SDKs and licenses
├── models/                       # Local segmentation weights
└── outputs/                      # Local run results and recordings
```

The camera driver supplies RGB-D frames in a common format. The perception layer extracts targets, and the grasp backend proposes candidates in camera coordinates. The planner applies the hand–eye and TCP transforms to produce grasp plans in the robot base frame. The execution layer then calls the B601 control interface. [Architecture and coordinate conventions (Chinese)](docs/architecture.md)

## 6. Hand–eye calibration and gripper replacement

### How hand–eye calibration works

The B601-RS SDK uses the URDF and measured joint angles to compute the pose of its end-effector frame, `gripper_end`. In eye-in-hand calibration, the camera moves with the arm and the ArUco board stays fixed. Each sample pairs the end-effector pose in the robot base frame with the board pose in the camera frame. The camera estimates the board pose using its intrinsics, the detected marker corners, and the marker's measured edge length.

OpenCV's `calibrateHandEye` uses samples from several arm poses to solve for the camera-to-end-effector transform that makes the estimated board pose in the base frame consistent:

```text
T_base_board = T_base_tcp @ T_tcp_camera @ T_camera_board
```

The project saves `T_tcp_camera` as `T_result` in `hand_eye.npz`. Its rotation gives the orientation of the camera optical frame relative to `gripper_end`. The first three entries of the last column give the camera optical origin's position in that frame, in meters. The length of this translation vector is the distance between the two frame origins. [Calibration implementation](src/rebot_grasp/calibration/hand_eye.py) · [Calibration file locations](config/calibration/README.md)

### What to update after replacing the gripper

- If the camera and `gripper_end` keep the same relative position and orientation, keep the existing `hand_eye.npz` and update the grasp-to-TCP transform.
- If the camera moves or rotates relative to `gripper_end`, rerun hand–eye calibration. If you redefine `gripper_end` in the URDF, transform the saved matrix into the new frame using the known change of frame, or recalibrate when that transform is unknown. [Hand–eye calibration steps](examples/hand_eye/README.md)

To measure the new gripper offset:

1. Open the jaws to a representative working width. Mark where each jaw will contact the object and use the midpoint between the two marks as the grasp center.
2. Use the URDF or CAD to identify the origin and axes of `gripper_end`. For a virtual origin, pick a measurable mounting datum whose location in `gripper_end` is known, then measure from that datum to the new grasp center. Express the resulting offset along the `gripper_end` axes. The grasp frame uses +X as the approach direction and +Y along the jaw opening axis.
3. Express the measured grasp frame in `gripper_end` as rotation `R` and translation `p` in meters. Invert that transform for `planning.T_grasp_tcp`. Use `Rᵀ` for rotation and `−Rᵀp` for translation. The planner applies `T_base_tcp = T_base_grasp @ T_grasp_tcp`.

For aligned axes, a measured offset of **+10 mm in X, +2 mm in Y, and 0 mm in Z** gives the following setting. Replace the existing `T_grasp_tcp` entry under `planning` in the profile you run. [AnyGrasp](config/anygrasp.yaml) · [Filtered AnyGrasp](config/anygrasp_filtered.yaml) · [Filtered OBB](config/obb_filtered.yaml)

```yaml
planning:
  T_grasp_tcp:
    - [1, 0, 0, -0.010]
    - [0, 1, 0, -0.002]
    - [0, 0, 1,  0.000]
    - [0, 0, 0,  1.000]
```
