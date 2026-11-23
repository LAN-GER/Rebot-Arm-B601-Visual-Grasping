# SDK setup and diagnostics

[简体中文](sdk_setup_zh.md)

From the repository root, run `bash scripts/check_env.sh`, use `bash scripts/setup_env.sh` to install missing dependencies, then run `conda activate rebot`. The sections below cover manual setup and troubleshooting by component. The camera driver, SDKs, weights, and hand–eye matrix depend on the device configuration. [New-user setup](../README.md)

## Cameras

Use `pyorbbecsdk2` for the Gemini 336 and `pyrealsense2` for RealSense. Configure Linux USB access as described by the manufacturer, then record one frame to check color and depth. The `camera` section of `config/default.yaml` selects the model, serial number, and resolution. [Orbbec Python SDK](https://github.com/orbbec/pyorbbecsdk/tree/v2-main) · [RealSense SDK](https://github.com/IntelRealSense/librealsense)

## reBot Arm B601

Clone the official control SDK to its default path and install kinematics and communication dependencies in `rebot`. If you created the environment from `environment.yml`, skip the `conda install` step. [reBotArm control SDK](https://github.com/Seeed-Projects/reBotArm_control_py)

```bash
git clone https://github.com/Seeed-Projects/reBotArm_control_py.git sdk/reBotArm_control_py
conda install -c conda-forge 'pinocchio=3.9.0' 'numpy<2'
python -m pip install 'motorbridge==0.6.0' 'meshcat>=0.3.2' 'matplotlib>=3.10'
python -c 'import pinocchio, motorbridge; print(pinocchio.__version__)'
```

- For the B601-RS, set `hardware_yaml: rebotarm_rs.yaml` in `sdk/reBotArm_control_py/config/rebotarm.yaml`. Confirm the CAN channel, joint and gripper settings, and kinematic model.
- Configure the CAN interface as described by the control SDK. Use `ip -details link show can0` to check it. Substitute the interface name if your hardware configuration uses a different channel. [B601-RS hardware and CAN settings](https://wiki.seeedstudio.com/cn/rebot_arm_b601_rs_pinocchio_meshcat/)
- Review `robot.ready_pose`, `robot.gripper`, and the hand–eye path in `config/default.yaml`. Verify the calibration board with `--check-marker` before automatic calibration or robot execution. [Hand–eye calibration](../examples/hand_eye/README.md)

## AnyGrasp

### Environment and extensions

Clone the SDK to the default path. The adapter uses `create_detector` and `get_grasp`. Match the gsnet binary to the Python ABI and system architecture. For Python 3.10 on Ubuntu x86-64, use the corresponding `cpython-310` binary. [AnyGrasp SDK setup](https://github.com/graspnet/anygrasp_sdk#installation)

```bash
git clone https://github.com/graspnet/anygrasp_sdk.git sdk/anygrasp_sdk
nvidia-smi
```

Install CUDA-enabled PyTorch, the matching CUDA Toolkit, MinkowskiEngine, PointNet2, and graspnetAPI in `rebot`. Match the toolkit for building native extensions to PyTorch's CUDA build. Python 3.10, PyTorch 2.5.1, and CUDA 12.1 were tested together. The following commands apply when your driver and GPU support that build. [PyTorch CUDA build table](https://pytorch.org/get-started/previous-versions/)

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

### License and detection weights

Read the feature ID on the deployment machine using its intended network adapter. Apply for the official license and detection checkpoint using that ID. [Official registration](https://github.com/graspnet/anygrasp_sdk/tree/main/license_registration)

```bash
bash scripts/anygrasp_license.sh 1
```

Pass the delivery directory to option `2`. Keep either the issued license ZIP or extracted license folder alongside `checkpoint_detection.tar`. The script validates both artifacts and prints their status at the end. If replacement is needed, it prints the previous license directory's backup path.

```bash
bash scripts/anygrasp_license.sh 2 "/path/to/delivery"
```

The default destinations are shown below. Change `checkpoint` in the relevant `config/backends/anygrasp*.yaml` file to use another weight path. Relative paths start at the directory containing the backend YAML. If only the license is installed and weights are still missing, the script reports `[PARTIAL]` and returns exit code `3`.

```text
sdk/anygrasp_sdk/grasp_detection/
├── license/
│   └── licenseCfg.json
└── log/
    └── checkpoint_detection.tar
```

The environment check verifies the license files, model checkpoint, and CUDA operators. Camera preview or offline replay verifies model initialization and inference. For the replay command, substitute a real recorded `frame.npz`.

```bash
bash scripts/check_env.sh --backend anygrasp
bash examples/anygrasp_filtered/run.sh
bash examples/anygrasp_filtered/run.sh --input /path/to/snapshot-001/frame.npz
```

## GraspNet and calibration with stationary samples

- Put GraspNet-Baseline under `sdk/graspnet-baseline`. Build the extensions as described upstream and set the checkpoint in `config/backends/graspnet.yaml`. [GraspNet-Baseline](https://github.com/graspnet/graspnet-baseline)
- Hand–eye calibration with stationary samples uses the collector and payload model from the sibling `rebot-arm-b601-robstride` checkout. [Calibration example](../examples/stationary_calibration/README.md)

## Python launcher

After `conda activate rebot`, `scripts/run_python.sh` uses the active environment's interpreter and runs examples from the repository root. Set `REBOT_PYTHON=/absolute/path/to/python` to select another interpreter.
