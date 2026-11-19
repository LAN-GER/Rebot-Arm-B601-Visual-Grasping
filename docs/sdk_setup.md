# SDK 安装与自检

先按项目 README 创建并激活 `rebot` 环境。下面的命令从仓库根目录运行。使用的相机驱动、SDK、权重和手眼标定文件随设备配置。[新用户安装指南](../README_zh.md)

## 相机

Gemini 336 使用 `pyorbbecsdk2`，RealSense 使用 `pyrealsense2`。Linux 上先按厂商说明配置相机的 USB 权限，再用项目 README 中的单帧录制命令检查彩色与深度画面。相机型号、序列号和分辨率在 `config/default.yaml` 的 `camera` 下。[Orbbec Python SDK](https://github.com/orbbec/pyorbbecsdk/tree/v2-main) · [RealSense SDK](https://github.com/IntelRealSense/librealsense)

## reBot Arm B601

把官方控制 SDK 克隆到默认路径，在当前 `rebot` 环境安装运动学和通信依赖。如果使用项目 `environment.yml` 创建环境，可以跳过下面的 `conda install`。[reBotArm 控制 SDK](https://github.com/Seeed-Projects/reBotArm_control_py)

```bash
git clone https://github.com/Seeed-Projects/reBotArm_control_py.git sdk/reBotArm_control_py
conda install -c conda-forge 'pinocchio=3.9.0' 'numpy<2'
python -m pip install 'motorbridge==0.6.0' 'meshcat>=0.3.2' 'matplotlib>=3.10'
python -c 'import pinocchio, motorbridge; print(pinocchio.__version__)'
```

- 使用 B601-RS 时，在 `sdk/reBotArm_control_py/config/rebotarm.yaml` 设置 `hardware_yaml: rebotarm_rs.yaml`，再核对对应的 CAN 通道、关节和夹爪设置，以及运动学模型。
- 按控制 SDK 的说明配置 CAN 接口，使用 `ip -details link show can0` 检查接口状态。如果硬件配置使用其他通道，替换命令中的接口名。[B601-RS 硬件配置与 CAN 设置](https://wiki.seeedstudio.com/cn/rebot_arm_b601_rs_pinocchio_meshcat/)
- 在 `config/default.yaml` 核对 `robot.ready_pose`、`robot.gripper` 和手眼文件路径。用 `--check-marker` 检查标定板后，再启动自动标定或真机抓取。[手眼标定](../examples/hand_eye/README.md)

## AnyGrasp

### 安装环境与扩展

将 SDK 克隆到项目默认路径。适配器使用提供 `create_detector` 和 `get_grasp` 的 SDK 接口。`gsnet` 二进制需与解释器 ABI 和架构匹配，Ubuntu x86-64 的 Python 3.10 使用对应的 `cpython-310` 文件。[AnyGrasp SDK 安装说明](https://github.com/graspnet/anygrasp_sdk#installation)

```bash
git clone https://github.com/graspnet/anygrasp_sdk.git sdk/anygrasp_sdk
nvidia-smi
```

在 `rebot` 环境中先安装匹配的 CUDA 版 PyTorch，再按照 SDK 说明配置 CUDA Toolkit、编译 MinkowskiEngine 和 PointNet2，以及安装 graspnetAPI。用于编译扩展的 Toolkit 要与 PyTorch 的 CUDA 构建匹配。已运行过的组合是 Python 3.10、PyTorch 2.5.1 和 CUDA 12.1。若驱动和 GPU 支持对应版本，可以从下面的命令开始。其他组合以所用 SDK 的安装说明为准。[PyTorch CUDA 版本表](https://pytorch.org/get-started/previous-versions/)

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

### 申请授权、放置权重与验证

在实际运行推理的机器上执行检查并取得特征码，按官方步骤提交申请。尽量使用部署时相同的网卡及连接方式。该检查运行小规模 CUDA 算子，并报告 SDK、授权配置与 checkpoint 是否存在。[官方授权步骤](https://github.com/graspnet/anygrasp_sdk/tree/main/license_registration)

```bash
bash scripts/run_python.sh examples/anygrasp/check_sdk.py --feature-id
```

收到文件后，按下列路径放置。授权目录中应包含 `licenseCfg.json` 及随授权提供的其余文件。若使用其他权重路径，在对应的 `config/backends/anygrasp*.yaml` 中调整 `checkpoint`，该字段相对于后端 YAML。

```text
sdk/anygrasp_sdk/grasp_detection/
├── license/
│   └── licenseCfg.json
└── log/
    └── checkpoint_detection.tar
```

打开过滤版相机预览即可验证授权、模型加载和实时推理。已有实拍帧时可以运行第二条命令离线复核。替换路径为实际保存的 `frame.npz`。

```bash
bash examples/anygrasp_filtered/run.sh
bash examples/anygrasp_filtered/run.sh --input /path/to/snapshot-001/frame.npz
```

## GraspNet 与稳定采样标定

- GraspNet Baseline 放在 `sdk/graspnet-baseline`，按官方说明编译扩展，权重路径写入 `config/backends/graspnet.yaml`。[GraspNet Baseline](https://github.com/graspnet/graspnet-baseline)
- 稳定采样标定示例依赖同级的 `rebot-arm-b601-robstride` 控制工程及其采集器和带载模型。[稳定采样示例](../examples/stationary_calibration/README.md)

## Python 启动器

执行 `conda activate rebot` 后，`scripts/run_python.sh` 使用当前环境的 Python，并从仓库根目录运行示例。需要指定其他解释器时，可设置 `REBOT_PYTHON=/absolute/path/to/python`。
