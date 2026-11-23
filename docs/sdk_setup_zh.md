# SDK 安装与自检

[English](sdk_setup.md)

新用户在仓库根目录先运行 `bash scripts/check_env.sh`，再运行 `bash scripts/setup_env.sh` 补齐工程环境，最后执行 `conda activate rebot`。需要按组件配置或排查时，可参考下列手动步骤。使用的相机驱动、SDK、权重和手眼标定文件随设备配置。[新用户安装指南](../README_zh.md)

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
- 在 `config/default.yaml` 核对 `robot.ready_pose`、`robot.gripper` 和手眼文件路径。用 `--check-marker` 检查标定板后，再启动自动标定或真机抓取。[手眼标定](../examples/hand_eye/README_zh.md)

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

### AnyGrasp 授权和检测权重

在实际运行推理的机器上读取特征码，使用部署时的网卡。用输出的机器码提交官方授权申请。[官方授权步骤](https://github.com/graspnet/anygrasp_sdk/tree/main/license_registration)

```bash
bash scripts/anygrasp_license.sh 1
```

将官方授权 ZIP 或解压目录与 `checkpoint_detection.tar` 放在同一交付目录，用选项 `2` 安装。脚本会验证两个文件，并在终端末尾显示结果。需要更换授权时，终端也会显示原授权目录的备份路径。

```bash
bash scripts/anygrasp_license.sh 2 "/path/to/delivery"
```

默认安装位置如下。权重仍缺失时，脚本显示 `[PARTIAL]` 并返回退出码 `3`。如需更换权重位置，在对应的 `config/backends/anygrasp*.yaml` 中修改 `checkpoint`，相对路径从后端 YAML 所在目录解析。

```text
sdk/anygrasp_sdk/grasp_detection/
├── license/
│   └── licenseCfg.json
└── log/
    └── checkpoint_detection.tar
```

环境检查验证授权文件、检测权重和 CUDA 算子。相机预览或离线回放可验证模型加载与推理。回放时使用实际保存的 `frame.npz`。

```bash
bash scripts/check_env.sh --backend anygrasp
bash examples/anygrasp_filtered/run.sh
bash examples/anygrasp_filtered/run.sh --input /path/to/snapshot-001/frame.npz
```

## GraspNet 与稳定采样标定

- GraspNet Baseline 放在 `sdk/graspnet-baseline`，按官方说明编译扩展，权重路径写入 `config/backends/graspnet.yaml`。[GraspNet Baseline](https://github.com/graspnet/graspnet-baseline)
- 稳定采样标定示例依赖同级的 `rebot-arm-b601-robstride` 控制工程及其采集器和带载模型。[稳定采样示例](../examples/stationary_calibration/README_zh.md)

## Python 启动器

执行 `conda activate rebot` 后，`scripts/run_python.sh` 使用当前环境的 Python，并从仓库根目录运行示例。启动器会隔离继承的 Python 路径、动态库路径和用户目录中的包，避免 ROS Humble 和用户级依赖覆盖 Conda 中的依赖。需要指定其他解释器时，可设置 `REBOT_PYTHON=/absolute/path/to/python`。
