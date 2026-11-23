# reBot Arm B601 视觉抓取

[English](README.md)

用 RGB-D 相机识别物体、估计抓取位姿，并控制 reBot Arm B601 执行抓取。项目提供 OBB、过滤版 OBB、AnyGrasp 和 GraspNet 示例。下面从新装的 Ubuntu 环境开始，以 B601-RS 和 Orbbec Gemini 336 为例。AnyGrasp 还需要 NVIDIA GPU、配套 CUDA 环境和官方授权。

在 Ubuntu x86-64 上先安装 Git 和 curl。使用 AnyGrasp 或 GraspNet 时还需要 NVIDIA 驱动，运行 `nvidia-smi -L` 应能看到显卡。克隆工程后，在仓库根目录执行后续命令。

```bash
git clone https://github.com/LAN-GER/Rebot-Arm-B601-Visual-Grasping.git
cd Rebot-Arm-B601-Visual-Grasping
```

先把 `camera.type` 设为实际相机型号。默认使用 `orbbec_gemini336`，分辨率为 640×480，帧率为 30 FPS。连接多台同型号相机时，可用 `camera.serial` 指定设备。[默认配置](config/default.yaml)

两种安装方式任选一种。[脚本自动配置](#1-使用脚本自动配置环境和-sdk) · [手动配置](#2-手动配置环境和-sdk) · [脚本入口](#脚本入口)

## 1. 使用脚本自动配置环境和 SDK

先用检查脚本查看缺项，再用配置脚本安装，最后独立复检。运行这三条命令时，相机和机械臂都可以断开。

```bash
bash scripts/check_env.sh
bash scripts/setup_env.sh
bash scripts/check_env.sh
```

配置脚本会沿用已有的 Conda，缺少时将 Miniforge 安装到 `~/miniforge3`。它会创建或补齐 Python 3.10 环境 `rebot`，按配置安装相机驱动。默认安装 B601 控制 SDK、YOLOE，以及 OBB、AnyGrasp 和 GraspNet 所需的公开依赖，并编译缺失的 CUDA 算子。只需部分功能时，给配置和检查命令传入相同的参数。

| 要运行的功能 | 两个脚本通用的参数 |
| --- | --- |
| 全部抓取后端和 B601 机械臂 | 无需参数 |
| OBB 或过滤版 OBB 相机预览，支持 CPU | `--backend obb` |
| AnyGrasp 或 GraspNet 相机预览 | `--backend anygrasp` 或 `--backend graspnet` |
| 单个后端加 B601 机械臂 | 追加 `--robot`，例如 `--backend anygrasp --robot` |

缺少的包或 SDK 会自动安装。已有包（含间接依赖）需要升级、降级或更换 Conda 构建时，终端会列出原版本和目标版本，输入 `y` 才执行该批变更。回车、回答 `n` 或在非交互环境运行都会跳过需要确认的批次。已有 Git SDK 会继续复用。构建明确要求切换 SDK 分支时，脚本会显示两个提交号并询问。结束时，`----` 下方列出本次实际发生的变更。

检查脚本在最后的 `----` 下列出当前所选后端已就绪和缺失或不可用的包、SDK 与运行文件。需要手动提供的文件和因前置条件不足而未检查的项目会单独列出。

新机器首次检查通常会报告缺项。`[MANUAL]` 表示需要自行提供的文件，例如 AnyGrasp 本机授权、配套检测权重和手眼标定矩阵。退出码 `3` 表示仅剩手动项，补齐后用相同参数重新检查。如果 GraspNet 下载受限，可给配置脚本加 `--checkpoint-file /path/to/checkpoint-rs.tar` 使用本地权重，或用 `GRASPNET_CHECKPOINT_URL` 指定可信直链。下载中断后可以重跑配置脚本。[AnyGrasp 授权与文件位置](docs/sdk_setup_zh.md) · [手眼标定](examples/hand_eye/README_zh.md)

使用 AnyGrasp 时，运行 `bash scripts/anygrasp_license.sh`。选择 `1` 读取本机机器码，选择 `2` 并输入交付目录即可同时配置授权和检测权重。[AnyGrasp 授权和检测权重](#anygrasp-授权和检测权重)

如果配置脚本刚安装了 Miniforge，先在当前终端执行 `source ~/miniforge3/etc/profile.d/conda.sh`。运行示例前执行 `conda activate rebot`，然后进入[第 3 节](#3-运行示例)。

## 2. 手动配置环境和 SDK

先配置 Python 和相机，再按要运行的示例选择对应 SDK。OBB 算法已包含在工程中，完成 2.1 和 2.2 即可运行相机预览。

### 2.1 创建 `rebot` 环境并选择 PyTorch 版本

在可使用 Conda 的终端中，用工程环境文件创建 `rebot`。该文件包含 Python 3.10、低于 2 的 NumPy 和 B601 所需的 Pinocchio 3.9.0。如果已经有 Python 3.10 的 `rebot` 环境，直接激活并在其中补齐依赖。[环境文件](environment.yml) · [Miniforge 安装说明](https://github.com/conda-forge/miniforge#install)

```bash
conda env create -f environment.yml
conda activate rebot
```

**AnyGrasp 或 GraspNet 使用 NVIDIA GPU。** 本机 RTX 4080 Laptop GPU 已验证的组合为 PyTorch 2.5.1 和 CUDA 12.1。下面的检查应输出 `12.1` 和 `True`。

```bash
python -m pip install 'torch==2.5.1+cu121' 'torchvision==0.20.1+cu121' 'torchaudio==2.5.1+cu121' \
  --index-url https://download.pytorch.org/whl/cu121
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

**OBB 使用 CPU。** 仅配置 CPU 环境时，选择下面这条 PyTorch 安装命令。

```bash
python -m pip install -r requirements-pytorch.txt --index-url https://download.pytorch.org/whl/cpu
```

### 2.2 安装相机驱动和 YOLOE

按 `camera.type` 选择驱动。Orbbec Gemini 336 执行 `python -m pip install 'pyorbbecsdk2==2.0.18'`。RealSense D405 或 D435i 执行 `python -m pip install 'pyrealsense2>=2.54'`。按相机厂商说明配置 USB 访问权限。[相机 SDK 说明](docs/sdk_setup_zh.md)

安装工程并下载默认 YOLOE 分割权重和文本编码器。分割权重放在 `models/`，文本编码器放在仓库根目录。

```bash
python -m pip install -e '.[obb,yolo,visualization]' scipy
python -m pip install 'git+https://github.com/ultralytics/CLIP.git@7ffa84b3bfa40c42ecc1c77147a855e69cb2dd40'
mkdir -p models
curl -fL --retry 3 -o models/yoloe-26s-seg.pt \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26s-seg.pt
curl -fL --retry 3 -o mobileclip2_b.ts \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/mobileclip2_b.ts
```

### 2.3 安装手眼标定和真机抓取所需的 B601 控制 SDK

```bash
git clone https://github.com/Seeed-Projects/reBotArm_control_py.git sdk/reBotArm_control_py
python -m pip install 'motorbridge==0.6.0' 'meshcat>=0.3.2' 'matplotlib>=3.10'
```

如果创建 `rebot` 时未使用工程环境文件，可执行 `conda install --override-channels -c conda-forge 'pinocchio=3.9.0' 'numpy<2'` 安装运动学依赖。使用 B601-RS 时，上机前在 `sdk/reBotArm_control_py/config/rebotarm.yaml` 中确认 `hardware_yaml: rebotarm_rs.yaml`，并核对 CAN 接口和 URDF。可选的稳定采样标定示例还需要独立的采集器和带载模型。[B601 与 CAN 配置](docs/sdk_setup_zh.md) · [稳定采样标定](examples/stationary_calibration/README_zh.md)

### 2.4 配置 AnyGrasp 和 GraspNet 共用的 CUDA 编译工具

使用与 CUDA 12.1 兼容的 C++ 编译器。需要安装时，可执行 `conda install --override-channels -c conda-forge 'gcc_linux-64=12' 'gxx_linux-64=12'`，然后重新激活 `rebot`。已有 CUDA 12.1 Toolkit 时，把 `CUDA_HOME` 设为其安装目录。也可以用下面的命令把 Toolkit 安装到 `rebot`。

```bash
conda install --override-channels -c nvidia/label/cuda-12.1.1 -c conda-forge 'cuda-toolkit=12.1'
export CUDA_HOME="$CONDA_PREFIX"
```

在同一终端中继续编译后端。确认 `nvcc` 显示 `release 12.1` 后，安装共用的 Python 依赖。

```bash
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=2
nvcc --version
python -m pip install wheel ninja
python -m pip install 'transforms3d>=0.4.2' \
  'graspnetAPI @ git+https://github.com/graspnet/graspnetAPI.git@eb57dd2092d8dbe05312a29c3d0c22f3226efbfc'
```

### 2.5 安装 AnyGrasp 并申请授权

将 AnyGrasp 使用的 PointNet2 编译到 `rebot`，用 OpenBLAS 编译 MinkowskiEngine，然后在实际运行推理的机器上读取特征码。

```bash
git clone https://github.com/graspnet/anygrasp_sdk.git sdk/anygrasp_sdk
git clone --branch cuda-12-1 https://github.com/chenxi-wang/MinkowskiEngine.git sdk/MinkowskiEngine-cuda-12-1
conda install --override-channels -c conda-forge 'openblas>=0.3.30,<0.4' 'numpy<2'
python -c 'import ctypes, sys; ctypes.CDLL(sys.prefix + "/lib/libopenblas.so")'
python -m pip install --no-build-isolation --no-deps ./sdk/anygrasp_sdk/pointnet2
(
  cd sdk/MinkowskiEngine-cuda-12-1 &&
  python setup.py bdist_wheel --force_cuda --blas=openblas \
    --blas_include_dirs="$CONDA_PREFIX/include" --blas_library_dirs="$CONDA_PREFIX/lib" &&
  python -m pip install --no-deps dist/*.whl
)
```

#### AnyGrasp 授权和检测权重

1. **读取本机机器码。** 在实际运行推理的机器上执行命令，并使用部署时的网卡。也可以启动交互菜单并选择 `1`。用输出的 `feature_id` 申请授权和检测权重。[AnyGrasp 官方授权申请](https://github.com/graspnet/anygrasp_sdk/tree/main/license_registration)

   ```bash
   bash scripts/anygrasp_license.sh 1
   ```

2. **从同一交付目录安装授权和权重。** 将官方授权 ZIP 或解压后的授权目录与 `checkpoint_detection.tar` 放在一起。运行下方命令，或在交互菜单选择 `2` 后输入该目录路径。

   ```text
   delivery/
   ├── license.zip
   └── checkpoint_detection.tar
   ```

   ```bash
   bash scripts/anygrasp_license.sh 2 "/path/to/delivery"
   ```

   脚本通过官方 SDK 校验授权，并检查检测权重能否读取，再根据 `config/backends/anygrasp.yaml` 安装。默认授权目录为 `sdk/anygrasp_sdk/grasp_detection/license/`，权重路径为 `sdk/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar`。如果过滤版使用独立的 SDK 或权重路径，在 `2` 之前加 `--backend anygrasp_filtered`。

3. **复核配置结果。** 终端在 `----` 下显示授权和权重各自的状态。只提供授权且权重仍缺失时，脚本显示 `[PARTIAL]` 并返回退出码 `3`。拿到权重后，可直接把 `checkpoint_detection.tar` 的路径传给选项 `2`。如果出现 `[ERROR]`，可查看具体原因，例如机器码不匹配或权重无法读取。

   ```bash
   bash scripts/check_env.sh --backend anygrasp
   ```

   如需验证模型加载和真实 RGB-D 推理，运行相机预览或回放录制帧。[过滤版 AnyGrasp 预览与回放](examples/anygrasp_filtered/README_zh.md)

### 2.6 安装 GraspNet-Baseline

将 Baseline 的 PointNet2 和 kNN 扩展安装到它自己的 SDK 目录，使 AnyGrasp 可以在同一环境中使用自己的 PointNet2。

```bash
git clone https://github.com/graspnet/graspnet-baseline.git sdk/graspnet-baseline
baseline="$PWD/sdk/graspnet-baseline"
python -m pip install --no-build-isolation --no-deps --upgrade --target "$baseline/pointnet2" "$baseline/pointnet2"
touch "$baseline/pointnet2/pointnet2/__init__.py"
python -m pip install --no-build-isolation --no-deps --upgrade --target "$baseline/knn" "$baseline/knn"
mkdir -p "$baseline/checkpoints"
```

下载官方 **RealSense** 预训练权重，保存为 `sdk/graspnet-baseline/checkpoints/checkpoint-rs.tar`。如需更换权重，在后端 YAML 中修改 `checkpoint`，相对路径以该文件所在目录为准。运行 `bash scripts/check_env.sh --backend graspnet` 检查相机预览所需的依赖和 CUDA 算子。[官方预训练权重](https://github.com/graspnet/graspnet-baseline#training-and-testing) · [GraspNet 配置](config/backends/graspnet.yaml)

手动配置完整工程后，运行 `bash scripts/check_env.sh`。如果只安装了单个后端，使用 `--backend obb`、`--backend anygrasp` 或 `--backend graspnet`。追加 `--robot` 可一起检查机械臂驱动和手眼矩阵。检查全部通过时退出码为 `0`。

## 3. 运行示例

### 离线验证

只用 CPU 就能跑通合成场景。第一条命令生成模拟 RGB-D 和 OBB 候选，第二条回放同一帧。运行成功后可在输出目录查看 `result.json` 和预览图。

```bash
run_dir="outputs/setup-$(date +%s%N)"
rebot-grasp demo --output "$run_dir/demo"
rebot-grasp preview --input "$run_dir/demo/frame.npz" --output "$run_dir/replay"
```

### 3.1 检查相机和抓取预览

先录制一帧确认 RGB-D 对齐，再打开过滤版 OBB 预览。预览只用相机，在窗口按 **G** 保存当前帧的抓取候选。结果中可对照原始与过滤后的抓取位置。[过滤版 OBB 说明](docs/obb_filtered_zh.md)

```bash
rebot-grasp record --output "outputs/rgbd/frame-$(date +%s%N).npz"
bash examples/obb_filtered/run.sh
```

### 3.2 手眼标定

用相机检查 ArUco 标记后，再做手眼标定。项目默认使用 **100 mm、ID 0** 的板。打印后实测黑色标记边长，填写 `calibration.aruco.marker_length_m`，单位为米。第一条命令只开相机，第二条会启动机械臂沿标定轨迹运动，运行前请清空运动区域。[标定步骤与打印板](examples/hand_eye/README_zh.md)

```bash
bash examples/hand_eye/run.sh --check-marker
bash examples/hand_eye/run.sh
```

用 Gemini 336 求解成功后得到 `config/calibration/orbbec_gemini336/hand_eye.npz`。更换相机或安装位置后重新标定，并将 `calibration.hand_eye_path` 指向与当前设备对应的矩阵。

### 3.3 执行 OBB 或 AnyGrasp 抓取

**过滤版 OBB。** 预览位置和标定结果正确后，加 `--execute` 启动。机械臂会先进入准备位，按 **G** 重新采集并抓取，按 **Q** 或 **Esc** 结束。

```bash
bash examples/obb_filtered/run.sh --execute
```

**过滤版 AnyGrasp。** 第一条命令只开相机和模型，左侧显示原始点云对应的抓取候选，右侧显示过滤后的候选。第二条命令进入准备位，由 **G** 触发一轮抓取，执行当前类别在右侧通过工作范围和 IK 检查的候选。

```bash
bash examples/anygrasp_filtered/run.sh
bash examples/anygrasp_filtered/run.sh --execute --target-class "yellow banana"
```

图像窗口里 **1** 选笔、**2** 选橙子、**3** 选香蕉，顺序由配置里的 `yolo.custom_classes` 决定。绿色框和白色抓取线对应当前目标，蓝色框和黄色抓取线对应其余目标。红点表示两指之间的抓取中心，`Z` 是该点在相机坐标系中的深度。目标缺失、深度被拒或无可达候选时跳过抓取。按 **Q** 或 **Esc** 结束。[过滤版 AnyGrasp 说明](examples/anygrasp_filtered/README_zh.md)

**如果夹取深度不足**，可以在对应的过滤版执行命令末尾加 `--insertion-depth-mm <毫米数>`，把 `<毫米数>` 换成所需数值。它设置本轮沿工具接近方向的额外下探总量，并覆盖后端配置。当前过滤版 OBB 默认 0 mm，过滤版 AnyGrasp 默认 5 mm。[过滤版 OBB 配置](config/backends/obb_filtered.yaml) · [过滤版 AnyGrasp 配置](config/backends/anygrasp_filtered.yaml)

普通版 AnyGrasp、单帧采集和回放有独立启动入口。[全部示例与参数](examples/README_zh.md)

### 3.4 使用 GraspNet-Baseline

**相机预览。** 启动后在图像窗口按 **G** 或**空格**生成抓取候选，在点云窗口查看结果。按 **R** 恢复实时画面，按 **Q** 或 **Esc** 退出。[GraspNet 相机示例](examples/graspnet_camera/README_zh.md)

```bash
bash examples/graspnet_camera/run.sh
```

**真机抓取。** 先完成当前相机的手眼标定，并用相机预览检查候选。真机示例读取 `config/calibration/<相机型号>/hand_eye.npz`。启动时机械臂会进入准备位，使用 `--dry-run` 也会移动。按 **G** 或**空格**推理并执行可达候选，按 **Q** 或 **Esc** 结束。用 `--target-class` 指定 YOLOE 词表中的目标类别。[GraspNet 真机示例](examples/graspnet_grasp/README_zh.md)

```bash
bash examples/graspnet_grasp/run.sh --target-class "yellow banana"
```

## 4. 修改配置和查看结果

| 要改什么 | 在哪里改 |
| --- | --- |
| 相机型号、分辨率、标定参数、准备位、夹爪力 | [config/default.yaml](config/default.yaml) |
| 初始目标、词表、检测置信度、机器人工作范围、TCP 换算 | [AnyGrasp](config/anygrasp.yaml) · [过滤版 AnyGrasp](config/anygrasp_filtered.yaml) · [过滤版 OBB](config/obb_filtered.yaml) |
| 额外下探、夹爪遮罩、相机坐标系工作区域、点云质量阈值 | [后端配置](config/backends/)，额外下探键名为 `insertion_depth_m`。过滤版可用 `--insertion-depth-mm` 覆盖本轮设置 |

实时示例按 **G** 后将结果保存到 `outputs/` 下本轮的 `snapshot-NNN/`。启动器会打印运行目录。手动加 `--output` 时使用尚不存在的目录。

- `frame.npz` 保存 RGB-D 帧，可供离线回放。
- `result.json` 保存目标、点云质量、候选和计划。过滤版额外保存 `comparison.png` 和 `comparison.json`。
- `execution.json` 保存已发起的真机执行结果。控制器用 `object_held` 报告夹持状态。

## 5. 工程结构

```text
Rebot-Arm-B601-Visual-Grasping/
├── environment.yml               # Conda 环境 rebot
├── pyproject.toml                # Python 包、可选依赖和 CLI 入口
├── config/
│   ├── default.yaml              # 相机、机械臂和通用参数
│   ├── backends/                 # OBB、AnyGrasp、GraspNet 后端参数
│   └── calibration/              # 各相机的标定文件存放位置
├── src/rebot_grasp/
│   ├── cli.py                    # 解析命令行并分派任务
│   ├── bootstrap.py              # 组装相机、后端和机械臂
│   ├── config.py                 # 读取并校验配置
│   ├── core/                     # 帧、目标、候选、执行计划和接口
│   ├── cameras/                  # Orbbec、RealSense、录制回放
│   ├── perception/               # YOLO 分割、目标选择、点云处理
│   ├── grasping/                 # 算法适配、候选过滤与排序
│   ├── geometry/                 # 坐标变换和 TCP 换算
│   ├── calibration/              # ArUco 和手眼求解
│   ├── robots/                   # B601、夹爪和动作控制
│   ├── pipeline/                 # 采集、规划、执行和实时会话
│   └── visualization/            # 图像叠加、点云窗口和诊断输出
├── examples/                     # 每个示例一个目录，含 run.sh 和说明
├── scripts/                      # 环境工具与公共 Python 启动器
│   ├── check_env.sh              # 检查依赖包、SDK 和所需文件
│   ├── setup_env.sh              # 补齐依赖和 SDK
│   ├── anygrasp_license.sh       # 读取机器码或配置 AnyGrasp
│   ├── run_python.sh            # 使用工程环境启动 Python
│   ├── env/                     # 环境与 SDK 内部辅助脚本
│   └── legacy/                  # 旧示例命令的兼容入口
├── docs/                         # SDK 配置、坐标约定和各模块说明
├── sdk/                          # 本地第三方 SDK 与授权
├── models/                       # 本地分割权重
└── outputs/                      # 本地运行结果和录制
```

相机驱动输出统一的 RGB-D 帧。检测层提取目标，抓取后端生成相机坐标系下的候选，规划层用手眼矩阵和 TCP 变换生成机器人基座坐标系下的抓取计划，执行层再调用 B601 的控制接口。[模块与坐标约定](docs/architecture_zh.md)

### 脚本入口

从工程根目录执行以下命令。

| 用途 | 命令 |
| --- | --- |
| 检查环境 | `bash scripts/check_env.sh` |
| 补齐依赖和 SDK | `bash scripts/setup_env.sh` |
| 读取 AnyGrasp 机器码或配置授权和权重 | `bash scripts/anygrasp_license.sh` |
| 查看工程 CLI 参数 | `bash scripts/run_python.sh -m rebot_grasp --help` |
| 启动过滤版 OBB 相机预览 | `bash examples/obb_filtered/run.sh` |
| 启动过滤版 AnyGrasp 相机预览 | `bash examples/anygrasp_filtered/run.sh` |

其他示例使用 `bash examples/<示例目录>/run.sh` 启动。支持 `--help` 的示例可追加该参数查看用法。[示例索引](examples/README_zh.md)

旧命令中的 `scripts/<文件名>` 改为 `scripts/legacy/<文件名>`，例如 `bash scripts/legacy/anygrasp.sh` 或 `python scripts/legacy/main.py`。只配置或检查 GraspNet 时，给顶层环境脚本加 `--backend graspnet`。

## 6. 手眼标定原理与更换夹爪

### 手眼标定怎么求相机位姿

B601-RS 的 SDK 根据 URDF 和实测关节角，计算末端坐标系 `gripper_end` 相对于机器人基座的位姿。眼在手上的标定让相机随机械臂运动，ArUco 标定板固定在原处。每次采样配对两组数据，一组是末端在基座坐标系中的位姿，另一组是标定板在相机坐标系中的位姿。后者由相机内参、识别到的标记角点和标记的实测边长求得。

工程用 OpenCV 的 `calibrateHandEye` 处理多个机械臂姿态的配对数据，求出一个固定的相机到末端变换，使各组数据算出的标定板基座位姿尽量一致。

```text
T_base_board = T_base_tcp @ T_tcp_camera @ T_camera_board
```

工程把 `T_tcp_camera` 保存到 `hand_eye.npz` 的 `T_result` 字段。旋转部分表示相机光学坐标系的朝向。最后一列的前三项是相机光学原点在 `gripper_end` 中的位置，单位为米。对这三个分量取向量长度，就是两坐标系原点间的距离。[标定实现](src/rebot_grasp/calibration/hand_eye.py) · [标定文件位置](config/calibration/README_zh.md)

### 更换夹爪后怎么改

- 如果相机和 `gripper_end` 的相对位置及朝向保持一致，继续用原来的 `hand_eye.npz`，更新抓取位姿到 TCP 的换算。
- 如果相机相对 `gripper_end` 移动或转动，重新做手眼标定。如果在 URDF 里重新定义了 `gripper_end`，可以用已知的新旧坐标系变换换算原标定矩阵。变换未知时重新标定。[手眼标定步骤](examples/hand_eye/README_zh.md)

按以下步骤测量新夹爪的偏移：

1. 把夹爪张到常用的抓取宽度。在两侧爪片各选一个实际接触物体的位置，用两点的中点作为夹持中心。
2. 在 URDF 或 CAD 中确认 `gripper_end` 的原点及轴向。如果原点是虚拟点，选一个模型中已知位置且方便测量的安装基准，再从该基准测到新夹持中心。把结果换算为沿 `gripper_end` 各轴的有符号偏移。抓取坐标系的 +X 指向接近方向，+Y 沿夹爪开合方向。
3. 将实测的夹持坐标系写成在 `gripper_end` 下的旋转 `R` 和平移 `p`，平移单位用米。填入 `planning.T_grasp_tcp` 时取逆，旋转填 `Rᵀ`，平移填 `−Rᵀp`。工程用 `T_base_tcp = T_base_grasp @ T_grasp_tcp` 算出机械臂应该到达的末端位姿。

例如，两坐标系的轴向一致，实测夹持中心沿 X 偏移 **+10 mm**、沿 Y 偏移 **+2 mm**、沿 Z 偏移 **0 mm**，就按下面填写。替换实际运行配置中 `planning` 下已有的 `T_grasp_tcp`。[AnyGrasp](config/anygrasp.yaml) · [过滤版 AnyGrasp](config/anygrasp_filtered.yaml) · [过滤版 OBB](config/obb_filtered.yaml)

```yaml
planning:
  T_grasp_tcp:
    - [1, 0, 0, -0.010]
    - [0, 1, 0, -0.002]
    - [0, 0, 1,  0.000]
    - [0, 0, 0,  1.000]
```
