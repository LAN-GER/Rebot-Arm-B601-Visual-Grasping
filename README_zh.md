# reBot Arm B601 视觉抓取

[English](README.md)

用 RGB-D 相机识别物体、估计抓取位姿，并控制 reBot Arm B601 执行抓取。项目提供 OBB、过滤版 OBB、AnyGrasp 和 GraspNet 示例。下面从新装的 Ubuntu 环境开始，以 B601-RS 和 Orbbec Gemini 336 为例。AnyGrasp 还需要 NVIDIA GPU、配套 CUDA 环境和官方授权。

## 1. 创建环境并运行离线示例

先安装 Git 和 Conda。从终端克隆项目，创建 Python 3.10 环境 `rebot`。后续命令都在仓库根目录和已激活的 `rebot` 环境中运行。

```bash
git clone https://github.com/LAN-GER/Rebot-Arm-B601-Visual-Grasping.git
cd Rebot-Arm-B601-Visual-Grasping
conda create -n rebot python=3.10 pip
conda activate rebot
python -m pip install -e '.[obb]'
```

只用 CPU 就能跑通合成场景。第一条命令生成模拟 RGB-D 和 OBB 候选，第二条回放同一帧。运行成功后可在输出目录查看 `result.json` 和预览图。

```bash
run_dir="outputs/setup-$(date +%s%N)"
rebot-grasp demo --output "$run_dir/demo"
rebot-grasp preview --input "$run_dir/demo/frame.npz" --output "$run_dir/replay"
```

如果选择用仓库的 [environment.yml](environment.yml) 创建含运动学依赖的环境，用 `conda env create -f environment.yml` 替代上面的 `conda create`，然后同样执行 `conda activate rebot` 和项目安装命令。

## 2. 配置相机、检测模型和 SDK

| 要运行的功能 | 所需组件 |
| --- | --- |
| 相机采集与 ArUco 检查 | 对应的相机 Python SDK、USB 权限和支持 GUI 的 OpenCV |
| 实时 OBB 预览 | 相机、PyTorch、YOLOE 分割权重和文本编码器。过滤版还需要 Open3D |
| 手眼标定与真机抓取 | 相机、reBotArm 控制 SDK、Pinocchio、MotorBridge 和可用的 CAN 接口 |
| AnyGrasp 预览与抓取 | 在实时 OBB 依赖基础上加入 NVIDIA GPU、CUDA 版 PyTorch、AnyGrasp SDK、权重与本机授权 |

**相机 SDK。** Gemini 336 可用下面的 Orbbec 安装命令。RealSense 用第二条。按实际型号选择，在 [config/default.yaml](config/default.yaml) 修改 `camera.type`；Gemini 336 的预设为 640×480、30 FPS。同型号相机有多台时，可设置 `camera.serial`。

```bash
# Orbbec Gemini 336
python -m pip install 'pyorbbecsdk2==2.0.18'

# RealSense
python -m pip install -e '.[realsense]'
```

**PyTorch 与 YOLOE。** 只跑相机 OBB 时可用 CPU 版 PyTorch。准备运行 AnyGrasp 时，按 SDK 的 Python、PyTorch 和 CUDA 兼容要求，替换下面的 CPU 安装命令，再安装 YOLOE。把分割权重放在 `models/`，把文本编码器放在项目根目录。[PyTorch 版本安装表](https://pytorch.org/get-started/previous-versions/) · [Ultralytics YOLOE 文档](https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/yoloe.md)

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

**机械臂和 AnyGrasp SDK。** 按下列文档分别配置控制 SDK 和 AnyGrasp。先确认 RS/DM 型号、CAN 通道和机械臂的 URDF，再运行涉及运动的示例。AnyGrasp 用户在目标机器上申请授权，并把授权及检测权重放入 SDK 对应目录。[SDK 安装与自检](docs/sdk_setup.md)

## 3. 上机操作

### 3.1 检查相机和抓取预览

先录制一帧确认 RGB-D 对齐，再打开过滤版 OBB 预览。预览只用相机，在窗口按 **G** 保存当前帧的抓取候选。结果中可对照原始与过滤后的抓取位置。[过滤版 OBB 说明](docs/obb_filtered.md)

```bash
rebot-grasp record --output "outputs/rgbd/frame-$(date +%s%N).npz"
bash examples/obb_filtered/run.sh
```

### 3.2 手眼标定

用相机检查 ArUco 标记后，再做手眼标定。项目默认使用 **100 mm、ID 0** 的板。打印后实测黑色标记边长，填写 `calibration.aruco.marker_length_m`，单位为米。第一条命令只开相机，第二条会启动机械臂沿标定轨迹运动，运行前请清空运动区域。[标定步骤与打印板](examples/hand_eye/README.md)

```bash
bash examples/hand_eye/run.sh --check-marker
bash examples/hand_eye/run.sh
```

用 Gemini 336 求解成功后得到 `config/calibration/orbbec_gemini336/hand_eye.npz`。更换相机或安装位置后重新标定，并将 `calibration.hand_eye_path` 指向与当前设备对应的矩阵。

### 3.3 执行 OBB 或 AnyGrasp 抓取

**过滤版 OBB。** 预览位置和标定结果正确后，加 `--execute` 启动。机械臂会先进入准备位，按 **G** 重新采集并抓取，按 **Q** 或 **Esc** 结束。下例使用 5 mm 额外下探，沿工具接近方向应用。

```bash
bash examples/obb_filtered/run.sh --execute --insertion-depth-mm 5
```

**过滤版 AnyGrasp。** 第一条命令只开相机和模型，左侧显示原始点云对应的抓取候选，右侧显示过滤后的候选。第二条命令进入准备位，由 **G** 触发一轮抓取，执行当前类别在右侧通过工作范围和 IK 检查的候选。

```bash
bash examples/anygrasp_filtered/run.sh
bash examples/anygrasp_filtered/run.sh --execute --target-class "yellow banana" --insertion-depth-mm 5
```

图像窗口里 **1** 选笔、**2** 选橙子、**3** 选香蕉，顺序由配置里的 `yolo.custom_classes` 决定。绿色框和白色抓取线对应当前目标，蓝色框和黄色抓取线对应其余目标。红点表示两指之间的抓取中心，`Z` 是该点在相机坐标系中的深度。目标缺失、深度被拒或无可达候选时跳过抓取。按 **Q** 或 **Esc** 结束。[过滤版 AnyGrasp 说明](examples/anygrasp_filtered/README.md)

普通版 AnyGrasp、GraspNet、单帧采集和回放有独立启动入口。[全部示例与参数](examples/README.md)

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
├── scripts/                      # Python 启动器和旧命令入口
├── tests/                        # 离线测试
├── docs/                         # SDK 配置、坐标约定和各模块说明
├── sdk/                          # 本地第三方 SDK 与授权
├── models/                       # 本地分割权重
└── outputs/                      # 本地运行结果和录制
```

相机驱动输出统一的 RGB-D 帧。检测层提取目标，抓取后端生成相机坐标系下的候选，规划层用手眼矩阵和 TCP 变换生成机器人基座坐标系下的抓取计划，执行层再调用 B601 的控制接口。[模块与坐标约定](docs/architecture.md)
