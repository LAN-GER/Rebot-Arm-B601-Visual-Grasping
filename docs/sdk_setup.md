# SDK 与运行环境

现场 Ubuntu 工作站已配置 Python 3.10 的 LJ 环境、NVIDIA RTX 4080 Laptop GPU、AnyGrasp 授权和抓取权重。香蕉抓取已在这套设备上运行。便携离线 OBB 环境的安装步骤见 [项目说明](../README.md)。

| 功能 | 依赖及安装位置 |
| --- | --- |
| B601-RS/DM | [reBotArm_control_py](https://github.com/Seeed-Projects/reBotArm_control_py) 放入 `sdk/reBotArm_control_py`，安装该 SDK 的运动学、驱动和 MotorBridge 依赖。在 SDK 的 `config/rebotarm.yaml` 选择实际硬件 |
| Orbbec / RealSense | 分别使用 `pyorbbecsdk`（安装包 pyorbbecsdk2）、`pyrealsense2`。Gemini 336 现场使用 640×480、30 fps 对齐采集 |
| YOLOE | PyTorch、Ultralytics、其 CLIP 依赖；检测权重放在 `models/`。开放词表还需匹配的文本编码器，本机使用仓库根目录的 `mobileclip2_b.ts` |
| AnyGrasp | [官方 SDK](https://github.com/graspnet/anygrasp_sdk) 放入 `sdk/anygrasp_sdk`，安装与 Python ABI、PyTorch/CUDA 匹配的 gsnet、MinkowskiEngine、PointNet2 和 graspnetAPI |
| GraspNet-Baseline | [官方仓库](https://github.com/graspnet/graspnet-baseline) 放入 `sdk/graspnet-baseline`，按其说明编译扩展，并在 `config/backends/graspnet.yaml` 指定 checkpoint。本机未安装这套基线模型 |
| 点云过滤与窗口 | 统计离群点过滤需要 Open3D，显示交互窗口另需桌面图形环境。纯离线 OBB 合成示例只需 OpenCV |

## AnyGrasp 授权检查与回放

将本机配套授权放在 `sdk/anygrasp_sdk/grasp_detection/license/`，默认权重为 `grasp_detection/log/checkpoint_detection.tar`。这些文件由使用者单独安装，Git 已忽略。当前适配器使用 `create_detector/get_grasp` 接口。采集特征码时使用申请授权时的机器和网卡组合。[官方授权说明](https://github.com/graspnet/anygrasp_sdk/tree/main/license_registration)

```bash
bash scripts/run_python.sh examples/anygrasp/check_sdk.py --feature-id
bash examples/replay/run.sh --config config/anygrasp.yaml --backend anygrasp \
  --input /path/to/snapshot-001/frame.npz --output outputs/anygrasp-replay
```

第一条检查 CUDA 扩展、文件和机器特征码。第二条真正加载授权及模型。仅 RGB-D 的文件需加 `--detect` 或选择 `--full-scene`。现场实时模式另需相机和图形桌面，执行模式还会读取手眼矩阵。

## Python 入口

`scripts/run_python.sh` 从仓库根目录解析相对路径。在本机优先激活 LJ，并加载 `sdk/runtime/gui` 和 `sdk/runtime/python` 里的本地 OpenCV 与 YOLO 依赖。其他电脑可把依赖直接安装在所用环境中；设置 `REBOT_PYTHON=/absolute/path/to/python` 可为示例选择具体解释器，不加载本机 LJ 专用目录。各启动命令无需历史输出文件。

稳定采样标定示例额外依赖本机同级的 `rebot-arm-b601-robstride` 控制工程、带载 URDF 与该工程校验过的采集器。普通手眼标定使用本仓库的采集器。[两种标定方式](../examples/hand_eye/README.md)
