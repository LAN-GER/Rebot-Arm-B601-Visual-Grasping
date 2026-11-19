# reBot Arm B601 视觉抓取

用 RGB-D 相机采集场景，选定目标后使用 OBB、过滤版 OBB、AnyGrasp 或 GraspNet 生成抓取候选。当前现场组合是 B601-RS、Orbbec Gemini 336 和带 NVIDIA GPU 的 Ubuntu PC。AnyGrasp 与过滤版 OBB 共用同一套 B601 张爪、预抓取、下探、夹持和返回动作。当前香蕉抓取使用沿工具接近方向的 **5 mm** 额外下探。

## 在现场运行

以下命令从仓库根目录执行。启动器在这台电脑上使用已有的 `LJ` 环境。实时会话的输出目录必须是新目录，每次启动时重新生成。

### AnyGrasp 多物体抓取

普通版和过滤版都支持抓取笔、橙子和香蕉。默认选香蕉，检测置信度阈值为 0.30。在图像窗口按数字键切换目标，再按 **G** 重新采集 RGB-D 帧，对选中类别规划一次抓取。

| 按键 | 抓取目标 | 启动时指定目标 |
| --- | --- | --- |
| **1** | 笔 | `--target-class pen` |
| **2** | 橙子 | `--target-class orange` |
| **3** | 香蕉 | `--target-class "yellow banana"` |

**先看过滤效果**。从仓库根目录运行下面的命令即可打开相机预览。左图使用原始点云，右图使用过滤后的点云。AnyGrasp 对每个分割实例单独推理。绿色框和白线对应当前目标，其他物体用蓝框和黄色抓取线显示。线段连接两指端点，红点是抓取中心，`Z` 是该中心在相机坐标系下的深度，单位为米。只有框时可查看旁边的状态，`DEPTH REJECTED` 表示点云质量未通过，`NO GRASP` 表示没有有效候选。

```bash
bash examples/anygrasp_filtered/run.sh
```

**真机抓取**。下面的命令以笔为初始目标，沿工具接近方向额外下探 5 mm。加上 `--execute` 后，机械臂会在启动时移到准备位。按 **G** 时只执行右图中当前目标通过工作区和可达性检查的候选。目标缺失或没有可达候选时跳过本次抓取。完成后自动恢复预览，可按 **2** 切换到橙子，再按 **G**。按 **Q** 或 **Esc** 结束会话。[过滤版示例说明](examples/anygrasp_filtered/README.md)

```bash
bash examples/anygrasp_filtered/run.sh --execute --target-class pen --insertion-depth-mm 5
```

**普通版 AnyGrasp** 使用相同的选键和抓取线。下面两条命令分别启动相机预览和香蕉真机抓取。按 **G** 后显示该次快照，按 **R** 恢复实时画面。[普通版示例说明](examples/anygrasp/README.md)

```bash
bash examples/anygrasp/run.sh --output "outputs/anygrasp-live/preview-$(date +%Y%m%d-%H%M%S-%N)"
bash examples/anygrasp/run.sh --execute --target-class "yellow banana" \
  --output "outputs/anygrasp-live/banana-$(date +%Y%m%d-%H%M%S-%N)"
```

### OBB 与手眼标定

**过滤版 OBB**使用完整二维分割轮廓确定中心和方向，使用过滤后的目标点估计深度并做质量检查。启动后可同时查看原始与过滤估计、点云和深度诊断。以下两行分别启动相机预览和现场用过的 5 mm 下探试验。

```bash
bash examples/obb_filtered/run.sh
bash examples/obb_filtered/run.sh --execute --insertion-depth-mm 5
```

**手眼标定**使用 100 mm、ID 0 的 ArUco 板。第一行只检查标记。第二行运行自动轨迹，求解成功后写入 `config/calibration/<相机型号>/hand_eye.npz`。另外提供等待关节稳定后抓帧、单独保存候选矩阵的示例。[标定说明](examples/hand_eye/README.md) · [稳定采样示例](examples/stationary_calibration/README.md)

```bash
bash examples/hand_eye/run.sh --check-marker
bash examples/hand_eye/run.sh
```

## 在新环境运行离线示例

Python 3.10+。合成 OBB 示例只需要 NumPy、PyYAML 和 OpenCV，在普通 CPU 上可运行。可从任意工作目录使用已安装的 `rebot-grasp`，此时请用 `--config` 指向仓库里的配置。各硬件和算法的额外依赖见 [SDK 安装说明](docs/sdk_setup.md)。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[obb]'
rebot-grasp demo --output outputs/demo
rebot-grasp preview --input outputs/demo/frame.npz --output outputs/demo-replay
```

用已有实拍录制复核 AnyGrasp 时，输入应包含目标 mask；仅含 RGB-D 的采集文件可加 `--detect` 重新分割。回放会使用录制中的相机位姿生成计划，缺少位姿时保存相机系候选。

```bash
bash examples/replay/run.sh --config config/anygrasp.yaml --backend anygrasp \
  --input /path/to/snapshot-001/frame.npz --output outputs/banana-replay
```

开发环境运行完整测试时，安装点云与测试依赖。

```bash
python -m pip install -e '.[obb,visualization,dev]'
python -m pytest -q
```

## 配置与调参

| 要调整的内容 | 配置位置 | 当前值与用途 |
| --- | --- | --- |
| 相机、标定板、手眼路径、夹持力与准备位 | [config/default.yaml](config/default.yaml) | Gemini 336，640×480，30 fps。RS 参数在 `robot.gripper.rs` |
| 检测词表、默认目标、置信度 | [普通版 AnyGrasp](config/anygrasp.yaml) · [过滤版 AnyGrasp](config/anygrasp_filtered.yaml) | `yolo.custom_classes` 的顺序对应数字键，`target_class` 指定初始目标，`detection.conf_threshold` 控制检测阈值 |
| 基坐标工作范围、TCP 换算 | [AnyGrasp 现场配置](config/anygrasp.yaml) · [过滤版 AnyGrasp 现场配置](config/anygrasp_filtered.yaml) · [过滤版 OBB 现场配置](config/obb_filtered.yaml) | 现场配置继承 `default.yaml`，通过 `planning` 覆盖规划参数 |
| AnyGrasp 额外下探 | [config/backends/anygrasp.yaml](config/backends/anygrasp.yaml) | `insertion_depth_m: 0.005`，在 SDK 候选深度基础上沿工具 +X 增加 5 mm |
| 过滤版 AnyGrasp 额外下探与点云过滤 | [config/backends/anygrasp_filtered.yaml](config/backends/anygrasp_filtered.yaml) | 独立参数，默认下探 0.005 m |
| 过滤版 OBB 额外下探 | [config/backends/obb_filtered.yaml](config/backends/obb_filtered.yaml) | 默认 `0.0`，单次运行可用 `--insertion-depth-mm 5` 覆盖 |
| 普通 OBB 参数 | [config/backends/obb.yaml](config/backends/obb.yaml) | 独立的轮廓估计配置，插入深度默认 0.015 m |
| 点云、夹爪遮罩与视觉工作区域 | 对应后端 YAML | `outlier_filter`、`gripper_exclusion`、`workspace_camera` 和 `pointcloud_quality`，XYZ 均在相机系 |
| 模型与授权 | [SDK 安装说明](docs/sdk_setup.md) | SDK 和检测权重路径相对于后端 YAML，手眼路径相对于应用配置目录 |

改变 `insertion_depth_m` 调整的是抓取点沿接近方向的位置。手眼矩阵描述相机的刚性安装关系，重新安装相机时应重新标定。实际执行时采用的配置和矩阵写入会话的 `session.json`。

## 工程结构

```text
├── pyproject.toml
├── config/
│   ├── default.yaml、anygrasp.yaml、anygrasp_filtered.yaml、obb_filtered.yaml
│   ├── backends/                 # obb、obb_filtered、anygrasp、anygrasp_filtered、graspnet
│   └── calibration/              # 每个相机的内参与手眼结果
├── src/rebot_grasp/
│   ├── cli.py、bootstrap.py、config.py
│   ├── core/                     # 帧、目标、候选、计划和接口
│   ├── cameras/                  # Orbbec、RealSense、录制与合成相机
│   ├── perception/               # YOLO、实例选择、点云过滤与质量检查
│   ├── grasping/                 # backends/、候选过滤与选择
│   ├── geometry/                 # 坐标变换、工具轴与 TCP
│   ├── calibration/              # ArUco、手眼求解、稳定采样与轨迹校验
│   ├── robots/                   # B601、动作与夹爪控制
│   ├── pipeline/                 # 采集、规划、执行与实时会话
│   └── visualization/            # 图像叠加、点云窗口与诊断导出
├── examples/                     # 每个示例独立目录，含 run.sh 和 README.md
├── scripts/                      # 运行环境与旧命令的轻量入口
├── tests/
├── docs/
└── sdk/                          # 本机外部 SDK、二进制依赖与授权
```

## 查找示例和结果

完整清单列出每个示例使用的设备、启动行为及旧命令对应关系。[示例索引](examples/README.md)

过滤版 AnyGrasp 默认把每次按 **G** 的结果写入 `outputs/anygrasp-filtered/run-*/result/snapshot-NNN/`。普通版使用 `--output` 指定的目录保存快照。

- 查看这次抓的是哪一类、点云是否通过以及生成了哪些计划，打开 `result.json`。其中 `target_class` 对应按 **G** 时选中的类别。
- 比较过滤前后的抓取位置，打开过滤版快照中的 `comparison.png` 和 `comparison.json`。`diagnostics.png` 显示分割和深度，`frame.npz` 可用于离线回放。
- 查看机械臂是否实际发起抓取，打开 `execution.json`。`object_held` 是控制器对夹持状态的判断，复核滑落情况时应结合现场观察。

过滤版 OBB 的位置和方向对比见[对照说明](docs/obb_filtered.md)。

[模块与坐标约定](docs/architecture.md) · [点云质量与拒绝原因](docs/pointcloud_preprocessing.md) · [上游 B601 抓取教程](https://wiki.seeedstudio.com/cn/rebot_arm_b601_rs_grasping_demo/)
