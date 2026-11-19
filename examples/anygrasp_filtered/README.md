# AnyGrasp 点云过滤对照

当前桌面词表为 `pen`（笔）、`orange`（橙子）和 `yellow banana`（香蕉）。窗口显示每个检测实例的类别和置信度。绿色框表示本轮抓取目标，蓝色框表示其他识别结果。默认抓香蕉，可用 `--target-class pen` 或 `--target-class orange` 切换目标。预览逐实例使用各自的 mask 调用 AnyGrasp。白线为当前目标的候选，黄色线为其他物体的候选，Z 表示候选中心的相机深度。点云质量被拒或无有效候选时会在实例旁标记原因。按图像窗口中的 **1 选笔、2 选橙子、3 选香蕉**（对应配置中的词表顺序）。选中后按 **G** 重新采图，执行模式只抓当前类别。目标缺失时跳过此次抓取。

同一张 RGB-D、同一个选中实例的分割和同一个 AnyGrasp 模型产生两幅图。左侧 RAW REFERENCE 使用深度范围内的原始点云及目标 mask；右侧 FILTERED 使用夹爪遮罩、相机工作区、离群点过滤和质量检查。两侧使用相同的模型、碰撞检测、宽度与分数限制，以及额外下探量。原有 AnyGrasp 示例仍使用自己的配置。

白线连接投影到图像上的两指位置，红点表示抓取中心。等待按 G 时显示各自最高分的有效候选。每个检测实例通常要做两次模型推理，底部帧龄表示画面的滞后。辅助窗口显示过滤点云和分割/深度，紫色标出排除的夹爪像素。

## 启动

从工程根目录运行。第一行仅用相机预览。第二行启动后机械臂会进入准备位，按 G 重新采集一帧，从右侧候选中选取通过工作区和 IK 检查的抓取。质量拒绝或无可达抓取时跳过该次动作。完成后自动恢复预览，Q/Esc 结束会话。

```bash
bash examples/anygrasp_filtered/run.sh
bash examples/anygrasp_filtered/run.sh --execute --target-class "yellow banana" --insertion-depth-mm 5
```

使用 `--single-window` 仅显示左右对照。配置在 [现场配置](../../config/anygrasp_filtered.yaml) 和 [后端参数](../../config/backends/anygrasp_filtered.yaml)，当前阈值 0.30、额外下探 0.005 m。下探沿工具 +X，应只在一个位置调整；命令行的毫米值会覆盖配置值。

## 回放和结果

```bash
bash examples/anygrasp_filtered/run.sh --input /path/to/snapshot-001/frame.npz
```

离线回放用录制的 RGB-D 重新检测目标，在 GPU 上运行同一套对照。默认写入 `outputs/anygrasp-filtered/run-*/result/`，终端会打印本轮目录。也可用 `--output` 指定新的目录。实时会话每次按 G 保存一个 `snapshot-NNN/`。

- `comparison.png` 和 `comparison.json` 给出两侧显示的抓取、相机系中心位移、夹爪宽度差与无向开合轴夹角。两次模型推理可能选中不同的抓取位置，单帧差值也包含候选排序的影响。
- `diagnostics.png`、`pointcloud_raw.npz`、`pointcloud.npz` 和各 mask 可用于复核分割、夹爪排除和离群点。未检出目标时保存空对照。
- `result.json` 记录过滤后的候选、计划和质量原因。实际执行时右图标注选中的可达候选，`execution.json` 记录动作。
