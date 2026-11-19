# 录制帧抓取回放

离线推理。从仓库根目录运行。

```bash
bash examples/replay/run.sh --input outputs/demo/frame.npz --output outputs/demo-replay
```

可以先运行[合成 OBB](../offline_demo/README.md)生成上述输入。选择 `--backend anygrasp --config config/anygrasp.yaml` 可对带目标 mask 的实拍帧做真实 GPU 推理；`--detect` 重新运行 YOLO。输出候选、计划和叠加图。
