# GraspNet 真机抓取

[English](README.md)

相机、GPU、机械臂，启动会使能并进入准备位。从仓库根目录运行。

```bash
bash examples/graspnet_grasp/run.sh --help
```

运行前准备 GraspNet-Baseline、权重和手眼矩阵。G 推理并执行候选。使用 `--dry-run` 时仍会进入准备位，需要仅查看候选时请运行[GraspNet 相机预览](../graspnet_camera/README_zh.md)。
