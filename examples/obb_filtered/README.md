# 过滤版 OBB 对照

从仓库根目录运行。第一条命令开启相机预览，第二条会让机械臂进入准备位。在窗口按 **G** 生成快照，执行模式下会从过滤后的可达候选中选择抓取。

```bash
bash examples/obb_filtered/run.sh
bash examples/obb_filtered/run.sh --execute
```

如果夹取深度不足，在执行命令末尾加 `--insertion-depth-mm <毫米数>`。这个值覆盖本轮沿工具接近方向的额外下探量，默认值为 0 mm。[后端配置](../../config/backends/obb_filtered.yaml)

主窗口并排显示同一帧的原始和过滤结果。默认输出在 `outputs/obb-filtered/run-*/result/`。[窗口、算法与输出](../../docs/obb_filtered.md)
