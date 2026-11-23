# 7 号电机张爪诊断

[English](README.md)

仅连接并驱动 B601-RS 的 7 号夹爪电机。从仓库根目录运行。

```bash
bash examples/gripper_check/run.sh --controller-open
```

使用控制器张爪逻辑复现单次张开。`--open` 选择单电机 MIT 测试。两者均会实际驱动夹爪。终端显示张爪结果或故障原因。
