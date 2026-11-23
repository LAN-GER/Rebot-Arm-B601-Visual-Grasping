# 稳定采样手眼标定

[English](README.md)

支持已存反馈离线预演，显式执行时跑轨迹。从仓库根目录运行。

```bash
bash examples/stationary_calibration/run.sh --feedback /path/to/feedback.json --output "outputs/calibration/preview-$(date +%s%N)"
```

示例调用同级 `rebot-arm-b601-robstride` 已校验的采集器及带载模型。上方命令用旧反馈演算轨迹。现场启动使用 `--execute --output "outputs/calibration/run-$(date +%s%N)"`。真实采集等待各关节连续稳定至少 1 秒，再配对图像和反馈。通过检查并求解后生成 `hand_eye_candidate.npz`。请复核候选再用于抓取。
