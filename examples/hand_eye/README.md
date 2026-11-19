# 手眼标定

标记检查只用相机，自动轨迹会移动机械臂。从仓库根目录运行。

```bash
bash examples/hand_eye/run.sh --check-marker
```

按实际打印后的黑色标记边长设置 `calibration.aruco.marker_length_m`。本机使用 0.100 m、ID 0。按 100% 比例打印后实测边长。[100 mm 打印板](boards/aruco_100mm.pdf) · [30 mm 打印板](boards/aruco_30mm.pdf)

去掉 `--check-marker` 会运行自动轨迹并采样；`--manual` 使用手动引导。求解成功将更新该相机的 `config/calibration/<相机型号>/hand_eye.npz`。要单独保存标定候选，可用[稳定采样标定](../stationary_calibration/README.md)。
