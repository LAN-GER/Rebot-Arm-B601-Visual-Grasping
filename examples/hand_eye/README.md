# Hand–eye calibration

[简体中文](README_zh.md)

The marker check uses the camera only. Automatic calibration moves the robot. Run from the repository root.

```bash
bash examples/hand_eye/run.sh --check-marker
```

Set `calibration.aruco.marker_length_m` to the actual printed length of the black marker edge. The current setup uses a 0.100 m marker with ID 0. Print at 100% scale and measure the edge. [100 mm board](boards/aruco_100mm.pdf) · [30 mm board](boards/aruco_30mm.pdf)

Omit `--check-marker` to run the automatic sampling trajectory or add `--manual` for manual guidance. A successful solve updates `config/calibration/<camera_type>/hand_eye.npz` for that camera. To save a separate candidate matrix, use [hand–eye calibration with stationary samples](../stationary_calibration/README.md).
