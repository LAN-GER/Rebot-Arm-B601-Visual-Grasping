# 手眼标定结果

[English](README.md)

每个相机安装位置使用独立的手眼矩阵。当前 B601-RS 的 Gemini 336 使用 `orbbec_gemini336/hand_eye.npz`。标定文件由设备现场生成，Git 忽略这些文件。运行配置中的 `calibration.hand_eye_path` 相对于配置文件所在目录。平移使用米。
