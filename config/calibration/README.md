# Hand–eye calibration results

[简体中文](README_zh.md)

Use a separate hand–eye matrix for each camera mounting position. The Gemini 336 on the current B601-RS uses `orbbec_gemini336/hand_eye.npz`. Generate the calibration files on the actual hardware. Git ignores these files. The runtime setting `calibration.hand_eye_path` is relative to the configuration file. Translations are in meters.
