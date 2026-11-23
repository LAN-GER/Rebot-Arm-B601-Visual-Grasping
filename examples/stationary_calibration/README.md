# Hand–eye calibration with stationary samples

[简体中文](README_zh.md)

Preview a trajectory using recorded feedback or explicitly run it on the arm. Run from the repository root.

```bash
bash examples/stationary_calibration/run.sh --feedback /path/to/feedback.json --output "outputs/calibration/preview-$(date +%s%N)"
```

The example uses the validated collector and payload model from the sibling `rebot-arm-b601-robstride` checkout. The command above previews the trajectory from saved feedback. For live capture use `--execute --output "outputs/calibration/run-$(date +%s%N)"`. During live capture, each joint must remain stable for at least one second before the collector pairs the image with feedback. A successful check and solve produce `hand_eye_candidate.npz`. Review the candidate before using it for grasping.
