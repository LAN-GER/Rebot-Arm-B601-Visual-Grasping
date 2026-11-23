# Grasp preview from a recording

[简体中文](README_zh.md)

This example runs offline inference. Run from the repository root.

```bash
bash examples/replay/run.sh --input outputs/demo/frame.npz --output outputs/demo-replay
```

Run the [synthetic OBB demo](../offline_demo/README.md) to generate the input above. Use `--backend anygrasp --config config/anygrasp.yaml` to run GPU inference on a recorded frame with a target mask. Add `--detect` to rerun YOLO. The output includes candidates, the plan, and an overlay.
