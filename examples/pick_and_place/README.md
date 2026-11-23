# Place a banana in a box

[简体中文](README_zh.md)

The original interactive workflow enables the arm and moves it to the ready pose on startup. Run from the repository root.

```bash
bash examples/pick_and_place/run.sh --help
```

The example detects a banana and a box, then runs the placement sequence. Omit `--help` to connect to the robot. `--dry-run` skips the pick-and-place sequence itself.
