# Motor 7 gripper-opening diagnostic

[简体中文](README_zh.md)

This example connects to and drives only gripper motor 7 on the B601-RS. Run from the repository root.

```bash
bash examples/gripper_check/run.sh --controller-open
```

Use `--controller-open` for one open command through the controller or `--open` for a single-motor MIT test. Both commands move the gripper. The terminal reports the result or fault.
