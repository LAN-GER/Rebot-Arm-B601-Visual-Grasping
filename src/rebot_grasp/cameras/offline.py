from pathlib import Path
import time

import numpy as np

from rebot_grasp.core.types import Capture, RGBDFrame, Target


def load_recording(path: str | Path) -> tuple[Capture, list[Target]]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        missing = {"color_bgr", "depth_m", "K"} - set(data.files)
        if missing:
            raise ValueError(f"Recording is missing {sorted(missing)}; depth must be stored in meters")
        frame_id = str(data["frame_id"].item()) if "frame_id" in data else path.name
        # Replay gets a host timestamp; it is never evidence of a fresh live scene.
        frame = RGBDFrame(data["color_bgr"], data["depth_m"], data["K"], frame_id, time.monotonic())
        T = data["T_base_camera"] if "T_base_camera" in data else None
        targets = []
        if "target_mask" in data:
            label = str(data["target_label"].item()) if "target_label" in data else "object"
            target = Target(label, data["target_mask"])
            if target.mask.shape != frame.depth_m.shape:
                raise ValueError("Recorded target mask is not aligned")
            targets.append(target)
        synthetic = data["synthetic"].item() if "synthetic" in data else False
        return Capture(frame, T, synthetic), targets


def save_recording(path: str | Path, capture: Capture, target: Target | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = capture.frame
    values = dict(color_bgr=frame.color_bgr, depth_m=frame.depth_m, K=frame.K, frame_id=frame.frame_id,
                  synthetic=capture.synthetic)
    if capture.T_base_camera is not None:
        values["T_base_camera"] = capture.T_base_camera
    if target is not None:
        if target.mask.shape != frame.depth_m.shape:
            raise ValueError("Target mask is not aligned")
        values.update(target_mask=target.mask, target_label=target.label)
    np.savez_compressed(path, **values)
