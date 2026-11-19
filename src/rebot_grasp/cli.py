"""Recordings, offline planning and opt-in B601-controlled AnyGrasp sessions."""

import argparse
from pathlib import Path
import sys

from rebot_grasp.config import BACKEND_KEYS
from rebot_grasp.pipeline import commands


COMMANDS = {
    "obb-filtered": commands.run_obb_filtered,
    "anygrasp-filtered": commands.run_anygrasp_filtered,
    "camera-check": commands.run_camera_check,
    "anygrasp-live": commands.run_anygrasp_live,
    "record": commands.run_record,
    "demo": commands.run_preview,
    "preview": commands.run_preview,
}


def parser():
    root = argparse.ArgumentParser(prog="rebot-grasp", description="RGB-D grasp planning and B601 control")
    commands = root.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run synthetic RGB-D through OBB and planning")
    preview = commands.add_parser("preview", help="Preview a recorded RGB-D NPZ")
    record = commands.add_parser("record", help="Save an RGB-D frame from the camera")
    check = commands.add_parser("camera-check", help="One-frame YOLO segmentation and point-cloud inspection")
    check.add_argument("--config", default="config/anygrasp.yaml")
    check.add_argument("--output", type=Path, required=True)
    check.add_argument("--input", type=Path, help="Inspect a recording instead of opening the camera")
    check.add_argument("--target-class")
    check.add_argument("--warmup", type=int, default=20)
    check.add_argument("--view", action="store_true", help="Show segmentation, then the selected target in 3D")
    live = commands.add_parser("anygrasp-live", help="AnyGrasp snapshots with optional original B601 execution")
    live.add_argument("--config", default="config/anygrasp.yaml")
    live.add_argument("--output", type=Path, required=True, help="New run directory")
    live_mode = live.add_mutually_exclusive_group()
    live_mode.add_argument("--diagnose-gripper", action="store_true",
                           help="Move ready, press G to infer and test opening only, then release/home")
    live_mode.add_argument("--execute", action="store_true", help="Enable B601 control, move ready on startup, grasp on G")
    live_target = live.add_mutually_exclusive_group()
    live_target.add_argument("--target-class")
    live_target.add_argument("--full-scene", action="store_true")
    for backend, name in (("obb_filtered", "OBB"), ("anygrasp_filtered", "AnyGrasp")):
        filtered = commands.add_parser(backend.replace("_", "-"), help=f"{name} raw/filtered point-cloud comparison")
        filtered.add_argument('--config', default=f'config/{backend}.yaml')
        filtered.add_argument('--output', type=Path, required=True, help='New run directory')
        filtered.add_argument('--target-class')
        filtered.add_argument('--single-window', action='store_true', help='Disable auxiliary 3D and diagnostic windows')
        filtered.add_argument('--insertion-depth-mm', type=float,
                              help='Extra approach along tool +X in millimeters; default from backend config')
        filtered_mode = filtered.add_mutually_exclusive_group()
        filtered_mode.add_argument('--execute', action='store_true', help='Enable B601 robot actions; move ready on startup')
        filtered_mode.add_argument('--input', type=Path, help='Offline RGB-D replay; never connects to robot or camera')
    record.add_argument("--config", default="config/default.yaml")
    record.add_argument("--output", type=Path, required=True)
    record.add_argument("--camera-type", dest="type", choices=(
        "orbbec_gemini2", "orbbec_gemini336", "realsense_d405", "realsense_d435i",
    ))
    record.add_argument("--width", dest="color_width", type=int)
    record.add_argument("--height", dest="color_height", type=int)
    record.add_argument("--fps", type=int)
    record.add_argument("--warmup", type=int, default=5)
    for command in (demo, preview):
        command.add_argument("--config", default="config/default.yaml")
        command.add_argument("--output", type=Path, default=Path("outputs/preview"))
        command.add_argument("--view-3d", action="store_true", help="Show scene and jaws in an Open3D window")
    preview.add_argument("--input", type=Path, required=True)
    preview.add_argument("--backend", choices=tuple(BACKEND_KEYS))
    target = preview.add_mutually_exclusive_group()
    target.add_argument("--target-class")
    target.add_argument("--full-scene", action="store_true", help="Scene-wide AnyGrasp or GraspNet inference")
    preview.add_argument("--detect", action="store_true", help="Run configured YOLO instead of recorded mask")
    preview.add_argument("--no-image", action="store_true", help="Export JSON without OpenCV overlay")
    return root


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except (ValueError, OSError, RuntimeError, ImportError, TypeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
