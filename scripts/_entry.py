"""Run a checkout example through a historical Python entry point."""
from pathlib import Path
import runpy
import sys


def run_example(name: str, filename: str = "app.py") -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root / "src"), str(root / "sdk/reBotArm_control_py")]
    runpy.run_path(str(root / "examples" / name / filename), run_name="__main__")
