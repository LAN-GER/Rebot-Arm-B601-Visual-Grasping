import json
from pathlib import Path

from rebot_grasp.core.types import Capture
from rebot_grasp.pipeline.planning import PlanningResult


def save_result(output: str | Path, capture: Capture, result: PlanningResult, *, target_class=None) -> Path:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "synthetic": capture.synthetic, "executed": False,
               "frame_id": capture.frame.frame_id, "status": result.status, "units": "meters",
               "pointcloud_report": result.pointcloud_report,
               "candidate_origin": "modeled fingertip-plane midpoint",
               "candidates": [{"backend": g.backend, "score": g.score, "width_m": g.width_m,
                               "T_camera_grasp": g.T_camera_grasp.tolist()} for g in result.candidates],
               "plans": [{"T_base_pregrasp": p.T_base_pregrasp.tolist(), "T_base_grasp": p.T_base_grasp.tolist(),
                          "T_base_retreat": p.T_base_retreat.tolist(), "opening_m": p.opening_m}
                         for p in result.plans]}
    if target_class is not None:
        payload["target_class"] = target_class
    if result.insertion_depth_m is not None:
        payload["insertion_depth_m"] = result.insertion_depth_m
        if result.insertion_depth_m:
            payload["candidate_origin"] = "fingertip-plane midpoint after insertion along tool +X"
    path = output / "result.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path
