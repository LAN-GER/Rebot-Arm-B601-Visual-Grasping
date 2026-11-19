from dataclasses import dataclass

from rebot_grasp.config import AppConfig
from rebot_grasp.core.interfaces import GraspBackend
from rebot_grasp.core.types import Capture, GraspCandidate, GraspPlan, Target
from rebot_grasp.geometry.tool_frames import grasp_to_tcp, retreat_along_approach
from rebot_grasp.grasping.filters import filter_candidates, filter_workspace
from rebot_grasp.perception.target_selection import select_target


@dataclass(frozen=True)
class PlanningResult:
    candidates: list[GraspCandidate]
    plans: list[GraspPlan]
    status: str
    pointcloud_report: dict | None = None
    insertion_depth_m: float | None = None
    detections: tuple[Target, ...] = ()


def make_plans(capture: Capture, candidates: list[GraspCandidate], config: AppConfig) -> list[GraspPlan]:
    if capture.T_base_camera is None or config.planning["T_grasp_tcp"] is None:
        return []
    settings = config.planning
    plans = []
    for candidate in candidates:
        if candidate.frame_id != capture.frame.frame_id:
            raise ValueError("Candidate and capture belong to different frames")
        pose = grasp_to_tcp(candidate, capture.T_base_camera, settings["T_grasp_tcp"])
        approach = capture.T_base_camera[:3, :3] @ candidate.T_camera_grasp[:3, 0]
        pre = retreat_along_approach(pose, approach, settings["pregrasp_offset_m"])
        retreat = retreat_along_approach(pose, approach, settings["retreat_offset_m"])
        opening_m = candidate.width_m + config.filters["opening_margin_m"]
        plans.append(GraspPlan(candidate, pre, pose, retreat, opening_m))
    return filter_workspace(plans, settings["workspace_min_m"], settings["workspace_max_m"])


def plan_capture(capture: Capture, targets: list[Target], backend: GraspBackend, config: AppConfig) -> PlanningResult:
    detections = tuple(targets)
    selected = None if config.full_scene else select_target(targets, config.target_class)
    if selected is None and not config.full_scene:
        return PlanningResult([], [], "target_not_found", detections=detections)
    candidates = backend.predict(capture.frame, selected)
    cloud_report = getattr(backend, 'last_preprocessing_report', None)
    if cloud_report is not None and not cloud_report['quality']['passed']:
        return PlanningResult([], [], 'pointcloud_rejected', cloud_report, detections=detections)
    if any(candidate.frame_id != capture.frame.frame_id for candidate in candidates):
        raise ValueError("Backend returned a candidate for another frame")
    candidates = filter_candidates(candidates, **config.filters)
    plans = make_plans(capture, candidates, config)
    if not candidates:
        status = "no_valid_grasp"
    elif capture.T_base_camera is None or config.planning["T_grasp_tcp"] is None:
        status = "camera_preview_only_missing_transforms"
    else:
        status = "planned_not_executed" if plans else "outside_workspace"
    insertion = getattr(backend, "insertion_depth_m", None) if config.backend in ("anygrasp", "anygrasp_filtered") else None
    return PlanningResult(candidates, plans, status, cloud_report, insertion, detections)
