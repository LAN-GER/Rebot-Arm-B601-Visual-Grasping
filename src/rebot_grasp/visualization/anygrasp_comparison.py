"""Render and export camera-frame AnyGrasp candidates on their source RGB-D frame."""
import numpy as np

from rebot_grasp.visualization.comparison_export import save_comparison_files
from rebot_grasp.visualization.overlay import draw_candidates, draw_detection_boxes, draw_instance_grasps


def displayed_candidate(backend, result):
    return backend.selected or next(iter(result.candidates), None)


def comparison_image(capture, backend, result, *, instances=None):
    filtered = displayed_candidate(backend, result)
    panels = []
    for label, candidate in (("RAW REFERENCE", backend.reference), ("FILTERED", filtered)):
        if candidate is None:
            status = "NO CANDIDATE" if label == "RAW REFERENCE" else result.status.upper()
        elif label == "FILTERED" and result.status in ("outside_workspace", "no_reachable_candidate"):
            status = result.status.upper()
        elif label == "FILTERED" and backend.selected is not None:
            status = "SELECTED FOR EXECUTION"
        else:
            status = f"score {candidate.score:.2f}"
        panel = draw_candidates(capture.frame, [candidate] if instances is None and candidate else [],
                                label=f"{label} | {status}", line_color=(255, 255, 255))
        if instances is None:
            draw_detection_boxes(panel, result.detections, backend.last_target)
        else:
            draw_instance_grasps(panel, capture.frame, instances, backend.last_target, raw=label == "RAW REFERENCE")
        panels.append(panel)
    return np.concatenate(panels, axis=1)


def candidate_record(candidate):
    if candidate is None:
        return None
    return dict(T_camera_grasp=candidate.T_camera_grasp.tolist(), score=candidate.score,
                width_m=candidate.width_m)


def save_comparison(output, capture, backend, result):
    raw = backend.reference
    filtered = displayed_candidate(backend, result)
    delta = None
    if raw is not None and filtered is not None:
        displacement = filtered.T_camera_grasp[:3, 3] - raw.T_camera_grasp[:3, 3]
        cosine = abs(np.dot(raw.T_camera_grasp[:3, 1], filtered.T_camera_grasp[:3, 1]))
        delta = dict(position_camera_m=displacement.tolist(), distance_m=float(np.linalg.norm(displacement)),
                     jaw_axis_angle_deg=float(np.degrees(np.arccos(np.clip(cosine, 0., 1.)))),
                     width_m=filtered.width_m - raw.width_m)
    comparison = dict(frame_id=capture.frame.frame_id, status=result.status,
                      raw_is_reference_only=True, filtered_selected_for_execution=backend.selected is not None,
                      insertion_depth_m=backend.insertion_depth_m, raw=candidate_record(raw),
                      filtered=candidate_record(filtered), delta=delta)
    save_comparison_files(output, capture, backend, result, comparison, comparison_image(capture, backend, result))
