import numpy as np

from rebot_grasp.core.types import GraspCandidate, RGBDFrame, Target


def draw_candidates(frame: RGBDFrame, candidates: list[GraspCandidate], label="PREVIEW ONLY",
                    *, line_color=(40, 255, 80)) -> np.ndarray:
    import cv2
    image = frame.color_bgr.copy()
    for grasp in candidates[:20]:
        draw_grasp_line(image, frame, grasp, line_color)
    cv2.putText(image, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return image


def draw_grasp_line(image, frame, grasp, line_color):
    import cv2

    T = grasp.T_camera_grasp
    center, opening = T[:3, 3], T[:3, 1]
    points = np.array([center - opening * grasp.width_m / 2, center, center + opening * grasp.width_m / 2])
    if (points[:, 2] <= 0).any():
        return
    projected = points @ frame.K.T
    uv = np.rint(projected[:, :2] / projected[:, 2:]).astype(int)
    cv2.line(image, tuple(uv[0]), tuple(uv[2]), (30, 30, 30), 4)
    cv2.line(image, tuple(uv[0]), tuple(uv[2]), line_color, 2)
    cv2.circle(image, tuple(uv[1]), 3, (0, 0, 255), -1)


def draw_target_box(image: np.ndarray, target: Target | None, *, top=18, color=(40, 230, 40), suffix="") -> None:
    """Draw the selected mask's bounds and detector confidence onto a display copy."""
    import cv2

    if target is None or not target.mask.any():
        cv2.putText(image, "Target not detected", (8, top + 7), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (0, 180, 255), 1)
        return
    y, x = np.nonzero(target.mask)
    x1, y1 = int(x.min()), int(y.min())
    cv2.rectangle(image, (x1, y1), (int(x.max()), int(y.max())), color, 2)
    label = f"{target.label} {target.confidence:.2f}{suffix}"
    width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .48, 1)[0][0]
    label_x = max(0, min(x1, image.shape[1] - width - 1))
    cv2.putText(image, label, (label_x, max(top, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, .48, color, 1)


def draw_detection_boxes(image, targets, selected):
    """Draw all current detections, with the chosen grasp instance on top."""
    for target in targets:
        if target is not selected and target.mask.any():
            draw_target_box(image, target, top=40, color=(230, 160, 40))
    if selected is not None:
        draw_target_box(image, selected, top=40)

def draw_instance_grasps(image, frame, instances, selected, *, raw=False):
    """Mark each preview with its own depth, or the reason no line is available."""
    for item in instances:
        candidate = item.raw if raw else item.filtered
        active = item.target is selected
        if candidate is None:
            status = "NO GRASP" if raw or item.status != "pointcloud_rejected" else "DEPTH REJECTED"
        else:
            status = f"Z={candidate.T_camera_grasp[2, 3]:.3f}m"
        draw_target_box(image, item.target, top=40, color=(40, 230, 40) if active else (230, 160, 40),
                        suffix="  " + status)
        if candidate is not None:
            draw_grasp_line(image, frame, candidate, (255, 255, 255) if active else (0, 205, 255))
