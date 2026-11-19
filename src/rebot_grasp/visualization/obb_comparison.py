"""Export same-frame OBB before/after evidence, independent of robot control."""
import numpy as np

from rebot_grasp.visualization.comparison_export import save_comparison_files
from rebot_grasp.visualization.overlay import draw_target_box


def comparison_image(capture, backend, result):
    import cv2
    image = capture.frame.color_bgr
    left, right = image.copy(), image.copy()
    prepared = backend.last_prepared
    if prepared is not None:
        for panel, mask, color in ((left, backend.last_target.mask, [0, 200, 255]),
                                   (right, prepared.gripper_mask, [230, 30, 230]),
                                   (right, backend.last_eligible_mask, [40, 230, 40])):
            panel[mask] = (.65 * panel[mask] + .35 * np.asarray(color)).astype(np.uint8)
    records = backend.last_comparison or {}
    status_label = {'camera_preview_only_missing_transforms': 'PREVIEW ONLY',
                    'planned_not_executed': 'PLAN READY', 'pointcloud_rejected': 'QUALITY REJECTED',
                    'target_not_found': 'TARGET NOT FOUND', 'no_valid_grasp': 'NO VALID GRASP',
                    'outside_workspace': 'OUTSIDE WORKSPACE',
                    'no_reachable_candidate': 'IK UNREACHABLE'}.get(result.status, result.status)
    for panel, name in ((left, 'raw'), (right, 'filtered')):
        draw_target_box(panel, backend.last_target, top=78)
        record = records.get(name)
        title = 'RAW REFERENCE - never executed' if name == 'raw' else 'FILTERED - ' + status_label
        cv2.putText(panel, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        if record is not None and record['position'] is not None:
            cv2.polylines(panel, [np.rint(record['rect_points']).astype(np.int32)], True, (255, 200, 0), 1)
            ends = np.rint(record['short_edge_points']).astype(int)
            cv2.line(panel, tuple(ends[0]), tuple(ends[1]), (255, 255, 255), 2)
            cv2.circle(panel, tuple(record['center_px']), 4, (0, 0, 255), -1)
            z = record['position'][2]
            text = f"{record['class_name']} {record['conf']:.2f} z={z:.3f}m jaw={record['jaw_width_m']*1000:.1f}mm"
            cv2.putText(panel, text, (8, 40), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1)
    if prepared is not None:
        q = prepared.report['quality']
        text = 'QUALITY PASS' if q['passed'] else 'REJECT: ' + ', '.join(q['reasons'])
        cv2.putText(right, text, (8, 60), cv2.FONT_HERSHEY_SIMPLEX, .4,
                    (40, 230, 40) if q['passed'] else (0, 0, 255), 1)
    insertion = records.get('insertion_depth_m', 0.)
    cv2.putText(right, f'Extra approach: {insertion*1000:.1f} mm along tool +X',
                (8, right.shape[0] - 32), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
    return np.concatenate([left, right], axis=1)


def save_comparison(output, capture, backend, result):
    comparison = backend.last_comparison or dict(frame_id=capture.frame.frame_id, status=result.status,
                                                raw=None, filtered=None, raw_is_reference_only=True)
    save_comparison_files(output, capture, backend, result, comparison, comparison_image(capture, backend, result))
