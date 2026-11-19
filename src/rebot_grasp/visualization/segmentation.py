"""Save and display a single segmented RGB-D frame and its aligned point cloud."""
import json
from pathlib import Path

import numpy as np

from rebot_grasp.perception.preprocessing import prepare_cloud
from rebot_grasp.perception.target_selection import select_target


def inspect_segmentation(capture, targets, output, *, target_class=None, min_depth_m=.05,
                         max_depth_m=1., view=False, outlier_filter=None, workspace_camera=None,
                         gripper_exclusion=None, pointcloud_quality=None):
    import cv2

    output = Path(output)
    frame = capture.frame
    selected = select_target(targets, target_class)
    prepared = prepare_cloud(frame, selected, min_depth_m, max_depth_m,
                             outlier_filter=outlier_filter, workspace_camera=workspace_camera,
                             gripper_exclusion=gripper_exclusion, pointcloud_quality=pointcloud_quality,
                             target_required=True)
    raw, cloud = prepared.raw, prepared.cloud
    raw_count = len(raw.points)
    raw_selected_count = int(raw.target_mask.sum()) if selected is not None else 0
    np.savez_compressed(output / "pointcloud_raw.npz", points_m=raw.points,
                        colors_rgb=raw.colors_rgb,
                        target_mask=raw.target_mask if selected is not None else np.zeros(raw_count, bool))
    selected_points = cloud.target_mask if selected is not None else np.zeros(len(cloud.points), bool)
    overlay = frame.color_bgr.copy()
    instances = []
    for index, target in enumerate(targets):
        if target.mask.shape != frame.depth_m.shape:
            raise ValueError("Segmentation mask is not aligned with depth")
        color = np.array([40, 230, 40] if target is selected else [230, 140, 30], dtype=np.uint8)
        overlay[target.mask] = (0.55 * overlay[target.mask] + 0.45 * color).astype(np.uint8)
        contours, _ = cv2.findContours(target.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color.tolist(), 2)
        ys, xs = np.nonzero(target.mask)
        if len(xs):
            cv2.putText(overlay, f"{index}: {target.label} {target.confidence:.2f}",
                        (int(xs.min()), max(15, int(ys.min()) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, color.tolist(), 1)
        if not cv2.imwrite(str(output / f"mask-{index:02d}.png"), target.mask.astype(np.uint8) * 255):
            raise OSError("Could not save target mask")
        valid = target.mask & np.isfinite(frame.depth_m) & (frame.depth_m > min_depth_m) & (frame.depth_m < max_depth_m)
        instances.append(dict(index=index, label=target.label, confidence=target.confidence,
                              pixels=int(target.mask.sum()), valid_depth_pixels=int(valid.sum()),
                              selected=target is selected))
    for name, image in (("color.png", frame.color_bgr), ("segmentation.png", overlay)):
        if not cv2.imwrite(str(output / name), image):
            raise OSError(f"Could not save {name}")
    np.savez_compressed(output / "pointcloud.npz", points_m=cloud.points,
                        colors_rgb=cloud.colors_rgb, target_mask=selected_points)
    report = dict(frame_id=frame.frame_id, synthetic=capture.synthetic,
                  resolution=[frame.depth_m.shape[1], frame.depth_m.shape[0]],
                  depth_range_m=[min_depth_m, max_depth_m], point_count=len(cloud.points),
                  selected_label=selected.label if selected else None,
                  outlier_filter=outlier_filter, raw_point_count=raw_count,
                  removed_point_count=raw_count - len(cloud.points),
                  raw_selected_point_count=raw_selected_count,
                  removed_selected_point_count=raw_selected_count - int(selected_points.sum()),
                  selected_point_count=int(selected_points.sum()), instances=instances,
                  preprocessing=prepared.report, quality=prepared.report['quality'])
    # Separate diagnostic image: green eligible target, magenta removed self,
    # gray outside workspace. Keep the original YOLO overlay for comparison.
    eligible = np.zeros(frame.depth_m.shape, bool)
    if len(cloud.points):
        xyz = cloud.points
        u = np.rint(xyz[:, 0] * frame.K[0, 0] / xyz[:, 2] + frame.K[0, 2]).astype(int)
        v = np.rint(xyz[:, 1] * frame.K[1, 1] / xyz[:, 2] + frame.K[1, 2]).astype(int)
        eligible[v[selected_points], u[selected_points]] = True
    processed = frame.color_bgr.copy()
    for mask, color in ((~prepared.workspace_mask, [100, 100, 100]),
                        (prepared.gripper_mask, [230, 30, 230]), (eligible, [40, 230, 40])):
        processed[mask] = (.55 * processed[mask] + .45 * np.asarray(color)).astype(np.uint8)
    quality = prepared.report['quality']
    label = 'PASS' if quality['passed'] else 'REJECT: ' + ', '.join(quality['reasons'])
    cv2.putText(processed, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .45,
                (40, 230, 40) if quality['passed'] else (0, 0, 255), 1)
    for name, data in (('preprocessing.png', processed),
                       ('gripper_exclusion.png', prepared.gripper_mask.astype(np.uint8) * 255),
                       ('eligible_target.png', eligible.astype(np.uint8) * 255)):
        if not cv2.imwrite(str(output / name), data):
            raise OSError(f'Could not save {name}')
    (output / "inspection.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print(f"Saved {output.resolve()}", flush=True)
    if view:
        window = "RGB-D segmentation: green=selected, Q=point cloud"
        try:
            cv2.imshow(window, processed)
            while cv2.waitKey(30) & 0xff not in (ord('q'), ord('Q'), 27):
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            cv2.destroyAllWindows()
        if len(cloud.points):
            import open3d as o3d
            geometry = o3d.geometry.PointCloud()
            geometry.points = o3d.utility.Vector3dVector(cloud.points)
            colors = cloud.colors_rgb.copy()
            colors[selected_points] = [0.1, 1., 0.1]
            geometry.colors = o3d.utility.Vector3dVector(colors)
            o3d.visualization.draw_geometries([geometry], window_name="RGB-D cloud: green=selected target")
    return quality['passed']
