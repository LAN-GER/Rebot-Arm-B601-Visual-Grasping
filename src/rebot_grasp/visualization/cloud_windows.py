"""Nonblocking filtered point-cloud diagnostic windows; no camera access or motion commands."""
import numpy as np

from rebot_grasp.visualization.pointcloud_viewer import _jaw_segments


def diagnostic_image(capture, backend, result):
    """Same-frame segmentation and metric depth, including missing-depth pixels."""
    import cv2
    frame = capture.frame
    color = frame.color_bgr.copy()
    z = frame.depth_m
    valid = np.isfinite(z) & (z > backend.min_depth_m) & (z < backend.max_depth_m)
    normalized = np.zeros(z.shape, np.uint8)
    normalized[valid] = np.rint(255 * np.clip(
        (z[valid] - backend.min_depth_m) / (backend.max_depth_m - backend.min_depth_m), 0, 1)).astype(np.uint8)
    depth = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    depth[~valid] = 0
    prepared = backend.last_prepared
    if prepared is not None:
        for mask, tint in ((backend.last_target.mask, [0, 200, 255]),
                           (prepared.gripper_mask, [230, 30, 230]),
                           (backend.last_eligible_mask, [40, 230, 40])):
            color[mask] = (.55 * color[mask] + .45 * np.asarray(tint)).astype(np.uint8)
        contours, _ = cv2.findContours(backend.last_target.mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(depth, contours, -1, (255, 255, 255), 1)
        metrics = prepared.report['quality']['metrics']
        status = ('PASS' if prepared.report['quality']['passed'] else 'REJECT')
        summary = f"{status}  points={metrics['target_points']}  depth={metrics['valid_depth_ratio']:.1%}  kept={metrics['retained_ratio']:.1%}"
    else:
        summary = 'No selected target - no filtered cloud'
    # Captions sit outside the image so masks and depth remain fully visible.
    panel = np.concatenate([color, depth], axis=1)
    header = np.zeros((64, panel.shape[1], 3), np.uint8)
    cv2.putText(header, 'Green=kept  Yellow=excluded target  Magenta=gripper',
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
    cv2.putText(header, f'Depth {backend.min_depth_m:.2f}-{backend.max_depth_m:.2f} m; black=invalid',
                (frame.depth_m.shape[1] + 8, 18), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
    cv2.putText(header, summary, (8, 38), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1)
    cv2.putText(header, 'Frame ' + frame.frame_id, (8, 56), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
    return np.concatenate([header, panel], axis=0)


def cloud_display_data(backend, result):
    """Always returns this snapshot's data; missing targets clear the previous cloud."""
    prepared = backend.last_prepared
    if prepared is None:
        return np.empty((0, 3)), np.empty((0, 3)), []
    cloud = prepared.cloud
    colors = cloud.colors_rgb.copy()
    if cloud.target_mask is not None:
        colors[cloud.target_mask] = [0.1, 1., 0.1]
    jaws = [_jaw_segments(candidate) for candidate in result.candidates]
    return cloud.points, colors, jaws


class CloudDiagnosticWindows:
    """Poll Open3D from the existing UI loop rather than blocking in draw_geometries.

    Closing either auxiliary window does not send a robot command. Preview frames
    update continuously; the data remain frozen during the B601 motion call.
    """
    def __init__(self, algorithm='OBB'):
        self.name = f'{algorithm} live segmentation | depth'
        self.algorithm = algorithm
        self.visualizer = None
        self.image_open = False
        self.has_view = False

    def start(self):
        import cv2
        import open3d as o3d
        self.o3d = o3d
        self.visualizer = o3d.visualization.Visualizer()
        if not self.visualizer.create_window(
                window_name=f'{self.algorithm} filtered cloud - green=target; orange=jaws; G in image window',
                width=800, height=600, left=980, top=50):
            self.close()
            raise RuntimeError('Cannot open point-cloud window; use --single-window to disable auxiliary views')
        self.visualizer.get_render_option().background_color = np.array([.08, .08, .08])
        self.visualizer.get_render_option().point_size = 2.
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        self.image_open = True
        cv2.resizeWindow(self.name, 960, 408)
        placeholder = np.zeros((544, 1280, 3), np.uint8)
        cv2.putText(placeholder, 'Waiting for live detection...',
                    (30, 100), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 1)
        cv2.imshow(self.name, placeholder)

    def poll(self):
        import cv2
        if self.image_open:
            try:
                self.image_open = cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE) >= 1
            except cv2.error:
                self.image_open = False
        if self.visualizer is not None:
            if not self.visualizer.poll_events():
                self.visualizer.destroy_window()
                self.visualizer = None
            else:
                self.visualizer.update_renderer()

    def update(self, capture, backend, result):
        import cv2
        self.poll()
        if self.image_open:
            cv2.imshow(self.name, diagnostic_image(capture, backend, result))
        if self.visualizer is None:
            return
        points, colors, jaws = cloud_display_data(backend, result)
        vis, o3d = self.visualizer, self.o3d
        vis.clear_geometries()
        if len(points):
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(points)
            cloud.colors = o3d.utility.Vector3dVector(colors)
            vis.add_geometry(cloud, reset_bounding_box=not self.has_view)
            for vertices in jaws:
                geometry = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(vertices),
                    lines=o3d.utility.Vector2iVector([[0, 1], [1, 2], [2, 3]]))
                geometry.paint_uniform_color([1., .65, 0.])
                vis.add_geometry(geometry, reset_bounding_box=False)
            if not self.has_view:
                control = vis.get_view_control()
                control.set_front([0, 0, -1])
                control.set_up([0, -1, 0])
                control.set_zoom(.7)
                self.has_view = True
        vis.poll_events()
        vis.update_renderer()

    def close(self):
        import cv2
        if self.visualizer is not None:
            self.visualizer.destroy_window()
            self.visualizer = None
        if self.image_open:
            try:
                cv2.destroyWindow(self.name)
            except cv2.error:
                pass
            self.image_open = False
