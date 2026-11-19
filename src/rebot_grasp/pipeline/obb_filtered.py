"""Filtered OBB snapshots with paired evidence and unchanged B601 robot actions."""
from rebot_grasp.pipeline import live_session
from rebot_grasp.visualization.obb_comparison import comparison_image, save_comparison
from rebot_grasp.core.types import Capture


def infer_snapshot(capture, detector, backend, config, output, robot=None):
    # A missing target bypasses predict(), so clear prior-frame diagnostics here.
    backend.reset_diagnostics()
    result, plan = live_session.infer_snapshot(capture, detector, backend, config, output, robot)
    save_comparison(output, capture, backend, result)
    print(f'[OBB filtered] {result.status}; saved {output}', flush=True)
    if backend.last_comparison is not None:
        print(f"[OBB comparison] {backend.last_comparison['delta']}", flush=True)
    return result, plan


def run_live(camera, backend, detector, config, output, robot=None, hand_eye=None, *, show_diagnostics=True):
    from rebot_grasp.visualization.obb_windows import OBBDiagnosticWindows
    from rebot_grasp.pipeline.obb_preview import LiveOBBPreview
    preview = LiveOBBPreview(detector, config)
    try:
        return live_session.run_live(camera, backend, detector, config, output, robot, hand_eye,
                                 snapshot_handler=infer_snapshot, snapshot_renderer=comparison_image,
                                 window_name='OBB raw reference | filtered execution',
                                 capture_handler=capture_at_rest,
                                 diagnostics=OBBDiagnosticWindows() if show_diagnostics else None,
                                 live_preview=preview, freeze_after_snapshot=False)
    finally:
        # Also cover failures before the shared session enters its cleanup block.
        preview.close()


def capture_at_rest(camera, robot, hand_eye):
    """B601-style single pose read while the robot is at rest awaiting G.

    No extra encoder-stability threshold or altered trajectory is introduced.
    The image and pose are not hardware-synchronized.
    """
    frame = camera.capture()
    return Capture(frame, robot.tcp_pose() @ hand_eye)
