"""Compare two point-cloud inputs per frame and dispatch only filtered plans."""
from types import SimpleNamespace
import time

from rebot_grasp.core.types import Capture
from rebot_grasp.pipeline import live_session
from rebot_grasp.pipeline.live_preview import LivePreview, PreviewFrame
from rebot_grasp.pipeline.instance_preview import plan_instance_previews
from rebot_grasp.visualization.anygrasp_comparison import comparison_image, save_comparison
from rebot_grasp.visualization.cloud_windows import CloudDiagnosticWindows


class ComparisonPreview(LivePreview):
    def __init__(self, detector, backend, config):
        super().__init__("anygrasp-comparison")
        self.detector, self.backend, self.config = detector, backend, config

    def _process(self, frame):
        start = time.monotonic()
        capture = Capture(frame)
        result, _, instances = plan_instance_previews(capture, self.detector.detect(frame), self.backend, self.config)
        # The worker can start another frame while the UI renders this one's cloud.
        view = SimpleNamespace(**{key: getattr(self.backend, key) for key in
                                 ("last_prepared", "last_target", "last_eligible_mask", "reference",
                                  "selected", "min_depth_m", "max_depth_m")})
        return PreviewFrame(capture, view, result, comparison_image(capture, view, result, instances=instances), time.monotonic() - start)


def infer_snapshot(capture, detector, backend, config, output, robot=None):
    backend.reset_diagnostics()
    result, plan = live_session.infer_snapshot(capture, detector, backend, config, output, robot)
    backend.selected = plan.candidate if plan else None
    save_comparison(output, capture, backend, result)
    print(f"[AnyGrasp filtered] {result.status}; saved {output}", flush=True)
    return result, plan


def run_live(camera, backend, detector, config, output, robot=None, hand_eye=None, *,
             show_diagnostics=True):
    preview = ComparisonPreview(detector, backend, config)
    try:
        return live_session.run_live(
            camera, backend, detector, config, output, robot, hand_eye,
            live_preview=preview, snapshot_handler=infer_snapshot, snapshot_renderer=comparison_image,
            diagnostics=CloudDiagnosticWindows("AnyGrasp") if show_diagnostics else None,
            window_name="AnyGrasp raw reference | filtered execution", freeze_after_snapshot=False,
        )
    finally:
        preview.close()
