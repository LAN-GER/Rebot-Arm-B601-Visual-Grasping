"""AnyGrasp live target preview and on-demand grasp snapshots."""
import time

from rebot_grasp.core.types import Capture
from rebot_grasp.pipeline.instance_preview import plan_instance_previews
from rebot_grasp.pipeline import live_session
from rebot_grasp.pipeline.live_preview import LivePreview, PreviewFrame
from rebot_grasp.pipeline.planning import plan_capture
from rebot_grasp.visualization.overlay import draw_candidates, draw_instance_grasps
from rebot_grasp.pipeline.live_session import execute_snapshot, infer_snapshot

__all__ = ["infer_snapshot", "execute_snapshot", "run_live"]


class LiveGraspPreview(LivePreview):
    """Project one filtered AnyGrasp candidate onto its source RGB-D frame."""

    def __init__(self, detector, backend, config):
        super().__init__("anygrasp-preview")
        self.detector = detector
        self.backend = backend
        self.config = config

    def _process(self, frame):
        start = time.monotonic()
        capture = Capture(frame)
        targets = [] if self.config.full_scene else self.detector.detect(frame)
        if self.config.full_scene:
            result = plan_capture(capture, targets, self.backend, self.config)
        else:
            result, selected, instances = plan_instance_previews(capture, targets, self.backend, self.config)
        best = result.candidates[:1]
        label = f"GRASP PREVIEW  {best[0].score:.2f}" if best else result.status.upper().replace("_", " ")
        image = draw_candidates(frame, best if self.config.full_scene else [], label=label, line_color=(255, 255, 255))
        if not self.config.full_scene:
            draw_instance_grasps(image, frame, instances, selected)
        return PreviewFrame(capture, None, result, image, time.monotonic() - start)


def run_live(camera, backend, detector, config, output, robot=None, hand_eye=None, **options):
    preview = LiveGraspPreview(detector, backend, config)
    try:
        return live_session.run_live(camera, backend, detector, config, output, robot, hand_eye,
                                     live_preview=preview, **options)
    finally:
        preview.close()
