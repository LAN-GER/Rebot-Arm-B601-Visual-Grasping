"""Filtered OBB live preview with same-frame point-cloud diagnostics."""
import time
from types import SimpleNamespace

from rebot_grasp.core.types import Capture
from rebot_grasp.grasping.backends.obb_filtered import FilteredOBBBackend
from rebot_grasp.pipeline.live_preview import LivePreview, PreviewFrame
from rebot_grasp.pipeline.planning import plan_capture
from rebot_grasp.visualization.obb_comparison import comparison_image


class LiveOBBPreview(LivePreview):
    def __init__(self, detector, config):
        self.detector, self.config = detector, config
        self.backend = FilteredOBBBackend(**config.backend_options)
        super().__init__("obb-preview")

    def _process(self, frame):
        start = time.monotonic()
        capture = Capture(frame)
        targets = self.detector.detect(frame)
        self.backend.reset_diagnostics()
        result = plan_capture(capture, targets, self.backend, self.config)
        # Snapshot references instead of sharing a backend whose fields are reset
        # by the next inference. These arrays are never mutated by later frames.
        view = SimpleNamespace(**{name: getattr(self.backend, name) for name in (
            'last_prepared', 'last_target', 'last_eligible_mask', 'last_comparison',
            'min_depth_m', 'max_depth_m')})
        return PreviewFrame(capture, view, result, comparison_image(capture, view, result),
                            time.monotonic() - start)
