from dataclasses import replace
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np

from rebot_grasp.config import load_config
from rebot_grasp.core.types import Capture
from rebot_grasp.pipeline.obb_preview import LiveOBBPreview, PreviewFrame
from rebot_grasp.pipeline.planning import PlanningResult
from rebot_grasp.pipeline.live_session import run_live


def preview_config():
    cfg = load_config(Path(__file__).resolve().parents[1] / 'config/obb_filtered.yaml')
    return replace(cfg, target_class='synthetic-block',
                   backend_options={**cfg.backend_options, 'gripper_exclusion': None})


def test_preview_drops_busy_frames_and_drains_before_snapshot(scene):
    started, finish = Event(), Event()
    worker = LiveOBBPreview(None, preview_config())
    calls = []
    def process(frame):
        calls.append(frame.frame_id)
        started.set()
        assert finish.wait(2)
        return frame.frame_id
    worker._process = process
    try:
        worker.tick(replace(scene[0].frame, frame_id='first'))
        assert started.wait(2)
        future = worker.future
        worker.tick(replace(scene[0].frame, frame_id='discarded'))
        assert worker.future is future and calls == ['first']
        finish.set()
        assert future.result(timeout=2) == 'first'
        assert worker.tick(replace(scene[0].frame, frame_id='newest')) == 'first'
        worker.pause()
        assert worker.future is None and worker.latest is None
        assert calls == ['first', 'newest']
    finally:
        finish.set()
        worker.close()


def test_preview_draws_detection_and_does_not_mutate_previous_view(scene):
    capture, target = scene
    detector = SimpleNamespace(detect=lambda frame: [target] if frame.frame_id == 'found' else [])
    worker = LiveOBBPreview(detector, preview_config())
    try:
        first = worker._process(replace(capture.frame, frame_id='found'))
        assert first.backend_view.last_target.label == 'synthetic-block'
        assert first.result.candidates and not first.result.plans
        y, x = np.where(target.mask)
        # The blue OBB may overlap the green detection border on rectangles.
        border = first.image[y.min()-1:y.max()+2, x.min()-1:x.max()+2]
        assert np.any(np.all(border == [40, 230, 40], axis=-1))
        second = worker._process(replace(capture.frame, frame_id='absent'))
        assert second.result.status == 'target_not_found'
        assert second.backend_view.last_prepared is None
        assert first.backend_view.last_prepared is not None
        assert first.capture.frame.frame_id == 'found'
    finally:
        worker.close()


def test_live_updates_views_without_g_and_never_runs_snapshot_or_robot(scene, tmp_path, monkeypatch):
    import cv2
    import itertools
    keys = iter([-1, -1, ord('q')])
    monkeypatch.setattr(cv2, 'namedWindow', lambda *a: None)
    monkeypatch.setattr(cv2, 'imshow', lambda *a: None)
    monkeypatch.setattr(cv2, 'waitKey', lambda _: next(keys))
    monkeypatch.setattr(cv2, 'getWindowProperty', lambda *a: 1)
    monkeypatch.setattr(cv2, 'destroyAllWindows', lambda: None)
    frames = itertools.count()
    camera = SimpleNamespace(open=lambda: None, close=lambda: None,
                              capture=lambda: replace(scene[0].frame, frame_id=str(next(frames))))
    backend = SimpleNamespace(start=lambda: None, close=lambda: None)
    result = PlanningResult([], [], 'camera_preview_only_missing_transforms')
    class Preview:
        latest = None
        closed = False
        def tick(self, frame):
            self.latest = PreviewFrame(Capture(frame), None, result, frame.color_bgr, .02)
            return self.latest
        def close(self):
            self.closed = True
    preview = Preview()
    updated = []
    diagnostics = SimpleNamespace(start=lambda: None, poll=lambda: None, close=lambda: None,
                                   update=lambda c, b, r: updated.append(c.frame.frame_id))
    def forbidden(*args):
        raise AssertionError('Preview must not invoke snapshot/execution')
    run_live(camera, backend, None, preview_config(), tmp_path / 'run',
              snapshot_handler=forbidden, diagnostics=diagnostics, live_preview=preview)
    assert len(updated) == 3 and len(set(updated)) == 3
    assert preview.closed
    assert not list((tmp_path / 'run').glob('snapshot-*'))
