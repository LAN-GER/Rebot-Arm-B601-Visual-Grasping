from types import SimpleNamespace

import numpy as np
import pytest

from rebot_grasp.calibration.stationary_capture import Feedback, collect_pair


class Replay:
    def __init__(self):
        self.now = 1.
        self.seq = 0
        self.frames = 0

    def sleep(self, dt):
        self.now += dt

    def feedback(self):
        self.seq += 1
        return Feedback(np.full(6, self.now - 0.005), np.full(6, self.seq),
                        np.zeros(6), np.zeros(6))

    def frame(self):
        self.frames += 1
        return self.frames, self.now - 0.01, int((self.now - 0.01) * 1e6), self.frames

    def run(self, read=None, frame=None):
        return collect_pair(read or self.feedback, frame or self.frame,
                            lambda bgr: SimpleNamespace(id=0), clock=lambda: self.now,
                            sleep=self.sleep)


def test_capture_after_continuous_stillness_and_bracket_frame():
    replay = Replay()
    _, _, data = replay.run()
    stamp = data['frame_monotonic_s']
    ts = data['joint_feedback_monotonic_s']
    assert data['settled_monotonic_s'] <= stamp
    assert np.all(ts[0] + 1 <= stamp)
    assert np.all(ts[-1] >= stamp)
    assert np.all(np.diff(data['joint_sequences'], axis=0) > 0)


@pytest.mark.parametrize("failure", ["frozen_motor", "missing", "moving", "old_frame"])
def test_reject_unqualified_capture(failure):
    replay = Replay()

    def read():
        row = replay.feedback()
        if failure == "missing":
            return None
        if failure == "moving":
            row.position[0] = replay.now * .01
        if failure == "frozen_motor":
            row.sequence[3] = 7
        return row

    def frame():
        bgr, stamp, us, seq = replay.frame()
        return bgr, (0.5 if failure == "old_frame" else stamp), us, seq

    with pytest.raises(RuntimeError, match="timed out"):
        replay.run(read, frame)


def test_motion_after_frame_discards_pair_and_requires_new_window():
    replay = Replay()
    first_photo = []

    def frame():
        replay.frames += 1
        # Frame arrives just after the latest joint measurement.
        if not first_photo:
            first_photo.append(replay.now)
        return replay.frames, replay.now, int(replay.now * 1e6), replay.frames

    def read():
        row = replay.feedback()
        if first_photo:
            row.position[0] = .01
        return row

    bgr, _, data = replay.run(read, frame)
    assert bgr > 1
    assert data['frame_monotonic_s'] >= first_photo[0] + 1
    assert data['paired_joints_rad'][0] == pytest.approx(.01)
