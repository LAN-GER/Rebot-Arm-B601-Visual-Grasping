"""Pair a new Orbbec color frame with stationary B601 encoder feedback."""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np


STILL_S = 1.0
MAX_SPAN_RAD = 0.002
MAX_GAP_S = 0.15
TIMEOUT_S = 6.0


@dataclass
class Feedback:
    timestamp: np.ndarray   # Host monotonic seconds, one receipt per joint
    sequence: np.ndarray
    position: np.ndarray
    velocity: np.ndarray


def read_feedback(controller, motion_fault):
    arm = controller.rebotarm
    fault = motion_fault(arm)
    if fault:
        raise RuntimeError(f"Motion fault: {fault}")
    if controller._moving or controller._send_thread.is_alive():
        return None
    arm.arm._poll_feedback()
    rows = []
    for joint in arm.arm._jcfgs:
        snap = arm._motor_map[joint.name].get_state_snapshot()
        now = time.monotonic()
        if snap is None or not 0 <= snap.feedback_age_s <= MAX_GAP_S:
            return None
        if snap.state.fault_code:
            raise RuntimeError(f"Motor fault on {joint.name}: {snap.state.fault_code}")
        rows.append((now - snap.feedback_age_s, snap.feedback_sequence,
                     snap.state.pos, snap.state.vel))
    t, seq, q, dq = np.array(rows).T
    if len(q) != 6 or not np.isfinite([t, seq, q, dq]).all():
        return None
    return Feedback(t, seq.astype(np.int64), q, dq)


class StillWindow:
    def __init__(self):
        self.rows = []

    def update(self, row):
        if row is None:
            self.rows.clear()
            return False
        if self.rows:
            last = self.rows[-1]
            gap = row.timestamp - last.timestamp
            if (np.any(row.sequence < last.sequence) or np.any(gap > MAX_GAP_S)
                    or np.any((row.sequence > last.sequence) & (gap <= 0))):
                self.rows.clear()
            elif np.any(row.sequence == last.sequence):
                return False
        self.rows.append(row)
        if np.any(np.ptp([r.position for r in self.rows], axis=0) > MAX_SPAN_RAD):
            self.rows = [row]
        return bool(np.all(row.timestamp - self.rows[0].timestamp >= STILL_S))


def read_color(camera):
    """Read the same configured color stream, retaining the SDK frame timestamp."""
    import cv2
    from pyorbbecsdk import OBFormat

    frames = camera._pipeline.wait_for_frames(100)
    color = frames.get_color_frame() if frames is not None else None
    if color is None:
        return None
    system_us = color.get_system_timestamp_us()
    age = time.time() - system_us * 1e-6
    received = time.monotonic()
    if not 0 <= age <= MAX_GAP_S:
        return None
    raw = np.asanyarray(color.get_data(), dtype=np.uint8)
    if color.get_format() == OBFormat.MJPG:
        bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    elif color.get_format() == OBFormat.RGB:
        rgb = raw.reshape(color.get_height(), color.get_width(), 3)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    else:
        raise RuntimeError(f"Unsupported B601 color format: {color.get_format()}")
    if bgr is None:
        return None
    return bgr, received - age, system_us, color.get_index()


def collect_pair(feedback, read_frame, detect, *, clock=time.monotonic, sleep=time.sleep):
    window = StillWindow()
    pending = None
    ready_at = None
    reason = "waiting for continuous encoder feedback and stationary joints"
    deadline = clock() + TIMEOUT_S
    while clock() < deadline:
        row = feedback()
        previous_start = window.rows[0] if window.rows else None
        settled = window.update(row)
        if not window.rows or window.rows[0] is not previous_start:
            ready_at = pending = None
            reason = "feedback interrupted or joint movement exceeded the limit"
        if settled:
            if ready_at is None:
                ready_at = clock()
            if pending is None:
                frame = read_frame()
                if frame is not None and ready_at <= frame[1] <= clock():
                    marker = detect(frame[0])
                    if marker is not None:
                        pending = (frame, marker)
                    else:
                        reason = "target marker missing from fresh frame"
                else:
                    reason = "waiting for color frame acquired after settling"
            if pending is not None:
                (bgr, stamp, system_us, frame_id), marker = pending
                if np.all(row.timestamp >= stamp):
                    ts = np.array([r.timestamp for r in window.rows])
                    q = np.array([r.position for r in window.rows])
                    paired_q = np.array([np.interp(stamp, ts[:, j], q[:, j])
                                         for j in range(q.shape[1])])
                    evidence = dict(frame_index=frame_id, frame_system_us=system_us,
                                    frame_monotonic_s=stamp, settled_monotonic_s=ready_at,
                                    joint_feedback_monotonic_s=ts, joint_positions_rad=q,
                                    joint_sequences=np.array([r.sequence for r in window.rows]),
                                    reported_velocities_rad_s=np.array([r.velocity for r in window.rows]),
                                    paired_joints_rad=paired_q)
                    return bgr, marker, evidence
        sleep(0.02)
    raise RuntimeError(f"Stationary capture timed out: {reason}")


def capture_stationary(camera, controller, *, motion_fault):
    from reBotArm_control_py.kinematics import compute_fk, pad_q_for_model

    print("[Capture] Waiting for 1 s of stationary encoder feedback.", flush=True)
    bgr, marker, evidence = collect_pair(lambda: read_feedback(controller, motion_fault),
                                         lambda: read_color(camera), camera.detect_aruco)
    model = controller._model
    q = pad_q_for_model(model, evidence["paired_joints_rad"], 6)
    tcp = compute_fk(model, q)[2]
    print(f"[Capture] Accepted stationary frame {evidence['frame_index']}.", flush=True)
    return bgr, tcp, marker.T_marker2cam, evidence
