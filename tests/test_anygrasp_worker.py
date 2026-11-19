import numpy as np
import pytest

from rebot_grasp.grasping.backends import _anygrasp_sdk as sdk


def replying_worker(connection, *args):
    connection.send((True, None))
    while True:
        request = connection.recv()
        if request is None:
            break
        points, options = request
        connection.send((True, [{"point_count": len(points)}]))
    connection.close()


def invalid_license_worker(connection, *args):
    connection.send((False, "license rejected"))
    connection.close()


def exiting_worker(connection, *args):
    connection.close()


def test_lazy_worker_reused_and_closed(tmp_path, monkeypatch):
    folder = tmp_path / "grasp_detection/license"
    folder.mkdir(parents=True)
    (folder / "licenseCfg.json").write_text("{}")
    checkpoint = tmp_path / "weights.tar"
    checkpoint.write_bytes(b"fake")
    monkeypatch.setattr(sdk, "_worker", replying_worker)
    detector = sdk.SDKDetector(tmp_path, checkpoint, .09, .03, timeout_s=5)
    assert detector._process is None
    try:
        detector.start()
        pid = detector._process.pid
        assert detector.predict(np.ones((4, 3)), {}) == [{"point_count": 4}]
        assert detector.predict(np.ones((8, 3)), {}) == [{"point_count": 8}]
        assert detector._process.pid == pid
    finally:
        detector.close()
    assert detector._process is None and detector._connection is None


@pytest.mark.parametrize("worker,match", [(invalid_license_worker, "license rejected"),
                                         (exiting_worker, "exited unexpectedly")])
def test_worker_startup_failure_releases_process(tmp_path, monkeypatch, worker, match):
    folder = tmp_path / "grasp_detection/license"
    folder.mkdir(parents=True)
    (folder / "licenseCfg.json").write_text("{}")
    checkpoint = tmp_path / "weights.tar"
    checkpoint.write_bytes(b"fake")
    monkeypatch.setattr(sdk, "_worker", worker)
    detector = sdk.SDKDetector(tmp_path, checkpoint, .09, .03, timeout_s=5)
    with pytest.raises(RuntimeError, match=match):
        detector.start()
    assert detector._process is None and detector._connection is None
