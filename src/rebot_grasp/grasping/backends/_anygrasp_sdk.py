"""Run the binary SDK in an isolated process with its required working directory.

The model remains loaded across requests. This is local IPC, not a network service.
The parent never changes cwd or imports torch/gsnet. SDK stdout cannot corrupt IPC.
"""

from argparse import Namespace
import importlib.util
import multiprocessing as mp
import os
from pathlib import Path
import sysconfig


def load_sdk_module(sdk_dir):
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    directory = Path(sdk_dir)
    candidates = [directory / ("gsnet" + suffix), directory / "gsnet.so",
                  directory / "gsnet_versions" / ("gsnet" + suffix)]
    binary = next((path.resolve() for path in candidates if path.is_file()), None)
    if binary is None:
        raise FileNotFoundError(f"No gsnet binary for this Python ABI ({suffix}) in {sdk_dir}")
    spec = importlib.util.spec_from_file_location("gsnet", binary)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "create_detector"):
        raise RuntimeError("This adapter requires the current create_detector/get_grasp SDK API")
    return module


def _worker(connection, sdk_dir, checkpoint, max_width, gripper_height):
    try:
        os.chdir(sdk_dir)  # SDK-relative license lookup, confined to this process.
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("AnyGrasp requires CUDA-enabled PyTorch and an available NVIDIA GPU")
        module = load_sdk_module(sdk_dir)
        detector = module.create_detector(Namespace(checkpoint_path=checkpoint, max_gripper_width=max_width, gripper_height=gripper_height))
        if detector is None:
            raise RuntimeError("AnyGrasp initialization failed; check machine license and checkpoint")
        connection.send((True, None))
        while True:
            request = connection.recv()
            if request is None:
                break
            points, options = request
            grasps = detector.get_grasp(points, options)
            records = []
            if grasps is not None and len(grasps):
                if not options.get("dense_grasp", False):
                    grasps = grasps.nms()
                grasps = grasps.sort_by_score()
                for grasp in grasps:
                    records.append(dict(rotation=grasp.rotation_matrix, translation=grasp.translation,
                                        depth=grasp.depth, width=grasp.width, score=grasp.score))
            connection.send((True, records))
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        try:
            connection.send((False, f"{type(exc).__name__}: {exc}"))
        except (EOFError, BrokenPipeError):
            pass
    finally:
        connection.close()


class SDKDetector:
    def __init__(self, sdk_root, checkpoint, max_width_m, gripper_height_m, timeout_s=120):
        self.sdk_dir = (Path(sdk_root).expanduser().resolve() / "grasp_detection")
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.sdk_dir.is_dir():
            raise FileNotFoundError(f"AnyGrasp SDK not found: {self.sdk_dir}")
        if not (self.sdk_dir / "license" / "licenseCfg.json").is_file():
            raise FileNotFoundError(f"AnyGrasp license missing under {self.sdk_dir / 'license'}")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"AnyGrasp checkpoint missing: {self.checkpoint}")
        if not 0 < max_width_m <= 0.1 or not 0 < gripper_height_m < 0.2 or not 0 < timeout_s < float("inf"):
            raise ValueError("Invalid AnyGrasp gripper dimensions or timeout")
        self.max_width = max_width_m
        self.gripper_height = gripper_height_m
        self.timeout_s = timeout_s
        self._process = None
        self._connection = None

    def _receive(self):
        if not self._connection.poll(self.timeout_s):
            self.close()
            raise TimeoutError("AnyGrasp SDK timed out; worker stopped")
        try:
            ok, payload = self._connection.recv()
        except EOFError as exc:
            self.close()
            raise RuntimeError("AnyGrasp SDK worker exited unexpectedly") from exc
        if not ok:
            self.close()
            raise RuntimeError(payload)
        return payload

    def start(self):
        if self._process is None:
            ctx = mp.get_context("spawn")
            self._connection, child = ctx.Pipe()
            self._process = ctx.Process(target=_worker, args=(child, str(self.sdk_dir), str(self.checkpoint), self.max_width, self.gripper_height), daemon=True)
            self._process.start()
            child.close()
            self._receive()

    def predict(self, points, options):
        self.start()
        try:
            self._connection.send((points, options))
        except (BrokenPipeError, OSError) as exc:
            self.close()
            raise RuntimeError("AnyGrasp SDK worker is unavailable") from exc
        return self._receive()

    def close(self):
        if self._process is not None:
            if self._connection is not None:
                try:
                    self._connection.send(None)
                except (BrokenPipeError, OSError):
                    pass
            self._process.join(timeout=1)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2)
            self._process = None
        if self._connection is not None:
            self._connection.close()
            self._connection = None
