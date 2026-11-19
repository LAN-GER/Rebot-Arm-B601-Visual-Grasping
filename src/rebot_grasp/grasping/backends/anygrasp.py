"""AnyGrasp's current create_detector/get_grasp API behind project contracts."""

import numpy as np

from rebot_grasp.core.types import GraspCandidate, RGBDFrame, Target
from rebot_grasp.geometry.transforms import transform
from rebot_grasp.perception.preprocessing import prepare_cloud, validate_preprocessing
from rebot_grasp.perception.spatial_filters import inside_workspace


class AnyGraspBackend:
    name = "anygrasp"

    def __init__(self, detector, *, min_depth_m=0.05, max_depth_m=1.0, dense_grasp=False,
                 collision_detection=True, approach_camera=None, approach_thresh_rad=np.pi,
                 outlier_filter=None, workspace_camera=None, gripper_exclusion=None,
                 pointcloud_quality=None, insertion_depth_m=0.0):
        self.detector = detector
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m
        if (isinstance(insertion_depth_m, bool) or not isinstance(insertion_depth_m, (int, float))
                or not np.isfinite(insertion_depth_m) or insertion_depth_m < 0):
            raise ValueError("insertion_depth_m must be finite and nonnegative")
        self.insertion_depth_m = float(insertion_depth_m)
        self.preprocessing = dict(outlier_filter=outlier_filter, workspace_camera=workspace_camera,
                                  gripper_exclusion=gripper_exclusion, pointcloud_quality=pointcloud_quality)
        validate_preprocessing(**self.preprocessing)
        self.last_preprocessing_report = None
        if not np.isfinite([min_depth_m, max_depth_m, approach_thresh_rad]).all():
            raise ValueError("AnyGrasp depth and angle settings must be finite")
        if not 0 <= min_depth_m < max_depth_m or not 0 <= approach_thresh_rad <= np.pi:
            raise ValueError("Invalid AnyGrasp depth or angle settings")
        if not isinstance(dense_grasp, bool) or not isinstance(collision_detection, bool):
            raise ValueError("AnyGrasp dense_grasp and collision_detection must be booleans")
        if approach_camera is not None:
            direction = np.asarray(approach_camera, dtype=float)
            if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-8:
                raise ValueError("approach_camera must be a nonzero 3-vector")
            approach_camera = direction.tolist()
        self.options = dict(dense_grasp=dense_grasp, collision_detection=collision_detection,
                            approach_steering=approach_camera, approach_thresh=approach_thresh_rad)

    def start(self):
        self.detector.start()

    def predict(self, frame: RGBDFrame, target: Target | None) -> list[GraspCandidate]:
        self.last_preprocessing_report = None
        prepared = prepare_cloud(frame, target, self.min_depth_m, self.max_depth_m, **self.preprocessing)
        return self._predict_prepared(frame, target, prepared)

    def _predict_prepared(self, frame, target, prepared):
        self.last_preprocessing_report = prepared.report
        if not prepared.report['quality']['passed']:
            return []
        return self._predict_cloud(frame, prepared.cloud, self.preprocessing['workspace_camera'])

    def _predict_cloud(self, frame, cloud, workspace):
        # The SDK interprets an all-false mask as DISABLING region steering.
        if len(cloud.points) == 0 or (cloud.target_mask is not None and not cloud.target_mask.any()):
            return []
        records = self.detector.predict(cloud.points, {**self.options, "region_steering": cloud.target_mask})
        result = []
        for record in records:
            rotation = np.asarray(record["rotation"], dtype=float)
            translation = np.asarray(record["translation"], dtype=float)
            depth = float(record["depth"])
            if not np.isfinite(depth) or depth < 0:
                continue
            if rotation.shape != (3, 3) or translation.shape != (3,):
                continue
            tip = translation + (depth + self.insertion_depth_m) * rotation[:, 0]
            # Region steering is a model hint; enforce the final fingertip bound too.
            if not inside_workspace(tip[None, :], workspace)[0]:
                continue
            try:
                result.append(GraspCandidate(transform(rotation, tip), float(record["width"]),
                                             float(record["score"]), self.name, frame.frame_id))
            except ValueError:
                continue
        return result

    def close(self):
        close = getattr(self.detector, "close", None)
        if close is not None:
            close()
