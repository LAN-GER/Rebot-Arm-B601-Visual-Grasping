"""Compare raw and preprocessed inputs to the same resident AnyGrasp detector."""
from rebot_grasp.grasping.backends.anygrasp import AnyGraspBackend
from rebot_grasp.grasping.filters import filter_candidates
from rebot_grasp.perception.pointcloud import target_pixel_mask


class FilteredAnyGraspBackend(AnyGraspBackend):
    name = "anygrasp_filtered"

    def __init__(self, detector, filters, **options):
        super().__init__(detector, **options)
        self.filters = filters
        self.reset_diagnostics()

    def reset_diagnostics(self):
        self.last_target = None
        self.last_prepared = None
        self.last_eligible_mask = None
        self.last_preprocessing_report = None
        self.reference = None
        self.selected = None

    def _predict_prepared(self, frame, target, prepared):
        if target is None:
            raise ValueError("Filtered AnyGrasp comparison requires a target")
        self.last_target = target
        self.last_prepared = prepared
        self.last_eligible_mask = target_pixel_mask(frame, prepared.cloud)
        raw = self._predict_cloud(frame, prepared.raw, workspace=None)
        ranked = filter_candidates(raw, **self.filters)
        self.reference = ranked[0] if ranked else None
        # Only the quality-gated branch is returned to planning and robot IK.
        return super()._predict_prepared(frame, target, prepared)
