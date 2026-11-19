"""Load the GraspNet-Baseline repository's flat modules only when requested."""

from pathlib import Path
import sys


def prepare_imports(sdk_root: Path) -> None:
    model_file = sdk_root / "models" / "graspnet.py"
    if not model_file.is_file():
        raise FileNotFoundError(f"GraspNet-Baseline source not found: {model_file}")
    paths = [sdk_root, *(sdk_root / name for name in ("models", "dataset", "utils", "pointnet2", "graspnetAPI"))]
    for directory in reversed(paths):
        path = str(directory)
        if path not in sys.path:
            sys.path.insert(0, path)


def build_net(checkpoint: Path, num_view: int):
    if not checkpoint.is_file():
        raise FileNotFoundError(f"GraspNet checkpoint not found: {checkpoint}")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("GraspNet pointnet2 operators require CUDA-enabled PyTorch")

    from graspnet import GraspNet

    device = torch.device("cuda:0")
    model = GraspNet(
        input_feature_dim=0, num_view=num_view, num_angle=12, num_depth=4,
        cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False,
    ).to(device)
    state = torch.load(str(checkpoint), map_location=device)
    model.load_state_dict(state["model_state_dict"])
    return model.eval()


def infer_group(model, end_points, scene_points, collision_thresh, voxel_size):
    import torch
    from collision_detector import ModelFreeCollisionDetector
    from graspnet import pred_decode
    from graspnetAPI import GraspGroup

    with torch.no_grad():
        decoded = pred_decode(model(end_points))
    group = GraspGroup(decoded[0].detach().cpu().numpy())
    initial_count = len(group)
    if initial_count and collision_thresh > 0:
        detector = ModelFreeCollisionDetector(scene_points, voxel_size=voxel_size)
        collision = detector.detect(group, approach_dist=0.05, collision_thresh=collision_thresh)
        group = group[~collision]
    return group, initial_count


class GraspNetDetector:
    def __init__(self, sdk_root, checkpoint, num_view):
        self.sdk_root = Path(sdk_root).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        self.num_view = num_view
        self._model = None

    def predict(self, sampled_points, scene_points, collision_thresh, voxel_size):
        if self._model is None:
            prepare_imports(self.sdk_root)
            self._model = build_net(self.checkpoint, self.num_view)

        import torch

        batch = torch.from_numpy(sampled_points[None]).to("cuda:0")
        group, _ = infer_group(
            self._model, {"point_clouds": batch}, scene_points, collision_thresh, voxel_size,
        )
        return group.nms().sort_by_score() if len(group) else group
