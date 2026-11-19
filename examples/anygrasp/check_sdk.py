"""Check AnyGrasp's local binary, CUDA extensions and required files."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config/default.yaml"))
    parser.add_argument("--feature-id", action="store_true")
    args = parser.parse_args()
    from rebot_grasp.config import load_config
    from rebot_grasp.grasping.backends._anygrasp_sdk import load_sdk_module

    report = {}
    try:
        import torch
        import MinkowskiEngine as ME
        from pointnet2 import pointnet2_utils
        import graspnetAPI

        config = load_config(args.config, "anygrasp")
        options = config.backend_options
        sdk_dir = Path(options["sdk_root"]) / "grasp_detection"
        module = load_sdk_module(sdk_dir)
        report.update(python=sys.version.split()[0], torch=torch.__version__,
                      cuda_build=torch.version.cuda, sdk_directory=str(sdk_dir))
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device unavailable")
        report["gpu"] = torch.cuda.get_device_name(0)
        points = torch.rand(1, 32, 3, device="cuda")
        indices = pointnet2_utils.furthest_point_sample(points, 8)
        coords = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.int32)
        sparse = ME.SparseTensor(torch.ones(2, 3, device="cuda"), coordinates=coords, device="cuda")
        convolution = ME.MinkowskiConvolution(3, 4, kernel_size=1, dimension=3).cuda()
        output = convolution(sparse)
        torch.cuda.synchronize()
        report["cuda_extensions_passed"] = bool(indices.shape == (1, 8) and output.F.shape == (2, 4))
        report["graspnet_api"] = graspnetAPI.__file__
        if args.feature_id:
            report["feature_id"] = module.get_feature_id()
        report["license_config_present"] = (sdk_dir / "license/licenseCfg.json").is_file()
        report["checkpoint_present"] = Path(options["checkpoint"]).is_file()
        report["license_validated"] = False
        report["model_inference_validated"] = False
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(report, indent=2))
    return 1 if "error" in report else 0


if __name__ == "__main__":
    raise SystemExit(main())
