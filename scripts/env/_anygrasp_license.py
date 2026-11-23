"""Read the SDK feature ID or install the detection license and weights."""
from contextlib import contextmanager, ExitStack
import ctypes
import fcntl
import filecmp
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
from zipfile import Path as ZipPath, ZipFile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
CONFIG = "licenseCfg.json"
WEIGHTS = "checkpoint_detection.tar"


def find_bundle(path):
    if not path.exists():
        raise FileNotFoundError(f"License path does not exist: {path}")
    if path.is_dir():
        if (path / CONFIG).is_file():
            return path
        candidates = [p for p in path.iterdir() if p.name != "__MACOSX" and
                      ((p.is_dir() and (p / CONFIG).is_file()) or
                       (p.is_file() and p.suffix.lower() == ".zip"))]
        if len(candidates) == 1:
            return candidates[0]
        if candidates:
            raise ValueError("Multiple licenses found. Specify one license directory or ZIP file.")
        raise FileNotFoundError(f"No {CONFIG} or license ZIP file found in {path}.")
    if path.name == CONFIG:
        return path.parent
    if path.suffix.lower() != ".zip":
        raise ValueError("Provide a license directory or ZIP file.")
    return path


def copy_files(folder, stage):
    # The official JSON names its three companion files. Never extract a whole
    # archive or copy unrelated files (e.g. model checkpoints) from the source.
    def copy(name):
        source = folder / name
        if not source.is_file() or (isinstance(source, Path) and source.is_symlink()):
            raise FileNotFoundError(f"The license bundle must contain a regular file named {name}.")
        with source.open("rb") as src, (stage / name).open("xb") as dest:
            shutil.copyfileobj(src, dest)
        if (stage / name).stat().st_size == 0:
            raise ValueError(f"License file is empty: {name}")

    copy(CONFIG)
    config = json.loads((stage / CONFIG).read_bytes())
    if not isinstance(config, dict):
        raise ValueError(f"{CONFIG} must contain a JSON object.")
    for key, suffix in (("license", ".lic"), ("signature", ".signature"), ("public_key", ".public_key")):
        name = config.get(key)
        if not isinstance(name, str) or Path(name).name != name or "\\" in name or not name.endswith(suffix):
            raise ValueError(f"The '{key}' field in {CONFIG} must be a {suffix} filename in the same directory.")
        copy(name)


@contextmanager
def delivery_files(source, installed_license):
    root = source if source.is_dir() else source.parent
    external = root / WEIGHTS
    if source.is_file() and source.name == WEIGHTS:
        bundle = installed_license
    else:
        try:
            bundle = find_bundle(source)
        except FileNotFoundError:
            if not (source.is_dir() and external.is_file()):
                raise
            bundle = installed_license
    if bundle == installed_license and not (bundle / CONFIG).is_file():
        raise FileNotFoundError("Detection weights were supplied without a license. Include the issued license in the delivery directory.")

    weights = [external] if external.is_file() else []
    with ExitStack() as stack:
        if bundle.is_dir():
            folder = bundle
            nested = folder / WEIGHTS
            if nested != external and nested.is_file():
                weights.append(nested)
        else:
            archive = stack.enter_context(ZipFile(bundle))
            members = [entry.filename for entry in archive.infolist() if not entry.is_dir()
                       and "__MACOSX" not in PurePosixPath(entry.filename).parts]
            configs = [n for n in members if PurePosixPath(n).name == CONFIG]
            if len(configs) != 1:
                raise ValueError(f"Expected exactly one {CONFIG} in the license ZIP. Found {len(configs)}.")
            folder = ZipPath(archive, at=configs[0][:-len(CONFIG)])
            weights.extend(ZipPath(archive, at=n) for n in members if PurePosixPath(n).name == WEIGHTS)
        if len(weights) > 1:
            raise ValueError(f"Multiple copies of {WEIGHTS} found. Supply one detection checkpoint.")
        yield folder, next(iter(weights), None)


@contextmanager
def configure_weights(source, target):
    from _environment import model_checkpoint

    if source is None:
        if target.exists():
            model_checkpoint(target)
            yield f"Detection weights already configured at {target}"
        else:
            yield None
        return

    if isinstance(source, Path) and target.is_file() and filecmp.cmp(source, target, shallow=False):
        model_checkpoint(target)
        yield f"Detection weights already configured at {target}"
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".weights-stage-") as work:
        staged = Path(work) / WEIGHTS
        with source.open("rb") as src, staged.open("xb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
        model_checkpoint(staged)
        if target.is_file() and filecmp.cmp(staged, target, shallow=False):
            yield f"Detection weights already configured at {target}"
            return
        previous = Path(work) / "previous"
        if target.exists():
            os.link(target, previous)
        os.replace(staged, target)
        try:
            yield f"Detection weights installed at {target}"
        except BaseException:
            if previous.exists():
                os.replace(previous, target)
            else:
                target.unlink()
            raise


def install_directory(stage, target, parent_fd):
    if not target.exists():
        stage.rename(target)
        return None

    # Linux renameat2(RENAME_EXCHANGE) keeps the SDK's license path valid even
    # if another process opens it during replacement or this installer exits.
    exchange = ctypes.CDLL(None, use_errno=True).renameat2
    backup = Path(tempfile.mkdtemp(dir=target.parent, prefix=".license-backup-")) / "license"
    try:
        stage.rename(backup)
        if exchange(parent_fd, os.fsencode(backup), parent_fd, os.fsencode(target), 2) != 0:
            errno = ctypes.get_errno()
            raise OSError(errno, os.strerror(errno), str(target))
    except Exception:
        shutil.rmtree(backup.parent)
        raise
    return backup


def configure(sdk, sdk_dir, source, checkpoint=None):
    from rebot_grasp.grasping.backends._anygrasp_sdk import verify_license

    target = sdk_dir / "license"
    checkpoint = checkpoint or sdk_dir / "log" / WEIGHTS
    fd = os.open(sdk_dir, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        with (
            delivery_files(source, target) as (folder, weights),
            tempfile.TemporaryDirectory(dir=sdk_dir, prefix=".license-stage-") as work,
        ):
            stage = Path(work) / "license"
            stage.mkdir(mode=0o700)
            copy_files(folder, stage)
            verify_license(sdk, stage)
            if target.is_symlink() or (target.exists() and not target.is_dir()):
                raise ValueError(f"The license destination must be a directory, and cannot be a symbolic link: {target}")
            same_license = target.is_dir() and all(
                (target / p.name).is_file() and not (target / p.name).is_symlink()
                and filecmp.cmp(target / p.name, p, shallow=False) for p in stage.iterdir())
            with configure_weights(weights, checkpoint) as weight_status:
                if same_license:
                    message = f"The SDK verified the existing license at {target}."
                else:
                    backup = install_directory(stage, target, fd)
                    message = f"License verified by the SDK and installed at {target}"
                    if backup:
                        message += f"\nPrevious license directory backed up to {backup}"
                if weight_status is None:
                    return f"{message}\nMissing detection weights: supply {WEIGHTS} for {checkpoint}", False
                return f"{message}\n{weight_status}", True
    finally:
        os.close(fd)


def main(backend, action, *args):
    from rebot_grasp.config import load_backend_options
    from rebot_grasp.grasping.backends._anygrasp_sdk import load_sdk_module, prepare_sdk_binary
    import torch  # Load the libtorch libraries used by the native gsnet module.

    source = Path(args[0]).expanduser().resolve() if action == "2" else None
    options = load_backend_options(ROOT / "config/default.yaml", backend)
    sdk_dir = Path(options["sdk_root"]) / "grasp_detection"
    os.chdir(sdk_dir)
    if action == "2":
        installed = prepare_sdk_binary(sdk_dir)
        if installed is not None:
            print(f"AnyGrasp extension installed at {installed}")
    sdk = load_sdk_module(sdk_dir)
    if action == "1":
        feature = sdk.get_feature_id()
        feature = feature.strip().removesuffix("%") if isinstance(feature, str) else ""
        if not re.fullmatch(r"N[0-9]+", feature):
            raise RuntimeError("The AnyGrasp SDK returned an invalid machine feature ID.")
        print(f"Machine feature ID (feature_id): {feature}")
        return 0
    message, complete = configure(sdk, sdk_dir, source, Path(options["checkpoint"]))
    print(message)
    return 0 if complete else 3


if __name__ == "__main__":
    try:
        sys.exit(main(*sys.argv[1:]))
    except Exception as error:
        sys.exit(f"{type(error).__name__}: {error}")
