"""Check public environment commands without modifying a real Conda environment."""
from pathlib import Path
import os
import subprocess
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]


def run_script(name, *args, **env):
    return subprocess.run(["bash", str(ROOT / "scripts" / name), *args], cwd="/tmp",
                          env={**os.environ, **env}, text=True, capture_output=True)


@pytest.fixture
def fake_conda(tmp_path):
    conda = tmp_path / "conda"
    conda.write_text("""#!/bin/sh
case "$*" in
    *" -I -c "*) echo 3.10.22 ;;
    *"_environment.py check "*) echo "$*" ;;
    *) exit 97 ;;
esac
""")
    conda.chmod(0o755)
    return str(conda)


@pytest.mark.parametrize(("entry", "args", "expected"), [
    ("check_env.sh", [], "all 1"),
    ("check_env.sh", ["--backend", "obb"], "obb 0"),
    ("check_env.sh", ["--backend", "anygrasp", "--robot"], "anygrasp 1"),
    ("check_graspnet_env.sh", [], "graspnet 0"),
])
def test_checker_forwards_scope(fake_conda, entry, args, expected):
    result = run_script(entry, *args, CONDA_EXE=fake_conda)
    assert result.returncode == 0, result.stderr
    assert f"_environment.py check {expected}" in result.stdout


def test_missing_environment_gives_setup_command_without_installing():
    result = run_script("check_env.sh", "--backend", "obb", "--robot", CONDA_EXE="/bin/false")
    assert result.returncode == 1
    assert "bash scripts/setup_env.sh --backend obb --robot" in result.stderr


@pytest.mark.parametrize(("entry", "args"), [
    ("setup_env.sh", ["--backend", "unknown"]),
    ("setup_env.sh", ["--checkpoint-file"]),
    ("check_env.sh", ["--checkpoint-file", "/does/not/exist"]),
])
def test_rejects_invalid_options_before_accessing_conda(entry, args):
    result = run_script(entry, *args, CONDA_EXE="/bin/false")
    assert result.returncode == 2


def test_failed_model_download_preserves_destination(tmp_path):
    source = tmp_path / "quota.html"
    source.write_text("<html>Download quota exceeded</html>")
    destination = tmp_path / "configured weights.tar"
    destination.write_bytes(b"")
    # Exercise the real downloader, validator and EXIT cleanup through a local URL.
    program = """
set -eu
source scripts/_sdk_assets.sh
probe() { python3 scripts/_environment.py probe "$@"; }
works() { probe "$@" >/dev/null 2>&1; }
install_model "$1" "$2"
"""
    result = subprocess.run(["bash", "-c", program, "test", str(destination), source.as_uri()],
                            cwd=ROOT, text=True, capture_output=True)
    assert result.returncode != 0
    assert destination.read_bytes() == b""
    assert not list(tmp_path.glob("*.part.*"))
    with zipfile.ZipFile(source, "w") as model:
        model.writestr("archive/data.pkl", b"test weights")
    subprocess.run(["bash", "-c", program, "test", str(destination), source.as_uri()],
                   cwd=ROOT, check=True)
    assert destination.read_bytes() == source.read_bytes()


def test_sdk_checkout_and_rerun_preserve_existing_files(tmp_path):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    subprocess.run(["git", "init", "-q", str(upstream)], check=True)
    (upstream / "setup.py").write_text("# upstream")
    subprocess.run(["git", "-C", str(upstream), "add", "setup.py"], check=True)
    subprocess.run(["git", "-C", str(upstream), "-c", "user.name=test", "-c",
                    "user.email=test@example.invalid", "commit", "-qm", "SDK fixture"], check=True)
    destination = tmp_path / "sdk" / "test sdk"
    command = ["bash", "-ec", 'source scripts/_sdk_assets.sh; clone_sdk "$1" setup.py "$2"',
               "test", str(destination), str(upstream)]
    subprocess.run(command, cwd=ROOT, check=True)
    assert (destination / "setup.py").read_text() == "# upstream"
    (destination / "setup.py").write_text("# local")
    subprocess.run(command, cwd=ROOT, check=True)
    assert (destination / "setup.py").read_text() == "# local"


def test_manual_only_report_requests_local_files(fake_conda):
    conda = Path(fake_conda)
    script = conda.read_text().replace('echo "$*" ;;', 'exit 3 ;;')
    conda.write_text(script)
    result = run_script("check_env.sh", "--backend", "obb", "--robot", CONDA_EXE=fake_conda)
    assert result.returncode == 3
    assert "[MANUAL]" in result.stderr
    assert "install missing software" not in result.stderr
