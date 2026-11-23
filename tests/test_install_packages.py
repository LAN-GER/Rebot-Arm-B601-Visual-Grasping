"""Exercise the real resolver and approval boundary without network or host mutations."""
import json
import os
from pathlib import Path
import pty
import subprocess
import sys
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts/_install_packages.py"


def wheel(directory, name, version, dependency=None):
    info = f"{name}-{version}.dist-info"
    path = directory / f"{name}-{version}-py3-none-any.whl"
    meta = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
    if dependency:
        meta += f"Requires-Dist: {dependency}\n"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"{name}.py", f"VERSION = '{version}'")
        z.writestr(f"{info}/METADATA", meta)
        z.writestr(f"{info}/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        z.writestr(f"{info}/RECORD", "".join(f"{n},,\n" for n in [*z.namelist(), f"{info}/RECORD"]))
    return path


def run(args, env, reply=None):
    if reply is None:
        return subprocess.run(args, env=env, stdin=subprocess.DEVNULL, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
    master, slave = pty.openpty()
    try:
        os.write(master, (reply + "\n").encode())
        return subprocess.run(args, env=env, stdin=slave, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
    finally:
        os.close(master)
        os.close(slave)


def test_pip_dependency_changes_require_confirmation(tmp_path):
    subprocess.run([sys.executable, "-m", "venv", str(tmp_path / "env")], check=True)
    py = tmp_path / "env/bin/python"
    journal = tmp_path / "journal"
    env = dict(os.environ, REBOT_INSTALL_JOURNAL=str(journal), PIP_NO_INDEX="1",
               PIP_FIND_LINKS=str(tmp_path), PIP_DISABLE_PIP_VERSION_CHECK="1")
    env.pop("PYTHONPATH", None)
    first = wheel(tmp_path, "fixturelib", "1.0")
    second = wheel(tmp_path, "fixturelib", "2.0")
    app = wheel(tmp_path, "fixtureapp", "1.0", "fixturelib==2.0")
    cmd = [py, HELPER, "pip"]
    direct = run([*cmd, first], env)
    assert direct.returncode == 0, direct.stdout
    assert "Installed fixturelib 1.0" in journal.read_text()
    for reply in (None, "n", ""):
        declined = run([*cmd, app], env, reply)
        assert declined.returncode != 0
        assert "Upgraded fixturelib: 1.0 -> 2.0" in declined.stdout
        assert "fixtureapp" not in journal.read_text()
    approved = run([*cmd, app], env, "y")
    assert approved.returncode == 0, approved.stdout
    assert "Upgraded fixturelib: 1.0 -> 2.0" in journal.read_text()
    assert "Installed fixtureapp 1.0" in journal.read_text()
    declined = run([*cmd, first], env, "n")
    assert declined.returncode != 0 and "Downgraded" in declined.stdout
    approved = run([*cmd, first], env, "yes")
    assert approved.returncode == 0, approved.stdout
    assert "Downgraded fixturelib: 2.0 -> 1.0" in journal.read_text()
    # An SDK-local extension occupies a different destination to the rebot module.
    target = tmp_path / "sdk with space"
    local = run([*cmd, "--upgrade", "--target", target, second], env)
    assert local.returncode == 0, local.stdout
    assert f"[pip {target}] Installed fixturelib 2.0" in journal.read_text()
    replace = [*cmd, "--upgrade", "--target", target, first]
    assert run(replace, env).returncode != 0
    approved = run(replace, env, "y")
    assert approved.returncode == 0, approved.stdout
    assert f"[pip {target}] Downgraded fixturelib: 2.0 -> 1.0" in journal.read_text()
    assert [d.name for d in target.glob("*.dist-info")] == ["fixturelib-1.0.dist-info"]


def test_conda_changes_require_confirmation(tmp_path):
    import hashlib
    import io
    import shutil
    import tarfile

    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not conda:
        pytest.skip("requires a Conda installation")
    base = subprocess.check_output([conda, "info", "--base"], text=True).strip()
    repo = tmp_path / "channel"
    for subdir in ["linux-64", "noarch"]:
        sub = repo / subdir
        sub.mkdir(parents=True)
        packages = {}
        if subdir == "linux-64":
            for ver in ["1.0", "2.0"]:
                record = dict(name="fixturelib", version=ver, build="0", build_number=0,
                              subdir=subdir, depends=[])
                archive = sub / f"fixturelib-{ver}-0.tar.bz2"
                with tarfile.open(archive, "w:bz2") as tar:
                    for filename, data in [("info/index.json", json.dumps(record)),
                                           ("info/files", "share/version.txt\n"), ("share/version.txt", ver)]:
                        info = tarfile.TarInfo(filename)
                        binary = data.encode()
                        info.size = len(binary)
                        tar.addfile(info, io.BytesIO(binary))
                packages[archive.name] = dict(record, sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                                              size=archive.stat().st_size)
        (sub / "repodata.json").write_text(json.dumps({"info": {"subdir": subdir}, "packages": packages}))
    prefix = tmp_path / "env"
    (prefix / "conda-meta").mkdir(parents=True)
    (prefix / "conda-meta/history").touch()
    journal = tmp_path / "journal"
    env = dict(os.environ, CONDA_PREFIX=str(prefix), CONDA_PKGS_DIRS=str(tmp_path / "pkgs"),
               CONDA_OFFLINE="false", CONDA_NOTIFY_OUTDATED_CONDA="false", REBOT_INSTALL_JOURNAL=str(journal))
    env.pop("PYTHONPATH", None)
    cmd = [Path(base) / "bin/python", HELPER, "conda", "-c", repo.as_uri()]
    for ver, reply, accepted in [("1.0", None, True), ("2.0", None, False), ("2.0", "n", False),
                                 ("2.0", "y", True), ("1.0", "y", True)]:
        result = run([*cmd, f"fixturelib={ver}"], env, reply)
        assert (result.returncode == 0) == accepted, result.stdout
        assert (prefix / "share/version.txt").read_text() == (ver if accepted else "1.0")
    for expected in ["Installed fixturelib 1.0", "Upgraded fixturelib: 1.0", "Downgraded fixturelib: 2.0"]:
        assert expected in journal.read_text()


def test_required_git_sdk_branch_change(tmp_path):
    upstream = tmp_path / "upstream"
    dest = tmp_path / "sdk"
    journal = tmp_path / "journal"
    env = dict(os.environ, REBOT_INSTALL_JOURNAL=str(journal))

    def git(repo, *args):
        return subprocess.check_output(["git", "-C", repo, "-c", "user.name=fixture", "-c",
                                        "user.email=test@example.invalid", *args], text=True).strip()

    upstream.mkdir()
    git(upstream, "init", "-q")
    (upstream / "setup.py").write_text("# original")
    git(upstream, "add", ".")
    git(upstream, "commit", "-qm", "original version")
    original = git(upstream, "rev-parse", "HEAD")
    args = ["bash", "-ec", 'source "$1"; shift; clone_sdk "$@"', "test",
            ROOT / "scripts/_sdk_assets.sh", dest, "setup.py", upstream]
    assert run(args, env).returncode == 0
    git(upstream, "checkout", "-qb", "cuda-12")
    (upstream / "setup.py").write_text("# new")
    git(upstream, "commit", "-qam", "CUDA build")
    new = git(upstream, "rev-parse", "HEAD")
    change = [*args, "--branch", "cuda-12"]
    for reply in (None, "n"):
        assert run(change, env, reply).returncode != 0
        assert git(dest, "rev-parse", "HEAD") == original
        assert len(journal.read_text().splitlines()) == 1
    (dest / "local.txt").touch()
    assert run(change, env, "y").returncode != 0
    assert git(dest, "rev-parse", "HEAD") == original
    (dest / "local.txt").unlink()
    approved = run(change, env, "y")
    assert approved.returncode == 0, approved.stdout
    assert git(dest, "rev-parse", "HEAD") == new
    assert git(dest, "branch", "--show-current") == "cuda-12"
    assert run(change, env).returncode == 0
    assert journal.read_text().splitlines()[1:] == [
        f"[SDK] Updated {dest}: {original[:12]} -> {new[:12]} (cuda-12)"]
