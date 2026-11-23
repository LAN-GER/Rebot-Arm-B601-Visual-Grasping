"""Review resolved changes to rebot and record what the package managers installed."""
from importlib import metadata
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import sysconfig
from tempfile import TemporaryDirectory


def normalized(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def python_packages(paths):
    return {normalized(d.metadata["Name"]): d.version for d in metadata.distributions(path=paths)
            if d.metadata["Name"]}


def prune_old_target_metadata(target, installed):
    # pip --target replaces the module but leaves the previous version's
    # .dist-info directory. Keep discovery consistent with the wheel just installed.
    for directory in target.glob("*.dist-info"):
        dist = metadata.PathDistribution(directory)
        name = dist.metadata.get("Name")
        if name and normalized(name) in installed and dist.version != installed[normalized(name)]:
            if directory.is_symlink():
                directory.unlink()
            else:
                shutil.rmtree(directory)


def conda_packages(prefix):
    packages = {}
    for file in (prefix / "conda-meta").glob("*.json"):
        record = json.loads(file.read_bytes())
        packages[record["name"]] = (record["version"], record["build"])
    return packages


def describe(name, old, new, version_key):
    if old is None:
        return f"Installed {name} {new}"
    if new is None:
        return f"Removed {name} {old}"
    before = version_key(old.split(" ", 1)[0])
    after = version_key(new.split(" ", 1)[0])
    action = "Replaced"
    if after > before:
        action = "Upgraded"
    elif after < before:
        action = "Downgraded"
    return f"{action} {name}: {old} -> {new}"


def changes(before, after, version_key):
    return [describe(name, before.get(name), after.get(name), version_key)
            for name in sorted(before.keys() | after.keys()) if before.get(name) != after.get(name)]


def confirm(lines):
    if not lines:
        return
    print("Changes requiring confirmation / 以下变更需要确认", *lines, sep="\n  ", flush=True)
    if sys.stdin.isatty():
        try:
            if input("Apply these changes? / 是否执行以上变更？ [y/N] ").strip().lower() in ("y", "yes"):
                return
        except EOFError:
            pass
    raise SystemExit("Skipped this transaction: updating an existing installation requires terminal confirmation.")


def record(manager, lines):
    with Path(os.environ["REBOT_INSTALL_JOURNAL"]).open("a", encoding="utf-8") as dest:
        for line in lines:
            print(f"[{manager}] {line}", file=dest)


def install_pip(args):
    from pip._vendor.packaging.version import Version

    target = Path(args[args.index("--target") + 1]) if "--target" in args else None
    paths = [str(target)] if target else list({sysconfig.get_path("purelib"), sysconfig.get_path("platlib")})
    before = python_packages(paths)
    pip = [sys.executable, "-m", "pip", "install"]
    label = f"pip {target}" if target else "pip"
    with TemporaryDirectory() as tmp:
        report = Path(tmp) / "report.json"
        subprocess.run([*pip, "--dry-run", "--report", str(report), *args], check=True)
        planned = {normalized(i["metadata"]["name"]): i["metadata"]["version"]
                   for i in json.loads(report.read_bytes())["install"]}
        confirm([describe(k, before[k], v, Version) for k, v in sorted(planned.items())
                 if k in before and Version(before[k]) != Version(v)])
        if not planned:
            return

        # A second resolver pass cannot select an unapproved version of an
        # existing package, even if the index changes after the preview.
        pins = Path(tmp) / "constraints.txt"
        pins.write_text("".join(f"{k}==={v}\n" for k, v in (before | planned).items()))
        installed = {}
        try:
            subprocess.run([*pip, "--constraint", str(pins), "--report", str(report), *args], check=True)
            installed = {normalized(i["metadata"]["name"]): i["metadata"]["version"]
                         for i in json.loads(report.read_bytes())["install"]}
            if target:
                prune_old_target_metadata(target, installed)
        finally:
            after = python_packages(paths)
            lines = changes(before, after, Version)
            for name in installed:
                if name in before and before[name] == after.get(name):
                    lines.append(f"Reinstalled {name} {after[name]}")
            record(label, lines)


def install_conda(args):
    # Use the base Conda interpreter, including when rebot has no Python yet.
    from conda.base.context import context
    from conda.models.match_spec import MatchSpec
    from conda.models.version import VersionOrder

    opts = argparse.ArgumentParser()
    opts.add_argument("-c", dest="channels", action="append", required=True)
    opts.add_argument("specs", nargs="+")
    options = opts.parse_args(args)
    prefix = Path(os.environ["CONDA_PREFIX"])
    solver = context.plugin_manager.get_solver_backend()(
        str(prefix), options.channels, context.subdirs,
        specs_to_add=[MatchSpec(s) for s in options.specs], command="install")
    txn = solver.solve_for_transaction()
    if txn.nothing_to_do:
        return
    plan = txn.prefix_setups[str(prefix)]
    before = conda_packages(prefix)
    # Conda can take ownership of a distribution previously installed by pip.
    site = list((prefix / "lib").glob("python*/site-packages"))
    python_before = python_packages([str(p) for p in site])
    linked = {p.name: (p.version, p.build) for p in plan.link_precs}
    affected = sorted(set(linked) | {p.name for p in plan.unlink_precs})
    review = []
    for name in affected:
        old = before.get(name)
        new = linked.get(name)
        if old and old != new:
            review.append(describe(name, " ".join(old), " ".join(new) if new else None, VersionOrder))
        installed = python_before.get(normalized(name))
        if new and installed and (not old or installed != old[0]) and installed != new[0]:
            review.append(describe(name, installed, new[0], VersionOrder))
    confirm(review)
    try:
        txn.download_and_extract()
        txn.execute()  # Apply exactly the transaction reviewed above.
    finally:
        after = conda_packages(prefix)
        previous = {k: " ".join(v) for k, v in before.items()}
        for k in linked.keys() - before.keys():
            if k in after and normalized(k) in python_before:
                previous[k] = python_before[normalized(k)]
        record("conda", changes(previous, {k: " ".join(v) for k, v in after.items()}, VersionOrder))


def requirements_met(specs):
    from pip._vendor.packaging.requirements import Requirement
    for spec in map(Requirement, specs):
        try:
            if metadata.version(spec.name) not in spec.specifier:
                return False
        except metadata.PackageNotFoundError:
            return False
    return True


if __name__ == "__main__":
    command, *args = sys.argv[1:]
    try:
        if command == "confirm":
            confirm(args)
        elif command == "pip":
            install_pip(args)
        elif command == "conda":
            install_conda(args)
        elif command == "created":
            record("conda", [f"Installed {name} {ver} {build}" for name, (ver, build)
                             in sorted(conda_packages(Path(args[0])).items())])
        elif command == "satisfies":
            sys.exit(not requirements_met(args))
        else:
            sys.exit(f"Unknown package operation: {command}")
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
