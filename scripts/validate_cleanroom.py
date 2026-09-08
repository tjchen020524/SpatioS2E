#!/usr/bin/env python3
"""Validate an installed release and write a machine-readable environment report."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "requirements-lock-linux-x86_64-py310.txt"
PARTITION_MANIFEST = ROOT / "configs" / "manuscript" / "gene_partition_manifest.json"
SOURCE_ROOTS = ("spatios2e", "tests", "scripts", "configs/manuscript", "docs", "experiments")
SOURCE_FILES = (
    "pyproject.toml",
    "MANIFEST.in",
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "CHANGELOG.md",
    LOCK.name,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _locked_packages() -> dict[str, str]:
    locked: dict[str, str] = {}
    for raw in LOCK.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "--")):
            continue
        name, version = line.split("==", maxsplit=1)
        locked[_canonical(name)] = version
    return locked


def _installed_packages() -> dict[str, str]:
    return {
        _canonical(dist.metadata["Name"]): dist.version
        for dist in importlib.metadata.distributions()
        if dist.metadata.get("Name")
    }


def _source_snapshot() -> dict[str, object]:
    paths: list[Path] = []
    for relative in SOURCE_ROOTS:
        root = ROOT / relative
        if root.exists():
            paths.extend(
                path
                for path in root.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix not in {".pyc", ".pyo"}
                and not {"artifacts", "runs", "trainable_backbone_runs"}.intersection(path.parts)
            )
    paths.extend(ROOT / relative for relative in SOURCE_FILES)
    records = []
    combined = hashlib.sha256()
    for path in sorted(set(paths)):
        relative = path.relative_to(ROOT).as_posix()
        digest = _sha256(path)
        records.append({"path": relative, "sha256": digest, "bytes": path.stat().st_size})
        combined.update(relative.encode())
        combined.update(b"\0")
        combined.update(digest.encode())
        combined.update(b"\n")
    return {"sha256": combined.hexdigest(), "n_files": len(records), "files": records}


def _read_cpu_model() -> str | None:
    path = Path("/proc/cpuinfo")
    if not path.exists():
        return None
    for line in path.read_text(errors="replace").splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", maxsplit=1)[1].strip()
    return None


def _read_memory_bytes() -> int | None:
    path = Path("/proc/meminfo")
    if not path.exists():
        return None
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    return None


def _gpu_inventory() -> dict[str, object]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=20)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "reason": type(exc).__name__, "devices": []}
    devices = []
    if completed.returncode == 0:
        for line in completed.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 5:
                devices.append(
                    {
                        "name": fields[0],
                        "uuid": fields[1],
                        "driver_version": fields[2],
                        "memory_mib": int(fields[3]),
                        "compute_capability": fields[4],
                    }
                )
    return {
        "available": bool(devices),
        "devices": devices,
        "return_code": completed.returncode,
        "stderr": completed.stderr.strip(),
    }


def _torch_inventory() -> dict[str, object]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - recorded rather than hidden
        return {"imported": False, "error": repr(exc)}
    return {
        "imported": True,
        "version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
    }


def _git_inventory() -> dict[str, object]:
    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", *args], cwd=ROOT, check=False, capture_output=True, text=True)

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def _portable_text(value: str) -> str:
    """Remove machine-local environment prefixes from the archived report."""
    text = value.replace(str(Path(sys.executable).resolve()), "python")
    text = text.replace(str(ROOT.resolve()), "<repository>")
    text = re.sub(
        r"/(?:[^/\s]+/)*lib/python\d+\.\d+",
        "<python-environment>/lib/python",
        text,
    )
    text = re.sub(r"/tmp/spatios2e-build-[^/\s]+", "<temporary-build-dir>", text)
    return text


def _run(label: str, command: list[str], cwd: Path = ROOT) -> dict[str, object]:
    started = time.monotonic()
    completed = subprocess.run(command, cwd=cwd, check=False, capture_output=True, text=True)
    return {
        "label": label,
        "command": [_portable_text(argument) for argument in command],
        "return_code": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "stdout": _portable_text(completed.stdout),
        "stderr": _portable_text(completed.stderr),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    locked = _locked_packages()
    installed = _installed_packages()
    lock_mismatches = {
        name: {"expected": version, "installed": installed.get(name)}
        for name, version in locked.items()
        if installed.get(name) != version
    }

    with tempfile.TemporaryDirectory(prefix="spatios2e-build-") as build_dir:
        checks = [
            _run("partition manifest", [sys.executable, "scripts/build_manuscript_partition_manifest.py", "--check"]),
            _run("unit and synthetic training tests", [sys.executable, "-m", "pytest", "-q"]),
            _run("ruff", [sys.executable, "-m", "ruff", "check", "spatios2e", "tests", "scripts", "experiments"]),
            _run(
                "source and wheel build",
                [sys.executable, "-m", "build", "--no-isolation", "--outdir", build_dir],
            ),
            _run("component-evaluator CLI", [sys.executable, "-m", "spatios2e.evaluation.components", "--help"]),
        ]

    report = {
        "schema_version": 1,
        "status": "PASS" if not lock_mismatches and all(item["return_code"] == 0 for item in checks) else "FAIL",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "CPU clean-room install, exact dependency check, partition-manifest regeneration, unit tests, "
            "synthetic held-out-decoder training/evaluation, lint, package build and CLI smoke test. "
            "This is not a rerun of the four-cohort manuscript analyses and does not validate CUDA execution."
        ),
        "platform_support": "CPython 3.10, Linux x86_64",
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "operating_system": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "hardware": {
            "cpu_model": _read_cpu_model(),
            "logical_cpu_count": os.cpu_count(),
            "memory_bytes": _read_memory_bytes(),
            "gpu": _gpu_inventory(),
        },
        "torch": _torch_inventory(),
        "git": _git_inventory(),
        "artifacts": {
            "dependency_lock": {"path": LOCK.name, "sha256": _sha256(LOCK)},
            "gene_partition_manifest": {
                "path": PARTITION_MANIFEST.relative_to(ROOT).as_posix(),
                "sha256": _sha256(PARTITION_MANIFEST),
            },
            "source_snapshot": _source_snapshot(),
        },
        "dependency_lock": {
            "n_locked": len(locked),
            "mismatches": lock_mismatches,
            "installed": dict(sorted(installed.items())),
        },
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"{report['status']}: wrote {args.output}")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
