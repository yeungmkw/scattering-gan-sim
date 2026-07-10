"""Runtime helpers shared by reconstruction and GAN experiment entrypoints."""

from __future__ import annotations

import json
import platform
import random
import shutil
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEPENDENCY_DISTRIBUTIONS = (
    "torch",
    "torchvision",
    "numpy",
    "scipy",
    "matplotlib",
    "pillow",
    "opencv-python-headless",
    "pytest",
)


def prepare_output_dir(output_dir: Path) -> None:
    """Prepare a fresh run directory for artifacts owned by an entrypoint."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for dirname in ("samples", "checkpoints"):
        path = output_dir / dirname
        if path.exists():
            shutil.rmtree(path)
        path.mkdir()
    for filename in (
        "config.json",
        "source_config.json",
        "history.json",
        "metrics.json",
        "manifest.json",
    ):
        path = output_dir / filename
        if path.exists():
            path.unlink()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def snapshot_config(config: dict[str, Any], *, output_dir: Path, config_path: Path | None) -> None:
    write_json(output_dir / "config.json", config)
    if config_path is not None and config_path.exists():
        shutil.copyfile(config_path, output_dir / "source_config.json")


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def select_device(name: str) -> torch.device:
    normalized = name.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(normalized)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_metadata() -> dict[str, Any]:
    """Return serializable environment and repository provenance for a run."""

    repository_root = Path(__file__).resolve().parent
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.system(),
        "dependencies": _dependency_versions(),
        "git": _git_metadata(repository_root),
    }


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in DEPENDENCY_DISTRIBUTIONS:
        try:
            versions[distribution] = version(distribution)
        except PackageNotFoundError:
            versions[distribution] = None
    return versions


def _git_metadata(repository_root: Path) -> dict[str, Any]:
    commit = _git_output(repository_root, "rev-parse", "HEAD")
    branch = _git_output(repository_root, "branch", "--show-current")
    status = _git_output(repository_root, "status", "--porcelain")
    return {
        "commit": commit or None,
        "branch": branch or None,
        "is_dirty": bool(status),
        "has_commit": commit is not None,
    }


def _git_output(repository_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()
