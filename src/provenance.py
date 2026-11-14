"""Provenance info: raw data hash, library versions and git commit."""

import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy
import optuna
import pandas
import sklearn


def file_sha256(path: Path) -> str:
    """Return the sha256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_git_commit() -> str | None:
    """Return the current git commit hash, or None if it cannot be read."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# Folders written by the pipeline itself (data report, metrics, plots). Changes there
# are outputs, not code, so they do not make the tree dirty: otherwise each step
# would block the next one (e.g. `make data` rewrites data_report.json, then
# `make train-*` would refuse to run). Data and models are already in .gitignore.
GENERATED_DIRS = ("reports",)


def get_git_dirty(repo_dir: Path | None = None) -> bool | None:
    """Return True if code or configuration has uncommitted changes, False if not.

    Uses `git status --porcelain` on the whole repository, excluding GENERATED_DIRS
    (ignored files such as data and models do not count either). Untracked files
    count as changes. repo_dir is any folder inside the repository (default: this
    project). Returns None if git is not available or this is not a git repository.
    """
    exclude = [f":(top,exclude){folder}" for folder in GENERATED_DIRS]
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--", ":(top)", *exclude],
            cwd=repo_dir or Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(result.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def ensure_clean_tree(allow_dirty: bool) -> None:
    """Raise RuntimeError if code or configuration has uncommitted changes.

    Reports must come from committed code, so that the recorded git commit is the
    code that produced them. Changes in GENERATED_DIRS (the reports themselves) are
    allowed, so the pipeline steps can run one after another and the reports can be
    committed at the end. allow_dirty=True skips the check (git_dirty is still
    recorded as True in the provenance).
    """
    if get_git_dirty() and not allow_dirty:
        raise RuntimeError(
            "The git working tree has uncommitted changes outside reports/, so the "
            "recorded commit would not match the code. Commit your changes first, or "
            "pass --allow-dirty."
        )


def get_environment() -> dict:
    """Return the Python and library versions, the git commit and whether the tree is dirty."""
    return {
        "python": platform.python_version(),
        "pandas": pandas.__version__,
        "numpy": numpy.__version__,
        "scikit-learn": sklearn.__version__,
        "optuna": optuna.__version__,
        "git_commit": get_git_commit(),
        "git_dirty": get_git_dirty(),
    }


def get_provenance(raw_path: Path) -> dict:
    """Collect provenance info for a raw data file."""
    return {"raw_sha256": file_sha256(Path(raw_path)), **get_environment()}


def get_run_provenance(data_report_path: Path) -> dict:
    """Provenance for a training/evaluation run.

    The dataset sha256 is read from the data report written by make_dataset;
    the library versions and git commit are those of the current run.
    """
    data_report_path = Path(data_report_path)
    if not data_report_path.exists():
        raise FileNotFoundError(
            f"Data report not found at {data_report_path}. Run 'python -m src.make_dataset' first."
        )
    report = json.loads(data_report_path.read_text(encoding="utf-8"))
    return {"dataset_sha256": report["provenance"]["raw_sha256"], **get_environment()}
