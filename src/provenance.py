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


def get_environment() -> dict:
    """Return the Python and library versions and the current git commit."""
    return {
        "python": platform.python_version(),
        "pandas": pandas.__version__,
        "numpy": numpy.__version__,
        "scikit-learn": sklearn.__version__,
        "optuna": optuna.__version__,
        "git_commit": get_git_commit(),
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
