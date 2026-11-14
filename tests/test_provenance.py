"""Tests for provenance info, using temporary files."""

import hashlib
import json
import subprocess

import pytest

import src.provenance as provenance
from src.provenance import get_provenance, get_run_provenance


def test_get_provenance_on_temp_file(tmp_path):
    path = tmp_path / "data.csv"
    path.write_bytes(b"a,b\n1,2\n")

    prov = get_provenance(path)

    assert prov["raw_sha256"] == hashlib.sha256(b"a,b\n1,2\n").hexdigest()
    for key in ["python", "pandas", "numpy", "scikit-learn", "optuna"]:
        assert isinstance(prov[key], str) and prov[key]
    assert prov["git_commit"] is None or isinstance(prov["git_commit"], str)


def test_sha256_changes_with_content(tmp_path):
    path = tmp_path / "data.csv"
    path.write_bytes(b"a,b\n1,2\n")
    first = get_provenance(path)["raw_sha256"]

    path.write_bytes(b"a,b\n1,3\n")
    second = get_provenance(path)["raw_sha256"]

    assert first != second


def test_get_run_provenance_reads_dataset_hash(tmp_path):
    report_path = tmp_path / "data_report.json"
    report_path.write_text(json.dumps({"provenance": {"raw_sha256": "abc123"}}))

    prov = get_run_provenance(report_path)

    assert prov["dataset_sha256"] == "abc123"
    for key in ["python", "pandas", "numpy", "scikit-learn", "optuna"]:
        assert isinstance(prov[key], str) and prov[key]
    assert "git_commit" in prov


def test_get_run_provenance_missing_report(tmp_path):
    with pytest.raises(FileNotFoundError, match="make_dataset"):
        get_run_provenance(tmp_path / "missing.json")


def fake_git_status(stdout):
    """Fake subprocess.run that returns `stdout`, as `git status --porcelain` would."""

    def run(*args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    return run


@pytest.mark.parametrize("stdout, expected", [("", False), (" M src/train.py\n", True)])
def test_get_git_dirty_reads_porcelain_output(monkeypatch, stdout, expected):
    monkeypatch.setattr(provenance.subprocess, "run", fake_git_status(stdout))
    assert provenance.get_git_dirty() is expected


@pytest.mark.parametrize(
    "error", [OSError("git not found"), subprocess.CalledProcessError(128, "git")]
)
def test_get_git_dirty_is_none_without_git(monkeypatch, error):
    def run(*args, **kwargs):
        raise error

    monkeypatch.setattr(provenance.subprocess, "run", run)
    assert provenance.get_git_dirty() is None


def test_environment_records_git_dirty(tmp_path):
    path = tmp_path / "data.csv"
    path.write_bytes(b"a\n1\n")
    assert get_provenance(path)["git_dirty"] in (True, False, None)


@pytest.mark.parametrize(
    "dirty, allow_dirty, raises",
    [
        (True, False, True),
        (True, True, False),
        (False, False, False),
        (None, False, False),
    ],
)
def test_ensure_clean_tree(monkeypatch, dirty, allow_dirty, raises):
    monkeypatch.setattr(provenance, "get_git_dirty", lambda: dirty)
    if raises:
        with pytest.raises(RuntimeError, match="--allow-dirty"):
            provenance.ensure_clean_tree(allow_dirty)
    else:
        provenance.ensure_clean_tree(allow_dirty)


def git(repo, *args):
    """Run a git command in `repo` with a throwaway identity."""
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repo(tmp_path):
    """Tiny committed git repository with code and a report."""
    git(tmp_path, "init", "-q")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "train.py").write_text("x = 1\n")
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "data_report.json").write_text("{}\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_clean_repo_is_not_dirty(repo):
    assert provenance.get_git_dirty(repo) is False
    assert provenance.get_git_dirty(repo / "src") is False  # any folder inside works


def test_changes_in_reports_do_not_count(repo):
    (repo / "reports" / "data_report.json").write_text('{"changed": true}\n')
    (repo / "reports" / "metrics_temporal.json").write_text("{}\n")  # new, untracked
    assert provenance.get_git_dirty(repo) is False
    assert provenance.get_git_dirty(repo / "src") is False


@pytest.mark.parametrize("change", ["modify", "untracked"])
def test_code_changes_count(repo, change):
    if change == "modify":
        (repo / "src" / "train.py").write_text("x = 2\n")
    else:
        (repo / "Makefile").write_text("all:\n")
    assert provenance.get_git_dirty(repo) is True
    assert provenance.get_git_dirty(repo / "src") is True


def test_outside_a_repository_is_none(tmp_path):
    assert provenance.get_git_dirty(tmp_path) is None
