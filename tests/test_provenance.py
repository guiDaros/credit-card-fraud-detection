"""Tests for provenance info, using temporary files."""

import hashlib

from src.provenance import get_provenance


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
