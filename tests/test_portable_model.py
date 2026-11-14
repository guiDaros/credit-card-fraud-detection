"""A saved model must load and predict without this project's code."""

import os
import subprocess
import sys
import textwrap

import joblib
import numpy as np
import pandas as pd
import pytest

import src.train as train

RAW_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Runs in a fresh interpreter. -I (isolated mode) ignores PYTHONPATH and does not
# add the current directory to sys.path, and cwd is outside the repository.
LOAD_AND_PREDICT = textwrap.dedent(
    """
    import sys

    import joblib
    import numpy as np
    import pandas as pd

    try:
        import src  # noqa: F401
    except ModuleNotFoundError:
        pass
    else:
        raise SystemExit("the project's src package should not be importable here")

    bundle = joblib.load(sys.argv[1])
    rng = np.random.default_rng(1)
    columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    df = pd.DataFrame(rng.normal(size=(5, len(columns))), columns=columns)
    df["Time"] = rng.uniform(0, 172800, 5)
    df["Amount"] = rng.uniform(0, 500, 5)

    proba = bundle["pipeline"].predict_proba(df[bundle["input_columns"]])[:, 1]
    assert proba.shape == (5,)
    assert "src" not in sys.modules
    print(" ".join(f"{p:.6f}" for p in proba))
    """
)


def make_raw_data(n_rows: int = 200):
    rng = np.random.default_rng(0)
    y = pd.Series((np.arange(n_rows) % 10 == 0).astype(int))
    X = pd.DataFrame(rng.normal(size=(n_rows, len(RAW_COLUMNS))), columns=RAW_COLUMNS)
    X["Time"] = np.sort(rng.uniform(0, 172800, n_rows))
    X["Amount"] = rng.uniform(0, 500, n_rows)
    X["V1"] += 2 * y
    return X, y


@pytest.mark.parametrize("use_hour", [True, False])
def test_saved_model_loads_without_src(tmp_path, use_hour):
    X, y = make_raw_data()
    pipeline = train.make_pipeline(use_hour=use_hour, n_estimators=5, max_depth=3).fit(X, y)
    model_file = tmp_path / "model.joblib"
    bundle = {"pipeline": pipeline, "threshold": 0.5, "input_columns": RAW_COLUMNS}
    joblib.dump(bundle, model_file)

    outside = tmp_path / "outside"
    outside.mkdir()
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "-I", "-c", LOAD_AND_PREDICT, str(model_file)],
        cwd=outside,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert len(result.stdout.split()) == 5
