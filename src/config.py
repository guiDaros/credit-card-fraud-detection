"""Project-wide configuration: paths and constants."""

from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "raw" / "creditcard.csv"
PROCESSED_PATH = ROOT_DIR / "data" / "processed" / "transactions.parquet"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
DATA_REPORT_PATH = REPORTS_DIR / "data_report.json"

# sha256 of the Kaggle creditcard.csv the results were produced with (the
# provenance.raw_sha256 of reports/data_report.json). make_dataset refuses a
# different file unless --skip-hash-check is passed.
RAW_SHA256 = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"

# Constants
SEED = 42
TARGET = "Class"
TEST_SIZE = 0.2
SPLITS = ("random", "temporal")


def add_run_args(parser) -> None:
    """Add the --split and --no-hour options shared by train and evaluate."""
    parser.add_argument(
        "--split",
        choices=SPLITS,
        default="temporal",
        help="Train/test split and CV scheme (default: temporal).",
    )
    parser.add_argument("--no-hour", action="store_true", help="Do not create the Hour feature.")


def run_suffix(split: str, use_hour: bool) -> str:
    """Name of a run configuration, e.g. 'temporal' or 'temporal_nohour'."""
    return split if use_hour else f"{split}_nohour"


def model_path(split: str, use_hour: bool) -> Path:
    """Path of the saved model, e.g. models/model_temporal.joblib."""
    return MODELS_DIR / f"model_{run_suffix(split, use_hour)}.joblib"


def metrics_path(split: str, use_hour: bool) -> Path:
    """Path of the metrics report, e.g. reports/metrics_temporal.json."""
    return REPORTS_DIR / f"metrics_{run_suffix(split, use_hour)}.json"


def pr_curve_path(split: str, use_hour: bool) -> Path:
    """Path of the PR curve plot, e.g. reports/pr_curve_temporal.png."""
    return REPORTS_DIR / f"pr_curve_{run_suffix(split, use_hour)}.png"


def baseline_metrics_path() -> Path:
    """Path of the logistic regression baseline report (temporal split)."""
    return REPORTS_DIR / "metrics_baseline_temporal.json"


def compare_hour_path() -> Path:
    """Path of the with/without Hour comparison report (scripts/compare_hour.py)."""
    return REPORTS_DIR / "compare_hour.json"
