"""Project-wide configuration: paths and constants."""

from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "raw" / "creditcard.csv"
PROCESSED_PATH = ROOT_DIR / "data" / "processed" / "transactions.parquet"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

# Constants
SEED = 42
TARGET = "Class"
TEST_SIZE = 0.2
