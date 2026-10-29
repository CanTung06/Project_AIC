"""Cấu hình tập trung cho pipeline CIC-IDS2017."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = ROOT / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
METRICS_DIR = OUTPUTS_DIR / "metrics"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"

FILTERED_DATA = PROCESSED_DIR / "password_attack_dataset.csv"
TRAIN_DATA = PROCESSED_DIR / "train.csv"
VALIDATION_DATA = PROCESSED_DIR / "validation.csv"
TEST_DATA = PROCESSED_DIR / "test.csv"
UNDERSAMPLED_TRAIN_DATA = PROCESSED_DIR / "train_undersampled.csv"

WEIGHTED_MODEL = MODELS_DIR / "05_sgd_class_weight.joblib"
UNDERSAMPLED_MODEL = MODELS_DIR / "05_sgd_undersampling.joblib"
SELECTED_MODEL_INFO = METRICS_DIR / "07_selected_model.json"

TARGET_COLUMN = "is_attack"
METADATA_COLUMNS = ["attack_type"]
SEED = 42
CHUNK_SIZE = 500_000
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
UNDERSAMPLE_NEGATIVE_PER_POSITIVE = 10
MAX_ERROR_EXAMPLES_PER_TYPE = 1_000


def ensure_dirs() -> None:
    for directory in (PROCESSED_DIR, TABLES_DIR, METRICS_DIR, FIGURES_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
