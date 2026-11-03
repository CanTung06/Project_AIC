"""Cấu hình tập trung cho pipeline phát hiện bất thường đăng nhập."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
AUTH_SOURCE_FILE = RAW_DIR / "synthetic_web_auth_logs.csv"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = ROOT / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
METRICS_DIR = OUTPUTS_DIR / "metrics"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"

FILTERED_DATA = PROCESSED_DIR / "auth_preprocessed.csv"
TRAIN_DATA = PROCESSED_DIR / "auth_train.csv"
VALIDATION_DATA = PROCESSED_DIR / "auth_validation.csv"
TEST_DATA = PROCESSED_DIR / "auth_test.csv"
UNDERSAMPLED_TRAIN_DATA = PROCESSED_DIR / "auth_train_undersampled.csv"
AUTH_FEATURE_SCHEMA = METRICS_DIR / "02_auth_feature_schema.json"

LOGISTIC_REGRESSION_MODEL = MODELS_DIR / "05_logistic_regression_auth_undersampling.joblib"
FULL_LOGISTIC_REGRESSION_MODEL = MODELS_DIR / "05_logistic_regression_auth_full_data.joblib"
XGBOOST_MODEL = MODELS_DIR / "06_xgboost_auth_undersampling.joblib"
FULL_XGBOOST_MODEL = MODELS_DIR / "06_xgboost_auth_full_data.joblib"
RANDOM_FOREST_MODEL = MODELS_DIR / "07_random_forest_auth_undersampling.joblib"
FULL_RANDOM_FOREST_MODEL = MODELS_DIR / "07_random_forest_auth_full_data.joblib"

TARGET_COLUMN = "is_anomaly"
SOURCE_LABEL_COLUMN = "Anomaly"
# Metadata nguồn chỉ dùng để tạo feature lịch sử; không xuất trong dataset đã xử lý.
SOURCE_METADATA_COLUMNS = ["User ID", "Timestamp"]
# Dataset sau s02 chỉ gồm feature số và nhãn.
METADATA_COLUMNS = []
SEED = 42
CHUNK_SIZE = 500_000
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
UNDERSAMPLE_NEGATIVE_PER_POSITIVE = 3
MAX_ERROR_EXAMPLES_PER_TYPE = 1_000


def ensure_dirs() -> None:
    for directory in (PROCESSED_DIR, TABLES_DIR, METRICS_DIR, FIGURES_DIR, MODELS_DIR, OUTPUTS_DIR / "predictions"):
        directory.mkdir(parents=True, exist_ok=True)
