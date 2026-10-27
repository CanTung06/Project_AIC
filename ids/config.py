"""Cấu hình tập trung cho pipeline Project_AIC."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = ROOT / "data" / "raw" / "rba-dataset.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
METRIC_DIR = OUTPUT_DIR / "metrics"
MODEL_DIR = OUTPUT_DIR / "models"
FIGURE_DIR = OUTPUT_DIR / "figures"

TARGET = "Is Account Takeover"
TRAIN_FILE = PROCESSED_DIR / "train.csv"
VALIDATION_FILE = PROCESSED_DIR / "validation.csv"
TEST_FILE = PROCESSED_DIR / "test.csv"
MODEL_FILE = MODEL_DIR / "03_xgboost_account_takeover.joblib"
MODEL_RESULTS_FILE = METRIC_DIR / "03_xgboost_validation_trials.json"
TEST_METRICS_FILE = METRIC_DIR / "04_xgboost_test_metrics.json"
TRIALS_TABLE_FILE = TABLE_DIR / "03_xgboost_validation_trials.csv"
VALIDATION_TOPK_TABLE_FILE = TABLE_DIR / "05_validation_top_risk.csv"
VALIDATION_TOPK_METRICS_FILE = METRIC_DIR / "05_validation_topk_metrics.json"
VALIDATION_ALERT_REVIEW_FILE = TABLE_DIR / "06_validation_alert_review.csv"
VALIDATION_ALERT_REVIEW_SUMMARY_FILE = METRIC_DIR / "06_validation_alert_review_summary.json"

SEED = 42
CSV_CHUNK_SIZE = 500_000
MAX_NEGATIVE_TRAIN_ROWS = 10_000
TRIAL_CONFIGS = (
    {"ratio": 20, "weight_multiplier": 1.0, "max_depth": 3, "min_child_weight": 5,
     "learning_rate": 0.05, "n_estimators": 500, "subsample": 0.8,
     "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 5.0},
    {"ratio": 50, "weight_multiplier": 1.0, "max_depth": 4, "min_child_weight": 5,
     "learning_rate": 0.05, "n_estimators": 500, "subsample": 0.8,
     "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 5.0},
    {"ratio": 100, "weight_multiplier": 1.0, "max_depth": 5, "min_child_weight": 10,
     "learning_rate": 0.03, "n_estimators": 700, "subsample": 0.8,
     "colsample_bytree": 0.7, "reg_alpha": 0.5, "reg_lambda": 10.0},
    {"ratio": 20, "weight_multiplier": 5.0, "max_depth": 3, "min_child_weight": 10,
     "learning_rate": 0.05, "n_estimators": 500, "subsample": 0.7,
     "colsample_bytree": 0.7, "reg_alpha": 1.0, "reg_lambda": 10.0},
    {"ratio": 50, "weight_multiplier": 5.0, "max_depth": 4, "min_child_weight": 10,
     "learning_rate": 0.03, "n_estimators": 700, "subsample": 0.8,
     "colsample_bytree": 0.7, "reg_alpha": 0.5, "reg_lambda": 10.0},
    {"ratio": 100, "weight_multiplier": 5.0, "max_depth": 5, "min_child_weight": 20,
     "learning_rate": 0.03, "n_estimators": 700, "subsample": 0.8,
     "colsample_bytree": 0.8, "reg_alpha": 1.0, "reg_lambda": 10.0},
)
PRECISION_TARGET = 0.90
RECALL_TARGET = 0.90
F1_TARGET = 0.90


def ensure_dirs() -> None:
    for folder in (TABLE_DIR, METRIC_DIR, MODEL_DIR, FIGURE_DIR):
        folder.mkdir(parents=True, exist_ok=True)
