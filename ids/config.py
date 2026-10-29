from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = ROOT / "data" / "raw" / "rba-dataset.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
METRIC_DIR = OUTPUT_DIR / "metrics"
MODEL_DIR = OUTPUT_DIR / "models"

RAW_LABEL = "Is Attack IP"
TARGET = "is_anomaly"
SOURCE_INDEX = "source_index"
TRAIN = PROCESSED_DIR / "train.csv"
VALIDATION = PROCESSED_DIR / "validation.csv"
TEST = PROCESSED_DIR / "test.csv"
SPLIT_SUMMARY = TABLE_DIR / "02_split_summary.csv"
FEATURE_SCHEMA = TABLE_DIR / "02_feature_schema.json"

MODEL = MODEL_DIR / "03_logistic_regression_is_attack_ip.joblib"
TRAINING_SUMMARY = METRIC_DIR / "03_logistic_regression_training_summary.json"
METRICS = METRIC_DIR / "04_logistic_regression_evaluation_metrics.json"
TEST_PR_CURVE = FIGURE_DIR / "04_logistic_regression_test_pr_curve.png"
SUPERVISED_MODEL = MODEL_DIR / "05_random_forest_is_attack_ip.joblib"
SUPERVISED_TRAINING_SUMMARY = METRIC_DIR / "05_random_forest_training_summary.json"
SUPERVISED_METRICS = METRIC_DIR / "06_random_forest_evaluation_metrics.json"
SUPERVISED_TEST_PR_CURVE = FIGURE_DIR / "06_random_forest_test_pr_curve.png"
SGD_MODEL = MODEL_DIR / "07_sgd_classifier_is_attack_ip.joblib"
SGD_TRAINING_SUMMARY = METRIC_DIR / "07_sgd_classifier_training_summary.json"
SGD_METRICS = METRIC_DIR / "08_sgd_classifier_evaluation_metrics.json"
SGD_TEST_PR_CURVE = FIGURE_DIR / "08_sgd_classifier_test_pr_curve.png"
MODEL_COMPARISON = METRIC_DIR / "09_model_comparison.json"
MODEL_COMPARISON_TABLE = TABLE_DIR / "09_model_comparison.csv"

CHUNK_SIZE = 500_000
LR_PER_CLASS_SAMPLE_SIZE = 250_000
LR_C = 1.0
LR_MAX_ITER = 300
LR_TOL = 1e-3
SEED = 42
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
RF_TRAIN_SAMPLE_SIZE = 500_000
RF_PER_CLASS_SAMPLE_SIZE = 250_000  # 250,000 mẫu mỗi lớp (Tỷ lệ cân bằng 50-50)
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 20
RF_MIN_SAMPLES_LEAF = 5
RF_MAX_SAMPLES = 100_000
SGD_EPOCHS = 3
SGD_ALPHA = 1e-5
SGD_LEARNING_RATE = "optimal"

FEATURES = [
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
    "month_sin", "month_cos", "weekend", "user_prior_logins_log1p",
    "login_successful", "login_successful_missing",
    "round_trip_time_log1p", "round_trip_time_missing",
    "ip_global_prior_count_log1p", "ip_global_prior_frequency",
    "ip_user_prior_count_log1p", "ip_user_prior_frequency",
    "country_user_prior_count_log1p", "country_user_prior_frequency",
    "browser_user_prior_count_log1p", "browser_user_prior_frequency",
    "device_type_user_prior_count_log1p", "device_type_user_prior_frequency",
    "asn_global_prior_count_log1p", "asn_global_prior_frequency",
    "asn_user_prior_count_log1p", "asn_user_prior_frequency",
    "region_user_prior_count_log1p", "region_user_prior_frequency",
    "os_user_prior_count_log1p", "os_user_prior_frequency",
    "device_familiarity_index",
    "network_vs_device_mismatch",
    "suspicious_auth_failure",
]


def ensure_dirs():
    for path in (PROCESSED_DIR, FIGURE_DIR, TABLE_DIR, METRIC_DIR, MODEL_DIR):
        path.mkdir(parents=True, exist_ok=True)


def elapsed_text(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def parse_label(values):
    normalized = values.fillna("").astype("string").str.strip().str.lower()
    mapped = normalized.map({"true": 1, "false": 0, "1": 1, "0": 0})
    if mapped.isna().any():
        examples = normalized[mapped.isna()].value_counts().head().to_dict()
        raise ValueError(f"Nhãn {RAW_LABEL} không hợp lệ: {examples}")
    return mapped.to_numpy(dtype=np.uint8)


def split_for_source_index(source_indices):
    """Chia ổn định xấp xỉ 70/15/15 theo source_index."""
    values = np.asarray(source_indices, dtype=np.uint64).copy()
    values += np.uint64(SEED)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    values ^= values >> np.uint64(31)
    bucket = values % np.uint64(1000)
    split = np.full(len(values), 2, dtype=np.uint8)
    split[bucket < 700] = 0
    split[(bucket >= 700) & (bucket < 850)] = 1
    return split
