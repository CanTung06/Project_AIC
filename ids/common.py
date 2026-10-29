"""Hàm đọc dữ liệu, tính chỉ số và lưu kết quả dùng chung."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import config as C


def signed_log_transform(values):
    values = np.asarray(values, dtype=np.float64)
    return np.sign(values) * np.log1p(np.abs(values))


def elapsed_text(seconds: float) -> str:
    seconds = int(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def load_feature_names(path: Path) -> list[str]:
    columns = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns.tolist()
    return [
        column for column in columns
        if column != C.TARGET_COLUMN and column not in C.METADATA_COLUMNS
    ]


def load_xy(path: Path, feature_names: list[str] | None = None):
    if feature_names is None:
        feature_names = load_feature_names(path)
    selected_columns = feature_names + [C.TARGET_COLUMN]
    frame = pd.read_csv(
        path,
        usecols=selected_columns,
        dtype={column: np.float32 for column in feature_names},
        encoding="utf-8-sig",
        low_memory=False,
    )
    X = frame[feature_names]
    y = frame[C.TARGET_COLUMN].astype(np.int8)
    return X, y, feature_names


def calculate_metrics(y_true, scores, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = (scores >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    result = {
        "threshold": float(threshold),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "support_0": int((y_true == 0).sum()),
        "support_1": int((y_true == 1).sum()),
    }
    if 0 < int(y_true.sum()) < len(y_true):
        result["pr_auc"] = float(average_precision_score(y_true, scores))
        result["roc_auc"] = float(roc_auc_score(y_true, scores))
    else:
        result["pr_auc"] = None
        result["roc_auc"] = None
    return result


def confusion_table(y_true, scores, threshold: float) -> pd.DataFrame:
    predictions = (np.asarray(scores) >= threshold).astype(np.int8)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    return pd.DataFrame(
        matrix,
        index=pd.Index(["Actual_0_Normal", "Actual_1_Attack"], name="actual"),
        columns=pd.Index(["Predicted_0_Normal", "Predicted_1_Attack"], name="predicted"),
    )


def save_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(content, output_file, ensure_ascii=False, indent=2)


def print_table(title: str, frame: pd.DataFrame, index: bool = False) -> None:
    print(f"\n{title}")
    print(frame.to_string(index=index, float_format=lambda value: f"{value:.4f}"))
