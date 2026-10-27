"""Chỉ số nhị phân và chọn ngưỡng chỉ dựa trên validation."""
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from . import config as C


def binary_metrics(y_true, scores, threshold: float) -> dict:
    y = np.asarray(y_true, dtype=np.uint8)
    score = np.asarray(scores, dtype=np.float64)
    predicted = score >= threshold
    actual_positive = y == 1
    actual_negative = ~actual_positive
    tp = int(np.count_nonzero(predicted & actual_positive))
    fp = int(np.count_nonzero(predicted & actual_negative))
    fn = int(np.count_nonzero(~predicted & actual_positive))
    tn = int(np.count_nonzero(~predicted & actual_negative))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    result = {
        "threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "precision": precision, "recall": recall, "f1": f1,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "fp_per_million_negatives": fp * 1_000_000 / (fp + tn) if fp + tn else 0.0,
        "brier_score": float(brier_score_loss(y, score)),
    }
    if np.unique(y).size == 2:
        result["pr_auc"] = float(average_precision_score(y, score))
        result["roc_auc"] = float(roc_auc_score(y, score))
    else:
        result["pr_auc"] = None
        result["roc_auc"] = None
    return result


def choose_validation_threshold(y_true, scores) -> tuple[float, dict]:
    """Tìm ngưỡng trên các điểm dương của validation; không đọc test."""
    y = np.asarray(y_true, dtype=np.uint8)
    score = np.asarray(scores, dtype=np.float64)
    positive_scores = np.unique(score[y == 1])
    if not positive_scores.size:
        raise ValueError("Validation không có mẫu takeover; không thể chọn ngưỡng.")
    candidates = np.unique(np.concatenate((positive_scores, np.array([0.5, 1.0 + 1e-12]))))
    rows = [binary_metrics(y, score, float(t)) for t in candidates]
    qualified = [
        row for row in rows
        if (
            row["precision"] >= C.PRECISION_TARGET
            and row["recall"] >= C.RECALL_TARGET
            and row["f1"] >= C.F1_TARGET
        )
    ]
    if qualified:
        best = min(qualified, key=lambda row: (row["fp"], -row["recall"], -row["precision"]))
        best["validation_target_met"] = True
        return best["threshold"], best
    best = max(rows, key=lambda row: (row["f1"], -row["fp"], row["recall"]))
    best["validation_target_met"] = False
    return best["threshold"], best
