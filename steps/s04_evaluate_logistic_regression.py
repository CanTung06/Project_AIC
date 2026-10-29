import json
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)

from ids import config as C


def score_file(path, features, model, split_name):
    labels_parts, scores_parts = [], []
    rows = 0
    started = time.perf_counter()
    for chunk in pd.read_csv(
        path,
        usecols=[*features, C.TARGET],
        chunksize=C.CHUNK_SIZE,
        low_memory=False,
    ):
        values = chunk[features].to_numpy(dtype=np.float32, copy=False)
        scores = model.predict_proba(values)[:, 1]
        labels_parts.append(chunk[C.TARGET].to_numpy(dtype=np.uint8, copy=True))
        scores_parts.append(scores.astype(np.float32, copy=False))
        rows += len(chunk)
        print(
            f"{split_name}: đã chấm {rows:,} dòng | "
            f"{C.elapsed_text(time.perf_counter() - started)}",
            flush=True,
        )
    if not labels_parts:
        raise ValueError(f"Tập {split_name} rỗng.")
    return np.concatenate(labels_parts), np.concatenate(scores_parts)


def choose_f1_threshold(y_true, scores):
    if not len(scores) or np.unique(y_true).size != 2:
        raise ValueError("Validation cần có cả lớp 0 và lớp 1 để chọn ngưỡng.")
    # Histogram giữ bộ nhớ thấp hơn PR-curve có hàng triệu ngưỡng.
    edges = np.linspace(0.0, 1.0, 10_001)
    positive_hist = np.histogram(scores[y_true == 1], bins=edges)[0]
    negative_hist = np.histogram(scores[y_true == 0], bins=edges)[0]
    tp = np.cumsum(positive_hist[::-1], dtype=np.int64)[::-1]
    fp = np.cumsum(negative_hist[::-1], dtype=np.int64)[::-1]
    fn = int(np.count_nonzero(y_true == 1)) - tp
    denominator = 2 * tp + fp + fn
    f1 = np.divide(
        2 * tp,
        denominator,
        out=np.zeros_like(tp, dtype=np.float64),
        where=denominator > 0,
    )
    best = int(np.argmax(f1))
    precision = int(tp[best]) / (int(tp[best]) + int(fp[best])) if tp[best] + fp[best] else 0.0
    recall = int(tp[best]) / int(np.count_nonzero(y_true == 1))
    return float(edges[best]), precision, recall, float(f1[best])


def calculate_metrics(y_true, scores, threshold):
    predicted = (scores >= threshold).astype(np.uint8)
    report = classification_report(
        y_true,
        predicted,
        labels=[0, 1],
        target_names=["Is Attack IP=0", "Is Attack IP=1"],
        output_dict=True,
        zero_division=0,
    )
    report.pop("accuracy", None)
    report_text = pd.DataFrame(report).T.to_string(
        float_format=lambda value: f"{value:.6f}"
    )
    tn, fp, fn, tp = (
        int(value)
        for value in confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    )
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "rows": int(len(y_true)),
        "actual_class_1_rate": float(np.mean(y_true)),
        "predicted_class_1_rows": int(predicted.sum()),
        "predicted_class_1_rate": float(np.mean(predicted)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision_class_1": precision,
        "recall_class_1": recall,
        "f1_class_1": f1,
        "classification_report": report,
        "classification_report_text": report_text,
    }


def save_pr_curve(y_true, scores):
    precision, recall, _ = precision_recall_curve(y_true, scores)
    pr_auc = float(average_precision_score(y_true, scores))
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"PR-AUC={pr_auc:.6f}")
    plt.xlabel("Recall: Is Attack IP")
    plt.ylabel("Precision: Is Attack IP")
    plt.title("Precision-Recall Logistic Regression trên test")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(C.TEST_PR_CURVE, dpi=150)
    plt.close()
    return pr_auc


def main():
    C.ensure_dirs()
    bundle = joblib.load(C.MODEL)
    model = bundle["model"]
    features = bundle["feature_columns"]

    y_validation, validation_scores = score_file(
        C.VALIDATION, features, model, "Validation Logistic Regression"
    )
    threshold, val_precision, val_recall, val_f1 = choose_f1_threshold(
        y_validation, validation_scores
    )
    del y_validation, validation_scores

    y_test, test_scores = score_file(C.TEST, features, model, "Test Logistic Regression")
    metrics_at_f1 = calculate_metrics(y_test, test_scores, threshold)
    pr_auc = save_pr_curve(y_test, test_scores)
    metrics_at_f1["pr_auc"] = pr_auc

    bundle["threshold"] = threshold
    bundle["threshold_rule"] = "Tối đa hóa F1 lớp 1 trên validation."
    joblib.dump(bundle, C.MODEL, compress=3)
    result = {
        "model": "LogisticRegression",
        "learning_type": "supervised",
        "target": C.RAW_LABEL,
        "threshold": threshold,
        "threshold_selection": "Tối đa hóa F1 lớp 1 trên validation; test không dùng để chọn ngưỡng.",
        "validation_precision_class_1_at_selected_threshold": val_precision,
        "validation_recall_class_1_at_selected_threshold": val_recall,
        "validation_f1_class_1_at_selected_threshold": val_f1,
        "test_metrics_at_f1_threshold": metrics_at_f1,
        "test_pr_curve": str(C.TEST_PR_CURVE),
        "note": "Mô hình dự đoán nhãn Is Attack IP; kết quả không xác nhận trực tiếp đăng nhập bất thường hoặc ATO.",
    }
    C.METRICS.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nLogistic Regression — kết quả test tại ngưỡng tối đa F1 validation")
    print(metrics_at_f1["classification_report_text"])
    print(
        f"PR-AUC={pr_auc:.6f} | Precision lớp 1={metrics_at_f1['precision_class_1']:.6f} | "
        f"Recall lớp 1={metrics_at_f1['recall_class_1']:.6f} | "
        f"F1 lớp 1={metrics_at_f1['f1_class_1']:.6f}"
    )
    print(
        f"TP={metrics_at_f1['tp']:,} | FP={metrics_at_f1['fp']:,} | "
        f"FN={metrics_at_f1['fn']:,} | TN={metrics_at_f1['tn']:,}"
    )
    print(f"Ngưỡng tối đa F1={threshold:.6f} | Metrics: {C.METRICS}")


if __name__ == "__main__":
    main()
