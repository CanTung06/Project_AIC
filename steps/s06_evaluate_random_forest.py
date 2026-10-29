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
    label_parts, score_parts = [], []
    rows = 0
    started = time.perf_counter()
    for chunk in pd.read_csv(
        path,
        usecols=[*features, C.TARGET],
        chunksize=C.CHUNK_SIZE,
        low_memory=False,
    ):
        values = chunk[features].to_numpy(dtype=np.float32, copy=False)
        labels = chunk[C.TARGET].to_numpy(dtype=np.uint8, copy=False)
        probabilities = model.predict_proba(values)[:, 1]
        label_parts.append(labels.copy())
        score_parts.append(probabilities.astype(np.float32, copy=False))
        rows += len(chunk)
        print(
            f"{split_name}: đã chấm {rows:,} dòng | "
            f"{C.elapsed_text(time.perf_counter() - started)}",
            flush=True,
        )
    if not label_parts:
        raise ValueError(f"Tập {split_name} rỗng.")
    return np.concatenate(label_parts), np.concatenate(score_parts)


def choose_f1_threshold(y_true, scores):
    if not len(scores) or np.unique(y_true).size != 2:
        raise ValueError("Validation cần có mẫu của cả hai lớp để chọn ngưỡng.")

    # Dùng histogram để tránh tạo mảng PR có hàng triệu ngưỡng và tốn RAM.
    edges = np.linspace(0.0, 1.0, 10_001)
    positive_hist = np.histogram(scores[y_true == 1], bins=edges)[0]
    negative_hist = np.histogram(scores[y_true == 0], bins=edges)[0]
    true_positive = np.cumsum(positive_hist[::-1], dtype=np.int64)[::-1]
    false_positive = np.cumsum(negative_hist[::-1], dtype=np.int64)[::-1]
    false_negative = int(np.count_nonzero(y_true == 1)) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=denominator > 0,
    )
    best = int(np.argmax(f1))
    tp = int(true_positive[best])
    fp = int(false_positive[best])
    positives = int(np.count_nonzero(y_true == 1))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / positives if positives else 0.0
    return float(edges[best]), precision, recall, float(f1[best])


def save_pr_curve(y_true, scores):
    precision, recall, _ = precision_recall_curve(y_true, scores)
    pr_auc = float(average_precision_score(y_true, scores))
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"PR-AUC={pr_auc:.6f}")
    plt.xlabel("Recall: Is Attack IP")
    plt.ylabel("Precision: Is Attack IP")
    plt.title("Precision-Recall Random Forest trên test")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(C.SUPERVISED_TEST_PR_CURVE, dpi=150)
    plt.close()
    return pr_auc


def evaluate(y_true, scores, threshold):
    y_pred = (scores >= threshold).astype(np.uint8)
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Is Attack IP=0", "Is Attack IP=1"],
        output_dict=True,
        zero_division=0,
    )
    report.pop("accuracy", None)
    report_text = pd.DataFrame(report).T.to_string(
        float_format=lambda value: f"{value:.6f}"
    )
    tn, fp, fn, tp = (int(value) for value in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "rows": int(len(y_true)),
        "actual_class_1_rate": float(np.mean(y_true)),
        "predicted_class_1_rows": int(y_pred.sum()),
        "predicted_class_1_rate": float(np.mean(y_pred)),
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


def main():
    C.ensure_dirs()
    bundle = joblib.load(C.SUPERVISED_MODEL)
    model = bundle["model"]
    features = bundle["feature_columns"]

    y_validation, validation_scores = score_file(
        C.VALIDATION, features, model, "Validation Random Forest"
    )
    threshold, val_precision, val_recall, val_f1 = choose_f1_threshold(
        y_validation, validation_scores
    )
    print(
        f"Ngưỡng chọn trên validation theo F1 lớp 1: {threshold:.8f} "
        f"(precision={val_precision:.4f}, recall={val_recall:.4f}, F1={val_f1:.4f})",
        flush=True,
    )
    del y_validation, validation_scores

    y_test, test_scores = score_file(C.TEST, features, model, "Test Random Forest")
    test_metrics_at_f1 = evaluate(y_test, test_scores, threshold)
    pr_auc = save_pr_curve(y_test, test_scores)
    test_metrics_at_f1["pr_auc"] = pr_auc
    result = {
        "model": "RandomForestClassifier",
        "learning_type": "supervised",
        "target": C.RAW_LABEL,
        "threshold": threshold,
        "threshold_selection": "Tối đa hóa F1 lớp 1 trên validation; không dùng test để chọn ngưỡng.",
        "validation_precision_class_1_at_selected_threshold": val_precision,
        "validation_recall_class_1_at_selected_threshold": val_recall,
        "validation_f1_class_1_at_selected_threshold": val_f1,
        "test_metrics_at_f1_threshold": test_metrics_at_f1,
        "test_pr_curve": str(C.SUPERVISED_TEST_PR_CURVE),
        "note": "Mô hình học nhãn Is Attack IP; chỉ số đo khả năng dự đoán nhãn này, không phải xác nhận ATO.",
    }
    bundle["threshold"] = threshold
    joblib.dump(bundle, C.SUPERVISED_MODEL, compress=3)
    C.SUPERVISED_METRICS.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nRandom Forest — đánh giá trên test")
    print("Ngưỡng tối đa F1 được chọn trên validation:")
    print(test_metrics_at_f1["classification_report_text"])
    print(
        f"PR-AUC={test_metrics_at_f1['pr_auc']:.8g} | "
        f"Precision lớp 1={test_metrics_at_f1['precision_class_1']:.8g} | "
        f"Recall lớp 1={test_metrics_at_f1['recall_class_1']:.8g} | "
        f"F1 lớp 1={test_metrics_at_f1['f1_class_1']:.8g}"
    )
    print(
        f"TP={test_metrics_at_f1['tp']:,} | FP={test_metrics_at_f1['fp']:,} | "
        f"FN={test_metrics_at_f1['fn']:,} | TN={test_metrics_at_f1['tn']:,}"
    )
    print(f"Metrics: {C.SUPERVISED_METRICS}")


if __name__ == "__main__":
    main()
