import json
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from ids import config as C
from steps.s04_evaluate_logistic_regression import calculate_metrics, choose_f1_threshold


def score_file(path, features, scaler, model, split_name):
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
        values = scaler.transform(values).astype(np.float32, copy=False)
        scores = model.predict_proba(values)[:, 1]
        label_parts.append(chunk[C.TARGET].to_numpy(dtype=np.uint8, copy=True))
        score_parts.append(scores.astype(np.float32, copy=False))
        rows += len(chunk)
        print(
            f"{split_name}: đã chấm {rows:,} dòng | "
            f"{C.elapsed_text(time.perf_counter() - started)}",
            flush=True,
        )
    if not label_parts:
        raise ValueError(f"Tập {split_name} rỗng.")
    return np.concatenate(label_parts), np.concatenate(score_parts)


def save_pr_curve(y_true, scores):
    precision, recall, _ = precision_recall_curve(y_true, scores)
    pr_auc = float(average_precision_score(y_true, scores))
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"PR-AUC={pr_auc:.6f}")
    plt.xlabel("Recall: Is Attack IP")
    plt.ylabel("Precision: Is Attack IP")
    plt.title("Precision-Recall SGDClassifier trên test")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(C.SGD_TEST_PR_CURVE, dpi=150)
    plt.close()
    return pr_auc


def main():
    C.ensure_dirs()
    bundle = joblib.load(C.SGD_MODEL)
    scaler = bundle["scaler"]
    model = bundle["model"]
    features = bundle["feature_columns"]

    y_validation, validation_scores = score_file(
        C.VALIDATION, features, scaler, model, "Validation SGDClassifier"
    )
    threshold, val_precision, val_recall, val_f1 = choose_f1_threshold(
        y_validation, validation_scores
    )
    del y_validation, validation_scores

    y_test, test_scores = score_file(
        C.TEST, features, scaler, model, "Test SGDClassifier"
    )
    test_metrics = calculate_metrics(y_test, test_scores, threshold)
    pr_auc = save_pr_curve(y_test, test_scores)
    test_metrics["pr_auc"] = pr_auc

    bundle["threshold"] = threshold
    bundle["threshold_rule"] = "Tối đa hóa F1 lớp 1 trên validation."
    joblib.dump(bundle, C.SGD_MODEL, compress=3)

    result = {
        "model": "SGDClassifier(log_loss)",
        "learning_type": "supervised_out_of_core",
        "target": C.RAW_LABEL,
        "threshold": threshold,
        "threshold_selection": "Tối đa hóa F1 lớp 1 trên validation; không dùng test để chọn ngưỡng.",
        "validation_precision_class_1_at_selected_threshold": val_precision,
        "validation_recall_class_1_at_selected_threshold": val_recall,
        "validation_f1_class_1_at_selected_threshold": val_f1,
        "test_metrics_at_f1_threshold": test_metrics,
        "test_pr_curve": str(C.SGD_TEST_PR_CURVE),
        "note": "Mô hình dự đoán nhãn Is Attack IP; kết quả không xác nhận trực tiếp đăng nhập bất thường hoặc ATO.",
    }
    C.SGD_METRICS.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nSGDClassifier — kết quả test tại ngưỡng tối đa F1 validation")
    print(test_metrics["classification_report_text"])
    print(
        f"PR-AUC={pr_auc:.6f} | Precision lớp 1={test_metrics['precision_class_1']:.6f} | "
        f"Recall lớp 1={test_metrics['recall_class_1']:.6f} | "
        f"F1 lớp 1={test_metrics['f1_class_1']:.6f}"
    )
    print(
        f"TP={test_metrics['tp']:,} | FP={test_metrics['fp']:,} | "
        f"FN={test_metrics['fn']:,} | TN={test_metrics['tn']:,}"
    )
    print(f"Ngưỡng tối đa F1={threshold:.6f} | Metrics: {C.SGD_METRICS}")


if __name__ == "__main__":
    main()
