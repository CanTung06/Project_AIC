"""Bước 07 - Chọn ngưỡng theo F1 trên validation và đánh giá lại trên test.

Ngưỡng chỉ được chọn từ validation; test chỉ dùng để báo cáo kết quả cuối.
Chạy: python steps/s07_tuning.py
"""

import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ids import config as C
from ids.common import calculate_metrics, confusion_table, elapsed_text, load_xy, print_table, save_json


MODELS = {
    "class_weight": C.WEIGHTED_MODEL,
    "random_undersampling": C.UNDERSAMPLED_MODEL,
}


def main() -> int:
    C.ensure_dirs()
    started = time.perf_counter()
    for path in [C.VALIDATION_DATA, C.TEST_DATA, *MODELS.values()]:
        if not path.exists():
            raise FileNotFoundError(f"Thiếu {path}; hãy chạy các bước trước theo thứ tự.")
    X_val, y_val, feature_names = load_xy(C.VALIDATION_DATA)
    X_test, y_test, _ = load_xy(C.TEST_DATA, feature_names)
    threshold_rows = []
    test_rows = []
    validation_scores = {}
    selected_thresholds = {}
    figure, axis = plt.subplots(figsize=(8, 6))

    for model_name, model_path in MODELS.items():
        model = joblib.load(model_path)
        scores_val = model.predict_proba(X_val)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_val, scores_val)
        if len(thresholds) == 0:
            raise ValueError("Không thể tìm ngưỡng: validation cần có cả hai lớp.")
        f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(
            precision[:-1] + recall[:-1], 1e-15
        )
        best_index = int(np.nanargmax(f1_values))
        threshold = float(thresholds[best_index])
        selected_thresholds[model_name] = threshold
        validation_scores[model_name] = {
            "threshold": threshold,
            "precision": float(precision[best_index]),
            "recall": float(recall[best_index]),
            "f1": float(f1_values[best_index]),
        }
        threshold_rows.append({"model": model_name, **validation_scores[model_name]})
        axis.plot(
            recall,
            precision,
            label=f"{model_name} (AP={average_precision_score(y_val, scores_val):.3f})",
        )

        scores_test = model.predict_proba(X_test)[:, 1]
        metrics = calculate_metrics(y_test, scores_test, threshold)
        test_rows.append({"model": model_name, **metrics})
        matrix = confusion_table(y_test, scores_test, threshold)
        matrix.to_csv(
            C.TABLES_DIR / f"07_confusion_{model_name}_test_tuned.csv",
            encoding="utf-8-sig",
        )
        print_table(f"Test confusion matrix | {model_name} | threshold={threshold:.6f}", matrix, index=True)

    axis.set_title("Precision-Recall Curve trên Validation")
    axis.set_xlabel("Recall lớp 1")
    axis.set_ylabel("Precision lớp 1")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(C.FIGURES_DIR / "07_validation_pr_curve.png", dpi=150)
    plt.close(figure)

    threshold_table = pd.DataFrame(threshold_rows)
    test_table = pd.DataFrame(test_rows)
    threshold_table.to_csv(C.TABLES_DIR / "07_validation_thresholds.csv", index=False, encoding="utf-8-sig")
    test_table.to_csv(C.TABLES_DIR / "07_tuned_test_metrics.csv", index=False, encoding="utf-8-sig")

    selected_name = max(validation_scores, key=lambda name: validation_scores[name]["f1"])
    selected_info = {
        "selected_model": selected_name,
        "threshold": selected_thresholds[selected_name],
        "selection_metric": "validation_f1_class_1",
        "validation_metrics": validation_scores[selected_name],
        "thresholds_by_model": selected_thresholds,
    }
    save_json(C.SELECTED_MODEL_INFO, selected_info)
    save_json(C.METRICS_DIR / "07_threshold_tuning.json", {
        "validation": validation_scores,
        "test_at_validation_threshold": test_rows,
        "selected_model": selected_info,
    })

    print_table("Ngưỡng tối ưu F1 trên validation", threshold_table)
    print_table("Kết quả test tại ngưỡng đã chọn trên validation", test_table)
    print(f"\nModel được chọn theo F1 validation: {selected_name}")
    print(f"Ngưỡng lưu tại: {C.SELECTED_MODEL_INFO}")
    print(f"Thời gian tuning và đánh giá: {elapsed_text(time.perf_counter() - started)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
