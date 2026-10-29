"""Bước 07 - Tối đa hóa precision trên validation với recall tối thiểu"""

import json
import gc
import sys
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, classification_report, precision_recall_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ids import config as C
from ids.common import calculate_metrics, confusion_table, elapsed_text, load_xy, print_table, save_json
from ids.thresholded import ThresholdedClassifier


MODELS = {
    "random_undersampling": C.UNDERSAMPLED_MODEL,
}


def main() -> int:
    C.ensure_dirs()
    started = time.perf_counter()
    for path in [C.VALIDATION_DATA, C.TEST_DATA, *MODELS.values()]:
        if not path.exists():
            raise FileNotFoundError(f"Thiếu {path}; hãy chạy các bước trước theo thứ tự.")
    X_val, y_val, feature_names = load_xy(C.VALIDATION_DATA)
    threshold_rows = []
    validation_scores = {}
    selected_thresholds = {}
    figure, axis = plt.subplots(figsize=(8, 6))

    for model_name, model_path in MODELS.items():
        model = joblib.load(model_path)
        if isinstance(model, ThresholdedClassifier):
            model = model.estimator
        scores_val = model.predict_proba(X_val)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_val, scores_val)
        if len(thresholds) == 0:
            raise ValueError("Không thể tìm ngưỡng: validation cần có cả hai lớp.")
        eligible_indices = np.flatnonzero(recall[:-1] >= C.MIN_RECALL_TARGET)
        if eligible_indices.size == 0:
            raise ValueError(
                f"{model_name}: không có ngưỡng nào đạt recall tối thiểu "
                f"{C.MIN_RECALL_TARGET:.0%} trên validation."
            )
        eligible_precision = precision[:-1][eligible_indices]
        highest_precision = np.max(eligible_precision)
        best_candidates = eligible_indices[eligible_precision == highest_precision]
        best_index = int(best_candidates[np.argmax(thresholds[best_candidates])])
        threshold = float(thresholds[best_index])
        f1_value = 2 * precision[best_index] * recall[best_index] / max(
            precision[best_index] + recall[best_index], 1e-15
        )
        selected_thresholds[model_name] = threshold
        validation_scores[model_name] = {
            "threshold": threshold,
            "precision": float(precision[best_index]),
            "recall": float(recall[best_index]),
            "f1": float(f1_value),
            "minimum_recall_target": C.MIN_RECALL_TARGET,
        }
        threshold_rows.append({"model": model_name, **validation_scores[model_name]})
        axis.plot(
            recall,
            precision,
            label=f"{model_name} (AP={average_precision_score(y_val, scores_val):.3f})",
        )

    axis.set_title("Precision-Recall Curve trên Validation")
    axis.set_xlabel("Recall lớp 1")
    axis.set_ylabel("Precision lớp 1")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(C.FIGURES_DIR / "07_validation_pr_curve.png", dpi=150)
    plt.close(figure)

    threshold_table = pd.DataFrame(threshold_rows)
    threshold_table.to_csv(C.TABLES_DIR / "07_validation_thresholds.csv", index=False, encoding="utf-8-sig")

    selected_name = max(
        validation_scores,
        key=lambda name: (
            validation_scores[name]["precision"],
            validation_scores[name]["recall"],
            validation_scores[name]["f1"],
        ),
    )
    selected_threshold = selected_thresholds[selected_name]
    selected_info = {
        "selected_model": selected_name,
        "threshold": selected_threshold,
        "selection_metric": "validation_precision_with_minimum_recall",
        "minimum_recall_target": C.MIN_RECALL_TARGET,
        "validation_metrics": validation_scores[selected_name],
        "thresholds_by_model": selected_thresholds,
    }

    del X_val, y_val
    gc.collect()
    X_test, y_test, _ = load_xy(C.TEST_DATA, feature_names)
    selected_model = joblib.load(MODELS[selected_name])
    if isinstance(selected_model, ThresholdedClassifier):
        selected_model = selected_model.estimator
    scores_test = selected_model.predict_proba(X_test)[:, 1]
    test_metrics = calculate_metrics(y_test, scores_test, selected_threshold)
    predictions_test = (scores_test >= selected_threshold).astype(np.int8)
    tuned_report = classification_report(
        y_test,
        predictions_test,
        labels=[0, 1],
        target_names=["Normal", "Attack"],
        digits=4,
        zero_division=0,
    )
    tuned_report_text = (
        f"Classification report | {selected_name} | tuned threshold={selected_threshold:.8f}\n"
        f"{tuned_report}\nAUC-ROC: {test_metrics['roc_auc']:.4f}\n"
    )
    (C.TABLES_DIR / "07_tuned_classification_report.txt").write_text(
        tuned_report_text,
        encoding="utf-8",
    )
    test_row = {"model": selected_name, **test_metrics}
    test_table = pd.DataFrame([test_row])
    test_table.to_csv(C.TABLES_DIR / "07_tuned_test_metrics.csv", index=False, encoding="utf-8-sig")
    matrix = confusion_table(y_test, scores_test, selected_threshold)
    for model_name in MODELS:
        if model_name != selected_name:
            stale_matrix_path = C.TABLES_DIR / f"07_confusion_{model_name}_test_tuned.csv"
            stale_matrix_path.unlink(missing_ok=True)
    matrix.to_csv(
        C.TABLES_DIR / f"07_confusion_{selected_name}_test_tuned.csv",
        encoding="utf-8-sig",
    )

    final_model = ThresholdedClassifier(
        estimator=selected_model,
        threshold=selected_threshold,
        model_name=selected_name,
    )
    joblib.dump(final_model, MODELS[selected_name], compress=3)
    selected_info["model_path"] = str(MODELS[selected_name])
    selected_info["threshold_embedded_in_model"] = True

    save_json(C.SELECTED_MODEL_INFO, selected_info)
    save_json(C.METRICS_DIR / "07_threshold_tuning.json", {
        "selection_rule": "maximize_validation_precision_subject_to_minimum_recall",
        "minimum_recall_target": C.MIN_RECALL_TARGET,
        "validation": validation_scores,
        "test_at_validation_threshold": [test_row],
        "selected_model": selected_info,
    })

    print_table(
        f"Ngưỡng precision cao nhất trên validation",
        threshold_table,
    )
    print_table("Kết quả test của mô hình được chọn", test_table)
    print("\n" + tuned_report_text)
    print_table(f"Confusion matrix test | {selected_name} | threshold={selected_threshold:.6f}", matrix, index=True)
    print(f"\nModel được chọn theo precision validation với recall tối thiểu: {selected_name}")
    print(f"Ngưỡng lưu tại: {C.SELECTED_MODEL_INFO}")
    print(f"Thời gian tuning và đánh giá: {elapsed_text(time.perf_counter() - started)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
