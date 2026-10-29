"""Bước 06 - Đánh giá SGDClassifier, in classification report và confusion matrix. Đánh giá validation/test với ngưỡng 0.5"""

import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ids import config as C
from ids.common import calculate_metrics, confusion_table, elapsed_text, load_xy, print_table


MODELS = {
    "random_undersampling": C.UNDERSAMPLED_MODEL,
}


def main() -> int:
    C.ensure_dirs()
    started = time.perf_counter()
    for path in [C.VALIDATION_DATA, C.TEST_DATA, *MODELS.values()]:
        if not path.exists():
            raise FileNotFoundError(f"Thiếu {path}; hãy chạy các bước trước theo thứ tự.")

    feature_names = None
    splits = {}
    for split_name, path in [("validation", C.VALIDATION_DATA), ("test", C.TEST_DATA)]:
        X, y, feature_names = load_xy(path, feature_names)
        splits[split_name] = (X, y)
        print(f"Đã nạp {split_name}: {len(y):,} dòng")

    metric_rows = []
    report_rows = []
    metrics_json = {}
    for model_name, model_path in MODELS.items():
        model = joblib.load(model_path)
        metrics_json[model_name] = {}
        for split_name, (X, y) in splits.items():
            split_started = time.perf_counter()
            scores = model.predict_proba(X)[:, 1]
            result = calculate_metrics(y, scores, threshold=0.5)
            metric_rows.append({"model": model_name, "split": split_name, **result})
            metrics_json[model_name][split_name] = result

            predicted = (scores >= 0.5).astype("int8")
            report = classification_report(
                y,
                predicted,
                labels=[0, 1],
                target_names=["Normal (0)", "Attack (1)"],
                output_dict=True,
                zero_division=0,
            )
            print(f"\nClassification report | {model_name} | {split_name} | threshold=0.5")
            print(classification_report(
                y,
                predicted,
                labels=[0, 1],
                target_names=["Normal", "Attack"],
                digits=4,
                zero_division=0,
            ))
            if result["roc_auc"] is not None:
                print(f"AUC-ROC: {result['roc_auc']:.4f}")
            for class_name in ["Normal (0)", "Attack (1)", "macro avg", "weighted avg"]:
                values = report[class_name]
                report_rows.append({
                    "model": model_name,
                    "split": split_name,
                    "class": class_name,
                    "precision": values["precision"],
                    "recall": values["recall"],
                    "f1_score": values["f1-score"],
                    "support": int(values["support"]),
                })

            matrix = confusion_table(y, scores, 0.5)
            matrix_path = C.TABLES_DIR / f"06_confusion_{model_name}_{split_name}.csv"
            matrix.to_csv(matrix_path, encoding="utf-8-sig")
            print_table(f"Confusion matrix | {model_name} | {split_name} | threshold=0.5", matrix, index=True)
            print(f"Hoàn tất {model_name}/{split_name} | {elapsed_text(time.perf_counter() - split_started)}")

    metrics_table = pd.DataFrame(metric_rows)
    report_table = pd.DataFrame(report_rows)
    metrics_table.to_csv(C.TABLES_DIR / "06_metrics_comparison.csv", index=False, encoding="utf-8-sig")
    report_table.to_csv(C.TABLES_DIR / "06_classification_report.csv", index=False, encoding="utf-8-sig")
    with (C.METRICS_DIR / "06_evaluation_metrics.json").open("w", encoding="utf-8") as output_file:
        json.dump(metrics_json, output_file, ensure_ascii=False, indent=2)

    print_table("Chỉ số tổng hợp (tập Attack (1) là trọng tâm)", metrics_table)
    print_table("Classification report", report_table)
    print(f"\nTổng thời gian đánh giá: {elapsed_text(time.perf_counter() - started)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
