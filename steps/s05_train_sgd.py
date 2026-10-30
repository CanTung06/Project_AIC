"""Bước 05: Train, evaluate và tuning threshold cho SGDClassifier."""

import gc
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ids import config as C
from ids.common import (
    calculate_metrics,
    confusion_table,
    elapsed_text,
    load_xy,
    print_table,
    save_json,
    signed_log_transform,
)
from ids.thresholded import ThresholdedClassifier


def evaluate_model(model) -> None:
    """Phần đánh giá của s05 trên validation và test với threshold 0.5."""
    print("\n" + "=" * 72)
    print("S05.2 - ĐÁNH GIÁ VALIDATION VÀ TEST VỚI THRESHOLD 0.5")
    print("=" * 72)

    model_name = "random_undersampling"
    metrics_by_split = {}
    reports_by_split = {}
    metric_rows = []
    report_rows = []

    for split_name, split_path in (
        ("validation", C.VALIDATION_DATA),
        ("test", C.TEST_DATA),
    ):
        if not split_path.exists():
            raise FileNotFoundError(f"Thiếu {split_path}; hãy chạy s03 trước.")

        split_started = time.perf_counter()
        X, y, _ = load_xy(split_path)
        print(f"Đã nạp {split_name}: {len(y):,} dòng")

        scores = model.predict_proba(X)[:, 1]
        metrics = calculate_metrics(y, scores, threshold=0.5)
        predictions = (scores >= 0.5).astype("int8")
        report_dict = classification_report(
            y,
            predictions,
            labels=[0, 1],
            target_names=["Normal (0)", "Attack (1)"],
            output_dict=True,
            zero_division=0,
        )
        report_text = classification_report(
            y,
            predictions,
            labels=[0, 1],
            target_names=["Normal", "Attack"],
            digits=4,
            zero_division=0,
        )

        metrics_by_split[split_name] = metrics
        reports_by_split[split_name] = report_dict
        metric_rows.append({"model": model_name, "split": split_name, **metrics})
        for class_name in ("Normal (0)", "Attack (1)", "macro avg", "weighted avg"):
            values = report_dict[class_name]
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
        save_json(
            C.METRICS_DIR / f"05_confusion_sgd_{split_name}.json",
            matrix.to_dict(orient="index"),
        )
        print(f"\nClassification report | SGDClassifier | {split_name} | threshold=0.5")
        print(report_text)
        if metrics["roc_auc"] is not None:
            print(f"AUC-ROC: {metrics['roc_auc']:.4f}")
        if metrics["pr_auc"] is not None:
            print(f"PR-AUC: {metrics['pr_auc']:.4f}")
        print_table(f"Confusion matrix | {model_name} | {split_name} | threshold=0.5", matrix, index=True)
        print(f"Hoàn tất đánh giá {split_name} | {elapsed_text(time.perf_counter() - split_started)}")

        del X, y, scores, predictions, matrix
        gc.collect()

    save_json(C.METRICS_DIR / "05_evaluation_metrics.json", {
        "metrics": {model_name: metrics_by_split},
        "classification_reports": {model_name: reports_by_split},
    })
    print_table("Chỉ số tổng hợp (Attack là trọng tâm)", pd.DataFrame(metric_rows))
    print_table("Classification report tổng hợp", pd.DataFrame(report_rows))


def tune_threshold(model) -> None:
    """Bước 07 (đã gộp): chọn threshold trên validation rồi đánh giá trên test."""
    started = time.perf_counter()
    print("\n" + "=" * 72)
    print("S05.3 - TUNING THRESHOLD TRÊN VALIDATION")
    print("=" * 72)

    model_name = "random_undersampling"
    X_val, y_val, feature_names = load_xy(C.VALIDATION_DATA)
    validation_scores_raw = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, validation_scores_raw)
    if len(thresholds) == 0:
        raise ValueError("Không thể tìm threshold: validation cần có cả hai lớp.")

    eligible_indices = np.flatnonzero(recall[:-1] >= C.MIN_RECALL_TARGET)
    if eligible_indices.size == 0:
        raise ValueError(
            f"Không có threshold nào đạt recall tối thiểu {C.MIN_RECALL_TARGET:.0%} trên validation."
        )
    eligible_precision = precision[:-1][eligible_indices]
    highest_precision = np.max(eligible_precision)
    best_candidates = eligible_indices[eligible_precision == highest_precision]
    best_index = int(best_candidates[np.argmax(thresholds[best_candidates])])
    threshold = float(thresholds[best_index])
    validation_metrics = {
        "threshold": threshold,
        "precision": float(precision[best_index]),
        "recall": float(recall[best_index]),
        "f1": float(
            2 * precision[best_index] * recall[best_index]
            / max(precision[best_index] + recall[best_index], 1e-15)
        ),
        "minimum_recall_target": C.MIN_RECALL_TARGET,
    }
    threshold_table = pd.DataFrame([{"model": model_name, **validation_metrics}])

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.plot(
        recall,
        precision,
        label=f"{model_name} (AP={average_precision_score(y_val, validation_scores_raw):.3f})",
    )
    axis.set_title("Precision-Recall Curve trên Validation")
    axis.set_xlabel("Recall lớp 1")
    axis.set_ylabel("Precision lớp 1")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(C.FIGURES_DIR / "05_sgd_validation_pr_curve.png", dpi=150)
    plt.close(figure)

    del X_val, y_val, validation_scores_raw
    gc.collect()

    X_test, y_test, _ = load_xy(C.TEST_DATA, feature_names)
    test_scores = model.predict_proba(X_test)[:, 1]
    test_metrics = calculate_metrics(y_test, test_scores, threshold)
    test_predictions = (test_scores >= threshold).astype("int8")
    test_report_text = classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=["Normal", "Attack"],
        digits=4,
        zero_division=0,
    )
    test_report_dict = classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=["Normal", "Attack"],
        output_dict=True,
        zero_division=0,
    )
    report_text = (
        f"Classification report | SGDClassifier | threshold={threshold:.8f}\n"
        f"{test_report_text}\n"
        f"AUC-ROC: {test_metrics['roc_auc']:.4f}\n"
        f"PR-AUC: {test_metrics['pr_auc']:.4f}\n"
    )
    (C.TABLES_DIR / "05_sgd_tuned_classification_report.txt").write_text(report_text, encoding="utf-8")

    test_row = {"model": model_name, **test_metrics}
    test_table = pd.DataFrame([test_row])
    matrix = confusion_table(y_test, test_scores, threshold)
    save_json(
        C.METRICS_DIR / "05_confusion_sgd_test_tuned.json",
        matrix.to_dict(orient="index"),
    )

    thresholded_model = ThresholdedClassifier(
        estimator=model,
        threshold=threshold,
        model_name=model_name,
    )
    joblib.dump(thresholded_model, C.UNDERSAMPLED_MODEL, compress=3)
    selected_info = {
        "selected_model": model_name,
        "threshold": threshold,
        "selection_metric": "validation_precision_with_minimum_recall",
        "minimum_recall_target": C.MIN_RECALL_TARGET,
        "validation_metrics": validation_metrics,
        "thresholds_by_model": {model_name: threshold},
        "model_path": str(C.UNDERSAMPLED_MODEL),
        "threshold_embedded_in_model": True,
    }
    save_json(C.SELECTED_MODEL_INFO, selected_info)
    save_json(C.METRICS_DIR / "05_threshold_tuning.json", {
        "selection_rule": "maximize_validation_precision_subject_to_minimum_recall",
        "minimum_recall_target": C.MIN_RECALL_TARGET,
        "validation": {model_name: validation_metrics},
        "validation_thresholds_by_model": [{"model": model_name, **validation_metrics}],
        "test_at_validation_threshold": [test_row],
        "test_classification_report": test_report_dict,
        "selected_model": selected_info,
    })

    print_table("Ngưỡng precision cao nhất trên validation", threshold_table)
    print_table("Kết quả test của mô hình được chọn", test_table)
    print("\n" + report_text)
    print_table(f"Confusion matrix test | SGDClassifier | threshold={threshold:.6f}", matrix, index=True)
    print(f"Ngưỡng lưu tại: {C.SELECTED_MODEL_INFO}")
    print(f"Thời gian tuning và đánh giá: {elapsed_text(time.perf_counter() - started)}")


def build_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("signed_log", FunctionTransformer(
            signed_log_transform,
            validate=False,
            feature_names_out="one-to-one",
        )),
        ("scaler", StandardScaler()),
        ("classifier", SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=0.0001,
            l1_ratio=0.15,
            max_iter=50,
            tol=0.001,
            shuffle=True,
            average=True,
            random_state=C.SEED,
        )),
    ])


def main() -> int:
    C.ensure_dirs()
    if not C.TRAIN_DATA.exists() or not C.UNDERSAMPLED_TRAIN_DATA.exists():
        raise FileNotFoundError("Hãy chạy s03_split_data.py và s04_imbalance.py trước.")
    # BƯỚC S05 - HUẤN LUYỆN SGDClassifier
    print("\n" + "=" * 72)
    print("BƯỚC S05 - HUẤN LUYỆN SGDClassifier")
    print("=" * 72)
    print("Đang đọc tập train undersampling...")
    X_train, y_train, feature_names = load_xy(C.UNDERSAMPLED_TRAIN_DATA)
    print(f"Train: {len(y_train):,} dòng, {len(feature_names):,} feature")

    model = build_pipeline()
    started = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started
    classifier = model.named_steps["classifier"]
    joblib.dump(model, C.UNDERSAMPLED_MODEL, compress=3)
    result = {
        "strategy": "random_undersampling_signed_log",
        "training_rows": int(len(y_train)),
        "feature_count": len(feature_names),
        "fit_seconds": round(fit_seconds, 2),
        "fit_time": elapsed_text(fit_seconds),
        "epochs_used": int(classifier.n_iter_),
        "feature_transform": "signed_log_then_standard_scaler",
        "model_path": str(C.UNDERSAMPLED_MODEL),
    }
    print(
        f"Đã lưu model: {C.UNDERSAMPLED_MODEL.name} | {fit_seconds:.1f} giây | "
        f"{classifier.n_iter_} epoch"
    )
    del X_train, y_train
    gc.collect()
    results = [result]

    result_table = pd.DataFrame(results)
    result_table.to_csv(C.TABLES_DIR / "05_training_summary.csv", index=False, encoding="utf-8-sig")
    with (C.METRICS_DIR / "05_training_summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(results, output_file, ensure_ascii=False, indent=2)

    print("\nTóm tắt huấn luyện:")
    print(result_table.to_string(index=False))

    # S05.2 - ĐÁNH GIÁ MODEL VỚI THRESHOLD 0.5
    evaluate_model(model)
    # S05.3 - TUNING THRESHOLD TRÊN VALIDATION, ĐÁNH GIÁ CUỐI TRÊN TEST
    tune_threshold(model)
    del model
    print(f"\nTổng thời gian s05: {elapsed_text(time.perf_counter() - started)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
