"""Bước 07: Huấn luyện, chọn threshold và đánh giá Random Forest."""

import gc
import sys
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ids import config as C
from ids.common import (
    calculate_metrics,
    confusion_table,
    elapsed_text,
    load_feature_names,
    print_model_evaluation,
    save_confusion_matrix_figure,
    save_json,
)
from ids.thresholded import ThresholdedClassifier


def load_split(path: Path, features: list[str]):
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns.tolist()
    actual_features = [
        column for column in header
        if column not in (C.TARGET_COLUMN, *C.METADATA_COLUMNS)
    ]
    if C.TARGET_COLUMN not in header or actual_features != features:
        raise ValueError(
            f"Cấu trúc feature hoặc cột nhãn trong {path} không khớp. "
            "Hãy chạy lại s03 để tạo các tập dữ liệu đồng nhất."
        )

    frame = pd.read_csv(
        path,
        usecols=features + [C.TARGET_COLUMN],
        dtype={column: np.float32 for column in features},
        encoding="utf-8-sig",
        low_memory=False,
    )
    labels = frame[C.TARGET_COLUMN]
    if labels.isna().any() or not labels.isin([0, 1]).all() or labels.nunique() != 2:
        raise ValueError(f"{path}: nhãn is_anomaly phải có đủ hai lớp 0 và 1, không có giá trị thiếu.")
    X = frame[features]
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError(f"{path}: feature có NaN hoặc vô hạn; hãy chạy lại s02/s03.")
    return X, labels.astype(np.int8)


def evaluate_predictions(y_true, scores, threshold):
    metrics = calculate_metrics(y_true, scores, threshold)
    predictions = (scores >= threshold).astype(np.int8)
    report = classification_report(
        y_true,
        predictions,
        labels=[0, 1],
        target_names=["Normal", "Anomaly"],
        output_dict=True,
        zero_division=0,
    )
    return metrics, predictions, report, confusion_table(y_true, scores, threshold)


def main(train_path: Path | None = None, model_path: Path | None = None, strategy: str | None = None) -> int:
    C.ensure_dirs()
    if train_path is None:
        experiments = (
            ("undersampling", C.UNDERSAMPLED_TRAIN_DATA, C.RANDOM_FOREST_MODEL),
            ("full_data", C.TRAIN_DATA, C.FULL_RANDOM_FOREST_MODEL),
        )
        required = [path for _, path, _ in experiments] + [C.VALIDATION_DATA, C.TEST_DATA]
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Thiếu dữ liệu đầu vào; hãy chạy s03 và s04 trước: "
                + ", ".join(map(str, missing))
            )
        for experiment_strategy, experiment_train_path, experiment_model_path in experiments:
            main(experiment_train_path, experiment_model_path, experiment_strategy)
        return 0

    if model_path is None or strategy is None:
        raise ValueError("Thiếu chiến lược train hoặc đường dẫn lưu model.")
    for path in (train_path, C.VALIDATION_DATA, C.TEST_DATA):
        if not path.is_file():
            raise FileNotFoundError(f"Thiếu tập dữ liệu {path}; hãy chạy s03 và s04 trước.")

    features = load_feature_names(train_path)
    if not features:
        raise ValueError("Tập train không có feature để huấn luyện.")

    print("BƯỚC S07 - TRAIN RANDOM FOREST", flush=True)
    print(f"Đang nạp train ({strategy}): {train_path}", flush=True)
    X_train, y_train = load_split(train_path, features)
    X_validation, y_validation = load_split(C.VALIDATION_DATA, features)
    validation_rows = len(y_validation)
    train_rows = len(y_train)
    train_counts = {
        "0": int((y_train == 0).sum()),
        "1": int((y_train == 1).sum()),
    }
    print(
        f"Train: {train_rows:,} dòng (Normal={train_counts['0']:,}, "
        f"Anomaly={train_counts['1']:,}); validation={len(y_validation):,}; "
        f"feature={len(features):,}"
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=18,
        min_samples_leaf=2,
        max_features="sqrt",
        max_samples=0.8,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=C.SEED,
    )
    fit_started = time.perf_counter()
    print("Đang huấn luyện Random Forest...", flush=True)
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - fit_started
    del X_train, y_train
    gc.collect()
    print(f"Huấn luyện xong | {elapsed_text(fit_seconds)}")

    # Chọn precision cao nhất với recall tối thiểu trên validation.
    print("\nS07.2 - CHỌN THRESHOLD TRÊN VALIDATION", flush=True)
    validation_scores = model.predict_proba(X_validation)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_validation, validation_scores)
    eligible = np.flatnonzero(recall[:-1] >= C.MIN_RECALL_TARGET)
    if eligible.size == 0:
        raise ValueError(
            f"Không có threshold đạt recall tối thiểu {C.MIN_RECALL_TARGET:.0%} trên validation."
        )
    best_precision = precision[:-1][eligible].max()
    candidates = eligible[np.isclose(precision[:-1][eligible], best_precision)]
    best_index = int(candidates[np.argmax(thresholds[candidates])])
    threshold = float(thresholds[best_index])
    validation_metrics = calculate_metrics(y_validation, validation_scores, threshold)
    validation_threshold = {
        "threshold": threshold,
        "precision": float(precision[best_index]),
        "recall": float(recall[best_index]),
        "f1": float(
            2 * precision[best_index] * recall[best_index]
            / max(precision[best_index] + recall[best_index], 1e-15)
        ),
        "threshold_selection_metric": "max_precision_with_minimum_recall",
        "minimum_recall_target": C.MIN_RECALL_TARGET,
        "pr_auc": validation_metrics["pr_auc"],
    }

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.plot(
        recall,
        precision,
        label=f"Random Forest (AP={average_precision_score(y_validation, validation_scores):.3f})",
    )
    axis.set_title("Random Forest Precision-Recall Curve trên Validation")
    axis.set_xlabel("Recall lớp 1")
    axis.set_ylabel("Precision lớp 1")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    pr_curve_path = C.FIGURES_DIR / f"07_random_forest_{strategy}_validation_pr_curve.png"
    figure.savefig(pr_curve_path, dpi=150)
    plt.close(figure)
    del X_validation, y_validation, validation_scores
    gc.collect()

    print("\nS07.3 - ĐÁNH GIÁ TRÊN TEST", flush=True)
    X_test, y_test = load_split(C.TEST_DATA, features)
    test_scores = model.predict_proba(X_test)[:, 1]
    test_metrics, test_predictions, test_report, test_matrix = evaluate_predictions(
        y_test, test_scores, threshold
    )
    report_text = (
        f"Classification report | Random Forest ({strategy}) | threshold={threshold:.8f}\n"
        f"{classification_report(y_test, test_predictions, labels=[0, 1], target_names=['Normal', 'Anomaly'], digits=4, zero_division=0)}\n"
        f"AUC-ROC: {test_metrics['roc_auc']:.4f}\n"
        f"PR-AUC: {test_metrics['pr_auc']:.4f}\n"
    )

    report_path = C.TABLES_DIR / f"07_random_forest_{strategy}_tuned_classification_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    confusion_json_path = C.METRICS_DIR / f"07_random_forest_{strategy}_confusion_test.json"
    save_json(confusion_json_path, test_matrix.to_dict(orient="index"))
    confusion_figure_path = C.FIGURES_DIR / f"07_confusion_random_forest_{strategy}_test.png"
    save_confusion_matrix_figure(
        test_matrix,
        confusion_figure_path,
        f"Confusion Matrix - Random Forest ({strategy})",
    )

    final_model = ThresholdedClassifier(
        estimator=model,
        threshold=threshold,
        model_name=f"random_forest_{strategy}",
    )
    joblib.dump(final_model, model_path, compress=3)
    metrics_path = C.METRICS_DIR / f"07_random_forest_{strategy}_metrics.json"
    result = {
        "model": "RandomForestClassifier",
        "strategy": strategy,
        "model_path": str(model_path),
        "classification_report_path": str(report_path),
        "confusion_matrix_json": str(confusion_json_path),
        "confusion_matrix_figure": str(confusion_figure_path),
        "validation_pr_curve": str(pr_curve_path),
        "training_summary": {
            "training_rows": int(train_rows),
            "train_class_counts": train_counts,
            "validation_rows": int(validation_rows),
            "feature_count": len(features),
            "feature_columns": features,
            "parameters": {
                "n_estimators": 200,
                "max_depth": 18,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
                "max_samples": 0.8,
                "class_weight": "balanced_subsample",
                "n_jobs": -1,
                "random_state": C.SEED,
            },
            "fit_seconds": round(fit_seconds, 2),
            "fit_time": elapsed_text(fit_seconds),
            "test_used_for_training": False,
        },
        "threshold_tuning": {
            "rule": "maximize_validation_precision_subject_to_minimum_recall",
            "minimum_recall_target": C.MIN_RECALL_TARGET,
            "validation": {
                "selected_threshold": validation_threshold,
            },
            "test": {
                "metrics": test_metrics,
                "classification_report": test_report,
            },
            "threshold_embedded_in_model": True,
        },
    }
    save_json(metrics_path, result)
    print_model_evaluation(
        f"Random Forest ({strategy})",
        validation_threshold,
        report_text,
        test_metrics,
        test_matrix,
        model_path,
        metrics_path,
        elapsed_text(time.perf_counter() - fit_started),
    )

    del X_test, y_test, test_scores, model, final_model
    gc.collect()
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
