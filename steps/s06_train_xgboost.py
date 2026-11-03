import gc
import sys
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, classification_report, precision_recall_curve

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
    matrix = confusion_table(y_true, scores, threshold)
    return metrics, predictions, report, matrix


def load_split(path: Path, features: list[str]):
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns.tolist()
    actual_features = [column for column in header if column not in (C.TARGET_COLUMN, *C.METADATA_COLUMNS)]
    if C.TARGET_COLUMN not in header or actual_features != features:
        raise ValueError(
            f"Cấu trúc đặc trưng hoặc cột nhãn trong {path} không khớp. "
            "Hãy dùng các tập dữ liệu đã chia chung ở bước s03."
        )
    if any(column in header for column in C.METADATA_COLUMNS):
        raise ValueError(f"{path}: dataset train phải chỉ gồm feature số và nhãn is_anomaly.")
    frame = pd.read_csv(
        path,
        usecols=features + [C.TARGET_COLUMN],
        dtype={column: np.float32 for column in features},
        encoding="utf-8-sig",
        low_memory=False,
    )
    labels = frame[C.TARGET_COLUMN]
    if labels.isna().any() or not labels.isin([0, 1]).all() or labels.nunique() != 2:
        raise ValueError(
            f"{path}: cột nhãn is_anomaly phải có đủ hai giá trị 0 và 1, "
            "không được chứa giá trị thiếu."
        )
    X = frame[features]
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError(
            f"{path}: các đặc trưng có chứa NaN hoặc giá trị vô hạn. "
            "Tệp dữ liệu gốc chưa bị thay đổi."
        )
    return X, labels.astype(np.int8)


def main(train_path: Path | None = None, model_path: Path | None = None, strategy: str | None = None) -> int:
    C.ensure_dirs()
    if train_path is None:
        experiments = (
            ("undersampling", C.UNDERSAMPLED_TRAIN_DATA, C.XGBOOST_MODEL),
            ("full_data", C.TRAIN_DATA, C.FULL_XGBOOST_MODEL),
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
        raise ValueError("Thiếu tên chiến lược hoặc đường dẫn lưu model.")
    for path in (train_path, C.VALIDATION_DATA):
        if not path.is_file():
            raise FileNotFoundError(
                f"Thiếu tập dữ liệu: {path}. Hãy chạy s03 và s04 trước."
            )
    features = load_feature_names(train_path)
    if not features:
        raise ValueError("Tệp train không có cột đặc trưng để huấn luyện.")
    print("BƯỚC S06 - TRAIN XGBOOST", flush=True)
    print(f"Đọc tập train ({strategy}): {train_path}", flush=True)
    X_train, y_train = load_split(train_path, features)
    X_validation, y_validation = load_split(C.VALIDATION_DATA, features)
    n_normal = int((y_train == 0).sum())
    n_anomaly = int((y_train == 1).sum())
    if n_normal == 0 or n_anomaly == 0:
        raise ValueError("Tập train cần có cả lớp bình thường (0) và lớp bất thường (1).")
    scale_pw = n_normal / n_anomaly
    print(f"Train: {len(y_train):,}; validation: {len(y_validation):,}; features: {len(features)}")
    print(f"scale_pos_weight = {scale_pw:.2f}", flush=True)

    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pw,
        reg_lambda=1.0,
        reg_alpha=0.0,
        eval_metric='auc',
        random_state=42,
    )
    started = time.perf_counter()
    model.set_params(early_stopping_rounds=15)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_validation, y_validation)],
        verbose=False,
    )
    fit_seconds = time.perf_counter() - started

    joblib.dump(model, model_path, compress=3)
    result = {
        "model": "XGBoost",
        "strategy": strategy,
        "model_code_source": "Project_AIC authentication anomaly pipeline",
        "training_rows": int(len(y_train)),
        "train_class_counts": {"0": n_normal, "1": n_anomaly},
        "validation_rows": int(len(y_validation)),
        "feature_count": len(features),
        "feature_columns": features,
        "scale_pos_weight": scale_pw,
        "fit_seconds": round(fit_seconds, 2),
        "fit_time": elapsed_text(fit_seconds),
        "best_iteration": int(model.best_iteration),
        "early_stopping_split": "validation",
        "early_stopping_rounds": 15,
        "data_resplit_or_resampled": False,
        "test_used_for_training": False,
        "model_path": str(model_path),
        "xgboost_version": xgb.__version__,
        "data_files": {
            name: {"path": str(path), "size_bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for name, path in (
                ("train", train_path),
                ("validation", C.VALIDATION_DATA),
            )
        },
    }
    print(f"Best iteration: {model.best_iteration}")
    print(f"Đã lưu model: {model_path} | {elapsed_text(fit_seconds)}")

    # Chọn threshold có F1 lớp 1 cao nhất trên validation
    print("\nS06.2 - TUNING THRESHOLD TRÊN VALIDATION")
    del X_train, y_train
    gc.collect()
    validation_scores = model.predict_proba(X_validation)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_validation, validation_scores)
    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    )
    best_f1 = f1_scores.max()
    candidates = np.flatnonzero(np.isclose(f1_scores, best_f1))
    best_index = int(candidates[np.argmax(thresholds[candidates])])
    selected_threshold = float(thresholds[best_index])
    validation_tuned, _, validation_tuned_report, _ = evaluate_predictions(
        y_validation, validation_scores, selected_threshold
    )
    validation_threshold_info = {
        "threshold": selected_threshold,
        "precision": float(precision[best_index]),
        "recall": float(recall[best_index]),
        "f1": float(f1_scores[best_index]),
        "threshold_selection_metric": "maximum_validation_f1",
        "pr_auc": validation_tuned["pr_auc"],
    }
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.plot(
        recall,
        precision,
        label=f"XGBoost (AP={average_precision_score(y_validation, validation_scores):.3f})",
    )
    axis.set_title("XGBoost Precision-Recall Curve trên Validation")
    axis.set_xlabel("Recall lớp 1")
    axis.set_ylabel("Precision lớp 1")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(C.FIGURES_DIR / f"06_xgboost_{strategy}_validation_pr_curve.png", dpi=150)
    plt.close(figure)

    del X_validation, y_validation, validation_scores
    gc.collect()
    X_test, y_test = load_split(C.TEST_DATA, features)
    test_scores = model.predict_proba(X_test)[:, 1]

    test_tuned, test_tuned_predictions, test_tuned_report, test_tuned_matrix = evaluate_predictions(
        y_test, test_scores, selected_threshold
    )
    tuned_report_text = (
        f"Classification report | XGBoost ({strategy}) | threshold={selected_threshold:.8f}\n"
        f"{classification_report(y_test, test_tuned_predictions, labels=[0, 1], target_names=['Normal', 'Anomaly'], digits=4, zero_division=0)}\n"
        f"AUC-ROC: {test_tuned['roc_auc']:.4f}\n"
        f"PR-AUC: {test_tuned['pr_auc']:.4f}\n"
    )
    report_path = C.TABLES_DIR / f"06_xgboost_{strategy}_tuned_classification_report.txt"
    report_path.write_text(
        tuned_report_text,
        encoding="utf-8",
    )
    save_json(
        C.METRICS_DIR / f"06_xgboost_{strategy}_confusion_test.json",
        test_tuned_matrix.to_dict(orient="index"),
    )
    confusion_figure_path = C.FIGURES_DIR / f"06_confusion_xgboost_{strategy}_test.png"
    save_confusion_matrix_figure(
        test_tuned_matrix,
        confusion_figure_path,
        f"Confusion Matrix - XGBoost ({strategy})",
    )

    selected_model = ThresholdedClassifier(
        estimator=model,
        threshold=selected_threshold,
        model_name=f"xgboost_{strategy}",
    )
    joblib.dump(selected_model, model_path, compress=3)
    selected_info = {
        "selected_model": f"xgboost_{strategy}",
        "threshold": selected_threshold,
        "selection_metric": "validation_f1",
        "validation_metrics": validation_threshold_info,
        "model_path": str(model_path),
        "threshold_embedded_in_model": True,
    }
    metrics_path = C.METRICS_DIR / f"06_xgboost_{strategy}_metrics.json"
    save_json(metrics_path, {
        "model": "XGBoost",
        "strategy": strategy,
        "model_path": str(model_path),
        "classification_report_path": str(report_path),
        "confusion_matrix_figure": str(confusion_figure_path),
        "training_summary": result,
        "threshold_tuning": {
            "rule": "maximize_validation_f1",
            "validation": {
                "metrics": validation_tuned,
                "classification_report": validation_tuned_report,
                "selected_threshold": validation_threshold_info,
            },
            "test": {
                "metrics": test_tuned,
                "classification_report": test_tuned_report,
            },
            "selected_model": selected_info,
        },
    })

    print_model_evaluation(
        f"XGBoost ({strategy})",
        validation_threshold_info,
        tuned_report_text,
        test_tuned,
        test_tuned_matrix,
        model_path,
        metrics_path,
        elapsed_text(time.perf_counter() - started),
    )

    del X_test, y_test, test_scores
    del model, selected_model
    gc.collect()
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
