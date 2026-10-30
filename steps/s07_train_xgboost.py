"""Bước 07: Train, evaluate và tuning threshold cho XGBoost."""

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
    print_table,
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
        target_names=["Normal", "Attack"],
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_table(y_true, scores, threshold)
    return metrics, predictions, report, matrix


def load_split(path: Path, features: list[str]):
    """Read prepared numeric features without cleaning, resampling, or splitting."""
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns.tolist()
    actual_features = [column for column in header if column not in (C.TARGET_COLUMN, *C.METADATA_COLUMNS)]
    if C.TARGET_COLUMN not in header or actual_features != features:
        raise ValueError(f"Feature schema/target mismatch in {path}. Use the team's common splits.")
    frame = pd.read_csv(
        path,
        usecols=features + [C.TARGET_COLUMN],
        dtype={column: np.float32 for column in features},
        encoding="utf-8-sig",
        low_memory=False,
    )
    labels = frame[C.TARGET_COLUMN]
    if labels.isna().any() or not labels.isin([0, 1]).all() or labels.nunique() != 2:
        raise ValueError(f"{path}: is_attack must contain both 0 and 1 without missing values.")
    X = frame[features]
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError(f"{path}: NaN/infinity in shared features. The source file was not modified.")
    return X, labels.astype(np.int8)


def main() -> int:
    C.ensure_dirs()
    data_dir = C.PROCESSED_DIR
    for path in (C.TRAIN_DATA, C.VALIDATION_DATA):
        if not path.is_file():
            raise FileNotFoundError(f"Thiếu tập dữ liệu đã chia: {path}. Hãy chạy s03 trước.")
    features = load_feature_names(C.TRAIN_DATA)
    if not features:
        raise ValueError("The training file has no feature columns.")
    print("BƯỚC S07 - TRAIN XGBOOST", flush=True)
    print(f"Đọc bộ data chung: {data_dir}", flush=True)
    X_train, y_train = load_split(C.TRAIN_DATA, features)
    X_validation, y_validation = load_split(C.VALIDATION_DATA, features)
    n_normal = int((y_train == 0).sum())
    n_attack = int((y_train == 1).sum())
    if n_normal == 0 or n_attack == 0:
        raise ValueError("Tập train cần có cả lớp bình thường (0) và lớp tấn công (1).")
    scale_pw = n_normal / n_attack
    print(f"Train: {len(y_train):,}; validation: {len(y_validation):,}; features: {len(features)}")
    print(f"scale_pos_weight = {scale_pw:.2f}", flush=True)

    # Hyperparameters from the user's uploaded 05_xgboost_ids.py.
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

    joblib.dump(model, C.XGBOOST_MODEL, compress=3)
    result = {
        "model": "XGBoost",
        "strategy": "full_shared_train_class_weight",
        "model_code_source": "user-provided 05_xgboost_ids.py",
        "training_rows": int(len(y_train)),
        "train_class_counts": {"0": n_normal, "1": n_attack},
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
        "model_path": str(C.XGBOOST_MODEL),
        "xgboost_version": xgb.__version__,
        "data_files": {
            name: {"path": str(path), "size_bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for name, path in (("train", C.TRAIN_DATA), ("validation", C.VALIDATION_DATA))
        },
    }
    save_json(C.METRICS_DIR / "07_training_summary.json", [result])
    table_keys = ("model", "training_rows", "feature_count", "scale_pos_weight",
                  "fit_seconds", "best_iteration", "model_path")
    pd.DataFrame([{key: result[key] for key in table_keys}]).to_csv(
        C.TABLES_DIR / "07_training_summary.csv", index=False, encoding="utf-8-sig"
    )
    print(f"Best iteration: {model.best_iteration}")
    print(f"Đã lưu model: {C.XGBOOST_MODEL} | {elapsed_text(fit_seconds)}")

    # S07.2 - ĐÁNH GIÁ VALIDATION/TEST VỚI THRESHOLD CỐ ĐỊNH 0.5.
    print("\nS07.2 - ĐÁNH GIÁ VỚI THRESHOLD 0.5")
    validation_scores = model.predict_proba(X_validation)[:, 1]
    validation_default, _, validation_default_report, validation_default_matrix = evaluate_predictions(
        y_validation, validation_scores, 0.5
    )
    print("\nClassification report | XGBoost | validation | threshold=0.5")
    print(classification_report(
        y_validation,
        (validation_scores >= 0.5).astype(np.int8),
        labels=[0, 1], target_names=["Normal", "Attack"], digits=4, zero_division=0,
    ))
    print(f"PR-AUC: {validation_default['pr_auc']:.4f} | AUC-ROC: {validation_default['roc_auc']:.4f}")
    print(
        f"TP={validation_default['tp']:,} FP={validation_default['fp']:,} "
        f"FN={validation_default['fn']:,} TN={validation_default['tn']:,}"
    )
    print_table("Confusion matrix | XGBoost | validation | threshold=0.5", validation_default_matrix, index=True)
    save_json(
        C.METRICS_DIR / "07_confusion_xgboost_validation.json",
        validation_default_matrix.to_dict(orient="index"),
    )

    if not C.TEST_DATA.is_file():
        raise FileNotFoundError(f"Thiếu tập test: {C.TEST_DATA}")
    del X_train, y_train
    gc.collect()
    X_test, y_test = load_split(C.TEST_DATA, features)
    test_scores = model.predict_proba(X_test)[:, 1]
    test_default, _, test_default_report, test_default_matrix = evaluate_predictions(
        y_test, test_scores, 0.5
    )
    print("\nClassification report | XGBoost | test | threshold=0.5")
    print(classification_report(
        y_test,
        (test_scores >= 0.5).astype(np.int8),
        labels=[0, 1], target_names=["Normal", "Attack"], digits=4, zero_division=0,
    ))
    print(f"PR-AUC: {test_default['pr_auc']:.4f} | AUC-ROC: {test_default['roc_auc']:.4f}")
    print(
        f"TP={test_default['tp']:,} FP={test_default['fp']:,} "
        f"FN={test_default['fn']:,} TN={test_default['tn']:,}"
    )
    print_table("Confusion matrix | XGBoost | test | threshold=0.5", test_default_matrix, index=True)
    save_json(
        C.METRICS_DIR / "07_confusion_xgboost_test.json",
        test_default_matrix.to_dict(orient="index"),
    )

    # S07.3 - CHỌN THRESHOLD TRÊN VALIDATION; KHÔNG DÙNG TEST ĐỂ CHỌN.
    precision, recall, thresholds = precision_recall_curve(y_validation, validation_scores)
    eligible = np.flatnonzero(recall[:-1] >= C.MIN_RECALL_TARGET)
    if eligible.size == 0:
        raise ValueError(
            f"Không có threshold đạt recall tối thiểu {C.MIN_RECALL_TARGET:.0%} trên validation."
        )
    best_precision = precision[:-1][eligible].max()
    candidates = eligible[precision[:-1][eligible] == best_precision]
    best_index = int(candidates[np.argmax(thresholds[candidates])])
    selected_threshold = float(thresholds[best_index])
    validation_tuned, _, validation_tuned_report, _ = evaluate_predictions(
        y_validation, validation_scores, selected_threshold
    )
    validation_threshold_info = {
        "threshold": selected_threshold,
        "precision": float(precision[best_index]),
        "recall": float(recall[best_index]),
        "f1": float(
            2 * precision[best_index] * recall[best_index]
            / max(precision[best_index] + recall[best_index], 1e-15)
        ),
        "minimum_recall_target": C.MIN_RECALL_TARGET,
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
    figure.savefig(C.FIGURES_DIR / "07_xgboost_validation_pr_curve.png", dpi=150)
    plt.close(figure)

    test_tuned, test_tuned_predictions, test_tuned_report, test_tuned_matrix = evaluate_predictions(
        y_test, test_scores, selected_threshold
    )
    tuned_report_text = (
        f"Classification report | XGBoost | threshold={selected_threshold:.8f}\n"
        f"{classification_report(y_test, test_tuned_predictions, labels=[0, 1], target_names=['Normal', 'Attack'], digits=4, zero_division=0)}\n"
        f"AUC-ROC: {test_tuned['roc_auc']:.4f}\n"
        f"PR-AUC: {test_tuned['pr_auc']:.4f}\n"
    )
    (C.TABLES_DIR / "07_xgboost_tuned_classification_report.txt").write_text(
        tuned_report_text,
        encoding="utf-8",
    )
    save_json(
        C.METRICS_DIR / "07_confusion_xgboost_test_tuned.json",
        test_tuned_matrix.to_dict(orient="index"),
    )

    selected_model = ThresholdedClassifier(
        estimator=model,
        threshold=selected_threshold,
        model_name="xgboost_full_train",
    )
    joblib.dump(selected_model, C.XGBOOST_MODEL, compress=3)
    selected_info = {
        "selected_model": "xgboost_full_train",
        "threshold": selected_threshold,
        "selection_metric": "validation_precision_with_minimum_recall",
        "minimum_recall_target": C.MIN_RECALL_TARGET,
        "validation_metrics": validation_threshold_info,
        "model_path": str(C.XGBOOST_MODEL),
        "threshold_embedded_in_model": True,
    }
    save_json(C.METRICS_DIR / "07_xgboost_selected_model.json", selected_info)
    save_json(C.METRICS_DIR / "07_xgboost_metrics.json", {
        "model": "XGBoost",
        "model_path": str(C.XGBOOST_MODEL),
        "training_summary": result,
        "threshold_0_5": {
            "validation": {
                "metrics": validation_default,
                "classification_report": validation_default_report,
            },
            "test": {
                "metrics": test_default,
                "classification_report": test_default_report,
            },
        },
        "threshold_tuning": {
            "rule": "maximize_validation_precision_subject_to_minimum_recall",
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

    print("\nS07.3 - KẾT QUẢ TEST VỚI THRESHOLD ĐÃ CHỌN")
    print_table("Threshold validation", pd.DataFrame([validation_threshold_info]))
    print("\n" + tuned_report_text)
    print(
        f"TP={test_tuned['tp']:,} FP={test_tuned['fp']:,} "
        f"FN={test_tuned['fn']:,} TN={test_tuned['tn']:,}"
    )
    print_table("Confusion matrix test | threshold đã chọn", test_tuned_matrix, index=True)
    print(f"Đã lưu model kèm threshold: {C.XGBOOST_MODEL}")
    print(f"Đã lưu metrics: {C.METRICS_DIR / '07_xgboost_metrics.json'}")
    print(f"Tổng thời gian s07: {elapsed_text(time.perf_counter() - started)}")

    del X_validation, y_validation, X_test, y_test, validation_scores, test_scores
    del model, selected_model
    gc.collect()
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
