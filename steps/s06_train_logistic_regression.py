"""Bước 06: Huấn luyện, chọn threshold và đánh giá Logistic Regression."""

import gc
import sys
import time
from pathlib import Path

import joblib
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ids import config as C
from ids.common import calculate_metrics, confusion_table, elapsed_text, load_xy, save_json
from ids.common import signed_log_transform
from ids.thresholded import ThresholdedClassifier


MIN_RECALL_TARGET = C.MIN_RECALL_TARGET


def build_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("signed_log", FunctionTransformer(
            signed_log_transform,
            validate=False,
            feature_names_out="one-to-one",
        )),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            tol=0.0001,
            class_weight=None,
            random_state=C.SEED,
        )),
    ])


def choose_threshold(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    eligible = (recall[:-1] >= MIN_RECALL_TARGET).nonzero()[0]
    if eligible.size == 0:
        raise ValueError(
            f"Không có threshold nào đạt recall tối thiểu {MIN_RECALL_TARGET:.0%} trên validation."
        )
    eligible_precision = precision[:-1][eligible]
    best_precision = eligible_precision.max()
    candidates = eligible[eligible_precision == best_precision]
    best_index = int(candidates[thresholds[candidates].argmax()])
    return float(thresholds[best_index]), {
        "precision": float(precision[best_index]),
        "recall": float(recall[best_index]),
        "f1": float(
            2 * precision[best_index] * recall[best_index]
            / max(precision[best_index] + recall[best_index], 1e-15)
        ),
    }


def main() -> int:
    C.ensure_dirs()
    required = [C.UNDERSAMPLED_TRAIN_DATA, C.VALIDATION_DATA, C.TEST_DATA]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Thiếu dữ liệu đầu vào; hãy chạy s03_split_data.py và s04_imbalance.py trước: "
            + ", ".join(map(str, missing))
        )

    started = time.perf_counter()
    print("Đang nạp tập train undersampling...")
    X_train, y_train, feature_names = load_xy(C.UNDERSAMPLED_TRAIN_DATA)
    print(
        f"Train Logistic Regression: {len(y_train):,} dòng "
        f"(Normal={(y_train == 0).sum():,}, Attack={(y_train == 1).sum():,})"
    )
    training_rows = int(len(y_train))
    training_support_0 = int((y_train == 0).sum())
    training_support_1 = int((y_train == 1).sum())

    model = build_model()
    fit_started = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - fit_started
    print(f"Huấn luyện xong | {elapsed_text(fit_seconds)}")
    del X_train, y_train
    gc.collect()

    print("Đang chọn threshold trên validation...")
    X_validation, y_validation, _ = load_xy(C.VALIDATION_DATA, feature_names)
    validation_scores = model.predict_proba(X_validation)[:, 1]
    threshold, validation_threshold_metrics = choose_threshold(
        y_validation.to_numpy(), validation_scores
    )
    validation_threshold_metrics["threshold"] = threshold
    validation_threshold_metrics["pr_auc"] = float(
        average_precision_score(y_validation, validation_scores)
    )
    del X_validation, y_validation, validation_scores
    gc.collect()

    print("Đang đánh giá trên test...")
    X_test, y_test, _ = load_xy(C.TEST_DATA, feature_names)
    test_scores = model.predict_proba(X_test)[:, 1]
    test_metrics = calculate_metrics(y_test, test_scores, threshold)
    test_predictions = (test_scores >= threshold).astype("int8")
    report_dict = classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=["Normal", "Attack"],
        output_dict=True,
        zero_division=0,
    )
    report_text = (
        f"Classification report | Logistic Regression | threshold={threshold:.8f}\n"
        f"{classification_report(y_test, test_predictions, labels=[0, 1], target_names=['Normal', 'Attack'], digits=4, zero_division=0)}\n"
        f"AUC-ROC: {test_metrics['roc_auc']:.4f}\n"
        f"PR-AUC: {test_metrics['pr_auc']:.4f}\n"
    )

    final_model = ThresholdedClassifier(
        estimator=model,
        threshold=threshold,
        model_name="logistic_regression_undersampling",
    )
    joblib.dump(final_model, C.LOGISTIC_REGRESSION_MODEL, compress=3)
    report_path = C.TABLES_DIR / "06_logistic_regression_classification_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    save_json(
        C.METRICS_DIR / "06_logistic_regression_confusion_test.json",
        confusion_table(y_test, test_scores, threshold).to_dict(orient="index"),
    )

    result = {
        "model": "LogisticRegression",
        "model_path": str(C.LOGISTIC_REGRESSION_MODEL),
        "training_rows": training_rows,
        "training_support_0": training_support_0,
        "training_support_1": training_support_1,
        "feature_count": len(feature_names),
        "parameters": {
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 1000,
            "tol": 0.0001,
            "class_weight": None,
            "training_data": "undersampled_train.csv, normal:attack = 3:1",
            "feature_transform": "signed_log_then_standard_scaler",
        },
        "fit_time": elapsed_text(fit_seconds),
        "minimum_recall_target": MIN_RECALL_TARGET,
        "validation_at_selected_threshold": validation_threshold_metrics,
        "test_metrics": test_metrics,
        "test_classification_report": report_dict,
        "classification_report_path": str(report_path),
    }
    save_json(C.METRICS_DIR / "06_logistic_regression_metrics.json", result)

    print("\n" + report_text)
    print(f"Đã lưu model: {C.LOGISTIC_REGRESSION_MODEL}")
    print(f"Đã lưu metrics: {C.METRICS_DIR / '06_logistic_regression_metrics.json'}")
    print(f"Tổng thời gian: {elapsed_text(time.perf_counter() - started)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
