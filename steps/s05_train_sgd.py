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
    print_model_evaluation,
    save_json,
    signed_log_transform,
)
from ids.thresholded import ThresholdedClassifier


def tune_threshold(model, total_started: float, strategy: str, model_path: Path) -> None:
    """Chọn threshold trên validation rồi đánh giá trên test."""
    print("\n" + "=" * 72)
    print("S05.2 - TUNING THRESHOLD TRÊN VALIDATION")
    print("=" * 72)

    model_name = f"sgd_{strategy}"
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
        "pr_auc": float(average_precision_score(y_val, validation_scores_raw)),
    }

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
    figure.savefig(C.FIGURES_DIR / f"05_sgd_{strategy}_validation_pr_curve.png", dpi=150)
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
        f"Classification report | SGDClassifier ({strategy}) | threshold={threshold:.8f}\n"
        f"{test_report_text}\n"
        f"AUC-ROC: {test_metrics['roc_auc']:.4f}\n"
        f"PR-AUC: {test_metrics['pr_auc']:.4f}\n"
    )
    report_path = C.TABLES_DIR / f"05_sgd_{strategy}_tuned_classification_report.txt"
    report_path.write_text(report_text, encoding="utf-8")

    test_row = {"model": model_name, **test_metrics}
    matrix = confusion_table(y_test, test_scores, threshold)
    save_json(
        C.METRICS_DIR / f"05_confusion_sgd_{strategy}_test_tuned.json",
        matrix.to_dict(orient="index"),
    )

    thresholded_model = ThresholdedClassifier(
        estimator=model,
        threshold=threshold,
        model_name=model_name,
    )
    joblib.dump(thresholded_model, model_path, compress=3)
    metrics_path = C.METRICS_DIR / f"05_sgd_{strategy}_threshold_tuning.json"
    save_json(metrics_path, {
        "model": "SGDClassifier",
        "strategy": strategy,
        "model_path": str(model_path),
        "selection_rule": "maximize_validation_precision_subject_to_minimum_recall",
        "minimum_recall_target": C.MIN_RECALL_TARGET,
        "validation_threshold": validation_metrics,
        "test_at_validation_threshold": [test_row],
        "test_classification_report": test_report_dict,
        "threshold_embedded_in_model": True,
    })

    print_model_evaluation(
        f"SGDClassifier ({strategy})",
        validation_metrics,
        report_text,
        test_metrics,
        matrix,
        model_path,
        metrics_path,
        elapsed_text(time.perf_counter() - total_started),
    )


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
    train_sets = (
        ("undersampling", C.UNDERSAMPLED_TRAIN_DATA, C.UNDERSAMPLED_MODEL),
        ("full_data", C.TRAIN_DATA, C.FULL_SGD_MODEL),
    )
    missing = [str(path) for _, path, _ in train_sets if not path.exists()]
    if missing or not C.VALIDATION_DATA.exists() or not C.TEST_DATA.exists():
        raise FileNotFoundError("Hãy chạy s03_split_data.py và s04_imbalance.py trước.")
    # BƯỚC S05 - HUẤN LUYỆN SGDClassifier
    print("\n" + "=" * 72)
    print("BƯỚC S05 - HUẤN LUYỆN SGDClassifier")
    print("=" * 72)
    results = []
    for strategy, train_path, model_path in train_sets:
        print(f"\nĐang đọc tập train ({strategy}): {train_path}")
        X_train, y_train, feature_names = load_xy(train_path)
        print(f"Train: {len(y_train):,} dòng, {len(feature_names):,} feature")

        model = build_pipeline()
        started = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started
        classifier = model.named_steps["classifier"]
        joblib.dump(model, model_path, compress=3)
        result = {
            "strategy": strategy,
            "training_rows": int(len(y_train)),
            "feature_count": len(feature_names),
            "fit_seconds": round(fit_seconds, 2),
            "fit_time": elapsed_text(fit_seconds),
            "epochs_used": int(classifier.n_iter_),
            "feature_transform": "signed_log_then_standard_scaler",
            "model_path": str(model_path),
        }
        results.append(result)
        print(
            f"Đã lưu model: {model_path.name} | {fit_seconds:.1f} giây | "
            f"{classifier.n_iter_} epoch"
        )
        del X_train, y_train
        gc.collect()
        tune_threshold(model, started, strategy, model_path)
        del model
        gc.collect()

    result_table = pd.DataFrame(results)
    result_table.to_csv(C.TABLES_DIR / "05_training_summary.csv", index=False, encoding="utf-8-sig")
    with (C.METRICS_DIR / "05_training_summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(results, output_file, ensure_ascii=False, indent=2)

    print("\nTóm tắt huấn luyện:")
    print(result_table.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
