import sys
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ids import config as C
from ids.common import calculate_metrics, load_xy, save_json


MODELS = (
    ("Logistic Regression", C.FULL_LOGISTIC_REGRESSION_MODEL),
    ("XGBoost", C.FULL_XGBOOST_MODEL),
    ("Random Forest", C.FULL_RANDOM_FOREST_MODEL),
)
MAX_EXAMPLES_PER_ERROR = 1_000


def main() -> int:
    C.ensure_dirs()
    missing = [path for path in (C.TEST_DATA, *(path for _, path in MODELS)) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Thiếu tập test/model; hãy chạy s03 và s05-s07 trước: "
            + ", ".join(map(str, missing))
        )

    started = time.perf_counter()
    X_test, y_test, feature_names = load_xy(C.TEST_DATA)
    summary_rows = []
    example_frames = []
    for model_name, model_path in MODELS:
        model = joblib.load(model_path)
        threshold = float(getattr(model, "threshold", 0.5))
        scores = model.predict_proba(X_test)[:, 1]
        predictions = (scores >= threshold).astype(np.int8)
        tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
        metrics = calculate_metrics(y_test, scores, threshold)
        summary_rows.append({
            "model": model_name,
            **metrics,
            "model_path": str(model_path),
        })

        error_mask = predictions != y_test.to_numpy()
        error_indices = np.flatnonzero(error_mask)
        if len(error_indices):
            chosen = error_indices[:MAX_EXAMPLES_PER_ERROR]
            examples = X_test.iloc[chosen].copy()
            examples.insert(0, "row_index", X_test.index.to_numpy()[chosen])
            examples.insert(1, "model", model_name)
            examples.insert(2, "true_label", y_test.to_numpy()[chosen])
            examples.insert(3, "predicted_label", predictions[chosen])
            examples.insert(4, "anomaly_score", scores[chosen])
            examples.insert(
                5,
                "error_type",
                np.where(y_test.to_numpy()[chosen] == 0, "false_positive", "false_negative"),
            )
            example_frames.append(examples)

    summary = pd.DataFrame(summary_rows)
    summary_path = C.TABLES_DIR / "09_error_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    examples = (
        pd.concat(example_frames, ignore_index=True)
        if example_frames else pd.DataFrame(
            columns=["row_index", "model", "true_label", "predicted_label", "anomaly_score", "error_type", *feature_names]
        )
    )
    examples_path = C.TABLES_DIR / "09_error_examples.csv"
    examples.to_csv(examples_path, index=False, encoding="utf-8-sig")

    figure, axis = plt.subplots(figsize=(8, 4.8))
    positions = np.arange(len(summary))
    width = 0.36
    axis.bar(positions - width / 2, summary["fp"], width, label="FP")
    axis.bar(positions + width / 2, summary["fn"], width, label="FN")
    axis.set_xticks(positions, summary["model"], rotation=12, ha="right")
    axis.set_ylabel("Số mẫu")
    axis.set_title("Lỗi trên tập test theo model")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure_path = C.FIGURES_DIR / "09_test_errors_by_model.png"
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)

    metrics_path = C.METRICS_DIR / "09_error_analysis.json"
    save_json(metrics_path, {
        "evaluation_split": str(C.TEST_DATA),
        "feature_count": len(feature_names),
        "error_examples_saved_per_model": MAX_EXAMPLES_PER_ERROR,
        "models": summary.to_dict(orient="records"),
        "summary_table": str(summary_path),
        "error_examples": str(examples_path),
        "error_figure": str(figure_path),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    })
    print("S09 - PHÂN TÍCH LỖI")
    print(summary[["model", "precision", "recall", "f1", "tp", "fp", "fn", "tn"]]
          .to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Đã lưu bảng lỗi: {summary_path}")
    print(f"Đã lưu mẫu FP/FN: {examples_path}")
    print(f"Đã lưu biểu đồ: {figure_path}")
    print(f"Đã lưu metrics: {metrics_path}")
    print(f"Thời gian: {time.perf_counter() - started:.1f} giây")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
