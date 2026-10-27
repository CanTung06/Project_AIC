"""Đánh giá đúng một lần trên test sau khi model và ngưỡng đã khóa."""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ids import config as C
from ids.metrics import binary_metrics


def main() -> None:
    if not C.MODEL_FILE.exists():
        raise FileNotFoundError("Chưa có model; chạy steps/s03_train_xgboost.py trước.")
    if not C.TEST_FILE.exists():
        raise FileNotFoundError(f"Thiếu tập test: {C.TEST_FILE}")

    bundle = joblib.load(C.MODEL_FILE)
    model = bundle["model"]
    features = bundle["feature_columns"]
    threshold = float(bundle["threshold"])
    y_parts, score_parts = [], []
    rows = 0
    started = time.perf_counter()
    for frame in pd.read_csv(C.TEST_FILE, chunksize=C.CSV_CHUNK_SIZE, dtype=np.float32):
        y_parts.append(frame[C.TARGET].to_numpy(dtype=np.uint8, copy=True))
        matrix = frame[features].to_numpy(dtype=np.float32, copy=False)
        score_parts.append(model.predict_proba(matrix)[:, 1].astype(np.float32))
        rows += len(frame)
        print(f"Test: đã chấm {rows:,} dòng", flush=True)

    elapsed = time.perf_counter() - started
    y = np.concatenate(y_parts)
    scores = np.concatenate(score_parts)
    metrics = binary_metrics(y, scores, threshold)
    metrics.update({
        "rows": int(len(y)),
        "positive_rows": int(y.sum()),
        "negative_rows": int(len(y) - y.sum()),
        "inference_seconds": elapsed,
        "inference_rows_per_second": len(y) / elapsed if elapsed else None,
        "model_size_bytes": C.MODEL_FILE.stat().st_size,
        "model_file": C.MODEL_FILE.name,
        "test_used_for_threshold_selection": False,
        "validation_target_met": bool(
            bundle["selected_validation_metrics"].get("validation_target_met", False)
        ),
        "test_target_met": bool(
            metrics["precision"] >= C.PRECISION_TARGET
            and metrics["recall"] >= C.RECALL_TARGET
            and metrics["f1"] >= C.F1_TARGET
        ),
    })
    C.ensure_dirs()
    C.TEST_METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nKết quả test độc lập")
    for name in (
        "threshold", "tp", "fp", "fn", "tn", "precision", "recall", "f1",
        "fpr", "fp_per_million_negatives", "pr_auc", "roc_auc",
        "inference_seconds", "model_size_bytes",
        "brier_score",
    ):
        print(f"{name}: {metrics[name]}")
    print(f"Đã lưu JSON: {C.TEST_METRICS_FILE}")


if __name__ == "__main__":
    main()
