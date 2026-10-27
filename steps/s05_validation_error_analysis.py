"""Xếp hạng rủi ro trên validation và báo cáo Precision@K."""
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


TOP_K_VALUES = (10, 50, 100)


def main() -> None:
    if not C.MODEL_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy model: {C.MODEL_FILE}; hãy chạy s03 trước.")
    if not C.VALIDATION_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy validation: {C.VALIDATION_FILE}; hãy chạy s02 trước.")

    bundle = joblib.load(C.MODEL_FILE)
    model = bundle["model"]
    features = bundle["feature_columns"]
    threshold = float(bundle["threshold"])
    max_k = max(TOP_K_VALUES)

    top_rows = None
    rows_seen = positives = tp = fp = 0
    started = time.perf_counter()
    C.ensure_dirs()

    for frame in pd.read_csv(C.VALIDATION_FILE, chunksize=C.CSV_CHUNK_SIZE, dtype=np.float32):
        missing = [column for column in features if column not in frame.columns]
        if missing:
            raise ValueError(f"Validation thiếu feature, ví dụ: {missing[:10]}")

        y = frame[C.TARGET].to_numpy(dtype=np.uint8, copy=True)
        matrix = frame[features].to_numpy(dtype=np.float32, copy=False)
        scores = model.predict_proba(matrix)[:, 1].astype(np.float32, copy=False)
        predicted = scores >= threshold
        positives += int(y.sum())
        tp += int(np.count_nonzero(predicted & (y == 1)))
        fp += int(np.count_nonzero(predicted & (y == 0)))

        local_k = min(max_k, len(scores))
        if local_k:
            local_idx = np.argpartition(scores, len(scores) - local_k)[-local_k:]
            candidate = frame.iloc[local_idx][features].copy().reset_index(drop=True)
            candidate.insert(0, "target", y[local_idx])
            candidate.insert(0, "predicted_at_validation_threshold", predicted[local_idx].astype(np.uint8))
            candidate.insert(0, "risk_score", scores[local_idx])
            candidate.insert(0, "validation_row", rows_seen + local_idx + 1)
            if top_rows is None:
                top_rows = candidate
            else:
                top_rows = pd.concat([top_rows, candidate], ignore_index=True)
            top_rows = top_rows.nlargest(max_k, "risk_score", keep="first").reset_index(drop=True)
        rows_seen += len(frame)
        print(f"Validation: đã chấm {rows_seen:,} dòng | {time.perf_counter() - started:.1f} giây", flush=True)

    if rows_seen == 0:
        raise ValueError("Tập validation rỗng.")

    top_rows = top_rows.sort_values("risk_score", ascending=False, kind="stable").reset_index(drop=True)
    top_rows.insert(0, "rank", np.arange(1, len(top_rows) + 1, dtype=np.int32))

    topk_metrics = {}
    top_labels = top_rows["target"].to_numpy(dtype=np.uint8)
    for requested_k in TOP_K_VALUES:
        actual_k = min(requested_k, rows_seen)
        true_positives_at_k = int(top_labels[:actual_k].sum())
        topk_metrics[str(requested_k)] = {
            "rows_reviewed": actual_k,
            "true_takeovers_found": true_positives_at_k,
            "precision_at_k": true_positives_at_k / actual_k if actual_k else 0.0,
            "recall_at_k": true_positives_at_k / positives if positives else 0.0,
        }

    fn = positives - tp
    negatives = rows_seen - positives
    tn = negatives - fp
    metrics = {
        "validation_rows": rows_seen,
        "validation_takeovers": positives,
        "selected_model_threshold": threshold,
        "threshold_confusion_matrix": [[tn, fp], [fn, tp]],
        "threshold_precision": tp / (tp + fp) if tp + fp else 0.0,
        "threshold_recall": tp / positives if positives else 0.0,
        "top_k": topk_metrics,
        "top_risk_csv": str(C.VALIDATION_TOPK_TABLE_FILE),
        "source_row_mapping": (
            "validation_row là số thứ tự trong validation.csv. Bản preprocess hiện tại không giữ index "
            "nguồn hoặc giá trị category gốc để ánh xạ trực tiếp về sự kiện raw."
        ),
        "test_used": False,
        "scoring_seconds": time.perf_counter() - started,
    }

    top_rows.to_csv(C.VALIDATION_TOPK_TABLE_FILE, index=False, encoding="utf-8-sig")
    C.VALIDATION_TOPK_METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nKết quả xếp hạng validation")
    for k, result in topk_metrics.items():
        print(
            f"Top {k}: takeover={result['true_takeovers_found']}, "
            f"Precision@{k}={result['precision_at_k']:.4f}, "
            f"Recall@{k}={result['recall_at_k']:.4f}"
        )
    print(f"Đã lưu danh sách top rủi ro: {C.VALIDATION_TOPK_TABLE_FILE}")
    print(f"Đã lưu metric: {C.VALIDATION_TOPK_METRICS_FILE}")
    print("Lưu ý: validation_row là số thứ tự trong validation.csv, không phải index của dữ liệu raw.")


if __name__ == "__main__":
    main()
