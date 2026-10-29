"""Bước 11 - Mô phỏng chấm điểm tuần tự theo chunk trên tập test."""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ids import config as C
from ids.common import elapsed_text, load_feature_names


def main() -> int:
    C.ensure_dirs()
    if not C.SELECTED_MODEL_INFO.exists():
        raise FileNotFoundError("Hãy chạy s07_tuning.py trước.")
    with C.SELECTED_MODEL_INFO.open("r", encoding="utf-8") as source:
        selected = json.load(source)
    model_path = C.WEIGHTED_MODEL if selected["selected_model"] == "class_weight" else C.UNDERSAMPLED_MODEL
    model = joblib.load(model_path)
    threshold = float(selected["threshold"])
    feature_names = load_feature_names(C.TEST_DATA)
    columns = feature_names + [C.TARGET_COLUMN, "attack_type"]

    output_path = C.TABLES_DIR / "11_stream_predictions.csv"
    temporary_path = output_path.with_suffix(".tmp.csv")
    temporary_path.unlink(missing_ok=True)
    total = 0
    alerts = 0
    tp = fp = fn = tn = 0
    wrote_header = False
    started = time.perf_counter()

    for chunk in pd.read_csv(
        C.TEST_DATA,
        usecols=columns,
        chunksize=C.CHUNK_SIZE,
        dtype={column: np.float32 for column in feature_names},
        encoding="utf-8-sig",
    ):
        X = chunk[feature_names]
        y_true = chunk[C.TARGET_COLUMN].astype(np.int8).to_numpy()
        scores = model.predict_proba(X)[:, 1]
        predictions = (scores >= threshold).astype(np.int8)
        tn_chunk, fp_chunk, fn_chunk, tp_chunk = confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).ravel()
        tn += int(tn_chunk)
        fp += int(fp_chunk)
        fn += int(fn_chunk)
        tp += int(tp_chunk)
        total += len(chunk)
        alerts += int(predictions.sum())

        result = pd.DataFrame({
            "record_number": np.arange(total - len(chunk) + 1, total + 1),
            "attack_score": scores,
            "predicted_is_attack": predictions,
            "actual_is_attack": y_true,
            "attack_type": chunk["attack_type"].astype("string").to_numpy(),
        })
        result.to_csv(
            temporary_path,
            index=False,
            mode="a" if wrote_header else "w",
            header=not wrote_header,
            encoding="utf-8" if wrote_header else "utf-8-sig",
        )
        wrote_header = True
        print(
            f"Đã chấm {total:,} dòng | phát hiện {alerts:,} cảnh báo | "
            f"{elapsed_text(time.perf_counter() - started)}",
            end="\r",
            flush=True,
        )

    if not wrote_header:
        raise RuntimeError("Tập test rỗng; không tạo file dự đoán.")
    temporary_path.replace(output_path)
    summary = {
        "selected_model": selected["selected_model"],
        "threshold": threshold,
        "test_records": total,
        "alerts": alerts,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "prediction_file": str(output_path),
        "elapsed": elapsed_text(time.perf_counter() - started),
    }
    with (C.METRICS_DIR / "11_stream_summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, ensure_ascii=False, indent=2)

    print("\n\nKết quả mô phỏng chấm điểm tuần tự:")
    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"\nĐã lưu dự đoán: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
