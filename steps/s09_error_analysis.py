"""Bước 09 - Phân tích false positive/false negative trên test."""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ids import config as C
from ids.common import elapsed_text, load_feature_names


def main() -> int:
    C.ensure_dirs()
    started = time.perf_counter()
    with C.SELECTED_MODEL_INFO.open("r", encoding="utf-8") as source:
        selected = json.load(source)
    model_path = C.WEIGHTED_MODEL if selected["selected_model"] == "class_weight" else C.UNDERSAMPLED_MODEL
    model = joblib.load(model_path)
    feature_names = load_feature_names(C.TEST_DATA)
    columns = feature_names + [C.TARGET_COLUMN, "attack_type"]
    data = pd.read_csv(
        C.TEST_DATA,
        usecols=columns,
        dtype={column: np.float32 for column in feature_names},
        encoding="utf-8-sig",
        low_memory=False,
    )
    X = data[feature_names]
    actual = data[C.TARGET_COLUMN].astype(np.int8).to_numpy()
    scores = model.predict_proba(X)[:, 1]
    predicted = (scores >= float(selected["threshold"])).astype(np.int8)

    false_positive_indices = np.flatnonzero((actual == 0) & (predicted == 1))
    false_negative_indices = np.flatnonzero((actual == 1) & (predicted == 0))
    rows = []
    summaries = []
    for error_type, indices in [
        ("false_positive", false_positive_indices),
        ("false_negative", false_negative_indices),
    ]:
        if error_type == "false_positive":
            indices = indices[np.argsort(scores[indices])[::-1]]
        else:
            indices = indices[np.argsort(scores[indices])]
        saved_indices = indices[:C.MAX_ERROR_EXAMPLES_PER_TYPE]
        for index in saved_indices:
            rows.append({
                "row_index": int(index),
                "error_type": error_type,
                "attack_type": str(data.iloc[index]["attack_type"]),
                "actual_is_attack": int(actual[index]),
                "predicted_is_attack": int(predicted[index]),
                "attack_score": float(scores[index]),
                "threshold": float(selected["threshold"]),
            })
        error_types = data.iloc[indices]["attack_type"].value_counts()
        for attack_type, count in error_types.items():
            summaries.append({
                "error_type": error_type,
                "attack_type": attack_type,
                "error_count": int(count),
                "examples_saved": int(min(count, C.MAX_ERROR_EXAMPLES_PER_TYPE)),
            })

    error_columns = [
        "row_index", "error_type", "attack_type", "actual_is_attack",
        "predicted_is_attack", "attack_score", "threshold",
    ]
    pd.DataFrame(rows, columns=error_columns).to_csv(
        C.TABLES_DIR / "09_error_examples.csv", index=False, encoding="utf-8-sig"
    )
    summary_table = pd.DataFrame(
        summaries,
        columns=["error_type", "attack_type", "error_count", "examples_saved"],
    )
    if not summary_table.empty:
        summary_table = summary_table.sort_values(
            ["error_type", "error_count"], ascending=[True, False]
        )
    summary_table.to_csv(
        C.TABLES_DIR / "09_errors_by_attack_type.csv", index=False, encoding="utf-8-sig"
    )
    report = {
        "selected_model": selected["selected_model"],
        "threshold": selected["threshold"],
        "false_positives": int(len(false_positive_indices)),
        "false_negatives": int(len(false_negative_indices)),
        "examples_saved_per_error_type_max": C.MAX_ERROR_EXAMPLES_PER_TYPE,
    }
    with (C.METRICS_DIR / "09_error_analysis.json").open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, ensure_ascii=False, indent=2)

    print(f"False Positive: {len(false_positive_indices):,}")
    print(f"False Negative: {len(false_negative_indices):,}")
    print("\nSai số theo loại nhãn thật:")
    print(summary_table.to_string(index=False))
    print(f"\nTối đa {C.MAX_ERROR_EXAMPLES_PER_TYPE:,} ví dụ mỗi loại lỗi được lưu.")
    print(f"Thời gian: {elapsed_text(time.perf_counter() - started)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
