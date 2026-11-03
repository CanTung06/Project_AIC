import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ids import config as C


def main() -> int:
    C.ensure_dirs()
    if not C.FILTERED_DATA.is_file():
        raise FileNotFoundError(f"Chưa có {C.FILTERED_DATA}; hãy chạy s02 trước.")
    started = time.perf_counter()
    frame = pd.read_csv(C.FILTERED_DATA, encoding="utf-8-sig", low_memory=False)
    frame.columns = [str(c).strip() for c in frame.columns]
    if C.TARGET_COLUMN not in frame:
        raise ValueError(f"Thiếu cột nhãn {C.TARGET_COLUMN}.")
    frame = frame.reset_index(drop=True)

    feature_columns = [
        c for c in frame.columns if c != C.TARGET_COLUMN and c not in C.METADATA_COLUMNS
    ]
    if not feature_columns:
        raise ValueError("Không tìm thấy feature.")
    non_numeric = [
        column for column in feature_columns
        if not pd.api.types.is_numeric_dtype(frame[column].dtype)
    ]
    if non_numeric:
        raise TypeError(f"s02 phải xuất feature dạng số; gặp cột chưa mã hóa: {non_numeric[:10]}")
    for column in feature_columns:
        values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        frame[column] = values.fillna(0).astype(np.float32)
    frame[C.TARGET_COLUMN] = pd.to_numeric(frame[C.TARGET_COLUMN], errors="raise").astype(np.int8)
    if not frame[C.TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError("Nhãn is_anomaly chỉ được chứa 0 hoặc 1.")

    row_count = len(frame)
    # Chia ngẫu nhiên có phân tầng theo nhãn để mỗi tập có cùng tỷ lệ lớp.
    # Vì các nguồn có mốc thời gian rất khác nhau (synthetic 2023, KREONET 2026),
    # chia theo thời gian sẽ đẩy gần như toàn bộ KREONET sang validation/test.
    labels = frame[C.TARGET_COLUMN]
    train_frame, remaining_frame = train_test_split(
        frame,
        test_size=C.VALIDATION_FRACTION + C.TEST_FRACTION,
        random_state=C.SEED,
        stratify=labels,
    )
    remaining_labels = remaining_frame[C.TARGET_COLUMN]
    validation_relative_fraction = C.VALIDATION_FRACTION / (
        C.VALIDATION_FRACTION + C.TEST_FRACTION
    )
    validation_frame, test_frame = train_test_split(
        remaining_frame,
        test_size=1.0 - validation_relative_fraction,
        random_state=C.SEED,
        stratify=remaining_labels,
    )
    splits = {
        "train": (train_frame, C.TRAIN_DATA),
        "validation": (validation_frame, C.VALIDATION_DATA),
        "test": (test_frame, C.TEST_DATA),
    }
    summary_rows = []
    class_counts = {}
    for split_name, (part, output_path) in splits.items():
        if part.empty:
            raise ValueError(f"Tập {split_name} rỗng.")
        temp_path = output_path.with_name(output_path.stem + ".tmp.csv")
        part.to_csv(temp_path, index=False, encoding="utf-8-sig")
        temp_path.replace(output_path)
        counts = part[C.TARGET_COLUMN].value_counts().to_dict()
        normal_count, anomaly_count = int(counts.get(0, 0)), int(counts.get(1, 0))
        class_counts[split_name] = {"0": normal_count, "1": anomaly_count}
        summary_rows.append({
            "split": split_name,
            "rows": len(part),
            "normal_count": normal_count,
            "anomaly_count": anomaly_count,
            "anomaly_percent": 100 * anomaly_count / len(part),
        })

    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(C.TABLES_DIR / "03_split_summary.csv", index=False, encoding="utf-8-sig")
    schema = {
        "feature_columns": feature_columns,
        "metadata_columns": C.METADATA_COLUMNS,
        "target_column": C.TARGET_COLUMN,
        "split_strategy": "stratified_random_split_by_target",
        "random_seed": C.SEED,
        "split_fractions": {
            "train": C.TRAIN_FRACTION,
            "validation": C.VALIDATION_FRACTION,
            "test": C.TEST_FRACTION,
        },
    }
    (C.TABLES_DIR / "03_feature_schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metrics = {
        "input_rows": row_count,
        "split_counts": class_counts,
        "split_strategy": "stratified_random_split_by_target",
        "random_seed": C.SEED,
        "split_fractions": schema["split_fractions"],
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "target_column": C.TARGET_COLUMN,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    (C.METRICS_DIR / "03_split_summary.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Chia dữ liệu ngẫu nhiên có phân tầng theo nhãn (70%/15%/15%):")
    print(summary_frame.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(f"Đã lưu train/validation/test trong {C.PROCESSED_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

