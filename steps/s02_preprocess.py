import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ids import config as C

CATEGORICAL_COLUMNS = ["Login Status", "IP Address", "Device Type", "Location"]
NUMERIC_COLUMNS = ["Session Duration", "Failed Attempts", "Behavioral Score"]
REQUIRED_COLUMNS = ["User ID", "Timestamp", *CATEGORICAL_COLUMNS, *NUMERIC_COLUMNS, C.SOURCE_LABEL_COLUMN]
OTHER = "__OTHER__"
MISSING = "__MISSING__"
UNKNOWN_VALUES = {"unknown", "unk", "n/a", "na", "none", "null"}


def normalize_header(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().casefold())


def canonicalize_columns(frame):
    expected = {normalize_header(name): name for name in REQUIRED_COLUMNS}
    frame = frame.rename(columns={c: expected[normalize_header(c)] for c in frame.columns
                                  if normalize_header(c) in expected})
    duplicates = frame.columns[frame.columns.duplicated()].tolist()
    if duplicates:
        raise ValueError(f"Các cột bị trùng sau khi chuẩn hóa: {duplicates}")
    return frame


def build_category_vocabulary(frame):
    result = {}
    for column in CATEGORICAL_COLUMNS:
        values = frame[column].astype("string").str.strip()
        valid = values.dropna().loc[lambda x: x.ne("")]
        valid = valid.loc[~valid.str.casefold().isin(UNKNOWN_VALUES)]
        observed = sorted(valid.unique().tolist())
        result[column] = [*observed, MISSING, OTHER]
    return result


def engineer_auth_features(frame, category_vocabulary):
    data = frame.copy()
    # CSV có thể trộn timestamp có/không có phần micro giây; format="mixed"
    # tránh pandas suy định dạng từ dòng đầu rồi đánh dấu các định dạng khác là lỗi.
    data["_event_time"] = pd.to_datetime(
        data["Timestamp"], format="mixed", errors="coerce", utc=True
    )
    if data["_event_time"].isna().any():
        raise ValueError(f"Có {int(data['_event_time'].isna().sum()):,} Timestamp không hợp lệ.")

    for column in CATEGORICAL_COLUMNS:
        values = data[column].astype("string").str.strip()
        values = values.mask(
            values.isna() | values.eq("") | values.str.casefold().isin(UNKNOWN_VALUES),
            MISSING,
        )
        canonical_values = {
            str(category).casefold(): category
            for category in category_vocabulary[column]
            if category not in (MISSING, OTHER)
        }
        values = values.map(lambda value: canonical_values.get(str(value).casefold(), value))
        data[column] = values.where(values.isin(category_vocabulary[column]), OTHER)
    for column in NUMERIC_COLUMNS:
        values = pd.to_numeric(data[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        data[column] = values.fillna(0).astype(np.float32)

    data = data.sort_values("_event_time", kind="stable").reset_index(drop=True)
    timestamp = data["_event_time"]
    hour = timestamp.dt.hour + timestamp.dt.minute / 60 + timestamp.dt.second / 3600
    weekday = timestamp.dt.dayofweek
    data["login_hour_sin"] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
    data["login_hour_cos"] = np.cos(2 * np.pi * hour / 24).astype(np.float32)
    data["login_weekday_sin"] = np.sin(2 * np.pi * weekday / 7).astype(np.float32)
    data["login_weekday_cos"] = np.cos(2 * np.pi * weekday / 7).astype(np.float32)

    # Tạo ma trận one-hot riêng rồi ghép một lần. Chèn từng cột vào DataFrame
    # tạo fragmentation và làm chậm đáng kể khi IP có nhiều giá trị duy nhất.
    one_hot_frames = []
    for column in CATEGORICAL_COLUMNS:
        categories = [
            category for category in category_vocabulary[column]
            if category not in (MISSING, OTHER)
        ]
        encoded = pd.get_dummies(
            pd.Categorical(data[column], categories=categories),
            dtype=np.int8,
        )
        encoded.columns = [f"{column}__{category}" for category in categories]
        one_hot_frames.append(encoded)
    if one_hot_frames:
        data = pd.concat([data, *one_hot_frames], axis=1, copy=False)

    return data.drop(columns=["_event_time"])


def main() -> int:
    C.ensure_dirs()
    files = [C.AUTH_SOURCE_FILE]
    if not C.AUTH_SOURCE_FILE.is_file():
        raise FileNotFoundError(f"Không tìm thấy dataset: {C.AUTH_SOURCE_FILE}")
    started = time.perf_counter()
    source_path = C.AUTH_SOURCE_FILE
    print(f"Đang đọc {source_path.name}", flush=True)
    raw = pd.read_csv(source_path, encoding="utf-8-sig", low_memory=False)
    raw.columns = [str(c).strip().lstrip("\ufeff") for c in raw.columns]
    raw = canonicalize_columns(raw)
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"{source_path.name} không đúng log xác thực; thiếu cột: {missing}")
    raw = raw[REQUIRED_COLUMNS].copy()
    total_rows = len(raw)
    labels = pd.to_numeric(raw[C.SOURCE_LABEL_COLUMN], errors="coerce")
    valid = labels.isin([0, 1])
    dropped_labels = int((~valid).sum())
    raw = raw.loc[valid].copy()
    raw[C.TARGET_COLUMN] = labels.loc[valid].astype(np.int8)
    raw.drop(columns=C.SOURCE_LABEL_COLUMN, inplace=True)
    # Các nguồn có thể dùng định dạng timestamp khác nhau (ISO-8601 của KREONET
    # và dạng có dấu cách của synthetic); parse từng giá trị để không loại nhầm dòng.
    timestamps = pd.to_datetime(
        raw["Timestamp"], format="mixed", errors="coerce", utc=True
    )
    invalid_times = int(timestamps.isna().sum())
    raw = raw.loc[timestamps.notna()].copy()

    categories = build_category_vocabulary(raw)
    processed = engineer_auth_features(raw, categories)
    # Dữ liệu đã xử lý chỉ giữ feature số và nhãn. Thứ tự thời gian đã được
    # sắp xếp trong engineer_auth_features trước khi loại ID/thời gian.
    processed.drop(
        columns=[*CATEGORICAL_COLUMNS, *C.SOURCE_METADATA_COLUMNS],
        inplace=True,
    )
    processed[C.TARGET_COLUMN] = pd.to_numeric(
        processed[C.TARGET_COLUMN], errors="raise"
    ).astype(np.int8)
    temp_path = C.FILTERED_DATA.with_name(C.FILTERED_DATA.stem + ".tmp.csv")
    processed.to_csv(temp_path, index=False, encoding="utf-8-sig")
    temp_path.replace(C.FILTERED_DATA)
    features = [c for c in processed.columns if c != C.TARGET_COLUMN]
    non_numeric_features = [
        column for column in features
        if not pd.api.types.is_numeric_dtype(processed[column].dtype)
    ]
    if not pd.api.types.is_numeric_dtype(processed[C.TARGET_COLUMN].dtype):
        non_numeric_features.append(C.TARGET_COLUMN)
    if non_numeric_features:
        raise TypeError(f"Feature chưa được mã hóa thành số: {non_numeric_features[:10]}")
    schema = {
        "task": "user authentication anomaly detection",
        "target_column": C.TARGET_COLUMN,
        "source_label_column": C.SOURCE_LABEL_COLUMN,
        "metadata_columns": [],
        "categorical_columns": CATEGORICAL_COLUMNS,
        "categorical_encoding": "one_hot",
        "excluded_source_columns": [],
        "unknown_category_handling": "missing or unknown values produce all-zero one-hot columns",
        "source_attribute_representation": {
            "User ID": [],
            "Timestamp": [
                "login_hour_sin",
                "login_hour_cos",
                "login_weekday_sin",
                "login_weekday_cos",
            ],
            "Login Status": [column for column in features if column.startswith("Login Status__")],
            "IP Address": [column for column in features if column.startswith("IP Address__")],
            "Device Type": [column for column in features if column.startswith("Device Type__")],
            "Location": [column for column in features if column.startswith("Location__")],
            "numeric_source_attributes": NUMERIC_COLUMNS,
        },
        "category_vocabulary": categories,
        "feature_columns": features,
        "history_features": False,
    }
    C.AUTH_FEATURE_SCHEMA.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    counts = processed[C.TARGET_COLUMN].value_counts().sort_index()
    summary = {
        "source_files": [p.name for p in files],
        "source_rows": total_rows,
        "processed_rows": len(processed),
        "dropped_invalid_labels": dropped_labels,
        "dropped_invalid_timestamps": invalid_times,
        "excluded_source_columns": [],
        "unknown_category_handling": "unknown and missing values create no one-hot feature",
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "feature_count": len(features),
        "feature_columns": features,
        "target_column": C.TARGET_COLUMN,
        "output_file": str(C.FILTERED_DATA),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    (C.METRICS_DIR / "02_preprocess_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Hoàn tất: đã đọc {total_rows:,}, giữ {len(processed):,} dòng.")
    print(f"Bình thường (0): {int(counts.get(0, 0)):,} | Bất thường (1): {int(counts.get(1, 0)):,}")
    print(f"Feature: {len(features)} | Thời gian: {time.perf_counter() - started:.1f} giây")
    print("Đã giữ các feature số Session Duration, Failed Attempts, Behavioral Score; Unknown được xử lý thành giá trị thiếu.")
    print(f"Dataset: {C.FILTERED_DATA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
