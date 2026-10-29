import json
import time
from collections import Counter

import pandas as pd

from ids import config as C


REQUIRED_COLUMNS = [
    "index", "Login Timestamp", "User ID", "IP Address", "Country",
    "Browser Name and Version", "Device Type", "ASN", "Region",
    "OS Name and Version", "Login Successful",
    "Round-Trip Time [ms]", C.RAW_LABEL,
]


def main():
    C.ensure_dirs()
    if not C.RAW_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy dataset: {C.RAW_FILE}")

    columns = [
        column for column in pd.read_csv(C.RAW_FILE, nrows=0).columns.tolist()
        if column != "Is Account Takeover"
    ]
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(columns))
    if missing_columns:
        raise ValueError(f"Dataset thiếu các cột bắt buộc: {missing_columns}")

    label_counts = Counter()
    missing_counts = Counter()
    timestamp_min, timestamp_max = None, None
    rows = 0
    started = time.perf_counter()

    for chunk in pd.read_csv(
        C.RAW_FILE,
        usecols=REQUIRED_COLUMNS,
        chunksize=C.CHUNK_SIZE,
        low_memory=False,
    ):
        labels = C.parse_label(chunk[C.RAW_LABEL])
        label_counts.update({0: int((labels == 0).sum()), 1: int((labels == 1).sum())})
        for column in REQUIRED_COLUMNS:
            missing_counts[column] += int(chunk[column].isna().sum())

        timestamps = pd.to_datetime(chunk["Login Timestamp"], errors="coerce", utc=True)
        valid_timestamps = timestamps.dropna()
        if not valid_timestamps.empty:
            current_min, current_max = valid_timestamps.min(), valid_timestamps.max()
            timestamp_min = current_min if timestamp_min is None else min(timestamp_min, current_min)
            timestamp_max = current_max if timestamp_max is None else max(timestamp_max, current_max)

        rows += len(chunk)
        print(
            f"Đã đọc {rows:,} dòng  | "
            f"{C.elapsed_text(time.perf_counter() - started)}",
            flush=True,
        )

    normal_count = int(label_counts[0])
    ato_count = int(label_counts[1])
    summary = {
        "dataset": str(C.RAW_FILE),
        "rows": rows,
        "columns": columns,
        "target": C.RAW_LABEL,
        "label_counts": {"not_attack_ip_0": normal_count, "attack_ip_1": ato_count},
        "attack_ip_rate": ato_count / rows if rows else 0.0,
        "timestamp_min_utc": timestamp_min.isoformat() if timestamp_min is not None else None,
        "timestamp_max_utc": timestamp_max.isoformat() if timestamp_max is not None else None,
        "missing_values_in_required_columns": dict(missing_counts),
        "notes": [
            "Is Attack IP là nhãn đối chiếu; không được đưa vào danh sách feature.",
            "Is Account Takeover không được sử dụng trong pipeline này.",
        ],
    }
    output = C.TABLE_DIR / "01_eda_summary.json"
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"EDA hoàn tất: {rows:,} dòng, {ato_count:,} mẫu có Is Attack IP=1.")
    print(f"Đã lưu: {output}")


if __name__ == "__main__":
    main()
