"""EDA theo chunk cho dataset RBA; không nạp toàn bộ file vào RAM."""
from collections import Counter
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ids import config as C

CHUNK_SIZE = 500_000
EXPECTED_COLUMNS = [
    "index", "Login Timestamp", "User ID", "Round-Trip Time [ms]", "IP Address",
    "Country", "Region", "City", "ASN", "User Agent String",
    "Browser Name and Version", "OS Name and Version", "Device Type",
    "Login Successful", "Is Attack IP", C.TARGET,
]
LABEL_COLUMNS = ["Login Successful", "Is Attack IP", C.TARGET]
CATEGORY_COLUMNS = [
    "IP Address", "Country", "Region", "City", "ASN", "User Agent String",
    "Browser Name and Version", "OS Name and Version", "Device Type",
]
TOP_N = 30


def elapsed_text(seconds: float) -> str:
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def main() -> None:
    if not C.RAW_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy dataset: {C.RAW_FILE}")
    C.ensure_dirs()
    header = pd.read_csv(C.RAW_FILE, nrows=0).columns.tolist()
    if header != EXPECTED_COLUMNS:
        raise ValueError(
            "Schema nguồn không đúng như đã xác nhận.\n"
            f"Cột mong đợi: {EXPECTED_COLUMNS}\nCột thực tế: {header}"
        )

    missing = Counter()
    labels = {column: Counter() for column in LABEL_COLUMNS}
    month_takeovers = Counter()
    category_candidates = {column: Counter() for column in CATEGORY_COLUMNS}
    total_rows = invalid_timestamps = invalid_indices = duplicate_indices = 0
    timestamp_min = timestamp_max = None
    rtt_missing = rtt_count = 0
    rtt_min = rtt_max = None
    rtt_sum = rtt_sum_sq = 0.0
    seen_indices = np.zeros(0, dtype=np.bool_)
    started = time.perf_counter()

    for chunk in pd.read_csv(
        C.RAW_FILE, usecols=EXPECTED_COLUMNS, dtype="string",
        chunksize=CHUNK_SIZE, low_memory=False,
    ):
        total_rows += len(chunk)
        for column in EXPECTED_COLUMNS:
            values = chunk[column].str.strip()
            missing[column] += int((values.isna() | values.eq("") | values.eq("-")).sum())

        raw_index = pd.to_numeric(chunk["index"], errors="coerce")
        valid = raw_index.notna() & raw_index.ge(0) & raw_index.mod(1).eq(0)
        invalid_indices += int((~valid).sum())
        if valid.any():
            indices = raw_index[valid].to_numpy(dtype=np.int64)
            unique, counts = np.unique(indices, return_counts=True)
            duplicate_indices += int(np.maximum(counts - 1, 0).sum())
            in_seen_range = unique < len(seen_indices)
            if in_seen_range.any():
                duplicate_indices += int(seen_indices[unique[in_seen_range]].sum())
            largest = int(unique[-1])
            if largest > 100_000_000:
                raise ValueError("Giá trị index quá lớn; dừng để tránh cấp phát bitmap vượt RAM.")
            if largest >= len(seen_indices):
                new_size = max(largest + 1, max(1024, len(seen_indices) * 2))
                seen_indices = np.pad(seen_indices, (0, new_size - len(seen_indices)))
            seen_indices[unique] = True

        for column in LABEL_COLUMNS:
            normalized = chunk[column].fillna("<missing>").str.strip().str.lower()
            labels[column].update(normalized.value_counts(dropna=False).to_dict())

        timestamps = pd.to_datetime(chunk["Login Timestamp"], errors="coerce")
        invalid_timestamps += int(timestamps.isna().sum())
        valid_timestamps = timestamps.dropna()
        if not valid_timestamps.empty:
            local_min, local_max = valid_timestamps.min(), valid_timestamps.max()
            timestamp_min = local_min if timestamp_min is None else min(timestamp_min, local_min)
            timestamp_max = local_max if timestamp_max is None else max(timestamp_max, local_max)
        takeover = chunk[C.TARGET].str.strip().str.lower().eq("true")
        month = timestamps.dt.strftime("%Y-%m")
        month_takeovers.update(month[takeover & month.notna()].value_counts().to_dict())

        rtt = pd.to_numeric(chunk["Round-Trip Time [ms]"], errors="coerce")
        valid_rtt = rtt.dropna()
        rtt_missing += int(rtt.isna().sum())
        if not valid_rtt.empty:
            values = valid_rtt.to_numpy(dtype=np.float64)
            rtt_count += len(values)
            rtt_sum += float(values.sum())
            rtt_sum_sq += float(np.square(values).sum())
            local_min, local_max = float(values.min()), float(values.max())
            rtt_min = local_min if rtt_min is None else min(rtt_min, local_min)
            rtt_max = local_max if rtt_max is None else max(rtt_max, local_max)

        # Top values are candidates aggregated from each chunk; avoids retaining
        # millions of unique IP/UA strings in memory.
        for column in CATEGORY_COLUMNS:
            category_candidates[column].update(
                chunk[column].fillna("<missing>").value_counts().head(500).to_dict()
            )

        print(
            f"EDA: {total_rows:,} dòng | thời gian {elapsed_text(time.perf_counter()-started)}",
            flush=True,
        )

    if total_rows == 0:
        raise ValueError("Dataset không có dòng dữ liệu.")

    summary = [
        {"metric": "file_size_bytes", "value": C.RAW_FILE.stat().st_size},
        {"metric": "total_rows", "value": total_rows},
        {"metric": "column_count", "value": len(header)},
        {"metric": "duplicate_index_rows", "value": duplicate_indices},
        {"metric": "invalid_index_rows", "value": invalid_indices},
        {"metric": "invalid_timestamps", "value": invalid_timestamps},
        {"metric": "timestamp_min", "value": timestamp_min},
        {"metric": "timestamp_max", "value": timestamp_max},
        {"metric": "takeover_true", "value": labels[C.TARGET].get("true", 0)},
        {"metric": "takeover_false", "value": labels[C.TARGET].get("false", 0)},
    ]
    summary.extend({"metric": f"missing::{column}", "value": missing[column]} for column in EXPECTED_COLUMNS)
    if rtt_count:
        mean = rtt_sum / rtt_count
        std = np.sqrt(max(rtt_sum_sq / rtt_count - mean**2, 0.0))
        summary.extend([
            {"metric": "rtt_non_missing", "value": rtt_count},
            {"metric": "rtt_missing", "value": rtt_missing},
            {"metric": "rtt_min", "value": rtt_min},
            {"metric": "rtt_max", "value": rtt_max},
            {"metric": "rtt_mean", "value": mean},
            {"metric": "rtt_std", "value": std},
        ])
    pd.DataFrame(summary).to_csv(C.TABLE_DIR / "01_rba_summary.csv", index=False, encoding="utf-8-sig")

    label_rows = [
        {"column": column, "value": value, "count": count}
        for column, counts in labels.items() for value, count in counts.most_common()
    ]
    pd.DataFrame(label_rows).to_csv(C.TABLE_DIR / "01_rba_label_counts.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [{"month": month, "takeover_rows": count} for month, count in sorted(month_takeovers.items())]
    ).to_csv(C.TABLE_DIR / "01_rba_monthly_takeovers.csv", index=False, encoding="utf-8-sig")
    top_rows = [
        {"column": column, "value": value, "chunk_top_count": count}
        for column, counts in category_candidates.items()
        for value, count in counts.most_common(TOP_N)
    ]
    pd.DataFrame(top_rows).to_csv(C.TABLE_DIR / "01_rba_top_categories.csv", index=False, encoding="utf-8-sig")

    print(f"\nEDA hoàn tất: {total_rows:,} dòng, {len(header)} cột.")
    print(f"Takeover true: {labels[C.TARGET].get('true', 0):,}")
    print(f"ID sự kiện lặp: {duplicate_indices:,}; ID không hợp lệ: {invalid_indices:,}")
    print(f"Bảng kết quả: {C.TABLE_DIR}")
    print(f"Thời gian: {elapsed_text(time.perf_counter()-started)}")


if __name__ == "__main__":
    main()
