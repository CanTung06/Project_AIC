from collections import Counter
from pathlib import Path
import time
import sys

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_DIR / "data" / "raw" / "rba-dataset.csv"
TABLES_DIR = PROJECT_DIR / "outputs" / "tables"

CHUNK_SIZE = 1_000_000
TOP_N = 30

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LABEL_COLUMNS = ["Login Successful", "Is Attack IP", "Is Account Takeover"]
CATEGORICAL_COLUMNS = [
    "Country",
    "Region",
    "City",
    "ASN",
    "Browser Name and Version",
    "OS Name and Version",
    "Device Type",
]
READ_COLUMNS = [
    "Login Timestamp",
    "Round-Trip Time [ms]",
    *LABEL_COLUMNS,
    *CATEGORICAL_COLUMNS,
]


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy dataset: {INPUT_FILE}")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    missing_counts = Counter()
    label_counts = {column: Counter() for column in LABEL_COLUMNS}
    label_combinations = Counter()
    category_counts = {column: Counter() for column in CATEGORICAL_COLUMNS}
    monthly_takeovers = Counter()

    total_rows = 0
    invalid_timestamps = 0
    timestamp_min = None
    timestamp_max = None
    rtt_count = 0
    rtt_missing = 0
    rtt_sum = 0.0
    rtt_sum_squares = 0.0
    rtt_min = None
    rtt_max = None
    started = time.time()

    reader = pd.read_csv(
        INPUT_FILE,
        usecols=READ_COLUMNS,
        dtype="string",
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for chunk in reader:
        total_rows += len(chunk)

        for column in READ_COLUMNS:
            values = chunk[column].str.strip()
            missing_counts[column] += int(
                (values.isna() | values.eq("") | values.eq("-")).sum()
            )

        timestamps = pd.to_datetime(
            chunk["Login Timestamp"],
            format="%Y-%m-%d %H:%M:%S.%f",
            errors="coerce",
        )
        invalid_timestamps += int(timestamps.isna().sum())
        valid_timestamps = timestamps.dropna()
        if not valid_timestamps.empty:
            current_min = valid_timestamps.min()
            current_max = valid_timestamps.max()
            timestamp_min = current_min if timestamp_min is None else min(timestamp_min, current_min)
            timestamp_max = current_max if timestamp_max is None else max(timestamp_max, current_max)

        for column in LABEL_COLUMNS:
            normalized = chunk[column].fillna("<missing>").str.strip().str.lower()
            label_counts[column].update(normalized.value_counts(dropna=False).to_dict())

        combinations = (
            chunk[LABEL_COLUMNS]
            .fillna("<missing>")
            .astype("string")
            .agg(" | ".join, axis=1)
        )
        label_combinations.update(combinations.value_counts().to_dict())

        for column in CATEGORICAL_COLUMNS:
            values = chunk[column].fillna("<missing>").str.strip()
            category_counts[column].update(values.value_counts(dropna=False).to_dict())

        takeover_mask = chunk["Is Account Takeover"].str.lower().eq("true")
        months = timestamps.dt.strftime("%Y-%m")
        monthly_takeovers.update(months[takeover_mask & months.notna()].value_counts().to_dict())

        rtt = pd.to_numeric(chunk["Round-Trip Time [ms]"], errors="coerce")
        valid_rtt = rtt.dropna()
        rtt_missing += len(rtt) - len(valid_rtt)
        if not valid_rtt.empty:
            values = valid_rtt.to_numpy(dtype=np.float64, copy=False)
            rtt_count += len(values)
            rtt_sum += float(values.sum())
            rtt_sum_squares += float(np.square(values).sum())
            local_min = float(values.min())
            local_max = float(values.max())
            rtt_min = local_min if rtt_min is None else min(rtt_min, local_min)
            rtt_max = local_max if rtt_max is None else max(rtt_max, local_max)

        elapsed_seconds = time.time() - started
        rows_per_second = total_rows / elapsed_seconds if elapsed_seconds else 0.0
        print(
            f"Tiến độ: đã phân tích {total_rows:,} dòng | "
            f"Tốc độ: {rows_per_second:,.0f} dòng/giây | "
            f"Đã chạy: {elapsed_seconds / 60:.1f} phút.",
            flush=True,
        )

    # Bảng tổng quan dataset, nhãn và chất lượng dữ liệu.
    positive_rows = int(label_counts["Is Account Takeover"].get("true", 0))
    summary_rows = [
        {"metric": "total_rows", "value": total_rows},
        {"metric": "file_size_bytes", "value": INPUT_FILE.stat().st_size},
        {"metric": "timestamp_min", "value": timestamp_min},
        {"metric": "timestamp_max", "value": timestamp_max},
        {"metric": "invalid_timestamps", "value": invalid_timestamps},
        {"metric": "account_takeover_true", "value": positive_rows},
        {
            "metric": "account_takeover_rate",
            "value": positive_rows / total_rows if total_rows else 0.0,
        },
    ]
    for column in READ_COLUMNS:
        summary_rows.append(
            {"metric": f"missing_or_dash::{column}", "value": missing_counts[column]}
        )
    pd.DataFrame(summary_rows).to_csv(
        TABLES_DIR / "01_rba_summary.csv", index=False
    )

    label_rows = []
    for column, counts in label_counts.items():
        for value, count in counts.most_common():
            label_rows.append(
                {"column": column, "value": value, "count": count, "rate": count / total_rows}
            )
    pd.DataFrame(label_rows).to_csv(
        TABLES_DIR / "01_rba_label_counts.csv", index=False
    )
    pd.DataFrame(
        [
            {"label_combination": value, "count": count, "rate": count / total_rows}
            for value, count in label_combinations.most_common()
        ]
    ).to_csv(TABLES_DIR / "01_rba_label_combinations.csv", index=False)

    category_rows = []
    for column, counts in category_counts.items():
        for value, count in counts.most_common(TOP_N):
            category_rows.append(
                {"column": column, "value": value, "count": count, "rate": count / total_rows}
            )
    pd.DataFrame(category_rows).to_csv(
        TABLES_DIR / "01_rba_top_categories.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "column": column,
                "distinct_values": len(counts),
                "top_value": counts.most_common(1)[0][0] if counts else None,
                "top_value_count": counts.most_common(1)[0][1] if counts else 0,
            }
            for column, counts in category_counts.items()
        ]
    ).to_csv(TABLES_DIR / "01_rba_category_cardinality.csv", index=False)

    if rtt_count:
        rtt_mean = rtt_sum / rtt_count if rtt_count else 0.0
        rtt_variance = max(rtt_sum_squares / rtt_count - rtt_mean**2, 0.0) if rtt_count else 0.0
        rtt_summary = {
            "non_missing_count": rtt_count,
            "missing_count": rtt_missing,
            "missing_rate": rtt_missing / total_rows if total_rows else 0.0,
            "min": rtt_min,
            "max": rtt_max,
            "mean": rtt_mean,
            "std": float(np.sqrt(rtt_variance)),
        }
        pd.DataFrame([{"metric": key, "value": value} for key, value in rtt_summary.items()]).to_csv(
            TABLES_DIR / "01_rba_rtt_summary.csv", index=False
        )

    pd.DataFrame(
        [
            {"month": month, "account_takeover_rows": count}
            for month, count in sorted(monthly_takeovers.items())
        ]
    ).to_csv(TABLES_DIR / "01_rba_monthly_takeovers.csv", index=False)

    print("\nHoàn tất EDA cho RBA.")
    print(f"Tổng số dòng: {total_rows:,}")
    print(f"Account takeover đã gắn nhãn: {positive_rows:,} ({positive_rows / total_rows:.6%})")
    print(f"Khoảng thời gian: {timestamp_min} đến {timestamp_max}")
    print(f"Bảng kết quả: {TABLES_DIR}")
    print(
        "Ghi chú: file này là dữ liệu tổng hợp; Is Account Takeover là nhãn mục tiêu, "
        "không đưa nhãn này vào feature."
    )


if __name__ == "__main__":
    main()
