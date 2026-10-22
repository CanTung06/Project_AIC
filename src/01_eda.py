from collections import Counter
from pathlib import Path
import time

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_DIR / "data" / "raw" / "auth.txt"
OUTPUT_DIR = PROJECT_DIR / "outputs"
CHUNK_SIZE = 500_000

COLUMNS = [
    "time",
    "source_user",
    "destination_user",
    "source_computer",
    "destination_computer",
    "authentication_type",
    "logon_type",
    "authentication_orientation",
    "success_failure",
]

ENTITY_COLUMNS = [
    "source_user",
    "destination_user",
    "source_computer",
    "destination_computer",
]


def main():
    if not INPUT_FILE.exists():
        print(f"Không tìm thấy file: {INPUT_FILE}")
        return

    tables_dir = OUTPUT_DIR / "tables"
    figures_dir = OUTPUT_DIR / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    category_columns = [
        "authentication_type",
        "logon_type",
        "authentication_orientation",
        "success_failure",
    ]
    missing_values = Counter()
    category_counts = {column: Counter() for column in category_columns}
    unique_entities = {column: set() for column in ENTITY_COLUMNS}
    total_rows = 0
    time_min = None
    time_max = None
    time_sum = 0
    time_count = 0
    start_time = time.time()
    
    chunks = pd.read_csv(
        INPUT_FILE,
        header=None,
        names=COLUMNS,
        chunksize=CHUNK_SIZE,
    )

    for chunk_number, data in enumerate(chunks, start=1):
        total_rows += len(data)
        missing_values.update(data.isna().sum().to_dict())

        # Cộng dồn thống kê cho cột thời gian.
        times = pd.to_numeric(data["time"], errors="coerce")
        if times.notna().any():
            chunk_min = times.min()
            chunk_max = times.max()
            time_min = chunk_min if time_min is None else min(time_min, chunk_min)
            time_max = chunk_max if time_max is None else max(time_max, chunk_max)
            time_sum += times.sum()
            time_count += times.count()

        # Cộng dồn số lần xuất hiện của từng giá trị phân loại.
        for column in category_columns:
            category_counts[column].update(
                data[column].fillna("<MISSING>").astype(str).value_counts().to_dict()
            )

        for column in ENTITY_COLUMNS:
            unique_entities[column].update(data[column].dropna().unique())

        elapsed = (time.time() - start_time) / 60
        print(
            f"Đã xử lý {total_rows:,} dòng ({elapsed:.1f} phút)",
            flush=True,
        )

    summary = pd.DataFrame([
        {
            "column": column,
            "missing_values": missing_values[column],
            "unique_values": len(unique_entities[column]) if column in ENTITY_COLUMNS else "",
        }
        for column in COLUMNS
    ])
    summary.to_csv(tables_dir / "01_eda_summary.csv", index=False)

    for column in category_columns:
        counts = pd.Series(category_counts[column]).sort_values(ascending=False)
        counts.rename_axis(column).reset_index(name="count").to_csv(
            tables_dir / f"01_eda_{column}.csv", index=False
        )

        print(f"\nPhân bố {column}:")
        print(counts.to_string())

        counts.head(10).sort_values().plot(kind="barh", title=f"Top values: {column}")
        plt.xlabel("Số lần xuất hiện")
        plt.tight_layout()
        plt.savefig(figures_dir / f"01_eda_{column}.png")
        plt.close()

    numeric_summary = {
        "rows": total_rows,
        "time_min": time_min,
        "time_max": time_max,
        "time_mean": time_sum / time_count if time_count else None,
        "runtime_minutes": (time.time() - start_time) / 60,
    }
    pd.DataFrame([numeric_summary]).to_csv(
        tables_dir / "01_eda_numeric_summary.csv", index=False
    )

    print(f"Số dòng đã phân tích: {total_rows:,}")
    print("\nSố giá trị duy nhất ở các cột người dùng/máy:")
    for column in ENTITY_COLUMNS:
        print(f"{column}: {len(unique_entities[column]):,}")
    print("\nGiá trị thiếu theo cột:")
    print(summary[["column", "missing_values"]].to_string(index=False))
    print(f"Bảng kết quả: {tables_dir}")
    print(f"Biểu đồ: {figures_dir}")
    print(f"Thời gian nhỏ nhất/lớn nhất: {time_min} / {time_max}")


if __name__ == "__main__":
    main()
