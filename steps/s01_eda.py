import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ids import config as C


def format_elapsed_time(seconds: float) -> str:
    """Đổi số giây thành HH:MM:SS để hiển thị tiến độ."""
    seconds = int(seconds)
    hours, minutes = divmod(seconds, 3600)
    minutes, seconds = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def find_label_column(columns: list[str]) -> str | None:
    """Tìm cột Label không phân biệt chữ hoa/thường."""
    for column in columns:
        if column.strip().casefold() == "label":
            return column
    return None


def save_label_plot(label_df: pd.DataFrame) -> None:
    """Lưu biểu đồ số lượng mẫu theo nhãn."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if label_df.empty:
        return

    plot_df = label_df.sort_values("sample_count", ascending=True)
    height = max(5, min(14, 0.35 * len(plot_df)))
    fig, ax = plt.subplots(figsize=(10, height))
    ax.barh(plot_df["label"], plot_df["sample_count"], color="#3973ac")
    ax.set_xscale("log")
    ax.set_title("Số lượng mẫu theo nhãn (trục số lượng dạng log)")
    ax.set_xlabel("Số mẫu")
    ax.set_ylabel("Nhãn")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "01_label_distribution.png", dpi=150)
    plt.close(fig)


def main() -> int:
    files = sorted(C.RAW_DIR.glob("*.csv"))
    if not files:
        print(f"Không tìm thấy CSV nào trong: {C.RAW_DIR}")
        return 1

    C.ensure_dirs()

    print(f"Bước 01 - EDA | Tìm thấy {len(files)} file CSV")
    print(f"Thư mục dữ liệu: {C.RAW_DIR}")
    print(f"Kích thước chunk: {C.CHUNK_SIZE:,} dòng")

    started = time.perf_counter()
    total_rows = 0
    label_counts: Counter[str] = Counter()
    label_counts_by_file: list[dict] = []
    missing_counts: Counter[str] = Counter()
    invalid_numeric_counts: Counter[str] = Counter()
    infinite_counts: Counter[str] = Counter()
    numeric_stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {"count": 0, "sum": 0.0, "sum_sq": 0.0, "min": float("inf"), "max": float("-inf")}
    )
    file_rows: list[dict] = []
    reference_schema: list[str] | None = None
    schema_mismatches: list[dict] = []
    duplicated_header_names: set[str] = set()

    for file_index, file_path in enumerate(files, start=1):
        file_started = time.perf_counter()
        file_row_count = 0
        current_file_labels: Counter[str] = Counter()
        file_bytes = file_path.stat().st_size
        print(f"\n[{file_index}/{len(files)}] Đang đọc: {file_path.name}")

        try:
            chunks = pd.read_csv(
                file_path,
                chunksize=C.CHUNK_SIZE,
                # CIC-IDS2017 có nhãn dùng ký tự Windows-1252 (dấu gạch ngang).
                encoding="cp1252",
                encoding_errors="replace",
                skipinitialspace=True,
                low_memory=False,
            )

            for chunk in chunks:
                chunk.columns = [str(column).strip().lstrip("\ufeff") for column in chunk.columns]
                schema = list(chunk.columns)
                if reference_schema is None:
                    reference_schema = schema
                elif schema != reference_schema and not any(item["file"] == file_path.name for item in schema_mismatches):
                    schema_mismatches.append({
                        "file": file_path.name,
                        "column_count": len(schema),
                        "schema_mismatch": True,
                    })

                for column in schema:
                    match = re.match(r"^(.*)\.(\d+)$", column)
                    if match:
                        duplicated_header_names.add(match.group(1))

                rows = len(chunk)
                file_row_count += rows
                total_rows += rows

                label_column = find_label_column(schema)
                if label_column is not None:
                    labels = chunk[label_column].astype("string").str.strip()
                    labels = labels.str.replace("\u00ef\u00bf\u00bd", "-", regex=False)
                    labels = labels.str.replace("\ufffd", "-", regex=False)
                    labels = labels.fillna("(THIẾU NHÃN)")
                    counts = labels.value_counts(dropna=False)
                    for label, count in counts.items():
                        label = str(label)
                        label_counts[label] += int(count)
                        current_file_labels[label] += int(count)

                for column in schema:
                    if column == label_column:
                        missing_counts[column] += int(chunk[column].isna().sum())
                        continue

                    raw_values = chunk[column]
                    missing_counts[column] += int(raw_values.isna().sum())
                    values = pd.to_numeric(raw_values, errors="coerce")
                    invalid_numeric_counts[column] += int((raw_values.notna() & values.isna()).sum())
                    finite_mask = np.isfinite(values.to_numpy(dtype="float64", na_value=np.nan))
                    infinite_counts[column] += int((values.notna().to_numpy() & ~finite_mask).sum())
                    finite_values = values.to_numpy(dtype="float64", na_value=np.nan)[finite_mask]
                    if finite_values.size:
                        stats = numeric_stats[column]
                        stats["count"] += int(finite_values.size)
                        stats["sum"] += float(finite_values.sum(dtype="float64"))
                        stats["sum_sq"] += float(np.square(finite_values).sum(dtype="float64"))
                        stats["min"] = min(stats["min"], float(finite_values.min()))
                        stats["max"] = max(stats["max"], float(finite_values.max()))

                elapsed = time.perf_counter() - started
                print(
                    f"  Đã đọc {file_row_count:,} dòng trong file | "
                    f"tổng {total_rows:,} dòng | {format_elapsed_time(elapsed)}",
                    end="\r",
                    flush=True,
                )

        except Exception as exc:
            print(f"\nLỗi khi đọc {file_path.name}: {exc}")
            raise

        print(
            f"  Hoàn tất {file_path.name}: {file_row_count:,} dòng | "
            f"{format_elapsed_time(time.perf_counter() - file_started)}"
        )
        file_rows.append({
            "file_name": file_path.name,
            "row_count": file_row_count,
            "column_count": len(reference_schema or []),
            "size_mb": round(file_bytes / (1024 * 1024), 2),
        })
        for label, count in current_file_labels.items():
            label_counts_by_file.append({"file_name": file_path.name, "label": label, "sample_count": count})

    total_elapsed = time.perf_counter() - started
    number_of_columns = len(reference_schema or [])

    label_df = pd.DataFrame(
        [{"label": label, "sample_count": count, "percentage": 100 * count / max(total_rows, 1)}
         for label, count in sorted(label_counts.items(), key=lambda item: item[1], reverse=True)]
    )
    file_label_df = pd.DataFrame(label_counts_by_file)
    file_df = pd.DataFrame(file_rows)

    missing_df = pd.DataFrame([
        {
            "column": column,
            "missing_count": int(count),
            "missing_percentage": 100 * count / max(total_rows, 1),
            "non_numeric_count": int(invalid_numeric_counts[column]),
            "infinite_count": int(infinite_counts[column]),
        }
        for column, count in missing_counts.items()
    ]).sort_values("missing_count", ascending=False)

    numeric_rows = []
    for column, stats in numeric_stats.items():
        count = int(stats["count"])
        mean = stats["sum"] / count if count else np.nan
        variance = max(stats["sum_sq"] / count - mean * mean, 0.0) if count else np.nan
        numeric_rows.append({
            "column": column,
            "valid_value_count": count,
            "mean": mean,
            "standard_deviation": float(np.sqrt(variance)) if count else np.nan,
            "minimum": stats["min"] if count else np.nan,
            "maximum": stats["max"] if count else np.nan,
        })
    numeric_df = pd.DataFrame(numeric_rows)

    file_df.to_csv(C.TABLES_DIR / "01_file_summary.csv", index=False, encoding="utf-8-sig")
    label_df.to_csv(C.TABLES_DIR / "01_label_distribution.csv", index=False, encoding="utf-8-sig")
    file_label_df.to_csv(C.TABLES_DIR / "01_label_by_file.csv", index=False, encoding="utf-8-sig")
    missing_df.to_csv(C.TABLES_DIR / "01_missing_values.csv", index=False, encoding="utf-8-sig")
    numeric_df.to_csv(C.TABLES_DIR / "01_numeric_summary.csv", index=False, encoding="utf-8-sig")
    save_label_plot(label_df)

    summary = {
        "raw_directory": str(C.RAW_DIR),
        "csv_file_count": len(files),
        "total_rows": total_rows,
        "column_count": number_of_columns,
        "schemas_match": len(schema_mismatches) == 0,
        "schema_mismatches": schema_mismatches,
        "duplicated_header_columns": sorted(duplicated_header_names),
        "label_distribution": {label: int(count) for label, count in label_counts.items()},
        "total_missing_values": int(sum(missing_counts.values())),
        "total_infinite_values": int(sum(infinite_counts.values())),
        "processing_time_seconds": round(total_elapsed, 2),
        "processing_time": format_elapsed_time(total_elapsed),
    }
    with (C.METRICS_DIR / "01_eda_summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print("TÓM TẮT EDA")
    print(f"Số file CSV       : {len(files):,}")
    print(f"Tổng số dòng      : {total_rows:,}")
    print(f"Số cột            : {number_of_columns:,}")
    print(f"Schema đồng nhất  : {'Có' if not schema_mismatches else 'Không'}")
    print(f"Nhãn              : {len(label_counts):,} loại")
    print(f"Thời gian xử lý   : {format_elapsed_time(total_elapsed)}")
    print(f"Kết quả           : {C.OUTPUTS_DIR}")
    print("\nPhân bố nhãn:")
    if not label_df.empty:
        print(label_df.to_string(index=False, formatters={"percentage": "{:.4f}%".format}))
    if duplicated_header_names:
        print("\nTên cột bị lặp trong CSV gốc:")
        print(", ".join(sorted(duplicated_header_names)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
