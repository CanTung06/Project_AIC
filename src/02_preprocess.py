from collections import defaultdict
from pathlib import Path
import time
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
TABLES_DIR = PROJECT_DIR / "outputs" / "tables"

AUTH_FILE = RAW_DIR / "auth.txt"
REDTEAM_FILE = RAW_DIR / "redteam.txt"
EDA_SUMMARY_FILE = TABLES_DIR / "01_eda_numeric_summary.csv"

CHUNK_SIZE = 1_000_000
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15

AUTH_COLUMNS = [
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

REDTEAM_COLUMNS = [
    "time",
    "source_user",
    "source_computer",
    "destination_computer",
]

MATCH_COLUMNS = ["time", "source_user", "source_computer", "destination_computer"]
CATEGORY_COLUMNS = [
    "authentication_type",
    "logon_type",
    "authentication_orientation",
    "success_failure",
]
NON_CATEGORY_COLUMNS = [
    column for column in AUTH_COLUMNS if column not in CATEGORY_COLUMNS
]
SPLIT_NAMES = ("train", "validation", "test")


def load_time_range() -> tuple[int, int]:
    """Đọc khoảng thời gian toàn bộ dữ liệu do bước EDA 01 tạo ra."""
    if not EDA_SUMMARY_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {EDA_SUMMARY_FILE}. Hãy chạy src/01_eda.py trước."
        )

    summary = pd.read_csv(EDA_SUMMARY_FILE)
    required = {"time_min", "time_max"}
    if not required.issubset(summary.columns) or summary.empty:
        raise ValueError(f"Bảng thời gian EDA không hợp lệ: {EDA_SUMMARY_FILE}")

    time_min = int(summary.loc[0, "time_min"])
    time_max = int(summary.loc[0, "time_max"])
    if time_max <= time_min:
        raise ValueError(f"Khoảng thời gian không hợp lệ: {time_min}..{time_max}")
    return time_min, time_max


def load_redteam_keys() -> set[tuple[int, str, str, str]]:
    """Đọc các khóa sự kiện red team duy nhất theo thứ tự cột của auth.txt."""
    redteam = pd.read_csv(
        REDTEAM_FILE,
        header=None,
        names=REDTEAM_COLUMNS,
        dtype={column: "string" for column in REDTEAM_COLUMNS[1:]},
    )
    redteam["time"] = pd.to_numeric(redteam["time"], errors="raise").astype("int64")
    for column in REDTEAM_COLUMNS[1:]:
        redteam[column] = redteam[column].fillna("Unknown").replace("?", "Unknown")
    return set(redteam[MATCH_COLUMNS].itertuples(index=False, name=None))


def load_category_values() -> dict[str, list[str]]:
    """Đọc danh sách giá trị phân loại từ kết quả EDA của bước 01."""
    categories = {}
    for column in CATEGORY_COLUMNS:
        category_file = TABLES_DIR / f"01_eda_{column}.csv"
        if not category_file.exists():
            raise FileNotFoundError(
                f"Không tìm thấy {category_file}. Hãy chạy src/01_eda.py trước."
            )
        values = pd.read_csv(category_file, usecols=[column])[column]
        values = values.fillna("Unknown").astype(str).replace("?", "Unknown")
        categories[column] = sorted(set(values) - {"Unknown", "Other"}) + [
            "Unknown",
            "Other",
        ]
    return categories


def make_one_hot_features(
    chunk: pd.DataFrame, categories: dict[str, list[str]]
) -> pd.DataFrame:
    """Mã hóa one-hot bốn trường phân loại, giữ bộ cột nhất quán giữa các khối."""
    encoded_columns = []
    for column in CATEGORY_COLUMNS:
        known_values = categories[column][:-2]
        values = chunk[column].where(chunk[column].isin(known_values), "Other")
        encoded = pd.get_dummies(
            values,
            prefix=column,
            prefix_sep="=",
            dtype="uint8",
        )
        expected_columns = [f"{column}={value}" for value in categories[column]]
        encoded_columns.append(
            encoded.reindex(columns=expected_columns, fill_value=0)
        )
    return pd.concat(encoded_columns, axis=1)


def time_split_masks(times: pd.Series, train_end: int, validation_end: int):
    return {
        "train": times <= train_end,
        "validation": (times > train_end) & (times <= validation_end),
        "test": times > validation_end,
    }


def main() -> None:
    for path in (AUTH_FILE, REDTEAM_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file đầu vào: {path}")

    time_min, time_max = load_time_range()
    total_rows = int(pd.read_csv(EDA_SUMMARY_FILE).loc[0, "rows"])
    time_span = time_max - time_min
    train_end = time_min + int(time_span * TRAIN_FRACTION)
    validation_end = time_min + int(
        time_span * (TRAIN_FRACTION + VALIDATION_FRACTION)
    )

    redteam_keys = load_redteam_keys()
    categories = load_category_values()
    output_columns = (
        NON_CATEGORY_COLUMNS
        + [
            f"{column}={value}"
            for column in CATEGORY_COLUMNS
            for value in categories[column]
        ]
        + ["is_redteam"]
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    output_files = {
        name: OUTPUT_DIR / f"02_dataset_{name}.csv.gz" for name in SPLIT_NAMES
    }
    # Dừng nếu file đã tồn tại để tránh ghi nối hoặc ghi đè kết quả lần chạy trước.
    existing = [str(path) for path in output_files.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "File đầu ra đã tồn tại; hãy di chuyển chúng hoặc chọn thư mục đầu ra mới: "
            + ", ".join(existing)
        )

    stats = defaultdict(lambda: {"rows": 0, "redteam_rows": 0})
    matched_redteam_keys: set[tuple[int, str, str, str]] = set()
    invalid_time_rows = 0
    rows_seen = 0
    started = time.time()

    chunks = pd.read_csv(
        AUTH_FILE,
        header=None,
        names=AUTH_COLUMNS,
        dtype={column: "string" for column in AUTH_COLUMNS},
        chunksize=CHUNK_SIZE,
    )

    for chunk_number, chunk in enumerate(chunks, start=1):
        rows_seen += len(chunk)
        chunk["time"] = pd.to_numeric(chunk["time"], errors="coerce")
        valid_time = chunk["time"].notna()
        invalid_time_rows += int((~valid_time).sum())
        chunk = chunk.loc[valid_time].copy()
        if chunk.empty:
            continue
        chunk["time"] = chunk["time"].astype("int64")

        for column in AUTH_COLUMNS[1:]:
            chunk[column] = chunk[column].fillna("Unknown").replace("?", "Unknown")

        event_keys = list(chunk[MATCH_COLUMNS].itertuples(index=False, name=None))
        labels = [key in redteam_keys for key in event_keys]
        chunk["is_redteam"] = pd.Series(labels, index=chunk.index, dtype="uint8")
        one_hot = make_one_hot_features(chunk, categories)
        processed_chunk = pd.concat(
            [chunk[NON_CATEGORY_COLUMNS], one_hot, chunk[["is_redteam"]]], axis=1
        )

        for split_name, mask in time_split_masks(
            chunk["time"], train_end, validation_end
        ).items():
            split_chunk = processed_chunk.loc[mask, output_columns]
            if split_chunk.empty:
                continue
            split_chunk.to_csv(
                output_files[split_name],
                mode="a",
                header=not output_files[split_name].exists(),
                index=False,
                compression="gzip",
            )
            stats[split_name]["rows"] += len(split_chunk)
            stats[split_name]["redteam_rows"] += int(split_chunk["is_redteam"].sum())

        if any(labels):
            matched_redteam_keys.update(
                key for key, label in zip(event_keys, labels) if label
            )

        elapsed_minutes = (time.time() - started) / 60
        print(
            f"Đã đọc {rows_seen:,} dòng / {total_rows:,} dòng "
            f"({elapsed_minutes:.1f} phút)",
            flush=True,
        )

    summary_rows = []
    for split_name in SPLIT_NAMES:
        summary_rows.append(
            {
                "split": split_name,
                "rows": stats[split_name]["rows"],
                "redteam_rows": stats[split_name]["redteam_rows"],
                "redteam_rate": (
                    stats[split_name]["redteam_rows"] / stats[split_name]["rows"]
                    if stats[split_name]["rows"]
                    else 0.0
                ),
                "output_file": str(output_files[split_name]),
            }
        )

    summary_rows.append(
        {
            "split": "overall",
            "rows": sum(stats[name]["rows"] for name in SPLIT_NAMES),
            "redteam_rows": sum(stats[name]["redteam_rows"] for name in SPLIT_NAMES),
            "redteam_rate": (
                sum(stats[name]["redteam_rows"] for name in SPLIT_NAMES)
                / sum(stats[name]["rows"] for name in SPLIT_NAMES)
                if sum(stats[name]["rows"] for name in SPLIT_NAMES)
                else 0.0
            ),
            "output_file": "",
        }
    )

    summary_file = TABLES_DIR / "02_preprocess_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_file, index=False)

    print("\nHoàn tất tiền xử lý.")
    print(f"Tổng số dòng đầu vào đã đọc: {rows_seen:,}")
    print(f"Số dòng có thời gian không hợp lệ: {invalid_time_rows:,}")
    print(f"Số khóa red team duy nhất trong file: {len(redteam_keys):,}")
    print(
        f"Số khóa red team khớp với auth: {len(matched_redteam_keys):,} "
        f"({len(matched_redteam_keys) / len(redteam_keys):.2%})"
        if redteam_keys
        else "Không tìm thấy khóa red team nào."
    )
    print(f"Mốc thời gian: train <= {train_end}; validation <= {validation_end}")
    print(f"File thống kê: {summary_file}")
    print(f"Các file đã xử lý: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
