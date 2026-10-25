from collections import defaultdict
from pathlib import Path
import time
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
TABLES_DIR = PROJECT_DIR / "outputs" / "tables"

AUTH_FILE = RAW_DIR / "auth.txt"
REDTEAM_FILE = RAW_DIR / "redteam.txt"
EDA_SUMMARY_FILE = TABLES_DIR / "01_eda_numeric_summary.csv"

# Có thể giảm xuống 500_000 nếu bộ nhớ tiến gần mức đầy.
CHUNK_SIZE = 1_000_000
SECONDS_PER_DAY = 86_400

# Chia theo thời gian tuyệt đối (giây kể từ đầu dữ liệu LANL).
# Số khóa red team duy nhất đến mốc 1.000.000 là 328, đến 2.000.000 là 669.
TRAIN_END_TIME = 1_000_000
VALIDATION_END_TIME = 2_000_000

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
ENTITY_COLUMNS = [
    "source_user",
    "destination_user",
    "source_computer",
    "destination_computer",
]
CATEGORY_COLUMNS = [
    "authentication_type",
    "logon_type",
    "authentication_orientation",
    "success_failure",
]
BASE_OUTPUT_COLUMNS = [
    "time",
    "time_of_day_sin",
    "time_of_day_cos",
    *[f"{column}_prior_log_count" for column in ENTITY_COLUMNS],
]
SPLIT_NAMES = ("train", "validation", "test")


def load_categories() -> dict[str, list[str]]:
    """Đọc từ vựng phân loại do bước EDA 01 tạo ra."""
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


def load_redteam_keys() -> tuple[pd.MultiIndex, set[tuple[int, str, str, str]]]:
    """Đọc khóa red team để gắn nhãn đánh giá, không dùng làm feature."""
    redteam = pd.read_csv(
        REDTEAM_FILE,
        header=None,
        names=REDTEAM_COLUMNS,
        dtype={column: "string" for column in REDTEAM_COLUMNS[1:]},
    )
    redteam["time"] = pd.to_numeric(redteam["time"], errors="raise").astype("int64")
    for column in REDTEAM_COLUMNS[1:]:
        redteam[column] = redteam[column].fillna("Unknown").replace("?", "Unknown")
    keys = redteam[MATCH_COLUMNS].drop_duplicates()
    return pd.MultiIndex.from_frame(keys), set(keys.itertuples(index=False, name=None))


def load_data_summary() -> tuple[int, int, int]:
    """Đọc số dòng và khoảng thời gian từ kết quả EDA."""
    if not EDA_SUMMARY_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {EDA_SUMMARY_FILE}. Hãy chạy src/01_eda.py trước."
        )
    summary = pd.read_csv(EDA_SUMMARY_FILE)
    required = {"rows", "time_min", "time_max"}
    if summary.empty or not required.issubset(summary.columns):
        raise ValueError(f"Bảng EDA không hợp lệ: {EDA_SUMMARY_FILE}")
    return (
        int(summary.loc[0, "rows"]),
        int(summary.loc[0, "time_min"]),
        int(summary.loc[0, "time_max"]),
    )


def split_masks(times: pd.Series) -> dict[str, pd.Series]:
    return {
        "train": times <= TRAIN_END_TIME,
        "validation": (times > TRAIN_END_TIME) & (times <= VALIDATION_END_TIME),
        "test": times > VALIDATION_END_TIME,
    }


def one_hot_encode(
    chunk: pd.DataFrame, categories: dict[str, list[str]]
) -> pd.DataFrame:
    """One-hot bốn trường phân loại với bộ cột cố định giữa các khối."""
    parts = []
    for column in CATEGORY_COLUMNS:
        known_values = categories[column][:-2]
        values = chunk[column].where(chunk[column].isin(known_values), "Other")
        encoded = pd.get_dummies(
            values, prefix=column, prefix_sep="=", dtype="uint8"
        )
        expected = [f"{column}={value}" for value in categories[column]]
        parts.append(encoded.reindex(columns=expected, fill_value=0))
    return pd.concat(parts, axis=1)


def history_features(
    chunk: pd.DataFrame, history: dict[str, dict[str, int]]
) -> pd.DataFrame:
    """Tính tần suất chỉ từ các sự kiện trước sự kiện hiện tại."""
    features = pd.DataFrame(index=chunk.index)
    for column in ENTITY_COLUMNS:
        previous = chunk[column].map(history[column]).fillna(0).to_numpy(
            dtype=np.float64
        )
        within_chunk = chunk.groupby(column, sort=False).cumcount().to_numpy(
            dtype=np.float64
        )
        features[f"{column}_prior_log_count"] = np.log1p(previous + within_chunk)
    return features.astype("float32")


def update_history(chunk: pd.DataFrame, history: dict[str, dict[str, int]]) -> None:
    """Cập nhật bộ đếm sau khi đã tính feature cho khối hiện tại."""
    for column in ENTITY_COLUMNS:
        counts = chunk[column].value_counts(sort=False)
        current = history[column]
        for entity, count in counts.items():
            current[entity] = current.get(entity, 0) + int(count)


def main() -> None:
    for path in (AUTH_FILE, REDTEAM_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file đầu vào: {path}")

    total_rows, time_min, time_max = load_data_summary()
    if not (time_min < TRAIN_END_TIME < VALIDATION_END_TIME < time_max):
        raise ValueError(
            "Mốc chia thời gian không nằm trong khoảng dữ liệu: "
            f"{time_min}..{time_max}"
        )

    categories = load_categories()
    redteam_index, redteam_key_set = load_redteam_keys()
    one_hot_columns = [
        f"{column}={value}"
        for column in CATEGORY_COLUMNS
        for value in categories[column]
    ]
    output_columns = BASE_OUTPUT_COLUMNS + one_hot_columns + ["is_redteam"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_files = {
        split: OUTPUT_DIR / f"02_dataset_{split}.csv.gz" for split in SPLIT_NAMES
    }
    existing = [str(path) for path in output_files.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "File đầu ra đã tồn tại; hãy chuyển chúng sang thư mục khác trước khi "
            "chạy lại: " + ", ".join(existing)
        )

    history: dict[str, dict[str, int]] = {column: {} for column in ENTITY_COLUMNS}
    stats = defaultdict(lambda: {"rows": 0, "redteam_rows": 0})
    matched_redteam_keys: set[tuple[int, str, str, str]] = set()
    invalid_time_rows = 0
    rows_seen = 0
    previous_time = None
    started = time.time()

    reader = pd.read_csv(
        AUTH_FILE,
        header=None,
        names=AUTH_COLUMNS,
        dtype={column: "string" for column in AUTH_COLUMNS},
        chunksize=CHUNK_SIZE,
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        rows_seen += len(chunk)
        chunk["time"] = pd.to_numeric(chunk["time"], errors="coerce")
        valid_time = chunk["time"].notna()
        invalid_time_rows += int((~valid_time).sum())
        chunk = chunk.loc[valid_time].copy()
        if chunk.empty:
            continue
        chunk["time"] = chunk["time"].astype("int64")

        if not chunk["time"].is_monotonic_increasing:
            raise ValueError(
                f"Thời gian không tăng dần trong khối {chunk_number}. "
                "Dừng để tránh tính feature lịch sử sai."
            )
        if previous_time is not None and int(chunk["time"].iloc[0]) < previous_time:
            raise ValueError(
                f"Thời gian bị lùi tại khối {chunk_number}; cần sắp xếp auth.txt "
                "theo time trước khi chạy."
            )

        for column in AUTH_COLUMNS[1:]:
            chunk[column] = chunk[column].fillna("Unknown").replace("?", "Unknown")

        matches = pd.MultiIndex.from_frame(chunk[MATCH_COLUMNS]).isin(redteam_index)
        chunk["is_redteam"] = matches.astype("uint8")
        event_keys = chunk.loc[matches, MATCH_COLUMNS].drop_duplicates()
        if not event_keys.empty:
            matched_redteam_keys.update(event_keys.itertuples(index=False, name=None))

        encoded = one_hot_encode(chunk, categories)
        behavioral = history_features(chunk, history)
        seconds_in_day = chunk["time"] % SECONDS_PER_DAY
        angle = 2 * np.pi * seconds_in_day / SECONDS_PER_DAY

        result = pd.DataFrame(index=chunk.index)
        result["time"] = chunk["time"]
        result["time_of_day_sin"] = np.sin(angle).astype("float32")
        result["time_of_day_cos"] = np.cos(angle).astype("float32")
        result[behavioral.columns] = behavioral
        result[one_hot_columns] = encoded[one_hot_columns]
        result["is_redteam"] = chunk["is_redteam"]
        result = result[output_columns]

        for split, mask in split_masks(chunk["time"]).items():
            split_data = result.loc[mask]
            if split_data.empty:
                continue
            split_data.to_csv(
                output_files[split],
                mode="a",
                header=not output_files[split].exists(),
                index=False,
                compression="gzip",
            )
            stats[split]["rows"] += len(split_data)
            stats[split]["redteam_rows"] += int(split_data["is_redteam"].sum())

        update_history(chunk, history)
        previous_time = int(chunk["time"].iloc[-1])

        elapsed_minutes = (time.time() - started) / 60
        print(
            f"Đã đọc {rows_seen:,} dòng / {total_rows:,} dòng "
            f"({elapsed_minutes:.1f} phút)",
            flush=True,
        )

    summary_rows = []
    for split in SPLIT_NAMES:
        count = stats[split]["rows"]
        redteam_count = stats[split]["redteam_rows"]
        summary_rows.append(
            {
                "split": split,
                "rows": count,
                "redteam_rows": redteam_count,
                "redteam_rate": redteam_count / count if count else 0.0,
                "output_file": str(output_files[split]),
            }
        )
    overall_rows = sum(stats[name]["rows"] for name in SPLIT_NAMES)
    overall_redteam = sum(stats[name]["redteam_rows"] for name in SPLIT_NAMES)
    summary_rows.append(
        {
            "split": "overall",
            "rows": overall_rows,
            "redteam_rows": overall_redteam,
            "redteam_rate": overall_redteam / overall_rows if overall_rows else 0.0,
            "output_file": "",
        }
    )

    summary_file = TABLES_DIR / "02_preprocess_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_file, index=False)

    print("\nHoàn tất tiền xử lý và tạo feature.")
    print(f"Tổng số dòng đã đọc: {rows_seen:,}")
    print(f"Dòng có thời gian không hợp lệ: {invalid_time_rows:,}")
    print(f"Khóa red team duy nhất: {len(redteam_key_set):,}")
    if redteam_key_set:
        coverage = len(matched_redteam_keys) / len(redteam_key_set)
        print(
            f"Khóa red team khớp auth: {len(matched_redteam_keys):,} "
            f"({coverage:.2%})"
        )
    else:
        print("Không có khóa red team trong file.")
    print(
        f"Mốc chia: train <= {TRAIN_END_TIME}; "
        f"validation <= {VALIDATION_END_TIME}; phần còn lại là test."
    )
    print(f"Bảng tổng kết: {summary_file}")
    print(f"File feature: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
