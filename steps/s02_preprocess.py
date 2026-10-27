"""Tiền xử lý RBA theo chunk, tạo feature và chia train/validation/test."""
from collections import Counter
import json
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
MAX_CATEGORIES_PER_COLUMN = 100
DENSE_BATCH_MAX_ROWS = 10_000
DENSE_BATCH_MAX_CELLS = 10_000_000
TRAIN_FRACTION, VALIDATION_FRACTION, TEST_FRACTION = 0.70, 0.15, 0.15
SEED = C.SEED

INDEX_COLUMN = "index"
TIMESTAMP_COLUMN = "Login Timestamp"
USER_ID_COLUMN = "User ID"
RTT_COLUMN = "Round-Trip Time [ms]"
LOGIN_SUCCESS_COLUMN = "Login Successful"
ATTACK_IP_COLUMN = "Is Attack IP"
CATEGORY_COLUMNS = [
    "IP Address", "Country", "Region", "City", "ASN", "User Agent String",
    "Browser Name and Version", "OS Name and Version", "Device Type",
]
READ_COLUMNS = [
    INDEX_COLUMN, TIMESTAMP_COLUMN, USER_ID_COLUMN, RTT_COLUMN,
    *CATEGORY_COLUMNS, LOGIN_SUCCESS_COLUMN, ATTACK_IP_COLUMN, C.TARGET,
]
MISSING = "<MISSING>"
OTHER = "__RBA_INTERNAL_OTHER_8F27A6__"
SPLITS = {0: "train", 1: "validation", 2: "test"}

TIME_FEATURES = [
    "login_hour", "login_day_of_week", "login_month", "login_is_weekend",
    "login_timestamp_missing", "login_hour_sin", "login_hour_cos",
    "login_day_of_week_sin", "login_day_of_week_cos",
    "login_day_of_year_sin", "login_day_of_year_cos",
]
RTT_FEATURES = ["round_trip_time_ms", "round_trip_time_log1p", "round_trip_time_missing"]
BOOLEAN_FEATURES = [
    "login_successful", "login_successful_missing", "is_attack_ip", "is_attack_ip_missing",
]
HISTORY_FEATURES = [
    "account_prior_login_count_log1p", "account_prior_failed_login_count_log1p",
    "seconds_since_previous_login_log1p", "account_has_previous_login",
    "previous_login_successful", "previous_login_failed",
    "previous_login_successful_missing",
    *[f"previous_{name}_changed" for name in (
        "ip_address", "country", "region", "city", "asn", "user_agent",
        "browser", "operating_system", "device_type",
    )],
]
NUMERIC_FEATURES = TIME_FEATURES + RTT_FEATURES + BOOLEAN_FEATURES + HISTORY_FEATURES


def elapsed_text(seconds: float) -> str:
    hours, rest = divmod(int(seconds), 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def parse_target(chunk: pd.DataFrame) -> np.ndarray:
    values = chunk[C.TARGET].fillna("").astype("string").str.strip().str.lower()
    mapped = values.map({"true": 1, "false": 0})
    if mapped.isna().any():
        raise ValueError(f"Nhãn {C.TARGET} không hợp lệ: {values[mapped.isna()].value_counts().head().to_dict()}")
    return mapped.to_numpy(dtype=np.uint8)


def normalize_category(values: pd.Series) -> pd.Series:
    result = values.astype("string").str.strip()
    missing = result.isna() | result.eq("") | result.eq("-")
    return result.mask(missing, MISSING).astype("string")


def parse_boolean(values: pd.Series, name: str) -> tuple[np.ndarray, np.ndarray]:
    normalized = values.astype("string").str.strip().str.lower()
    mapped = normalized.map({"true": 1.0, "false": 0.0})
    missing = normalized.isna() | normalized.eq("") | normalized.eq("-")
    invalid = mapped.isna() & ~missing
    if invalid.any():
        raise ValueError(f"Giá trị {name} không hợp lệ: {normalized[invalid].value_counts().head().to_dict()}")
    return mapped.fillna(0).to_numpy(dtype=np.float32), missing.to_numpy(dtype=np.float32)


def check_index_chunk(chunk: pd.DataFrame, seen: np.ndarray) -> np.ndarray:
    numeric = pd.to_numeric(chunk[INDEX_COLUMN], errors="coerce")
    valid = numeric.notna() & numeric.ge(0) & numeric.mod(1).eq(0)
    if not valid.all():
        raise ValueError("Cột index có giá trị thiếu, âm hoặc không phải số nguyên.")
    indices = numeric.to_numpy(dtype=np.int64)
    unique, counts = np.unique(indices, return_counts=True)
    duplicates = int(np.maximum(counts - 1, 0).sum())
    in_range = unique < len(seen)
    if in_range.any():
        duplicates += int(seen[unique[in_range]].sum())
    if duplicates:
        raise ValueError(f"Phát hiện {duplicates:,} index sự kiện trùng; dừng để không âm thầm bỏ/giữ sai bản ghi.")
    largest = int(unique[-1])
    if largest >= len(seen):
        new_size = max(largest + 1, max(1024, len(seen) * 2))
        if new_size > 100_000_000:
            raise ValueError("index quá lớn để kiểm tra duy nhất an toàn trong RAM.")
        seen = np.pad(seen, (0, new_size - len(seen)))
    seen[unique] = True
    return seen


def validate_time_order(chunk: pd.DataFrame, previous_ns: int | None) -> int:
    timestamps = pd.to_datetime(chunk[TIMESTAMP_COLUMN], errors="coerce")
    if timestamps.isna().any():
        raise ValueError(f"{TIMESTAMP_COLUMN} có giá trị không hợp lệ; không thể tạo feature lịch sử.")
    ns = timestamps.to_numpy(dtype="datetime64[ns]").view(np.int64)
    if len(ns) > 1 and np.any(np.diff(ns) < 0):
        raise ValueError("Nguồn chưa sắp xếp tăng dần theo timestamp; dừng để tránh dùng tương lai làm lịch sử.")
    if previous_ns is not None and len(ns) and ns[0] < previous_ns:
        raise ValueError("Timestamp bị lùi giữa hai chunk.")
    return int(ns[-1]) if len(ns) else previous_ns


def assign_splits(labels: np.ndarray, row_start: int, rng, positive_assignment=None) -> np.ndarray:
    random_values = rng.random(len(labels))
    split = np.where(
        random_values < TRAIN_FRACTION,
        0,
        np.where(random_values < TRAIN_FRACTION + VALIDATION_FRACTION, 1, 2),
    ).astype(np.uint8)
    if positive_assignment:
        for local in np.flatnonzero(labels == 1):
            split[local] = positive_assignment[row_start + int(local)]
    return split


def positive_split_counts(total: int) -> tuple[int, int, int]:
    if total < 3:
        raise ValueError(f"Chỉ có {total} takeover; cần ít nhất 3 để chia vào cả ba tập.")
    raw = np.array([TRAIN_FRACTION, VALIDATION_FRACTION, TEST_FRACTION]) * total
    counts = np.floor(raw).astype(int)
    for split_id in np.argsort(-(raw - counts))[: total - int(counts.sum())]:
        counts[split_id] += 1
    for split_id in range(3):
        if counts[split_id] == 0:
            donor = int(np.argmax(counts))
            counts[donor] -= 1
            counts[split_id] += 1
    return tuple(int(value) for value in counts)


def read_chunks():
    return pd.read_csv(
        C.RAW_FILE, usecols=READ_COLUMNS, dtype="string",
        chunksize=CHUNK_SIZE, low_memory=False,
    )


class AccountHistory:
    """Tạo lịch sử chỉ từ các sự kiện trước đó theo thứ tự timestamp."""
    def __init__(self, category_levels: dict[str, list[str]]):
        self.state: dict[str, tuple] = {}
        self.last_timestamp_ns: int | None = None
        self.processed_rows = 0
        self.category_maps = {}
        self.other_codes = {}
        for column, levels in category_levels.items():
            kept = [value for value in levels if value != OTHER]
            self.category_maps[column] = {value: code for code, value in enumerate(kept)}
            self.other_codes[column] = len(kept)

    def transform(self, chunk: pd.DataFrame) -> np.ndarray:
        timestamps = pd.to_datetime(chunk[TIMESTAMP_COLUMN], errors="coerce")
        ns = timestamps.to_numpy(dtype="datetime64[ns]").view(np.int64)
        if len(ns) > 1 and np.any(np.diff(ns) < 0):
            raise ValueError("Timestamp trong chunk không tăng dần.")
        if self.last_timestamp_ns is not None and len(ns) and ns[0] < self.last_timestamp_ns:
            raise ValueError("Timestamp lùi giữa các chunk khi tính lịch sử.")
        if len(ns):
            self.last_timestamp_ns = int(ns[-1])

        raw_users = normalize_category(chunk[USER_ID_COLUMN]).astype(str).to_numpy(dtype=object)
        missing_user = raw_users == MISSING
        row_start = self.processed_rows
        # Tài khoản thiếu ID được xem như một tài khoản riêng cho từng event.
        raw_users[missing_user] = [f"__MISSING_USER_{row_start+i}" for i in np.flatnonzero(missing_user)]
        codes, unique_users = pd.factorize(raw_users, sort=False)
        row_count, user_count = len(chunk), len(unique_users)
        categories = {}
        for column in CATEGORY_COLUMNS:
            normalized = normalize_category(chunk[column])
            encoded = normalized.map(self.category_maps[column])
            categories[column] = encoded.fillna(self.other_codes[column]).to_numpy(dtype=np.uint8)
        success, success_missing = parse_boolean(chunk[LOGIN_SUCCESS_COLUMN], LOGIN_SUCCESS_COLUMN)
        failed = ((success == 0) & (success_missing == 0)).astype(np.int64)

        source = pd.DataFrame({
            "timestamp": timestamps.to_numpy(dtype="datetime64[ns]"),
            "success": success,
            "success_missing": success_missing,
            **categories,
        })
        previous = source.groupby(codes, sort=False).shift(1)
        previous_ns = previous["timestamp"].to_numpy(dtype="datetime64[ns]").view(np.int64).copy()
        previous_success = previous["success"].fillna(0).to_numpy(dtype=np.float32, copy=True)
        previous_success_missing = previous["success_missing"].fillna(0).to_numpy(dtype=np.float32, copy=True)
        previous_categories = {
            col: previous[col].fillna(self.other_codes[col]).to_numpy(dtype=np.uint8, copy=True)
            for col in CATEGORY_COLUMNS
        }
        prior_in_chunk = (
            pd.Series(np.ones(row_count, dtype=np.int64))
            .groupby(codes, sort=False).cumcount().to_numpy(dtype=np.int64)
        )
        base_logins = np.zeros(user_count, dtype=np.int64)
        base_failed = np.zeros(user_count, dtype=np.int64)
        for position in np.flatnonzero(prior_in_chunk == 0):
            key = str(unique_users[codes[position]])
            state = self.state.get(key)
            if state is None:
                continue
            code = codes[position]
            base_logins[code], base_failed[code] = state[0], state[1]
            previous_ns[position], previous_success[position] = state[2], state[3]
            previous_success_missing[position] = state[4]
            for category_index, column in enumerate(CATEGORY_COLUMNS):
                previous_categories[column][position] = state[5 + category_index]

        prior_logins = base_logins[codes] + prior_in_chunk
        prior_failed = base_failed[codes] + (
            pd.Series(failed).groupby(codes, sort=False).cumsum().to_numpy(dtype=np.int64) - failed
        )
        has_previous = prior_logins > 0
        has_previous_time = previous_ns != np.iinfo(np.int64).min
        seconds = np.zeros(row_count, dtype=np.float64)
        seconds[has_previous_time] = np.maximum(
            (ns[has_previous_time] - previous_ns[has_previous_time]) / 1_000_000_000.0, 0.0
        )
        columns = [
            np.log1p(prior_logins).astype(np.float32),
            np.log1p(prior_failed).astype(np.float32),
            np.log1p(seconds).astype(np.float32),
            has_previous.astype(np.float32),
            (previous_success * has_previous).astype(np.float32),
            ((previous_success == 0) & (previous_success_missing == 0) & has_previous).astype(np.float32),
            (previous_success_missing * has_previous).astype(np.float32),
        ]
        for column in CATEGORY_COLUMNS:
            columns.append(((categories[column] != previous_categories[column]) & has_previous).astype(np.float32))
        result = np.column_stack(columns).astype(np.float32, copy=False)

        local_counts = np.bincount(codes, minlength=user_count)
        local_failed = np.bincount(codes, weights=failed, minlength=user_count).astype(np.int64)
        last_positions = np.full(user_count, -1, dtype=np.int64)
        np.maximum.at(last_positions, codes, np.arange(row_count, dtype=np.int64))
        for code, user in enumerate(unique_users):
            position = int(last_positions[code])
            if missing_user[position]:
                continue
            key = str(user)
            old = self.state.get(key)
            base_login = int(old[0]) if old else 0
            base_fail = int(old[1]) if old else 0
            self.state[key] = (
                base_login + int(local_counts[code]),
                base_fail + int(local_failed[code]),
                int(ns[position]),
                float(success[position]),
                float(success_missing[position]),
                *[int(categories[col][position]) for col in CATEGORY_COLUMNS],
            )
        self.processed_rows += row_count
        return result


def numeric_features(chunk: pd.DataFrame, history: np.ndarray) -> np.ndarray:
    timestamps = pd.to_datetime(chunk[TIMESTAMP_COLUMN], errors="coerce")
    hour = timestamps.dt.hour + timestamps.dt.minute / 60.0
    weekday = timestamps.dt.dayofweek
    day_year = timestamps.dt.dayofyear
    time_frame = pd.DataFrame({
        "login_hour": timestamps.dt.hour.fillna(0),
        "login_day_of_week": weekday.fillna(0),
        "login_month": timestamps.dt.month.fillna(0),
        "login_is_weekend": (weekday >= 5).fillna(False),
        "login_timestamp_missing": timestamps.isna(),
        "login_hour_sin": np.sin(2*np.pi*hour/24),
        "login_hour_cos": np.cos(2*np.pi*hour/24),
        "login_day_of_week_sin": np.sin(2*np.pi*weekday/7),
        "login_day_of_week_cos": np.cos(2*np.pi*weekday/7),
        "login_day_of_year_sin": np.sin(2*np.pi*day_year/365.2425),
        "login_day_of_year_cos": np.cos(2*np.pi*day_year/365.2425),
    }).fillna(0).astype(np.float32)
    rtt = pd.to_numeric(chunk[RTT_COLUMN], errors="coerce")
    rtt_features = np.column_stack((
        rtt.fillna(0).to_numpy(dtype=np.float32),
        np.log1p(rtt.clip(lower=0).fillna(0).to_numpy(dtype=np.float32)),
        rtt.isna().to_numpy(dtype=np.float32),
    ))
    success, success_missing = parse_boolean(chunk[LOGIN_SUCCESS_COLUMN], LOGIN_SUCCESS_COLUMN)
    attack_ip, attack_ip_missing = parse_boolean(chunk[ATTACK_IP_COLUMN], ATTACK_IP_COLUMN)
    boolean = np.column_stack((success, success_missing, attack_ip, attack_ip_missing))
    return np.concatenate((time_frame.to_numpy(), rtt_features, boolean, history), axis=1).astype(np.float32, copy=False)


def make_feature_schema(category_levels: dict[str, list[str]]) -> tuple[list[str], list[str]]:
    one_hot_names = []
    for column in CATEGORY_COLUMNS:
        for value in category_levels[column]:
            one_hot_names.append(
                f"{column}=OTHER" if value == OTHER else f"{column}=VALUE::{value}"
            )
    return one_hot_names, NUMERIC_FEATURES + one_hot_names


def main() -> None:
    if not C.RAW_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy dataset: {C.RAW_FILE}")
    actual_columns = pd.read_csv(C.RAW_FILE, nrows=0).columns.tolist()
    if actual_columns != READ_COLUMNS:
        raise ValueError(
            "Schema/delimiter nguồn không khớp. File cần là CSV dấu phẩy với đúng 16 cột "
            f"đã xác nhận. Cột thực tế: {actual_columns}"
        )
    if C.PROCESSED_DIR.exists() and any(C.PROCESSED_DIR.iterdir()):
        raise FileExistsError(f"{C.PROCESSED_DIR} không rỗng; hãy chuyển kết quả cũ ra trước để tránh ghi đè.")

    started = time.perf_counter()
    total_rows = 0
    positives: dict[int, list[str]] = {}
    train_category_counts = {column: Counter() for column in CATEGORY_COLUMNS}
    rng = np.random.default_rng(SEED)
    previous_ns = None
    seen_indices = np.zeros(0, dtype=np.bool_)

    print("Lượt 1/2: kiểm tra schema, index, thời gian, nhãn và category train.")
    for chunk in read_chunks():
        seen_indices = check_index_chunk(chunk, seen_indices)
        previous_ns = validate_time_order(chunk, previous_ns)
        y = parse_target(chunk)
        split = assign_splits(y, total_rows, rng)
        train_negative = (split == 0) & (y == 0)
        for column in CATEGORY_COLUMNS:
            values = normalize_category(chunk[column])
            if values.eq(OTHER).any():
                raise ValueError("Giá trị nguồn đụng token nội bộ OTHER; đổi token trước khi chạy.")
            train_category_counts[column].update(values[train_negative].value_counts().to_dict())
        for local in np.flatnonzero(y == 1):
            row_id = total_rows + int(local)
            positives[row_id] = [str(normalize_category(chunk[col].iloc[[local]]).iloc[0]) for col in CATEGORY_COLUMNS]
        total_rows += len(chunk)
        print(
            f"Lượt 1: {total_rows:,} dòng, takeover={len(positives):,} | "
            f"{elapsed_text(time.perf_counter()-started)}",
            flush=True,
        )

    if not positives:
        raise ValueError("Dataset không có mẫu takeover.")
    positive_counts = positive_split_counts(len(positives))
    positive_ids = np.array(list(positives), dtype=np.int64)
    shuffled = np.random.default_rng(SEED + 1).permutation(positive_ids)
    positive_assignment: dict[int, int] = {}
    cursor = 0
    for split_id, count in enumerate(positive_counts):
        for row_id in shuffled[cursor:cursor+count]:
            positive_assignment[int(row_id)] = split_id
        cursor += count
    for row_id, values in positives.items():
        if positive_assignment[row_id] == 0:
            for column, value in zip(CATEGORY_COLUMNS, values):
                train_category_counts[column][value] += 1

    category_levels = {}
    for column, counter in train_category_counts.items():
        top_values = [
            value for value, _ in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
            if value != OTHER
        ][:MAX_CATEGORIES_PER_COLUMN]
        category_levels[column] = top_values + [OTHER]
    one_hot_names, feature_names = make_feature_schema(category_levels)
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("Tên feature bị trùng sau mã hóa category.")

    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    split_paths = {name: C.PROCESSED_DIR / f"{name}.csv" for name in SPLITS.values()}
    row_start = 0
    rng = np.random.default_rng(SEED)
    history = AccountHistory(category_levels)
    split_rows, split_positives = Counter(), Counter()
    batch_counts = Counter()
    batch_rows = max(1, min(DENSE_BATCH_MAX_ROWS, DENSE_BATCH_MAX_CELLS // max(len(one_hot_names), 1)))
    headers_written = set()

    with (
        split_paths["train"].open("w", encoding="utf-8-sig", newline="") as f_train,
        split_paths["validation"].open("w", encoding="utf-8-sig", newline="") as f_validation,
        split_paths["test"].open("w", encoding="utf-8-sig", newline="") as f_test,
    ):
        file_handles = {"train": f_train, "validation": f_validation, "test": f_test}
        print("Lượt 2/2: tạo feature quá khứ, one-hot category và ghi CSV.")
        for chunk in read_chunks():
            y = parse_target(chunk)
            split = assign_splits(y, row_start, rng, positive_assignment)
            history_array = history.transform(chunk)
            numeric = numeric_features(chunk, history_array)
            normalized = {col: normalize_category(chunk[col]) for col in CATEGORY_COLUMNS}

            for split_id, split_name in SPLITS.items():
                local_rows = np.flatnonzero(split == split_id)
                for start in range(0, len(local_rows), batch_rows):
                    idx = local_rows[start:start+batch_rows]
                    cats = pd.DataFrame({col: normalized[col].iloc[idx].to_numpy() for col in CATEGORY_COLUMNS})
                    for col in CATEGORY_COLUMNS:
                        cats[col] = cats[col].where(cats[col].isin(category_levels[col]), OTHER)
                    encoded = pd.get_dummies(cats, columns=CATEGORY_COLUMNS, prefix_sep="=", dtype=np.uint8)
                    rename = {}
                    for col in CATEGORY_COLUMNS:
                        for value in category_levels[col]:
                            if value == OTHER:
                                rename[f"{col}={OTHER}"] = f"{col}=OTHER"
                            else:
                                rename[f"{col}={value}"] = f"{col}=VALUE::{value}"
                    encoded = encoded.rename(columns=rename).reindex(columns=one_hot_names, fill_value=0)
                    numeric_frame = pd.DataFrame(numeric[idx], columns=NUMERIC_FEATURES)
                    output = pd.concat([numeric_frame, encoded], axis=1)
                    output[C.TARGET] = y[idx]
                    output.to_csv(file_handles[split_name], index=False, header=split_name not in headers_written)
                    headers_written.add(split_name)
                    batch_counts[split_id] += 1
                    split_rows[split_id] += len(idx)
                    split_positives[split_id] += int(y[idx].sum())
            row_start += len(chunk)
            print(
                f"Lượt 2: {row_start:,}/{total_rows:,} dòng | "
                f"{elapsed_text(time.perf_counter()-started)}",
                flush=True,
            )

    if row_start != total_rows:
        raise RuntimeError(f"Số dòng đổi giữa hai lượt: {total_rows:,} và {row_start:,}.")
    summary = []
    for split_id, split_name in SPLITS.items():
        n_rows = int(split_rows[split_id])
        n_positive = int(split_positives[split_id])
        summary.append({
            "split": split_name, "rows": n_rows, "takeover_true": n_positive,
            "takeover_rate": n_positive / n_rows if n_rows else 0.0,
            "batches": int(batch_counts[split_id]), "file": split_paths[split_name].name,
        })
    pd.DataFrame(summary).to_csv(C.TABLE_DIR / "02_split_summary.csv", index=False, encoding="utf-8-sig")
    schema = {
        "target": C.TARGET,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_columns": CATEGORY_COLUMNS,
        "category_levels": category_levels,
        "one_hot_features": one_hot_names,
        "feature_columns": feature_names,
        "max_categories_per_column": MAX_CATEGORIES_PER_COLUMN,
        "split_method": "random row split; positive labels apportioned to all splits",
        "history_rule": "past events in timestamp order only",
        "history_category_policy": "history-change features use the train category vocabulary; unseen values map to OTHER",
        "raw_id_policy": "index excluded; User ID used only as history key",
    }
    (C.TABLE_DIR / "02_feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(f"Hoàn tất: {total_rows:,} dòng, {len(feature_names):,} feature.")
    print(f"Chia tập: {summary}")
    print(f"Kết quả: {C.PROCESSED_DIR}; tổng thời gian {elapsed_text(time.perf_counter()-started)}")


if __name__ == "__main__":
    main()
