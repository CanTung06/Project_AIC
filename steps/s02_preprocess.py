import json
import time

import numpy as np
import pandas as pd

from ids import config as C


INDEX_COLUMN = "index"
TIME_COLUMN = "Login Timestamp"
USER_COLUMN = "User ID"
MISSING = "<MISSING>"
CATEGORY_COLUMNS = {
    "IP Address": "ip",
    "Country": "country",
    "Browser Name and Version": "browser",
    "Device Type": "device_type",
    "ASN": "asn",
    "Region": "region",
    "OS Name and Version": "os",
}
BOOLEAN_COLUMNS = {
    "Login Successful": ("login_successful", "login_successful_missing"),
}
RTT_COLUMN = "Round-Trip Time [ms]"
REQUIRED_COLUMNS = [
    INDEX_COLUMN, TIME_COLUMN, USER_COLUMN, *CATEGORY_COLUMNS.keys(),
    *BOOLEAN_COLUMNS.keys(), RTT_COLUMN, C.RAW_LABEL,
]
SPLIT_NAMES = ("train", "validation", "test")


def normalize(series):
    values = series.astype("string").str.strip()
    missing = values.isna() | values.isin(["", "-"])
    values = values.mask(missing, MISSING)
    return values.to_numpy(dtype=object), missing.to_numpy(dtype=bool)


def parse_boolean_feature(series):
    normalized = series.fillna("").astype("string").str.strip().str.lower()
    missing = normalized.isin(["", "-"])
    mapped = normalized.map({"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0})
    invalid = mapped.isna() & ~missing
    if invalid.any():
        examples = normalized[invalid].value_counts().head().to_dict()
        raise ValueError(f"Giá trị boolean không hợp lệ ở {series.name}: {examples}")
    return mapped.fillna(0).to_numpy(dtype=np.float32), missing.to_numpy(dtype=np.float32)


def local_prior_counts(local_codes, base_counts, keys, state):
    """Tính số lần trước đó trong chunk và cập nhật trạng thái toàn cục."""
    if not len(local_codes):
        return np.empty(0, dtype=np.int64)
    order = np.argsort(local_codes, kind="stable")
    sorted_codes = local_codes[order]
    starts = np.r_[0, np.flatnonzero(sorted_codes[1:] != sorted_codes[:-1]) + 1]
    group_sizes = np.diff(np.r_[starts, len(local_codes)])
    within_group = np.arange(len(local_codes), dtype=np.int64) - np.repeat(starts, group_sizes)
    prior = np.empty(len(local_codes), dtype=np.int64)
    prior[order] = base_counts[local_codes[order]] + within_group
    counts = np.bincount(local_codes, minlength=len(keys))
    for code, key in enumerate(keys):
        state[key] = int(base_counts[code] + counts[code])
    return prior


def category_prior_counts(category_codes, state_counts):
    """Tính số lần trước đó theo category bằng mảng đếm gọn, không dùng nhãn."""
    codes = np.asarray(category_codes, dtype=np.int64)
    if not len(codes):
        return np.empty(0, dtype=np.uint64)

    max_code = int(codes.max())
    if max_code >= len(state_counts):
        new_size = max(max_code + 1, len(state_counts) * 2)
        expanded = np.zeros(new_size, dtype=np.uint64)
        expanded[: len(state_counts)] = state_counts
        state_counts = expanded

    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    starts = np.r_[0, np.flatnonzero(sorted_codes[1:] != sorted_codes[:-1]) + 1]
    group_sizes = np.diff(np.r_[starts, len(codes)])
    within_group = np.arange(len(codes), dtype=np.int64) - np.repeat(starts, group_sizes)
    prior = np.empty(len(codes), dtype=np.uint64)
    prior[order] = state_counts[sorted_codes] + within_group.astype(np.uint64)
    state_counts[sorted_codes[starts]] += group_sizes.astype(np.uint64)
    return prior, state_counts


class UserHistory:
    def __init__(self):
        self.user_ids = {}
        self.next_user_code = 1
        self.category_ids = {column: {} for column in CATEGORY_COLUMNS}
        self.next_category_code = {column: 1 for column in CATEGORY_COLUMNS}
        self.user_counts = {}
        self.user_category_counts = {column: {} for column in CATEGORY_COLUMNS}
        self.global_category_counts = {
            column: np.zeros(1024, dtype=np.uint64)
            for column in ("IP Address", "ASN")
        }
        self.rows_seen = 0

    def encode_users(self, values, missing_mask):
        local_codes, unique_values = pd.factorize(values, sort=False)
        unique_codes = np.zeros(len(unique_values), dtype=np.uint64)
        for position, value in enumerate(unique_values):
            if value == MISSING:
                continue
            key = str(value)
            if key not in self.user_ids:
                self.user_ids[key] = self.next_user_code
                self.next_user_code += 1
            unique_codes[position] = self.user_ids[key]
        encoded = unique_codes[local_codes]
        missing_positions = np.flatnonzero(missing_mask)
        if len(missing_positions):
            encoded[missing_positions] = np.arange(
                self.next_user_code,
                self.next_user_code + len(missing_positions),
                dtype=np.uint64,
            )
            self.next_user_code += len(missing_positions)
        return encoded

    def encode_categories(self, column, values):
        local_codes, unique_values = pd.factorize(values, sort=False)
        unique_codes = np.zeros(len(unique_values), dtype=np.uint64)
        mapping = self.category_ids[column]
        next_code = self.next_category_code[column]
        for position, value in enumerate(unique_values):
            if value == MISSING:
                continue
            key = str(value)
            if key not in mapping:
                mapping[key] = next_code
                next_code += 1
            unique_codes[position] = mapping[key]
        self.next_category_code[column] = next_code
        return unique_codes[local_codes]

    def transform(self, chunk):
        timestamp = pd.to_datetime(chunk[TIME_COLUMN], errors="coerce", utc=True)
        hour = timestamp.dt.hour + timestamp.dt.minute / 60.0
        weekday = timestamp.dt.dayofweek
        month = timestamp.dt.month - 1
        hour = hour.fillna(0).to_numpy(dtype=np.float32)
        weekday = weekday.fillna(0).to_numpy(dtype=np.float32)
        month = month.fillna(0).to_numpy(dtype=np.float32)

        users, missing_users = normalize(chunk[USER_COLUMN])
        user_codes = self.encode_users(users, missing_users)
        user_local, user_keys_raw = pd.factorize(user_codes, sort=False)
        user_keys = [int(key) for key in user_keys_raw]
        user_base = np.fromiter(
            (self.user_counts.get(key, 0) for key in user_keys),
            dtype=np.int64,
            count=len(user_keys),
        )
        user_prior = local_prior_counts(
            user_local, user_base, user_keys, self.user_counts
        )

        features = [
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * weekday / 7),
            np.cos(2 * np.pi * weekday / 7),
            np.sin(2 * np.pi * month / 12),
            np.cos(2 * np.pi * month / 12),
            (weekday >= 5).astype(np.float32),
            np.log1p(user_prior),
        ]

        login_successful_val = None
        for column in BOOLEAN_COLUMNS:
            values, missing = parse_boolean_feature(chunk[column])
            features.extend([values, missing])
            if column == "Login Successful":
                login_successful_val = values

        rtt = pd.to_numeric(chunk[RTT_COLUMN], errors="coerce").to_numpy(dtype=np.float32)
        rtt_missing = ~np.isfinite(rtt)
        rtt = np.nan_to_num(rtt, nan=0.0, posinf=0.0, neginf=0.0)
        rtt = np.log1p(np.maximum(rtt, 0.0))
        features.extend([rtt, rtt_missing.astype(np.float32)])

        user_freq_by_category = {}
        for column, prefix in CATEGORY_COLUMNS.items():
            values, _ = normalize(chunk[column])
            category_codes = self.encode_categories(column, values)
            if column in self.global_category_counts:
                global_prior, self.global_category_counts[column] = category_prior_counts(
                    category_codes, self.global_category_counts[column]
                )
                prior_row_numbers = self.rows_seen + np.arange(len(chunk), dtype=np.int64)
                global_frequency = global_prior / np.maximum(prior_row_numbers, 1)
                features.extend([
                    np.log1p(global_prior),
                    global_frequency.astype(np.float32, copy=False),
                ])

            pair_local, pair_values = pd.factorize(
                pd.MultiIndex.from_arrays([user_codes, category_codes]), sort=False
            )
            pair_keys = [(int(user), int(category)) for user, category in pair_values]
            state = self.user_category_counts[column]
            pair_base = np.fromiter(
                (state.get(key, 0) for key in pair_keys),
                dtype=np.int64,
                count=len(pair_keys),
            )
            prior_count = local_prior_counts(pair_local, pair_base, pair_keys, state)
            frequency = prior_count / np.maximum(user_prior, 1)
            features.extend([np.log1p(prior_count), frequency])
            user_freq_by_category[column] = frequency

        # 3 Đặc trưng tương tác mới bảo vệ người dùng lành tính
        ip_freq = user_freq_by_category["IP Address"]
        browser_freq = user_freq_by_category["Browser Name and Version"]
        device_type_freq = user_freq_by_category["Device Type"]
        os_freq = user_freq_by_category["OS Name and Version"]

        device_familiarity = ((browser_freq + device_type_freq + os_freq) / 3.0).astype(np.float32, copy=False)
        network_device_mismatch = ((1.0 - ip_freq) * device_familiarity).astype(np.float32, copy=False)
        suspicious_failure = ((1.0 - login_successful_val) * (1.0 - ip_freq)).astype(np.float32, copy=False)

        features.extend([
            device_familiarity,
            network_device_mismatch,
            suspicious_failure,
        ])

        self.rows_seen += len(chunk)
        return np.column_stack(features).astype(np.float32, copy=False)


def main():
    C.ensure_dirs()
    if not C.RAW_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy dataset: {C.RAW_FILE}")
    eda_file = C.TABLE_DIR / "01_eda_summary.json"
    if not eda_file.exists():
        raise FileNotFoundError("Hãy chạy bước 01 EDA trước.")
    eda = json.loads(eda_file.read_text(encoding="utf-8"))
    expected_rows = int(eda["rows"])

    header = pd.read_csv(C.RAW_FILE, nrows=0).columns.tolist()
    missing = sorted(set(REQUIRED_COLUMNS) - set(header))
    if missing:
        raise ValueError(f"Dataset thiếu các cột bắt buộc: {missing}")

    paths = {"train": C.TRAIN, "validation": C.VALIDATION, "test": C.TEST}
    for path in paths.values():
        if path.exists():
            path.unlink()

    history = UserHistory()
    seen_headers = set()
    row_counts = {name: 0 for name in SPLIT_NAMES}
    anomaly_counts = {name: 0 for name in SPLIT_NAMES}
    scanned = 0
    started = time.perf_counter()

    for chunk in pd.read_csv(
        C.RAW_FILE,
        usecols=REQUIRED_COLUMNS,
        chunksize=C.CHUNK_SIZE,
        low_memory=False,
    ):
        source_index = pd.to_numeric(chunk[INDEX_COLUMN], errors="raise").to_numpy(dtype=np.int64)
        labels = C.parse_label(chunk[C.RAW_LABEL])
        features = history.transform(chunk)

        output = pd.DataFrame(features, columns=C.FEATURES)
        output.insert(0, C.SOURCE_INDEX, source_index)
        output[C.TARGET] = labels
        split_codes = C.split_for_source_index(source_index)

        for split_code, split_name in enumerate(SPLIT_NAMES):
            mask = split_codes == split_code
            if not np.any(mask):
                continue
            part = output.loc[mask]
            part.to_csv(
                paths[split_name],
                mode="a",
                index=False,
                header=split_name not in seen_headers,
                encoding="utf-8",
            )
            seen_headers.add(split_name)
            row_counts[split_name] += len(part)
            anomaly_counts[split_name] += int(part[C.TARGET].sum())

        scanned += len(chunk)
        print(
            f"Đã xử lý {scanned:,} / {expected_rows:,} dòng | "
            f"{C.elapsed_text(time.perf_counter() - started)}",
            flush=True,
        )

    if scanned != expected_rows:
        raise ValueError(f"EDA ghi nhận {expected_rows:,} dòng nhưng preprocess đọc {scanned:,}.")
    total_anomalies = sum(anomaly_counts.values())
    expected_anomalies = int(eda["label_counts"]["attack_ip_1"])
    if total_anomalies != expected_anomalies:
        raise ValueError(f"Nhãn Is Attack IP không khớp: EDA={expected_anomalies:,}, preprocess={total_anomalies:,}.")

    split_rows = []
    for split_name in SPLIT_NAMES:
        normal_count = row_counts[split_name] - anomaly_counts[split_name]
        split_rows.append({
            "split": split_name,
            "rows": row_counts[split_name],
            "normal_rows": normal_count,
            "attack_ip_rows": anomaly_counts[split_name],
            "attack_ip_rate": anomaly_counts[split_name] / row_counts[split_name] if row_counts[split_name] else 0.0,
            "file": paths[split_name].name,
        })
    pd.DataFrame(split_rows).to_csv(C.SPLIT_SUMMARY, index=False, encoding="utf-8")

    schema = {
        "target_column": C.TARGET,
        "source_label": C.RAW_LABEL,
        "feature_columns": C.FEATURES,
        "target_meaning": "0 = Is Attack IP False; 1 = Is Attack IP True",
        "split": "Phân hoạch giả ngẫu nhiên ổn định 70/15/15 theo source_index; giữ nguyên toàn bộ tỷ lệ lớp.",
        "history": "Tần suất IP, quốc gia, trình duyệt, thiết bị, ASN, region và OS được tính từ các lần đăng nhập trước của cùng user; số lần IP và ASN xuất hiện toàn cục chỉ tính từ các dòng đứng trước.",
        "label_usage": "Feature lịch sử IP và ASN không dùng nhãn Is Attack IP; chỉ s05 mới huấn luyện theo nhãn.",
        "additional_signals": ["Login Successful", "Round-Trip Time [ms]"],
        "prediction_timing": "Có thể chấm điểm trong hoặc ngay sau lần đăng nhập; Login Successful chỉ có sau khi có kết quả đăng nhập.",
        "excluded_from_features": [C.SOURCE_INDEX, C.TARGET, USER_COLUMN, C.RAW_LABEL, "Is Account Takeover"],
    }
    C.FEATURE_SCHEMA.write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nTiền xử lý hoàn tất")
    print(pd.DataFrame(split_rows).to_string(index=False))
    print(f"Thời gian: {C.elapsed_text(time.perf_counter() - started)}")
    print(f"Feature schema: {C.FEATURE_SCHEMA}")


if __name__ == "__main__":
    main()
