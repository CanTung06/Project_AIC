"""Bước 10 - So sánh phân phối feature giữa train và test."""

import numpy as np
import pandas as pd
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ids import config as C
from ids.common import elapsed_text, load_feature_names


def compute_statistics(path, feature_names):
    count = np.zeros(len(feature_names), dtype=np.int64)
    sums = np.zeros(len(feature_names), dtype=np.float64)
    sums_squared = np.zeros(len(feature_names), dtype=np.float64)
    minimum = np.full(len(feature_names), np.inf)
    maximum = np.full(len(feature_names), -np.inf)

    for chunk in pd.read_csv(
        path,
        usecols=feature_names,
        dtype={column: np.float32 for column in feature_names},
        chunksize=C.CHUNK_SIZE,
        encoding="utf-8-sig",
    ):
        values = chunk.to_numpy(dtype=np.float64, copy=False)
        finite = np.isfinite(values)
        valid_counts = finite.sum(axis=0)
        safe_values = np.where(finite, values, 0.0)
        count += valid_counts
        sums += safe_values.sum(axis=0)
        sums_squared += np.square(safe_values).sum(axis=0)
        for index in range(values.shape[1]):
            column_values = values[finite[:, index], index]
            if column_values.size:
                minimum[index] = min(minimum[index], float(column_values.min()))
                maximum[index] = max(maximum[index], float(column_values.max()))

    means = sums / np.maximum(count, 1)
    variances = np.maximum(sums_squared / np.maximum(count, 1) - np.square(means), 0.0)
    return count, means, np.sqrt(variances), minimum, maximum


def main() -> int:
    C.ensure_dirs()
    started = time.perf_counter()
    C.ensure_dirs()
    feature_names = load_feature_names(C.TRAIN_DATA)
    train_count, train_mean, train_std, train_min, train_max = compute_statistics(C.TRAIN_DATA, feature_names)
    test_count, test_mean, test_std, test_min, test_max = compute_statistics(C.TEST_DATA, feature_names)
    standardized_shift = np.abs(test_mean - train_mean) / np.maximum(train_std, 1e-12)

    table = pd.DataFrame({
        "feature": feature_names,
        "train_mean": train_mean,
        "test_mean": test_mean,
        "standardized_mean_shift": standardized_shift,
        "train_std": train_std,
        "test_std": test_std,
        "train_min": train_min,
        "train_max": train_max,
        "test_min": test_min,
        "test_max": test_max,
        "train_valid_count": train_count,
        "test_valid_count": test_count,
    }).sort_values("standardized_mean_shift", ascending=False)
    table.to_csv(C.TABLES_DIR / "10_feature_drift.csv", index=False, encoding="utf-8-sig")
    print(f"Đã so sánh {len(feature_names)} feature trên train/test | {elapsed_text(time.perf_counter() - started)}")
    print("Top 20 feature có độ dịch chuyển trung bình chuẩn hóa lớn nhất:")
    print(table.head(20).to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print("\nĐây là chỉ báo dịch chuyển mô tả; chưa phải kiểm định thống kê hay bằng chứng model giảm chất lượng.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
