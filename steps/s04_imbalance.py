"""Bước 04 - Xử lý mất cân bằng bằng class weight và random undersampling.

Tạo class weight cho SGDClassifier trên toàn bộ train và một bản train undersampled
với đúng 3 mẫu bình thường cho mỗi mẫu tấn công. Validation/test không bị lấy mẫu lại.
Chạy: python steps/s04_imbalance.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ids import config as C
from ids.common import elapsed_text


def main() -> int:
    C.ensure_dirs()
    if not C.TRAIN_DATA.exists():
        raise FileNotFoundError(f"Chưa có {C.TRAIN_DATA}; hãy chạy s03_split_data.py trước.")

    class_counts = {0: 0, 1: 0}
    for chunk in pd.read_csv(
        C.TRAIN_DATA,
        usecols=[C.TARGET_COLUMN],
        chunksize=C.CHUNK_SIZE,
        encoding="utf-8-sig",
    ):
        values = chunk[C.TARGET_COLUMN].value_counts()
        for class_id, count in values.items():
            class_counts[int(class_id)] += int(count)

    positive_count = class_counts[1]
    negative_count = class_counts[0]
    if positive_count == 0 or negative_count == 0:
        raise ValueError(f"Train phải có cả hai lớp; số lượng hiện tại: {class_counts}")

    total = positive_count + negative_count
    weights = {
        "0": total / (2 * negative_count),
        "1": total / (2 * positive_count),
    }
    requested_negative_count = min(
        negative_count,
        positive_count * C.UNDERSAMPLE_NEGATIVE_PER_POSITIVE,
    )
    temporary_path = C.UNDERSAMPLED_TRAIN_DATA.with_suffix(".tmp.csv")
    temporary_path.unlink(missing_ok=True)
    wrote_header = False
    sampled_counts = {0: 0, 1: 0}
    rng = np.random.default_rng(C.SEED)
    sampled_negative_positions = np.sort(
        rng.choice(negative_count, size=requested_negative_count, replace=False)
    )
    negative_rows_seen = 0
    rows_read = 0
    started = time.perf_counter()

    for chunk in pd.read_csv(
        C.TRAIN_DATA,
        chunksize=C.CHUNK_SIZE,
        encoding="utf-8-sig",
        low_memory=False,
    ):
        labels = chunk[C.TARGET_COLUMN].astype(np.int8).to_numpy()
        negative_indices = np.flatnonzero(labels == 0)
        negative_stop = negative_rows_seen + len(negative_indices)
        sample_start = np.searchsorted(sampled_negative_positions, negative_rows_seen, side="left")
        sample_stop = np.searchsorted(sampled_negative_positions, negative_stop, side="left")
        selected_negative_offsets = (
            sampled_negative_positions[sample_start:sample_stop] - negative_rows_seen
        )
        mask = labels == 1
        mask[negative_indices[selected_negative_offsets]] = True
        negative_rows_seen = negative_stop
        selected = chunk.loc[mask]
        for class_id, count in selected[C.TARGET_COLUMN].value_counts().items():
            sampled_counts[int(class_id)] += int(count)
        if not selected.empty:
            selected.to_csv(
                temporary_path,
                index=False,
                mode="a" if wrote_header else "w",
                header=not wrote_header,
                encoding="utf-8" if wrote_header else "utf-8-sig",
            )
            wrote_header = True
        rows_read += len(chunk)
        print(
            f"Đã lấy mẫu {rows_read:,} dòng train | "
            f"{elapsed_text(time.perf_counter() - started)}",
            end="\r",
            flush=True,
        )

    if not wrote_header:
        raise RuntimeError("Không tạo được train undersampled.")
    if negative_rows_seen != negative_count:
        raise RuntimeError(
            f"Số mẫu âm đã đọc ({negative_rows_seen:,}) không khớp thống kê "
            f"ban đầu ({negative_count:,})."
        )
    temporary_path.replace(C.UNDERSAMPLED_TRAIN_DATA)

    strategy_table = pd.DataFrame([
        {
            "strategy": "class_weight_balanced",
            "normal_count": negative_count,
            "attack_count": positive_count,
            "attack_to_normal_ratio": positive_count / negative_count,
            "normal_class_weight": weights["0"],
            "attack_class_weight": weights["1"],
        },
        {
            "strategy": "random_undersampling_3_to_1",
            "normal_count": sampled_counts[0],
            "attack_count": sampled_counts[1],
            "attack_to_normal_ratio": sampled_counts[1] / max(sampled_counts[0], 1),
            "normal_class_weight": 1.0,
            "attack_class_weight": 1.0,
        },
    ])
    strategy_table.to_csv(
        C.TABLES_DIR / "04_imbalance_strategies.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "class_counts_full_train": {str(key): value for key, value in class_counts.items()},
        "balanced_class_weight": weights,
        "undersampling_negative_per_positive_target": C.UNDERSAMPLE_NEGATIVE_PER_POSITIVE,
        "undersampled_counts": {str(key): value for key, value in sampled_counts.items()},
        "undersampled_train_path": str(C.UNDERSAMPLED_TRAIN_DATA),
        "validation_test_resampled": False,
    }
    with (C.METRICS_DIR / "04_imbalance_summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, ensure_ascii=False, indent=2)

    print("\n\nCác chiến lược mất cân bằng:")
    print(strategy_table.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print(f"\nĐã lưu tập train undersampled: {C.UNDERSAMPLED_TRAIN_DATA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
