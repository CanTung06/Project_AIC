"""Bước 03 - Làm sạch số và chia train/validation/test theo từng chunk"""

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
    if not C.FILTERED_DATA.exists():
        raise FileNotFoundError(f"Chưa có {C.FILTERED_DATA}; hãy chạy s02_preprocess.py trước.")

    split_paths = {
        "train": C.TRAIN_DATA,
        "validation": C.VALIDATION_DATA,
        "test": C.TEST_DATA,
    }
    temp_paths = {name: path.with_suffix(".tmp.csv") for name, path in split_paths.items()}
    for path in temp_paths.values():
        path.unlink(missing_ok=True)
    wrote_header = {name: False for name in split_paths}
    split_counts = {name: {0: 0, 1: 0} for name in split_paths}
    attack_type_counts: dict[tuple[str, str], int] = {}
    feature_columns: list[str] | None = None
    rng = np.random.default_rng(C.SEED)
    rows_read = 0
    started = time.perf_counter()

    print("Bước 03 - Chia dữ liệu 70% / 15% / 15%")
    for chunk in pd.read_csv(
        C.FILTERED_DATA,
        chunksize=C.CHUNK_SIZE,
        encoding="utf-8-sig",
        low_memory=False,
    ):
        chunk.columns = [str(column).strip() for column in chunk.columns]
        required = {C.TARGET_COLUMN, "attack_type"}
        if not required.issubset(chunk.columns):
            raise ValueError(f"Thiếu cột bắt buộc: {sorted(required - set(chunk.columns))}")

        feature_columns = [
            column for column in chunk.columns
            if column != C.TARGET_COLUMN and column not in C.METADATA_COLUMNS
        ]
        for column in feature_columns:
            values = pd.to_numeric(chunk[column], errors="coerce")
            chunk[column] = values.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
        chunk[C.TARGET_COLUMN] = chunk[C.TARGET_COLUMN].astype(np.int8)
        chunk["attack_type"] = chunk["attack_type"].astype("string").fillna("UNKNOWN")

        random_values = rng.random(len(chunk))
        train_mask = random_values < C.TRAIN_FRACTION
        validation_mask = (random_values >= C.TRAIN_FRACTION) & (
            random_values < C.TRAIN_FRACTION + C.VALIDATION_FRACTION
        )
        masks = {"train": train_mask, "validation": validation_mask, "test": ~(
            train_mask | validation_mask
        )}

        for split_name, mask in masks.items():
            subset = chunk.loc[mask]
            if subset.empty:
                continue
            for class_id, count in subset[C.TARGET_COLUMN].value_counts().items():
                split_counts[split_name][int(class_id)] += int(count)
            for (attack_type, class_id), count in subset.groupby(
                ["attack_type", C.TARGET_COLUMN], observed=True
            ).size().items():
                key = (split_name, str(attack_type))
                attack_type_counts[key] = attack_type_counts.get(key, 0) + int(count)

            subset.to_csv(
                temp_paths[split_name],
                index=False,
                mode="a" if wrote_header[split_name] else "w",
                header=not wrote_header[split_name],
                encoding="utf-8" if wrote_header[split_name] else "utf-8-sig",
            )
            wrote_header[split_name] = True

        rows_read += len(chunk)
        print(
            f"Đã chia {rows_read:,} dòng | "
            f"{elapsed_text(time.perf_counter() - started)}",
            end="\r",
            flush=True,
        )

    if not feature_columns:
        raise ValueError("Dataset tiền xử lý không có dòng hoặc không có feature.")
    if not all(wrote_header.values()):
        raise RuntimeError("Có tập split không có dữ liệu; không thay thế các file split cũ.")
    for split_name, path in split_paths.items():
        temp_paths[split_name].replace(path)

    split_rows = []
    type_rows = []
    for split_name, counts in split_counts.items():
        total = counts[0] + counts[1]
        split_rows.append({
            "split": split_name,
            "row_count": total,
            "normal_count": counts[0],
            "attack_count": counts[1],
            "attack_percentage": 100 * counts[1] / max(total, 1),
        })
        for (which_split, attack_type), count in attack_type_counts.items():
            if which_split == split_name:
                type_rows.append({"split": split_name, "attack_type": attack_type, "sample_count": count})

    split_df = pd.DataFrame(split_rows)
    type_df = pd.DataFrame(type_rows).sort_values(["split", "sample_count"], ascending=[True, False])
    split_df.to_csv(C.TABLES_DIR / "03_split_summary.csv", index=False, encoding="utf-8-sig")
    type_df.to_csv(C.TABLES_DIR / "03_attack_type_by_split.csv", index=False, encoding="utf-8-sig")

    schema = {
        "feature_columns": feature_columns,
        "metadata_columns": C.METADATA_COLUMNS,
        "target_column": C.TARGET_COLUMN,
        "split_fractions": {
            "train": C.TRAIN_FRACTION,
            "validation": C.VALIDATION_FRACTION,
            "test": C.TEST_FRACTION,
        },
        "random_seed": C.SEED,
    }
    with (C.TABLES_DIR / "03_feature_schema.json").open("w", encoding="utf-8") as output_file:
        json.dump(schema, output_file, ensure_ascii=False, indent=2)

    print("\n\nPhân bố các tập:")
    print(split_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    with (C.METRICS_DIR / "03_split_summary.json").open("w", encoding="utf-8") as output_file:
        json.dump({
            "input_rows": rows_read,
            "split_counts": split_counts,
            "split_fractions": schema["split_fractions"],
            "feature_count": len(feature_columns),
            "feature_columns": feature_columns,
            "metadata_columns": C.METADATA_COLUMNS,
            "target_column": C.TARGET_COLUMN,
            "random_seed": C.SEED,
        }, output_file, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu train, validation, test trong: {C.PROCESSED_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
