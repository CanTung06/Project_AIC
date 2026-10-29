import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ids import config as C


def sample_training_data(features):
    """Lấy mẫu cân bằng 50-50: 250k lớp bình thường (0) và 250k lớp tấn công (1)."""
    rng = np.random.default_rng(C.SEED)
    target_per_class = getattr(C, "RF_PER_CLASS_SAMPLE_SIZE", 250_000)

    kept_x_0, kept_x_1 = [], []
    count_0 = count_1 = 0
    scanned = total_rows = 0
    started = time.perf_counter()

    for chunk in pd.read_csv(
        C.TRAIN,
        usecols=[*features, C.TARGET],
        chunksize=C.CHUNK_SIZE,
        low_memory=False,
    ):
        chunk_0 = chunk[chunk[C.TARGET] == 0]
        chunk_1 = chunk[chunk[C.TARGET] == 1]

        # Thu thập lớp 0 (Bình thường) cho đủ 250,000
        if count_0 < target_per_class and len(chunk_0) > 0:
            take_0 = min(len(chunk_0), target_per_class - count_0)
            kept_x_0.append(chunk_0[features].iloc[:take_0].to_numpy(dtype=np.float32))
            count_0 += take_0

        # Thu thập lớp 1 (IP Tấn công) cho đủ 250,000
        if count_1 < target_per_class and len(chunk_1) > 0:
            take_1 = min(len(chunk_1), target_per_class - count_1)
            kept_x_1.append(chunk_1[features].iloc[:take_1].to_numpy(dtype=np.float32))
            count_1 += take_1

        scanned += len(chunk)
        total_rows += len(chunk)
        print(
            f"Train cân bằng 50-50: Đã quét {scanned:,} dòng | "
            f"Lớp 0: {count_0:,}/{target_per_class:,} | "
            f"Lớp 1: {count_1:,}/{target_per_class:,} | "
            f"{C.elapsed_text(time.perf_counter() - started)}",
            flush=True,
        )

        # Đã gom đủ 250,000 cho cả hai lớp thì dừng quét để tối ưu thời gian I/O
        if count_0 >= target_per_class and count_1 >= target_per_class:
            break

    if count_0 == 0 or count_1 == 0:
        raise ValueError("Không đủ mẫu cho cả 2 lớp trong tập train.")

    x_all = np.concatenate(kept_x_0 + kept_x_1, axis=0)
    y_all = np.concatenate([np.zeros(count_0, dtype=np.uint8), np.ones(count_1, dtype=np.uint8)], axis=0)

    shuffle_indices = rng.permutation(len(y_all))
    return x_all[shuffle_indices], y_all[shuffle_indices], total_rows


def main():
    C.ensure_dirs()
    schema = json.loads(C.FEATURE_SCHEMA.read_text(encoding="utf-8"))
    features = schema["feature_columns"]
    x_train, y_train, train_rows_scanned = sample_training_data(features)

    model = RandomForestClassifier(
        n_estimators=C.RF_N_ESTIMATORS,
        max_depth=C.RF_MAX_DEPTH,
        min_samples_leaf=C.RF_MIN_SAMPLES_LEAF,
        max_features="sqrt",
        class_weight="balanced_subsample",
        bootstrap=True,
        max_samples=C.RF_MAX_SAMPLES,
        n_jobs=-1,
        random_state=C.SEED,
    )

    started = time.perf_counter()
    print(
        f"Đang huấn luyện Random Forest trên {len(y_train):,} mẫu "
        f"(lớp 0: {int((y_train == 0).sum()):,}, lớp 1: {int(y_train.sum()):,})...",
        flush=True,
    )
    model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - started

    joblib.dump(
        {
            "model": model,
            "feature_columns": features,
            "target_column": C.TARGET,
            "target_name": C.RAW_LABEL,
            "seed": C.SEED,
        },
        C.SUPERVISED_MODEL,
        compress=3,
    )

    summary = {
        "model": "RandomForestClassifier",
        "learning_type": "supervised",
        "target": C.RAW_LABEL,
        "training_sample_method": "Balanced Subsampling 50-50: 250,000 mẫu lớp 0 và 250,000 mẫu lớp 1 để triệt tiêu độ lệch lớp.",
        "train_rows_scanned": int(train_rows_scanned),
        "train_rows_sampled": int(len(y_train)),
        "class_0_rows_sampled": int(np.count_nonzero(y_train == 0)),
        "class_1_rows_sampled": int(np.count_nonzero(y_train == 1)),
        "class_1_rate_sampled": float(np.mean(y_train)),
        "feature_count": len(features),
        "class_weight": "balanced_subsample",
        "hyperparameters": {
            "n_estimators": C.RF_N_ESTIMATORS,
            "max_depth": C.RF_MAX_DEPTH,
            "min_samples_leaf": C.RF_MIN_SAMPLES_LEAF,
            "max_features": "sqrt",
            "max_samples_per_tree": C.RF_MAX_SAMPLES,
            "bootstrap": True,
            "n_jobs": -1,
        },
        "fit_seconds": fit_seconds,
        "model_file": str(C.SUPERVISED_MODEL),
        "note": "Mô hình học nhãn Is Attack IP; đây là đối chứng có giám sát với Isolation Forest không giám sát.",
    }
    C.SUPERVISED_TRAINING_SUMMARY.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nHuấn luyện Random Forest hoàn tất")
    print(
        f"Train: {len(y_train):,} mẫu; lớp 0={int((y_train == 0).sum()):,}; "
        f"lớp 1={int(y_train.sum()):,} ({np.mean(y_train):.2%})"
    )
    print(f"Class weight: balanced_subsample | Thời gian fit: {C.elapsed_text(fit_seconds)}")
    print(f"Model: {C.SUPERVISED_MODEL}")
    print(f"Metrics: {C.SUPERVISED_TRAINING_SUMMARY}")


if __name__ == "__main__":
    main()
