import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ids import config as C


def sample_training_data(features):
    """Lấy ngẫu nhiên đều tối đa số mẫu quy định cho từng lớp."""
    rng = np.random.default_rng(C.SEED)
    limit = C.LR_PER_CLASS_SAMPLE_SIZE
    kept_x = {
        0: np.empty((0, len(features)), dtype=np.float32),
        1: np.empty((0, len(features)), dtype=np.float32),
    }
    kept_keys = {0: np.empty(0), 1: np.empty(0)}
    scanned = 0
    started = time.perf_counter()

    for chunk in pd.read_csv(
        C.TRAIN,
        usecols=[*features, C.TARGET],
        chunksize=C.CHUNK_SIZE,
        low_memory=False,
    ):
        labels = chunk[C.TARGET].to_numpy(dtype=np.uint8, copy=False)
        values = chunk[features].to_numpy(dtype=np.float32, copy=False)

        for label in (0, 1):
            positions = np.flatnonzero(labels == label)
            if not len(positions):
                continue
            keys = rng.random(len(positions))
            combined_x = np.concatenate((kept_x[label], values[positions]), axis=0)
            combined_keys = np.concatenate((kept_keys[label], keys))
            if len(combined_keys) > limit:
                selected = np.argpartition(combined_keys, limit - 1)[:limit]
                kept_x[label] = combined_x[selected].copy()
                kept_keys[label] = combined_keys[selected].copy()
            else:
                kept_x[label] = combined_x
                kept_keys[label] = combined_keys

        scanned += len(chunk)
        print(
            f"Train Logistic Regression: đã đọc {scanned:,} dòng; "
            f"đang giữ lớp 0={len(kept_keys[0]):,}, lớp 1={len(kept_keys[1]):,} "
            f"| {C.elapsed_text(time.perf_counter() - started)}",
            flush=True,
        )

    if not len(kept_keys[0]) or not len(kept_keys[1]):
        raise ValueError("Tập train cần có mẫu của cả lớp 0 và lớp 1.")

    x_train = np.concatenate((kept_x[0], kept_x[1]), axis=0)
    y_train = np.concatenate((
        np.zeros(len(kept_x[0]), dtype=np.uint8),
        np.ones(len(kept_x[1]), dtype=np.uint8),
    ))
    order = rng.permutation(len(y_train))
    return x_train[order], y_train[order], scanned


def main():
    C.ensure_dirs()
    schema = json.loads(C.FEATURE_SCHEMA.read_text(encoding="utf-8"))
    features = schema["feature_columns"]
    x_train, y_train, train_rows_scanned = sample_training_data(features)

    model = Pipeline([
        ("standard_scaler", StandardScaler()),
        ("logistic_regression", LogisticRegression(
            C=C.LR_C,
            solver="lbfgs",
            class_weight="balanced",
            max_iter=C.LR_MAX_ITER,
            tol=C.LR_TOL,
            random_state=C.SEED,
        )),
    ])

    started = time.perf_counter()
    print(
        f"Đang huấn luyện Logistic Regression trên {len(y_train):,} mẫu "
        f"(lớp 0={int((y_train == 0).sum()):,}, lớp 1={int(y_train.sum()):,})...",
        flush=True,
    )
    model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - started
    classifier = model.named_steps["logistic_regression"]

    joblib.dump(
        {
            "model": model,
            "feature_columns": features,
            "target_column": C.TARGET,
            "target_name": C.RAW_LABEL,
            "seed": C.SEED,
        },
        C.MODEL,
        compress=3,
    )

    summary = {
        "model": "LogisticRegression",
        "learning_type": "supervised",
        "target": C.RAW_LABEL,
        "training_sample_method": "Lấy ngẫu nhiên đều tối đa 250.000 mẫu mỗi lớp từ toàn bộ tập train.",
        "train_rows_scanned": int(train_rows_scanned),
        "train_rows_sampled": int(len(y_train)),
        "class_0_rows_sampled": int(np.count_nonzero(y_train == 0)),
        "class_1_rows_sampled": int(np.count_nonzero(y_train == 1)),
        "feature_count": len(features),
        "standardization": "StandardScaler được fit trên train.",
        "class_weight": "balanced",
        "hyperparameters": {
            "C": C.LR_C,
            "regularization": "L2 (mặc định của LogisticRegression)",
            "solver": "lbfgs",
            "max_iter": C.LR_MAX_ITER,
            "tol": C.LR_TOL,
            "n_iter_actual": int(classifier.n_iter_[0]),
        },
        "fit_seconds": fit_seconds,
        "model_file": str(C.MODEL),
        "note": "Ngưỡng phân loại sẽ được chọn theo F1 lớp 1 trên validation ở bước 04.",
    }
    C.TRAINING_SUMMARY.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nHuấn luyện Logistic Regression hoàn tất")
    print(
        f"Train: {len(y_train):,} mẫu | lớp 0={int((y_train == 0).sum()):,} | "
        f"lớp 1={int(y_train.sum()):,}"
    )
    print(
        f"Solver={classifier.solver} | Số vòng lặp={classifier.n_iter_[0]} | "
        f"Fit={C.elapsed_text(fit_seconds)}"
    )
    print(f"Model: {C.MODEL}")
    print(f"Metrics: {C.TRAINING_SUMMARY}")


if __name__ == "__main__":
    main()
