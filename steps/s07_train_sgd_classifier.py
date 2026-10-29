import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from ids import config as C


def main():
    C.ensure_dirs()
    if not C.TRAIN.exists():
        raise FileNotFoundError(f"Không tìm thấy tập train: {C.TRAIN}")
    schema = json.loads(C.FEATURE_SCHEMA.read_text(encoding="utf-8"))
    features = schema["feature_columns"]

    # Lượt 1: đọc toàn bộ train để tính thống kê chuẩn hóa và trọng số lớp.
    scaler = StandardScaler()
    class_counts = np.zeros(2, dtype=np.int64)
    rows_seen = 0
    started = time.perf_counter()
    print("Lượt 1/2: quét toàn bộ train để tính scaler và số lượng từng lớp...", flush=True)
    for chunk in pd.read_csv(
        C.TRAIN,
        usecols=[*features, C.TARGET],
        chunksize=C.CHUNK_SIZE,
        low_memory=False,
    ):
        x_chunk = chunk[features].to_numpy(dtype=np.float32, copy=False)
        y_chunk = chunk[C.TARGET].to_numpy(dtype=np.uint8, copy=False)
        if not np.isin(y_chunk, [0, 1]).all():
            raise ValueError("Nhãn train phải chỉ gồm 0 và 1.")
        scaler.partial_fit(x_chunk)
        class_counts += np.bincount(y_chunk, minlength=2)[:2]
        rows_seen += len(chunk)
        print(
            f"Scaler: đã đọc {rows_seen:,} dòng | "
            f"{C.elapsed_text(time.perf_counter() - started)}",
            flush=True,
        )

    if rows_seen == 0 or np.any(class_counts == 0):
        raise ValueError("Tập train rỗng hoặc thiếu một trong hai lớp.")

    total_rows = int(class_counts.sum())
    class_weights = {
        label: total_rows / (2.0 * int(class_counts[label]))
        for label in (0, 1)
    }
    classifier = SGDClassifier(
        loss="log_loss",
        alpha=C.SGD_ALPHA,
        learning_rate=C.SGD_LEARNING_RATE,
        class_weight=class_weights,
        average=True,
        shuffle=True,
        random_state=C.SEED,
    )

    # Lượt 2 trở đi: partial_fit xử lý từng chunk; mỗi epoch đi qua mọi dòng train.
    for epoch in range(1, C.SGD_EPOCHS + 1):
        epoch_started = time.perf_counter()
        rows_this_epoch = 0
        print(
            f"Lượt train {epoch}/{C.SGD_EPOCHS}: huấn luyện trên toàn bộ train...",
            flush=True,
        )
        for chunk in pd.read_csv(
            C.TRAIN,
            usecols=[*features, C.TARGET],
            chunksize=C.CHUNK_SIZE,
            low_memory=False,
        ):
            x_chunk = chunk[features].to_numpy(dtype=np.float32, copy=False)
            y_chunk = chunk[C.TARGET].to_numpy(dtype=np.uint8, copy=False)
            x_scaled = scaler.transform(x_chunk).astype(np.float32, copy=False)
            if epoch == 1 and rows_this_epoch == 0:
                classifier.partial_fit(x_scaled, y_chunk, classes=np.array([0, 1]))
            else:
                classifier.partial_fit(x_scaled, y_chunk)
            rows_this_epoch += len(chunk)
            print(
                f"Epoch {epoch}/{C.SGD_EPOCHS}: đã huấn luyện {rows_this_epoch:,} / "
                f"{total_rows:,} dòng | {C.elapsed_text(time.perf_counter() - epoch_started)}",
                flush=True,
            )
        if rows_this_epoch != total_rows:
            raise ValueError(
                f"Epoch {epoch} đọc {rows_this_epoch:,} dòng, "
                f"khác lượt scaler {total_rows:,} dòng."
            )

    joblib.dump(
        {
            "scaler": scaler,
            "model": classifier,
            "feature_columns": features,
            "target_column": C.TARGET,
            "target_name": C.RAW_LABEL,
            "seed": C.SEED,
            "threshold": None,
        },
        C.SGD_MODEL,
        compress=3,
    )

    summary = {
        "model": "SGDClassifier(log_loss)",
        "learning_type": "supervised_out_of_core",
        "target": C.RAW_LABEL,
        "training_data": "Toàn bộ tập train; đọc theo chunk, không lấy mẫu con.",
        "train_rows": total_rows,
        "class_0_rows": int(class_counts[0]),
        "class_1_rows": int(class_counts[1]),
        "class_weights": {str(key): value for key, value in class_weights.items()},
        "feature_count": len(features),
        "standardization": "StandardScaler.partial_fit trên toàn bộ train trước khi huấn luyện.",
        "hyperparameters": {
            "loss": "log_loss",
            "alpha": C.SGD_ALPHA,
            "learning_rate": C.SGD_LEARNING_RATE,
            "epochs": C.SGD_EPOCHS,
            "average": True,
            "shuffle": True,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "model_file": str(C.SGD_MODEL),
        "note": "Validation sẽ chọn ngưỡng tối đa F1 lớp 1; test chỉ dùng để đánh giá cuối cùng.",
    }
    C.SGD_TRAINING_SUMMARY.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nHuấn luyện SGDClassifier hoàn tất")
    print(
        f"Đã dùng {total_rows:,} dòng train qua {C.SGD_EPOCHS} epoch "
        f"(lớp 0={class_counts[0]:,}, lớp 1={class_counts[1]:,})"
    )
    print(f"Model: {C.SGD_MODEL}")
    print(f"Training summary: {C.SGD_TRAINING_SUMMARY}")


if __name__ == "__main__":
    main()
