"""Thử một lưới nhỏ XGBoost trên train undersample; validation giữ nguyên."""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ids import config as C
from ids.metrics import binary_metrics, choose_validation_threshold


def load_training_sample(feature_columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Đọc train theo chunk và lấy reservoir âm đều trên toàn file."""
    rng = np.random.default_rng(C.SEED)
    reservoir_x = np.empty((0, len(feature_columns)), dtype=np.float32)
    reservoir_keys = np.empty(0, dtype=np.float64)
    positive_blocks = []
    scanned = 0
    started = time.perf_counter()

    for frame in pd.read_csv(
        C.TRAIN_FILE,
        chunksize=C.CSV_CHUNK_SIZE,
        dtype=np.float32,
        low_memory=False,
    ):
        y = frame[C.TARGET].to_numpy(dtype=np.uint8, copy=False)
        x = frame[feature_columns].to_numpy(dtype=np.float32, copy=False)
        positive = y == 1
        if positive.any():
            positive_blocks.append(x[positive].copy())

        negative = ~positive
        if negative.any():
            negative_x = x[negative]
            keys = rng.random(len(negative_x))
            combined_x = np.concatenate((reservoir_x, negative_x), axis=0)
            combined_keys = np.concatenate((reservoir_keys, keys))
            keep = min(C.MAX_NEGATIVE_TRAIN_ROWS, len(combined_keys))
            if len(combined_keys) > keep:
                selected = np.argpartition(combined_keys, keep - 1)[:keep]
                reservoir_x = combined_x[selected].copy()
                reservoir_keys = combined_keys[selected].copy()
            else:
                reservoir_x, reservoir_keys = combined_x, combined_keys

        scanned += len(frame)
        elapsed = time.perf_counter() - started
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        print(
            f"Đã lấy mẫu train: {scanned:,} dòng | "
            f"Thời gian: {hours:02d}:{minutes:02d}:{seconds:02d}",
            flush=True,
        )
        del frame, x

    if not positive_blocks:
        raise ValueError("Tập train không có mẫu takeover.")
    positives = np.concatenate(positive_blocks, axis=0)
    if len(positives) < 3:
        raise ValueError("Train có dưới 3 mẫu takeover; không đủ để huấn luyện đáng tin cậy.")
    return reservoir_x, positives


def predict_in_chunks(model, path: Path, feature_columns: list[str]):
    labels = []
    scores = []
    rows = 0
    started = time.perf_counter()
    for frame in pd.read_csv(path, chunksize=C.CSV_CHUNK_SIZE, dtype=np.float32):
        labels.append(frame[C.TARGET].to_numpy(dtype=np.uint8, copy=True))
        matrix = frame[feature_columns].to_numpy(dtype=np.float32, copy=False)
        scores.append(model.predict_proba(matrix)[:, 1].astype(np.float32))
        rows += len(frame)
        elapsed = time.perf_counter() - started
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)
        print(f"Validation: {rows:,} dòng | thời gian {h:02d}:{m:02d}:{s:02d}", flush=True)
    return np.concatenate(labels), np.concatenate(scores), time.perf_counter() - started


def main() -> None:
    C.ensure_dirs()
    print(f"XGBoost {xgboost.__version__}; early stopping dùng holdout tách từ train.")
    for required in (C.TRAIN_FILE, C.VALIDATION_FILE):
        if not required.exists():
            raise FileNotFoundError(f"Thiếu {required}; chạy bước s02_preprocess trước.")

    header = pd.read_csv(C.TRAIN_FILE, nrows=0).columns.tolist()
    feature_columns = [column for column in header if column != C.TARGET]
    if not feature_columns:
        raise ValueError("Không tìm thấy feature trong train.csv.")
    reservoir_x, positives = load_training_sample(feature_columns)
    negative_pool_size = len(reservoir_x)
    if not negative_pool_size:
        raise ValueError("Train không có mẫu âm.")

    y_val = None
    trials = []
    best = None
    trial_number = 0
    for trial_config in C.TRIAL_CONFIGS:
        trial_number += 1
        ratio = trial_config["ratio"]
        multiplier = trial_config["weight_multiplier"]
        n_negative = min(negative_pool_size, ratio * len(positives))
        rng = np.random.default_rng(C.SEED + trial_number)
        selected = rng.choice(negative_pool_size, size=n_negative, replace=False)
        x_train = np.concatenate((positives, reservoir_x[selected]), axis=0)
        y_train = np.concatenate((
            np.ones(len(positives), dtype=np.uint8),
            np.zeros(n_negative, dtype=np.uint8),
        ))
        order = rng.permutation(len(y_train))
        x_train, y_train = x_train[order], y_train[order]
        x_fit, x_stop, y_fit, y_stop = train_test_split(
            x_train, y_train, test_size=0.20,
            random_state=C.SEED + trial_number, stratify=y_train,
        )

        # Trọng số dựa trên tỷ lệ train sau undersampling.
        weight = (n_negative / len(positives)) * multiplier
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_estimators=trial_config["n_estimators"],
            learning_rate=trial_config["learning_rate"],
            max_depth=trial_config["max_depth"],
            min_child_weight=trial_config["min_child_weight"],
            subsample=trial_config["subsample"],
            colsample_bytree=trial_config["colsample_bytree"],
            reg_alpha=trial_config["reg_alpha"],
            reg_lambda=trial_config["reg_lambda"],
            scale_pos_weight=weight,
            early_stopping_rounds=30,
            n_jobs=4,
            random_state=C.SEED + trial_number,
            verbosity=0,
        )
        started = time.perf_counter()
        model.fit(x_fit, y_fit, eval_set=[(x_stop, y_stop)], verbose=False)
        fit_seconds = time.perf_counter() - started

        # Validation nguyên vẹn, theo chunk; không cân bằng nhân tạo.
        y_val, val_scores, predict_seconds = predict_in_chunks(
            model, C.VALIDATION_FILE, feature_columns
        )
        threshold, val_metrics = choose_validation_threshold(y_val, val_scores)
        record = {
            "trial": trial_number,
            **trial_config,
            "negative_rows": n_negative,
            "positive_rows": len(positives),
            "scale_pos_weight": weight,
            "best_iteration": int(model.best_iteration),
            "fit_seconds": fit_seconds,
            "validation_predict_seconds": predict_seconds,
            **val_metrics,
        }
        trials.append(record)
        print(
            f"Thử nghiệm {trial_number}: tỷ lệ âm:dương={ratio}:1, "
            f"weight={weight:.3g}, depth={trial_config['max_depth']}, "
            f"P={val_metrics['precision']:.4f}, R={val_metrics['recall']:.4f}, "
            f"F1={val_metrics['f1']:.4f}, FP={val_metrics['fp']:,}",
            flush=True,
        )

        key = (
            int(val_metrics["validation_target_met"]),
            -val_metrics["fp"] if val_metrics["validation_target_met"] else val_metrics["f1"],
            val_metrics["recall"],
            -val_metrics["fp"],
        )
        if best is None or key > best["key"]:
            best = {"key": key, "model": model, "record": record, "threshold": threshold}
        # Giữ lại kết quả đã hoàn thành nếu một thử nghiệm sau bị gián đoạn.
        pd.DataFrame(trials).to_csv(C.TRIALS_TABLE_FILE, index=False, encoding="utf-8-sig")
        C.MODEL_RESULTS_FILE.write_text(
            json.dumps({"selected_trial_so_far": best["record"], "trials": trials}, indent=2),
            encoding="utf-8",
        )

    assert best is not None
    bundle = {
        "model": best["model"],
        "feature_columns": feature_columns,
        "threshold": best["threshold"],
        "selected_validation_metrics": best["record"],
        "xgboost_version": __import__("xgboost").__version__,
    }
    joblib.dump(bundle, C.MODEL_FILE, compress=3)
    pd.DataFrame(trials).to_csv(C.TRIALS_TABLE_FILE, index=False, encoding="utf-8-sig")
    C.MODEL_RESULTS_FILE.write_text(
        json.dumps({"selected_trial": best["record"], "trials": trials}, indent=2),
        encoding="utf-8",
    )
    print(f"Đã lưu model: {C.MODEL_FILE}")
    print(f"Đã lưu kết quả validation: {C.MODEL_RESULTS_FILE}")
    print(f"Bảng so sánh: {C.TRIALS_TABLE_FILE}")


if __name__ == "__main__":
    main()
