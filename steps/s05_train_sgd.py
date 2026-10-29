"""Bước 05 - Huấn luyện SGDClassifier với hai chiến lược mất cân bằng.

1) class_weight trên toàn bộ train; 2) random undersampling train theo tỷ lệ 10:1.
Validation và test không được resample. Chạy: python steps/s05_train_sgd.py
"""

import gc
import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ids import config as C
from ids.common import elapsed_text, load_xy


def build_pipeline(class_weight):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("scaler", StandardScaler()),
        ("classifier", SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=0.0001,
            l1_ratio=0.15,
            class_weight=class_weight,
            max_iter=50,
            tol=0.001,
            shuffle=True,
            average=True,
            random_state=C.SEED,
        )),
    ])


def main() -> int:
    C.ensure_dirs()
    if not C.TRAIN_DATA.exists() or not C.UNDERSAMPLED_TRAIN_DATA.exists():
        raise FileNotFoundError("Hãy chạy s03_split_data.py và s04_imbalance.py trước.")
    with (C.METRICS_DIR / "04_imbalance_summary.json").open("r", encoding="utf-8") as source:
        imbalance = json.load(source)
    class_weight = {int(key): float(value) for key, value in imbalance["balanced_class_weight"].items()}

    strategies = [
        ("class_weight", C.TRAIN_DATA, class_weight, C.WEIGHTED_MODEL),
        ("random_undersampling", C.UNDERSAMPLED_TRAIN_DATA, None, C.UNDERSAMPLED_MODEL),
    ]
    results = []
    feature_names = None

    for strategy_name, training_path, weights, model_path in strategies:
        print(f"\nĐang đọc dữ liệu cho chiến lược: {strategy_name}")
        X_train, y_train, feature_names = load_xy(training_path, feature_names)
        print(f"Train: {len(y_train):,} dòng, {len(feature_names):,} feature")

        model = build_pipeline(weights)
        started = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started
        classifier = model.named_steps["classifier"]
        joblib.dump(model, model_path, compress=3)
        result = {
            "strategy": strategy_name,
            "training_rows": int(len(y_train)),
            "feature_count": len(feature_names),
            "fit_seconds": round(fit_seconds, 2),
            "fit_time": elapsed_text(fit_seconds),
            "epochs_used": int(classifier.n_iter_),
            "class_weight": weights,
            "model_path": str(model_path),
        }
        results.append(result)
        print(
            f"Đã lưu model: {model_path.name} | {fit_seconds:.1f} giây | "
            f"{classifier.n_iter_} epoch"
        )

        del X_train, y_train, model
        gc.collect()

    result_table = pd.DataFrame(results)
    result_table.drop(columns=["class_weight"], inplace=True)
    result_table.to_csv(C.TABLES_DIR / "05_training_summary.csv", index=False, encoding="utf-8-sig")
    with (C.METRICS_DIR / "05_training_summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(results, output_file, ensure_ascii=False, indent=2)

    print("\nTóm tắt huấn luyện:")
    print(result_table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
