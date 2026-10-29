"""Bước 05: Huấn luyện một SGDClassifier với undersampling và signed-log toàn bộ đặc trưng."""

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
from sklearn.preprocessing import FunctionTransformer, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ids import config as C
from ids.common import elapsed_text, load_xy, signed_log_transform


def build_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("signed_log", FunctionTransformer(
            signed_log_transform,
            validate=False,
            feature_names_out="one-to-one",
        )),
        ("scaler", StandardScaler()),
        ("classifier", SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=0.0001,
            l1_ratio=0.15,
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
    print("\nĐang đọc tập train undersampling...")
    X_train, y_train, feature_names = load_xy(C.UNDERSAMPLED_TRAIN_DATA)
    print(f"Train: {len(y_train):,} dòng, {len(feature_names):,} feature")

    model = build_pipeline()
    started = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started
    classifier = model.named_steps["classifier"]
    joblib.dump(model, C.UNDERSAMPLED_MODEL, compress=3)
    result = {
        "strategy": "random_undersampling_signed_log",
        "training_rows": int(len(y_train)),
        "feature_count": len(feature_names),
        "fit_seconds": round(fit_seconds, 2),
        "fit_time": elapsed_text(fit_seconds),
        "epochs_used": int(classifier.n_iter_),
        "feature_transform": "signed_log_then_standard_scaler",
        "model_path": str(C.UNDERSAMPLED_MODEL),
    }
    print(
        f"Đã lưu model: {C.UNDERSAMPLED_MODEL.name} | {fit_seconds:.1f} giây | "
        f"{classifier.n_iter_} epoch"
    )
    del X_train, y_train, model
    gc.collect()
    results = [result]

    result_table = pd.DataFrame(results)
    result_table.to_csv(C.TABLES_DIR / "05_training_summary.csv", index=False, encoding="utf-8-sig")
    with (C.METRICS_DIR / "05_training_summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(results, output_file, ensure_ascii=False, indent=2)

    print("\nTóm tắt huấn luyện:")
    print(result_table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
