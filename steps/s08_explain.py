"""Bước 08 - Giải thích SGDClassifier bằng hệ số tuyến tính đã chuẩn hóa."""

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ids import config as C
from ids.common import elapsed_text, load_feature_names
from ids.thresholded import ThresholdedClassifier


def main() -> int:
    C.ensure_dirs()
    started = time.perf_counter()
    models = (
        ("undersampling", C.UNDERSAMPLED_MODEL),
        ("full_data", C.FULL_SGD_MODEL),
    )
    available_models = [(name, path) for name, path in models if path.is_file()]
    if not available_models:
        raise FileNotFoundError("Hãy chạy s05 trước.")
    feature_names = load_feature_names(C.TRAIN_DATA)
    tables = []
    for strategy, model_path in available_models:
        saved_model = joblib.load(model_path)
        threshold = saved_model.threshold if isinstance(saved_model, ThresholdedClassifier) else None
        model = saved_model.estimator if isinstance(saved_model, ThresholdedClassifier) else saved_model
        classifier = model.named_steps["classifier"]
        coefficients = classifier.coef_.ravel()
        if len(feature_names) != len(coefficients):
            raise ValueError(f"Số feature không khớp số hệ số của model {model_path.name}.")

        table = pd.DataFrame({
            "feature": feature_names,
            "coefficient": coefficients,
            "absolute_coefficient": np.abs(coefficients),
            "direction": np.where(coefficients >= 0, "higher_attack_score", "lower_attack_score"),
        }).sort_values("absolute_coefficient", ascending=False)
        table.insert(0, "rank", range(1, len(table) + 1))
        table.insert(0, "strategy", strategy)
        tables.append(table)
        print(f"\nModel được giải thích: SGDClassifier ({strategy}) | {model_path.name}")
        if threshold is not None:
            print(f"Threshold lưu trong model: {threshold:.8f}")
        print("Top 20 hệ số có độ lớn cao nhất (trên feature đã chuẩn hóa):")
        print(table.head(20).to_string(index=False, float_format=lambda value: f"{value:.5f}"))

    combined_table = pd.concat(tables, ignore_index=True)
    combined_table.to_csv(
        C.TABLES_DIR / "08_sgd_feature_coefficients.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("\nLưu ý: hệ số thể hiện liên hệ trong model, không chứng minh quan hệ nhân quả.")
    print(f"Thời gian: {elapsed_text(time.perf_counter() - started)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
