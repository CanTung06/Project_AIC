"""Bước 08 - Giải thích SGDClassifier bằng hệ số tuyến tính đã chuẩn hóa."""

import json
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
    if not C.SELECTED_MODEL_INFO.exists():
        raise FileNotFoundError("Hãy chạy s05 trước.")
    with C.SELECTED_MODEL_INFO.open("r", encoding="utf-8") as source:
        selected = json.load(source)
    model_path = C.UNDERSAMPLED_MODEL
    model = joblib.load(model_path)
    if isinstance(model, ThresholdedClassifier):
        model = model.estimator
    classifier = model.named_steps["classifier"]
    feature_names = load_feature_names(C.TRAIN_DATA)
    coefficients = classifier.coef_.ravel()
    if len(feature_names) != len(coefficients):
        raise ValueError("Số feature không khớp số hệ số của model.")

    table = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
        "absolute_coefficient": np.abs(coefficients),
        "direction": np.where(coefficients >= 0, "higher_attack_score", "lower_attack_score"),
    }).sort_values("absolute_coefficient", ascending=False)
    table.insert(0, "rank", range(1, len(table) + 1))
    table.to_csv(C.TABLES_DIR / "08_sgd_feature_coefficients.csv", index=False, encoding="utf-8-sig")
    print(f"Model được giải thích: {selected['selected_model']}")
    print("Top 20 hệ số có độ lớn cao nhất (trên feature đã chuẩn hóa):")
    print(table.head(20).to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print("\nLưu ý: hệ số thể hiện liên hệ trong model, không chứng minh quan hệ nhân quả.")
    print(f"Thời gian: {elapsed_text(time.perf_counter() - started)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
