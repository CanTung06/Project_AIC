"""Chấm điểm file CSV đã qua cùng quy tắc feature của bước s02."""
import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ids import config as C


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Dự đoán trên CSV đã được biến đổi theo feature schema của bước s02."
    )
    parser.add_argument("input", type=Path, help="CSV feature, gồm đúng các cột feature đã train")
    parser.add_argument("--output", type=Path, default=C.OUTPUT_DIR / "predictions.csv")
    parser.add_argument("--threshold", type=float, help="Ngưỡng tùy chọn; mặc định dùng ngưỡng validation đã lưu")
    args = parser.parse_args(argv)
    if not C.MODEL_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy model: {C.MODEL_FILE}")
    if not args.input.exists():
        raise FileNotFoundError(f"Không tìm thấy input: {args.input}")

    bundle = joblib.load(C.MODEL_FILE)
    model = bundle["model"]
    features = bundle["feature_columns"]
    threshold = float(bundle["threshold"] if args.threshold is None else args.threshold)
    if not 0 <= threshold <= 1:
        raise ValueError("threshold phải nằm trong [0, 1].")
    header = pd.read_csv(args.input, nrows=0).columns.tolist()
    missing = [column for column in features if column not in header]
    if missing:
        raise ValueError(
            f"Input thiếu {len(missing)} feature. Dữ liệu thô cần được biến đổi bằng schema "
            "đã lưu ở outputs/tables/02_feature_schema.json trước khi chấm điểm."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total, started = 0, time.perf_counter()
    with args.output.open("w", encoding="utf-8-sig", newline="") as output_file:
        for frame in pd.read_csv(args.input, usecols=features, chunksize=C.CSV_CHUNK_SIZE, dtype=np.float32):
            matrix = frame[features].to_numpy(dtype=np.float32, copy=False)
            scores = model.predict_proba(matrix)[:, 1]
            result = pd.DataFrame({
                "record_number": np.arange(total, total + len(frame), dtype=np.int64),
                "takeover_score": scores,
                "predicted_takeover": (scores >= threshold).astype(np.uint8),
            })
            result.to_csv(output_file, index=False, header=(total == 0))
            total += len(frame)
            elapsed = time.perf_counter() - started
            h, rest = divmod(int(elapsed), 3600)
            m, s = divmod(rest, 60)
            print(f"Đã chấm {total:,} dòng | {h:02d}:{m:02d}:{s:02d}", flush=True)
    print(f"Ngưỡng: {threshold:.8f}; file dự đoán: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
