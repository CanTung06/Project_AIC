"""Tạo bảng gọn để rà soát các cảnh báo validation theo ngưỡng của model."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ids import config as C


def describe_feature(name: str, value: float) -> str:
    if "=VALUE::" in name:
        return name.replace("=VALUE::", "=", 1)
    if name.endswith("=OTHER"):
        return name
    if abs(value - round(value)) < 1e-6:
        return f"{name}={int(round(value))}"
    return f"{name}={value:.4g}"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not C.VALIDATION_TOPK_TABLE_FILE.exists():
        raise FileNotFoundError(
            f"Thiếu {C.VALIDATION_TOPK_TABLE_FILE}; hãy chạy s05_validation_error_analysis.py trước."
        )
    if not C.MODEL_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy model: {C.MODEL_FILE}")

    bundle = joblib.load(C.MODEL_FILE)
    feature_columns = bundle["feature_columns"]
    threshold = float(bundle["threshold"])
    top = pd.read_csv(C.VALIDATION_TOPK_TABLE_FILE)
    required = {"rank", "validation_row", "risk_score", "predicted_at_validation_threshold", "target"}
    missing = required - set(top.columns)
    if missing:
        raise ValueError(f"Bảng s05 thiếu cột: {sorted(missing)}")

    alerts = top.loc[top["predicted_at_validation_threshold"].eq(1)].copy()
    review_rows = []
    for _, row in alerts.iterrows():
        active = []
        for name in feature_columns:
            value = float(row[name])
            if abs(value) > 1e-12:
                active.append(describe_feature(name, value))
        label = int(row["target"])
        review_rows.append({
            "rank": int(row["rank"]),
            "validation_row": int(row["validation_row"]),
            "risk_score": float(row["risk_score"]),
            "threshold": threshold,
            "dataset_label": label,
            "result_by_dataset_label": "TP theo nhãn" if label == 1 else "FP theo nhãn",
            "active_feature_count": len(active),
            "active_features": " | ".join(active),
        })

    review = pd.DataFrame(review_rows)
    C.ensure_dirs()
    review.to_csv(C.VALIDATION_ALERT_REVIEW_FILE, index=False, encoding="utf-8-sig")
    tp = int((alerts["target"] == 1).sum())
    fp = int((alerts["target"] == 0).sum())
    summary = {
        "alerts_at_selected_validation_threshold": len(alerts),
        "tp_by_dataset_label": tp,
        "fp_by_dataset_label": fp,
        "threshold": threshold,
        "precision_by_dataset_label": tp / (tp + fp) if tp + fp else 0.0,
        "review_csv": str(C.VALIDATION_ALERT_REVIEW_FILE),
        "validation_row_note": "Số thứ tự trong validation.csv; không phải ID sự kiện trong raw.",
        "label_note": "TP/FP được tính theo nhãn hiện có trong dataset; chưa xác minh độc lập bằng log gốc.",
    }
    C.VALIDATION_ALERT_REVIEW_SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("Bảng rà soát cảnh báo validation")
    print(f"Số cảnh báo: {len(alerts)} | TP theo nhãn: {tp} | FP theo nhãn: {fp}")
    print(f"Precision theo nhãn: {summary['precision_by_dataset_label']:.4f}")
    print(f"Đã lưu bảng: {C.VALIDATION_ALERT_REVIEW_FILE}")
    print(f"Đã lưu tóm tắt: {C.VALIDATION_ALERT_REVIEW_SUMMARY_FILE}")
    print("validation_row chỉ là số thứ tự trong validation.csv, không truy ngược trực tiếp được về raw.")


if __name__ == "__main__":
    main()
