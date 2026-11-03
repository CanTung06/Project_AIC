import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ids import config as C


def main() -> int:
    C.ensure_dirs()
    files = sorted(C.RAW_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy CSV trong {C.RAW_DIR}")
    started = time.perf_counter()
    file_rows, label_rows, missing_rows, numeric_rows = [], [], [], []
    all_label_counts = {}
    schemas = []
    for file_index, path in enumerate(files, 1):
        print(f"[{file_index}/{len(files)}] Đang phân tích {path.name}", flush=True)
        frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        frame.columns = [str(c).strip().lstrip("\ufeff") for c in frame.columns]
        label_column = next((c for c in frame.columns if c.casefold() == "anomaly"), None)
        if label_column is None:
            raise ValueError(f"{path.name}: không tìm thấy cột nhãn Anomaly.")
        schemas.append(frame.columns.tolist())
        file_rows.append({
            "file_name": path.name,
            "rows": len(frame),
            "columns": len(frame.columns),
            "size_mb": round(path.stat().st_size / (1024 ** 2), 3),
        })
        labels = frame[label_column].astype("string").fillna("(MISSING)").str.strip()
        counts = labels.value_counts(dropna=False)
        for label, count in counts.items():
            all_label_counts[str(label)] = all_label_counts.get(str(label), 0) + int(count)
            label_rows.append({
                "file_name": path.name,
                "label": str(label),
                "sample_count": int(count),
                "percentage": 100 * int(count) / max(len(frame), 1),
            })
        for column in frame.columns:
            values = frame[column]
            missing_rows.append({
                "file_name": path.name,
                "column": column,
                "missing_count": int(values.isna().sum()),
                "missing_percentage": 100 * float(values.isna().mean()),
                "data_type": str(values.dtype),
                "unique_values": int(values.nunique(dropna=True)),
            })
            if column == label_column:
                continue
            numeric = pd.to_numeric(values, errors="coerce")
            valid = numeric.replace([np.inf, -np.inf], np.nan).dropna()
            if not valid.empty:
                numeric_rows.append({
                    "file_name": path.name,
                    "column": column,
                    "valid_count": len(valid),
                    "mean": float(valid.mean()),
                    "std": float(valid.std()),
                    "minimum": float(valid.min()),
                    "maximum": float(valid.max()),
                })

    file_frame = pd.DataFrame(file_rows)
    labels_frame = pd.DataFrame(label_rows)
    missing_frame = pd.DataFrame(missing_rows)
    numeric_frame = pd.DataFrame(numeric_rows)
    file_frame.to_csv(C.TABLES_DIR / "01_file_summary.csv", index=False, encoding="utf-8-sig")
    labels_frame.to_csv(C.TABLES_DIR / "01_label_distribution.csv", index=False, encoding="utf-8-sig")
    missing_frame.to_csv(C.TABLES_DIR / "01_column_summary.csv", index=False, encoding="utf-8-sig")
    numeric_frame.to_csv(C.TABLES_DIR / "01_numeric_summary.csv", index=False, encoding="utf-8-sig")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not labels_frame.empty:
        grouped = labels_frame.groupby("label", as_index=False)["sample_count"].sum().sort_values("sample_count")
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.barh(grouped["label"], grouped["sample_count"], color="#3973ac")
        axis.set_title("Số lượng đăng nhập theo nhãn Anomaly")
        axis.set_xlabel("Số dòng")
        axis.set_ylabel("Anomaly (0 = bình thường, 1 = bất thường)")
        figure.tight_layout()
        figure.savefig(C.FIGURES_DIR / "01_label_distribution.png", dpi=150)
        plt.close(figure)
    if not file_frame.empty:
        figure, axis = plt.subplots(figsize=(9, max(4, len(file_frame) * 0.45)))
        axis.barh(file_frame["file_name"], file_frame["columns"], color="#4c956c")
        axis.set_title("Số lượng thuộc tính trong từng file")
        axis.set_xlabel("Số cột")
        figure.tight_layout()
        figure.savefig(C.FIGURES_DIR / "01_attribute_count_by_file.png", dpi=150)
        plt.close(figure)

    total_rows = int(file_frame["rows"].sum())
    summary = {
        "dataset_type": "synthetic web authentication logs",
        "raw_directory": str(C.RAW_DIR),
        "files": file_rows,
        "total_rows": total_rows,
        "label_distribution": all_label_counts,
        "schemas_match": all(schema == schemas[0] for schema in schemas),
        "columns": schemas[0] if schemas else [],
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    (C.METRICS_DIR / "01_eda_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nTóm tắt EDA")
    print(f"Số file: {len(files)} | Tổng dòng: {total_rows:,} | Số cột: {len(schemas[0])}")
    print("Phân bố nhãn Anomaly:")
    print(labels_frame.groupby("label", as_index=False)["sample_count"].sum().to_string(index=False))
    print(f"Kết quả: {C.OUTPUTS_DIR} | Thời gian: {time.perf_counter() - started:.1f} giây")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

