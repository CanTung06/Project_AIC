"""Đánh giá các model đã huấn luyện trên CSV thô trong data/test."""

import json
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ids import config as C
from ids.common import calculate_metrics, confusion_table, elapsed_text, load_feature_names, save_json
from ids.thresholded import ThresholdedClassifier
from steps.s02_preprocess import KEEP_LABELS, normalize_label


EXTERNAL_DATA_DIR = C.DATA_DIR / "test"
CHUNK_SIZE = 100_000
MODELS = {
    "SGDClassifier": C.UNDERSAMPLED_MODEL,
    "Logistic Regression": C.LOGISTIC_REGRESSION_MODEL,
    "XGBoost": C.XGBOOST_MODEL,
}
THRESHOLD_FALLBACKS = {
    "SGDClassifier": C.METRICS_DIR / "05_threshold_tuning.json",
    "Logistic Regression": C.METRICS_DIR / "06_logistic_regression_metrics.json",
    "XGBoost": C.METRICS_DIR / "07_xgboost_metrics.json",
}


def load_threshold(model_name: str, model) -> tuple[object, float]:
    if isinstance(model, ThresholdedClassifier):
        return model.estimator, float(model.threshold)

    with THRESHOLD_FALLBACKS[model_name].open("r", encoding="utf-8-sig") as file:
        details = json.load(file)
    if model_name.startswith("SGDClassifier"):
        threshold = details["selected_model"]["threshold"]
    elif model_name == "Logistic Regression":
        threshold = details["validation_at_selected_threshold"]["threshold"]
    else:
        threshold = details["threshold_tuning"]["selected_model"]["threshold"]
    return model, float(threshold)


def main() -> int:
    C.ensure_dirs()
    started = time.perf_counter()
    csv_files = sorted(EXTERNAL_DATA_DIR.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Không tìm thấy CSV trong {EXTERNAL_DATA_DIR}")

    missing_models = [path for path in MODELS.values() if not path.exists()]
    if missing_models:
        raise FileNotFoundError("Thiếu model đã huấn luyện: " + ", ".join(map(str, missing_models)))

    feature_names = load_feature_names(C.TRAIN_DATA)
    loaded_models = {}
    thresholds = {}
    for model_name, model_path in MODELS.items():
        loaded_models[model_name], thresholds[model_name] = load_threshold(
            model_name, joblib.load(model_path)
        )

    y_parts = []
    score_parts = {name: [] for name in MODELS}
    source_label_counts = Counter()
    retained_label_counts = Counter()
    rows_read = 0
    rows_retained = 0

    print(f"Đánh giá ngoài trên {len(csv_files)} CSV trong {EXTERNAL_DATA_DIR}")
    print(f"Giữ nhãn: {', '.join(KEEP_LABELS.values())}")
    print(f"Chunk: {CHUNK_SIZE:,} dòng")

    for file_index, file_path in enumerate(csv_files, start=1):
        file_rows = 0
        file_retained = 0
        print(f"\n[{file_index}/{len(csv_files)}] Đang đọc: {file_path.name}")
        for chunk in pd.read_csv(
            file_path,
            chunksize=CHUNK_SIZE,
            encoding="cp1252",
            encoding_errors="replace",
            skipinitialspace=True,
            low_memory=False,
        ):
            chunk.columns = [str(column).strip().lstrip("\ufeff") for column in chunk.columns]
            label_columns = [column for column in chunk.columns if column.casefold() == "label"]
            if len(label_columns) != 1:
                raise ValueError(f"{file_path.name}: cần đúng một cột Label, tìm thấy {len(label_columns)}")

            missing_features = [column for column in feature_names if column not in chunk.columns]
            if missing_features:
                raise ValueError(
                    f"{file_path.name}: thiếu {len(missing_features)} feature của tập train; "
                    f"ví dụ: {missing_features[:8]}"
                )

            label_column = label_columns[0]
            raw_labels = chunk[label_column].astype("string").str.strip()
            normalized_labels = raw_labels.map(normalize_label)
            for raw_label, count in raw_labels.fillna("(MISSING)").value_counts().items():
                source_label_counts[str(raw_label)] += int(count)
                normalized = normalize_label(raw_label)
                if normalized in KEEP_LABELS:
                    retained_label_counts[KEEP_LABELS[normalized]] += int(count)

            keep_mask = normalized_labels.isin(KEEP_LABELS)
            selected = chunk.loc[keep_mask]
            if not selected.empty:
                y = normalized_labels.loc[keep_mask].ne("benign").astype("int8").to_numpy()
                X = selected[feature_names].apply(pd.to_numeric, errors="coerce")
                X = X.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
                y_parts.append(y)
                for model_name, model in loaded_models.items():
                    score_parts[model_name].append(model.predict_proba(X)[:, 1].astype(np.float32))

            file_rows += len(chunk)
            file_retained += int(keep_mask.sum())
            rows_read += len(chunk)
            rows_retained += int(keep_mask.sum())
            print(
                f"  Đã đọc {rows_read:,} dòng, giữ {rows_retained:,} | "
                f"{elapsed_text(time.perf_counter() - started)}",
                end="\r",
                flush=True,
            )

        print(
            f"\n  Hoàn tất: {file_rows:,} dòng, giữ {file_retained:,} | "
            f"tổng {rows_read:,} dòng, giữ {rows_retained:,} | "
            f"{elapsed_text(time.perf_counter() - started)}"
        )

    if not y_parts:
        raise ValueError("Không có dòng nào khớp các nhãn được giữ trong data/test.")

    y_true = np.concatenate(y_parts)
    if np.unique(y_true).size < 2:
        raise ValueError("Tập data/test sau lọc phải còn cả BENIGN và ít nhất một nhãn tấn công.")

    results = {
        "source_directory": str(EXTERNAL_DATA_DIR),
        "source_file_count": len(csv_files),
        "source_row_count": rows_read,
        "retained_row_count": rows_retained,
        "ignored_row_count": rows_read - rows_retained,
        "source_label_counts": dict(source_label_counts),
        "retained_label_counts": dict(retained_label_counts),
        "label_mapping": {"0": "BENIGN", "1": [value for key, value in KEEP_LABELS.items() if key != "benign"]},
        "test_class_counts": {
            "normal": int((y_true == 0).sum()),
            "attack": int((y_true == 1).sum()),
        },
        "models": {},
        "processing_time": elapsed_text(time.perf_counter() - started),
    }

    for model_name in MODELS:
        scores = np.concatenate(score_parts[model_name])
        threshold = thresholds[model_name]
        metrics = calculate_metrics(y_true, scores, threshold)
        predictions = (scores >= threshold).astype(np.int8)
        report = classification_report(
            y_true,
            predictions,
            labels=[0, 1],
            target_names=["Normal", "Attack"],
            output_dict=True,
            zero_division=0,
        )
        matrix = confusion_table(y_true, scores, threshold)
        if model_name.startswith("SGDClassifier"):
            safe_name = "sgd_undersampling"
        elif model_name == "Logistic Regression":
            safe_name = "logistic_regression"
        else:
            safe_name = "xgboost"
        save_json(C.METRICS_DIR / f"11_external_confusion_{safe_name}.json", matrix.to_dict(orient="index"))
        results["models"][model_name] = {
            "model_path": str(MODELS[model_name]),
            "threshold_source": "embedded validation threshold" if isinstance(joblib.load(MODELS[model_name]), ThresholdedClassifier) else str(THRESHOLD_FALLBACKS[model_name]),
            "metrics": metrics,
            "classification_report": report,
            "confusion_matrix": matrix.to_dict(orient="index"),
        }
        print(f"\nClassification report | {model_name} | threshold={threshold:.8f}")
        print(classification_report(
            y_true,
            predictions,
            labels=[0, 1],
            target_names=["Normal", "Attack"],
            digits=4,
            zero_division=0,
        ))
        print(f"PR-AUC: {metrics['pr_auc']:.4f} | AUC-ROC: {metrics['roc_auc']:.4f}")
        print(f"TP={metrics['tp']:,} FP={metrics['fp']:,} FN={metrics['fn']:,} TN={metrics['tn']:,}")

    results["processing_time"] = elapsed_text(time.perf_counter() - started)
    output_path = C.METRICS_DIR / "11_external_test_metrics.json"
    save_json(output_path, results)
    print(f"\nMetrics saved: {output_path}")
    print(f"Total time: {results['processing_time']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
