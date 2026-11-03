"""Dự đoán CSV log đăng nhập bằng Random Forest full data."""

import json
import re
import sys
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ids import config as C
from ids.common import confusion_table, save_confusion_matrix_figure
from steps.s02_preprocess import (
    CATEGORICAL_COLUMNS,
    MISSING,
    NUMERIC_COLUMNS,
    REQUIRED_COLUMNS,
    canonicalize_columns,
    engineer_auth_features,
)


CHUNK_SIZE = 10_000
LABEL_COLUMNS = (C.TARGET_COLUMN, C.SOURCE_LABEL_COLUMN, "Is Attack IP", "Is Account Takeover")
RAW_COLUMNS = set(REQUIRED_COLUMNS) | {"Login Successful", *LABEL_COLUMNS}
FALLBACK_TIME = "1970-01-01T00:00:00Z"
COUNTRY_ALIASES = {
    "US": "USA", "USA": "USA", "UNITED STATES": "USA",
    "UNITED STATES OF AMERICA": "USA", "CA": "Canada", "DE": "Germany",
    "JP": "Japan", "AQ": "Antarctica",
}


def elapsed(start):
    seconds = int(time.perf_counter() - start)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def load_schema_and_model():
    if not C.AUTH_FEATURE_SCHEMA.is_file():
        raise FileNotFoundError(f"Thiếu schema: {C.AUTH_FEATURE_SCHEMA}; hãy chạy s02 trước.")
    schema = json.loads(C.AUTH_FEATURE_SCHEMA.read_text(encoding="utf-8"))
    model_path = C.FULL_RANDOM_FOREST_MODEL
    if not model_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy model Random Forest: {model_path}")
    model = joblib.load(model_path)
    estimator = getattr(model, "estimator", model)
    features = getattr(estimator, "feature_names_in_", None)
    if features is None:
        features = estimator.get_booster().feature_names
    features = [] if features is None else list(features)
    if not features or features != schema.get("feature_columns"):
        raise ValueError("Feature của model không khớp schema s02; hãy kiểm tra lại phiên bản model.")
    return model, features, schema["category_vocabulary"]


def prepare_chunk(frame, features, vocabulary):
    """Chuẩn hóa log thô hoặc dữ liệu s02; bỏ cột lạ và đệm cột thiếu."""
    frame.columns = [str(c).strip().lstrip("\ufeff") for c in frame.columns]
    frame = canonicalize_columns(frame)
    aliases = {"Login Timestamp": "Timestamp", "Country": "Location"}
    frame = frame.rename(columns={s: t for s, t in aliases.items() if s in frame and t not in frame})
    source_columns = list(frame.columns)

    if "Login Status" not in frame and "Login Successful" in frame:
        values = frame["Login Successful"].astype("string").str.strip().str.casefold()
        frame["Login Status"] = np.select(
            [values.isin({"true", "1", "1.0", "yes", "success", "successful"}),
             values.isin({"false", "0", "0.0", "no", "fail", "failed"})],
            ["Success", "Fail"], default=MISSING,
        )
    if "Location" in frame:
        country = frame["Location"].astype("string").str.strip()
        frame["Location"] = country.str.upper().map(COUNTRY_ALIASES).fillna(country)

    is_processed = any(
        name in frame.columns for name in features
        if "__" in name or name.startswith(("login_hour_", "login_weekday_"))
    )
    labels = [c for c in LABEL_COLUMNS if c in frame]
    if is_processed:
        # Extra columns are discarded. Missing model features are later filled with zero.
        return frame[[c for c in features if c in frame] + labels].copy(), True, [], labels

    # Keep only recognized source columns; add absent source fields with safe defaults.
    kept = [c for c in source_columns if c in RAW_COLUMNS]
    if "Login Successful" in source_columns and "Login Status" in frame:
        kept.append("Login Status")
    frame = frame.loc[:, list(dict.fromkeys(kept))].copy()
    defaults = []
    for column in REQUIRED_COLUMNS:
        if column == C.SOURCE_LABEL_COLUMN or column in frame:
            continue
        defaults.append(column)
        if column == "Timestamp":
            frame[column] = FALLBACK_TIME
        elif column == "User ID":
            frame[column] = "__MISSING_USER__"
        elif column in CATEGORICAL_COLUMNS:
            frame[column] = MISSING
        else:
            frame[column] = 0

    parsed = pd.to_datetime(frame["Timestamp"], format="mixed", errors="coerce", utc=True)
    invalid_time = int(parsed.isna().sum())
    if invalid_time:
        parsed = parsed.fillna(pd.Timestamp(FALLBACK_TIME))
        defaults.append("Timestamp không hợp lệ")
    frame["Timestamp"] = parsed
    for column in CATEGORICAL_COLUMNS:
        if column not in frame:
            frame[column] = MISSING
        values = frame[column].astype("string").str.strip()
        lookup = {str(v).casefold(): v for v in vocabulary[column] if v not in (MISSING, "__OTHER__")}
        values = values.map(lambda v: lookup.get(str(v).casefold(), v) if pd.notna(v) and str(v).strip() else MISSING)
        frame[column] = values
    for column in NUMERIC_COLUMNS:
        if column not in frame:
            frame[column] = 0
            defaults.append(column)
        frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)

    return frame, False, list(dict.fromkeys(defaults)), list(dict.fromkeys(kept + labels))


def make_features(frame, is_processed, features, vocabulary):
    data = frame if is_processed else engineer_auth_features(frame, vocabulary)
    X = data.reindex(columns=features, fill_value=0).apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    return data, X


def save_figures(name, y_true, scores, threshold, output_path):
    folder = output_path.parent / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_").lower()
    cm_path = folder / f"{output_path.stem}_{slug}_confusion_matrix.png"
    save_confusion_matrix_figure(
        confusion_table(y_true, scores, threshold), cm_path, f"Confusion Matrix - {name}"
    )
    paths = [cm_path]
    if np.unique(y_true).size == 2:
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, label=f"PR curve (AP={ap:.4f})")
        ax.set(xlabel="Recall", ylabel="Precision", title=f"Precision-Recall Curve - {name}", xlim=(0, 1), ylim=(0, 1.02))
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        pr_path = folder / f"{output_path.stem}_{slug}_pr_curve.png"
        fig.savefig(pr_path, dpi=160)
        plt.close(fig)
        paths.append(pr_path)
    return paths


def make_report(label, truth, scores, threshold):
    predicted = (scores >= threshold).astype(np.int8)
    text = classification_report(
        truth, predicted, labels=[0, 1], target_names=["Normal", "Anomaly"],
        digits=4, zero_division=0,
    ).rstrip()
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
    lines = [f"Đối chiếu theo nhãn: {label}", text]
    if np.unique(truth).size == 2:
        lines += [f"AUC-ROC: {roc_auc_score(truth, scores):.4f}",
                  f"PR-AUC: {average_precision_score(truth, scores):.8f}"]
    else:
        lines += ["AUC-ROC: không tính được vì chỉ có một lớp nhãn.",
                  "PR-AUC: không tính được vì chỉ có một lớp nhãn."]
    lines.append(f"TP={tp:,} FP={fp:,} FN={fn:,} TN={tn:,}")
    return "\n".join(lines)


def predict(input_path):
    started = time.perf_counter()
    model, features, vocabulary = load_schema_and_model()
    threshold = float(getattr(model, "threshold", 0.5))
    output_dir = C.OUTPUTS_DIR / "predictions" / input_path.name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_random_forest_full_data_predictions.csv"
    temp_path = output_path.with_name(output_path.stem + ".tmp.csv")
    temp_path.unlink(missing_ok=True)

    # Header pass tells us whether the input is raw or already preprocessed.
    header = pd.read_csv(input_path, nrows=0, encoding="utf-8-sig")
    _, is_processed, defaults, output_columns = prepare_chunk(header, features, vocabulary)
    total_rows, positives = 0, 0
    truths, score_parts = {}, {}
    print(f"Bắt đầu dự đoán bằng Random Forest | threshold={threshold:.8f}")
    for chunk in pd.read_csv(input_path, chunksize=CHUNK_SIZE, encoding="utf-8-sig", low_memory=False):
        data, processed, chunk_defaults, kept_columns = prepare_chunk(chunk, features, vocabulary)
        data, X = make_features(data, processed, features, vocabulary)
        scores = model.predict_proba(X)[:, 1].astype(np.float32)
        predictions = (scores >= threshold).astype(np.int8)
        data["random_forest_anomaly_probability"] = scores
        data["random_forest_is_anomaly"] = predictions
        data["random_forest_prediction"] = np.where(predictions == 1, "Anomaly", "Normal")
        keep = [c for c in kept_columns if c in data]
        keep += ["random_forest_anomaly_probability", "random_forest_is_anomaly", "random_forest_prediction"]
        data[keep].to_csv(temp_path, index=False, mode="a" if total_rows else "w",
                          header=(total_rows == 0), encoding="utf-8" if total_rows else "utf-8-sig")
        defaults.extend(chunk_defaults)
        for label in LABEL_COLUMNS:
            if label not in data:
                continue
            values = pd.to_numeric(data[label], errors="coerce").to_numpy()
            valid = np.isfinite(values) & np.isin(values, [0, 1])
            if valid.any():
                truths.setdefault(label, []).append(values[valid].astype(np.int8))
                score_parts.setdefault(label, []).append(scores[valid])
        total_rows += len(data)
        positives += int(predictions.sum())
        print(f"Đã dự đoán {total_rows:,} dòng | {elapsed(started)}", flush=True)

    if not total_rows:
        raise ValueError("CSV đầu vào không có dòng dữ liệu.")
    temp_path.replace(output_path)
    report = [
        f"Classification report | Random Forest (full_data) | threshold={threshold:.8f}",
        f"Input CSV: {input_path}", f"Số dòng: {total_rows:,}",
        f"Định dạng đầu vào: {'CSV đã tiền xử lý' if is_processed else 'CSV log thô'}",
    ]
    for label, parts in truths.items():
        truth, scores = np.concatenate(parts), np.concatenate(score_parts[label])
        section = make_report(label, truth, scores, threshold)
        print(f"\n{section}")
        report.append(section)
        report.extend(f"Biểu đồ: {p}" for p in save_figures(label, truth, scores, threshold, output_path))
    if not truths:
        report.append(f"Không có nhãn thật hợp lệ; số dự đoán Normal={total_rows - positives:,}, Anomaly={positives:,}.")
        print(report[-1])
    if defaults:
        warning = "Trường/feature thiếu hoặc không hợp lệ đã được điền mặc định: " + ", ".join(sorted(set(defaults)))
        report.append(warning)
        print(warning)
    report.extend([f"Số feature: {len(features):,}", f"Kết quả CSV: {output_path}", f"Thời gian: {elapsed(started)}"])
    report_path = output_path.with_suffix(".txt")
    report_path.write_text("\n\n".join(report) + "\n", encoding="utf-8")
    print(f"Kết quả CSV: {output_path}\nBáo cáo TXT: {report_path}\nHoàn tất | {elapsed(started)}")


def main():
    path = Path(input("Nhập đường dẫn CSV log đăng nhập: ").strip().strip('"')).expanduser()
    if not path.is_file():
        print(f"Không tìm thấy file: {path}")
        return 1
    try:
        predict(path)
    except (ValueError, KeyError, OSError, ImportError) as error:
        print(f"Không thể dự đoán bằng Random Forest: {error}")
        return 1
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
