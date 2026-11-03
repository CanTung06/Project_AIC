"""Dự đoán bất thường đăng nhập từ CSV bên ngoài bằng XGBoost đã huấn luyện.

Đầu vào có thể là CSV log xác thực thô hoặc CSV đã qua s02. Với dữ liệu thô,
chương trình tạo feature thời gian và one-hot theo schema của mô hình.
"""

import json
import re
import sys
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ids import config as C
from ids.common import confusion_table, save_confusion_matrix_figure
from steps.s02_preprocess import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    REQUIRED_COLUMNS,
    canonicalize_columns,
    engineer_auth_features,
)


SCORE_COLUMN = "xgb_anomaly_probability"
PREDICTION_COLUMN = "xgb_is_anomaly"
LABEL_COLUMN = "xgb_prediction"
PREDICTION_CHUNK_SIZE = 10_000
LINE_COUNT_CHUNK_SIZE = 8 * 1024 * 1024
MAX_IN_MEMORY_PREDICTION_ROWS = 5_000
COUNTRY_ALIASES = {
    "US": "USA",
    "USA": "USA",
    "UNITED STATES": "USA",
    "UNITED STATES OF AMERICA": "USA",
    "CA": "Canada",
    "DE": "Germany",
    "JP": "Japan",
    "AQ": "Antarctica",
}


def normalize_external_schema(frame, vocabulary):
    """Đổi tên và chuẩn hóa các trường RBA về schema mà model đã học."""
    aliases = {
        "Login Timestamp": "Timestamp",
        "Country": "Location",
    }
    rename = {
        source: target
        for source, target in aliases.items()
        if source in frame.columns and target not in frame.columns
    }
    frame = frame.rename(columns=rename)

    if "Login Status" not in frame.columns and "Login Successful" in frame.columns:
        status_values = frame["Login Successful"].astype("string").str.strip().str.casefold()
        success_values = {"true", "1", "1.0", "yes", "success", "successful"}
        failure_values = {"false", "0", "0.0", "no", "fail", "failed"}
        frame["Login Status"] = np.select(
            [status_values.isin(success_values), status_values.isin(failure_values)],
            ["Success", "Fail"],
            default="__MISSING__",
        )

    if "Location" in frame.columns:
        country = frame["Location"].astype("string").str.strip()
        normalized_country = country.str.upper().map(COUNTRY_ALIASES)
        frame["Location"] = normalized_country.fillna(country)

    # Chấp nhận khác biệt chữ hoa/thường (ví dụ mobile -> Mobile) và dùng đúng
    # cách viết của danh mục lúc train. Giá trị chưa từng học sẽ thành OTHER ở s02.
    for column in CATEGORICAL_COLUMNS:
        if column not in frame.columns:
            continue
        lookup = {
            str(category).casefold(): category
            for category in vocabulary[column]
            if category not in ("__MISSING__", "__OTHER__")
        }
        values = frame[column].astype("string").str.strip()
        frame[column] = values.map(
            lambda value: lookup.get(str(value).casefold(), value)
            if not pd.isna(value) and str(value).strip()
            else "__MISSING__"
        )

    missing_numeric = [column for column in NUMERIC_COLUMNS if column not in frame.columns]
    if "Failed Attempts" in missing_numeric and "Login Status" in frame.columns:
        frame["Failed Attempts"] = frame["Login Status"].eq("Fail").astype(np.float32)
        missing_numeric.remove("Failed Attempts")
    # RBA không có Session Duration hoặc Behavioral Score tương ứng; giữ các
    # feature trong đúng vị trí của model bằng 0 và báo rõ cho người chạy.
    for column in missing_numeric:
        frame[column] = np.float32(0)

    return frame, missing_numeric


def feature_names_from_model(model):
    estimator = getattr(model, "estimator", model)
    names = getattr(estimator, "feature_names_in_", None)
    if names is None:
        names = estimator.get_booster().feature_names
    if names is None or len(names) == 0:
        raise ValueError("Không đọc được danh sách feature từ model XGBoost.")
    return list(names)


def make_label_report(label_name, y_true, scores, threshold):
    predictions = (scores >= threshold).astype(np.int8)
    lines = [
        f"Đối chiếu theo nhãn: {label_name}",
        classification_report(
            y_true,
            predictions,
            labels=[0, 1],
            target_names=["Normal", "Anomaly"],
            digits=4,
            zero_division=0,
        ).rstrip(),
    ]
    if np.unique(y_true).size == 2:
        lines.append(f"AUC-ROC: {roc_auc_score(y_true, scores):.4f}")
        lines.append(f"PR-AUC: {average_precision_score(y_true, scores):.8f}")
    else:
        lines.append("AUC-ROC: không tính được vì dữ liệu chỉ có một lớp nhãn.")
        lines.append("PR-AUC: không tính được vì dữ liệu chỉ có một lớp nhãn.")
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    lines.append(f"TP={tp:,} FP={fp:,} FN={fn:,} TN={tn:,}")
    return "\n".join(lines)


def print_label_report(label_name, y_true, scores, threshold):
    """In báo cáo nhãn và trả nội dung để ghi thành file TXT."""
    report = make_label_report(label_name, y_true, scores, threshold)
    print(f"\n{report}")
    return report


def make_output_path(input_path, model_key):
    output_directory = C.OUTPUTS_DIR / "predictions" / input_path.name
    output_directory.mkdir(parents=True, exist_ok=True)
    base = output_directory / f"{input_path.stem}_{model_key}_full_data_predictions.csv"
    output = base
    number = 2
    while output.exists():
        output = base.with_name(f"{base.stem}_{number}{base.suffix}")
        number += 1
    return output


def save_evaluation_figures(label_name, y_true, scores, threshold, output_path):
    """Save PR curve and confusion matrix for an available ground-truth label."""
    figure_directory = output_path.parent / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)
    label_slug = re.sub(r"[^A-Za-z0-9_-]+", "_", label_name).strip("_").lower()
    base_name = f"{output_path.stem}_{label_slug}"
    paths = {}

    matrix_path = figure_directory / f"{base_name}_confusion_matrix.png"
    matrix = confusion_table(y_true, scores, threshold)
    save_confusion_matrix_figure(
        matrix,
        matrix_path,
        f"Confusion Matrix - {label_name}",
    )
    paths["confusion_matrix"] = matrix_path

    if np.unique(y_true).size == 2:
        precision, recall, _ = precision_recall_curve(y_true, scores)
        average_precision = average_precision_score(y_true, scores)
        curve_path = figure_directory / f"{base_name}_pr_curve.png"
        figure, axis = plt.subplots(figsize=(8, 6))
        axis.plot(recall, precision, label=f"PR curve (AP={average_precision:.4f})")
        axis.set_title(f"Precision-Recall Curve - {label_name}")
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(curve_path, dpi=160)
        plt.close(figure)
        paths["pr_curve"] = curve_path
    return paths


def format_elapsed(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def count_csv_rows(input_path):
    """Count physical CSV lines for an exact row-progress denominator.

    Authentication log CSVs use one record per line. This scans in binary
    blocks, so it uses little memory even for multi-gigabyte inputs.
    """
    file_size = input_path.stat().st_size
    bytes_read = 0
    line_count = 0
    last_byte = b""
    started = time.perf_counter()
    next_report = 0.1
    print("Đang đếm số dòng để hiển thị phần trăm tiến trình...")
    with input_path.open("rb") as source_file:
        while True:
            block = source_file.read(LINE_COUNT_CHUNK_SIZE)
            if not block:
                break
            bytes_read += len(block)
            line_count += block.count(b"\n")
            last_byte = block[-1:]
            percent = bytes_read / max(file_size, 1)
            if percent >= next_report:
                print(
                    f"Đếm dòng: {min(percent, 1.0) * 100:.0f}% dung lượng file"
                    f" | {format_elapsed(time.perf_counter() - started)}",
                    flush=True,
                )
                next_report += 0.1
    if last_byte and last_byte != b"\n":
        line_count += 1
    rows = max(0, line_count - 1)  # Bỏ dòng tiêu đề.
    print(f"Đã đếm xong {rows:,} dòng dữ liệu | {format_elapsed(time.perf_counter() - started)}")
    return rows


def predict(input_path, model_path, model_key, model_label):
    # Chọn cách đọc theo số dòng, không chỉ theo dung lượng file: ma trận one-hot
    # có thể phình rất lớn dù CSV nguồn còn nhỏ.
    expected_rows = count_csv_rows(input_path)
    if expected_rows > MAX_IN_MEMORY_PREDICTION_ROWS:
        return predict_large_csv(input_path, model_path, model_key, model_label, expected_rows)
    if not C.AUTH_FEATURE_SCHEMA.is_file():
        raise FileNotFoundError(
            f"Thiếu schema {C.AUTH_FEATURE_SCHEMA}; hãy chạy s02_preprocess trước."
        )
    schema = json.loads(C.AUTH_FEATURE_SCHEMA.read_text(encoding="utf-8"))
    vocabulary = schema["category_vocabulary"]
    schema_features = schema.get("feature_columns")
    if not schema_features:
        raise ValueError("Schema s02 không có danh sách feature; hãy chạy lại s02_preprocess.")

    started = time.perf_counter()
    print("Tiến trình [1/4]: Đang đọc CSV vào bộ nhớ...")
    raw = pd.read_csv(input_path, encoding="utf-8-sig", low_memory=False)
    print(f"Tiến trình [1/4]: Đã đọc {len(raw):,} dòng | {format_elapsed(time.perf_counter() - started)}")
    raw.columns = [str(column).strip().lstrip("\ufeff") for column in raw.columns]
    raw = canonicalize_columns(raw)
    raw, defaulted_numeric_columns = normalize_external_schema(raw, vocabulary)
    model = joblib.load(model_path)
    feature_names = feature_names_from_model(model)
    if feature_names != schema_features:
        raise ValueError(
            "Schema one-hot hiện tại không khớp model. Hãy chạy lại s02 đến s06 để "
            "tạo dataset, schema và model cùng một phiên bản."
        )

    # Cho phép đưa thẳng tập đã qua s02 (ví dụ auth_test.csv) vào dự đoán.
    # Nếu chưa có đủ feature one-hot thì xem đầu vào là log thô và tự biến đổi.
    if all(column in raw.columns for column in feature_names):
        transformed = raw.copy()
        input_mode = "CSV đã tiền xử lý (dùng feature one-hot có sẵn)"
        output_columns = [column for column in raw.columns if column not in feature_names]
    else:
        required_without_label = [
            column for column in REQUIRED_COLUMNS if column != C.SOURCE_LABEL_COLUMN
        ]
        missing = [column for column in required_without_label if column not in raw.columns]
        if missing:
            raise ValueError(
                "CSV phải là log thô đủ cột hoặc dữ liệu đã qua s02. Cột log thô còn thiếu: "
                + ", ".join(missing)
            )
        transformed = engineer_auth_features(raw, vocabulary)
        input_mode = "CSV log thô (đã tạo feature one-hot)"
        output_columns = list(raw.columns)

    print(f"Tiến trình [2/4]: Đã chuẩn hóa feature | {format_elapsed(time.perf_counter() - started)}")

    missing_features = [column for column in feature_names if column not in transformed.columns]
    if missing_features:
        raise ValueError(
            "Feature sau tiền xử lý không khớp model: " + ", ".join(missing_features[:15])
        )
    # Dùng đúng danh sách feature từ s02: các cột phân loại đã được one-hot,
    # còn ID/thời gian và nhãn (nếu có) không được đưa vào model.
    X = transformed[feature_names].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    print("Tiến trình [3/4]: Đang dự đoán bằng model...")
    probabilities = model.predict_proba(X)[:, 1]
    threshold = float(getattr(model, "threshold", 0.5))
    predictions = (probabilities >= threshold).astype(np.int8)
    score_column = f"{model_key}_anomaly_probability"
    prediction_column = f"{model_key}_is_anomaly"
    label_column = f"{model_key}_prediction"
    transformed[score_column] = probabilities
    transformed[prediction_column] = predictions
    transformed[label_column] = np.where(predictions == 1, "Anomaly", "Normal")

    output_path = make_output_path(input_path, model_key)
    print("Tiến trình [4/4]: Đang lưu kết quả...")
    output_columns = [column for column in output_columns if column in transformed.columns]
    output_columns.extend([score_column, prediction_column, label_column])
    transformed[output_columns].to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Tiến trình [4/4]: Hoàn tất | {format_elapsed(time.perf_counter() - started)}")

    report_path = output_path.with_suffix(".txt")
    report_sections = [
        f"Classification report | {model_label} (full_data) | threshold={threshold:.8f}",
        f"Input CSV: {input_path}",
        f"Số dòng: {len(transformed):,}",
        f"Định dạng đầu vào: {input_mode}",
    ]
    evaluation_labels = [
        column for column in (
            C.TARGET_COLUMN,
            C.SOURCE_LABEL_COLUMN,
            "Is Attack IP",
            "Is Account Takeover",
        )
        if column in transformed.columns
    ]
    valid_evaluation_labels = []
    for label_name in evaluation_labels:
        candidate = pd.to_numeric(transformed[label_name], errors="coerce")
        if candidate.notna().all() and candidate.isin([0, 1]).all():
            valid_evaluation_labels.append((label_name, candidate.astype(np.int8).to_numpy()))

    if valid_evaluation_labels:
        for label_name, y_true in valid_evaluation_labels:
            section = make_label_report(label_name, y_true, probabilities, threshold)
            print(f"\n{section}")
            report_sections.append(section)
            figure_paths = save_evaluation_figures(
                f"{model_key}_{label_name}", y_true, probabilities, threshold, output_path
            )
            for figure_type, figure_path in figure_paths.items():
                report_sections.append(f"{figure_type}: {figure_path}")
                print(f"Đã lưu biểu đồ {figure_type}: {figure_path}")
            if "pr_curve" not in figure_paths:
                report_sections.append("PR curve: không tạo được vì nhãn thật chỉ có một lớp.")
    else:
        print("Không có nhãn thật hợp lệ; chỉ in số lượng dự đoán, không tính metric.")
        print(f"Số dòng đã dự đoán: {len(transformed):,}")
        print(f"Normal predictions: {int((predictions == 0).sum()):,}")
        print(f"Anomaly predictions: {int((predictions == 1).sum()):,}")
        report_sections.append("Không có nhãn thật hợp lệ; không tính classification report hoặc AUC.")
        report_sections.append(f"Normal predictions: {int((predictions == 0).sum()):,}")
        report_sections.append(f"Anomaly predictions: {int((predictions == 1).sum()):,}")
    print(f"Số feature số (đã one-hot): {len(feature_names):,}")
    print(f"Định dạng đầu vào: {input_mode}")
    if defaulted_numeric_columns:
        warning = "Cảnh báo: đã điền 0 cho feature không có trong đầu vào: " + ", ".join(defaulted_numeric_columns)
        print(warning)
        report_sections.append(warning)
    print(f"Kết quả CSV: {output_path}")
    report_sections.extend([
        f"Số feature số (đã one-hot): {len(feature_names):,}",
        f"Định dạng đầu vào: {input_mode}",
        f"Kết quả CSV: {output_path}",
        f"Thời gian: {format_elapsed(time.perf_counter() - started)}",
    ])
    report_path.write_text("\n\n".join(report_sections) + "\n", encoding="utf-8")
    print(f"Báo cáo TXT: {report_path}")
    return output_path


def predict_large_csv(input_path, model_path, model_key, model_label, expected_rows=None):
    """Dự đoán CSV dung lượng lớn theo từng chunk."""
    if not C.AUTH_FEATURE_SCHEMA.is_file():
        raise FileNotFoundError(
            f"Thiếu schema {C.AUTH_FEATURE_SCHEMA}; hãy chạy s02_preprocess trước."
        )
    schema = json.loads(C.AUTH_FEATURE_SCHEMA.read_text(encoding="utf-8"))
    vocabulary = schema["category_vocabulary"]
    schema_features = schema.get("feature_columns")
    if not schema_features:
        raise ValueError("Schema s02 không có danh sách feature; hãy chạy lại s02_preprocess.")

    model = joblib.load(model_path)
    feature_names = feature_names_from_model(model)
    if feature_names != schema_features:
        raise ValueError(
            "Schema one-hot hiện tại không khớp model. Hãy chạy lại s02 đến s06 để "
            "tạo dataset, schema và model cùng một phiên bản."
        )
    header = pd.read_csv(input_path, nrows=0, encoding="utf-8-sig")
    header.columns = [str(column).strip().lstrip("\ufeff") for column in header.columns]
    header, _ = normalize_external_schema(canonicalize_columns(header), vocabulary)
    is_preprocessed = all(column in header.columns for column in feature_names)
    if is_preprocessed:
        output_columns = [column for column in header.columns if column not in feature_names]
    else:
        output_columns = list(header.columns)

    output_path = make_output_path(input_path, model_key)
    temporary_output = output_path.with_name(output_path.stem + ".tmp" + output_path.suffix)
    temporary_output.unlink(missing_ok=True)
    label_parts = {}
    score_parts = {}
    label_candidates = (C.TARGET_COLUMN, C.SOURCE_LABEL_COLUMN, "Is Attack IP", "Is Account Takeover")
    threshold = float(getattr(model, "threshold", 0.5))
    total_rows = 0
    positive_predictions = 0
    missing_numeric_columns = None
    started = time.perf_counter()
    if expected_rows is None:
        expected_rows = count_csv_rows(input_path)
    print(f"Bắt đầu dự đoán {expected_rows:,} dòng theo chunk {PREDICTION_CHUNK_SIZE:,}.")

    for chunk_number, chunk in enumerate(pd.read_csv(
        input_path,
        chunksize=PREDICTION_CHUNK_SIZE,
        encoding="utf-8-sig",
        low_memory=False,
    ), start=1):
        chunk.columns = [str(column).strip().lstrip("\ufeff") for column in chunk.columns]
        chunk = canonicalize_columns(chunk)
        chunk, defaulted_columns = normalize_external_schema(chunk, vocabulary)
        if missing_numeric_columns is None:
            missing_numeric_columns = defaulted_columns

        if is_preprocessed:
            transformed = chunk
        else:
            required = [column for column in REQUIRED_COLUMNS if column != C.SOURCE_LABEL_COLUMN]
            missing = [column for column in required if column not in chunk.columns]
            if missing:
                raise ValueError("CSV thiếu trường sau chuẩn hóa: " + ", ".join(missing))
            transformed = engineer_auth_features(chunk, vocabulary)

        missing_features = [column for column in feature_names if column not in transformed.columns]
        if missing_features:
            raise ValueError("Thiếu feature của model: " + ", ".join(missing_features[:15]))
        X = transformed[feature_names].apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
        scores = model.predict_proba(X)[:, 1].astype(np.float32)
        predictions = (scores >= threshold).astype(np.int8)
        score_column = f"{model_key}_anomaly_probability"
        prediction_column = f"{model_key}_is_anomaly"
        label_column = f"{model_key}_prediction"
        transformed[score_column] = scores
        transformed[prediction_column] = predictions
        transformed[label_column] = np.where(predictions == 1, "Anomaly", "Normal")

        keep_columns = [
            column for column in output_columns
            if column in transformed.columns
        ] + [score_column, prediction_column, label_column]
        transformed[keep_columns].to_csv(
            temporary_output,
            index=False,
            mode="a" if total_rows else "w",
            header=(total_rows == 0),
            encoding="utf-8" if total_rows else "utf-8-sig",
        )

        for label_name in label_candidates:
            if label_name not in transformed.columns:
                continue
            values = pd.to_numeric(transformed[label_name], errors="coerce").to_numpy()
            valid = np.isfinite(values) & np.isin(values, [0, 1])
            if valid.any():
                label_parts.setdefault(label_name, []).append(values[valid].astype(np.int8))
                score_parts.setdefault(label_name, []).append(scores[valid])

        total_rows += len(transformed)
        positive_predictions += int(predictions.sum())
        elapsed = time.perf_counter() - started
        percent = min(total_rows / expected_rows, 1.0) if expected_rows else 1.0
        rows_per_second = total_rows / elapsed if elapsed > 0 else 0.0
        remaining_seconds = (
            (expected_rows - total_rows) / rows_per_second
            if rows_per_second > 0 and expected_rows > total_rows
            else 0.0
        )
        print(
            f"Tiến trình: {percent * 100:6.2f}% | {total_rows:,}/{expected_rows:,} dòng"
            f" | {rows_per_second:,.0f} dòng/giây | đã chạy {format_elapsed(elapsed)}"
            f" | còn khoảng {format_elapsed(remaining_seconds)}",
            flush=True,
        )

    if total_rows == 0:
        raise ValueError("CSV đầu vào không có dòng dữ liệu.")
    temporary_output.replace(output_path)
    print(f"Classification report | {model_label} (full_data) | threshold={threshold:.8f}")
    report_path = output_path.with_suffix(".txt")
    report_sections = [
        f"Classification report | {model_label} (full_data) | threshold={threshold:.8f}",
        f"Input CSV: {input_path}",
        f"Số dòng: {total_rows:,}",
        f"Định dạng đầu vào: {'CSV đã tiền xử lý' if is_preprocessed else 'CSV log thô'}",
    ]
    if label_parts:
        for label_name, parts in label_parts.items():
            y_true = np.concatenate(parts)
            label_scores = np.concatenate(score_parts[label_name])
            section = print_label_report(
                f"{model_key}_{label_name}",
                y_true,
                label_scores,
                threshold,
            )
            report_sections.append(section)
            figure_paths = save_evaluation_figures(
                label_name, y_true, label_scores, threshold, output_path
            )
            for figure_type, figure_path in figure_paths.items():
                report_sections.append(f"{figure_type}: {figure_path}")
                print(f"Đã lưu biểu đồ {figure_type}: {figure_path}")
            if "pr_curve" not in figure_paths:
                report_sections.append("PR curve: không tạo được vì nhãn thật chỉ có một lớp.")
    else:
        print("Không có nhãn thật hợp lệ; chỉ in số lượng dự đoán, không tính metric.")
        print(f"Normal predictions: {total_rows - positive_predictions:,}")
        print(f"Anomaly predictions: {positive_predictions:,}")
        report_sections.append("Không có nhãn thật hợp lệ; không tính classification report hoặc AUC.")
        report_sections.append(f"Normal predictions: {total_rows - positive_predictions:,}")
        report_sections.append(f"Anomaly predictions: {positive_predictions:,}")
    print(f"Số dòng đã dự đoán: {total_rows:,}")
    print(f"Số feature số (đã one-hot): {len(feature_names):,}")
    if missing_numeric_columns:
        warning = "Cảnh báo: đã điền 0 cho feature không có trong đầu vào: " + ", ".join(missing_numeric_columns)
        print(warning)
        report_sections.append(warning)
    print(f"Kết quả CSV: {output_path}")
    report_sections.extend([
        f"Số feature số (đã one-hot): {len(feature_names):,}",
        f"Kết quả CSV: {output_path}",
        f"Thời gian: {format_elapsed(time.perf_counter() - started)}",
    ])
    report_path.write_text("\n\n".join(report_sections) + "\n", encoding="utf-8")
    print(f"Báo cáo TXT: {report_path}")
    return output_path


def main():
    print("ĐÁNH GIÁ BA MODEL FULL DATA TRÊN CSV BÊN NGOÀI")
    input_path = Path(input("Nhập đường dẫn CSV log đăng nhập: ").strip().strip('"')).expanduser()
    if not input_path.is_file():
        print(f"Không tìm thấy file: {input_path}")
        return 1
    models = [
        ("logistic_regression", "Logistic Regression", C.FULL_LOGISTIC_REGRESSION_MODEL),
        ("xgboost", "XGBoost", C.FULL_XGBOOST_MODEL),
        ("random_forest", "Random Forest", C.FULL_RANDOM_FOREST_MODEL),
    ]
    missing_models = [f"{name}: {path}" for name, _, path in models if not path.is_file()]
    if missing_models:
        print("Thiếu model full data; hãy chạy bước huấn luyện tương ứng:\n" + "\n".join(missing_models))
        return 1
    for model_key, model_label, model_path in models:
        print(f"\n{'=' * 72}\nĐANG ĐÁNH GIÁ {model_label.upper()} (FULL DATA)\n{'=' * 72}")
        try:
            predict(input_path, model_path, model_key, model_label)
        except (ValueError, KeyError, OSError, ImportError) as error:
            print(f"Không thể dự đoán bằng {model_label}: {error}")
            return 1
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())

