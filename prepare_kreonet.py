"""Convert KREONET IAM JSONL records to this project's raw auth-log schema.

Run directly; this file is intentionally not called by run_all.py.
"""

from pathlib import Path
import json

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "raw" / "kreonet_auth_compatible.csv"

OUTPUT_COLUMNS = [
    "User ID",
    "Timestamp",
    "Login Status",
    "IP Address",
    "Device Type",
    "Location",
    "Session Duration",
    "Failed Attempts",
    "Behavioral Score",
    "Anomaly",
]


def clean_text(value, default=""):
    if value is None or pd.isna(value):
        return default
    return str(value).strip()


def classify_device(user_agent):
    text = clean_text(user_agent).casefold()
    if any(token in text for token in ("iphone", "android", "mobile", "windows phone")):
        return "Mobile"
    if any(token in text for token in ("ipad", "tablet", "kindle")):
        return "Tablet"
    if not text:
        return "Unknown"
    return "Desktop"


def event_status(row):
    """Return an observed login outcome when an event explicitly indicates one."""
    event = clean_text(row.get("event_type")).casefold()
    message = clean_text(row.get("message")).casefold()
    combined = f"{event} {message}"

    if any(token in combined for token in ("login_success", "login successful", "authentication successful")):
        return "Success"
    if any(token in combined for token in ("login_failure", "login failed", "authentication failed", "invalid password")):
        return "Failure"

    try:
        status_code = int(row.get("http_status"))
    except (TypeError, ValueError):
        status_code = 0
    if status_code >= 400:
        return "Failure"
    return "Unknown"


def is_login_flow(frame):
    """Select authentication/login traces; do not use these fields as model inputs."""
    module = frame.get("module", pd.Series("", index=frame.index)).fillna("").astype(str).str.casefold()
    category = frame.get("category", pd.Series("", index=frame.index)).fillna("").astype(str).str.casefold()
    event_type = frame.get("event_type", pd.Series("", index=frame.index)).fillna("").astype(str).str.casefold()
    uri = frame.get("uri", pd.Series("", index=frame.index)).fillna("").astype(str).str.casefold()
    message = frame.get("message", pd.Series("", index=frame.index)).fillna("").astype(str).str.casefold()

    return (
        module.eq("auth")
        | category.str.contains(r"login|credential|account", regex=True)
        | event_type.str.contains(r"login|auth", regex=True)
        | uri.str.contains(r"/auth/|login", regex=True)
        | message.str.contains(r"login|password|authentication", regex=True)
    )


def convert_group(group):
    group = group.sort_values("_parsed_timestamp", kind="stable")
    first = group.iloc[0]
    statuses = group.apply(event_status, axis=1)
    known_statuses = statuses[statuses.ne("Unknown")]
    login_status = known_statuses.iloc[-1] if not known_statuses.empty else "Unknown"

    labels = group.get("label", pd.Series("", index=group.index)).fillna("").astype(str).str.casefold()
    anomaly = int(labels.isin(["anomaly", "attack"]).any())

    return {
        "User ID": clean_text(first.get("user_id"), "Unknown"),
        "Timestamp": first["_parsed_timestamp"].isoformat(),
        "Login Status": login_status,
        "IP Address": clean_text(first.get("client_ip"), "Unknown"),
        "Device Type": classify_device(first.get("user_agent")),
        # KREONET does not provide a trustworthy geographic location.
        "Location": "Unknown",
        # duration_ms is per-event processing latency, not login-session duration.
        "Session Duration": 0,
        "Failed Attempts": int(statuses.eq("Failure").sum()),
        # No equivalent behavioral score exists in the KREONET schema.
        "Behavioral Score": 0,
        "Anomaly": anomaly,
    }


def main():
    input_text = input("Đường dẫn file KREONET .jsonl: ").strip().strip('"')
    if not input_text:
        raise ValueError("Bạn chưa nhập đường dẫn file JSONL.")

    input_path = Path(input_text).expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file: {input_path}")

    output_text = input(f"Đường dẫn CSV đầu ra [{DEFAULT_OUTPUT}]: ").strip().strip('"')
    output_path = Path(output_text).expanduser() if output_text else DEFAULT_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Đang đọc JSONL...")
    source = pd.read_json(input_path, lines=True)
    required = {"timestamp", "trace_id", "user_id", "client_ip", "user_agent", "label"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"JSONL thiếu các trường KREONET cần thiết: {missing}")

    source["_parsed_timestamp"] = pd.to_datetime(source["timestamp"], errors="coerce", utc=True)
    invalid_times = int(source["_parsed_timestamp"].isna().sum())
    source = source.loc[source["_parsed_timestamp"].notna()].copy()

    login_rows = source.loc[is_login_flow(source)].copy()
    if login_rows.empty:
        raise ValueError("Không tìm thấy dòng đăng nhập/xác thực trong file KREONET.")

    # One output row per authentication trace. Rows with no trace_id remain
    # separate instead of being accidentally merged into a single event.
    missing_trace = login_rows["trace_id"].isna() | login_rows["trace_id"].astype(str).str.strip().eq("")
    login_rows["_trace_key"] = login_rows["trace_id"].astype("string")
    login_rows.loc[missing_trace, "_trace_key"] = [
        f"__row_{index}" for index in login_rows.index[missing_trace]
    ]

    records = [convert_group(group) for _, group in login_rows.groupby("_trace_key", sort=False)]
    result = pd.DataFrame(records, columns=OUTPUT_COLUMNS)
    result["Anomaly"] = result["Anomaly"].astype("int8")
    result["Session Duration"] = result["Session Duration"].astype("int32")
    result["Failed Attempts"] = result["Failed Attempts"].astype("int32")
    result["Behavioral Score"] = result["Behavioral Score"].astype("float32")
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    counts = result["Anomaly"].value_counts().sort_index()
    summary = {
        "source_file": str(input_path.resolve()),
        "source_rows": int(len(source)),
        "invalid_timestamps_dropped": invalid_times,
        "login_related_rows": int(len(login_rows)),
        "output_login_traces": int(len(result)),
        "class_counts": {str(label): int(count) for label, count in counts.items()},
        "output_file": str(output_path.resolve()),
        "label_mapping": {"benign": 0, "anomaly": 1, "attack": 1},
        "limitations": [
            "KREONET is synthetic IAM data; its labels are not identical to the source dataset's Anomaly label.",
            "Location is set to Unknown because KREONET has no geographic location field.",
            "Session Duration and Behavioral Score are set to 0 because no equivalent fields exist.",
            "User-Agent is reduced to Mobile, Tablet, Desktop, or Unknown.",
            "Rows are grouped by trace_id; multi-record authentication flows become one CSV row.",
        ],
    }
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Hoàn tất: {len(source):,} dòng nguồn, {len(login_rows):,} dòng đăng nhập, {len(result):,} trace đầu ra.")
    print(f"Bình thường (0): {int(counts.get(0, 0)):,} | Bất thường (1): {int(counts.get(1, 0)):,}")
    print(f"CSV tương thích: {output_path}")
    print(f"Tóm tắt: {summary_path}")
    print("Lưu ý: Session Duration, Behavioral Score = 0; Location = Unknown do KREONET không có các trường tương đương.")


if __name__ == "__main__":
    main()
