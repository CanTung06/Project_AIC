import json
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ids import config as C

# Chuẩn hóa Unicode và dấu câu trước khi so khớp để nhận diện được nhãn có dấu
# gạch ngang khác nhau hoặc ký tự thay thế do lỗi encoding của file nguồn.
KEEP_LABELS = {
    "benign": "BENIGN",
    "ftp patator": "FTP-Patator",
    "ssh patator": "SSH-Patator",
    "web attack brute force": "Web Attack - Brute Force",
}


def normalize_label(value: object) -> str:
    """Chuẩn hóa nhãn để so khớp ổn định giữa các encoding/dấu gạch ngang."""
    text = str(value).replace("\u00ef\u00bf\u00bd", " ").replace("\ufffd", " ")
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def format_elapsed_time(seconds: float) -> str:
    seconds = int(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def main() -> int:
    files = sorted(C.RAW_DIR.glob("*.csv"))
    if not files:
        print(f"Không tìm thấy CSV nào trong: {C.RAW_DIR}")
        return 1

    C.ensure_dirs()

    output_path = C.FILTERED_DATA
    temporary_path = output_path.with_name(output_path.stem + ".tmp.csv")
    temporary_path.unlink(missing_ok=True)

    kept_label_counts: Counter[str] = Counter()
    dropped_label_counts: Counter[str] = Counter()
    file_summary: list[dict] = []
    total_rows = 0
    kept_rows = 0
    wrote_header = False
    started = time.perf_counter()

    print(f"Bước 02 - Tiền xử lý | {len(files)} file CSV")
    print("Giữ lại: BENIGN, FTP-Patator, SSH-Patator, Web Attack - Brute Force")
    print(f"Chunk: {C.CHUNK_SIZE:,} dòng")

    for file_index, file_path in enumerate(files, start=1):
        file_started = time.perf_counter()
        file_total = 0
        file_kept = 0
        print(f"\n[{file_index}/{len(files)}] Đang xử lý: {file_path.name}")

        chunks = pd.read_csv(
            file_path,
            chunksize=C.CHUNK_SIZE,
            encoding="cp1252",
            encoding_errors="replace",
            skipinitialspace=True,
            low_memory=False,
        )

        for chunk in chunks:
            chunk.columns = [str(column).strip().lstrip("\ufeff") for column in chunk.columns]
            label_columns = [
                column for column in chunk.columns
                if column.strip().casefold() == "label"
            ]
            if len(label_columns) != 1:
                raise ValueError(
                    f"{file_path.name}: cần đúng một cột Label, tìm thấy {len(label_columns)}"
                )

            label_column = label_columns[0]
            raw_labels = chunk[label_column].astype("string").str.strip()
            normalized_labels = raw_labels.map(normalize_label)
            counts = raw_labels.fillna("(THIẾU NHÃN)").value_counts(dropna=False)
            for raw_label, count in counts.items():
                display_label = (
                    str(raw_label)
                    .replace("\u00ef\u00bf\u00bd", "-")
                    .replace("\ufffd", "-")
                )
                normalized = normalize_label(display_label)
                if normalized in KEEP_LABELS:
                    kept_label_counts[KEEP_LABELS[normalized]] += int(count)
                else:
                    dropped_label_counts[display_label] += int(count)

            keep_mask = normalized_labels.isin(KEEP_LABELS)
            selected = chunk.loc[keep_mask].copy()
            rows_in_chunk = len(chunk)
            selected_count = len(selected)
            file_total += rows_in_chunk
            total_rows += rows_in_chunk
            file_kept += selected_count
            kept_rows += selected_count

            if selected_count:
                selected_labels = normalized_labels.loc[keep_mask]
                selected["attack_type"] = selected_labels.map(KEEP_LABELS).to_numpy()
                selected["is_attack"] = selected_labels.ne("benign").astype("int8").to_numpy()
                selected.drop(columns=[label_column], inplace=True)
                selected.to_csv(
                    temporary_path,
                    index=False,
                    mode="a" if wrote_header else "w",
                    header=not wrote_header,
                    encoding="utf-8-sig" if not wrote_header else "utf-8",
                )
                wrote_header = True

            print(
                f"  Đã đọc {file_total:,} dòng trong file, giữ {file_kept:,} | "
                f"tổng đã đọc {total_rows:,}, giữ {kept_rows:,} | "
                f"{format_elapsed_time(time.perf_counter() - started)}",
                end="\r",
                flush=True,
            )

        print(
            "\r\033[K"
            f"  Hoàn tất: {file_total:,} dòng, giữ {file_kept:,} | "
            f"{format_elapsed_time(time.perf_counter() - file_started)} | "
            f"tổng đã đọc {total_rows:,}, giữ {kept_rows:,} | "
            f"{format_elapsed_time(time.perf_counter() - started)}"
        )
        file_summary.append({
            "file_name": file_path.name,
            "total_rows": file_total,
            "retained_rows": file_kept,
            "dropped_rows": file_total - file_kept,
        })

    if not wrote_header:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError("Không có dòng nào khớp các nhãn được giữ lại; chưa tạo dataset đầu ra.")

    temporary_path.replace(output_path)
    elapsed = time.perf_counter() - started
    benign_count = kept_label_counts.get("BENIGN", 0)
    attack_count = kept_rows - benign_count

    attack_summary_rows = [
        {
            "original_label": label,
            "decision": "GIỮ",
            "sample_count": int(count),
            "reason": "Dữ liệu bình thường" if label == "BENIGN" else "Tấn công brute-force mật khẩu",
        }
        for label, count in kept_label_counts.items()
    ]
    attack_summary_rows.extend(
        {
            "original_label": label,
            "decision": "LOẠI",
            "sample_count": int(count),
            "reason": "Không thuộc nhóm brute-force mật khẩu được chọn",
        }
        for label, count in dropped_label_counts.items()
    )
    pd.DataFrame(attack_summary_rows).sort_values(
        ["decision", "sample_count"], ascending=[True, False]
    ).to_csv(C.TABLES_DIR / "02_attack_filter_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(file_summary).to_csv(
        C.TABLES_DIR / "02_file_summary.csv", index=False, encoding="utf-8-sig"
    )

    summary = {
        "task": "Binary classification of BENIGN and password brute-force traffic",
        "source_directory": str(C.RAW_DIR),
        "source_file_count": len(files),
        "source_row_count": total_rows,
        "retained_row_count": kept_rows,
        "dropped_row_count": total_rows - kept_rows,
        "retained_label_counts": dict(kept_label_counts),
        "dropped_label_counts": dict(dropped_label_counts),
        "benign_sample_count": benign_count,
        "attack_sample_count": attack_count,
        "attack_rate_percent": round(100 * attack_count / max(kept_rows, 1), 6),
        "target_column": C.TARGET_COLUMN,
        "label_mapping": {"0": "BENIGN", "1": ["FTP-Patator", "SSH-Patator", "Web Attack - Brute Force"]},
        "output_file": str(output_path),
        "chunk_size": C.CHUNK_SIZE,
        "processing_time_seconds": round(elapsed, 2),
        "processing_time": format_elapsed_time(elapsed),
    }
    with (C.METRICS_DIR / "02_preprocess_summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print("HOÀN TẤT TIỀN XỬ LÝ")
    print(f"Tổng dòng nguồn : {total_rows:,}")
    print(f"Dòng giữ lại    : {kept_rows:,}")
    print(f"Dòng loại bỏ    : {total_rows - kept_rows:,}")
    print(f"BENIGN (0)      : {benign_count:,}")
    print(f"Brute-force (1) : {attack_count:,}")
    print(f"Tỷ lệ tấn công  : {summary['attack_rate_percent']:.4f}%")
    print(f"Thời gian       : {format_elapsed_time(elapsed)}")
    print(f"Dataset         : {output_path}")
    print(f"Bảng thống kê   : {C.TABLES_DIR / '02_attack_filter_summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
