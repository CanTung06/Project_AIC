"""Kiểm tra auth.txt theo từng dòng, không nạp cả file vào RAM."""

import csv
import time
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_DIR / "data" / "raw" / "auth.txt"


def main():
    if not INPUT_FILE.exists():
        print(f"Không tìm thấy file: {INPUT_FILE}")
        return

    # Đọc vài dòng đầu để đoán dấu phân cách.
    with INPUT_FILE.open("r", encoding="utf-8", errors="replace") as file:
        sample = [file.readline().rstrip("\n") for _ in range(10)]

    delimiter = ","
    for possible_delimiter in [",", "\t", ";", "|"]:
        if sample[0].count(possible_delimiter) > 0:
            delimiter = possible_delimiter
            break

    # Đếm số cột dựa trên dòng đầu tiên.
    column_count = len(next(csv.reader([sample[0]], delimiter=delimiter)))
    print(f"File: {INPUT_FILE}")
    print(f"Dung lượng: {INPUT_FILE.stat().st_size:,} bytes")
    print(f"Dấu phân cách: {repr(delimiter)}")
    print(f"Số cột dự kiến: {column_count}")
    print("\nMột vài dòng đầu:")
    for line in sample[:5]:
        print(line[:300])

    # Quét toàn bộ file bằng một CSV reader và in tiến độ mỗi 1 GB.
    file_size = INPUT_FILE.stat().st_size
    report_every = 1024**3
    next_report = report_every
    start_time = time.time()
    total_rows = 0
    empty_rows = 0
    wrong_column_rows = 0

    with INPUT_FILE.open("r", encoding="utf-8", errors="replace", newline="") as file:
        reader = csv.reader(file, delimiter=delimiter)
        for fields in reader:
            total_rows += 1
            if not fields or not any(value.strip() for value in fields):
                empty_rows += 1
            elif len(fields) != column_count:
                wrong_column_rows += 1

            if total_rows % 100_000 == 0:
                position = file.buffer.tell()
            else:
                position = 0
            if position >= next_report:
                elapsed = time.time() - start_time
                percent = position / file_size * 100
                speed = position / max(elapsed, 1) / (1024**2)
                print(
                    f"Đã đọc {position / (1024**3):.1f} / {file_size / (1024**3):.1f} GB "
                    f"({percent:.1f}%) | {speed:.1f} MB/s",
                    flush=True,
                )
                next_report += report_every

    print("\nKết quả kiểm tra toàn bộ file:")
    print(f"Tổng số dòng dữ liệu: {total_rows:,}")
    print(f"Dòng trống: {empty_rows:,}")
    print(f"Dòng có số cột khác dự kiến: {wrong_column_rows:,}")
    print(f"Thời gian: {(time.time() - start_time) / 60:.1f} phút")


if __name__ == "__main__":
    main()
