"""Stream-inspect auth.txt without loading the full file into memory.

Run from any directory:
    python src/01_inspect_auth.py
    python src/01_inspect_auth.py --sample-lines 20 --max-errors 10

The report is printed as JSON and, by default, saved to outputs/metrics/01_inspection.json.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import TextIO


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_DIR / "data" / "raw" / "auth.txt"
DEFAULT_REPORT = PROJECT_DIR / "outputs" / "metrics" / "01_inspection.json"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect a delimited text file one line at a time."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help=f"Input file (default: {DEFAULT_INPUT})")
    parser.add_argument("--sample-lines", type=int, default=100,
                        help="Number of initial non-empty lines to inspect (default: 100)")
    parser.add_argument("--max-errors", type=int, default=20,
                        help="Maximum malformed line examples to include (default: 20)")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT,
                        help=f"JSON report path (default: {DEFAULT_REPORT})")
    return parser


def detect_delimiter(lines: list[str]) -> str:
    """Choose a likely delimiter from a small sample; fall back to whitespace."""
    candidates = [",", "\t", ";", "|"]
    scores: dict[str, tuple[int, int]] = {}
    for delimiter in candidates:
        widths: list[int] = []
        for line in lines:
            try:
                widths.append(len(next(csv.reader([line], delimiter=delimiter))))
            except (csv.Error, StopIteration):
                continue
        # Prefer a delimiter that consistently produces multiple fields.
        multi = [width for width in widths if width > 1]
        scores[delimiter] = (len(multi), len(set(multi)) == 1 if multi else False)
    best = max(candidates, key=lambda item: (scores[item][0], scores[item][1]))
    if scores[best][0] == 0:
        return "whitespace"
    return best


def open_text(path: Path) -> TextIO:
    # utf-8-sig transparently strips a possible UTF-8 BOM from the first field.
    return path.open("r", encoding="utf-8-sig", errors="replace", newline="")


def parse_line(line: str, delimiter: str) -> list[str]:
    if delimiter == "whitespace":
        return line.split()
    return next(csv.reader([line], delimiter=delimiter, strict=True))


def inspect(path: Path, sample_limit: int, max_errors: int) -> dict:
    if sample_limit < 1:
        raise ValueError("--sample-lines must be at least 1")
    if max_errors < 0:
        raise ValueError("--max-errors cannot be negative")
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    sample: list[str] = []
    total_lines = 0
    empty_lines = 0
    with open_text(path) as source:
        for raw_line in source:
            total_lines += 1
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                empty_lines += 1
                continue
            if len(sample) < sample_limit:
                sample.append(line)

    if not sample:
        return {
            "input": str(path), "size_bytes": path.stat().st_size,
            "total_lines": total_lines, "empty_lines": empty_lines,
            "delimiter": None, "header_detected": None, "column_count": 0,
            "column_names": [], "field_count_distribution": {},
            "malformed_nonempty_lines": 0, "malformed_examples": [],
            "sample_rows": [],
        }

    delimiter = detect_delimiter(sample)
    parsed_sample: list[list[str]] = []
    for line in sample:
        try:
            parsed_sample.append(parse_line(line, delimiter))
        except (csv.Error, UnicodeError):
            continue

    if not parsed_sample:
        raise ValueError("Could not parse any sampled lines")

    expected_width = len(parsed_sample[0])
    # A first row is considered a header only when its fields look textual and
    # later sampled rows contain at least one numeric-looking value.
    first = [field.strip() for field in parsed_sample[0]]
    later = [field.strip() for row in parsed_sample[1:] for field in row]
    has_header = bool(later) and all(first) and any(
        not value.replace(".", "", 1).replace("-", "", 1).isdigit()
        for value in first
    ) and any(value.replace(".", "", 1).replace("-", "", 1).isdigit() for value in later)
    column_names = first if has_header else [f"column_{i + 1}" for i in range(expected_width)]

    field_counts: Counter[str] = Counter()
    malformed_examples: list[dict[str, object]] = []
    malformed_count = 0
    missing_fields = [0] * expected_width
    nonempty_data_rows = 0
    with open_text(path) as source:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                continue
            try:
                fields = parse_line(line, delimiter)
            except (csv.Error, UnicodeError) as exc:
                malformed_count += 1
                if len(malformed_examples) < max_errors:
                    malformed_examples.append({"line": line_number, "error": str(exc), "text": line[:300]})
                continue
            field_counts[str(len(fields))] += 1
            if len(fields) != expected_width:
                malformed_count += 1
                if len(malformed_examples) < max_errors:
                    malformed_examples.append({
                        "line": line_number,
                        "error": f"expected {expected_width} fields, found {len(fields)}",
                        "text": line[:300],
                    })
                continue
            if has_header and line_number == 1:
                continue
            nonempty_data_rows += 1
            for index, value in enumerate(fields):
                if not value.strip():
                    missing_fields[index] += 1

    report = {
        "input": str(path),
        "size_bytes": path.stat().st_size,
        "total_lines": total_lines,
        "empty_lines": empty_lines,
        "delimiter": "whitespace" if delimiter == "whitespace" else repr(delimiter),
        "header_detected": has_header,
        "column_count": expected_width,
        "column_names": column_names,
        "data_rows_with_expected_width": nonempty_data_rows,
        "field_count_distribution": dict(sorted(field_counts.items(), key=lambda item: int(item[0]))),
        "malformed_nonempty_lines": malformed_count,
        "missing_values_by_column": {
            name: count for name, count in zip(column_names, missing_fields)
        },
        "malformed_examples": malformed_examples,
        "sample_rows": parsed_sample[:5],
    }
    return report


def main() -> int:
    args = make_parser().parse_args()
    try:
        report = inspect(args.input.resolve(), args.sample_lines, args.max_errors)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    except (OSError, ValueError, csv.Error) as exc:
        raise SystemExit(f"Inspection failed: {exc}") from exc

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nReport saved to: {args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
