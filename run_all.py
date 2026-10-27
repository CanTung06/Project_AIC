"""Chạy các bước của pipeline theo thứ tự; có thể chọn một bước riêng."""
import argparse
import importlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STEPS = [
    ("s01", "steps.s01_eda"),
    ("s02", "steps.s02_preprocess"),
    ("s03", "steps.s03_train_xgboost"),
    ("s04", "steps.s04_evaluate"),
    ("s05", "steps.s05_validation_error_analysis"),
    ("s06", "steps.s06_validation_alert_review"),
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Pipeline XGBoost phát hiện account takeover")
    parser.add_argument("--only", help="Chạy riêng một bước: s01, s02, s03, s04, s05 hoặc s06")
    parser.add_argument("--from", dest="start", help="Chạy từ bước được chỉ định trở đi")
    args = parser.parse_args(argv)
    ids = [step_id for step_id, _ in STEPS]
    if args.only and args.only not in ids:
        parser.error(f"Bước không hợp lệ. Chọn một trong: {', '.join(ids)}")
    if args.start and args.start not in ids:
        parser.error(f"Bước không hợp lệ. Chọn một trong: {', '.join(ids)}")
    if args.only:
        selected = [args.only]
    elif args.start:
        selected = ids[ids.index(args.start):]
    else:
        selected = ids

    from ids import config as C
    C.ensure_dirs()
    started = time.perf_counter()
    for step_id, module in STEPS:
        if step_id not in selected:
            continue
        print(f"\n===== Bắt đầu {step_id} ({module}) =====", flush=True)
        importlib.import_module(module).main()
    print(f"\nHoàn tất pipeline; thời gian: {time.perf_counter() - started:.1f} giây")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
