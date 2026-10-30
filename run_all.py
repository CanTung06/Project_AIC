"""Chạy tuần tự pipeline phát hiện brute-force trên CIC-IDS2017.

    python run_all.py
    python run_all.py --only s05,s06,s07
    python run_all.py --from s08
"""

import argparse
import importlib
import time

from ids import config as C


STEPS = [
    ("s01", "steps.s01_eda"),
    ("s02", "steps.s02_preprocess"),
    ("s03", "steps.s03_split_data"),
    ("s04", "steps.s04_imbalance"),
    ("s05", "steps.s05_train_sgd"),
    ("s06", "steps.s06_train_logistic_regression"),
    ("s07", "steps.s07_train_xgboost"),
    ("s08", "steps.s08_explain"),
    ("s09", "steps.s09_drift"),
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Pipeline CIC-IDS2017 dùng SGD, Logistic Regression và XGBoost")
    parser.add_argument("--only", default="", help="Chỉ chạy các bước, ví dụ: s05,s06,s07")
    parser.add_argument("--from", dest="start", default="", help="Chạy từ bước, ví dụ: s08")
    args = parser.parse_args(argv)

    step_ids = [step_id for step_id, _ in STEPS]
    if args.only:
        selected = [item.strip() for item in args.only.split(",") if item.strip()]
    elif args.start:
        if args.start not in step_ids:
            parser.error(f"Bước không hợp lệ: {args.start}; chọn trong {step_ids}")
        selected = step_ids[step_ids.index(args.start):]
    else:
        selected = step_ids
    unknown = sorted(set(selected) - set(step_ids))
    if unknown:
        parser.error(f"Bước không hợp lệ: {unknown}; chọn trong {step_ids}")

    C.ensure_dirs()
    overall_start = time.perf_counter()
    timings = []
    for step_id, module_name in STEPS:
        if step_id not in selected:
            continue
        print(f"\n{'=' * 76}\nBẮT ĐẦU {step_id}: {module_name}\n{'=' * 76}")
        step_start = time.perf_counter()
        importlib.import_module(module_name).main()
        elapsed = time.perf_counter() - step_start
        timings.append({"step": step_id, "seconds": round(elapsed, 2)})
        print(f"\nHoàn tất {step_id} | {elapsed:.1f} giây")

    print("\nTỔNG THỜI GIAN")
    for item in timings:
        print(f"{item['step']}: {item['seconds']:.1f} giây")
    print(f"Toàn pipeline: {time.perf_counter() - overall_start:.1f} giây")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
