import argparse
import importlib


STEPS = [
    ("s01", "steps.s01_eda"),
    ("s02", "steps.s02_preprocess"),
    ("s03", "steps.s03_train_logistic_regression"),
    ("s04", "steps.s04_evaluate_logistic_regression"),
    ("s05", "steps.s05_train_random_forest"),
    ("s06", "steps.s06_evaluate_random_forest"),
    ("s07", "steps.s07_compare_models"),
]


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline phát hiện bất thường đăng nhập và so sánh mô hình."
    )
    parser.add_argument("--only", choices=[step[0] for step in STEPS])
    parser.add_argument("--from-step", choices=[step[0] for step in STEPS])
    args = parser.parse_args()

    if args.only:
        selected = [args.only]
    elif args.from_step:
        start = [step[0] for step in STEPS].index(args.from_step)
        selected = [step[0] for step in STEPS[start:]]
    else:
        selected = [step[0] for step in STEPS]

    for step_name, module_name in STEPS:
        if step_name in selected:
            print(f"\n===== {step_name}: {module_name} =====", flush=True)
            importlib.import_module(module_name).main()


if __name__ == "__main__":
    main()
