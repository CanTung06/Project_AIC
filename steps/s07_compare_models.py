import json

import pandas as pd

from ids import config as C


def comparison_row(model_name, comparison_name, metrics):
    return {
        "model": model_name,
        "comparison": comparison_name,
        "pr_auc": metrics["pr_auc"],
        "precision_class_1": metrics["precision_class_1"],
        "recall_class_1": metrics["recall_class_1"],
        "f1_class_1": metrics["f1_class_1"],
        "predicted_class_1_rate": metrics["predicted_class_1_rate"],
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tn": metrics["tn"],
    }


def main():
    logistic = json.loads(C.METRICS.read_text(encoding="utf-8"))
    forest = json.loads(C.SUPERVISED_METRICS.read_text(encoding="utf-8"))

    rows = [
        comparison_row(
            "Logistic Regression",
            "Ngưỡng validation tối đa F1 lớp 1",
            logistic["test_metrics_at_f1_threshold"],
        ),
        comparison_row(
            "Random Forest",
            "Ngưỡng validation tối đa F1 lớp 1",
            forest["test_metrics_at_f1_threshold"],
        ),
    ]
    table = pd.DataFrame(rows)
    table.to_csv(C.MODEL_COMPARISON_TABLE, index=False, encoding="utf-8-sig")

    result = {
        "target": C.RAW_LABEL,
        "test_rows": logistic["test_metrics_at_f1_threshold"]["rows"],
        "positive_rate_in_test": logistic["test_metrics_at_f1_threshold"]["actual_class_1_rate"],
        "primary_comparison": "So sánh Logistic Regression và Random Forest trên cùng test; mỗi mô hình dùng ngưỡng tối đa F1 lớp 1 được chọn riêng trên validation.",
        "models": rows,
        "table_file": str(C.MODEL_COMPARISON_TABLE),
        "note": "Các metric đo khả năng dự đoán nhãn Is Attack IP; không khẳng định nhãn này đồng nghĩa với đăng nhập bất thường hoặc ATO.",
    }
    C.MODEL_COMPARISON.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nSo sánh Logistic Regression và Random Forest trên cùng tập test")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Bảng: {C.MODEL_COMPARISON_TABLE}")
    print(f"JSON: {C.MODEL_COMPARISON}")


if __name__ == "__main__":
    main()
