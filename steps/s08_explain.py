import sys
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier, export_text

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ids import config as C
from ids.common import load_xy, save_json


MODELS = (
    ("Logistic Regression", C.FULL_LOGISTIC_REGRESSION_MODEL),
    ("XGBoost", C.FULL_XGBOOST_MODEL),
    ("Random Forest", C.FULL_RANDOM_FOREST_MODEL),
)
PERMUTATION_SAMPLE_SIZE = 1_000
PERMUTATION_REPEATS = 5
TOP_FEATURES = 15


def model_importance(model, feature_names):
    estimator = getattr(model, "estimator", model)
    if hasattr(estimator, "named_steps") and "classifier" in estimator.named_steps:
        values = np.abs(estimator.named_steps["classifier"].coef_).ravel()
    elif hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "get_booster"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    else:
        return np.full(len(feature_names), np.nan)
    if len(values) != len(feature_names):
        return np.full(len(feature_names), np.nan)
    return values


def main() -> int:
    C.ensure_dirs()
    required = [C.TRAIN_DATA, C.VALIDATION_DATA, *(path for _, path in MODELS)]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Thiếu dữ liệu/model; hãy chạy s03 và s05-s07 trước: "
            + ", ".join(map(str, missing))
        )

    started = time.perf_counter()
    X_validation, y_validation, feature_names = load_xy(C.VALIDATION_DATA)
    if len(X_validation) > PERMUTATION_SAMPLE_SIZE:
        sample = X_validation.sample(
            n=PERMUTATION_SAMPLE_SIZE, random_state=C.SEED
        ).index
        X_sample = X_validation.loc[sample]
        y_sample = y_validation.loc[sample]
    else:
        X_sample, y_sample = X_validation, y_validation

    importance_rows = []
    summaries = {}
    for model_name, model_path in MODELS:
        model = joblib.load(model_path)
        estimator = getattr(model, "estimator", model)
        native = model_importance(model, feature_names)
        permutation = permutation_importance(
            estimator,
            X_sample,
            y_sample,
            scoring="average_precision",
            n_repeats=PERMUTATION_REPEATS,
            random_state=C.SEED,
            n_jobs=1,
        ).importances_mean
        frame = pd.DataFrame({
            "feature": feature_names,
            "model": model_name,
            "model_importance": native,
            "permutation_importance_pr_auc": permutation,
        })
        frame["model_rank"] = frame["model_importance"].rank(
            ascending=False, method="min"
        )
        frame["permutation_rank"] = frame["permutation_importance_pr_auc"].rank(
            ascending=False, method="min"
        )
        importance_rows.append(frame)
        summaries[model_name] = {
            "model_path": str(model_path),
            "top_permutation_features": (
                frame.sort_values("permutation_importance_pr_auc", ascending=False)
                .head(TOP_FEATURES)
                .set_index("feature")["permutation_importance_pr_auc"]
                .astype(float)
                .to_dict()
            ),
            "top_native_features": (
                frame.sort_values("model_importance", ascending=False)
                .head(TOP_FEATURES)
                .set_index("feature")["model_importance"]
                .astype(float)
                .to_dict()
            ),
        }

    combined = pd.concat(importance_rows, ignore_index=True)
    table_path = C.TABLES_DIR / "08_feature_importance.csv"
    combined.to_csv(table_path, index=False, encoding="utf-8-sig")

    rf = combined.loc[combined["model"] == "Random Forest"].sort_values(
        "permutation_importance_pr_auc", ascending=False
    ).head(TOP_FEATURES).sort_values("permutation_importance_pr_auc")
    figure, axis = plt.subplots(figsize=(9, max(4.5, len(rf) * 0.32)))
    axis.barh(rf["feature"], rf["permutation_importance_pr_auc"], color="#3973ac")
    axis.set_title("Random Forest: permutation importance trên validation")
    axis.set_xlabel("Mức giảm PR-AUC khi xáo trộn feature")
    axis.set_ylabel("Feature")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure_path = C.FIGURES_DIR / "08_random_forest_feature_importance.png"
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)

    train_X, train_y, train_features = load_xy(C.TRAIN_DATA, feature_names)
    readable_tree = DecisionTreeClassifier(
        max_depth=3, min_samples_leaf=5, class_weight="balanced", random_state=C.SEED
    )
    readable_tree.fit(train_X, train_y)
    rules = export_text(readable_tree, feature_names=train_features, decimals=4)
    rules_path = C.TABLES_DIR / "08_readable_rules.txt"
    rules_path.write_text(
        "Cây quyết định nông để tham khảo; đây không phải model production.\n\n"
        + rules,
        encoding="utf-8",
    )

    metrics_path = C.METRICS_DIR / "08_explain_summary.json"
    save_json(metrics_path, {
        "purpose": "Global feature explanation; permutation importance computed on validation",
        "permutation_scoring": "average_precision",
        "validation_sample_rows": int(len(X_sample)),
        "feature_count": len(feature_names),
        "models": summaries,
        "feature_table": str(table_path),
        "importance_figure": str(figure_path),
        "readable_tree_rules": str(rules_path),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    })
    print("S08 - GIẢI THÍCH MODEL")
    print(combined.loc[combined["model"] == "Random Forest"]
          .sort_values("permutation_importance_pr_auc", ascending=False)
          .head(TOP_FEATURES)[["feature", "model_importance", "permutation_importance_pr_auc"]]
          .to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print(f"Đã lưu bảng: {table_path}")
    print(f"Đã lưu biểu đồ: {figure_path}")
    print(f"Đã lưu luật cây tham khảo: {rules_path}")
    print(f"Đã lưu metrics: {metrics_path}")
    print(f"Thời gian: {time.perf_counter() - started:.1f} giây")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
