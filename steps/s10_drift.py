import sys
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ids import config as C
from ids.common import load_xy, save_json


MODELS = (
    ("Logistic Regression", C.FULL_LOGISTIC_REGRESSION_MODEL),
    ("XGBoost", C.FULL_XGBOOST_MODEL),
    ("Random Forest", C.FULL_RANDOM_FOREST_MODEL),
)
PSI_EPSILON = 1e-6


def _bucket(values, edges=None):
    values = pd.Series(values).astype(float)
    if values.nunique(dropna=False) <= 10:
        return values.fillna(np.nan).astype("string").fillna("__MISSING__")
    if edges is None:
        quantiles = np.unique(np.quantile(values.dropna(), np.linspace(0, 1, 11)))
        inner = quantiles[1:-1]
    else:
        inner = np.asarray(edges, dtype=float)
    return pd.cut(values, bins=np.r_[-np.inf, inner, np.inf], include_lowest=True).astype("string")


def population_stability_index(reference, current) -> float:
    reference = pd.Series(reference).astype(float)
    current = pd.Series(current).astype(float)
    if reference.empty or current.empty:
        return float("nan")

    if reference.nunique(dropna=False) <= 10:
        ref_bucket = _bucket(reference)
        current_bucket = _bucket(current)
        keys = sorted(set(ref_bucket.dropna()) | set(current_bucket.dropna()))
    else:
        quantiles = np.unique(np.quantile(reference.dropna(), np.linspace(0, 1, 11)))
        inner = quantiles[1:-1]
        edges = np.r_[-np.inf, inner, np.inf]
        ref_bucket = pd.cut(reference, bins=edges, include_lowest=True).astype("string")
        current_bucket = pd.cut(current, bins=edges, include_lowest=True).astype("string")
        keys = sorted(set(ref_bucket.dropna()) | set(current_bucket.dropna()))

    if not keys:
        return 0.0
    ref_share = ref_bucket.value_counts(normalize=True).reindex(keys, fill_value=0).to_numpy()
    current_share = current_bucket.value_counts(normalize=True).reindex(keys, fill_value=0).to_numpy()
    ref_share = np.maximum(ref_share, PSI_EPSILON)
    current_share = np.maximum(current_share, PSI_EPSILON)
    return float(np.sum((current_share - ref_share) * np.log(current_share / ref_share)))


def main() -> int:
    C.ensure_dirs()
    required = [
        C.TRAIN_DATA, C.VALIDATION_DATA, C.TEST_DATA,
        *(path for _, path in MODELS),
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Thiếu dữ liệu/model; hãy chạy s03 và s05-s07 trước: "
            + ", ".join(map(str, missing))
        )

    started = time.perf_counter()
    X_train, y_train, features = load_xy(C.TRAIN_DATA)
    X_validation, y_validation, _ = load_xy(C.VALIDATION_DATA, features)
    X_test, y_test, _ = load_xy(C.TEST_DATA, features)

    feature_rows = []
    for feature in features:
        value = population_stability_index(X_train[feature], X_test[feature])
        feature_rows.append({
            "feature": feature,
            "psi": value,
            "level": "major" if value >= 0.25 else "moderate" if value >= 0.10 else "stable",
            "train_mean": float(X_train[feature].mean()),
            "test_mean": float(X_test[feature].mean()),
        })
    feature_drift = pd.DataFrame(feature_rows).sort_values("psi", ascending=False)
    feature_path = C.TABLES_DIR / "10_feature_drift.csv"
    feature_drift.to_csv(feature_path, index=False, encoding="utf-8-sig")

    model_rows = []
    score_distributions = {}
    for model_name, model_path in MODELS:
        model = joblib.load(model_path)
        threshold = float(getattr(model, "threshold", 0.5))
        validation_scores = model.predict_proba(X_validation)[:, 1]
        test_scores = model.predict_proba(X_test)[:, 1]
        val_alerts = validation_scores >= threshold
        test_alerts = test_scores >= threshold
        score_psi = population_stability_index(validation_scores, test_scores)
        model_rows.append({
            "model": model_name,
            "threshold": threshold,
            "validation_score_psi_test": score_psi,
            "validation_alert_rate": float(val_alerts.mean()),
            "test_alert_rate": float(test_alerts.mean()),
            "test_precision": float(precision_score(y_test, test_alerts, zero_division=0)),
            "test_recall": float(recall_score(y_test, test_alerts, zero_division=0)),
            "test_positive_support": int((y_test == 1).sum()),
        })
        score_distributions[model_name] = {
            "validation": validation_scores,
            "test": test_scores,
            "threshold": threshold,
            "psi": score_psi,
        }

    model_drift = pd.DataFrame(model_rows)
    model_path = C.TABLES_DIR / "10_model_score_drift.csv"
    model_drift.to_csv(model_path, index=False, encoding="utf-8-sig")

    top = feature_drift.head(min(20, len(feature_drift))).sort_values("psi")
    figure, axes = plt.subplots(1, 2, figsize=(14, max(5, len(top) * 0.28)))
    axes[0].barh(top["feature"], top["psi"], color="#3973ac")
    axes[0].axvline(0.10, color="#d89b2b", linestyle="--", label="0.10")
    axes[0].axvline(0.25, color="#b23a48", linestyle="--", label="0.25")
    axes[0].set_title("PSI feature: train so với test")
    axes[0].set_xlabel("Population Stability Index")
    axes[0].legend()
    colors = ["#3973ac", "#d89b2b"]
    for index, (name, values) in enumerate(score_distributions.items()):
        axes[1].hist(
            values["validation"], bins=20, density=True, histtype="step",
            linewidth=1.8, color=colors[index % 2], label=f"{name} validation",
        )
        axes[1].hist(
            values["test"], bins=20, density=True, histtype="step",
            linewidth=1.8, linestyle="--", color=colors[index % 2], label=f"{name} test",
        )
    axes[1].set_title("Phân phối score validation và test")
    axes[1].set_xlabel("Điểm bất thường")
    axes[1].set_ylabel("Mật độ")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure_path = C.FIGURES_DIR / "10_drift_overview.png"
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)

    metrics_path = C.METRICS_DIR / "10_drift_summary.json"
    save_json(metrics_path, {
        "feature_reference_split": str(C.TRAIN_DATA),
        "feature_current_split": str(C.TEST_DATA),
        "score_reference_split": str(C.VALIDATION_DATA),
        "score_current_split": str(C.TEST_DATA),
        "psi_interpretation": {"stable_below": 0.10, "moderate_below": 0.25, "major_at_or_above": 0.25},
        "feature_psi_major_count": int((feature_drift["psi"] >= 0.25).sum()),
        "feature_psi_moderate_count": int(((feature_drift["psi"] >= 0.10) & (feature_drift["psi"] < 0.25)).sum()),
        "top_feature_drift": feature_drift.head(20).to_dict(orient="records"),
        "model_score_drift": model_rows,
        "feature_drift_table": str(feature_path),
        "model_drift_table": str(model_path),
        "overview_figure": str(figure_path),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    })
    print("S10 - PHÂN TÍCH DRIFT")
    print("\nFeature PSI cao nhất:")
    print(feature_drift.head(15).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nScore drift theo model:")
    print(model_drift.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Đã lưu metrics: {metrics_path}")
    print(f"Thời gian: {time.perf_counter() - started:.1f} giây")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
