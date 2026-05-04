"""
run_pipeline.py – End-to-end xPTS Shot Quality Model pipeline.

Generates synthetic data, engineers features, trains XGBoost + Logistic
Regression models, evaluates them with rigorous ML metrics (ROC-AUC,
PR-AUC, log-loss, Brier score, ECE), runs stratified k-fold cross-
validation, performs randomised hyperparameter search, and saves a rich
set of diagnostic charts + a model artifact.

Usage:
    python run_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow importing from src/ without an install step
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

from src.generate_synthetic_data import generate_shots
from src.features import build_model_frame
from src.model import (
    train_model,
    cross_validate_model,
    tune_model,
    add_expected_points,
    get_feature_importance,
    get_permutation_importance,
    get_calibration_data,
    get_roc_data,
    get_pr_curve_data,
    get_learning_curve_data,
    save_artifacts,
)

OUTPUTS = Path("outputs")
OUTPUTS.mkdir(exist_ok=True)

sns.set_theme(style="darkgrid", palette="muted")


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
print("=" * 60)
print("Step 1 – Generating synthetic shot data …")
raw_path = Path("data/raw/shots.csv")
raw_path.parent.mkdir(parents=True, exist_ok=True)
shots_raw = generate_shots(12_000)
shots_raw.to_csv(raw_path, index=False)
print(f"  {len(shots_raw):,} shots saved → {raw_path}")

# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------
print("Step 2 – Engineering features …")
shots = build_model_frame(shots_raw)

processed_path = Path("data/processed/shots_model_input.csv")
processed_path.parent.mkdir(parents=True, exist_ok=True)
shots.to_csv(processed_path, index=False)
print(f"  Feature matrix saved → {processed_path}")
print(f"  Columns: {list(shots.columns)}")

# ---------------------------------------------------------------------------
# 3. Stratified k-fold cross-validation
#    Leakage-free: player_zone_fg_pct is refit from training-fold labels
#    in each iteration of the CV loop.
# ---------------------------------------------------------------------------
print("\nStep 3 – 5-fold stratified cross-validation …")
for model_type in ("xgboost", "logistic"):
    cv_results = cross_validate_model(shots, model_type=model_type, n_splits=5)
    mean_row = cv_results[cv_results["fold"] == "mean"].iloc[0]
    std_row  = cv_results[cv_results["fold"] == "std"].iloc[0]
    print(f"\n  [{model_type.upper()}] cross-validation (5-fold):")
    for metric in ("roc_auc", "pr_auc", "log_loss", "brier_score", "ece"):
        print(f"    {metric:>14s}: {mean_row[metric]:.4f} ± {std_row[metric]:.4f}")

cv_results_xgb = cross_validate_model(shots, model_type="xgboost", n_splits=5)
cv_results_xgb.to_csv(OUTPUTS / "cv_results_xgboost.csv", index=False)

# ---------------------------------------------------------------------------
# 4. Hyperparameter tuning (randomised search, XGBoost)
# ---------------------------------------------------------------------------
print("\nStep 4 – Randomised hyperparameter search (XGBoost, n_iter=15) …")
tuning_result = tune_model(shots, model_type="xgboost", n_iter=15, n_splits=3)
print(f"  Best CV ROC-AUC : {tuning_result['best_cv_roc_auc']:.4f}")
print(f"  Test  ROC-AUC   : {tuning_result['test_roc_auc']:.4f}")
print(f"  Test  PR-AUC    : {tuning_result['test_pr_auc']:.4f}")
print(f"  Test  ECE       : {tuning_result['test_ece']:.4f}")
print("  Best params:")
for k, v in tuning_result["best_params"].items():
    print(f"    {k}: {v}")

# ---------------------------------------------------------------------------
# 5. Final model training (with tuned-inspired defaults)
# ---------------------------------------------------------------------------
print("\nStep 5 – Training final models …")
xgb_artifacts = train_model(shots, model_type="xgboost")
lr_artifacts  = train_model(shots, model_type="logistic")

print("\n  XGBoost metrics (hold-out):")
for k, v in xgb_artifacts.metrics.items():
    print(f"    {k:>14s}: {v:.4f}")

print("\n  Logistic Regression metrics (hold-out):")
for k, v in lr_artifacts.metrics.items():
    print(f"    {k:>14s}: {v:.4f}")

# Save XGBoost as the primary model
model_path = save_artifacts(xgb_artifacts, "models/xpts_model.pkl")
print(f"\n  Model artifact saved → {model_path}")

# Attach xpts predictions to the full dataset
shots = add_expected_points(shots, xgb_artifacts.pipeline, xgb_artifacts.feature_columns)
shots.to_csv(processed_path, index=False)

# ---------------------------------------------------------------------------
# 6. Charts
# ---------------------------------------------------------------------------
print("\nStep 6 – Generating charts …")


# ── 6a. NBA half-court shot chart coloured by xPTS ─────────────────────────
def draw_half_court(ax: plt.Axes, color: str = "#aaaaaa") -> None:
    """Draw a simplified NBA half-court outline on *ax*."""
    from matplotlib.patches import Arc, Circle, Rectangle
    ax.add_patch(Rectangle((-250, -52), 500, 940, fill=False, color=color, lw=1.5))
    ax.add_patch(Rectangle((-80, -52), 160, 190, fill=False, color=color, lw=1.2))
    ax.add_patch(Rectangle((-60, -52), 120, 190, fill=False, color=color, lw=0.8))
    ax.add_patch(Circle((0, 0), 7.5, fill=False, color=color, lw=1.5))
    ax.plot([-30, 30], [0, 0], color=color, lw=1.5)
    ax.add_patch(Circle((0, 142), 60, fill=False, color=color, lw=1.2))
    ax.add_patch(Arc((0, 0), 475, 475, theta1=22, theta2=158, color=color, lw=1.5))
    ax.plot([-220, -220], [-52, 90], color=color, lw=1.5)
    ax.plot([220, 220], [-52, 90], color=color, lw=1.5)
    ax.plot([-250, 250], [470, 470], color=color, lw=1.5)


fig, ax = plt.subplots(figsize=(10, 9))
ax.set_facecolor("#1a1a2e")
fig.patch.set_facecolor("#1a1a2e")
draw_half_court(ax, color="#555577")

sample = shots.sample(min(3000, len(shots)), random_state=1)
sc = ax.scatter(
    sample["loc_x"], sample["loc_y"],
    c=sample["xpts"], cmap="RdYlGn",
    s=12, alpha=0.7, vmin=0.4, vmax=1.6, linewidths=0,
)
plt.colorbar(sc, ax=ax, label="Expected Points (xPTS)", fraction=0.03, pad=0.02)
ax.set_xlim(-260, 260)
ax.set_ylim(-60, 500)
ax.set_aspect("equal")
ax.set_title("Shot Chart – Coloured by xPTS", color="white", fontsize=15, pad=12)
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#555577")
plt.tight_layout()
plt.savefig(OUTPUTS / "shot_chart_xpts.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: outputs/shot_chart_xpts.png")


# ── 6b. Calibration curves ─────────────────────────────────────────────────
cal_xgb = get_calibration_data(xgb_artifacts, n_bins=10)
cal_lr  = get_calibration_data(lr_artifacts,  n_bins=10)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration")
ax.plot(cal_xgb["mean_predicted"], cal_xgb["fraction_positive"],
        "o-", lw=2,
        label=f"XGBoost  (AUC={xgb_artifacts.metrics['roc_auc']:.3f},"
              f" ECE={xgb_artifacts.metrics['ece']:.3f})")
ax.plot(cal_lr["mean_predicted"], cal_lr["fraction_positive"],
        "s-", lw=2,
        label=f"Logistic (AUC={lr_artifacts.metrics['roc_auc']:.3f},"
              f" ECE={lr_artifacts.metrics['ece']:.3f})")
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Fraction of Positives (Actual Make Rate)")
ax.set_title("Calibration Curves – Shot Make Probability")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUTS / "calibration_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: outputs/calibration_curves.png")


# ── 6c. ROC curves ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
for arts, label in [(xgb_artifacts, "XGBoost"), (lr_artifacts, "Logistic Regression")]:
    roc = get_roc_data(arts)
    ax.plot(roc["fpr"], roc["tpr"], lw=2,
            label=f"{label} (AUC={arts.metrics['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1.2)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves – Shot Make Prediction")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUTS / "roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: outputs/roc_curves.png")


# ── 6d. Precision-Recall curves ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
for arts, label in [(xgb_artifacts, "XGBoost"), (lr_artifacts, "Logistic Regression")]:
    pr = get_pr_curve_data(arts)
    ax.plot(pr["recall"], pr["precision"], lw=2,
            label=f"{label} (PR-AUC={arts.metrics['pr_auc']:.3f})")
baseline = xgb_artifacts.y_test.mean()
ax.axhline(baseline, color="k", linestyle="--", lw=1.2,
           label=f"No-skill baseline ({baseline:.3f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves – Shot Make Prediction")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUTS / "pr_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: outputs/pr_curves.png")


# ── 6e. Feature importance (XGBoost impurity gain) ─────────────────────────
fi = get_feature_importance(xgb_artifacts)
if fi is not None:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("viridis", len(fi))
    ax.barh(fi["feature"], fi["importance"], color=colors)
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title("XGBoost – Impurity-Based Feature Importance")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUTPUTS / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/feature_importance.png")


# ── 6f. Permutation importance (test-set ROC-AUC drop) ─────────────────────
print("  Computing permutation importance (20 repeats) …")
perm_imp = get_permutation_importance(xgb_artifacts, n_repeats=20)
if not perm_imp.empty:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(perm_imp["feature"], perm_imp["importance_mean"],
            xerr=perm_imp["importance_std"],
            color=sns.color_palette("magma", len(perm_imp)),
            capsize=3)
    ax.axvline(0, color="k", lw=0.8, linestyle="--")
    ax.set_xlabel("Mean decrease in ROC-AUC (20 permutations ± std)")
    ax.set_title("XGBoost – Permutation Feature Importance (Test Set)")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUTPUTS / "permutation_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    perm_imp.to_csv(OUTPUTS / "permutation_importance.csv", index=False)
    print("  Saved: outputs/permutation_importance.png")


# ── 6g. Learning curves (bias-variance diagnostic) ─────────────────────────
print("  Computing learning curves …")
lc = get_learning_curve_data(shots, model_type="xgboost", n_splits=5)

fig, ax = plt.subplots(figsize=(8, 5))
ax.fill_between(lc["train_size"],
                lc["train_mean"] - lc["train_std"],
                lc["train_mean"] + lc["train_std"],
                alpha=0.15)
ax.fill_between(lc["train_size"],
                lc["test_mean"] - lc["test_std"],
                lc["test_mean"] + lc["test_std"],
                alpha=0.15)
ax.plot(lc["train_size"], lc["train_mean"], "o-", lw=2, label="Training ROC-AUC")
ax.plot(lc["train_size"], lc["test_mean"],  "s-", lw=2, label="CV Validation ROC-AUC")
ax.set_xlabel("Training Set Size")
ax.set_ylabel("ROC-AUC")
ax.set_title("Learning Curves – XGBoost (Bias-Variance Diagnostic)")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUTS / "learning_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: outputs/learning_curves.png")


# ── 6h. Player summary – average xPTS per player ───────────────────────────
player_summary = (
    shots.groupby("player_name")
    .agg(
        shots_taken=("xpts", "count"),
        avg_xpts=("xpts", "mean"),
        make_rate=("shot_made_flag", "mean"),
        avg_distance=("shot_distance", "mean"),
    )
    .sort_values("avg_xpts", ascending=False)
    .reset_index()
)
player_summary["xpts_vs_average"] = player_summary["avg_xpts"] - player_summary["avg_xpts"].mean()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

palette = ["#2ecc71" if v >= 0 else "#e74c3c" for v in player_summary["xpts_vs_average"]]
axes[0].barh(player_summary["player_name"], player_summary["avg_xpts"], color=palette)
axes[0].axvline(player_summary["avg_xpts"].mean(), color="white", linestyle="--", lw=1.5,
                label=f"League avg {player_summary['avg_xpts'].mean():.3f}")
axes[0].set_xlabel("Average xPTS per Shot Attempt")
axes[0].set_title("Average xPTS by Player")
axes[0].legend()
axes[0].invert_yaxis()

for _, row in player_summary.iterrows():
    axes[1].scatter(row["avg_xpts"], row["make_rate"], s=120, zorder=3)
    axes[1].annotate(
        row["player_name"].split()[-1],
        (row["avg_xpts"], row["make_rate"]),
        xytext=(4, 2), textcoords="offset points", fontsize=8,
    )
axes[1].set_xlabel("Average xPTS")
axes[1].set_ylabel("Actual Make Rate")
axes[1].set_title("xPTS vs Actual Make Rate by Player")

plt.suptitle("Player-Level Shot Quality Summary", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUTS / "player_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: outputs/player_summary.png")


# ── 6i. xPTS distribution by shot zone ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
zone_order = (
    shots.groupby("shot_zone_basic")["xpts"]
    .median().sort_values(ascending=False).index.tolist()
)
sns.violinplot(
    data=shots, x="shot_zone_basic", y="xpts",
    hue="shot_zone_basic", order=zone_order, hue_order=zone_order,
    palette="Set2", ax=ax, inner="box", density_norm="width", legend=False,
)
ax.set_xticks(range(len(zone_order)))
ax.set_xticklabels(zone_order, rotation=30, ha="right")
ax.set_title("xPTS Distribution by Shot Zone")
ax.set_xlabel("Shot Zone")
ax.set_ylabel("Expected Points (xPTS)")
plt.tight_layout()
plt.savefig(OUTPUTS / "xpts_by_zone.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: outputs/xpts_by_zone.png")


# ── 6j. Comprehensive model metrics table ──────────────────────────────────
metrics_df = pd.DataFrame([
    {
        "Model": "XGBoost",
        **{k: round(v, 4) for k, v in xgb_artifacts.metrics.items()},
    },
    {
        "Model": "Logistic Regression",
        **{k: round(v, 4) for k, v in lr_artifacts.metrics.items()},
    },
])
metrics_df.to_csv(OUTPUTS / "model_metrics.csv", index=False)
print("\n  Model Metrics (hold-out test set):")
print(metrics_df.to_string(index=False))

player_summary.to_csv(OUTPUTS / "player_summary.csv", index=False)
print("\n  Player Summary:")
print(player_summary.to_string(index=False))

print("\n" + "=" * 60)
print("Pipeline complete. All outputs written to outputs/")
print("=" * 60)
