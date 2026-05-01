"""
run_pipeline.py – End-to-end xPTS Shot Quality Model pipeline.

Generates synthetic data, engineers features, trains XGBoost + Logistic
Regression models, evaluates them, and saves charts + a model artifact.

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
    add_expected_points,
    get_feature_importance,
    get_calibration_data,
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
# 3. Train models
# ---------------------------------------------------------------------------
print("Step 3 – Training models …")
xgb_artifacts   = train_model(shots, model_type="xgboost")
lr_artifacts    = train_model(shots, model_type="logistic")

print("\n  XGBoost metrics:")
for k, v in xgb_artifacts.metrics.items():
    print(f"    {k:>15s}: {v:.4f}")

print("\n  Logistic Regression metrics:")
for k, v in lr_artifacts.metrics.items():
    print(f"    {k:>15s}: {v:.4f}")

# Save XGBoost as the primary model
model_path = save_artifacts(xgb_artifacts, "models/xpts_model.pkl")
print(f"\n  Model artifact saved → {model_path}")

# Attach xpts predictions to the full dataset
shots = add_expected_points(shots, xgb_artifacts.pipeline, xgb_artifacts.feature_columns)
shots.to_csv(processed_path, index=False)

# ---------------------------------------------------------------------------
# 4. Charts
# ---------------------------------------------------------------------------
print("\nStep 4 – Generating charts …")


# ── 4a. NBA half-court shot chart coloured by xPTS ─────────────────────────
def draw_half_court(ax: plt.Axes, color: str = "#aaaaaa") -> None:
    """Draw a simplified NBA half-court outline on *ax*."""
    from matplotlib.patches import Arc, Circle, Rectangle
    # Court border
    ax.add_patch(Rectangle((-250, -52), 500, 940, fill=False, color=color, lw=1.5))
    # Paint
    ax.add_patch(Rectangle((-80, -52), 160, 190, fill=False, color=color, lw=1.2))
    ax.add_patch(Rectangle((-60, -52), 120, 190, fill=False, color=color, lw=0.8))
    # Basket
    ax.add_patch(Circle((0, 0), 7.5, fill=False, color=color, lw=1.5))
    ax.plot([-30, 30], [0, 0], color=color, lw=1.5)  # backboard
    # Free-throw circle
    ax.add_patch(Circle((0, 142), 60, fill=False, color=color, lw=1.2))
    # Three-point arc
    ax.add_patch(Arc((0, 0), 475, 475, theta1=22, theta2=158, color=color, lw=1.5))
    ax.plot([-220, -220], [-52, 90], color=color, lw=1.5)
    ax.plot([220, 220], [-52, 90], color=color, lw=1.5)
    # Half-court line
    ax.plot([-250, 250], [470, 470], color=color, lw=1.5)


fig, ax = plt.subplots(figsize=(10, 9))
ax.set_facecolor("#1a1a2e")
fig.patch.set_facecolor("#1a1a2e")
draw_half_court(ax, color="#555577")

sample = shots.sample(min(3000, len(shots)), random_state=1)
sc = ax.scatter(
    sample["loc_x"],
    sample["loc_y"],
    c=sample["xpts"],
    cmap="RdYlGn",
    s=12,
    alpha=0.7,
    vmin=0.4,
    vmax=1.6,
    linewidths=0,
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


# ── 4b. Calibration curves for both models ─────────────────────────────────
cal_xgb = get_calibration_data(xgb_artifacts, n_bins=10)
cal_lr  = get_calibration_data(lr_artifacts, n_bins=10)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration")
ax.plot(cal_xgb["mean_predicted"], cal_xgb["fraction_positive"],
        "o-", lw=2, label=f"XGBoost  (AUC={xgb_artifacts.metrics['roc_auc']:.3f})")
ax.plot(cal_lr["mean_predicted"], cal_lr["fraction_positive"],
        "s-", lw=2, label=f"Logistic (AUC={lr_artifacts.metrics['roc_auc']:.3f})")
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Fraction of Positives (Actual Make Rate)")
ax.set_title("Calibration Curves – Shot Make Probability")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUTS / "calibration_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: outputs/calibration_curves.png")


# ── 4c. Feature importance (XGBoost) ───────────────────────────────────────
fi = get_feature_importance(xgb_artifacts)
if fi is not None:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("viridis", len(fi))
    ax.barh(fi["feature"], fi["importance"], color=colors)
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title("XGBoost – Feature Importance")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUTPUTS / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/feature_importance.png")


# ── 4d. Player summary – average xPTS per player ───────────────────────────
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

# Left: average xPTS per player
palette = ["#2ecc71" if v >= 0 else "#e74c3c" for v in player_summary["xpts_vs_average"]]
axes[0].barh(player_summary["player_name"], player_summary["avg_xpts"], color=palette)
axes[0].axvline(player_summary["avg_xpts"].mean(), color="white", linestyle="--", lw=1.5,
                label=f"League avg {player_summary['avg_xpts'].mean():.3f}")
axes[0].set_xlabel("Average xPTS per Shot Attempt")
axes[0].set_title("Average xPTS by Player")
axes[0].legend()
axes[0].invert_yaxis()

# Right: scatter make_rate vs avg_xpts
for _, row in player_summary.iterrows():
    axes[1].scatter(row["avg_xpts"], row["make_rate"], s=120, zorder=3)
    axes[1].annotate(
        row["player_name"].split()[-1],
        (row["avg_xpts"], row["make_rate"]),
        xytext=(4, 2),
        textcoords="offset points",
        fontsize=8,
    )
axes[1].set_xlabel("Average xPTS")
axes[1].set_ylabel("Actual Make Rate")
axes[1].set_title("xPTS vs Actual Make Rate by Player")

plt.suptitle("Player-Level Shot Quality Summary", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUTS / "player_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: outputs/player_summary.png")


# ── 4e. xPTS distribution by shot zone ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
zone_order = (
    shots.groupby("shot_zone_basic")["xpts"]
    .median()
    .sort_values(ascending=False)
    .index.tolist()
)
sns.violinplot(
    data=shots,
    x="shot_zone_basic",
    y="xpts",
    hue="shot_zone_basic",
    order=zone_order,
    hue_order=zone_order,
    palette="Set2",
    ax=ax,
    inner="box",
    density_norm="width",
    legend=False,
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


# ── 4f. ROC comparison ─────────────────────────────────────────────────────
from sklearn.metrics import roc_curve

fig, ax = plt.subplots(figsize=(7, 6))
for arts, label in [(xgb_artifacts, "XGBoost"), (lr_artifacts, "Logistic Regression")]:
    fpr, tpr, _ = roc_curve(arts.y_test, arts.probabilities)
    ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC={arts.metrics['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1.2)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves – Shot Make Prediction")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUTS / "roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: outputs/roc_curves.png")


# ── 4g. Model metrics comparison table (printed + saved) ───────────────────
metrics_df = pd.DataFrame(
    {
        "Model": ["XGBoost", "Logistic Regression"],
        "ROC AUC": [xgb_artifacts.metrics["roc_auc"], lr_artifacts.metrics["roc_auc"]],
        "Log-Loss": [xgb_artifacts.metrics["log_loss"], lr_artifacts.metrics["log_loss"]],
        "Brier Score": [xgb_artifacts.metrics["brier_score"], lr_artifacts.metrics["brier_score"]],
    }
)
metrics_df.to_csv(OUTPUTS / "model_metrics.csv", index=False)
print("\n  Model Metrics:")
print(metrics_df.to_string(index=False))

# ── 4h. Player summary table ────────────────────────────────────────────────
player_summary.to_csv(OUTPUTS / "player_summary.csv", index=False)
print("\n  Player Summary:")
print(player_summary.to_string(index=False))

print("\n" + "=" * 60)
print("Pipeline complete. All outputs written to outputs/")
print("=" * 60)
