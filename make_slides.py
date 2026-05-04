"""
make_slides.py  –  Generate presentation-ready slide charts for the xPTS model.

Reads pre-computed CSVs and the trained model artifact from outputs/ and
models/, then writes polished, high-DPI PNG slides to outputs/slides/.

Usage (run AFTER run_pipeline.py):
    python make_slides.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Arc, Circle, Rectangle
import numpy as np
import pandas as pd
import seaborn as sns
import joblib

# ---------------------------------------------------------------------------
# Global presentation style
# ---------------------------------------------------------------------------
SLIDE_DIR = Path("outputs/slides")
SLIDE_DIR.mkdir(parents=True, exist_ok=True)

TITLE_SIZE  = 22
LABEL_SIZE  = 16
TICK_SIZE   = 13
LEGEND_SIZE = 13
DPI         = 180
BG         = "white"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    "#f7f7f7",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    TITLE_SIZE,
    "axes.labelsize":    LABEL_SIZE,
    "xtick.labelsize":   TICK_SIZE,
    "ytick.labelsize":   TICK_SIZE,
    "legend.fontsize":   LEGEND_SIZE,
    "axes.titlepad":     16,
})

PALETTE = sns.color_palette("tab10")
XGB_COLOR  = PALETTE[0]   # blue
LR_COLOR   = PALETTE[1]   # orange

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data …")
shots         = pd.read_csv("data/processed/shots_model_input.csv")
metrics_df    = pd.read_csv("outputs/model_metrics.csv")
player_df     = pd.read_csv("outputs/player_summary.csv")
perm_df       = pd.read_csv("outputs/permutation_importance.csv")
cv_df         = pd.read_csv("outputs/cv_results_xgboost.csv")
artifact      = joblib.load("models/xpts_model.pkl")
pipeline      = artifact["pipeline"]
feature_cols  = artifact["feature_columns"]

shots["make_probability"] = pipeline.predict_proba(shots[feature_cols])[:, 1]
if "xpts" not in shots.columns:
    shots["xpts"] = shots["make_probability"] * shots["shot_value"].fillna(2)

# Reconstruct test-split for curves (deterministic split, same seed as training)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    log_loss, brier_score_loss,
)

# We need a quick re-train of the logistic model to get its test probabilities
# for the ROC / PR / calibration slides.  Re-import training helpers.
from src.model import train_model, get_calibration_data, get_roc_data, get_pr_curve_data

print("Re-training logistic for slide curves …")
xgb_arts = train_model(shots, model_type="xgboost")
lr_arts  = train_model(shots, model_type="logistic")


# ===========================================================================
# Slide 1 – Shot Chart (xPTS)
# ===========================================================================
def draw_half_court(ax: plt.Axes, color: str = "#bbbbbb") -> None:
    ax.add_patch(Rectangle((-250, -52), 500, 940, fill=False, color=color, lw=2))
    ax.add_patch(Rectangle((-80, -52),  160, 190, fill=False, color=color, lw=1.5))
    ax.add_patch(Rectangle((-60, -52),  120, 190, fill=False, color=color, lw=1))
    ax.add_patch(Circle((0, 0), 7.5,   fill=False, color=color, lw=2))
    ax.plot([-30, 30], [0, 0], color=color, lw=2)
    ax.add_patch(Circle((0, 142), 60,  fill=False, color=color, lw=1.5))
    ax.add_patch(Arc((0, 0), 475, 475, theta1=22, theta2=158, color=color, lw=2))
    ax.plot([-220, -220], [-52, 90], color=color, lw=2)
    ax.plot([ 220,  220], [-52, 90], color=color, lw=2)
    ax.plot([-250, 250], [470, 470], color=color, lw=1.5)


print("Slide 1 – Shot Chart …")
fig, ax = plt.subplots(figsize=(11, 10))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#1a1a2e")
draw_half_court(ax, color="#555577")

sample = shots.sample(min(3500, len(shots)), random_state=1)
sc = ax.scatter(
    sample["loc_x"], sample["loc_y"],
    c=sample["xpts"], cmap="RdYlGn",
    s=14, alpha=0.75, vmin=0.4, vmax=1.6, linewidths=0,
)
cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Expected Points (xPTS)", color="white", fontsize=LABEL_SIZE)
cbar.ax.yaxis.set_tick_params(color="white", labelsize=TICK_SIZE)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
ax.set_xlim(-265, 265); ax.set_ylim(-65, 510); ax.set_aspect("equal")
ax.set_title("NBA Shot Chart – Coloured by Expected Points (xPTS)",
             color="white", fontsize=TITLE_SIZE, pad=14)
ax.axis("off")

# Annotate the three zones
for txt, x, y in [("Corner 3\n(High xPTS)", -240, 30),
                   ("Restricted Area\n(Highest xPTS)", 0, -45),
                   ("Mid-Range\n(Lower xPTS)", 130, 180)]:
    ax.text(x, y, txt, color="white", fontsize=10, ha="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="#1a1a2e", ec="white", alpha=0.7))

plt.tight_layout()
fig.savefig(SLIDE_DIR / "01_shot_chart.png", dpi=DPI, bbox_inches="tight",
            facecolor="#1a1a2e")
plt.close()
print("  → 01_shot_chart.png")


# ===========================================================================
# Slide 2 – ROC Curves
# ===========================================================================
print("Slide 2 – ROC Curves …")
fig, ax = plt.subplots(figsize=(9, 7))
for arts, label, color in [
    (xgb_arts, "XGBoost",             XGB_COLOR),
    (lr_arts,  "Logistic Regression", LR_COLOR),
]:
    roc = get_roc_data(arts)
    ax.plot(roc["fpr"], roc["tpr"], lw=2.5, color=color,
            label=f"{label}  (AUC = {arts.metrics['roc_auc']:.3f})")

ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random classifier (AUC = 0.500)")
ax.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")
ax.set_xlabel("False Positive Rate  (1 – Specificity)")
ax.set_ylabel("True Positive Rate  (Sensitivity / Recall)")
ax.set_title("ROC Curves  –  Shot Make Probability")
ax.legend(loc="lower right")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
# Add AUC annotation box
ax.text(0.60, 0.15,
        "AUC closer to 1.0 = better\nAUC = 0.5 = coin flip",
        fontsize=12, transform=ax.transAxes,
        bbox=dict(boxstyle="round", fc="lightyellow", ec="#cccc00", alpha=0.85))
plt.tight_layout()
fig.savefig(SLIDE_DIR / "02_roc_curves.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("  → 02_roc_curves.png")


# ===========================================================================
# Slide 3 – Precision-Recall Curves
# ===========================================================================
print("Slide 3 – PR Curves …")
baseline = xgb_arts.y_test.mean()
fig, ax = plt.subplots(figsize=(9, 7))
for arts, label, color in [
    (xgb_arts, "XGBoost",             XGB_COLOR),
    (lr_arts,  "Logistic Regression", LR_COLOR),
]:
    pr = get_pr_curve_data(arts)
    ax.plot(pr["recall"], pr["precision"], lw=2.5, color=color,
            label=f"{label}  (PR-AUC = {arts.metrics['pr_auc']:.3f})")

ax.axhline(baseline, color="gray", linestyle="--", lw=1.5,
           label=f"No-skill baseline  ({baseline:.3f})")
ax.set_xlabel("Recall  (True Positive Rate)")
ax.set_ylabel("Precision  (Positive Predictive Value)")
ax.set_title("Precision-Recall Curves  –  Shot Make Probability")
ax.legend(loc="upper right")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.text(0.05, 0.10,
        "PR curve is more informative\nwhen class balance matters",
        fontsize=12, transform=ax.transAxes,
        bbox=dict(boxstyle="round", fc="lightyellow", ec="#cccc00", alpha=0.85))
plt.tight_layout()
fig.savefig(SLIDE_DIR / "03_pr_curves.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("  → 03_pr_curves.png")


# ===========================================================================
# Slide 4 – Calibration Curves
# ===========================================================================
print("Slide 4 – Calibration …")
cal_xgb = get_calibration_data(xgb_arts, n_bins=10)
cal_lr  = get_calibration_data(lr_arts,  n_bins=10)

fig, ax = plt.subplots(figsize=(9, 7))
ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration", zorder=1)
ax.fill_between([0, 1], [0, 1], alpha=0.04, color="gray")

ax.plot(cal_xgb["mean_predicted"], cal_xgb["fraction_positive"],
        "o-", lw=2.5, ms=8, color=XGB_COLOR,
        label=(f"XGBoost  "
               f"(AUC={xgb_arts.metrics['roc_auc']:.3f}, "
               f"ECE={xgb_arts.metrics['ece']:.3f})"))
ax.plot(cal_lr["mean_predicted"], cal_lr["fraction_positive"],
        "s-", lw=2.5, ms=8, color=LR_COLOR,
        label=(f"Logistic  "
               f"(AUC={lr_arts.metrics['roc_auc']:.3f}, "
               f"ECE={lr_arts.metrics['ece']:.3f})"))

ax.set_xlabel("Mean Predicted Probability (Confidence)")
ax.set_ylabel("Fraction of Positives (Actual Make Rate)")
ax.set_title("Calibration Curves  –  How Reliable Are the Probabilities?")
ax.legend(loc="upper left")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.text(0.55, 0.08,
        "Points on diagonal = perfectly calibrated\nECE closer to 0 = better",
        fontsize=12, transform=ax.transAxes,
        bbox=dict(boxstyle="round", fc="lightyellow", ec="#cccc00", alpha=0.85))
plt.tight_layout()
fig.savefig(SLIDE_DIR / "04_calibration.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("  → 04_calibration.png")


# ===========================================================================
# Slide 5 – Learning Curves
# ===========================================================================
print("Slide 5 – Learning Curves …")
lc = pd.read_csv("outputs/learning_curves.csv") if Path("outputs/learning_curves.csv").exists() else None

# Re-compute if CSV was not saved previously
if lc is None:
    from src.model import get_learning_curve_data
    lc = get_learning_curve_data(shots, model_type="xgboost", n_splits=5)
    lc.to_csv("outputs/learning_curves.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 7))
ax.fill_between(lc["train_size"],
                lc["train_mean"] - lc["train_std"],
                lc["train_mean"] + lc["train_std"],
                alpha=0.15, color=XGB_COLOR)
ax.fill_between(lc["train_size"],
                lc["test_mean"] - lc["test_std"],
                lc["test_mean"] + lc["test_std"],
                alpha=0.15, color=LR_COLOR)
ax.plot(lc["train_size"], lc["train_mean"], "o-", lw=2.5, color=XGB_COLOR,
        label="Training ROC-AUC")
ax.plot(lc["train_size"], lc["test_mean"],  "s-", lw=2.5, color=LR_COLOR,
        label="CV Validation ROC-AUC")

ax.set_xlabel("Number of Training Samples")
ax.set_ylabel("ROC-AUC")
ax.set_title("Learning Curves  –  Bias vs. Variance Diagnostic  (XGBoost)")
ax.legend(loc="lower right")
ax.set_ylim(0.48, 0.75)

# Annotate gap
gap_y = (lc["train_mean"].iloc[-1] + lc["test_mean"].iloc[-1]) / 2
ax.annotate("← Gap = variance\n   (over-fitting)",
            xy=(lc["train_size"].iloc[-1], gap_y),
            xytext=(lc["train_size"].iloc[-3], gap_y + 0.06),
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=11)
plt.tight_layout()
fig.savefig(SLIDE_DIR / "05_learning_curves.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("  → 05_learning_curves.png")


# ===========================================================================
# Slide 6 – Permutation Feature Importance
# ===========================================================================
print("Slide 6 – Permutation Importance …")
perm_df_sorted = perm_df.sort_values("importance_mean", ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in perm_df_sorted["importance_mean"]]
bars = ax.barh(perm_df_sorted["feature"], perm_df_sorted["importance_mean"],
               xerr=perm_df_sorted["importance_std"],
               color=colors, capsize=4, edgecolor="white", linewidth=0.5)
ax.axvline(0, color="black", lw=1, linestyle="--")
ax.set_xlabel("Mean Decrease in ROC-AUC\n(when feature values are randomly shuffled)")
ax.set_title("Permutation Feature Importance  –  Test Set  (XGBoost)")

# Add value labels
for bar, val, std in zip(bars, perm_df_sorted["importance_mean"], perm_df_sorted["importance_std"]):
    if val >= 0:
        ax.text(val + std + 0.001, bar.get_y() + bar.get_height() / 2,
                f"+{val:.4f}", va="center", fontsize=9)
    else:
        ax.text(0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)

plt.tight_layout()
fig.savefig(SLIDE_DIR / "06_permutation_importance.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("  → 06_permutation_importance.png")


# ===========================================================================
# Slide 7 – Cross-Validation Results (bar chart of folds)
# ===========================================================================
print("Slide 7 – CV Results …")
cv_folds = cv_df[cv_df["fold"].apply(lambda x: str(x).isdigit())].copy()
cv_folds["fold"] = cv_folds["fold"].astype(int)
mean_row = cv_df[cv_df["fold"] == "mean"].iloc[0]
std_row  = cv_df[cv_df["fold"] == "std"].iloc[0]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: ROC-AUC per fold
x = cv_folds["fold"]
axes[0].bar(x, cv_folds["roc_auc"], color=XGB_COLOR, alpha=0.8, edgecolor="white", width=0.6)
axes[0].axhline(float(mean_row["roc_auc"]), color="red", lw=2, linestyle="--",
                label=f"Mean = {float(mean_row['roc_auc']):.3f}")
axes[0].fill_between(
    [0.5, 5.5],
    float(mean_row["roc_auc"]) - float(std_row["roc_auc"]),
    float(mean_row["roc_auc"]) + float(std_row["roc_auc"]),
    color="red", alpha=0.10, label=f"± 1 std ({float(std_row['roc_auc']):.3f})"
)
axes[0].set_xlabel("Fold")
axes[0].set_ylabel("ROC-AUC")
axes[0].set_title("5-Fold CV  –  ROC-AUC per Fold  (XGBoost)")
axes[0].set_ylim(0.50, 0.70)
axes[0].legend()
axes[0].set_xticks(x)

# Right: all metrics grouped bar
metrics_show = ["roc_auc", "pr_auc", "brier_score", "ece"]
labels_show  = ["ROC-AUC", "PR-AUC", "Brier Score", "ECE"]
vals = [float(mean_row[m]) for m in metrics_show]
errs = [float(std_row[m])  for m in metrics_show]
bar_colors = [XGB_COLOR, LR_COLOR, PALETTE[2], PALETTE[3]]
b = axes[1].bar(labels_show, vals, yerr=errs, capsize=6, color=bar_colors,
                edgecolor="white", alpha=0.85)
axes[1].set_ylabel("Score  (mean ± std over 5 folds)")
axes[1].set_title("5-Fold CV  –  All Metrics  (XGBoost, mean ± std)")
for bar, v, e in zip(b, vals, errs):
    axes[1].text(bar.get_x() + bar.get_width() / 2, v + e + 0.003,
                 f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")

plt.tight_layout()
fig.savefig(SLIDE_DIR / "07_cv_results.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("  → 07_cv_results.png")


# ===========================================================================
# Slide 8 – Player Shot Quality
# ===========================================================================
print("Slide 8 – Player Summary …")
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

player_df_sorted = player_df.sort_values("avg_xpts", ascending=True)
colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in player_df_sorted["xpts_vs_average"]]
axes[0].barh(player_df_sorted["player_name"], player_df_sorted["avg_xpts"],
             color=colors, edgecolor="white", linewidth=0.5)
axes[0].axvline(player_df["avg_xpts"].mean(), color="black", lw=2, linestyle="--",
                label=f"League avg  {player_df['avg_xpts'].mean():.3f}")
axes[0].set_xlabel("Average Expected Points per Shot Attempt")
axes[0].set_title("Shot Quality by Player  (avg xPTS)")
axes[0].legend()

for _, row in player_df.iterrows():
    axes[1].scatter(row["avg_xpts"], row["make_rate"],
                    s=140, zorder=3, color=XGB_COLOR, edgecolors="white", linewidths=1)
    axes[1].annotate(row["player_name"].split()[-1],
                     (row["avg_xpts"], row["make_rate"]),
                     xytext=(5, 3), textcoords="offset points", fontsize=10)

# Add a 45-degree reference line (if xPTS were perfectly predictive)
xmin, xmax = player_df["avg_xpts"].min() - 0.05, player_df["avg_xpts"].max() + 0.05
axes[1].set_xlabel("Average xPTS")
axes[1].set_ylabel("Actual Field Goal Make Rate")
axes[1].set_title("xPTS vs Actual Make Rate by Player")

plt.suptitle("Player-Level Shot Quality Summary", fontsize=TITLE_SIZE + 2, y=1.01,
             fontweight="bold")
plt.tight_layout()
fig.savefig(SLIDE_DIR / "08_player_quality.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("  → 08_player_quality.png")


# ===========================================================================
# Slide 9 – xPTS Distribution by Shot Zone
# ===========================================================================
print("Slide 9 – xPTS by Zone …")
zone_order = (
    shots.groupby("shot_zone_basic")["xpts"]
    .median().sort_values(ascending=False).index.tolist()
)

fig, ax = plt.subplots(figsize=(12, 7))
sns.violinplot(
    data=shots, x="shot_zone_basic", y="xpts",
    hue="shot_zone_basic", order=zone_order, hue_order=zone_order,
    palette="Set2", ax=ax, inner="box", density_norm="width", legend=False,
)
ax.set_xticks(range(len(zone_order)))
ax.set_xticklabels(zone_order, rotation=28, ha="right", fontsize=TICK_SIZE)
ax.set_title("Distribution of Expected Points (xPTS) by Shot Zone")
ax.set_xlabel("Shot Zone")
ax.set_ylabel("Expected Points (xPTS)")

# Add median labels
for i, zone in enumerate(zone_order):
    med = shots[shots["shot_zone_basic"] == zone]["xpts"].median()
    ax.text(i, med + 0.04, f"{med:.2f}", ha="center", fontsize=10,
            fontweight="bold", color="black")

plt.tight_layout()
fig.savefig(SLIDE_DIR / "09_xpts_by_zone.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("  → 09_xpts_by_zone.png")


# ===========================================================================
# Slide 10 – Model Metrics Comparison (clean table chart)
# ===========================================================================
print("Slide 10 – Metrics Table …")
metric_labels = {
    "roc_auc":     "ROC-AUC ↑",
    "pr_auc":      "PR-AUC ↑",
    "log_loss":    "Log-Loss ↓",
    "brier_score": "Brier Score ↓",
    "ece":         "ECE ↓",
}
models_list  = ["XGBoost", "Logistic Regression"]
metric_keys  = list(metric_labels.keys())
metric_names = list(metric_labels.values())

fig, ax = plt.subplots(figsize=(12, 4.5))
ax.axis("off")

xgb_row = metrics_df[metrics_df["Model"] == "XGBoost"].iloc[0]
lr_row  = metrics_df[metrics_df["Model"] == "Logistic Regression"].iloc[0]
cell_data = [
    [f"{xgb_row[k]:.4f}" for k in metric_keys],
    [f"{lr_row[k]:.4f}"  for k in metric_keys],
]

table = ax.table(
    cellText=cell_data,
    rowLabels=models_list,
    colLabels=metric_names,
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(14)
table.scale(1.4, 2.8)

# Header row styling
for j in range(len(metric_keys)):
    table[(0, j)].set_facecolor("#2c3e50")
    table[(0, j)].set_text_props(color="white", fontweight="bold")

# XGBoost row
for j in range(len(metric_keys)):
    table[(1, j)].set_facecolor("#d6eaf8")

# Logistic row
for j in range(len(metric_keys)):
    table[(2, j)].set_facecolor("#d5f5e3")

# Row labels
for i, lbl in enumerate(models_list, 1):
    table[(i, -1)].set_facecolor("#ecf0f1")
    table[(i, -1)].set_text_props(fontweight="bold")

ax.set_title("Hold-Out Test Set  –  Model Metrics Comparison", fontsize=TITLE_SIZE,
             pad=20, fontweight="bold")
plt.tight_layout()
fig.savefig(SLIDE_DIR / "10_metrics_table.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("  → 10_metrics_table.png")


# ===========================================================================
# Slide 11 – Pipeline Overview Diagram
# ===========================================================================
print("Slide 11 – Pipeline Diagram …")
fig, ax = plt.subplots(figsize=(14, 4))
ax.axis("off")
ax.set_facecolor(BG)
fig.patch.set_facecolor(BG)

steps = [
    ("1. Synthetic\nData Generation", "#3498db"),
    ("2. Feature\nEngineering", "#9b59b6"),
    ("3. Leakage-Free\nCV (5-fold)", "#e67e22"),
    ("4. Hyper-\nparameter Tuning", "#e74c3c"),
    ("5. Model\nTraining", "#27ae60"),
    ("6. Evaluation &\nCharts", "#1abc9c"),
]

box_w, box_h = 0.13, 0.50
gap = 0.155
start_x = 0.02

for i, (label, color) in enumerate(steps):
    x = start_x + i * gap
    rect = mpatches.FancyBboxPatch(
        (x, 0.25), box_w, box_h,
        boxstyle="round,pad=0.02",
        facecolor=color, edgecolor="white", linewidth=1.5,
        transform=ax.transAxes, clip_on=False,
    )
    ax.add_patch(rect)
    ax.text(x + box_w / 2, 0.25 + box_h / 2, label,
            ha="center", va="center", fontsize=11, color="white",
            fontweight="bold", transform=ax.transAxes, clip_on=False)
    if i < len(steps) - 1:
        ax.annotate("", xy=(x + box_w + gap - box_w + 0.005, 0.50),
                    xytext=(x + box_w + 0.005, 0.50),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color="#555555", lw=2))

ax.set_title("xPTS Model Pipeline  –  End-to-End Overview",
             fontsize=TITLE_SIZE, fontweight="bold", pad=8)
plt.tight_layout()
fig.savefig(SLIDE_DIR / "11_pipeline_overview.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("  → 11_pipeline_overview.png")


print(f"\n✓  All slides saved to {SLIDE_DIR}/")
print("  Files:")
for f in sorted(SLIDE_DIR.iterdir()):
    print(f"    {f.name}")
