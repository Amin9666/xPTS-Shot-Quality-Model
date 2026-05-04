"""
run_pipeline.py – End-to-end xPTS Shot Quality Model pipeline.

Data priority:
  1. Load data/raw/shots.csv if it exists and looks like real NBA data
     (row count > 5,000 or contains game_id / player_id columns).
  2. Fetch real NBA data via nba_api for all 30 teams (2023-24 season).
  3. Fall back to synthetic Curry data (all chart titles marked SYNTHETIC DATA).

Engineers features, trains XGBoost + Logistic Regression models, evaluates them
with rigorous ML metrics (ROC-AUC, PR-AUC, log-loss, Brier score, ECE), runs
stratified k-fold cross-validation, performs randomised hyperparameter search,
and saves a rich set of diagnostic charts + a model artifact.

Usage:
    python run_pipeline.py
"""
from __future__ import annotations

import os
import sys
import time
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
# 1. Data  – priority: saved CSV → full-league nba_api fetch → synthetic
# ---------------------------------------------------------------------------
print("=" * 60)
print("Step 1 – Loading NBA shot data …")
raw_path = Path("data/raw/shots.csv")
raw_path.parent.mkdir(parents=True, exist_ok=True)

_REAL_DATA_MIN_ROWS = 5_000  # anything above this threshold is treated as real data
_EPSILON = 1e-9              # small value to avoid atan2 singularity at x=0
_RETRY_DELAY_SECS = 3.0      # seconds to wait between nba_api retry attempts
_TEAM_REQUEST_DELAY_SECS = 1.5  # seconds to wait between team requests (rate limiting)


def _looks_like_real_data(df: pd.DataFrame) -> bool:
    """Return True if *df* appears to be real NBA data (not synthetic)."""
    if len(df) > _REAL_DATA_MIN_ROWS:
        return True
    real_cols = {"game_id", "player_id"}
    return bool(real_cols.intersection(df.columns))


def _raise_if_ci() -> None:
    """Raise EnvironmentError when running inside a CI/cloud environment.

    GitHub Actions and Codespaces both set the ``CI`` environment variable to
    ``"true"``.  ``stats.nba.com`` blocks requests from datacenter IP ranges,
    so attempting the nba_api fetch in those environments always times out.
    Raising here lets the surrounding ``except`` block fall through to the
    synthetic-data fallback immediately.
    """
    if os.getenv("CI"):
        raise EnvironmentError(
            "Running in CI/cloud environment — skipping nba_api fetch "
            "(stats.nba.com blocks datacenter IPs). Using synthetic data fallback."
        )


def _normalise_league_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a full-league CSV (notebook 01 schema) to the pipeline schema."""
    # Lowercase all column names (notebook 01 already does this, but be defensive)
    df.columns = df.columns.str.lower()

    # event_type → shot_result
    if "event_type" in df.columns and "shot_result" not in df.columns:
        df = df.rename(columns={"event_type": "shot_result"})

    # action_type → shot_type (if shot_type missing)
    if "shot_type" not in df.columns and "action_type" in df.columns:
        df["shot_type"] = df["action_type"]

    # Derive shot_value from shot_type
    if "shot_value" not in df.columns:
        df["shot_value"] = df["shot_type"].apply(lambda t: 3 if "3PT" in str(t) else 2)

    # Derive shot_angle
    if "shot_angle" not in df.columns:
        safe_x = df["loc_x"].replace(0, _EPSILON)
        df["shot_angle"] = np.degrees(np.arctan2(df["loc_y"], safe_x))

    # Placeholder columns not present in the league CSV
    for col, default in [
        ("score_diff", 0),
        ("shot_clock", 12.0),
        ("home_score", 0),
        ("away_score", 0),
        ("true_make_prob", 0.0),
    ]:
        if col not in df.columns:
            df[col] = default

    # Use shot_made_flag as a proxy for true_make_prob when unavailable
    if (df["true_make_prob"] == 0).all() and "shot_made_flag" in df.columns:
        df["true_make_prob"] = df["shot_made_flag"].astype(float)

    return df


def _fetch_full_league(season: str = "2023-24") -> pd.DataFrame:
    """Fetch shot-chart data for all 30 NBA teams via nba_api."""
    from nba_api.stats.endpoints.shotchartdetail import ShotChartDetail  # type: ignore[import]
    from nba_api.stats.static import teams as nba_teams  # type: ignore[import]

    all_teams = nba_teams.get_teams()
    frames: list[pd.DataFrame] = []
    for i, team in enumerate(all_teams, start=1):
        tid = team["id"]
        print(f"    Fetching team {i}/{len(all_teams)}: {team['full_name']} …")
        for attempt in range(3):
            try:
                sc = ShotChartDetail(
                    team_id=tid,
                    player_id=0,
                    season_nullable=season,
                    season_type_all_star="Regular Season",
                    context_measure_simple="FGA",
                )
                df = sc.get_data_frames()[0]
                frames.append(df)
                break
            except Exception as exc:
                print(f"      attempt {attempt + 1} failed: {exc}", file=sys.stderr)
                if attempt < 2:
                    time.sleep(_RETRY_DELAY_SECS)
        else:
            print(f"      Skipping {team['full_name']} after 3 failures.", file=sys.stderr)
        time.sleep(_TEAM_REQUEST_DELAY_SECS)

    if not frames:
        raise RuntimeError("nba_api returned no data for any team")

    result = pd.concat(frames, ignore_index=True)
    result.columns = result.columns.str.lower()
    return result


using_real_data = True
shots_raw: pd.DataFrame

# ── Priority 1: existing data/raw/shots.csv ──────────────────────────────────
if raw_path.exists():
    _candidate = pd.read_csv(raw_path)
    if _looks_like_real_data(_candidate):
        shots_raw = _normalise_league_csv(_candidate)
        chart_title = "NBA 2023-24 League Shot Chart – Coloured by xPTS"
        print(
            f"  ✓ Data source: REAL NBA data (data/raw/shots.csv) — "
            f"{len(shots_raw):,} shots loaded"
        )
    else:
        print(
            f"  data/raw/shots.csv exists but looks synthetic ({len(_candidate):,} rows). "
            "Trying nba_api …"
        )

        # ── Priority 2: fetch full league via nba_api ─────────────────────────
        try:
            _raise_if_ci()
            print("  Fetching full 2023-24 league shot data via nba_api (all 30 teams) …")
            shots_raw = _fetch_full_league(season="2023-24")
            shots_raw = _normalise_league_csv(shots_raw)
            shots_raw.to_csv(raw_path, index=False)
            chart_title = "NBA 2023-24 League Shot Chart – Coloured by xPTS"
            print(
                f"  ✓ Data source: REAL NBA data (nba_api, all teams) — "
                f"{len(shots_raw):,} shots fetched and saved → {raw_path}"
            )
        except Exception as exc:
            print(
                f"  WARNING: nba_api fetch failed ({type(exc).__name__}: {exc}). "
                "Falling back to synthetic data.",
                file=sys.stderr,
            )
            from src.generate_synthetic_data import generate_curry_shots
            shots_raw = generate_curry_shots()
            chart_title = "Stephen Curry Shot Chart – Coloured by xPTS (SYNTHETIC DATA)"
            using_real_data = False
            print(f"  ✓ Data source: SYNTHETIC Curry data — {len(shots_raw):,} shots generated")
else:
    # ── Priority 2: fetch full league via nba_api ─────────────────────────────
    try:
        _raise_if_ci()
        print("  data/raw/shots.csv not found. Fetching full 2023-24 league data via nba_api …")
        shots_raw = _fetch_full_league(season="2023-24")
        shots_raw = _normalise_league_csv(shots_raw)
        shots_raw.to_csv(raw_path, index=False)
        chart_title = "NBA 2023-24 League Shot Chart – Coloured by xPTS"
        print(
            f"  ✓ Data source: REAL NBA data (nba_api, all teams) — "
            f"{len(shots_raw):,} shots fetched and saved → {raw_path}"
        )
    except Exception as exc:
        # ── Priority 3: synthetic fallback ────────────────────────────────────
        print(
            f"  WARNING: nba_api fetch failed ({type(exc).__name__}: {exc}). "
            "Falling back to synthetic data.",
            file=sys.stderr,
        )
        from src.generate_synthetic_data import generate_curry_shots
        shots_raw = generate_curry_shots()
        chart_title = "Stephen Curry Shot Chart – Coloured by xPTS (SYNTHETIC DATA)"
        using_real_data = False
        print(f"  ✓ Data source: SYNTHETIC Curry data — {len(shots_raw):,} shots generated")

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

# For the full-league dataset sample up to 3,000 shots for readability.
# For synthetic data restrict to Curry's shots only.
if using_real_data:
    chart_shots = shots
else:
    if shots["player_name"].nunique() > 1:
        chart_shots = shots[shots["player_name"] == "Stephen Curry"]
        chart_title = "Stephen Curry Shot Chart – Coloured by xPTS (SYNTHETIC DATA)"
    else:
        chart_shots = shots

sample = chart_shots.sample(min(3000, len(chart_shots)), random_state=1)
sc = ax.scatter(
    sample["loc_x"], sample["loc_y"],
    c=sample["xpts"], cmap="RdYlGn",
    s=18, alpha=0.75, vmin=0.4, vmax=1.6, linewidths=0,
)
plt.colorbar(sc, ax=ax, label="Expected Points (xPTS)", fraction=0.03, pad=0.02)
ax.set_xlim(-260, 260)
ax.set_ylim(-60, 500)
ax.set_aspect("equal")
ax.set_title(chart_title, color="white", fontsize=14, pad=12)
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
_cal_title = "Calibration Curves – Shot Make Probability"
if not using_real_data:
    _cal_title += " (SYNTHETIC DATA)"
ax.set_title(_cal_title)
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
_roc_title = "ROC Curves – Shot Make Prediction"
if not using_real_data:
    _roc_title += " (SYNTHETIC DATA)"
ax.set_title(_roc_title)
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
_pr_title = "Precision-Recall Curves – Shot Make Prediction"
if not using_real_data:
    _pr_title += " (SYNTHETIC DATA)"
ax.set_title(_pr_title)
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
    _fi_title = "XGBoost – Impurity-Based Feature Importance"
    if not using_real_data:
        _fi_title += " (SYNTHETIC DATA)"
    ax.set_title(_fi_title)
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
    _perm_title = "XGBoost – Permutation Feature Importance (Test Set)"
    if not using_real_data:
        _perm_title += " (SYNTHETIC DATA)"
    ax.set_title(_perm_title)
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
_lc_title = "Learning Curves – XGBoost (Bias-Variance Diagnostic)"
if not using_real_data:
    _lc_title += " (SYNTHETIC DATA)"
ax.set_title(_lc_title)
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

if player_summary.shape[0] > 1:
    # Multi-player comparison – limit to top 20 by shot volume for readability
    top20 = (
        player_summary.nlargest(20, "shots_taken")
        .sort_values("avg_xpts", ascending=False)
        .reset_index(drop=True)
    )
    top20["xpts_vs_average"] = top20["avg_xpts"] - player_summary["avg_xpts"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    palette = ["#2ecc71" if v >= 0 else "#e74c3c" for v in top20["xpts_vs_average"]]
    axes[0].barh(top20["player_name"], top20["avg_xpts"], color=palette)
    axes[0].axvline(player_summary["avg_xpts"].mean(), color="white", linestyle="--", lw=1.5,
                    label=f"Full league avg {player_summary['avg_xpts'].mean():.3f}")
    axes[0].set_xlabel("Average xPTS per Shot Attempt")
    axes[0].set_title("Average xPTS by Player (Top 20 by Volume)")
    axes[0].legend()
    axes[0].invert_yaxis()

    for _, row in top20.iterrows():
        axes[1].scatter(row["avg_xpts"], row["make_rate"], s=120, zorder=3)
        axes[1].annotate(
            row["player_name"].split()[-1],
            (row["avg_xpts"], row["make_rate"]),
            xytext=(4, 2), textcoords="offset points", fontsize=8,
        )
    axes[1].set_xlabel("Average xPTS")
    axes[1].set_ylabel("Actual Make Rate")
    axes[1].set_title("xPTS vs Actual Make Rate by Player (Top 20 by Volume)")

    _ps_suptitle = "Player-Level Shot Quality Summary"
    if not using_real_data:
        _ps_suptitle += " (SYNTHETIC DATA)"
    plt.suptitle(_ps_suptitle, fontsize=14, y=1.02)
else:
    # Single-player view: show shot selection (volume) and quality by zone
    player_name = player_summary["player_name"].iloc[0]
    zone_summary = (
        shots.groupby("shot_zone_basic")
        .agg(
            shots_taken=("xpts", "count"),
            avg_xpts=("xpts", "mean"),
            make_rate=("shot_made_flag", "mean"),
        )
        .sort_values("shots_taken", ascending=False)
        .reset_index()
    )
    zone_colors = sns.color_palette("Set2", len(zone_summary))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].barh(zone_summary["shot_zone_basic"], zone_summary["shots_taken"], color=zone_colors)
    axes[0].set_xlabel("Shots Taken")
    axes[0].set_title("Shot Volume by Zone (Selection)")
    axes[0].invert_yaxis()

    axes[1].barh(zone_summary["shot_zone_basic"], zone_summary["avg_xpts"], color=zone_colors)
    axes[1].set_xlabel("Average xPTS per Shot Attempt")
    axes[1].set_title("Shot Quality (avg xPTS) by Zone")
    axes[1].invert_yaxis()

    plt.suptitle(f"{player_name} – Shot Selection & Quality by Zone"
                 + (" (SYNTHETIC DATA)" if not using_real_data else ""),
                 fontsize=14, y=1.02)

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
_zone_title = "xPTS Distribution by Shot Zone"
if not using_real_data:
    _zone_title += " (SYNTHETIC DATA)"
ax.set_title(_zone_title)
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