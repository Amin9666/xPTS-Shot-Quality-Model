from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance as sklearn_permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    learning_curve,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


FEATURE_COLUMNS = [
    "shot_distance",
    "shot_angle",
    "distance_sq",
    "log1p_distance",
    "dist_angle_ix",
    "period",
    "game_seconds_remaining",
    "score_diff_abs",
    "player_zone_fg_pct",
    "late_clock",
    "shot_clock",
]

# Three feature tiers for ablation study
# Each tier adds a new group of predictors so the incremental value
# of geometry → game context → player skill can be quantified.
FEATURE_TIERS: dict[str, list[str]] = {
    "Location Only": [
        "shot_distance",
        "shot_angle",
        "distance_sq",
        "log1p_distance",
        "dist_angle_ix",
    ],
    "Location + Context": [
        "shot_distance",
        "shot_angle",
        "distance_sq",
        "log1p_distance",
        "dist_angle_ix",
        "period",
        "game_seconds_remaining",
        "score_diff_abs",
        "late_clock",
        "shot_clock",
    ],
    "Full (+ Player Skill)": [
        "shot_distance",
        "shot_angle",
        "distance_sq",
        "log1p_distance",
        "dist_angle_ix",
        "period",
        "game_seconds_remaining",
        "score_diff_abs",
        "late_clock",
        "shot_clock",
        "player_zone_fg_pct",
    ],
}

# Auxiliary columns needed for leakage-free zone encoding – not used as
# model inputs directly, but passed alongside X so the encoder can refit
# on each training fold.
_ZONE_AUX_COLS = ["player_name", "shot_zone_basic"]

ModelType = Literal["logistic", "xgboost"]


@dataclass(slots=True)
class TrainingArtifacts:
    pipeline: Pipeline
    metrics: dict[str, float]
    feature_columns: list[str]
    model_name: str = "logistic"
    X_test: Optional[pd.DataFrame] = field(default=None)
    y_test: Optional[pd.Series] = field(default=None)
    probabilities: Optional[np.ndarray] = field(default=None)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    transformers = []
    if numeric_columns:
        transformers.append((
            "num",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_columns,
        ))
    if categorical_columns:
        transformers.append((
            "cat",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_columns,
        ))
    return ColumnTransformer(transformers=transformers)


def _build_classifier(model_type: ModelType, **overrides) -> XGBClassifier | LogisticRegression:
    """Instantiate the base classifier for a given model type."""
    if model_type == "xgboost":
        defaults = dict(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        )
        defaults.update(overrides)
        return XGBClassifier(**defaults)
    else:
        defaults = dict(max_iter=1_000, C=1.0, solver="liblinear")
        defaults.update(overrides)
        return LogisticRegression(**defaults)


def _recompute_zone_fg_pct(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    smoothing: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace the pre-computed ``player_zone_fg_pct`` with a leakage-free
    version fitted exclusively on the training fold.

    Uses Bayesian (James–Stein-style) shrinkage toward the global mean::

        θ̂_pz = (n_pz · ȳ_pz + k · ȳ_global) / (n_pz + k)

    where *k* = ``smoothing`` controls regularisation strength toward the
    global shooting rate.  Small-sample player–zone cells are pulled toward
    the league average, avoiding the over-fitting that arises from raw
    group means on rare combinations.

    Note
    ----
    Without this fix the raw ``player_zone_fg_pct`` computed on the full
    dataset leaks test-set outcomes into the training features, inflating
    all reported metrics (a classic *target leakage* bug).
    """
    if "player_zone_fg_pct" not in X_train.columns:
        return X_train, X_test
    if not {"player_name", "shot_zone_basic"}.issubset(X_train.columns):
        return X_train, X_test

    global_mean = float(y_train.mean())

    tmp = X_train[["player_name", "shot_zone_basic"]].copy()
    tmp["_y"] = y_train.values

    agg = tmp.groupby(["player_name", "shot_zone_basic"])["_y"].agg(["count", "mean"])
    agg["smoothed"] = (
        agg["count"] * agg["mean"] + smoothing * global_mean
    ) / (agg["count"] + smoothing)
    encoding: dict = agg["smoothed"].to_dict()

    def _encode(df: pd.DataFrame) -> pd.Series:
        return df.apply(
            lambda r: encoding.get((r["player_name"], r["shot_zone_basic"]), global_mean),
            axis=1,
        )

    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train["player_zone_fg_pct"] = _encode(X_train)
    X_test["player_zone_fg_pct"] = _encode(X_test)
    return X_train, X_test


def _split_and_fix_leakage(
    model_frame: pd.DataFrame,
    available_features: list[str],
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split followed by leakage-free zone encoding."""
    aux_cols = [c for c in _ZONE_AUX_COLS if c in model_frame.columns]
    X_full = model_frame[available_features + aux_cols].copy()
    y = model_frame[target_column]

    X_full_train, X_full_test, y_train, y_test = train_test_split(
        X_full, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if "player_zone_fg_pct" in available_features and aux_cols:
        X_full_train, X_full_test = _recompute_zone_fg_pct(X_full_train, X_full_test, y_train)

    X_train = X_full_train[available_features]
    X_test = X_full_test[available_features]
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Public metrics utilities
# ---------------------------------------------------------------------------

def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE).

    ECE is the probability-weighted mean absolute deviation between a
    model's predicted confidence and its empirical accuracy across *n_bins*
    equal-width probability intervals::

        ECE = Σ_b (|B_b| / n) · |acc(B_b) − conf(B_b)|

    ECE ∈ [0, 1]; lower is better. A perfectly calibrated model has ECE = 0.
    Calibration matters whenever predicted probabilities are interpreted
    directly (e.g. computing expected points as p̂ × shot_value).
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for b, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        # Include the upper boundary in the last bin so that predictions
        # equal to exactly 1.0 are not silently dropped.
        if b < n_bins - 1:
            mask = (y_prob >= lo) & (y_prob < hi)
        else:
            mask = (y_prob >= lo) & (y_prob <= hi)
        if not mask.any():
            continue
        acc = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        ece += mask.sum() / n * abs(acc - conf)
    return float(ece)


# ---------------------------------------------------------------------------
# Core training
# ---------------------------------------------------------------------------

def train_model(
    dataframe: pd.DataFrame,
    target_column: str = "shot_made_flag",
    model_type: ModelType = "xgboost",
) -> TrainingArtifacts:
    """Train a shot-quality model and return evaluation artifacts.

    Improvements over a naïve pipeline:

    * **Leakage fix** – ``player_zone_fg_pct`` is refit on training data only
      inside :func:`_recompute_zone_fg_pct` using Bayesian shrinkage.
    * **PR-AUC** – Area under the Precision-Recall curve, which is more
      informative than ROC-AUC for imbalanced or skewed class distributions.
    * **Expected Calibration Error** – validates that predicted probabilities
      are reliable (essential for interpreting *p̂* as shot quality).
    """
    model_frame = dataframe.copy()
    available_features = [col for col in FEATURE_COLUMNS if col in model_frame.columns]
    if not available_features:
        raise ValueError("No model features are available. Run feature engineering before training.")
    if target_column not in model_frame.columns:
        raise ValueError(f"Missing target column: {target_column}")

    X_train, X_test, y_train, y_test = _split_and_fix_leakage(
        model_frame, available_features, target_column
    )

    numeric_columns = [c for c in available_features if model_frame[c].dtype != "object"]
    categorical_columns = [c for c in available_features if model_frame[c].dtype == "object"]
    preprocessor = _build_preprocessor(numeric_columns, categorical_columns)

    classifier = _build_classifier(model_type)
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])
    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "log_loss": float(log_loss(y_test, probabilities)),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "ece": expected_calibration_error(y_test.values, probabilities),
    }

    return TrainingArtifacts(
        pipeline=pipeline,
        metrics=metrics,
        feature_columns=available_features,
        model_name=model_type,
        X_test=X_test,
        y_test=y_test,
        probabilities=probabilities,
    )


# Keep backward-compatible alias
def train_baseline_model(dataframe: pd.DataFrame, target_column: str = "shot_made_flag") -> TrainingArtifacts:
    return train_model(dataframe, target_column, model_type="logistic")


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_model(
    dataframe: pd.DataFrame,
    target_column: str = "shot_made_flag",
    model_type: ModelType = "xgboost",
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Stratified k-fold cross-validation with leakage-free zone encoding.

    Because ``player_zone_fg_pct`` is a target-encoded feature, it must be
    recomputed from training-fold labels in each iteration to avoid leakage.
    A manual loop over :class:`~sklearn.model_selection.StratifiedKFold`
    splits gives full control over this re-encoding step.

    Returns a DataFrame with per-fold metrics **and** a summary row
    (``mean ± std``) so you can report calibrated uncertainty estimates
    alongside point estimates.
    """
    model_frame = dataframe.copy()
    available_features = [col for col in FEATURE_COLUMNS if col in model_frame.columns]
    aux_cols = [c for c in _ZONE_AUX_COLS if c in model_frame.columns]

    X_full = model_frame[available_features + aux_cols].copy()
    y = model_frame[target_column]

    numeric_columns = [c for c in available_features if model_frame[c].dtype != "object"]
    categorical_columns = [c for c in available_features if model_frame[c].dtype == "object"]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_metrics: list[dict] = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_full, y), 1):
        X_train_full = X_full.iloc[train_idx].copy()
        X_test_full = X_full.iloc[test_idx].copy()
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Leakage-free zone encoding per fold
        if "player_zone_fg_pct" in available_features and aux_cols:
            X_train_full, X_test_full = _recompute_zone_fg_pct(
                X_train_full, X_test_full, y_train
            )

        X_train = X_train_full[available_features]
        X_test = X_test_full[available_features]

        preprocessor = _build_preprocessor(numeric_columns, categorical_columns)
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", _build_classifier(model_type)),
        ])
        pipeline.fit(X_train, y_train)
        probs = pipeline.predict_proba(X_test)[:, 1]

        fold_metrics.append({
            "fold": fold,
            "roc_auc": float(roc_auc_score(y_test, probs)),
            "pr_auc": float(average_precision_score(y_test, probs)),
            "log_loss": float(log_loss(y_test, probs)),
            "brier_score": float(brier_score_loss(y_test, probs)),
            "ece": expected_calibration_error(y_test.values, probs),
        })

    results = pd.DataFrame(fold_metrics)
    metric_cols = [c for c in results.columns if c != "fold"]
    mean_row = {"fold": "mean", **results[metric_cols].mean().to_dict()}
    std_row = {"fold": "std", **results[metric_cols].std().to_dict()}
    return pd.concat([results, pd.DataFrame([mean_row, std_row])], ignore_index=True)


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_model(
    dataframe: pd.DataFrame,
    target_column: str = "shot_made_flag",
    model_type: ModelType = "xgboost",
    n_iter: int = 20,
    n_splits: int = 3,
    random_state: int = 42,
) -> dict:
    """Randomised hyperparameter search with stratified cross-validation.

    Uses :class:`~sklearn.model_selection.RandomizedSearchCV` over a broad
    parameter space, then evaluates the winning configuration on a held-out
    test set.  The leakage fix is applied **before** the inner CV loop so
    that the zone encoding seen during the search reflects the outer
    training partition only (a mild, accepted approximation analogous to
    warm-starting a feature encoder).

    Returns a dict containing ``best_params``, per-metric test scores, and
    the fitted best estimator.
    """
    model_frame = dataframe.copy()
    available_features = [col for col in FEATURE_COLUMNS if col in model_frame.columns]

    X_train, X_test, y_train, y_test = _split_and_fix_leakage(
        model_frame, available_features, target_column
    )

    numeric_columns = [c for c in available_features if model_frame[c].dtype != "object"]
    categorical_columns = [c for c in available_features if model_frame[c].dtype == "object"]
    preprocessor = _build_preprocessor(numeric_columns, categorical_columns)

    param_distributions: dict = {
        "xgboost": {
            "classifier__n_estimators": [100, 200, 300, 400, 500],
            "classifier__max_depth": [3, 4, 5, 6, 7],
            "classifier__learning_rate": [0.01, 0.03, 0.05, 0.10, 0.15, 0.20],
            "classifier__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "classifier__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "classifier__min_child_weight": [1, 3, 5, 7],
            "classifier__gamma": [0.0, 0.05, 0.10, 0.20, 0.50],
            "classifier__reg_alpha": [0.0, 0.01, 0.1, 1.0],
            "classifier__reg_lambda": [0.5, 1.0, 2.0, 5.0],
        },
        "logistic": {
            "classifier__C": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
            "classifier__penalty": ["l1", "l2"],
            "classifier__solver": ["liblinear"],
        },
    }

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", _build_classifier(model_type)),
    ])

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions[model_type],
        n_iter=n_iter,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state),
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)

    probs = search.best_estimator_.predict_proba(X_test)[:, 1]
    return {
        "best_params": search.best_params_,
        "best_cv_roc_auc": float(search.best_score_),
        "test_roc_auc": float(roc_auc_score(y_test, probs)),
        "test_pr_auc": float(average_precision_score(y_test, probs)),
        "test_log_loss": float(log_loss(y_test, probs)),
        "test_brier_score": float(brier_score_loss(y_test, probs)),
        "test_ece": expected_calibration_error(y_test.values, probs),
        "best_estimator": search.best_estimator_,
    }


# ---------------------------------------------------------------------------
# Post-hoc analysis utilities
# ---------------------------------------------------------------------------

def get_feature_importance(artifacts: TrainingArtifacts) -> pd.DataFrame | None:
    """Return a sorted feature-importance DataFrame for tree-based models."""
    clf = artifacts.pipeline.named_steps.get("classifier")
    if clf is None or not hasattr(clf, "feature_importances_"):
        return None
    importances = clf.feature_importances_
    feature_names = artifacts.feature_columns
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def get_permutation_importance(
    artifacts: TrainingArtifacts,
    n_repeats: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """Permutation feature importance evaluated on the held-out test set.

    Unlike impurity-based importance (which is computed on training data and
    biased toward high-cardinality features), permutation importance directly
    measures the drop in *test-set* ROC-AUC when each feature's values are
    randomly shuffled – giving a more reliable picture of out-of-sample
    relevance, especially for correlated predictors.
    """
    if artifacts.X_test is None or artifacts.y_test is None:
        return pd.DataFrame()

    result = sklearn_permutation_importance(
        artifacts.pipeline,
        artifacts.X_test,
        artifacts.y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="roc_auc",
        n_jobs=-1,
    )

    return (
        pd.DataFrame({
            "feature": artifacts.feature_columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        })
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def get_calibration_data(artifacts: TrainingArtifacts, n_bins: int = 10) -> pd.DataFrame:
    """Return fraction_of_positives and mean_predicted_value arrays for calibration plot."""
    frac_pos, mean_pred = calibration_curve(
        artifacts.y_test, artifacts.probabilities, n_bins=n_bins, strategy="uniform"
    )
    return pd.DataFrame({"mean_predicted": mean_pred, "fraction_positive": frac_pos})


def get_roc_data(artifacts: TrainingArtifacts) -> pd.DataFrame:
    """Return FPR/TPR arrays for an ROC curve plot."""
    fpr, tpr, thresholds = roc_curve(artifacts.y_test, artifacts.probabilities)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})


def get_pr_curve_data(artifacts: TrainingArtifacts) -> pd.DataFrame:
    """Return precision/recall arrays for a Precision-Recall curve plot.

    The PR curve is particularly informative when the positive class (made
    shots) is the minority or when the cost of false positives differs from
    false negatives, situations where ROC-AUC can be overly optimistic.
    """
    precision, recall, thresholds = precision_recall_curve(
        artifacts.y_test, artifacts.probabilities
    )
    # precision_recall_curve returns len(thresholds) = len(precision) - 1
    thresholds = np.append(thresholds, np.nan)
    return pd.DataFrame({"precision": precision, "recall": recall, "threshold": thresholds})


def get_learning_curve_data(
    dataframe: pd.DataFrame,
    target_column: str = "shot_made_flag",
    model_type: ModelType = "xgboost",
    n_splits: int = 5,
    train_sizes: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute train and CV test ROC-AUC for increasing training set sizes.

    The learning curve reveals the bias-variance trade-off:

    * A large gap between training and validation score → *high variance*
      (over-fitting) → more data or stronger regularisation will help.
    * Both scores low and close together → *high bias* (under-fitting) →
      a more expressive model or richer features are needed.
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 8)

    model_frame = dataframe.copy()
    available_features = [col for col in FEATURE_COLUMNS if col in model_frame.columns]
    X = model_frame[available_features]
    y = model_frame[target_column]

    numeric_columns = [c for c in available_features if model_frame[c].dtype != "object"]
    categorical_columns = [c for c in available_features if model_frame[c].dtype == "object"]
    preprocessor = _build_preprocessor(numeric_columns, categorical_columns)
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", _build_classifier(model_type)),
    ])

    train_sizes_abs, train_scores, test_scores = learning_curve(
        pipeline,
        X,
        y,
        train_sizes=train_sizes,
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42),
        scoring="roc_auc",
        n_jobs=-1,
    )

    return pd.DataFrame({
        "train_size": train_sizes_abs,
        "train_mean": train_scores.mean(axis=1),
        "train_std": train_scores.std(axis=1),
        "test_mean": test_scores.mean(axis=1),
        "test_std": test_scores.std(axis=1),
    })


# ---------------------------------------------------------------------------
# Expected-points annotation
# ---------------------------------------------------------------------------

def add_expected_points(
    dataframe: pd.DataFrame,
    pipeline: Pipeline,
    feature_columns: list[str],
    shot_value_column: str = "shot_value",
) -> pd.DataFrame:
    frame = dataframe.copy()
    # Use only the features the pipeline was actually trained on (in training order)
    frame["make_probability"] = pipeline.predict_proba(frame[feature_columns])[:, 1]
    frame["xpts"] = frame["make_probability"] * frame[shot_value_column].fillna(2)
    return frame


def save_artifacts(artifacts: TrainingArtifacts, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": artifacts.pipeline,
            "metrics": artifacts.metrics,
            "feature_columns": artifacts.feature_columns,
            "model_name": artifacts.model_name,
        },
        path,
    )
    return path


# ---------------------------------------------------------------------------
# Ablation study
# ---------------------------------------------------------------------------

def run_ablation_study(
    dataframe: pd.DataFrame,
    target_column: str = "shot_made_flag",
    model_type: ModelType = "xgboost",
    test_size: float = 0.2,
    random_state: int = 7,
) -> pd.DataFrame:
    """Quantify the incremental predictive value of each feature tier.

    Trains the same model type (XGBoost by default) three times, each with
    a progressively richer feature set, using the **same** train/test split:

    1. **Location Only** – pure geometry (distance, angle, polynomial terms).
    2. **Location + Context** – adds game-state variables (period, clock, score).
    3. **Full (+ Player Skill)** – adds Bayesian-smoothed player-zone efficiency.

    Comparing ROC-AUC and Log-Loss across tiers answers the research question:
    *How much of shot outcome variance is explained by geometry vs context vs
    player skill?*

    Returns a DataFrame with one row per tier and columns for each metric
    plus ``tier`` (name), ``n_features`` (feature count), and ``delta_roc_auc``
    (incremental AUC gain over the previous tier).
    """
    model_frame = dataframe.copy()
    rows: list[dict] = []

    # Use a fixed split across all tiers so comparisons are valid
    # The full feature set is used for the initial split; subsets are
    # taken from the same indices afterward.
    full_features = FEATURE_TIERS["Full (+ Player Skill)"]
    available_full = [f for f in full_features if f in model_frame.columns]
    aux_cols = [c for c in _ZONE_AUX_COLS if c in model_frame.columns]

    X_full_all = model_frame[available_full + aux_cols].copy()
    y = model_frame[target_column]

    X_all_train, X_all_test, y_train, y_test = train_test_split(
        X_full_all, y, test_size=test_size, random_state=random_state, stratify=y
    )

    prev_auc: float | None = None
    for tier_name, tier_features in FEATURE_TIERS.items():
        available = [f for f in tier_features if f in model_frame.columns]
        if not available:
            continue

        try:
            # Apply leakage fix only for the tier that includes player_zone_fg_pct
            X_train_tier = X_all_train[available + aux_cols].copy()
            X_test_tier = X_all_test[available + aux_cols].copy()

            if "player_zone_fg_pct" in available and aux_cols:
                X_train_tier, X_test_tier = _recompute_zone_fg_pct(
                    X_train_tier, X_test_tier, y_train
                )

            X_train = X_train_tier[available]
            X_test = X_test_tier[available]

            numeric_cols = [c for c in available if model_frame[c].dtype != "object"]
            categorical_cols = [c for c in available if model_frame[c].dtype == "object"]
            preprocessor = _build_preprocessor(numeric_cols, categorical_cols)

            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("classifier", _build_classifier(model_type)),
            ])
            pipeline.fit(X_train, y_train)
            probs = pipeline.predict_proba(X_test)[:, 1]

            auc = float(roc_auc_score(y_test, probs))
            row: dict = {
                "tier": tier_name,
                "n_features": len(available),
                "roc_auc": auc,
                "pr_auc": float(average_precision_score(y_test, probs)),
                "log_loss": float(log_loss(y_test, probs)),
                "brier_score": float(brier_score_loss(y_test, probs)),
                "ece": expected_calibration_error(y_test.values, probs),
                "delta_roc_auc": (auc - prev_auc) if prev_auc is not None else float("nan"),
            }
            rows.append(row)
            prev_auc = auc

        except Exception as exc:
            print(
                f"  Ablation: tier '{tier_name}' skipped — {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 500,
    ci: float = 0.95,
    random_state: int = 42,
) -> dict[str, tuple[float, float, float]]:
    """Bootstrap confidence intervals for held-out test-set metrics.

    Resamples ``(y_true, y_prob)`` with replacement *n_bootstrap* times and
    computes the empirical *ci*% confidence interval for each metric.  This
    quantifies the uncertainty in point-estimate metrics (e.g. ROC-AUC)
    arising from the finite test-set size.

    Parameters
    ----------
    y_true, y_prob:
        Ground-truth labels and predicted probabilities from the test set.
    n_bootstrap:
        Number of bootstrap resamples.  500 is sufficient for 95% CIs.
    ci:
        Desired confidence level (default: 0.95 → 95% CI).

    Returns
    -------
    dict mapping metric name → (mean, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    alpha = (1.0 - ci) / 2.0

    boot: dict[str, list[float]] = {
        "roc_auc": [],
        "pr_auc": [],
        "log_loss": [],
        "brier_score": [],
    }

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_prob[idx]
        # Skip degenerate bootstrap samples (only one class present)
        if yt.sum() == 0 or yt.sum() == n:
            continue
        boot["roc_auc"].append(float(roc_auc_score(yt, yp)))
        boot["pr_auc"].append(float(average_precision_score(yt, yp)))
        boot["log_loss"].append(float(log_loss(yt, yp)))
        boot["brier_score"].append(float(brier_score_loss(yt, yp)))

    output: dict[str, tuple[float, float, float]] = {}
    for metric, values in boot.items():
        arr = np.array(values)
        output[metric] = (
            float(arr.mean()),
            float(np.percentile(arr, 100.0 * alpha)),
            float(np.percentile(arr, 100.0 * (1.0 - alpha))),
        )
    return output


# ---------------------------------------------------------------------------
# Post-hoc calibrated model
# ---------------------------------------------------------------------------

def train_calibrated_model(
    dataframe: pd.DataFrame,
    target_column: str = "shot_made_flag",
    model_type: ModelType = "xgboost",
    method: str = "isotonic",
    cv: int = 5,
) -> TrainingArtifacts:
    """Train with post-hoc probability calibration (CalibratedClassifierCV).

    Wraps the base classifier in
    :class:`~sklearn.calibration.CalibratedClassifierCV` using *cv*-fold
    cross-validation to fit an isotonic regression (or Platt sigmoid) on top
    of the classifier's raw probability estimates.

    **Why this matters for xPTS:** the expected-points formula
    ``xPTS = P̂(make) × shot_value`` is sensitive to calibration quality.
    A model that is systematically over-confident (P̂ > true rate) inflates
    xPTS for high-probability shots; one that is under-confident deflates it.
    Isotonic calibration corrects this systematic bias monotonically.

    The calibrated model typically has lower ECE than the uncalibrated version
    while sacrificing minimal ROC-AUC, making it a strong candidate for
    deployment when probability magnitudes matter.
    """
    model_frame = dataframe.copy()
    available_features = [col for col in FEATURE_COLUMNS if col in model_frame.columns]
    if not available_features:
        raise ValueError("No model features available. Run feature engineering first.")
    if target_column not in model_frame.columns:
        raise ValueError(f"Missing target column: {target_column}")

    X_train, X_test, y_train, y_test = _split_and_fix_leakage(
        model_frame, available_features, target_column
    )

    numeric_columns = [c for c in available_features if model_frame[c].dtype != "object"]
    categorical_columns = [c for c in available_features if model_frame[c].dtype == "object"]
    preprocessor = _build_preprocessor(numeric_columns, categorical_columns)

    base_classifier = _build_classifier(model_type)
    calibrated_classifier = CalibratedClassifierCV(
        base_classifier, cv=cv, method=method
    )
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", calibrated_classifier),
    ])
    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "log_loss": float(log_loss(y_test, probabilities)),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "ece": expected_calibration_error(y_test.values, probabilities),
    }

    return TrainingArtifacts(
        pipeline=pipeline,
        metrics=metrics,
        feature_columns=available_features,
        model_name=f"{model_type}_calibrated_{method}",
        X_test=X_test,
        y_test=y_test,
        probabilities=probabilities,
    )
