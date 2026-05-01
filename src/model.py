from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


FEATURE_COLUMNS = [
    "shot_distance",
    "shot_angle",
    "period",
    "game_seconds_remaining",
    "score_diff_abs",
    "player_zone_fg_pct",
    "late_clock",
    "shot_clock",
]

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


def train_model(
    dataframe: pd.DataFrame,
    target_column: str = "shot_made_flag",
    model_type: ModelType = "xgboost",
) -> TrainingArtifacts:
    model_frame = dataframe.copy()
    available_features = [col for col in FEATURE_COLUMNS if col in model_frame.columns]
    if not available_features:
        raise ValueError("No model features are available. Run feature engineering before training.")
    if target_column not in model_frame.columns:
        raise ValueError(f"Missing target column: {target_column}")

    X_train, X_test, y_train, y_test = train_test_split(
        model_frame[available_features],
        model_frame[target_column],
        test_size=0.2,
        random_state=7,
        stratify=model_frame[target_column],
    )

    numeric_columns = [c for c in available_features if model_frame[c].dtype != "object"]
    categorical_columns = [c for c in available_features if model_frame[c].dtype == "object"]
    preprocessor = _build_preprocessor(numeric_columns, categorical_columns)

    if model_type == "xgboost":
        classifier = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        )
    else:
        classifier = LogisticRegression(max_iter=1000, C=1.0)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])
    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "log_loss": float(log_loss(y_test, probabilities)),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
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


def get_feature_importance(artifacts: TrainingArtifacts) -> pd.DataFrame | None:
    """Return a sorted feature-importance DataFrame for tree-based models."""
    clf = artifacts.pipeline.named_steps.get("classifier")
    if clf is None or not hasattr(clf, "feature_importances_"):
        return None
    importances = clf.feature_importances_
    feature_names = artifacts.feature_columns
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def get_calibration_data(artifacts: TrainingArtifacts, n_bins: int = 10) -> pd.DataFrame:
    """Return fraction_of_positives and mean_predicted_value arrays for calibration plot."""
    frac_pos, mean_pred = calibration_curve(
        artifacts.y_test, artifacts.probabilities, n_bins=n_bins, strategy="uniform"
    )
    return pd.DataFrame({"mean_predicted": mean_pred, "fraction_positive": frac_pos})


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
