from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FEATURE_COLUMNS = [
    "shot_distance",
    "shot_angle",
    "period",
    "game_seconds_remaining",
    "score_diff_abs",
    "player_zone_fg_pct",
    "late_clock",
]


@dataclass(slots=True)
class TrainingArtifacts:
    pipeline: Pipeline
    metrics: dict[str, float]
    feature_columns: list[str]


def train_baseline_model(dataframe: pd.DataFrame, target_column: str = "shot_made_flag") -> TrainingArtifacts:
    model_frame = dataframe.copy()
    available_features = [column for column in FEATURE_COLUMNS if column in model_frame.columns]
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

    numeric_columns = [column for column in available_features if model_frame[column].dtype != "object"]
    categorical_columns = [column for column in available_features if model_frame[column].dtype == "object"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": roc_auc_score(y_test, probabilities),
        "log_loss": log_loss(y_test, probabilities),
        "brier_score": brier_score_loss(y_test, probabilities),
    }

    return TrainingArtifacts(pipeline=pipeline, metrics=metrics, feature_columns=available_features)


def add_expected_points(
    dataframe: pd.DataFrame,
    pipeline: Pipeline,
    feature_columns: list[str],
    shot_value_column: str = "shot_value",
) -> pd.DataFrame:
    frame = dataframe.copy()
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
        },
        path,
    )
    return path
