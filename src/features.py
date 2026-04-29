from __future__ import annotations

import numpy as np
import pandas as pd


def add_geometry_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    frame = dataframe.copy()
    frame["shot_distance"] = np.sqrt(frame["loc_x"] ** 2 + frame["loc_y"] ** 2)
    frame["shot_angle"] = np.degrees(np.arctan2(frame["loc_y"], frame["loc_x"].replace(0, np.nan)))
    frame["shot_angle"] = frame["shot_angle"].fillna(90.0)
    return frame


def add_game_context_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    frame = dataframe.copy()
    if {"home_score", "away_score"}.issubset(frame.columns):
        frame["score_diff"] = frame["home_score"] - frame["away_score"]
        frame["score_diff_abs"] = frame["score_diff"].abs()

    frame["game_seconds_remaining"] = (
        frame["minutes_remaining"].fillna(0) * 60 + frame["seconds_remaining"].fillna(0)
    )

    if "shot_clock" in frame.columns:
        frame["late_clock"] = (frame["shot_clock"].fillna(24) <= 4).astype(int)

    return frame


def add_zone_history_feature(
    dataframe: pd.DataFrame,
    player_column: str = "player_name",
    zone_column: str = "shot_zone_basic",
    target_column: str = "shot_made_flag",
) -> pd.DataFrame:
    frame = dataframe.copy()
    if not {player_column, zone_column, target_column}.issubset(frame.columns):
        return frame

    history = (
        frame.groupby([player_column, zone_column], dropna=False)[target_column]
        .mean()
        .rename("player_zone_fg_pct")
        .reset_index()
    )
    return frame.merge(history, on=[player_column, zone_column], how="left")


def build_model_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    frame = add_geometry_features(dataframe)
    frame = add_game_context_features(frame)
    frame = add_zone_history_feature(frame)

    if "shot_made_flag" not in frame.columns and "shot_result" in frame.columns:
        frame["shot_made_flag"] = frame["shot_result"].astype(str).str.lower().eq("made shot").astype(int)

    if "shot_type" in frame.columns:
        frame["shot_value"] = frame["shot_type"].astype(str).str.extract(r"(\d)").fillna(2).astype(int)
    else:
        frame["shot_value"] = 2

    return frame
