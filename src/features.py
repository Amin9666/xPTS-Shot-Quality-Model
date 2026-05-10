from __future__ import annotations

import numpy as np
import pandas as pd


def add_geometry_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    frame = dataframe.copy()
    frame["shot_distance"] = np.sqrt(frame["loc_x"] ** 2 + frame["loc_y"] ** 2)
    frame["shot_angle"] = np.degrees(np.arctan2(frame["loc_y"], frame["loc_x"].replace(0, np.nan)))
    frame["shot_angle"] = frame["shot_angle"].fillna(90.0)
    return frame


def add_polynomial_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add non-linear and interaction terms motivated by shot geometry.

    ``distance_sq``
        Squared distance captures the *accelerating* difficulty of longer
        shots: empirically the log-odds of making a shot is approximately
        linear in distance, so make-probability is convex in distance and a
        squared term helps linear learners (logistic regression) model this
        without feature crosses.

    ``log1p_distance``
        log(1 + distance) emphasises the near-basket regime where small
        changes in range have a large effect on make-probability (the curve
        is steep inside ~10 ft and flattens beyond the arc).  The +1 offset
        ensures continuity at zero distance.

    ``dist_angle_ix``
        Distance × |angle| interaction. Shots from the same distance are
        harder when taken from a severe angle (e.g. extreme left/right),
        so the product encodes the joint penalty of range and lateral
        difficulty that neither feature captures alone.
    """
    frame = dataframe.copy()
    if "shot_distance" in frame.columns:
        frame["distance_sq"] = frame["shot_distance"] ** 2
        frame["log1p_distance"] = np.log1p(frame["shot_distance"])
    if {"shot_distance", "shot_angle"}.issubset(frame.columns):
        frame["dist_angle_ix"] = frame["shot_distance"] * frame["shot_angle"].abs()
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


def _name_court_region(cx: float, cy: float) -> str:
    """Return a basketball zone label for a court position (tenths-of-a-foot coords).

    The NBA shot-chart coordinate system places the basket at (0, 0), with
    ``loc_x`` spanning −250 to +250 (left–right) and ``loc_y`` running from
    −52 (behind backboard) upward toward half-court (~470).  All thresholds
    below are expressed in the same tenths-of-a-foot units.
    """
    dist = np.sqrt(cx ** 2 + cy ** 2)
    abs_cx = abs(cx)
    if dist < 60:
        return "Rim"
    if dist < 130:
        return "Paint"
    if dist < 220:
        return "Mid-Range"
    # 3-point territory
    if cy < 90 and abs_cx > 200:
        return "Corner 3"
    if abs_cx > 165:
        return "Wing 3"
    return "Above-Break 3"


def add_shot_archetype_clusters(
    dataframe: pd.DataFrame,
    n_clusters: int = 6,
    random_state: int = 42,
) -> pd.DataFrame:
    """Cluster shots into spatial archetypes using K-Means on court coordinates.

    Each cluster is assigned a human-readable basketball label derived from
    the cluster centroid's court region (e.g. "Rim", "Wing 3", "Corner 3").
    The resulting ``shot_archetype`` column is intended for post-hoc analysis
    and visualisation rather than as a direct model input.

    Parameters
    ----------
    n_clusters:
        Number of K-Means clusters.  Six is a natural choice that roughly
        corresponds to the main strategic shot locations in the NBA.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler as _SS

    frame = dataframe.copy()
    coord_cols = [c for c in ("loc_x", "loc_y") if c in frame.columns]
    if len(coord_cols) < 2:
        frame["shot_archetype"] = "Unknown"
        return frame

    coords = frame[["loc_x", "loc_y"]].fillna(0).values.astype(float)
    scaled = _SS().fit_transform(coords)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(scaled)

    # Name each cluster by its average court position
    cluster_names: dict[int, str] = {}
    for cid in range(n_clusters):
        mask = labels == cid
        cx = float(frame["loc_x"].values[mask].mean())
        cy = float(frame["loc_y"].values[mask].mean())
        cluster_names[cid] = _name_court_region(cx, cy)

    frame["shot_archetype"] = [cluster_names[lbl] for lbl in labels]
    return frame


def add_shot_decision_quality(
    dataframe: pd.DataFrame,
    xpts_column: str = "xpts",
    zone_column: str = "shot_zone_basic",
) -> pd.DataFrame:
    """Add a ``decision_quality`` column measuring each shot's xPTS above/below zone average.

    Shot decision quality = xPTS(shot) − mean(xPTS | shot_zone)

    A positive value means the shooter generated a better look than the
    league average for that zone; negative means it was below average.
    This separates *shot selection* (did you take the right shot?) from
    *execution* (did you make it?), a distinction central to modern
    offensive evaluation.

    Requires ``xpts_column`` to already be present (i.e. call after
    :func:`~src.model.add_expected_points`).
    """
    frame = dataframe.copy()
    if xpts_column not in frame.columns or zone_column not in frame.columns:
        return frame
    zone_avg = frame.groupby(zone_column)[xpts_column].transform("mean")
    frame["decision_quality"] = frame[xpts_column] - zone_avg
    return frame


def build_model_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    frame = add_geometry_features(dataframe)
    frame = add_polynomial_features(frame)
    frame = add_game_context_features(frame)
    frame = add_zone_history_feature(frame)
    frame = add_shot_archetype_clusters(frame)

    if "shot_made_flag" not in frame.columns and "shot_result" in frame.columns:
        frame["shot_made_flag"] = frame["shot_result"].astype(str).str.lower().eq("made shot").astype(int)

    if "shot_type" in frame.columns:
        frame["shot_value"] = frame["shot_type"].astype(str).str.extract(r"(\d)").fillna(2).astype(int)
    else:
        frame["shot_value"] = 2

    return frame
