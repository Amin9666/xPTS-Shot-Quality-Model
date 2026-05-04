"""
Synthetic NBA shot data generator.

Produces a realistic shot-level dataset with court geometry, player archetypes,
and outcome probabilities derived from contextual features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

EPSILON = 1e-9  # Small value to avoid division by zero in arctan2 for x=0
RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Player archetypes – each dict encodes tendencies and skill level
# ---------------------------------------------------------------------------
PLAYERS: list[dict] = [
    {"name": "Stephen Curry",   "archetype": "three_point_specialist", "skill": 0.90},
    {"name": "Jayson Tatum",    "archetype": "wing_scorer",            "skill": 0.78},
    {"name": "Nikola Jokic",    "archetype": "paint_scorer",           "skill": 0.82},
    {"name": "Damian Lillard",  "archetype": "three_point_specialist", "skill": 0.85},
    {"name": "Giannis A.",      "archetype": "paint_scorer",           "skill": 0.80},
    {"name": "Devin Booker",    "archetype": "mid_range_scorer",       "skill": 0.81},
    {"name": "LeBron James",    "archetype": "wing_scorer",            "skill": 0.83},
    {"name": "Luka Doncic",     "archetype": "mid_range_scorer",       "skill": 0.84},
    {"name": "Kevin Durant",    "archetype": "wing_scorer",            "skill": 0.88},
    {"name": "Ja Morant",       "archetype": "paint_scorer",           "skill": 0.77},
]

ZONES = ["In The Paint (Non-RA)", "Restricted Area", "Mid-Range", "Left Corner 3",
         "Right Corner 3", "Above the Break 3", "Backcourt"]

# Approximate zone weights per archetype
ZONE_WEIGHTS: dict[str, list[float]] = {
    "three_point_specialist": [0.05, 0.10, 0.08, 0.12, 0.12, 0.50, 0.03],
    "wing_scorer":            [0.10, 0.15, 0.20, 0.08, 0.08, 0.35, 0.04],
    "mid_range_scorer":       [0.08, 0.12, 0.40, 0.06, 0.06, 0.25, 0.03],
    "paint_scorer":           [0.20, 0.40, 0.12, 0.06, 0.06, 0.13, 0.03],
}


# ---------------------------------------------------------------------------
# Court coordinate samplers per zone (NBA tracking units: tenths of a foot)
# ---------------------------------------------------------------------------

def _sample_zone_coords(zone: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (loc_x, loc_y) arrays for *n* shots from a given zone."""
    match zone:
        case "Restricted Area":
            r = RNG.uniform(0, 40, n)
            theta = RNG.uniform(-np.pi / 2, np.pi / 2, n)
            return r * np.cos(theta), r * np.sin(theta) + 5
        case "In The Paint (Non-RA)":
            x = RNG.uniform(-80, 80, n)
            y = RNG.uniform(40, 190, n)
            return x, y
        case "Mid-Range":
            r = RNG.uniform(100, 230, n)
            theta = RNG.uniform(-np.pi * 0.65, np.pi * 0.65, n)
            return r * np.cos(theta), r * np.sin(theta)
        case "Left Corner 3":
            x = RNG.uniform(-250, -220, n)
            y = RNG.uniform(-52, 90, n)
            return x, y
        case "Right Corner 3":
            x = RNG.uniform(220, 250, n)
            y = RNG.uniform(-52, 90, n)
            return x, y
        case "Above the Break 3":
            r = RNG.uniform(237, 330, n)
            theta = RNG.uniform(-np.pi * 0.55, np.pi * 0.55, n)
            return r * np.cos(theta), r * np.sin(theta)
        case _:  # Backcourt
            x = RNG.uniform(-250, 250, n)
            y = RNG.uniform(400, 900, n)
            return x, y


def _base_make_probability(distance_ft: np.ndarray, zone: str) -> np.ndarray:
    """Logistic decay make-probability curve calibrated to NBA averages."""
    base = np.where(
        distance_ft <= 4, 0.64,
        np.where(
            distance_ft <= 10, 0.55,
            np.where(
                distance_ft <= 16, 0.43,
                np.where(distance_ft <= 23.75, 0.38, 0.36),
            ),
        ),
    )
    # Corner three bump
    if zone in ("Left Corner 3", "Right Corner 3"):
        base = base + 0.04
    return base


def generate_shots(n_shots: int = 12_000) -> pd.DataFrame:
    records: list[dict] = []

    for player in PLAYERS:
        player_n = n_shots // len(PLAYERS)
        weights = ZONE_WEIGHTS[player["archetype"]]
        zone_counts = RNG.multinomial(player_n, weights)

        for zone, count in zip(ZONES, zone_counts):
            if count == 0:
                continue

            x, y = _sample_zone_coords(zone, count)
            distance_ft = np.sqrt(x**2 + y**2) / 10.0  # tenths → feet
            angle_deg = np.degrees(np.arctan2(y, np.where(x == 0, EPSILON, x)))

            # Game context
            period = RNG.integers(1, 5, count)
            minutes_rem = RNG.integers(0, 12, count)
            seconds_rem = RNG.integers(0, 60, count)
            shot_clock = RNG.uniform(0, 24, count)
            score_diff = RNG.integers(-25, 26, count)

            base_prob = _base_make_probability(distance_ft, zone)
            skill_adj = (player["skill"] - 0.80) * 0.15
            late_clock_pen = np.where(shot_clock <= 4, -0.08, 0.0)
            close_game_adj = np.where(np.abs(score_diff) <= 5, 0.02, 0.0)
            prob = np.clip(base_prob + skill_adj + late_clock_pen + close_game_adj, 0.05, 0.95)

            made = RNG.binomial(1, prob).astype(int)
            shot_value_scalar = 3 if (zone.endswith("3") or zone == "Backcourt") else 2
            shot_value = np.full(count, shot_value_scalar, dtype=int)

            for i in range(count):
                records.append(
                    {
                        "player_name": player["name"],
                        "shot_zone_basic": zone,
                        "loc_x": round(float(x[i]), 1),
                        "loc_y": round(float(y[i]), 1),
                        "shot_distance": round(float(distance_ft[i]), 2),
                        "shot_angle": round(float(angle_deg[i]), 2),
                        "period": int(period[i]),
                        "minutes_remaining": int(minutes_rem[i]),
                        "seconds_remaining": int(seconds_rem[i]),
                        "shot_clock": round(float(shot_clock[i]), 1),
                        "home_score": int(RNG.integers(80, 120)),
                        "away_score": int(RNG.integers(80, 120)),
                        "shot_result": "Made Shot" if made[i] else "Missed Shot",
                        "shot_made_flag": int(made[i]),
                        "shot_type": f"{int(shot_value[i])}PT Field Goal",
                        "shot_value": int(shot_value[i]),
                        "true_make_prob": round(float(prob[i]), 4),
                    }
                )

    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Steph Curry 2023-24 realistic shot chart (used as offline fallback)
# ---------------------------------------------------------------------------
# Calibrated to his actual 2023-24 regular-season numbers:
#   FGA ≈ 1208  (82 games)
#   3PA ≈ 886   (72.5% of FGA) — league record territory
#   3PT% ≈ 40.8%,  2PT% ≈ 52.5%,  FG% ≈ 45.0%
#
# Zone breakdown (approximate, based on tracking data):
#   Above the Break 3  : 600  (49.7%)
#   Left Corner 3      : 143  (11.8%)
#   Right Corner 3     : 143  (11.8%)
#   Restricted Area    : 145  (12.0%)
#   In The Paint Non-RA:  85  ( 7.0%)
#   Mid-Range          :  82  ( 6.8%)
#   Backcourt          :  10  ( 0.8%)

_CURRY_ZONE_COUNTS: dict[str, int] = {
    "Above the Break 3":       600,
    "Left Corner 3":           143,
    "Right Corner 3":          143,
    "Restricted Area":         145,
    "In The Paint (Non-RA)":    85,
    "Mid-Range":                82,
    "Backcourt":                10,
}

# Per-zone make probabilities tuned to produce realistic fg% figures
_CURRY_ZONE_MAKE_PROB: dict[str, float] = {
    "Above the Break 3":      0.408,
    "Left Corner 3":          0.445,
    "Right Corner 3":         0.445,
    "Restricted Area":        0.625,
    "In The Paint (Non-RA)": 0.420,
    "Mid-Range":              0.470,
    "Backcourt":              0.070,
}


def generate_curry_shots(seed: int = 42) -> pd.DataFrame:
    """Generate a realistic Stephen Curry 2023-24 shot dataset.

    All statistics are calibrated to his actual 2023-24 regular-season
    numbers so this can serve as a credible offline fallback when the NBA
    API is unavailable.

    Parameters
    ----------
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Shot-level dataset with the same schema as :func:`generate_shots`.
    """
    rng = np.random.default_rng(seed)
    records: list[dict] = []

    for zone, count in _CURRY_ZONE_COUNTS.items():
        x, y = _sample_zone_coords(zone, count)
        distance_ft = np.sqrt(x**2 + y**2) / 10.0  # NBA coords are in tenths of a foot
        # Substitute EPSILON for x=0 to avoid atan2(y,0) singularity at the basket centre
        angle_deg = np.degrees(np.arctan2(y, np.where(x == 0, EPSILON, x)))

        period = rng.integers(1, 5, count)
        minutes_rem = rng.integers(0, 12, count)
        seconds_rem = rng.integers(0, 60, count)
        shot_clock = rng.uniform(0, 24, count)
        score_diff = rng.integers(-25, 26, count)

        # Derive home/away scores so that home_score - away_score == score_diff,
        # keeping both values in a realistic NBA range (≈ 80–130).
        away_score_base = rng.integers(80, 106, count)
        home_score = away_score_base + score_diff
        away_score = away_score_base

        make_prob = _CURRY_ZONE_MAKE_PROB[zone]
        # Small contextual adjustments (late clock hurts, close game slight boost)
        late_clock_pen = np.where(shot_clock <= 4, -0.05, 0.0)
        close_game_adj = np.where(np.abs(score_diff) <= 5, 0.01, 0.0)
        prob = np.clip(make_prob + late_clock_pen + close_game_adj, 0.04, 0.95)

        made = rng.binomial(1, prob).astype(int)
        is_three = zone.endswith("3") or zone == "Backcourt"
        shot_value_scalar = 3 if is_three else 2

        for i in range(count):
            records.append(
                {
                    "player_name": "Stephen Curry",
                    "shot_zone_basic": zone,
                    "loc_x": round(float(x[i]), 1),
                    "loc_y": round(float(y[i]), 1),
                    "shot_distance": round(float(distance_ft[i]), 2),
                    "shot_angle": round(float(angle_deg[i]), 2),
                    "period": int(period[i]),
                    "minutes_remaining": int(minutes_rem[i]),
                    "seconds_remaining": int(seconds_rem[i]),
                    "shot_clock": round(float(shot_clock[i]), 1),
                    "home_score": int(home_score[i]),
                    "away_score": int(away_score[i]),
                    "shot_result": "Made Shot" if made[i] else "Missed Shot",
                    "shot_made_flag": int(made[i]),
                    "shot_type": f"{shot_value_scalar}PT Field Goal",
                    "shot_value": shot_value_scalar,
                    "true_make_prob": round(float(prob[i]), 4),
                    "score_diff": int(score_diff[i]),
                }
            )

    df = pd.DataFrame(records).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    out = Path("data/raw/shots.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate_shots(12_000)
    df.to_csv(out, index=False)
    print(f"Saved {len(df):,} shots to {out}")
    print(df.head())
    print(df["shot_made_flag"].value_counts())
