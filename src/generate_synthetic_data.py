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


if __name__ == "__main__":
    out = Path("data/raw/shots.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate_shots(12_000)
    df.to_csv(out, index=False)
    print(f"Saved {len(df):,} shots to {out}")
    print(df.head())
    print(df["shot_made_flag"].value_counts())
