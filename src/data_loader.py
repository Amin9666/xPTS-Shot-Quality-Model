from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "shot_result",
    "shot_type",
    "loc_x",
    "loc_y",
    "period",
    "minutes_remaining",
    "seconds_remaining",
}


@dataclass(slots=True)
class ShotDatasetConfig:
    raw_data_path: Path = Path("data/raw/shots.csv")
    processed_data_path: Path = Path("data/processed/shots_model_input.csv")


def load_local_shots(csv_path: str | Path) -> pd.DataFrame:
    dataframe = pd.read_csv(csv_path)
    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Shot dataset is missing required columns: {missing_list}")
    return dataframe


def get_player_id(player_name: str) -> int:
    """Return the NBA player ID for *player_name* using nba_api's static lookup.

    Parameters
    ----------
    player_name:
        Full player name, e.g. ``"Stephen Curry"``.

    Returns
    -------
    int
        The NBA player ID.

    Raises
    ------
    ValueError
        If no player with that name is found.
    """
    from nba_api.stats.static.players import find_players_by_full_name  # type: ignore[import]

    results = find_players_by_full_name(player_name)
    if not results:
        raise ValueError(f"No player found for name: {player_name!r}")
    return int(results[0]["id"])


def fetch_player_shot_chart(
    player_id: int = 201939,
    season: str = "2023-24",
) -> pd.DataFrame:
    """Fetch a real NBA player shot chart from the nba_api.

    Pulls shot-level data for *player_id* in *season* and renames / derives
    columns so the result is compatible with the xPTS pipeline schema.

    Parameters
    ----------
    player_id:
        NBA player ID. Default is **201939** (Stephen Curry).
    season:
        NBA season string, e.g. ``"2023-24"``.

    Returns
    -------
    pd.DataFrame
        Shot chart data with all columns required by the pipeline.

    Raises
    ------
    RuntimeError
        If the nba_api request fails for any reason.
    """
    import requests  # type: ignore[import]

    from nba_api.stats.endpoints.shotchartdetail import ShotChartDetail  # type: ignore[import]

    time.sleep(0.6)  # respect NBA API rate limits

    _MAX_RETRIES = 3
    _RETRY_SLEEP = 30

    raw: pd.DataFrame | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            shot_chart = ShotChartDetail(
                player_id=player_id,
                team_id=0,
                season_nullable=season,
                season_type_all_star="Regular Season",
                context_measure_simple="FGA",
            )
            raw = shot_chart.get_data_frames()[0]
            break  # success
        except (json.JSONDecodeError, requests.exceptions.ReadTimeout) as exc:
            if attempt < _MAX_RETRIES:
                print(
                    f"  [nba_api] Attempt {attempt}/{_MAX_RETRIES} failed "
                    f"({type(exc).__name__}: {exc}). "
                    f"Retrying in {_RETRY_SLEEP}s …"
                )
                time.sleep(_RETRY_SLEEP)
            else:
                raise RuntimeError(
                    f"nba_api request failed after {_MAX_RETRIES} attempts "
                    f"for player_id={player_id}, season={season!r}. "
                    f"Original error: {exc}"
                ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"nba_api request failed for player_id={player_id}, season={season!r}. "
                "Check your network connection and that nba_api is installed correctly. "
                f"Original error: {exc}"
            ) from exc

    if raw is None:  # should never be reached, but guards against future changes
        raise RuntimeError(
            f"nba_api request returned no data for player_id={player_id}, season={season!r}."
        )

    # Rename raw NBA API columns to the pipeline schema
    column_map = {
        "LOC_X": "loc_x",
        "LOC_Y": "loc_y",
        "PERIOD": "period",
        "MINUTES_REMAINING": "minutes_remaining",
        "SECONDS_REMAINING": "seconds_remaining",
        "SHOT_TYPE": "shot_type",
        "EVENT_TYPE": "shot_result",
        "SHOT_MADE_FLAG": "shot_made_flag",
        "SHOT_DISTANCE": "shot_distance",
        "SHOT_ZONE_BASIC": "shot_zone_basic",
        "PLAYER_NAME": "player_name",
    }
    df = raw.rename(columns=column_map)

    # Derived geometry features
    df["shot_angle"] = np.degrees(np.arctan2(df["loc_y"], df["loc_x"]))
    df["shot_value"] = df["shot_type"].apply(lambda t: 3 if "3PT" in str(t) else 2)

    # Placeholder columns that the downstream pipeline expects but are not
    # available from the shot chart endpoint
    df["score_diff"] = 0
    df["shot_clock"] = 12.0
    df["home_score"] = 0
    df["away_score"] = 0

    # Simple placeholder: treat the actual make/miss flag as the true probability
    df["true_make_prob"] = df["shot_made_flag"].astype(float)

    # Validate required columns are present
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise RuntimeError(
            f"Fetched shot chart is missing required columns after renaming: {missing_list}"
        )

    return df


def fetch_league_shot_chart(*_args, **_kwargs) -> pd.DataFrame:
    """Placeholder hook kept for backwards compatibility."""
    raise NotImplementedError(
        "Use fetch_player_shot_chart() to pull real player data from nba_api."
    )


def save_processed_dataset(dataframe: pd.DataFrame, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output, index=False)
    return output
