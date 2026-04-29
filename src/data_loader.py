from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


def fetch_league_shot_chart(*_args, **_kwargs) -> pd.DataFrame:
    """Placeholder hook for future nba_api integration."""
    raise NotImplementedError(
        "Connect this function to nba_api shot chart endpoints when you add credentials or season parameters."
    )


def save_processed_dataset(dataframe: pd.DataFrame, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output, index=False)
    return output
