from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


DATA_PATH = Path("data/processed/shots_model_input.csv")


def load_dashboard_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)

    return pd.DataFrame(
        {
            "player_name": ["Stephen Curry", "Jayson Tatum", "Nikola Jokic", "A'ja Wilson"],
            "shot_distance": [24.0, 14.5, 7.0, 12.0],
            "shot_angle": [12.0, -18.0, 4.0, 30.0],
            "xpts": [1.18, 0.96, 1.34, 0.99],
            "shot_made_flag": [1, 0, 1, 1],
            "shot_type": ["3PT Field Goal", "2PT Field Goal", "2PT Field Goal", "2PT Field Goal"],
        }
    )


def main() -> None:
    st.set_page_config(page_title="xPTS Shot Quality Model", layout="wide")
    st.title("xPTS Shot Quality Dashboard")
    st.caption("Expected points per shot based on context, geometry, and player tendencies.")

    dataframe = load_dashboard_data()

    player_options = sorted(dataframe["player_name"].dropna().unique().tolist()) if "player_name" in dataframe.columns else []
    selected_players = st.sidebar.multiselect("Players", player_options, default=player_options[:4])

    filtered = dataframe.copy()
    if selected_players and "player_name" in filtered.columns:
        filtered = filtered[filtered["player_name"].isin(selected_players)]

    metric_columns = st.columns(3)
    metric_columns[0].metric("Shots", f"{len(filtered):,}")
    metric_columns[1].metric("Average xPTS", f"{filtered['xpts'].mean():.2f}" if "xpts" in filtered.columns else "N/A")
    metric_columns[2].metric(
        "Make Rate",
        f"{filtered['shot_made_flag'].mean():.1%}" if "shot_made_flag" in filtered.columns else "N/A",
    )

    left, right = st.columns((2, 1))

    with left:
        chart = px.scatter(
            filtered,
            x="shot_distance",
            y="shot_angle",
            color="xpts" if "xpts" in filtered.columns else None,
            size="xpts" if "xpts" in filtered.columns else None,
            hover_name="player_name" if "player_name" in filtered.columns else None,
            title="Shot Profile by Distance and Angle",
            color_continuous_scale="blues",
        )
        st.plotly_chart(chart, use_container_width=True)

    with right:
        summary_columns = [column for column in ["player_name", "shot_type", "xpts", "shot_made_flag"] if column in filtered.columns]
        st.subheader("Shot Sample")
        st.dataframe(filtered[summary_columns], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
