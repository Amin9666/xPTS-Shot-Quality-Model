from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROCESSED_DATA_PATH = Path("data/processed/shots_model_input.csv")
MODEL_PATH = Path("models/xpts_model.pkl")
OUTPUTS_PATH = Path("outputs")


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if PROCESSED_DATA_PATH.exists():
        return pd.read_csv(PROCESSED_DATA_PATH)
    # Minimal fallback stub so the app still renders without the pipeline having run
    return pd.DataFrame({
        "player_name": ["Stephen Curry", "Jayson Tatum", "Nikola Jokic", "A'ja Wilson"],
        "shot_distance": [24.0, 14.5, 7.0, 12.0],
        "shot_angle": [12.0, -18.0, 4.0, 30.0],
        "xpts": [1.18, 0.96, 1.34, 0.99],
        "shot_made_flag": [1, 0, 1, 1],
        "shot_type": ["3PT Field Goal", "2PT Field Goal", "2PT Field Goal", "2PT Field Goal"],
        "shot_zone_basic": ["Above the Break 3", "Mid-Range", "Restricted Area", "Mid-Range"],
        "loc_x": [200, 100, 10, 80],
        "loc_y": [200, 150, 20, 100],
    })


@st.cache_resource(show_spinner=False)
def load_model() -> dict | None:
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


def _model_metrics_section(model_artifact: dict) -> None:
    metrics = model_artifact.get("metrics", {})
    st.subheader("Model Performance")
    cols = st.columns(3)
    cols[0].metric("ROC AUC",     f"{metrics.get('roc_auc', 0):.4f}")
    cols[1].metric("Log-Loss",    f"{metrics.get('log_loss', 0):.4f}")
    cols[2].metric("Brier Score", f"{metrics.get('brier_score', 0):.4f}")
    st.caption(f"Model type: **{model_artifact.get('model_name', 'unknown').upper()}** — "
               "Lower log-loss/Brier is better; higher AUC is better.")


def main() -> None:
    st.set_page_config(page_title="xPTS Shot Quality Model", layout="wide")
    st.title("🏀 xPTS Shot Quality Dashboard")
    st.caption("Expected points per shot attempt — modelled from distance, angle, context, and player tendencies.")

    df = load_data()
    model_artifact = load_model()

    # ── Sidebar filters ────────────────────────────────────────────────────
    st.sidebar.header("Filters")
    player_options = sorted(df["player_name"].dropna().unique().tolist()) if "player_name" in df.columns else []
    selected_players = st.sidebar.multiselect("Players", player_options, default=player_options)

    zone_options = sorted(df["shot_zone_basic"].dropna().unique().tolist()) if "shot_zone_basic" in df.columns else []
    selected_zones = st.sidebar.multiselect("Shot Zones", zone_options, default=zone_options)

    shot_type_options = sorted(df["shot_type"].dropna().unique().tolist()) if "shot_type" in df.columns else []
    selected_types = st.sidebar.multiselect("Shot Types", shot_type_options, default=shot_type_options)

    filtered = df.copy()
    if selected_players and "player_name" in filtered.columns:
        filtered = filtered[filtered["player_name"].isin(selected_players)]
    if selected_zones and "shot_zone_basic" in filtered.columns:
        filtered = filtered[filtered["shot_zone_basic"].isin(selected_zones)]
    if selected_types and "shot_type" in filtered.columns:
        filtered = filtered[filtered["shot_type"].isin(selected_types)]

    # ── Top-line metrics ───────────────────────────────────────────────────
    top_cols = st.columns(4)
    top_cols[0].metric("Shots in view", f"{len(filtered):,}")
    top_cols[1].metric("Avg xPTS",      f"{filtered['xpts'].mean():.3f}" if "xpts" in filtered.columns else "N/A")
    top_cols[2].metric("Actual Make %", f"{filtered['shot_made_flag'].mean():.1%}" if "shot_made_flag" in filtered.columns else "N/A")
    if "xpts" in filtered.columns and "shot_made_flag" in filtered.columns:
        avg_xpts = filtered["xpts"].mean()
        avg_shot_value = filtered["shot_value"].mean() if "shot_value" in filtered.columns else 2.0
        actual_pts = filtered["shot_made_flag"].mean() * avg_shot_value
        xpts_diff = avg_xpts - actual_pts
    else:
        xpts_diff = 0
    top_cols[3].metric("xPTS vs Actual PTS diff", f"{xpts_diff:+.3f}")

    st.divider()

    # ── Shot chart ─────────────────────────────────────────────────────────
    st.subheader("Shot Chart – coloured by xPTS")
    if {"loc_x", "loc_y", "xpts"}.issubset(filtered.columns):
        fig_chart = px.scatter(
            filtered.sample(min(3000, len(filtered)), random_state=1),
            x="loc_x",
            y="loc_y",
            color="xpts",
            color_continuous_scale="RdYlGn",
            range_color=[0.4, 1.8],
            hover_name="player_name" if "player_name" in filtered.columns else None,
            hover_data={"xpts": ":.3f", "shot_zone_basic": True, "shot_distance": ":.1f"},
            title="",
            opacity=0.65,
            size_max=6,
        )
        fig_chart.update_traces(marker=dict(size=5))
        fig_chart.update_layout(
            plot_bgcolor="#1a1a2e",
            paper_bgcolor="#1a1a2e",
            font_color="white",
            height=480,
            xaxis=dict(range=[-260, 260], showgrid=False, title=""),
            yaxis=dict(range=[-60, 500], showgrid=False, title="", scaleanchor="x", scaleratio=1),
        )
        st.plotly_chart(fig_chart, use_container_width=True)

    # ── Player summary ─────────────────────────────────────────────────────
    st.subheader("Player Summary – Shot Quality")
    if {"player_name", "xpts", "shot_made_flag", "shot_distance"}.issubset(filtered.columns):
        player_tbl = (
            filtered.groupby("player_name")
            .agg(
                Shots=("xpts", "count"),
                Avg_xPTS=("xpts", "mean"),
                Make_Rate=("shot_made_flag", "mean"),
                Avg_Distance_ft=("shot_distance", "mean"),
            )
            .sort_values("Avg_xPTS", ascending=False)
            .reset_index()
        )
        player_tbl["Avg_xPTS"] = player_tbl["Avg_xPTS"].map("{:.3f}".format)
        player_tbl["Make_Rate"] = player_tbl["Make_Rate"].map("{:.1%}".format)
        player_tbl["Avg_Distance_ft"] = player_tbl["Avg_Distance_ft"].map("{:.1f}".format)
        player_tbl.columns = ["Player", "Shots", "Avg xPTS", "Make %", "Avg Dist (ft)"]

        left_col, right_col = st.columns((1, 1))
        with left_col:
            st.dataframe(player_tbl, use_container_width=True, hide_index=True)
        with right_col:
            bar_data = filtered.groupby("player_name")["xpts"].mean().sort_values(ascending=True)
            fig_bar = px.bar(
                x=bar_data.values,
                y=bar_data.index,
                orientation="h",
                color=bar_data.values,
                color_continuous_scale="RdYlGn",
                labels={"x": "Average xPTS", "y": ""},
                title="Average xPTS per Shot",
            )
            fig_bar.update_layout(showlegend=False, coloraxis_showscale=False, height=350)
            st.plotly_chart(fig_bar, use_container_width=True)

    # ── xPTS by zone box plot ──────────────────────────────────────────────
    if {"shot_zone_basic", "xpts"}.issubset(filtered.columns):
        st.subheader("xPTS Distribution by Shot Zone")
        zone_medians = filtered.groupby("shot_zone_basic")["xpts"].median().sort_values(ascending=False)
        fig_box = px.box(
            filtered,
            x="shot_zone_basic",
            y="xpts",
            color="shot_zone_basic",
            category_orders={"shot_zone_basic": zone_medians.index.tolist()},
            title="",
            points=False,
        )
        fig_box.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_box, use_container_width=True)

    # ── Model performance section ──────────────────────────────────────────
    if model_artifact:
        st.divider()
        _model_metrics_section(model_artifact)

    # ── Saved output charts ────────────────────────────────────────────────
    chart_files = {
        "Shot Chart (Matplotlib)": OUTPUTS_PATH / "shot_chart_xpts.png",
        "ROC Curves": OUTPUTS_PATH / "roc_curves.png",
        "Calibration Curves": OUTPUTS_PATH / "calibration_curves.png",
        "Feature Importance": OUTPUTS_PATH / "feature_importance.png",
        "Player Summary": OUTPUTS_PATH / "player_summary.png",
        "xPTS by Zone": OUTPUTS_PATH / "xpts_by_zone.png",
    }
    available = {name: path for name, path in chart_files.items() if path.exists()}
    if available:
        st.divider()
        st.subheader("Model Output Charts")
        names = list(available.keys())
        tabs = st.tabs(names)
        for tab, name in zip(tabs, names):
            with tab:
                st.image(str(available[name]), use_container_width=True)


if __name__ == "__main__":
    main()

