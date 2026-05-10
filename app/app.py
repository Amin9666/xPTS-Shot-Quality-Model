from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
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
        "shot_value": [3, 2, 2, 2],
        "shot_archetype": ["Above-Break 3", "Mid-Range", "Rim", "Mid-Range"],
        "decision_quality": [0.05, -0.03, 0.08, -0.01],
    })


@st.cache_resource(show_spinner=False)
def load_model() -> dict | None:
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


@st.cache_data(show_spinner=False)
def load_ablation() -> pd.DataFrame | None:
    p = OUTPUTS_PATH / "ablation_study.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


@st.cache_data(show_spinner=False)
def load_metrics() -> pd.DataFrame | None:
    p = OUTPUTS_PATH / "model_metrics.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


def _sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")
    player_options = sorted(df["player_name"].dropna().unique().tolist()) if "player_name" in df.columns else []
    selected_players = st.sidebar.multiselect("Players", player_options, default=player_options[:50] if len(player_options) > 50 else player_options)

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
    return filtered


def _top_metrics(filtered: pd.DataFrame) -> None:
    top_cols = st.columns(4)
    top_cols[0].metric("Shots in view", f"{len(filtered):,}")
    top_cols[1].metric("Avg xPTS", f"{filtered['xpts'].mean():.3f}" if "xpts" in filtered.columns else "N/A")
    top_cols[2].metric("Actual Make %", f"{filtered['shot_made_flag'].mean():.1%}" if "shot_made_flag" in filtered.columns else "N/A")
    if "xpts" in filtered.columns and "shot_made_flag" in filtered.columns:
        avg_xpts = filtered["xpts"].mean()
        avg_sv = filtered["shot_value"].mean() if "shot_value" in filtered.columns else 2.0
        actual_pts = filtered["shot_made_flag"].mean() * avg_sv
        xpts_diff = avg_xpts - actual_pts
    else:
        xpts_diff = 0.0
    top_cols[3].metric("xPTS − Actual PTS diff", f"{xpts_diff:+.3f}")


def _tab_shot_chart(filtered: pd.DataFrame) -> None:
    st.subheader("Shot Chart — coloured by xPTS")
    if {"loc_x", "loc_y", "xpts"}.issubset(filtered.columns):
        fig = px.scatter(
            filtered.sample(min(3000, len(filtered)), random_state=1),
            x="loc_x", y="loc_y",
            color="xpts",
            color_continuous_scale="RdYlGn",
            range_color=[0.4, 1.8],
            hover_name="player_name" if "player_name" in filtered.columns else None,
            hover_data={"xpts": ":.3f", "shot_zone_basic": True, "shot_distance": ":.1f"},
            opacity=0.65,
        )
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(
            plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e", font_color="white",
            height=480,
            xaxis=dict(range=[-260, 260], showgrid=False, title=""),
            yaxis=dict(range=[-60, 500], showgrid=False, title="", scaleanchor="x", scaleratio=1),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Shot chart requires loc_x, loc_y, and xpts columns.")

    if "shot_archetype" in filtered.columns and {"loc_x", "loc_y"}.issubset(filtered.columns):
        st.subheader("Shot Archetypes (K-Means Clusters)")
        fig_arch = px.scatter(
            filtered.sample(min(3000, len(filtered)), random_state=2),
            x="loc_x", y="loc_y",
            color="shot_archetype",
            hover_name="player_name" if "player_name" in filtered.columns else None,
            opacity=0.55,
        )
        fig_arch.update_traces(marker=dict(size=5))
        fig_arch.update_layout(
            plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e", font_color="white",
            height=420,
            xaxis=dict(range=[-260, 260], showgrid=False, title=""),
            yaxis=dict(range=[-60, 500], showgrid=False, title="", scaleanchor="x", scaleratio=1),
        )
        st.plotly_chart(fig_arch, use_container_width=True)

    if "shot_zone_basic" in filtered.columns and "xpts" in filtered.columns:
        st.subheader("xPTS Distribution by Shot Zone")
        zone_medians = filtered.groupby("shot_zone_basic")["xpts"].median().sort_values(ascending=False)
        fig_box = px.box(
            filtered, x="shot_zone_basic", y="xpts",
            color="shot_zone_basic",
            category_orders={"shot_zone_basic": zone_medians.index.tolist()},
            points=False,
        )
        fig_box.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_box, use_container_width=True)


def _tab_player_analysis(filtered: pd.DataFrame) -> None:
    if not {"player_name", "xpts", "shot_made_flag", "shot_distance"}.issubset(filtered.columns):
        st.warning("Player analysis requires player_name, xpts, shot_made_flag, and shot_distance columns.")
        return

    agg: dict = {
        "Shots": ("xpts", "count"),
        "Avg_xPTS": ("xpts", "mean"),
        "Make_Rate": ("shot_made_flag", "mean"),
        "Avg_Distance_ft": ("shot_distance", "mean"),
    }
    if "shot_value" in filtered.columns:
        agg["Avg_Shot_Value"] = ("shot_value", "mean")
    if "decision_quality" in filtered.columns:
        agg["Avg_Decision_Quality"] = ("decision_quality", "mean")

    player_tbl = (
        filtered.groupby("player_name")
        .agg(**agg)
        .sort_values("Avg_xPTS", ascending=False)
        .reset_index()
    )

    if "Avg_Shot_Value" in player_tbl.columns:
        player_tbl["Expected_Make_Rate"] = (
            player_tbl["Avg_xPTS"] / player_tbl["Avg_Shot_Value"].replace(0, 2.0)
        )
        player_tbl["Performance_Delta"] = player_tbl["Make_Rate"] - player_tbl["Expected_Make_Rate"]

    # ── Summary bar chart ──────────────────────────────────────────────
    st.subheader("Average xPTS per Shot — Top Players by Volume")
    top_vol = player_tbl.nlargest(min(25, len(player_tbl)), "Shots").sort_values("Avg_xPTS")
    fig_bar = px.bar(
        top_vol, x="Avg_xPTS", y="player_name",
        orientation="h",
        color="Avg_xPTS",
        color_continuous_scale="RdYlGn",
        labels={"Avg_xPTS": "Average xPTS", "player_name": ""},
    )
    fig_bar.update_layout(showlegend=False, coloraxis_showscale=False, height=max(350, len(top_vol) * 18))
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Residual / over-under performance view ────────────────────────
    if "Performance_Delta" in player_tbl.columns:
        st.subheader("Player Portfolio Efficiency: Actual vs Expected Make Rate")
        st.caption(
            "**Performance Delta** = Actual Make % − Expected Make % (derived from avg xPTS). "
            "Green = outperforming shot quality model; red = underperforming."
        )
        fig_res = px.scatter(
            player_tbl.nlargest(min(60, len(player_tbl)), "Shots"),
            x="Avg_xPTS", y="Make_Rate",
            color="Performance_Delta",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            size="Shots", size_max=25,
            hover_name="player_name",
            hover_data={"Performance_Delta": ":.3f", "Shots": True},
            labels={"Avg_xPTS": "Avg xPTS (shot quality)", "Make_Rate": "Actual Make %"},
        )
        # Reference line: perfect calibration (actual = expected)
        min_xpts = float(player_tbl["Avg_xPTS"].min())
        max_xpts = float(player_tbl["Avg_xPTS"].max())
        avg_shot_value = float(filtered["shot_value"].mean()) if "shot_value" in filtered.columns else 2.0
        fig_res.add_trace(go.Scatter(
            x=[min_xpts, max_xpts],
            y=[min_xpts / avg_shot_value, max_xpts / avg_shot_value],
            mode="lines",
            line=dict(color="white", dash="dash", width=1.5),
            name="Expected (model-calibrated)",
        ))
        fig_res.update_layout(height=480)
        st.plotly_chart(fig_res, use_container_width=True)

    # ── Data table ────────────────────────────────────────────────────
    st.subheader("Player Summary Table")
    display_cols = {
        "player_name": "Player",
        "Shots": "Shots",
        "Avg_xPTS": "Avg xPTS",
        "Make_Rate": "Make %",
        "Avg_Distance_ft": "Avg Dist (ft)",
    }
    if "Performance_Delta" in player_tbl.columns:
        display_cols["Performance_Delta"] = "Perf. Δ"
    if "Avg_Decision_Quality" in player_tbl.columns:
        display_cols["Avg_Decision_Quality"] = "Avg Decision Quality"

    show_tbl = player_tbl[[c for c in display_cols if c in player_tbl.columns]].copy()
    show_tbl = show_tbl.rename(columns=display_cols)
    for col in ["Avg xPTS", "Make %", "Avg Dist (ft)", "Perf. Δ", "Avg Decision Quality"]:
        if col in show_tbl.columns:
            if col == "Make %":
                show_tbl[col] = show_tbl[col].map("{:.1%}".format)
            else:
                show_tbl[col] = show_tbl[col].map("{:.3f}".format)
    st.dataframe(show_tbl, use_container_width=True, hide_index=True)


def _tab_model_insights(model_artifact: dict | None) -> None:
    # ── Model comparison table ─────────────────────────────────────────
    metrics_df = load_metrics()
    if metrics_df is not None and not metrics_df.empty:
        st.subheader("Model Comparison")
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        metric_cols = [c for c in metrics_df.columns if c != "Model"]
        melt = metrics_df.melt(id_vars="Model", value_vars=metric_cols, var_name="Metric", value_name="Value")
        fig_comp = px.bar(
            melt, x="Metric", y="Value", color="Model",
            barmode="group",
            title="Model Metrics Comparison (hold-out test set)",
        )
        fig_comp.update_layout(height=400)
        st.plotly_chart(fig_comp, use_container_width=True)
    elif model_artifact:
        metrics = model_artifact.get("metrics", {})
        st.subheader("Model Performance")
        cols = st.columns(3)
        cols[0].metric("ROC AUC", f"{metrics.get('roc_auc', 0):.4f}")
        cols[1].metric("Log-Loss", f"{metrics.get('log_loss', 0):.4f}")
        cols[2].metric("Brier Score", f"{metrics.get('brier_score', 0):.4f}")
        st.caption(f"Model type: **{model_artifact.get('model_name', 'unknown').upper()}** — "
                   "Lower log-loss/Brier is better; higher AUC is better.")

    # ── Ablation study ─────────────────────────────────────────────────
    ablation_df = load_ablation()
    if ablation_df is not None and not ablation_df.empty:
        st.subheader("Ablation Study — Incremental Value of Feature Groups")
        st.caption(
            "Research question: *How much of shot-outcome variance is explained by geometry "
            "vs game context vs player skill?* Each tier adds one feature group; all other "
            "settings (model, split) are held constant."
        )
        abl_fig = px.bar(
            ablation_df, x="roc_auc", y="tier", orientation="h",
            color="tier",
            text=ablation_df["roc_auc"].map("{:.4f}".format),
            labels={"roc_auc": "ROC-AUC", "tier": "Feature Tier"},
            title="Ablation Study: ROC-AUC by Feature Tier (XGBoost)",
        )
        abl_fig.update_traces(textposition="outside")
        abl_fig.update_layout(showlegend=False, height=320)
        st.plotly_chart(abl_fig, use_container_width=True)
        st.dataframe(ablation_df.round(4), use_container_width=True, hide_index=True)

    # ── Saved diagnostic charts ────────────────────────────────────────
    chart_files = {
        "Calibration Comparison": OUTPUTS_PATH / "calibration_comparison.png",
        "Calibration Curves": OUTPUTS_PATH / "calibration_curves.png",
        "ROC Curves": OUTPUTS_PATH / "roc_curves.png",
        "PR Curves": OUTPUTS_PATH / "pr_curves.png",
        "Bootstrap CI": OUTPUTS_PATH / "bootstrap_ci.png",
        "Ablation Study": OUTPUTS_PATH / "ablation_study.png",
        "Feature Importance": OUTPUTS_PATH / "feature_importance.png",
        "Permutation Importance": OUTPUTS_PATH / "permutation_importance.png",
        "Learning Curves": OUTPUTS_PATH / "learning_curves.png",
    }
    available = {name: path for name, path in chart_files.items() if path.exists()}
    if available:
        st.subheader("Diagnostic Charts")
        names = list(available.keys())
        tabs = st.tabs(names)
        for tab, name in zip(tabs, names):
            with tab:
                st.image(str(available[name]), use_container_width=True)


def _tab_whatif(model_artifact: dict | None, df: pd.DataFrame) -> None:
    st.subheader("What-If Explorer")
    st.caption(
        "Adjust shot parameters and see how the model's predicted xPTS changes in real time. "
        "Derived features (distance², log-distance, dist×angle, late-clock flag) are computed automatically."
    )

    if model_artifact is None:
        st.warning("No model artifact found. Run `python run_pipeline.py` first.")
        return

    feature_columns: list[str] = model_artifact.get("feature_columns", [])

    col_l, col_r = st.columns(2)
    with col_l:
        shot_distance = st.slider("Shot Distance (ft)", 0, 35, 15, key="wi_dist")
        shot_angle = st.slider("Shot Angle (°, 90°=straight-on)", -90, 90, 0, key="wi_ang")
        period = st.slider("Period", 1, 4, 2, key="wi_per")
        game_seconds_remaining = st.slider("Game Seconds Remaining", 0, 2880, 600, key="wi_gsec")
    with col_r:
        score_diff_abs = st.slider("Score Differential |Δ|", 0, 40, 5, key="wi_sd")
        shot_clock = st.slider("Shot Clock (sec)", 0, 24, 12, key="wi_sc")
        league_avg_fg = 0.467  # approximate 2024-25 NBA FG% (source: nba.com/stats)
        if "player_zone_fg_pct" in feature_columns:
            player_zone_fg_pct = st.slider(
                "Player Zone FG% (use league avg as default)",
                0.0, 1.0, float(df["player_zone_fg_pct"].mean()) if "player_zone_fg_pct" in df.columns else league_avg_fg,
                step=0.01, key="wi_pz",
            )
        else:
            player_zone_fg_pct = league_avg_fg
        is_3pt = st.checkbox("3-Point Attempt", value=False, key="wi_3pt")

    # Compute derived features
    distance_sq = float(shot_distance) ** 2
    log1p_distance = float(np.log1p(shot_distance))
    dist_angle_ix = float(shot_distance) * float(abs(shot_angle))
    late_clock = 1 if shot_clock <= 4 else 0
    shot_value = 3 if is_3pt else 2

    input_row: dict = {
        "shot_distance": shot_distance,
        "shot_angle": float(shot_angle),
        "distance_sq": distance_sq,
        "log1p_distance": log1p_distance,
        "dist_angle_ix": dist_angle_ix,
        "period": float(period),
        "game_seconds_remaining": float(game_seconds_remaining),
        "score_diff_abs": float(score_diff_abs),
        "player_zone_fg_pct": player_zone_fg_pct,
        "late_clock": late_clock,
        "shot_clock": float(shot_clock),
    }
    input_df = pd.DataFrame([input_row])
    available_feats = [f for f in feature_columns if f in input_df.columns]

    try:
        make_prob = float(
            model_artifact["pipeline"].predict_proba(input_df[available_feats])[0, 1]
        )
        xpts = make_prob * shot_value

        res_cols = st.columns(3)
        res_cols[0].metric("Make Probability", f"{make_prob:.3f}")
        res_cols[1].metric("Shot Value (pts)", str(shot_value))
        res_cols[2].metric("xPTS", f"{xpts:.3f}", delta=f"{xpts - df['xpts'].mean():.3f} vs league avg" if "xpts" in df.columns else None)

        # Distance sensitivity curve
        distances = list(range(0, 36))
        probs_curve = []
        for d in distances:
            row_d = input_row.copy()
            row_d["shot_distance"] = float(d)
            row_d["distance_sq"] = float(d) ** 2
            row_d["log1p_distance"] = float(np.log1p(d))
            row_d["dist_angle_ix"] = float(d) * float(abs(shot_angle))
            dfr = pd.DataFrame([row_d])
            p = float(model_artifact["pipeline"].predict_proba(dfr[available_feats])[0, 1])
            probs_curve.append(p * shot_value)

        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(
            x=distances, y=probs_curve, mode="lines+markers",
            name="xPTS vs Distance",
            line=dict(color="#2ecc71", width=2),
        ))
        fig_sens.add_vline(x=shot_distance, line_dash="dash", line_color="#e74c3c",
                           annotation_text=f"Current ({shot_distance} ft)")
        fig_sens.update_layout(
            title="xPTS Sensitivity to Shot Distance (all other inputs held fixed)",
            xaxis_title="Shot Distance (ft)",
            yaxis_title="xPTS",
            height=350,
        )
        st.plotly_chart(fig_sens, use_container_width=True)

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")


def main() -> None:
    st.set_page_config(page_title="xPTS Shot Quality Model", layout="wide")
    st.title("🏀 xPTS Shot Quality Dashboard")
    st.caption(
        "Expected points per shot attempt — calibrated probabilistic model of NBA shot quality "
        "using geometry, game context, and player-zone shooting history."
    )

    df = load_data()
    model_artifact = load_model()
    filtered = _sidebar_filters(df)

    _top_metrics(filtered)
    st.divider()

    tab_names = ["Shot Chart & Zones", "Player Analysis", "Model Insights", "What-If Explorer"]
    tab1, tab2, tab3, tab4 = st.tabs(tab_names)

    with tab1:
        _tab_shot_chart(filtered)

    with tab2:
        _tab_player_analysis(filtered)

    with tab3:
        _tab_model_insights(model_artifact)

    with tab4:
        _tab_whatif(model_artifact, df)


if __name__ == "__main__":
    main()

