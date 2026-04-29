# xPTS Shot Quality Model

An end-to-end NBA analytics project for estimating expected points from a shot attempt by modeling shot quality beyond raw make or miss outcomes.

## Problem Statement

NBA teams and analysts need a better way to evaluate shot selection than field goal percentage alone. A make or miss is noisy on a single possession, but the underlying quality of the shot is more stable. This project builds an `xPTS` style shot quality model that estimates the expected value of each shot attempt using contextual features such as distance, angle, clock pressure, and shooter tendencies.

## Project Goals

- Build a reproducible data pipeline for NBA shot-level data.
- Explore how shot context affects expected scoring value.
- Train a classification or probabilistic model that estimates shot make probability.
- Convert make probability into expected points.
- Visualize results with shot charts, calibration curves, and player-level summaries.
- Ship a Streamlit dashboard that can be shared with recruiters or hiring managers.

## Data Sources

All suggested sources are free to access.

- `nba_api`: official NBA stats endpoints for shot charts, play-by-play, and game logs.
- Basketball-Reference: useful for player and team tables via `pandas.read_html` or `BeautifulSoup`.
- `pbpstats.com`: advanced play-by-play context for possessions and game state.
- Kaggle NBA datasets: fast starting point for pre-cleaned CSVs and historical shot logs.

## Recommended Tech Stack

| Layer | Tools |
| --- | --- |
| Data collection | `nba_api`, `requests`, `BeautifulSoup` |
| Data processing | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| Modeling | `scikit-learn`, `xgboost`, `lightgbm` |
| App layer | `streamlit` |
| Version control | `git`, GitHub |

## Suggested Project Structure

```text
xPTS-Shot-Quality-Model/
├── app/
│   └── app.py
├── data/
│   ├── processed/
│   └── raw/
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_modeling.ipynb
├── src/
│   ├── data_loader.py
│   ├── features.py
│   └── model.py
├── README.md
└── requirements.txt
```

## Core Features To Engineer

These features make the project more realistic and more compelling to interviewers.

- Shot distance and shot angle.
- Shot clock remaining.
- Game context such as quarter, time remaining, and score differential.
- Player shooting history from the same area of the floor.
- Dribbles before shot and touch time when that data is available.
- Defender distance if tracking data is available.

## Methodology

### 1. Data Pipeline

- Pull raw shot and play-by-play data by season and game.
- Standardize columns and save raw extracts under `data/raw/`.
- Build a processed shot-level table under `data/processed/`.

### 2. Exploratory Data Analysis

- Inspect class balance, shot zones, and team or player distributions.
- Plot spatial shot density and make rate heatmaps.
- Review how shot quality shifts by quarter, score margin, and clock pressure.

### 3. Feature Engineering

- Create geometry-based features such as distance and angle.
- Create situational features from score margin and game clock.
- Aggregate historical zone efficiency at player and team levels.
- Handle missing tracking features with fallbacks.

### 4. Modeling

- Start with logistic regression as a baseline.
- Compare with tree-based models such as XGBoost and LightGBM.
- Evaluate using AUC, log-loss, Brier score, and calibration.

### 5. Results and Communication

- Convert predicted make probability into expected points.
- Surface the highest and lowest quality shots by player, team, or game context.
- Package the results in a dashboard with interactive filters.

## Metrics To Report

- ROC AUC
- Log-loss
- Brier score
- Calibration curve
- Feature importance or model explainability summary

## What To Show In The Dashboard

- Interactive shot chart colored by predicted `xPTS`.
- Filters for season, team, player, and shot zone.
- Calibration and performance views.
- Player summary tables such as average shot quality and expected points per attempt.

## Example Findings To Highlight

- Which players create the highest quality looks within a team offense.
- Which teams generate the most efficient corner threes.
- Whether a player is outperforming or underperforming shot quality expectations.

## Getting Started

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the dashboard

```bash
streamlit run app/app.py
```

## Next Build Targets

- Connect `src/data_loader.py` to real `nba_api` endpoints.
- Save a processed modeling dataset to `data/processed/shots.csv`.
- Train and serialize a baseline model artifact.
- Replace placeholder charts in the Streamlit app with modeled outputs.

## Deliverables

- Clean notebook workflow for collection, EDA, feature engineering, and modeling.
- Modular Python pipeline in `src/`.
- Public GitHub repository with a strong README.
- Streamlit demo link once deployed.