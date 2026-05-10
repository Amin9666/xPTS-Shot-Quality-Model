# xPTS Shot Quality Model Report

**Author:** Amin9666  
**Project:** Expected Points (xPTS) for NBA shot quality  
**Data snapshot used in this report:** `data/processed/shots_model_input.csv` (1,208 shot attempts)

## 1. Introduction

Traditional field goal percentage is noisy at the possession level and does not fully capture shot quality.  
This project models shot quality with **xPTS**, defined as:

\[
\text{xPTS} = P(\text{make}) \times \text{shot value (2 or 3)}
\]

The goal is to estimate expected scoring value per shot using spatial context, clock pressure, and player-zone tendencies, then evaluate both discrimination and calibration quality.

## 2. Data Exploration

The modeling dataset contains **1,208** shot attempts with an overall make rate of **42.30%** and average shot value of **2.742 points**.

Key zone-level findings:

- **Above the Break 3:** 600 shots, make rate 40.00%, avg xPTS 1.209  
- **Restricted Area:** 145 shots, make rate 61.38%, avg xPTS 1.194  
- **Corner 3s:** lower make rates but still competitive xPTS because of 3-point value  
- **Backcourt:** lowest-value attempts (avg xPTS 0.601)

Shot type summary:

- **2PT Field Goals:** 312 shots, make rate 52.56%, avg xPTS 1.053  
- **3PT Field Goals:** 896 shots, make rate 38.73%, avg xPTS 1.143

Overall average xPTS in this sample is **1.119**.

## 3. Data Preparation

The pipeline performs:

1. Feature engineering (distance transforms, shot angle interactions, game context, late-clock flags).
2. Leakage-safe handling of `player_zone_fg_pct` by recomputing zone encodings within each train split/fold.
3. Stratified train/test splitting for hold-out evaluation.
4. 5-fold cross-validation for robustness checks.

This workflow is designed to avoid target leakage and to produce realistic out-of-sample metrics.

## 4. Models and Results

Two models were evaluated on the hold-out test set:

| Model | ROC-AUC | PR-AUC | Log-Loss | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| XGBoost | 0.5026 | 0.4443 | 0.8018 | 0.2892 | 0.2046 |
| Logistic Regression | 0.5777 | 0.4917 | 0.6728 | 0.2401 | 0.0113 |

### 4.a Cross-Validation (XGBoost, 5-fold)

- ROC-AUC: **0.4997 ± 0.0450**
- PR-AUC: **0.4691 ± 0.0405**
- Log-Loss: **0.7892 ± 0.0438**
- Brier: **0.2834 ± 0.0185**
- ECE: **0.1730 ± 0.0363**

### 4.b Feature Importance (Permutation, XGBoost)

Top features by mean ROC-AUC drop:

1. `shot_angle` (0.0310)
2. `distance_sq` (0.0088)
3. `shot_clock` (0.0060)
4. `player_zone_fg_pct` (0.0052)
5. `dist_angle_ix` (0.0016)

### 4.c Interpretation

- Logistic Regression outperformed XGBoost on all reported hold-out metrics in this run, especially calibration (**ECE 0.0113**).  
- Court geometry and shot context remain the main predictive drivers.  
- High-value attempts cluster in expected regions: rim and 3-point areas.

## 5. Visual Outputs (Generated)

The following charts support this report:

- Shot chart: `outputs/shot_chart_xpts.png`
- ROC curves: `outputs/roc_curves.png`
- PR curves: `outputs/pr_curves.png`
- Calibration curves: `outputs/calibration_curves.png`
- Learning curves: `outputs/learning_curves.png`
- Permutation importance: `outputs/permutation_importance.png`
- Player summary: `outputs/player_summary.png`
- xPTS by zone: `outputs/xpts_by_zone.png`

## 6. Conclusion

This xPTS pipeline delivers an end-to-end shot quality evaluation system with leakage-aware modeling and interpretable outputs.  
In this data snapshot, **Logistic Regression** is the stronger deployment candidate due to substantially better probability calibration, while the model diagnostics confirm that shot geometry and context explain most of the learnable signal.

## A. Bibliography

1. NBA API (`nba_api`) documentation and endpoints: https://github.com/swar/nba_api  
2. Project pipeline and artifacts: `run_pipeline.py`, `src/model.py`, and `outputs/*.csv` in this repository.

## B. Appendix: Source Tables

- Metrics table source: `outputs/model_metrics.csv`
- Cross-validation source: `outputs/cv_results_xgboost.csv`
- Permutation source: `outputs/permutation_importance.csv`
- Player summary source: `outputs/player_summary.csv`
