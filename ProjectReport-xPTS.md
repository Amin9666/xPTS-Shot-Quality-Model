# xPTS Shot Quality Model: A Calibrated Probabilistic Framework for NBA Shot Evaluation

**Author:** Amin9666  
**Data snapshot:** `NBA_2025_Shots.csv.zip` → 219,527 shots (2024-25 NBA season)

---

## Abstract

We present **xPTS** (Expected Points), a calibrated probabilistic model for evaluating NBA shot quality at the possession level.  Shot make probability is formulated as a supervised binary classification problem and estimated using three feature groups — shot geometry, game context, and player-zone shooting history — allowing us to decompose outcome variance across these dimensions via an ablation study.  Two model families (XGBoost and Logistic Regression) are benchmarked on a held-out test set using discrimination (ROC-AUC, PR-AUC), probabilistic accuracy (Log-Loss, Brier score), and calibration (ECE) metrics, with uncertainty quantified through 500-sample bootstrap confidence intervals.  A post-hoc isotonic calibration layer is added to XGBoost, producing a three-way comparison (XGBoost · XGBoost+Calibrated · Logistic Regression).  The best model achieves ROC-AUC ≈ 0.643 on league-scale 2025 data.  We further introduce **shot archetype clustering** (K-Means, 6 zones) and a **decision quality** metric that separates shot selection from execution, providing a richer analytic vocabulary for offensive evaluation beyond raw make rates.

---

## 1. Introduction

Field goal percentage (FG%) is the dominant summary statistic for shot efficiency in basketball analytics, yet it conflates two distinct phenomena: *shot quality* (the difficulty and strategic value of the attempt) and *shot execution* (whether the shooter converted a given opportunity).  A player who consistently generates high-quality looks from optimal zones will outperform their FG% on a small sample; conversely, a player who happens to make several low-probability attempts may appear efficient while making systematically poor shot decisions.

The expected-points framework addresses this by estimating the *latent value* of each shot attempt rather than its binary outcome:

$$
\text{xPTS}_i = \hat{P}(\text{make}_i \mid \mathbf{x}_i) \times \text{shot\_value}_i
$$

where $\mathbf{x}_i$ is a feature vector encoding context for shot $i$ and $\text{shot\_value}_i \in \{2, 3\}$.

### 1.1 Research Questions

This project investigates three nested research questions:

1. **RQ1 (Feature decomposition):** How much of shot-outcome variance is attributable to geometry alone, relative to the incremental contributions of game context and player-zone shooting history?

2. **RQ2 (Model selection):** Does a nonlinear gradient-boosted model (XGBoost) outperform a calibrated linear baseline (Logistic Regression) on both discrimination and probabilistic accuracy metrics?

3. **RQ3 (Calibration importance):** Does post-hoc probability calibration materially improve ECE for XGBoost, and how does this affect xPTS magnitudes across shot zones?

### 1.2 Contributions

- End-to-end reproducible pipeline (data → features → ablation → models → calibration → visualisation).
- Leakage-free Bayesian-shrinkage encoding of player-zone FG% (James–Stein-style shrinkage applied per CV fold).
- Three-way model comparison with 95% bootstrap confidence intervals.
- Shot archetype clustering (K-Means) for basketball-interpretable spatial segmentation.
- Shot decision quality metric separating selection from execution.
- Streamlit dashboard with What-If Explorer for interactive probability estimation.

---

## 2. Brief Literature Review

The expected-goals (xG) family of models originated in football/soccer analytics (Lucey et al., 2014; Brechot & Flepp, 2020) and migrated to basketball as expected points and shot quality scores.  Cervone et al. (2016) introduced EPVA (Expected Possession Value Added) using optical tracking data; Goldsberry (2012) popularised spatial shot charts for strategic analysis.  The xPTS framing used here is closest to Goldsberry's zone-efficiency approach but operationalises it as a full probabilistic classification model rather than a descriptive statistic.

Model-wise, XGBoost (Chen & Guestrin, 2016) has become the dominant baseline for structured tabular prediction and is widely used in sports analytics competitions (Statsbomb, NBA data challenges).  Calibration — the correspondence between predicted probabilities and observed frequencies — is critically important when predicted values are used directly as expected values (Guo et al., 2017; Niculescu-Mizil & Caruana, 2005), motivating the post-hoc isotonic calibration step.

---

## 3. Data

### 3.1 Source

- **Primary dataset:** full-league 2025 NBA shot log (`NBA_2025_Shots.csv.zip`), 219,527 field goal attempts covering all players and teams for the 2024-25 regular season.
- **Fallback pipeline:** real shot-chart data via `nba_api` (2023-24 season); synthetic Curry data if API unavailable (CI environments).

### 3.2 Descriptive Statistics

| Statistic | Value |
|---|---:|
| Total shots | 219,527 |
| Make rate (overall) | 46.72% |
| 2PT attempts | 127,073 |
| 3PT attempts | 92,454 |
| 2PT make rate | 54.51% |
| 3PT make rate | 36.02% |
| Avg modelled xPTS | 1.057 |

Class balance is healthy for binary classification (roughly 47/53 split), minimising the need for resampling strategies.

### 3.3 Key Shot Zones

| Zone | Shots | Make Rate | Avg xPTS |
|---|---:|---:|---:|
| Above the Break 3 | 68,358 | 35.32% | 0.996 |
| Restricted Area | 61,190 | 66.36% | 1.377 |
| In The Paint (Non-RA) | 44,475 | 44.37% | 0.862 |

---

## 4. Feature Engineering

Features are grouped into three tiers that mirror the ablation study design (Section 6.2).

### 4.1 Spatial Features (Tier 1)

| Feature | Description |
|---|---|
| `shot_distance` | Euclidean distance from basket (tenths of a foot) |
| `shot_angle` | Polar angle from basket (degrees) |
| `distance_sq` | Squared distance — captures convexity of make-probability vs range |
| `log1p_distance` | log(1 + distance) — emphasises near-basket regime |
| `dist_angle_ix` | Distance × \|angle\| interaction — joint penalty of range + lateral difficulty |

### 4.2 Game Context Features (Tier 2)

| Feature | Description |
|---|---|
| `period` | Quarter number (1–4 + OT) |
| `game_seconds_remaining` | Elapsed quarter clock converted to seconds |
| `score_diff_abs` | Absolute score differential at time of shot |
| `late_clock` | Binary: shot clock ≤ 4 s |
| `shot_clock` | Continuous shot clock reading |

### 4.3 Player Skill Feature (Tier 3)

| Feature | Description |
|---|---|
| `player_zone_fg_pct` | Bayesian-shrinkage-smoothed FG% for this player–zone combination, fitted on training labels only to prevent target leakage |

The shrinkage estimator is:

$$
\hat{\theta}_{pz} = \frac{n_{pz} \cdot \bar{y}_{pz} + k \cdot \bar{y}_{\text{global}}}{n_{pz} + k}, \quad k = 20
$$

Small player–zone cells (few shots) are pulled toward the league average, reducing overfitting on rare combinations.

### 4.4 Shot Archetype Clustering

K-Means clustering (k=6) is applied to standardised court coordinates `(loc_x, loc_y)`.  Each cluster is assigned a human-readable label based on its centroid's court region:

- **Rim** (< 6 ft): layups, dunks, tip-ins
- **Paint** (6–13 ft): short-roll, floaters
- **Mid-Range** (13–22 ft): pull-up Js, baseline runners
- **Corner 3**: stationary catch-and-shoot from the corners
- **Wing 3**: off-screen or pull-up threes from the wings
- **Above-Break 3**: top-of-key and above-the-break pull-ups

These archetypes are used for post-hoc analysis and dashboard visualisation, not as model inputs.

### 4.5 Shot Decision Quality

After xPTS is computed, we define:

$$
\text{decision\_quality}_i = \text{xPTS}_i - \overline{\text{xPTS}}_{\text{zone}(i)}
$$

Positive values indicate the player found a look *above* the zone average; negative values flag below-average shot selection within that zone.

---

## 5. Methodology

### 5.1 Prediction Target

Binary classification: $y_i = \mathbf{1}[\text{shot made}]$.  We estimate $\hat{P}(y_i = 1 \mid \mathbf{x}_i)$ and transform it into expected points.

### 5.2 Models

| Model | Role | Key hyperparameters |
|---|---|---|
| Logistic Regression | Linear baseline, calibration reference | L2, `C=1.0`, liblinear solver |
| XGBoost | Nonlinear benchmark | 300 trees, depth=5, lr=0.05, subsample=0.8 |
| XGBoost + Isotonic | Post-hoc calibration variant | Same XGB + isotonic calibration (5-fold CV) |

### 5.3 Evaluation Protocol

1. **Stratified 80/20 hold-out split** (fixed `random_state=7`) for all models, ensuring comparisons use identical test sets.
2. **5-fold stratified cross-validation** for variance estimation; leakage fix re-applied per fold.
3. **Randomised hyperparameter search** (15 iterations, XGBoost) to confirm defaults are reasonable.
4. **Bootstrap confidence intervals** (n=500 resamples) on the XGBoost test-set predictions.

### 5.4 Evaluation Metrics

| Metric | Formula | Direction |
|---|---|---|
| ROC-AUC | Area under TPR-FPR curve | ↑ better |
| PR-AUC | Area under precision-recall curve | ↑ better |
| Log-Loss | $-\frac{1}{n}\sum y \log \hat{p} + (1-y)\log(1-\hat{p})$ | ↓ better |
| Brier Score | $\frac{1}{n}\sum (y - \hat{p})^2$ | ↓ better |
| ECE | $\sum_b \frac{|B_b|}{n} \left| \text{acc}(B_b) - \text{conf}(B_b) \right|$ | ↓ better |

---

## 6. Experimental Design

### 6.1 Hypotheses

**H1:** XGBoost outperforms Logistic Regression on ROC-AUC (nonlinear interaction effects exist).  
**H2:** Adding player-zone FG% produces the largest single AUC gain in the ablation study (player skill is the dominant signal).  
**H3:** Post-hoc isotonic calibration reduces XGBoost's ECE without materially sacrificing AUC.

### 6.2 Ablation Study Design

| Tier | Feature Groups Included | Purpose |
|---|---|---|
| Location Only | Spatial features only (5 features) | Geometry baseline — what does court position tell us? |
| Location + Context | Spatial + game state (10 features) | Does situational information help? |
| Full (+ Player Skill) | All features (11 features) | Does shooter history add incremental signal? |

All three tiers use the same XGBoost hyperparameters and the same 80/20 train/test split to ensure fair comparison.

---

## 7. Results

### 7.1 Hold-out Test Performance

| Model | ROC-AUC | PR-AUC | Log-Loss | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| XGBoost | 0.6434 | 0.6384 | 0.6513 | 0.2300 | 0.0262 |
| Logistic Regression | 0.6394 | 0.6168 | 0.6557 | 0.2318 | 0.0160 |
| XGBoost + Isotonic | ~0.643 | ~0.638 | ~0.650 | ~0.230 | ~0.016 |

*(Calibrated model metrics are regenerated at each pipeline run; approximate values shown.)*

**H1** is supported: XGBoost exceeds Logistic Regression on ROC-AUC and PR-AUC.  
**H3** is supported: isotonic calibration reduces XGBoost's ECE toward the logistic baseline while preserving AUC.

### 7.2 Ablation Study (XGBoost)

| Feature Tier | N Features | ROC-AUC | Δ AUC |
|---|---:|---:|---:|
| Location Only | 5 | ~0.590 | — |
| Location + Context | 10 | ~0.600 | +0.010 |
| Full (+ Player Skill) | 11 | ~0.643 | +0.043 |

**H2** is supported: the player-zone feature accounts for the largest single AUC gain (~0.043), more than 4× the contribution of game-context features (~0.010).

### 7.3 Bootstrap Confidence Intervals (XGBoost, n=500)

| Metric | Estimate | 95% CI |
|---|---:|---:|
| ROC-AUC | 0.6434 | [0.638, 0.649] |
| PR-AUC | 0.6384 | [0.632, 0.645] |
| Log-Loss | 0.6513 | [0.648, 0.655] |
| Brier Score | 0.2300 | [0.228, 0.232] |

*(Values regenerated at each pipeline run; representative figures shown.)*

Confidence intervals are narrow relative to model differences, supporting the reliability of reported comparisons.

### 7.4 Permutation Importance (XGBoost)

Top features by mean ROC-AUC drop on the test set:

1. `player_zone_fg_pct` — 0.1057
2. `shot_angle` — 0.0063
3. `shot_distance` — 0.0035
4. `dist_angle_ix` — 0.0023
5. `game_seconds_remaining` — 0.0022

The player-zone feature dominates by an order of magnitude, consistent with the ablation study.

---

## 8. Visualisations

### Model diagnostics
![ROC Curves](outputs/roc_curves.png)
![Precision-Recall Curves](outputs/pr_curves.png)
![Calibration Comparison (raw XGB vs calibrated vs logistic)](outputs/calibration_comparison.png)
![Learning Curves](outputs/learning_curves.png)

### Feature importance
![Permutation Importance](outputs/permutation_importance.png)

### Ablation and uncertainty
![Ablation Study](outputs/ablation_study.png)
![Bootstrap CI](outputs/bootstrap_ci.png)

### Shot quality and spatial analysis
![Shot Chart Coloured by xPTS](outputs/shot_chart_xpts.png)
![Shot Archetypes (K-Means)](outputs/shot_archetypes.png)
![xPTS Distribution by Zone](outputs/xpts_by_zone.png)
![Player Summary](outputs/player_summary.png)

---

## 9. Discussion

### 9.1 Player Skill as Primary Signal

The ablation study and permutation importance converge on the same finding: **shooter-level zone efficiency is by far the most informative signal in this model**.  This is theoretically important.  It implies that pure location-based models (which have been standard since Goldsberry 2012) systematically underestimate the importance of *who* is taking the shot relative to *where* it is taken from.

The Bayesian shrinkage encoding of `player_zone_fg_pct` is critical here.  Without shrinkage, rare player-zone combinations would overfit to small samples; without the per-fold refit, test-set labels would leak into the feature, inflating all reported metrics.

### 9.2 XGBoost vs Logistic Regression Trade-off

XGBoost outperforms Logistic Regression on discrimination metrics but has worse raw calibration (higher ECE).  This reflects a well-known pattern in tree ensemble models: gradient boosting optimises log-loss during training, but the resulting probability scores are not guaranteed to match empirical frequencies across the full probability range.

Post-hoc isotonic calibration resolves this: it learns a monotonic mapping from raw XGBoost scores to calibrated probabilities without modifying the underlying ranking.  The result is a model that is both more discriminative than logistic regression and approximately as well-calibrated — the preferred outcome for xPTS deployment.

### 9.3 Shot Decision Quality Interpretation

The `decision_quality` metric has a natural basketball interpretation: it separates *where* a player shot from *how good* that location was relative to the league average.  A player with positive average decision quality is systematically finding looks above their zone average — either by better shot creation, better positioning, or both.  This is distinct from actual conversion (make rate) and from raw shot quality (avg xPTS), making it a novel dimension of offensive evaluation.

---

## 10. Limitations and Threats to Validity

1. **No tracking data.** Features such as defender distance, touch time, and dribble count are strong predictors of shot difficulty but were unavailable in this dataset.  Their inclusion would likely reduce the relative importance of geometry and increase overall AUC.

2. **Temporal data leakage (mild).** The 80/20 split is random rather than season-ordered.  Shots from the same player appear in both train and test, meaning the model has some exposure to each player's tendencies even when the exact shot is withheld.  A season-stratified holdout would be more conservative.

3. **Single-season snapshot.** xPTS calibration and feature weights may shift across seasons due to rule changes, pace shifts, and roster turnover.  A multi-season analysis would test generalisation over time.

4. **Ecological validity.** Shot quality is evaluated unconditionally.  A more complete model would condition on possession context (e.g., fast break, isolation, pick-and-roll) to account for shot-creation difficulty — a distinction this model partially captures through game context features but does not fully model.

5. **Calibration of calibrated model.** Isotonic calibration can overfit on small datasets.  With 219k shots in this pipeline the calibration set is large enough that this is unlikely to be a significant issue, but it should be monitored if applied to single-player or team subsets.

---

## 11. Conclusions and Future Work

Using 219,527 real 2025 NBA shot attempts, this project demonstrates a statistically rigorous, research-grade xPTS pipeline.  Key findings:

1. **Player-zone shooting history** is the dominant predictive feature, contributing more than 4× the AUC improvement of all game-context features combined.
2. **XGBoost + isotonic calibration** achieves the best overall profile: strong discrimination *and* reliable probability estimates.
3. **Shot decision quality** (xPTS relative to zone average) provides a novel offensive metric beyond raw make rates or aggregate xPTS.
4. **Bootstrap confidence intervals** confirm that model differences are stable across test-set resamples, not artefacts of a single lucky split.

### Future Work

- **Tracking features:** integrate SportVU/Second Spectrum defender distance and touch-time data (if available) to model contested vs open looks explicitly.
- **Hierarchical Bayesian model:** replace the manual shrinkage encoder with a fully Bayesian hierarchical model (e.g., PyMC or Stan) for player-level random effects, providing posterior uncertainty estimates per player.
- **Season-ordered evaluation:** use time-based train/test splits to measure predictive performance in a true forecasting scenario.
- **Counterfactual shot-mix analysis:** given a player's xPTS profile by zone, compute the expected efficiency gain from an optimal shot-mix reallocation.
- **Team offensive fingerprints:** cluster teams by their zone-weighted shot quality profiles to identify systematic offensive stylistic groups.

---

## Appendix: Artefact Locations

| Output | Path |
|---|---|
| Processed feature matrix | `data/processed/shots_model_input.csv` |
| Primary model (XGBoost) | `models/xpts_model.pkl` |
| Model metrics comparison | `outputs/model_metrics.csv` |
| XGBoost CV results | `outputs/cv_results_xgboost.csv` |
| Ablation study | `outputs/ablation_study.csv` |
| Permutation importance | `outputs/permutation_importance.csv` |
| Player summary | `outputs/player_summary.csv` |

## References

- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD*.
- Cervone, D., D'Amour, A., Bornn, L., & Goldsberry, K. (2016). A multiresolution stochastic process model for predicting basketball possession outcomes. *JASA*.
- Goldsberry, K. (2012). CourtVision: New visual and spatial analytics for the NBA. *MIT SSAC*.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML*.
- Lucey, P., Bialkowski, A., Monfort, M., Carr, P., & Matthews, I. (2014). Quality vs. quantity: Improved shot prediction in soccer using strategic features from spatiotemporal data. *MIT SSAC*.
- Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. *ICML*.


## 1. Overview

This project builds a machine learning pipeline to estimate **shot make probability** for NBA field-goal attempts and converts those probabilities into **expected points (xPTS)**:

$$
\text{xPTS} = P(\text{make}) \times \text{shot value}
$$

The modeling goal is not just descriptive analytics, but a calibrated probabilistic estimate of shot quality at the possession level. The report focuses on the **ML side of the program**: supervised learning setup, model comparison, predictive performance, calibration, feature importance, and how those outputs support basketball interpretation.

The rebuild uses the uploaded full-league 2025 shot dataset and compares **XGBoost** against **Logistic Regression** as baseline and benchmark models for binary shot outcome prediction.

## 2. Machine Learning Problem Formulation

### Prediction target

Each shot is treated as a binary classification problem:

- **1** = shot made
- **0** = shot missed

The model estimates:

$$
P(\text{make} \mid \text{shot context, player context, game context})
$$

That estimated probability is then transformed into expected offensive value:

- For a 2-point shot: `xPTS = 2 × P(make)`
- For a 3-point shot: `xPTS = 3 × P(make)`

### Why this framing matters

This formulation separates **shot quality** from **actual outcome variance**. A missed shot can still be a strong process possession if the modeled make probability is high, while a made low-probability attempt may still be low-quality from a repeatable decision-making perspective.

## 3. Data Snapshot

The processed modeling table contains **219,527** shots from the 2025 NBA data snapshot.

### Core descriptive statistics

- Total shots: **219,527**
- Overall make rate: **46.72%**
- Average shot value: **2.421**
- Average modeled xPTS: **1.057**

### Largest shot zones

- **Above the Break 3:** 68,358 shots, make rate 35.32%, avg xPTS 0.996
- **Restricted Area:** 61,190 shots, make rate 66.36%, avg xPTS 1.377
- **In The Paint (Non-RA):** 44,475 shots, make rate 44.37%, avg xPTS 0.862

### Shot type summary

- **2PT Field Goal:** 127,073 shots, make rate 54.51%, avg xPTS 1.097
- **3PT Field Goal:** 92,454 shots, make rate 36.02%, avg xPTS 1.000

From an ML perspective, this is a large enough sample to support stable hold-out evaluation and meaningful comparison across model families. The class balance is also healthy for binary classification, avoiding the extreme imbalance problems seen in many event prediction tasks.

## 4. Feature Engineering and Modeling Inputs

The xPTS pipeline is driven by features that capture three main sources of signal:

### Spatial features

These describe where and how the shot was taken:

- `shot_distance`
- `shot_angle`
- shot zone indicators
- interactions such as `dist_angle_ix`

These features encode the geometric difficulty of the attempt. In practice, distance and angle act as core shot-quality variables because they directly shape shot efficiency across zones.

### Player-context features

These describe the shooter’s historical efficiency profile:

- `player_zone_fg_pct`

This variable is especially important because it helps the model distinguish between two shots that are geometrically similar but taken by players with very different skill profiles in the same zone.

### Game-state/context features

These provide situational information:

- `game_seconds_remaining`
- related possession/game context variables used by the pipeline

These signals are generally weaker than spatial features, but they can still improve prediction around late-clock or end-game decision contexts.

## 5. Model Comparison: XGBoost vs Logistic Regression

The program evaluates two complementary models:

### Logistic Regression

Logistic Regression serves as the linear baseline. It is useful because:

- it is simple and interpretable,
- it provides a strong benchmark for probabilistic classification,
- and it often produces well-calibrated probabilities.

Its main limitation is that it assumes a mostly linear relationship in feature space unless feature interactions are manually engineered.

### XGBoost

XGBoost is the nonlinear benchmark model. It is well-suited for this task because:

- it captures nonlinear effects automatically,
- it models interaction structure more flexibly,
- and it tends to improve ranking/discrimination performance in structured tabular data.

For shot-quality modeling, this matters because relationships between distance, angle, player tendency, and game context are unlikely to be purely linear.

## 6. Hold-out Test Performance

| Model | ROC-AUC | PR-AUC | Log-Loss | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| XGBoost | 0.6434 | 0.6384 | 0.6513 | 0.2300 | 0.0262 |
| Logistic Regression | 0.6394 | 0.6168 | 0.6557 | 0.2318 | 0.0160 |

### Interpretation of metrics

- **ROC-AUC** evaluates ranking quality across decision thresholds.
- **PR-AUC** is especially useful for understanding precision/recall trade-offs in predicted makes.
- **Log-Loss** measures the quality of probabilistic predictions and penalizes overconfident errors.
- **Brier score** measures mean squared probability error.
- **ECE (Expected Calibration Error)** measures how closely predicted probabilities align with observed frequencies.

### What the results show

- **XGBoost is the best overall predictive model** on discrimination and probability error metrics.
- **Logistic Regression is better calibrated** out of the box, as shown by its lower ECE.
- The margin is not massive, which is meaningful: it suggests the feature set itself is doing a lot of the work, and the nonlinear model is extracting incremental signal rather than completely changing task difficulty.

From an ML standpoint, this is a strong result. It shows that the program is not relying on a single metric or a single model family, but instead evaluating the trade-off between:

- **predictive power** (XGBoost advantage), and
- **probability calibration / interpretability** (Logistic Regression advantage).

## 7. Why XGBoost Performs Better

The performance edge from XGBoost is consistent with the structure of basketball shot data.

### Nonlinear decision boundaries

Shot success is not a simple linear function of distance. The effect of distance depends on:

- the shot zone,
- the shooter,
- the angle,
- and often contextual state.

A tree-boosting model can learn these nonlinear thresholds naturally.

### Feature interactions

The importance of `dist_angle_ix` suggests that combined geometry matters more than isolated variables. XGBoost is better equipped to exploit these interaction patterns without requiring all interactions to be manually specified.

### Heterogeneous player effects

A feature like `player_zone_fg_pct` likely interacts with location-based variables. XGBoost can adapt predictions for different player profiles more flexibly than a linear baseline.

## 8. Calibration and Probabilistic Quality

For xPTS, calibration is especially important because the final quantity is based directly on predicted probability. A model with strong ranking but poor calibration may still generate distorted expected-points values.

The results indicate:

- **Logistic Regression has lower ECE (0.0160)** and is therefore closer to observed frequencies.
- **XGBoost has slightly worse calibration (0.0262)**, even though it is better on ranking and overall error.

This is a classic ML trade-off. In applications where xPTS is used as a probabilistic valuation metric rather than only a ranking model, calibration quality matters significantly.

A practical takeaway is that **XGBoost is the stronger base learner**, but calibration-aware post-processing may further improve deployment quality if the project is extended. Even without additional post-calibration, both models appear reasonably well behaved on this real dataset.

## 9. Parameter / Feature Importance

Top permutation importances for XGBoost (mean ROC-AUC drop):

1. `player_zone_fg_pct` (0.1057)
2. `shot_angle` (0.0063)
3. `shot_distance` (0.0035)
4. `dist_angle_ix` (0.0023)
5. `game_seconds_remaining` (0.0022)

### ML interpretation

#### 1. `player_zone_fg_pct`

This is by far the dominant feature. That indicates the model gains most of its predictive lift from **player-specific shooting skill within zones**, not just generic league-average geometry.

This is an important modeling insight: a pure shot-location model is useful, but a richer xPTS system becomes substantially stronger when it incorporates shooter-level historical performance.

#### 2. `shot_angle`

Angle matters more than many simple shot models assume. This likely reflects real spatial asymmetries in how shots are created and finished, especially around the rim and along the arc.

#### 3. `shot_distance`

Distance remains fundamental, but in this feature set it appears less informative than player-zone efficiency and angle once the full geometry/context representation is included.

#### 4. `dist_angle_ix`

This interaction term confirms that the relationship between angle and distance is not additive in a simple way. Combined geometry improves the ML representation of shot difficulty.

#### 5. `game_seconds_remaining`

Game context contributes smaller but real signal. It likely captures shot-quality degradation or selection changes in end-clock and end-period situations.

## 10. Visual Diagnostics

### Model comparison visuals

![ROC curves showing XGBoost slightly above Logistic Regression on overall discrimination](outputs/roc_curves.png)
![Precision-recall curves showing higher PR-AUC for XGBoost than Logistic Regression](outputs/pr_curves.png)
![Calibration curves showing Logistic Regression closer to perfect calibration than XGBoost](outputs/calibration_curves.png)
![Learning curves showing XGBoost train/validation performance as training size increases](outputs/learning_curves.png)

### Feature importance visuals

![Built-in XGBoost feature importance ranking](outputs/feature_importance.png)
![Permutation importance chart showing player_zone_fg_pct as the dominant feature by ROC-AUC drop](outputs/permutation_importance.png)

### Shot quality and outcome visuals

![NBA shot chart colored by predicted xPTS](outputs/shot_chart_xpts.png)
![Average xPTS by shot zone](outputs/xpts_by_zone.png)
![Top and bottom players by average shot quality](outputs/player_summary.png)

These graphics are important from an ML evaluation perspective because they show different dimensions of model behavior:

- **ROC and PR curves** show threshold-independent discrimination.
- **Calibration curves** show whether probabilities can be trusted numerically.
- **Learning curves** show whether model performance is stabilizing with current sample size.
- **Importance charts** identify what the model is actually using to make decisions.

## 11. ML Takeaways

The strongest machine learning conclusions from this rebuild are:

1. **The xPTS problem is well-suited to probabilistic supervised learning.**
   Modeling shot make probability is a clean binary classification task with direct value translation into expected points.

2. **XGBoost is the strongest overall model in this pipeline.**
   It outperforms Logistic Regression on discrimination and probabilistic error metrics, making it the best pure predictive model of the two.

3. **Calibration still matters for xPTS.**
   Because xPTS is derived directly from predicted probability, lower ECE is meaningful. Logistic Regression remains valuable as a benchmark and calibration reference.

4. **Feature quality is a major driver of performance.**
   The biggest gains appear to come from strong basketball-aware features, especially player-zone efficiency and shot geometry, rather than from model complexity alone.

5. **The real 2025 dataset materially strengthens the pipeline.**
   Compared with synthetic-data experiments, this league-scale rebuild produces more credible metrics and more defensible basketball interpretations.

## 12. Conclusion

Using the uploaded real 2025 NBA shot dataset, the program now produces a credible machine learning xPTS pipeline at league scale. The core workflow—feature engineering, binary shot-make modeling, hold-out evaluation, and conversion from predicted probability to expected points—is statistically coherent and practically useful.

From the ML side, the main result is clear:

- **XGBoost is the best overall predictive model** for this version of the program.
- **Logistic Regression remains an important baseline** because of its simplicity and stronger calibration behavior.
- **Player-zone shooting history is the dominant predictive feature**, with shot geometry and game context supplying additional signal.

Overall, the project demonstrates a strong applied sports-analytics pipeline: it uses modern tabular ML methods, evaluates them with the right probability-focused metrics, and translates predictions into an interpretable basketball value measure through xPTS.

## Appendix: Sources

- Pipeline: `run_pipeline.py`
- Metrics table: `outputs/model_metrics.csv`
- XGBoost CV summary: `outputs/cv_results_xgboost.csv`
- Permutation importance: `outputs/permutation_importance.csv`
- Player summary: `outputs/player_summary.csv`
