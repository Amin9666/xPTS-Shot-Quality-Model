# xPTS Shot Quality Model Report (Real 2025 NBA Data)

**Author:** Amin9666  
**Project:** Expected Points (xPTS) for NBA shot quality  
**Data snapshot used in this report:** `NBA_2025_Shots.csv.zip` → `data/processed/shots_model_input.csv` (**219,527** shots)

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
