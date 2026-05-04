# xPTS Shot Quality Model – Presentation Notes

> **How to use this file:**  Each section corresponds to one slide image in `outputs/slides/`.
> The **Headline** is your slide title, **Bullet Points** are for on-slide text, and **Speaker Notes** give you the words to say out loud.

---

## Slide 1  –  `01_shot_chart.png`
### Headline
**NBA Shot Chart – Where Are the Best Shots?**

### Bullet Points (on slide)
- Each dot = one **real shot attempt** — coloured by model-predicted xPTS
- **Green** = high-value shot (near basket, corner 3)
- **Red** = low-value shot (long mid-range, extreme angle)
- xPTS = *P(make)* × *shot value* (2 or 3 pts)

### Speaker Notes
> "Every dot on the court is a shot attempt. The colour tells you not whether it went in, but how *good* the attempt was. Shots near the basket and in the corners light up green; long mid-range jumpers go red because they combine poor make probability with only 2-point value. The model converts a binary made/missed outcome into a continuous quality score for every shot."

---

## Slide 2  –  `09_xpts_by_zone.png`
### Headline
**xPTS by Shot Zone – The Value Map of the Court**

### Bullet Points (on slide)
- **Violin plots** show the full xPTS distribution within each zone
- **Restricted Area** and **Corner 3s** have the highest median xPTS
- **Mid-Range** sits at the bottom — low make% with only 2-point value
- This is *why* analytics-driven teams deprioritise mid-range volume

### Speaker Notes
> "The restricted area has the highest expected value — you're likely to make it, and it's worth 2 points. Corner threes are close behind: it's the shortest 3-point attempt on the court. The mid-range is the dead zone — 2-point value with substantially lower make probability. This is the analytical foundation behind teams that have eliminated mid-range attempts from their offense."

---

## Slide 3  –  `08_player_quality.png`
### Headline
**Player Shot Quality – Who Generates the Best Looks?**

### Bullet Points (on slide)
- **Left:** average xPTS per attempt by player (green = above average, red = below)
- **Right:** xPTS vs actual make rate — do better shot-selectors actually convert more?
- Lillard & Curry lead — high 3-point volume from range drives xPTS up
- Jokic highest *actual* make rate but average xPTS — efficient near-basket shots

### Speaker Notes
> "Lillard and Curry generate the highest expected value per attempt — largely because the 3-point multiplier inflates xPTS even at moderate make probabilities. Jokic has the highest actual make rate but slightly lower xPTS because he shoots mostly 2-point paint shots. The scatter plot is the validation check: players who generate better shots broadly score more."

---

## Slide 4  –  `06_permutation_importance.png`
### Headline
**Permutation Feature Importance – What Drives Predictions?**

### Bullet Points (on slide)
- Shuffle one feature at a time on the **test set** → measure AUC drop
- Unbiased toward correlated features (unlike impurity-gain importance)
- **Top driver: `player_zone_fg_pct`** – player's shooting history in that zone
- **Second: `shot_clock`** – urgency under pressure
- **Third: `shot_angle`** – lateral difficulty

### Speaker Notes
> "We shuffle one feature's values in the test set, breaking its relationship to the outcome, and measure how much AUC drops. A big drop means the feature is doing real work. Player historical zone shooting percentage is by far the most predictive feature, followed by shot-clock pressure and angle. Basic geometry like distance-squared adds little once raw distance and angle are already in the model."

---

## Slide 5  –  `04_calibration.png`
### Headline
**Calibration – Can We Trust the Probabilities?**

### Bullet Points (on slide)
- A **calibrated** model: if it says 60%, shots go in ≈ 60% of the time
- Points on the diagonal = perfect calibration
- Logistic ECE ≈ 0.030 ✓ | XGBoost ECE ≈ 0.054 (slightly over-confident)
- Calibration is **critical** for xPTS: we multiply raw probability by shot value

### Speaker Notes
> "A model can rank shots correctly but still have biased probabilities. For xPTS that matters enormously — we multiply probability by 2 or 3 to get expected points, so a 10% probability error biases every xPTS output. Logistic regression is better-calibrated here, which is a genuine advantage when interpreting shot quality scores directly."

---

## Slide 6  –  `07_cv_results.png`
### Headline
**5-Fold Cross-Validation – Robust, Leakage-Free Evaluation**

### Bullet Points (on slide)
- K-fold CV: train on 4 folds, test on 1, rotate 5 times → **mean ± std**
- Key fix: **player zone FG% refit per fold** to prevent data leakage
- XGBoost ROC-AUC: **0.586 ± 0.010** across 5 folds

### Speaker Notes
> "A single train/test split number can move by 0.01+ with a different random seed. CV gives you a distribution of performance estimates. The subtle methodological point: player zone FG% had to be recomputed from training-fold labels only inside each CV iteration — most implementations skip this and end up with inflated metrics due to data leakage."

---

## Slide 7  –  `10_metrics_table.png`
### Headline
**Model Comparison – Five Rigorous Evaluation Metrics**

### Bullet Points (on slide)
| Metric | Measures | Ideal |
|---|---|---|
| **ROC-AUC** | Ranking / discrimination | ↑ closer to 1.0 |
| **PR-AUC** | Precision-recall trade-off | ↑ closer to 1.0 |
| **Log-Loss** | Probability sharpness | ↓ closer to 0 |
| **Brier Score** | Probability accuracy (MSE) | ↓ closer to 0 |
| **ECE** | Calibration reliability | ↓ closer to 0 |

- XGBoost wins on discrimination (ROC-AUC, PR-AUC) by a thin margin
- Logistic Regression wins on calibration (Log-Loss, Brier, ECE)
- **No single metric tells the whole story**

### Speaker Notes
> "There is no single 'best model'. XGBoost slightly better separates makes from misses; logistic regression produces more accurate probabilities. For xPTS — where you multiply probability by shot value — calibration arguably matters more than discrimination. Reporting all five metrics, including ECE which most papers skip, is what separates a rigorous analysis from a marketing exercise."

---

## Suggested Slide Order & Talk Structure

| # | Slide | Time (min) |
|---|---|---|
| 1 | Shot Chart — the hook | 2 |
| 2 | xPTS by Zone — basketball intuition | 2 |
| 3 | Player Quality — the headline finding | 2 |
| 4 | Permutation Importance — what drives predictions | 1.5 |
| 5 | Calibration — probability reliability | 1.5 |
| 6 | 5-Fold CV — rigorous, leakage-free evaluation | 1.5 |
| 7 | Metrics Table — honest comparison | 1.5 |
| — | Q & A | 3 |
| **Total** | | **~15 min** |

---

## Key Takeaways (closing slide text)

- **xPTS converts binary make/miss into a continuous shot quality score**
- **Restricted Area and Corner 3s dominate** — mid-range is inefficient value
- **Player zone history is the single most predictive feature** (permutation importance)
- **Logistic Regression calibrates better** than XGBoost for probability outputs
- **Target leakage is the #1 silent killer** in sports analytics ML — we fixed it
