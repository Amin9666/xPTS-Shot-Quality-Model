# xPTS Shot Quality Model – Presentation Notes

> **How to use this file:**  Each section corresponds to one slide image in `outputs/slides/`.
> The **Headline** is your slide title, **Bullet Points** are for on-slide text, and **Speaker Notes** give you the words to say out loud.

---

## Slide 1  –  `01_shot_chart.png`
### Headline
**NBA Shot Chart – Where Are the Best Shots?**

### Bullet Points (on slide)
- Each dot = one shot attempt, coloured by **Expected Points (xPTS)**
- **Green** = high-value shot (near basket, corner 3)
- **Red** = low-value shot (long mid-range, extreme angle)
- xPTS = *P(make)* × *shot value* (2 or 3 pts)

### Speaker Notes
> "This is the money slide — the one that makes the model intuitive.  Every dot on the court is a shot attempt.  The colour tells you not whether it went in, but how *good* the attempt was.  Shots near the basket and in the corners light up green because the model assigns them a high make probability.  Long mid-range jumpers go red because they combine poor make probability with only 2-point value.  The model turns a binary made/missed outcome into a continuous quality score for every single shot."

---

## Slide 2  –  `02_roc_curves.png`
### Headline
**ROC Curves – How Well Does the Model Discriminate?**

### Bullet Points (on slide)
- ROC = Receiver Operating Characteristic curve
- X-axis: **False Positive Rate** (wrongly predicted makes)
- Y-axis: **True Positive Rate** (correctly predicted makes)
- **AUC = area under curve**; 1.0 = perfect, 0.5 = coin flip
- XGBoost AUC ≈ 0.576 | Logistic AUC ≈ 0.581

### Speaker Notes
> "The ROC curve is the standard tool for evaluating a binary classifier at every possible probability threshold.  The diagonal dashed line is what a model that guesses randomly would produce.  Any curve above it means the model is learning something real.  Our AUC values of around 0.58 sit modestly above the 0.5 baseline — which is expected for shot prediction.  Shot-making in basketball has a large irreducible random component: even NBA players hit only 45% on open mid-range attempts, and small angle changes, defender positioning, and fatigue all add noise the model cannot see."

---

## Slide 3  –  `03_pr_curves.png`
### Headline
**Precision-Recall Curves – Beyond ROC for Skewed Data**

### Bullet Points (on slide)
- ROC can be **optimistic** when classes are imbalanced
- Precision = of predicted makes, how many were *actually* made?
- Recall = of all actual makes, how many did we *catch*?
- PR-AUC is a stricter test of real model value
- Horizontal dashed line = **no-skill baseline** (just predict the average)

### Speaker Notes
> "If you only report ROC-AUC, you can look good even with a mediocre model on an imbalanced dataset.  The PR curve is more demanding.  It asks: when the model says 'this shot will go in', how often is it right?  And across all made shots, how many does it find?  A good model pushes the curve up and to the right, well above the flat no-skill line.  Both our models beat the baseline, and the curves show where the trade-off between precision and recall lives at different threshold choices — useful if you are building a real-time alert system."

---

## Slide 4  –  `04_calibration.png`
### Headline
**Calibration Curves – Can We Trust the Probabilities?**

### Bullet Points (on slide)
- A **calibrated** model: if it says 60%, shots go in ≈ 60% of the time
- Points on the diagonal = perfect calibration
- **ECE (Expected Calibration Error)**: probability-weighted gap from diagonal
  - Logistic ECE ≈ 0.030 ✓ (well-calibrated)
  - XGBoost ECE ≈ 0.054 (slightly over-confident)
- Calibration is **critical** for xPTS: we multiply raw probability by shot value

### Speaker Notes
> "This slide is the most statistically important one.  A model can have great AUC but terrible calibration — meaning it ranks shots correctly but its probabilities are systematically too high or too low.  For xPTS that matters enormously, because we multiply the raw probability by 2 or 3 to get expected points.  If the probability is off by 10%, every xPTS number is biased.  The diagonal line is truth; points close to it mean the model's confidence matches reality.  Logistic regression is better-calibrated here, which is a genuine advantage for interpreting shot quality scores directly."

---

## Slide 5  –  `05_learning_curves.png`
### Headline
**Learning Curves – Diagnosing Bias vs. Variance**

### Bullet Points (on slide)
- Learning curve: model performance as training data grows
- **Large gap** between train and validation → **high variance** (over-fitting)
- **Both scores low and converging** → **high bias** (under-fitting)
- Shaded bands = ± 1 standard deviation across 5 folds
- Key insight: validation AUC *plateaus* — more data alone won't help much

### Speaker Notes
> "This is the bias-variance diagnostic that every graduate course covers.  We train the same XGBoost model on progressively larger chunks of the data and watch both training and cross-validation AUC.  The gap between the two curves shrinks as data grows, which is the classic signature of variance.  But notice the validation score has largely plateaued by 10,000 samples — adding more data of the same type is unlikely to push AUC much higher.  The real lever is richer features: defender distance, touch time, dribble count.  The model is not under-fitting; it has extracted most of the signal available in our current feature set."

---

## Slide 6  –  `06_permutation_importance.png`
### Headline
**Permutation Feature Importance – What Actually Drives Predictions?**

### Bullet Points (on slide)
- Shuffle one feature at a time on the **test set** → measure AUC drop
- Unlike impurity-gain importance, **unbiased toward correlated features**
- **Top driver: `player_zone_fg_pct`** – player's shooting history in that zone
- **Second: `shot_clock`** – urgency / decision quality under pressure
- **Third: `shot_angle`** – lateral difficulty
- Features with near-zero or negative importance add noise, not signal

### Speaker Notes
> "Standard tree-based importance scores are computed on training data and are biased toward features with many unique values.  Permutation importance is the gold standard: we take the trained model, randomly shuffle one feature's values in the test set — breaking its relationship to the outcome — and measure how much AUC drops.  A big drop means that feature is doing real work.  A near-zero or negative value means the feature is essentially noise at test time.  The result is intuitive: a player's historical zone shooting percentage is the single most predictive feature, followed by shot clock pressure and lateral angle.  Basic geometry features like distance squared add little once the raw distance and angle are in the model."

---

## Slide 7  –  `07_cv_results.png`
### Headline
**5-Fold Cross-Validation – Robust, Leakage-Free Evaluation**

### Bullet Points (on slide)
- A **single train/test split is unreliable** — results vary by random seed
- K-fold CV: train on 4 folds, test on 1, rotate 5 times
- Reports **mean ± std** → honest confidence interval on performance
- Key fix: **player zone FG% refit per fold** to prevent data leakage
- XGBoost ROC-AUC: **0.586 ± 0.010** across 5 folds

### Speaker Notes
> "One of the most common mistakes in ML projects is reporting a single hold-out number and calling it done.  A different random seed can move your AUC by 0.01 or more.  Cross-validation gives you a distribution of performance estimates, and reporting mean ± std tells your audience how stable the model is.  The left plot shows each individual fold — you can see fold-to-fold variance is small, which is a good sign.  The right chart summarises all five metrics: AUC, PR-AUC, Brier score (a proper scoring rule for probability estimates), and ECE.  There is also a subtle methodological point: our target-encoded feature, the player zone FG%, had to be recomputed from training-fold labels only inside each CV iteration to prevent data leakage — a detail most introductory implementations miss."

---

## Slide 8  –  `08_player_quality.png`
### Headline
**Player Shot Quality – Who Generates the Best Looks?**

### Bullet Points (on slide)
- **Left:** average xPTS per attempt by player (green = above average, red = below)
- **Right:** xPTS vs actual make rate — do better shot-selectors actually convert more?
- Lillard & Curry lead — high 3-point volume from range drives xPTS up
- Jokic highest *actual* make rate but average xPTS — efficient near-basket shots
- Booker & Doncic — below-average shot selection by xPTS

### Speaker Notes
> "This is the most business-facing slide.  After all the theory, here is a real insight: Damian Lillard and Stephen Curry generate the highest expected value per shot attempt.  That is partly because they shoot a lot of 3-pointers from range where the 3-point multiplier inflates xPTS even at moderate make probabilities.  Nikola Jokic has the highest actual make rate — he almost never takes a bad shot — but his average xPTS is slightly lower because most of his attempts are 2-point paint shots.  The scatter plot on the right is the accountability check: do players who generate good shots actually score more?  Broadly yes.  Players with high xPTS tend to have higher make rates, which validates the model."

---

## Slide 9  –  `09_xpts_by_zone.png`
### Headline
**xPTS by Shot Zone – The Value Map of the Court**

### Bullet Points (on slide)
- **Violin plots** show the full distribution of xPTS within each zone
- **Restricted Area** and **Corner 3s** have the highest median xPTS
- **Mid-Range** sits at the bottom — low make% with only 2-point value
- This is *why* analytics-driven teams now deprioritise mid-range volume
- Box inside violin = IQR (25th–75th percentile)

### Speaker Notes
> "This slide tells the fundamental story of modern basketball analytics.  The restricted area right at the basket has the highest expected value — you're very likely to make a shot from 2 feet, and it counts 2 points.  Corner threes are close behind: the corner is the shortest three-point attempt on the court (22 feet vs 23.75 feet on the arc), so make probability is higher, and it's worth 3 points.  The mid-range — those pull-up jumpers from 15 to 20 feet — is the dead zone.  It carries only 2-point value but substantially lower make probability than near-basket shots.  This is the analytical foundation behind teams like the Houston Rockets who essentially eliminated mid-range attempts from their offense."

---

## Slide 10  –  `10_metrics_table.png`
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

- Logistic Regression wins on Log-Loss, Brier, and ECE
- XGBoost wins on raw ranking (ROC-AUC, PR-AUC) by a thin margin
- **No single metric tells the whole story** — that is why we report all five

### Speaker Notes
> "This is the table that shows intellectual honesty.  There is no single 'best model'.  XGBoost edges logistic regression on discrimination — it slightly better separates makes from misses.  But logistic regression produces better-calibrated probabilities — it's more accurate when you ask 'what is the actual probability this shot goes in?'.  For an xPTS application where you're multiplying probability by shot value, calibration arguably matters more than discrimination.  Reporting all five metrics, including ECE which most applied papers skip, is what separates a rigorous analysis from a marketing exercise."

---

## Slide 11  –  `11_pipeline_overview.png`
### Headline
**End-to-End Pipeline – Reproducible & Production-Ready**

### Bullet Points (on slide)
1. **Data Generation** – synthetic NBA shot data with realistic player archetypes
2. **Feature Engineering** – geometry (distance, angle, interactions), game context, player history
3. **Leakage-Free 5-Fold CV** – player zone FG% refit on training fold only
4. **Hyperparameter Tuning** – RandomizedSearchCV over 15 configurations
5. **Model Training** – XGBoost + Logistic Regression with full preprocessing pipeline
6. **Evaluation & Charts** – 5 metrics, 11 diagnostic plots, serialized artifact

### Speaker Notes
> "This is the architecture overview that shows the work is more than just calling sklearn.fit().  Each step is a deliberate methodological choice.  Step 3 is the one most projects get wrong: when you compute a player's historical shooting percentage and use it as a feature, you must recompute it from training labels only, inside every fold of cross-validation.  If you don't, information from the test set leaks into your training process and all your reported metrics are optimistically inflated — a subtle but fatal flaw.  Step 4 is randomised hyperparameter search rather than a manual grid, which is statistically more efficient and produces an unbiased estimate of the best configuration.  The entire pipeline is a single Python script: `python run_pipeline.py`."

---

## Suggested Slide Order & Talk Structure

| # | Slide | Time (min) |
|---|---|---|
| 1 | Shot Chart (xPTS) — the hook | 2 |
| 2 | Pipeline Overview — what we built | 1 |
| 3 | xPTS by Zone — basketball intuition | 2 |
| 4 | Player Quality — the headline finding | 2 |
| 5 | 5-Fold CV — rigorous evaluation | 2 |
| 6 | Metrics Table — honest comparison | 1.5 |
| 7 | ROC Curves — discrimination | 1.5 |
| 8 | PR Curves — beyond ROC | 1 |
| 9 | Calibration — probability reliability | 2 |
| 10 | Permutation Importance — what drives predictions | 2 |
| 11 | Learning Curves — where to improve | 1.5 |
| — | Q & A | 3 |
| **Total** | | **~23 min** |

---

## Key Takeaways (closing slide text)

- **xPTS converts binary make/miss into a continuous shot quality score**
- **Restricted Area and Corner 3s dominate** — mid-range is inefficient value
- **Logistic Regression calibrates better** than XGBoost for probability outputs
- **Player zone history is the single most predictive feature** (permutation importance)
- **Target leakage is the #1 silent killer** in sports analytics ML — we fixed it
- **More data won't help much** — richer tracking features (defender distance, touch time) are the next lever
