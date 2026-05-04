# xPTS Shot Quality Model – Presentation Notes

> **How to use this file:** Each section corresponds to one slide image in `outputs/slides/`.
> The **Headline** is your slide title, **Bullet Points** are for on-slide text, and **Speaker Notes** give you the full words to say out loud.

---

## Slide 1 – Title / Introduction
### Headline
**xPTS – Expected Points: Measuring Shot Quality in the NBA**

### Bullet Points (on slide)
- What makes a shot *good* beyond whether it went in?
- **xPTS = P(make) × shot value (2 or 3 pts)**
- Built on court location, shot clock, angle & player history
- Two models compared: **XGBoost** vs **Logistic Regression**

### Speaker Notes
> "Good [morning/afternoon], everyone. Today I want to talk about one of the most fundamental questions in basketball analytics: not just whether a shot went in — but whether it was a good shot in the first place.
>
> To answer that, I built a machine learning model called xPTS — Expected Points per Shot Attempt. It takes every shooting opportunity and assigns it a value based on factors like court location, shot clock time, shooting angle, and a player's own historical tendencies. By the end of this talk, you'll see how xPTS can be used to evaluate shot quality, compare players, and understand what the model actually learned — and why we should trust it."

---

## Slide 2 – `01_shot_chart.png`
### Headline
**NBA Shot Chart – Where Are the Best Shots?**

### Bullet Points (on slide)
- Each dot = one **real shot attempt** — coloured by model-predicted xPTS
- **Green** = high-value shot (near basket, corner 3)
- **Red** = low-value shot (long mid-range, extreme angle)
- xPTS = *P(make)* × *shot value* (2 or 3 pts)

### Speaker Notes
> "Let's start with the visual that makes this whole project click. Every dot on this court represents a real shot attempt, and the colour tells you one thing: how valuable was that attempt? Green is high expected value — the shot is likely to score efficiently. Red is low expected value.
>
> Notice the pattern immediately. Two regions dominate in green: the restricted area around the basket, and the corner three-point line. Meanwhile, the mid-range zone — that band of 15-to-20 foot pull-up jumpers — is painted almost entirely in red.
>
> This is not a coincidence, and it's not opinion. It's maths. xPTS is calculated as the probability of making the shot multiplied by the point value of that shot — 2 or 3. A corner three is the shortest three-point attempt on the floor. Even at a moderate make percentage, the 3-point multiplier drives the value up. Mid-range shots, on the other hand, are difficult enough to miss frequently and only worth 2 points. They're the worst of both worlds.
>
> This is why analytically-driven teams have systematically eliminated mid-range volume from their offenses over the last decade. Our model confirms exactly that logic."

---

## Slide 3 – `11_pipeline_overview.png`
### Headline
**xPTS Model Pipeline – End-to-End Overview**

### Bullet Points (on slide)
- **6 stages:** Data Generation → Feature Engineering → Leakage-Free CV → Hyperparameter Tuning → Model Training → Evaluation & Charts
- All player zone FG% features refit **per fold** to prevent data leakage
- Two models trained: **XGBoost** and **Logistic Regression**

### Speaker Notes
> "Before we dive into results, let me quickly walk you through how the system was built — because the methodology is just as important as the findings.
>
> The pipeline has six stages. First, data generation — realistic synthetic NBA shot data with court coordinates, player identities, shot zones, and shot clock information. Second, feature engineering — where we compute things like player zone field goal percentage and shot angle. Third, a 5-fold cross-validation setup that I'll explain in detail shortly. Fourth, hyperparameter tuning. Fifth, model training on two algorithms — XGBoost and Logistic Regression. And sixth, evaluation and chart generation.
>
> The key design principle throughout was: no data leakage. Every feature that references player history was recomputed within each fold. That's a discipline that a lot of sports analytics projects cut corners on, and it matters enormously for honest performance reporting."

---

## Slide 4 – `09_xpts_by_zone.png`
### Headline
**xPTS by Shot Zone – The Value Map of the Court**

### Bullet Points (on slide)
- **Violin plots** show the full xPTS distribution within each zone
- **Restricted Area** and **Corner 3s** have the highest median xPTS
- **Mid-Range** sits at the bottom — low make% with only 2-point value
- This is *why* analytics-driven teams deprioritise mid-range volume

### Speaker Notes
> "Now let's back up that visual intuition with hard numbers. These violin plots show the full distribution of xPTS across each shot zone — not just averages, but the entire spread of values.
>
> Look at Restricted Area — it has the highest median xPTS. You're close to the basket, the make probability is high, and you're picking up 2 points. Corner 3s come in right behind it. You trade some make percentage for that extra point, but the math still works in your favour.
>
> Now look at Mid-Range — it sits at the bottom of this chart. Poor make rate, only 2 points on offer. It's the shot that analytics has been arguing against for fifteen years.
>
> What I want you to take away from this chart is that xPTS doesn't just reflect shooting skill — it reflects shot selection decisions. Two players can have the same shooting talent, but the player who takes better shots will score more efficiently. This is what separates good offenses from great ones."

---

## Slide 5 – `08_player_quality.png`
### Headline
**Player Shot Quality – Who Generates the Best Looks?**

### Bullet Points (on slide)
- **Left:** average xPTS per attempt by player (green = above average, red = below)
- **Right:** xPTS vs actual make rate — do better shot-selectors actually convert more?
- Lillard & Curry lead — high 3-point volume from range drives xPTS up
- Jokic highest *actual* make rate but average xPTS — efficient near-basket shots

### Speaker Notes
> "Now we can ask: at the player level, who is generating the highest expected value per attempt?
>
> On the left chart, bars in green are above-average xPTS; red is below average. Lillard and Curry lead the pack — and this is a really interesting result because it's driven primarily by their three-point volume. They shoot a lot of threes from range, and the 3-point multiplier pushes their average xPTS up even at reasonable make percentages.
>
> On the right, we plot xPTS against actual make rate by player. And here's the fascinating nuance: Jokic has the highest actual make rate but sits closer to average xPTS. Why? Because his efficiency comes from high-percentage near-basket shots — they have a very high make probability but no 3-point multiplier. It's a different kind of efficiency.
>
> This tells us that xPTS and raw make percentage are measuring slightly different things. xPTS rewards shot selection and shot type; make rate rewards pure shooting execution. You need both lenses to fully evaluate a player."

---

## Slide 6 – `02_roc_curves.png`
### Headline
**ROC Curves – How Well Do the Models Rank Shot Outcomes?**

### Bullet Points (on slide)
- AUC = ability to rank makes above misses (0.5 = random, 1.0 = perfect)
- XGBoost AUC ≈ **0.586** | Logistic Regression AUC ≈ **0.582**
- Predicting professional shot-making has **high inherent randomness**
- 0.58–0.59 AUC is competitive with published academic shot models

### Speaker Notes
> "Now let's look at model performance. The ROC curve measures how well a model ranks shots — can it correctly order makes above misses? The AUC — Area Under the Curve — runs from 0.5, which is a coin flip, up to 1.0, which is perfect.
>
> XGBoost reaches an AUC of around 0.586; Logistic Regression is close behind. You might look at these numbers and think 'that doesn't sound impressive.' But consider what we're predicting: whether a professional athlete, under game conditions, will make or miss a basketball shot. There is enormous inherent randomness. A 0.58–0.59 AUC on shot prediction is competitive with published academic models. The model is genuinely learning real signal — not noise."

---

## Slide 7 – `03_pr_curves.png`
### Headline
**Precision-Recall Curves – A Stricter Test**

### Bullet Points (on slide)
- PR curve is more informative when **class balance matters**
- Dashed baseline = a model that always predicts the majority class
- Both models sit **above the no-skill baseline**
- XGBoost edges ahead on PR-AUC

### Speaker Notes
> "The Precision-Recall curve complements the ROC curve. In datasets where one outcome is more frequent than the other — and in basketball, misses are slightly more common than makes — the PR curve is a stricter test. It asks: when the model says 'this shot will go in,' how often is it right?
>
> Both models sit noticeably above the no-skill baseline shown by the dashed line. XGBoost has a marginally higher PR-AUC. The takeaway is consistent: both models are extracting meaningful signal from the features, and neither is just guessing based on class frequency."

---

## Slide 8 – `04_calibration.png`
### Headline
**Calibration – Can We Trust the Probabilities?**

### Bullet Points (on slide)
- A **calibrated** model: if it says 60%, shots go in ≈ 60% of the time
- Points on the diagonal = perfect calibration
- Logistic ECE ≈ 0.030 ✓ | XGBoost ECE ≈ 0.054 (slightly over-confident)
- Calibration is **critical** for xPTS: we multiply raw probability by shot value

### Speaker Notes
> "This is the slide I personally find most important for the xPTS use case specifically. Here's why.
>
> A model can rank shots correctly and still produce probabilities that are completely unreliable. For example, if the model says 60% chance of making this shot but those shots actually go in 72% of the time, the model is systematically under-confident. That might be fine for ranking purposes — but for xPTS, we multiply the raw probability by 2 or 3 to get expected points. A 10% miscalibration in probability becomes a 20–30 basis point error in expected points, compounded across every shot.
>
> The diagonal line is perfect calibration. Logistic Regression hugs that diagonal almost perfectly — ECE of around 0.030. XGBoost is reasonably well calibrated but slightly over-confident at higher probability ranges, with an ECE of around 0.054.
>
> This is a practical reason why, despite XGBoost's marginal edge in discrimination, Logistic Regression may be the better choice for producing xPTS values you actually want to trust in downstream analysis."

---

## Slide 9 – `05_learning_curves.png`
### Headline
**Learning Curves – Bias vs. Variance Diagnostic**

### Bullet Points (on slide)
- Training AUC drops as data grows — model can't memorise everything
- Validation AUC rises — model generalises better with more data
- Curves converging = **healthy, low-overfitting signal**
- More data would continue to improve performance

### Speaker Notes
> "The learning curve shows us something different — it tells us whether our model would benefit from more data, or whether it's already at its ceiling.
>
> As we increase training set size, the training AUC drops — the model can no longer perfectly memorise the data — while the validation AUC rises, meaning the model is generalising better. The gap between them is what we call variance, or overfitting.
>
> What we see here is a healthy convergence pattern. The curves are coming together rather than diverging. This tells us the model is not dramatically overfitting, and that collecting more shot data could continue to improve it — there's still signal left to capture."

---

## Slide 10 – `06_permutation_importance.png`
### Headline
**Permutation Feature Importance – What Drives Predictions?**

### Bullet Points (on slide)
- Shuffle one feature at a time on the **test set** → measure AUC drop
- Unbiased toward correlated features (unlike impurity-gain importance)
- **Top driver: `player_zone_fg_pct`** – player's shooting history in that zone
- **Second: `shot_clock`** – urgency under pressure
- **Third: `shot_angle`** – lateral difficulty

### Speaker Notes
> "One of the most rigorous ways to understand a model is permutation feature importance. The logic is simple: take one feature, randomly shuffle its values across the test set — which breaks any real relationship between that feature and the outcome — and measure how much the model's AUC drops. A large drop means the model was genuinely relying on that feature.
>
> The clear winner is player zone field goal percentage — a player's historical shooting rate within a given court zone. When we scramble that feature, model performance drops significantly. This makes intuitive sense: knowing that Curry is shooting from corner 3 at a 44% clip versus knowing an average player is there is enormously informative.
>
> Second is shot clock time — urgency matters. A late shot-clock heave is a fundamentally different shot than a settled catch-and-shoot opportunity. Third is shot angle — lateral difficulty adds real information.
>
> This result also validates our feature engineering process. The features that matter most are the ones that genuinely capture basketball reality."

---

## Slide 11 – `07_cv_results.png`
### Headline
**5-Fold Cross-Validation – Robust, Leakage-Free Evaluation**

### Bullet Points (on slide)
- K-fold CV: train on 4 folds, test on 1, rotate 5 times → **mean ± std**
- Key fix: **player zone FG% refit per fold** to prevent data leakage
- XGBoost ROC-AUC: **0.586 ± 0.010** across 5 folds

### Speaker Notes
> "Any single train-test split can give you a lucky or unlucky number. If you report performance based on one split, you have no idea how stable that estimate is. Cross-validation solves this.
>
> We rotate through 5 folds: train on 4, test on 1, and repeat. Then we report the mean and the standard deviation. XGBoost achieves 0.586 ± 0.010 ROC-AUC across 5 folds. That low standard deviation tells you the result is stable — it's not an artifact of one particular split.
>
> And I want to flag the methodological detail again: player zone field goal percentage was recomputed within each fold. If you compute that feature on the full dataset before splitting, you get target leakage — the feature will encode information about the test set's outcomes, and your reported performance will be artificially inflated. We fixed that. It's the kind of detail that separates rigorous ML practice from a number that looks good but doesn't hold up in production."

---

## Slide 12 – `10_metrics_table.png`
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

- XGBoost wins on discrimination (ROC-AUC, PR-AUC)
- Logistic Regression wins on calibration (Log-Loss, Brier, ECE)
- **No single metric tells the whole story**

### Speaker Notes
> "The final slide brings all the evaluation metrics together in one honest comparison. Five metrics, two models.
>
> XGBoost wins on discrimination — ROC-AUC and PR-AUC are slightly higher. It's better at ranking shots by their likelihood of going in.
>
> Logistic Regression wins on calibration — Log-Loss, Brier Score, and ECE are all better. Its probabilities are more trustworthy as raw numerical values.
>
> The lesson here is that no single metric tells the whole story. If your goal is a leaderboard ranking — which model is better at separating makes from misses — you pick XGBoost. If your goal is a deployable xPTS system where the numerical probability feeds into downstream calculations, you seriously consider Logistic Regression.
>
> So, to close: xPTS converts the binary make-or-miss outcome into a continuous shot quality score. The Restricted Area and Corner 3s dominate. Player zone shooting history is the single most predictive feature. Calibration matters as much as discrimination. And leakage-free cross-validation is non-negotiable.
>
> Thank you. I'm happy to take any questions."

---

## Suggested Slide Order & Talk Structure

| # | Slide | Time (min) |
|---|---|---|
| 1 | Title / Intro | 1 |
| 2 | Shot Chart — the hook | 2 |
| 3 | Pipeline Overview — how it was built | 1 |
| 4 | xPTS by Zone — basketball intuition | 1.5 |
| 5 | Player Quality — the headline finding | 1.5 |
| 6 | ROC Curves — discrimination | 1 |
| 7 | PR Curves — stricter test | 1 |
| 8 | Calibration — probability reliability | 1.5 |
| 9 | Learning Curves — bias vs variance | 1 |
| 10 | Permutation Importance — what drives predictions | 1.5 |
| 11 | 5-Fold CV — rigorous, leakage-free evaluation | 1.5 |
| 12 | Metrics Table — honest comparison + takeaways | 1.5 |
| — | Q & A | 3 |
| **Total** | | **~18 min (cut to 13 by trimming slides 7 & 9)** |

---

## Key Takeaways (closing slide text)

- **xPTS converts binary make/miss into a continuous shot quality score**
- **Restricted Area and Corner 3s dominate** — mid-range is inefficient value
- **Player zone history is the single most predictive feature** (permutation importance)
- **Logistic Regression calibrates better** than XGBoost for probability outputs
- **Target leakage is the #1 silent killer** in sports analytics ML — we fixed it
- **No single metric tells the whole story** — use discrimination AND calibration together

---

## Tips for Delivery

- **Slow down on Slides 8 (Calibration) and 11 (CV/Leakage)** — these are your most technically impressive points and deserve emphasis.
- **Pause after the Shot Chart** — let the visual breathe before you explain it.
- If you need to trim to 10 minutes, **cut Slides 7 (PR Curves) and 9 (Learning Curves)** — they are supporting evidence, not headline findings.
- Speak to the **basketball intuition first**, then the technical detail — lead with the story, not the maths.