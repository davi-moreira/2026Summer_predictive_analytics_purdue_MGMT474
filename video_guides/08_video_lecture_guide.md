# Video Lecture Guide: Notebook 08 — Resampling and CV - How to Compare Models Without Fooling Yourself

## At a Glance

This guide covers:

1. **Why NB08 exists** — It teaches students to obtain reliable, low-variance performance estimates through cross-validation, replacing the fragile single train/val split with a systematic evaluation framework

2. **Why it must follow NB07** — NB07 provides the metric vocabulary (ROC-AUC, accuracy, F1) that NB08 passes as the `scoring` parameter to `cross_val_score`. Without NB07, students would not know which metric to choose or why

3. **Why it must precede NB09** — NB09 uses `GridSearchCV` and `RandomizedSearchCV`, which embed cross-validation inside the hyperparameter search loop. Students must understand standalone CV before they can understand how it works inside grid search

4. **Why each library/tool is used:**
   - `KFold` — the basic k-fold splitter for regression, demonstrates fold-to-fold variance
   - `StratifiedKFold` — the classification-aware splitter that preserves class proportions
   - `cross_val_score` and `cross_validate` — convenience functions that handle the train-evaluate loop
   - `RepeatedStratifiedKFold` — repeated CV for more stable estimates when precision matters
   - `RandomForestClassifier` — first appearance as a comparison model, previewing Week 3

5. **Common student questions** with answers (e.g., how many folds should I use?, does CV replace the test set?)

6. **Connection to the full course arc** — how CV becomes the inner loop of grid search in NB09, the evaluation backbone of the midterm in NB10, and the standard evaluation protocol for the final project

7. **Suggested video structure** — both single-video (~12 min) and three-video (~4-5 min each) formats with speaking prompts, timestamps, and notebook cell references

---

## Purpose of This Document

This guide is your **lecture companion** for recording the video(s) that accompany `08_cross_validation_model_comparison.ipynb`. It explains the *why* behind every design choice in the notebook — why k-fold CV is introduced before grid search, why stratified CV is essential for classification, and why the reusable comparison function is the most project-relevant artifact. Use it as speaking notes, not as a script: the goal is to internalize the reasoning so you can explain it naturally on camera.

---

## 1. Why This Notebook Exists

Notebooks 06 and 07 evaluate models on a single validation set. This is fast and intuitive, but it has a fundamental flaw: the performance estimate depends on *which* samples happen to land in the validation set. A lucky split might inflate the score; an unlucky split might deflate it. With only 114 validation samples (from the breast cancer dataset), even a few atypical samples can shift accuracy by 2-3 percentage points.

Notebook 08 solves this problem by introducing **cross-validation**: instead of one split, the data is partitioned into k folds, and the model is trained and evaluated k times, each time using a different fold as the validation set. The resulting k scores reveal both the central tendency (mean) and the uncertainty (standard deviation) of the performance estimate.

This notebook answers the question: **"How do I know if my model's performance is real, or just an artifact of the data split?"** The answer is: use cross-validation to get multiple independent estimates, and report both the mean and the standard deviation.

---

## 2. Why It Must Come After Notebook 07

Notebook 07 establishes the metric vocabulary that notebook 08 *consumes*:

| NB07 Concept | How NB08 Uses It |
|---|---|
| **ROC-AUC as a classification metric** | NB08 passes `scoring='roc_auc'` to `cross_val_score` and `cross_validate`. Students must know what ROC-AUC measures to interpret the results. |
| **Accuracy vs. precision vs. recall vs. F1** | NB08's model comparison table includes multiple metrics. Students must understand what each one means to choose a primary metric and interpret secondary metrics. |
| **The accuracy paradox with imbalanced data** | NB08 uses `StratifiedKFold` specifically to avoid folds with distorted class ratios. Students who understand the accuracy trap from NB07 immediately see why stratification matters. |
| **The metrics dashboard pattern** | NB08 builds a `cv_report()` function that reports train/val mean and std, overfitting gap, and fit time. This is the CV analogue of NB07's metrics dashboard. |

**In short:** NB07 answers "what to measure" and NB08 answers "how to measure it reliably."

If you tried to teach CV *before* metrics, students would pass `scoring='accuracy'` everywhere without understanding its limitations. If you skipped CV entirely, students would compare models on a single split — which, as NB08 demonstrates, can produce misleading rankings.

---

## 3. Why It Must Come Before Notebook 09

Notebook 09 introduces **GridSearchCV** and **RandomizedSearchCV** — tools that embed cross-validation inside the hyperparameter search loop. Their API looks like `GridSearchCV(pipeline, param_grid, cv=StratifiedKFold(5), scoring='roc_auc')`. To use these tools correctly, students must already understand:

- What `KFold` and `StratifiedKFold` do (they become the `cv` parameter)
- What `scoring='roc_auc'` means (they choose which metric guides the search)
- Why the same CV folds must be used for all parameter combinations (fairness)
- How to read `cv_results_` with mean and std (they evaluate grid search output)

NB08 builds each of these skills explicitly. Without it, GridSearchCV would be a black box — students would run it, get `best_params_`, and have no idea whether the search was conducted fairly or whether the results are stable.

The deliberate sequence is:

```
NB07: What metrics exist and what do they mean? (Metric vocabulary)
            |
NB08: How do we estimate metrics reliably? (Cross-validation)
            |
NB09: How do we search for the best hyperparameters? (Grid search with CV inside)
```

Each notebook adds exactly one conceptual layer.

---

## 4. Why These Specific Libraries and Tools

### 4.1 `sklearn.model_selection.KFold`

**What it does:** Splits data into k non-overlapping folds. With `shuffle=True`, samples are randomly shuffled before splitting.

**Why we use it:**

1. **Demonstrates the basic CV mechanism.** Before students use convenience functions like `cross_val_score`, they need to understand what is happening under the hood: the data is partitioned, and the model is trained and evaluated k times.

2. **Shows fold-to-fold variance.** The bar chart of 5 R-squared scores makes the variance visible. Students see that a single-split estimate could land anywhere in the range, while the mean of all 5 is more stable.

3. **Regression use case.** `KFold` is used for the California Housing regression example because class proportions are not a concern with continuous targets. This contrasts with `StratifiedKFold` used for classification.

**What to emphasize on camera:** The `shuffle=True` and `random_state=RANDOM_SEED` parameters ensure reproducibility. Without shuffling, the folds might contain systematic patterns (e.g., all high-value houses in fold 5 because the data was sorted by price). Shuffling randomizes the assignment.

### 4.2 `sklearn.model_selection.StratifiedKFold`

**What it does:** Same as `KFold`, but ensures each fold has approximately the same class distribution as the full dataset.

**Why we use it:**

- **Prevents class-imbalance artifacts in CV.** Without stratification, a fold in the breast cancer dataset might end up with 50% malignant instead of the dataset-wide 37%. This distorts both the training (model sees a different class balance) and the evaluation (metrics are computed on a non-representative sample).
- **Reduces variance in estimates.** The notebook shows side-by-side that stratified CV typically has lower standard deviation than regular CV, because folds are more similar to each other.
- **Required for classification.** This is the CV splitter students should use for every classification task in the course and in their projects.

**What to emphasize on camera:** For classification, always use `StratifiedKFold`. It is a free improvement — same computational cost as regular `KFold`, but lower-variance estimates. Think of it as insurance against unlucky fold assignments.

### 4.3 `sklearn.model_selection.cross_val_score` and `cross_validate`

**What they do:** `cross_val_score` runs k-fold CV and returns an array of scores. `cross_validate` extends this by also returning train scores, fit times, and score times.

**Why we use them:**

- **Convenience.** Instead of writing a manual loop (split data, fit model, evaluate, repeat), students call one function. This reduces boilerplate and lets them focus on interpretation.
- **Fairness guarantee.** Both functions automatically use the same folds for all evaluations. This ensures apples-to-apples comparison when comparing models.
- **`cross_validate` adds diagnostic information.** Train scores reveal overfitting (large train-val gap), fit times reveal computational cost, and the combination helps students make informed model choices.

**What to emphasize on camera:** `cross_val_score` is for quick checks (one metric, one model). `cross_validate` is for serious evaluation (multiple metrics, train scores, timing). Use `cross_validate` when building comparison tables.

### 4.4 `sklearn.model_selection.RepeatedStratifiedKFold`

**What it does:** Repeats stratified k-fold CV multiple times with different random shuffles. For example, 5-fold CV repeated 3 times produces 15 evaluation scores.

**Why we use it:**

- **More stable estimates.** With only 5 fold scores, the standard deviation itself is noisy. With 15 scores, the histogram is smoother and the confidence interval is tighter.
- **Demonstrates the bias-variance tradeoff in estimation.** More evaluations cost more compute but produce more reliable results. Students must learn when the extra precision is worth the cost.

**What to emphasize on camera:** Use repeated CV for final reports or publications where you need high confidence in the numbers. Use single-fold CV during early exploration when speed matters more.

### 4.5 `sklearn.ensemble.RandomForestClassifier`

**What it does:** An ensemble of decision trees that combines their predictions via majority voting (classification) or averaging (regression).

**Why we use it here:**

- **First exposure to an ensemble model.** NB08 is the first notebook where Random Forest appears. By including it in the model comparison table alongside logistic regression, students see that different algorithm families can be compared fairly using the same CV protocol.
- **Preview of Week 3.** Trees and ensembles are the focus of Week 3 (NB11-NB15). Introducing Random Forest briefly in NB08 creates anticipation and shows students that the CV comparison framework scales to any model.
- **Fit time contrast.** Random Forest takes noticeably longer than logistic regression, which introduces the practical consideration of computational cost in model selection.

**What to emphasize on camera:** Do not explain how Random Forest works yet — that is NB11's job. Here, treat it as a "different model" that plugs into the same pipeline and gets compared using the same CV folds. The point is the comparison *framework*, not the model itself.

---

## 5. Key Concepts to Explain on Camera

### 5.1 Why Single Splits Are Fragile

With 114 validation samples (breast cancer), moving just 3-4 samples between the training and validation sets can shift accuracy by 2-3%. This instability means that a model comparison based on a single split might select the wrong model purely due to luck. Cross-validation averages over multiple splits, producing a stable estimate that reflects the model's true generalization ability.

**Analogy for students:** Imagine judging a basketball player's ability by watching one free throw. They might make it or miss it, and you would have no idea whether they are a 90% or 50% shooter. Watching 50 free throws (cross-validation) gives you a reliable estimate. The more throws you watch, the more confident you can be.

### 5.2 Stratification — Why Class Balance Matters in Every Fold

If the breast cancer dataset is 37% malignant / 63% benign overall, but one CV fold ends up 50% malignant / 50% benign, two problems arise: (1) the model trained on that fold sees a different class balance than the model in production will encounter, and (2) the evaluation on that fold is computed on a non-representative sample. Stratified CV fixes both problems by enforcing the 37/63 ratio in every fold.

**When it matters most:** Small datasets and high imbalance. With 100,000 samples and 50/50 balance, random KFold works fine. With 569 samples and 37/63 balance, stratification makes a meaningful difference.

### 5.3 Fair Comparison Requires Identical Conditions

The golden rule of model comparison: **same folds, same metric, same data.** If you compare Model A on folds {1,2,3,4,5} and Model B on folds {6,7,8,9,10}, the comparison is invalid because the folds might have different difficulty levels. By passing the same `StratifiedKFold` object to both models, every model sees the exact same training and validation data at each iteration.

### 5.4 Reporting Uncertainty — Mean AND Standard Deviation

A performance report that says "ROC-AUC = 0.99" is incomplete. A report that says "ROC-AUC = 0.99 +/- 0.01" is informative. The standard deviation tells you whether the estimate is stable (small std) or noisy (large std). If two models have means within one standard deviation of each other, the difference is likely not meaningful — and you should prefer the simpler model.

---

## 6. What Students Should Take Away

After watching your video and completing this notebook, students should be able to answer these questions confidently:

1. **"Why is cross-validation better than a single train/val split?"** — Because a single split produces one score that depends on which samples ended up in the validation set. CV produces k scores whose mean is a more stable estimate and whose standard deviation quantifies the uncertainty.

2. **"When should I use StratifiedKFold instead of KFold?"** — Always for classification. Stratified CV preserves the class distribution in every fold, preventing fold-to-fold variance caused by random class-proportion fluctuations.

3. **"How many folds should I use?"** — 5 or 10 is standard. More folds mean each training set is larger (less bias) but the folds are more correlated (more variance in the variance estimate). For small datasets, 5-fold is typical. For very small datasets (<100 samples), use leave-one-out (k = n).

4. **"Does cross-validation replace the test set?"** — No. CV replaces the *validation* set. The test set remains locked and is used only for the final evaluation of the chosen model. CV is for model selection and development; the test set is for the final, unbiased performance report.

5. **"How do I know if one model is meaningfully better than another?"** — Compare their mean CV scores relative to their standard deviations. If Model A scores 0.98 +/- 0.01 and Model B scores 0.97 +/- 0.02, the overlap is substantial and the difference may not be meaningful. In such cases, prefer the simpler, faster, or more interpretable model.

---

## 7. Common Student Questions (Anticipate These)

**Q: "Why 5 folds and not 10 or 20?"**
A: 5-fold CV uses 80% of the data for training in each iteration, which is close to the 60% we use in our fixed train/val/test split. 10-fold uses 90% for training, producing slightly less biased estimates but at 2x the computational cost. For most problems, 5-fold is a good default. If you need higher precision (e.g., for a final report), use repeated 5-fold or 10-fold.

**Q: "If CV uses all the data for validation, why do we still need a separate test set?"**
A: Because CV is used during model *development* — you try different models, tune hyperparameters, and select the best one. All of these decisions are based on CV scores, which means they are optimized to the training data distribution. The test set provides a *truly unseen* evaluation that is free from any selection bias. Without it, your reported performance would be slightly optimistic.

**Q: "Can I use CV with time-series data?"**
A: Not standard k-fold — shuffling time-series data breaks temporal order. Use `TimeSeriesSplit` instead, which always trains on past data and validates on future data. This is not covered in this notebook but is mentioned in the course plan for reference.

**Q: "Why does `cross_validate` return `test_score` instead of `val_score`?"**
A: This is a scikit-learn naming convention that confuses many students. In the CV context, "test" refers to the held-out fold (what we call "validation" in our train/val/test framework). It is NOT the locked test set. The naming is unfortunately overloaded. In this course, we always clarify: "CV test score" = validation fold score.

---

## 8. Connection to the Broader Course Arc

| Week | Notebooks | What NB08's CV Framework Enables |
|---|---|---|
| **Week 2** | 09 | `GridSearchCV` and `RandomizedSearchCV` embed CV as their inner evaluation loop. The `cv=StratifiedKFold(5)` parameter is the same object from NB08. Students who understand standalone CV instantly understand how grid search evaluates each parameter combination |
| **Week 2** | 10 | The midterm asks students to design evaluation plans. Proper answers include CV strategy choice (stratified vs. time-based), fold count justification, and metric selection — all from NB08 |
| **Week 3** | 11-15 | Tree-based models are compared using the `compare_models()` function from NB08. The CV protocol transfers directly: same folds, same metrics, different models |
| **Week 4** | 16-20 | Final project requires CV-based model comparison. The `compare_models()` function from NB08 is the template students adapt for their own datasets |

**NB08 is the methodological backbone of model evaluation.** From NB09 onward, every model comparison uses cross-validation. NB08 is where students learn the protocol, build the reusable function, and internalize the rules for fair comparison.

---

## 9. Suggested Video Structure

Below are two recording options. Each segment includes the **speaking prompt** (what to say), **timestamps**, and the **notebook cells** to show on screen.

All cell references use the format `Cell N` where N is the zero-indexed cell number in `08_cross_validation_model_comparison.ipynb`.

---

### Option A: Single Video (~12 minutes)

#### Segment 1 — Opening & Motivation `[0:00–1:30]`

> **Show:** Cell 0 (header) and Cell 1 (learning objectives) on screen.

**Say:**

"So far in this course, we have evaluated every model on a single validation set — the same 114 samples from the breast cancer dataset, or the same 4,128 samples from California Housing. But here is a question we have not asked: what if we got lucky with that split? What if a different random partition would have given us a very different score? Today we answer that question. Cross-validation replaces the fragile single-split estimate with k independent evaluations, giving us both a more stable average and a measure of uncertainty. By the end of this notebook, you will have a reusable function that compares any set of models using the same CV folds and the same metrics — a function you will use in your course project."

> **Action:** Scroll through the 5 learning objectives in Cell 1, briefly reading each one aloud.

---

#### Segment 2 — Setup & K-Fold Regression Example `[1:30–4:00]`

> **Show:** Cell 2 (setup code), then **run** it.

**Say:**

"The new imports here are the CV machinery: KFold, StratifiedKFold, cross_val_score, cross_validate, and RepeatedKFold. We also import RandomForestClassifier for the first time — we will use it in the model comparison section, not to explain how it works, but to show that different models can be compared fairly using the same CV protocol."

> **Show:** Cell 4 (why CV exists), Cell 5 (regression intro), then **run** Cell 6 (5-fold CV on Ridge + bar chart).

**Say:**

"We start with a regression example — California Housing with a Ridge pipeline. Five-fold CV splits the data into 5 equal parts, trains on 4, validates on 1, and rotates 5 times. The result: 5 R-squared scores. Look at the bar chart — the scores vary from fold to fold. The highest fold might score 0.61, the lowest 0.58. A single train/val split would have given us just one of these bars. Cross-validation gives us the full distribution: mean R-squared about 0.60, standard deviation about 0.01, and a 95% confidence interval that tells us where the true performance likely falls. This is why CV matters — it turns one fragile number into a distribution with quantified uncertainty."

---

#### Segment 3 — Stratified CV for Classification `[4:00–6:30]`

> **Show:** Cell 8 (stratified CV explanation), then **run** Cell 9 (regular vs. stratified comparison).

**Say:**

"Now classification. The breast cancer dataset is 37% malignant, 63% benign. With regular KFold, some folds might end up with 45% malignant while others get 30%. This random fluctuation adds noise to our estimates. StratifiedKFold fixes this by enforcing the 37/63 ratio in every fold. Look at the results: regular KFold and stratified KFold have similar means, but stratified typically has a lower standard deviation — the scores are more consistent across folds. The variance reduction percentage at the bottom quantifies the improvement. For classification, always use StratifiedKFold. It costs nothing extra and gives you more reliable estimates."

> **Show:** Cell 11 (Exercise 1 prompt), then **run** Cell 12 (cv_report function).

**Say:**

"Exercise 1 asks you to build a cv_report function. Here it is: pass in a model, data, a CV splitter, and a scoring metric, and it returns mean, standard deviation, overfitting gap, and fit time. The overfitting gap — the difference between train and validation scores — is a key diagnostic. A gap near zero means the model generalizes well. A large gap, say greater than 0.05, signals overfitting. This function is the CV analogue of the metrics dashboard from the previous notebook. Pause the video and examine the output, then answer the analysis questions."

---

#### Segment 4 — Fair Model Comparison `[6:30–9:30]`

> **Show:** Cell 15 (rules for fair comparison), then **run** Cell 16 (three-model comparison table).

**Say:**

"This is the most important section of the notebook. We compare three models — Logistic Regression with C equals 1, Logistic Regression with C equals 0.1, and Random Forest — using the exact same 5 stratified folds, the exact same scoring metrics, and the exact same data. This is the golden rule of model comparison: same folds, same metrics, same data. Change any one of these and the comparison becomes invalid.

Look at the results table. All three models score above 0.98 ROC-AUC. The two logistic regressions are nearly identical, which tells us regularization strength does not matter much here. Random Forest might be slightly higher or lower, but check the standard deviations — if the difference between means is smaller than the standard deviation, the difference is likely not meaningful. In that case, prefer the simpler model: Logistic Regression trains faster, is easier to interpret, and is less likely to overfit. The Fit_Time column confirms this — Random Forest takes noticeably longer because it builds 100 decision trees."

> **Mention:** "Pause the video and complete Exercise 2. Based on the comparison table, which model would you choose and why? Consider not just the mean score but also the standard deviation and fit time."

---

#### Segment 5 — Repeated CV & Reusable Function `[9:30–11:00]`

> **Show:** Cell 20 (repeated CV explanation), then **run** Cell 21 (single vs. repeated CV comparison + histograms).

**Say:**

"For even more stable estimates, you can repeat CV multiple times. Repeated 5-fold CV with 3 repeats gives you 15 scores instead of 5. The histograms show the difference: the single-CV histogram has only 5 bars, while the repeated-CV histogram is smoother and more bell-shaped. The means are similar, but the repeated version gives you more confidence in the estimate. The tradeoff is computation: 15 model fits instead of 5. For fast models like logistic regression, this is negligible. For slow models, it adds up."

> **Show:** Cell 23 (reusable function explanation), then **run** Cell 24 (compare_models function).

**Say:**

"Finally, we package the entire comparison workflow into a single function: compare_models. Pass in a dictionary of named models, data, a CV splitter, and one or more scoring metrics. It returns a tidy DataFrame with means, standard deviations, and fit times for every model. This function is project-ready — copy it into your course project, swap in your own models, and you get a standardized comparison table. The footer prints the CV object and scoring parameters so you always know exactly how the comparison was conducted."

---

#### Segment 6 — Wrap-Up & Closing `[11:00–12:00]`

> **Show:** Cell 26 (Wrap-Up).

**Say:**

"Five takeaways from today. One: CV reduces variance — it gives you a more honest performance estimate than a single split. Two: always use stratified CV for classification — it preserves class balance in every fold. Three: fair model comparison requires identical folds, identical metrics, identical data — change any one of these and the comparison is invalid. Four: always report mean and standard deviation — a mean without uncertainty is a half-truth. Five: build reusable functions — the compare_models function you built today will serve you throughout the rest of the course and beyond.

In the next notebook, we combine cross-validation with hyperparameter tuning. GridSearchCV tests every combination of parameters using the CV framework you learned today. It is the tool that automates the search for the best model configuration. We will also start preparing the project baseline model. See you there."

> **Show:** Cell 27 (Submission Instructions) briefly, then Cell 28 (Thank You).

---

### Option B: Three Shorter Videos (~4-5 minutes each)

---

#### Video 1: "Why Cross-Validation? K-Fold for Regression" (~5 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–0:45` | Show header + objectives | Cell 0, Cell 1 | "Until now, every model evaluation has been based on a single validation set. Today we ask: what if that split was lucky? Cross-validation replaces one fragile number with k independent estimates, giving us both a mean and a standard deviation. Here are our five learning objectives." (Read briefly.) |
| `0:45–1:30` | Run setup | Cell 2, Cell 3 | "New imports: KFold, StratifiedKFold, cross_val_score, cross_validate, RepeatedKFold. We also import RandomForestClassifier for the first time — we will use it for comparison, not explanation. Random seed 474 as always." |
| `1:30–3:00` | Why CV exists | Cell 4, Cell 5 | "The problem: with 114 validation samples, moving a few atypical samples between train and val can shift accuracy by 2-3 points. That means a model comparison based on a single split might pick the wrong winner. The solution: k-fold CV. Split into k folds, train on k-1, validate on 1, rotate, average. Every sample gets validated exactly once." |
| `3:00–5:00` | K-fold regression example | Cell 6, Cell 7 | "California Housing with Ridge regression, 5-fold CV. Five R-squared scores ranging from about 0.58 to 0.61. The bar chart makes the variance visible. A single split would give you one bar — maybe the best, maybe the worst. The mean is about 0.60 with a standard deviation of 0.01. The 95% confidence interval says the true performance is likely between 0.58 and 0.62. This is what honest evaluation looks like: a range, not a point." |

---

#### Video 2: "Stratified CV and Fair Model Comparison" (~5 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–0:30` | Recap | — | "Welcome back. K-fold CV gives us stable estimates for regression. For classification, we need one more ingredient: stratification. Let us see why." |
| `0:30–2:00` | Stratified vs. regular CV | Cell 8, Cell 9, Cell 10 | "Breast cancer dataset, 37% malignant, 63% benign. Regular KFold might create a fold with 45% malignant — that is not representative. StratifiedKFold enforces 37/63 in every fold. The side-by-side comparison shows similar means but stratified has lower standard deviation. The variance reduction can be substantial. Rule: for classification, always use StratifiedKFold. It is free insurance against unlucky fold compositions." |
| `2:00–3:00` | cv_report function | Cell 11, Cell 12, Cell 13 | "Our cv_report function wraps cross_validate and prints mean, std, overfitting gap, and fit time. The overfitting gap — train minus validation — is the key diagnostic. Small gap means good generalization. Large gap means overfitting. Pause and examine the output. Which numbers would concern you?" |
| `3:00–5:00` | Fair model comparison | Cell 15, Cell 16, Cell 17 | "Three models, same 5 stratified folds, same metrics. This is the golden rule: same folds, same metrics, same data. The results table shows that all three models perform similarly. When means are within one standard deviation, the difference is not meaningful — prefer the simpler model. Logistic Regression is faster, more interpretable, and just as accurate as Random Forest here. The Fit_Time column quantifies the computational cost difference." |

---

#### Video 3: "Repeated CV, Reusable Functions, and Wrap-Up" (~4 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–1:30` | Repeated CV | Cell 20, Cell 21, Cell 22 | "Repeated 5-fold CV with 3 repeats gives 15 scores instead of 5. The histogram is smoother, the standard deviation is tighter. The tradeoff: 3x the computation. For fast models, negligible. For slow models, use single CV during exploration and repeated CV for final reports. The side-by-side histograms make the stability improvement obvious." |
| `1:30–3:00` | Reusable compare_models function | Cell 23, Cell 24, Cell 25 | "The compare_models function packages everything into one call. Pass a dictionary of models, data, a CV splitter, and scoring metrics. Get back a tidy DataFrame. This function goes directly into your project. It enforces fair comparison by using the same folds for every model. The footer prints the CV configuration for reproducibility." |
| `3:00–4:00` | Wrap-up + closing | Cell 26, Cell 27, Cell 28 | "Five rules. One: CV reduces variance. Two: stratified CV for classification. Three: same folds, same metrics, same data. Four: report mean AND standard deviation. Five: build reusable functions. Next notebook, we combine CV with hyperparameter tuning — GridSearchCV tests every parameter combination using the CV framework you learned today. We also start building the project baseline. Submit your notebook, complete the quiz. See you there." |

---

*This guide was created to support video lecture recording for MGMT 47400 — Predictive Analytics, Purdue University.*
