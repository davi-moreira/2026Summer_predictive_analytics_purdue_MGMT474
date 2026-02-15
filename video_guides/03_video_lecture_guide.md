# Video Lecture Guide: Notebook 03 — Regression Metrics & Baseline Modeling

## At a Glance

This guide covers:

1. **Why NB03 exists** — It teaches students how to *measure* model quality using formal regression metrics (MAE, RMSE, R^2) and how to establish baselines that give every future comparison a meaningful floor

2. **Why it must follow NB02** — NB02 builds the preprocessing pipeline; NB03 assumes the pipeline is a solved problem and shifts focus entirely to evaluation. Without NB02, students would be wrestling with preprocessing and metrics simultaneously

3. **Why it must precede NB04** — NB04 introduces feature engineering (interactions, polynomials) and diagnostics. Students need to know *how to measure improvement* before they start engineering features, or they cannot tell whether their changes helped

4. **Why each library/tool is used:**
   - `DummyRegressor` — provides the simplest possible baselines (mean, median) that set the performance floor
   - `mean_absolute_error`, `mean_squared_error`, `r2_score` — the three core regression metrics, each with different sensitivity to outliers and different interpretability
   - `Pipeline` + `StandardScaler` + `LinearRegression` — reuses the NB02 pattern so students see the pipeline template in action a second time
   - Residual plots and predicted-vs-actual plots — visual diagnostics that reveal *where* the model fails, not just *how much*

5. **Common student questions** with answers (e.g., when to use MAE vs RMSE, what negative R^2 means, why the test set stays locked)

6. **Connection to the full course arc** — how baseline + metric discipline from NB03 carries through every subsequent model comparison

7. **Suggested video structure** — both single-video (~12 min) and three-video (~4 min each) formats with speaking prompts, timestamps, and notebook cell references

---

## Purpose of This Document

This guide is your **lecture companion** for recording the video(s) that accompany `03_regression_metrics_baselines.ipynb`. It explains the *why* behind every design choice in the notebook — why it exists, why it sits between notebooks 02 and 04, why specific metrics and tools are introduced, and why baseline modeling is treated as a formal step rather than an afterthought. Use it as speaking notes, not as a script: the goal is to internalize the reasoning so you can explain it naturally on camera.

---

## 1. Why This Notebook Exists

Notebook 02 ends with a working pipeline that produces an R^2 of approximately 0.60 on validation data. Students leave NB02 knowing *how* to build a pipeline, but they do not know how to properly *evaluate* or *contextualize* that number. Is 0.60 good? Compared to what? What does it mean in dollar terms?

Notebook 03 answers those questions. It introduces the three standard regression metrics (MAE, RMSE, R^2), shows students how to build baseline models that define a performance floor, and teaches the discipline of comparing every model against that floor. It also introduces residual analysis — the first visual diagnostic tool — so students can see not just *how much* the model errs but *where* and *how* it errs.

This notebook establishes the **evaluation framework** that every subsequent notebook in the course will reuse. Without it, students would have no principled way to decide whether feature engineering (NB04), regularization (NB05), or more flexible models (NB11-14) actually improve anything.

---

## 2. Why It Must Come After Notebook 02

Notebook 02 establishes three capabilities that notebook 03 *consumes*:

| NB02 Concept | How NB03 Uses It |
|---|---|
| **Pipeline template** (ColumnTransformer + Pipeline) | NB03 wraps `StandardScaler` + `LinearRegression` in a pipeline without re-explaining the mechanics. Students already know how `pipeline.fit()` prevents leakage. |
| **60/20/20 split with RANDOM_SEED = 474** | NB03 uses the identical split. Because students built and inspected splits in NB02, NB03 can load data, split in two lines, and move on. |
| **"Fit on train only" principle** | NB03 fits baselines and linear regression exclusively on training data and evaluates on validation. The discipline was taught in NB02; NB03 applies it without re-deriving the rationale. |

**In short:** NB02 gives the preprocessing infrastructure, NB03 gives the evaluation infrastructure. Attempting both in one notebook would overload students. Separating them lets each concept receive full attention.

If you tried to teach metrics *before* NB02, students would not have a pipeline to evaluate. If you skipped NB03 entirely, students would add features in NB04 and regularization in NB05 without knowing how to tell whether those additions actually improved the model.

---

## 3. Why It Must Come Before Notebook 04

Notebook 04 introduces **feature engineering** (manual interactions, polynomial features) and **residual diagnostics** to compare models. Its learning objectives assume students already know how to:

- Compute MAE, RMSE, and R^2 on both training and validation sets
- Compare a new model's metrics against a baseline
- Recognize overfitting by inspecting the train-validation gap
- Read a residual plot and a predicted-vs-actual plot

NB03 teaches all four of these skills. NB04 then asks students to *use* them to decide whether polynomial features are worth the added complexity. Without the metric vocabulary from NB03, students in NB04 would not be able to articulate why a model with 44 features and a 0.02 larger overfit gap might not be worth the improvement.

The deliberate sequence is:

```
NB02: How do we preprocess safely? (Pipeline + ColumnTransformer)
            |
NB03: How do we measure model quality? (Metrics + baselines + residuals)
            |
NB04: How do we improve the model? (Feature engineering + diagnostics)
```

Each notebook adds exactly one conceptual layer. By NB04, evaluation is a solved problem — students can focus entirely on feature engineering decisions.

---

## 4. Why These Specific Libraries and Tools

### 4.1 `sklearn.dummy.DummyRegressor`

**What it does:** Produces predictions that ignore the input features entirely. With `strategy='mean'`, it predicts the training-set mean for every sample. With `strategy='median'`, it predicts the training-set median.

**Why we use it:**

1. **It defines the floor.** Any model that cannot beat the mean predictor has learned nothing useful from the features. This sounds obvious, but in practice it catches real problems: misconfigured pipelines, target leakage that inflates training scores while validation collapses, or features that have no predictive relationship with the target.

2. **It calibrates expectations.** The mean-baseline MAE tells students: "If you predict nothing but the average house price, you will be off by about $91,000 per prediction." That number gives the linear model's MAE of ~$53,000 concrete meaning — it is a 40% improvement over doing nothing.

3. **It is trivial to compute but easy to forget.** Many real-world projects skip the baseline step and report model performance in isolation. Students who internalize the baseline habit will always have context for their numbers.

**What to emphasize on camera:** A baseline is not optional. It is the first number you compute in any evaluation, and every subsequent model comparison is relative to it. If your model cannot beat the mean, debug before proceeding.

### 4.2 `sklearn.metrics.mean_absolute_error` (MAE)

**What it does:** Computes the average of the absolute differences between predicted and true values: MAE = (1/n) * sum(|y_true - y_pred|).

**Why we use it:**

- **Same units as the target.** An MAE of 0.53 on California Housing means the model is off by about $53,000 on average. This is immediately interpretable by a non-technical stakeholder.
- **Treats all errors equally.** A $10,000 error and a $100,000 error contribute proportionally to the average. This is appropriate when the business cost of error is linear (e.g., each dollar of mis-estimate costs the same).
- **Robust to outliers.** Compared to RMSE, MAE is less affected by a few extreme predictions.

**What to emphasize on camera:** MAE answers the question "On average, how far off am I?" in the original units. It is the most intuitive metric for business audiences.

### 4.3 `sklearn.metrics.mean_squared_error` (used to compute RMSE)

**What it does:** Computes the average of squared differences. We take the square root to get RMSE, which returns to the original units.

**Why we use it:**

- **Penalizes large errors disproportionately.** Squaring amplifies big mistakes. An error of 10 contributes 100 to MSE; an error of 100 contributes 10,000. This is appropriate when large errors are catastrophically more expensive than small ones (e.g., undervaluing a house by $500,000 is not just 5x worse than undervaluing by $100,000 — it could mean a denied mortgage).
- **RMSE >= MAE always.** The gap between RMSE and MAE reveals how much the error distribution is dominated by outliers. A large gap means a few predictions are very wrong.
- **Standard in academic literature.** Most textbook comparisons report RMSE, making it essential for students who read papers.

**What to emphasize on camera:** Compare MAE and RMSE for the same model. If they are close, errors are fairly uniform. If RMSE is much larger, the model has some extreme mispredictions worth investigating.

### 4.4 `sklearn.metrics.r2_score` (R^2, Coefficient of Determination)

**What it does:** Measures the proportion of variance in the target that is explained by the model, relative to a mean-only predictor. R^2 = 1 - (sum of squared residuals / total sum of squares).

**Why we use it:**

- **Scale-free.** R^2 = 0.60 means 60% of variance explained, regardless of whether the target is in dollars, degrees, or millions. This makes it easy to compare models across different datasets.
- **Baseline-anchored.** R^2 = 0 means the model is exactly as good as the mean predictor (the DummyRegressor). R^2 < 0 means it is *worse* than the mean. This built-in baseline comparison is why R^2 is the first metric many practitioners check.
- **Commonly misinterpreted.** Students need to understand that R^2 = 0.60 does not mean the model is "60% correct." It means 60% of the variance is explained. The remaining 40% is noise, missing features, or nonlinear relationships the model cannot capture.

**What to emphasize on camera:** R^2 is relative, not absolute. A "good" R^2 depends on the domain. In physics, 0.99 is expected. In social science, 0.30 might be excellent. In housing, 0.60 is a reasonable starting point.

### 4.5 Residual Plots and Predicted-vs-Actual Plots

**What they do:** Residual plots show (prediction, residual) pairs as a scatter. Predicted-vs-actual plots show (true value, predicted value) pairs against the 45-degree line.

**Why we use them:**

- **They reveal *patterns* in errors.** A funnel shape means heteroskedasticity (error variance changes with prediction level). A curve means the model misses a nonlinear relationship. Random scatter means the model has extracted all learnable signal.
- **They show where the model struggles.** In the California Housing data, the predicted-vs-actual plot reveals a horizontal ceiling near $500,000 — the dataset caps median values. No linear model can predict above this cap.
- **They complement scalar metrics.** Two models can have identical R^2 but very different residual patterns. A model with random residuals is trustworthy; a model with patterned residuals has systematic bias worth investigating.

**What to emphasize on camera:** Always plot residuals. The number alone (MAE, R^2) tells you *how much* the model errs. The plot tells you *how* and *where*.

---

## 5. Key Concepts to Explain on Camera

### 5.1 "Beat the Mean or Go Home" — The Baseline Principle

The single most important idea in this notebook is that every model must be compared to a baseline, and the simplest regression baseline is predicting the training-set mean for every observation. If a model cannot beat this trivial strategy, it has learned nothing.

**Analogy for students:** Imagine you are a weather forecaster. Your baseline is "predict tomorrow's temperature will be the same as the historical average for this date." If your model cannot beat that simple guess, no one should trust your forecast. The mean-predictor is the same idea for house prices.

### 5.2 MAE vs RMSE — Choosing the Right Loss

MAE treats all errors equally. RMSE penalizes large errors more than small ones. The choice depends on business context:

- **MAE:** Every dollar of error costs the same. Appropriate for scenarios like average inventory surplus.
- **RMSE:** Large errors are disproportionately costly. Appropriate for scenarios like catastrophic under-pricing of real estate or medical dosage errors.

**Analogy for students:** MAE is like a parking meter — every minute costs the same. RMSE is like a speeding fine — going 5 mph over is a small fine, but going 50 mph over is a massive penalty. Choose the metric that matches how your business penalizes mistakes.

### 5.3 The Test Set Lockbox

NB03 explicitly reinforces the test-set discipline introduced in NB01 and NB02. The test set is never evaluated in this notebook. All model selection and comparison use the validation set.

**Analogy for students:** The test set is a sealed envelope containing your final exam. You can take as many practice tests as you want (validation set), but you only open the sealed envelope once — at the very end. If you peek, the exam no longer measures your real ability.

### 5.4 Residuals Reveal What Numbers Cannot

A residual is the difference between the true value and the prediction. Plotting residuals against predicted values is the most information-dense diagnostic in regression:

- **Random scatter:** The model is working correctly.
- **Funnel shape:** The model's errors grow with the prediction level (heteroskedasticity).
- **Curved pattern:** The model misses a nonlinear relationship (motivates NB04's polynomial features).
- **Clusters:** Subgroups in the data behave differently.

---

## 6. What Students Should Take Away

After watching your video and completing this notebook, students should be able to answer these questions confidently:

1. **"What is the difference between MAE and RMSE?"** — MAE is the average absolute error (treats all errors equally). RMSE penalizes large errors more because it squares them before averaging. RMSE is always >= MAE.

2. **"What does R^2 = 0 mean?"** — The model is exactly as good as predicting the mean for every observation. It has learned nothing beyond the average.

3. **"Why compute a baseline before building a real model?"** — Because without a baseline, you have no reference point. An MAE of 0.53 is meaningless until you know the mean-predictor's MAE is 0.91 — then you see the model cut the error by 40%.

4. **"What does a funnel-shaped residual plot tell you?"** — The model's errors are not constant across the prediction range. Errors grow larger for higher (or lower) predictions. This is called heteroskedasticity and suggests the model may need log-transformed targets or more flexible features.

5. **"Why do we never evaluate on the test set in this notebook?"** — Because every time you look at test performance and adjust your model based on it, you leak information. The test set must remain untouched until the final evaluation.

---

## 7. Common Student Questions (Anticipate These)

**Q: "When should I use MAE vs RMSE vs R^2?"**
A: Use MAE when you want an interpretable, dollar-denominated error and all errors cost the same. Use RMSE when large errors are disproportionately costly. Use R^2 when you want a scale-free number to compare models across different datasets. In practice, report all three — they tell you different things.

**Q: "Can R^2 be negative?"**
A: Yes. R^2 < 0 means the model is *worse* than predicting the mean. This typically happens when you evaluate on a different dataset or when the model is severely mis-specified. It is a strong signal that something is wrong.

**Q: "Why is the mean baseline's R^2 exactly zero?"**
A: By definition. R^2 measures improvement over the mean predictor. The mean predictor compared to itself shows zero improvement. This is not a coincidence — it is how R^2 is constructed.

**Q: "The residual plot shows a horizontal ceiling. Is that a model problem or a data problem?"**
A: It is a data problem. The California Housing dataset caps median house values at $500,000. No model can predict above this ceiling because the true values are capped. This is a form of *censoring* in the data. In a real project, you would investigate whether the cap affects your business question and possibly filter or transform the target.

**Q: "My model's train R^2 is 0.60 and validation R^2 is 0.62. How can validation be *higher* than training?"**
A: Small differences like this are normal random variation from the split. It does not mean the model generalizes better than it fits. The key check is whether the gap is *large* (say, train 0.90 and validation 0.60) — that would signal overfitting. A gap of 0.02 in either direction is noise.

---

## 8. Connection to the Broader Course Arc

| Week | Notebooks | What NB03's Evaluation Framework Enables |
|---|---|---|
| **Week 1** | 01-05 | NB03 metrics (MAE, RMSE, R^2) and baselines are used directly in NB04 (comparing polynomial vs linear) and NB05 (comparing Ridge/Lasso vs OLS). The `evaluate_regression()` helper function becomes a reusable tool. |
| **Week 2** | 06-10 | Classification metrics (accuracy, log loss, F1) introduced in NB06-07 follow the same baseline-first, metric-comparison pattern established here. The residual analysis concept translates to confusion matrices and ROC curves. |
| **Week 3** | 11-15 | Tree-based and ensemble models are evaluated using the same train/val framework and the same philosophy: always compare to a baseline, always check the train-val gap, always plot diagnostics. |
| **Week 4** | 16-20 | The final project report uses the evaluation documentation template introduced in NB03 (Section 9). Students write metric rationale, baseline performance, and improvement percentages — all skills practiced here for the first time. |

**The evaluation framework is the second backbone of the course** (after the pipeline from NB02). Every model comparison from NB04 onward follows the pattern: baseline metric, model metric, improvement percentage, diagnostic plots, overfit check.

---

## 9. Suggested Video Structure

Below are two recording options. Each segment includes the **speaking prompt** (what to say), **timestamps**, and the **notebook cells** to show on screen.

All cell references use the format `Cell N` where N is the zero-indexed cell number in `03_regression_metrics_baselines.ipynb`.

---

### Option A: Single Video (~12 minutes)

#### Segment 1 — Opening & Motivation `[0:00–1:30]`

> **Show:** Cell 0 (header) and Cell 1 (learning objectives) on screen.

**Say:**

"Welcome back. In the last notebook we built a preprocessing pipeline — StandardScaler plus LinearRegression, all wrapped in a Pipeline object that prevents leakage. We got an R-squared of about 0.60. But what does 0.60 actually mean? Is it good? Is it bad? Compared to what? That is what this notebook is about. Today we learn how to measure model quality using three standard regression metrics — MAE, RMSE, and R-squared — and we learn the most important habit in predictive analytics: always establish a baseline before evaluating any model. By the end, you will have a complete evaluation framework that you will reuse in every notebook for the rest of this course."

> **Action:** Scroll through the 5 learning objectives in Cell 1, briefly reading each one aloud.

---

#### Segment 2 — Setup & Data Loading `[1:30–2:30]`

> **Show:** Cell 2 (setup explanation), then **run** Cell 3 (imports).

**Say:**

"Standard setup — same as every notebook. Notice a few new imports: `mean_absolute_error`, `mean_squared_error`, and `r2_score` from sklearn.metrics. These are the three regression metrics we will use today. We also import `DummyRegressor`, which will serve as our baseline model. Random seed 474 as always."

> **Run** Cell 3. Point to the `Setup complete!` output.

> **Run** Cell 6 (load data and split). Point to the 60/20/20 sizes and the test-set warning.

**Say:**

"Same California Housing dataset, same 60-20-20 split. The test set is locked — we will not touch it. Everything we do today uses training and validation only."

---

#### Segment 3 — Metrics & Evaluation Function `[2:30–4:30]`

> **Show:** Cell 8 (metrics explanation in markdown).

**Say:**

"Let me explain the three metrics we are going to use. First, Mean Absolute Error, or MAE. This is the simplest: take the absolute difference between each prediction and the true value, then average them. The result is in the same units as the target — here, hundreds of thousands of dollars. An MAE of 0.53 means on average we are off by about 53,000 dollars. Second, Root Mean Squared Error, or RMSE. This is similar but it squares the errors before averaging, then takes the square root. The squaring means large errors get penalized much more than small ones. Think of it this way: MAE is a parking meter, every minute costs the same. RMSE is a speeding fine — going 5 over is small, going 50 over is enormous. RMSE is always greater than or equal to MAE. The bigger the gap, the more your model suffers from a few extreme mispredictions. Third, R-squared, the coefficient of determination. This measures the proportion of variance explained by the model relative to simply predicting the mean. R-squared of 0 means you are no better than the mean. R-squared of 1 means perfect predictions. R-squared can even be negative if your model is worse than the mean."

> **Run** Cell 9 (evaluation function). Point to the `Evaluation function created` output.

**Say:**

"We wrap all three metrics into a single helper function called `evaluate_regression`. This function takes true values, predicted values, and a label, and returns a dictionary with MAE, RMSE, and R-squared. Every model in this notebook — and in future notebooks — will be scored through this same function, so comparisons are always consistent."

---

#### Segment 4 — Baseline Models `[4:30–6:30]`

> **Show:** Cell 11 (baselines explanation), then **run** Cell 12 (fit baselines).

**Say:**

"Now the most important step in any evaluation: establishing a baseline. A baseline is the simplest possible model that requires zero intelligence. For regression, the two standard baselines are: predict the training-set mean for every observation, and predict the training-set median for every observation. We use scikit-learn's DummyRegressor, which ignores the features entirely and just predicts a constant. This sounds trivial, but it gives us the single most important reference number in the evaluation."

> **Run** Cell 15 (PAUSE-AND-DO 1 solution — baseline comparison table).

**Say:**

"Look at this table. Both baselines have an R-squared of approximately zero — which makes sense, because R-squared is defined relative to the mean predictor. The MAE tells us something much more useful: the mean predictor is off by about 91,000 dollars per prediction on average. That is our floor. Any model we build must do better than this, or it has learned nothing. Notice also that RMSE is larger than MAE for both baselines. This tells us the error distribution has some large outliers — houses where even the mean guess is far off."

> **Mention:** "Pause the video here and complete Exercise 1 — evaluate both baselines and create the comparison table." Point to Cell 14 (PAUSE-AND-DO 1).

---

#### Segment 5 — Linear Model & Comparison `[6:30–8:30]`

> **Show:** Cell 17 (linear model explanation), then **run** Cell 18 (fit linear pipeline).

**Say:**

"Now let us see if even the simplest real model can beat the baseline. We build the same StandardScaler plus LinearRegression pipeline from notebook 02. Same code, same pattern. One call to fit, one call to predict."

> **Run** Cell 21 (full comparison table with improvement).

**Say:**

"And here is the payoff. The full comparison table shows baselines and linear regression side by side. Linear regression has an MAE of about 0.53, compared to the baseline's 0.91. That is a roughly 40 percent reduction in average error. R-squared jumps from 0 to about 0.60, meaning the model now explains 60 percent of the variance in house prices. And the train-validation gap is tiny — less than 0.02 — which means essentially no overfitting. This is the pattern you will follow in every notebook: baseline first, then model, then comparison. If your model cannot beat the baseline, stop and debug before proceeding."

> **Mention:** "Pause the video and complete Exercise 2 — interpret the delta between baseline and linear regression." Point to Cell 20 (PAUSE-AND-DO 2).

---

#### Segment 6 — Residual Analysis `[8:30–10:30]`

> **Show:** Cell 24 (residual explanation), then **run** Cell 25 (residual plots — 2x2 grid).

**Say:**

"Numbers tell you how much the model errs. Plots tell you how and where. These are residual plots. Top row: residuals plotted against predicted values for training and validation. What you want to see is a formless cloud centered on the red zero line. What you actually see is a slight funnel shape — errors are wider for higher predictions. This is called heteroskedasticity. It means the model's accuracy varies across the price range. You can also see a hard edge on the right side — that is the 500,000 dollar ceiling in the data. The model cannot predict above this cap because the true values are capped.

Bottom row: histograms of residuals. They are roughly bell-shaped and centered near zero, which is good. But there is a noticeable right tail, meaning the model tends to under-predict the most expensive houses."

> **Run** Cell 28 (predicted vs actual plots).

**Say:**

"Predicted-versus-actual plot: every dot should lie on the red 45-degree line. The tighter the cluster, the better the model. You can see the dispersion is wider at higher values, and there is that horizontal ceiling effect again. This is the fastest way to communicate model quality to a non-technical audience — they can immediately see that most predictions are in the right ballpark, but the model struggles with high-value properties."

---

#### Segment 7 — Test Set Discipline, Documentation & Closing `[10:30–12:00]`

> **Show:** Cell 30 (test set lockbox rules), then **run** Cell 31 (test set status).

**Say:**

"One more critical concept: the test set lockbox. We have not computed a single metric on the test set in this entire notebook. That is deliberate. The test set is your final exam — you open it once, at the very end, and report whatever number comes out. If you evaluate on test repeatedly and tweak your model based on those numbers, you are effectively fitting to the test set, and your reported performance is a lie."

> **Show:** Cell 33 (evaluation documentation template). Skim through the sections.

**Say:**

"Finally, this evaluation documentation template. Every time you evaluate a model — in this course or in your career — you should document: the primary metric and why you chose it, the split strategy, the baseline performance, the model performance, the improvement over baseline, and any assumptions or risks. This template will be part of your final project deliverable."

> **Show:** Cell 34 (Wrap-Up). Read the two critical rules.

**Say:**

"Two rules to remember. One: if your model cannot beat the mean, debug before proceeding. Two: the test set is a lockbox — open it once. This evaluation framework — baselines, metrics, residual plots, documentation — is the second backbone of the course after the pipeline from notebook 02. In the next notebook, we will use this framework to evaluate feature engineering: interactions, polynomial features, and their effect on model quality. See you there."

> **Show:** Cell 35 (Submission Instructions) briefly, then Cell 36 (Thank You).

---

### Option B: Three Shorter Videos (~4 minutes each)

---

#### Video 1: "Metrics and Baselines — The Evaluation Floor" (~4 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–0:45` | Show header + objectives | Cell 0, Cell 1 | "Welcome back. In notebook 02 we built a preprocessing pipeline and got an R-squared of 0.60. But what does that number actually mean? Today we learn how to measure model quality properly — three metrics, two baselines, and the habit of always comparing to a floor. Here are our five learning objectives." (Read them briefly.) |
| `0:45–1:30` | Run setup + load data | Cell 2, Cell 3, Cell 6 | "Standard setup. New imports: mean_absolute_error, mean_squared_error, r2_score, and DummyRegressor. Same California Housing dataset, same 60-20-20 split. Test set is locked as always." |
| `1:30–2:30` | Explain metrics | Cell 8 | "Three metrics. MAE: average absolute error, same units as target, treats all errors equally. RMSE: penalizes large errors more via squaring, always larger than MAE. R-squared: proportion of variance explained, zero means no better than the mean, one is perfect, can be negative if the model is terrible." |
| `2:30–3:15` | Fit baselines + compare | Cell 9, Cell 12, Cell 15 | "We fit two DummyRegressors: mean and median. Both have R-squared near zero by definition. The mean predictor's MAE is about 0.91 — on average, guessing the mean is off by $91,000. This is our floor. Any model that cannot beat this has learned nothing." |
| `3:15–4:00` | Exercise 1 prompt | Cell 14 | "Pause the video now and complete Exercise 1. Evaluate both baselines on train and validation sets, create the comparison table, and interpret the results. Come back when you are done." |

---

#### Video 2: "Linear Regression vs Baseline — Your First Model Comparison" (~4 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–0:30` | Recap | — | "Welcome back. We have our baselines: mean predictor with MAE around 0.91. Now let us see how much a simple linear model improves on that." |
| `0:30–1:30` | Fit linear model + compare | Cell 17, Cell 18, Cell 21 | "Same pipeline pattern from notebook 02: StandardScaler plus LinearRegression. One call to fit, one call to predict. The comparison table shows linear regression cuts MAE by about 40 percent — from 0.91 to 0.53. R-squared jumps from 0 to about 0.60. The train-validation gap is tiny — no overfitting. This is the pattern: baseline first, model second, comparison third." |
| `1:30–2:30` | Exercise 2 prompt | Cell 20 | "Pause and complete Exercise 2: interpret the improvement over baseline. How much did MAE drop? What does R-squared of 0.60 mean in practical terms? What does the small train-val gap tell you?" |
| `2:30–4:00` | Residual analysis | Cell 24, Cell 25, Cell 28 | "Now we move beyond numbers to pictures. Residual plots show prediction on x, error on y. Random scatter is good. Funnel shape means error varies with prediction level — we see that here, plus a hard ceiling near $500k from the dataset's value cap. Predicted-vs-actual plot: dots on the 45-degree line are perfect predictions. Wider dispersion at high values confirms the model struggles with expensive properties. These plots tell you *where* the model fails — numbers alone cannot do that." |

---

#### Video 3: "Test Set Discipline and Documentation" (~4 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–1:00` | Test set lockbox | Cell 30, Cell 31 | "We have not touched the test set once in this entire notebook. That is the rule. The test set is your sealed final exam. You open it once at the very end and report whatever comes out. If you peek repeatedly and adjust your model each time, you are fitting to the test set and your numbers are meaningless." |
| `1:00–2:00` | Evaluation documentation template | Cell 33 | "Here is a documentation template for every evaluation you do: metric, rationale, split strategy, baseline, model performance, improvement, assumptions, risks. Fill this out for every model. In your final project, this structure will be part of your deliverable." |
| `2:00–3:00` | Wrap-Up | Cell 34 | "Five key takeaways. One: choose metrics based on business cost. Two: always compute a baseline first. Three: train on train, evaluate on validation, lock the test set. Four: residual plots reveal what numbers cannot. Five: document everything. Two critical rules: if your model cannot beat the mean, debug. The test set is a lockbox — open it once." |
| `3:00–4:00` | Closing + next notebook | Cell 35, Cell 36 | "This evaluation framework is the second backbone of the course. Every model comparison from now on follows this pattern: baseline, model, improvement, diagnostics, overfit check. Next notebook, we use this framework to evaluate feature engineering — interactions and polynomial features. Submit your notebook to Brightspace and complete the quiz. See you in notebook 04." |

---

*This guide was created to support video lecture recording for MGMT 47400 — Predictive Analytics, Purdue University.*
