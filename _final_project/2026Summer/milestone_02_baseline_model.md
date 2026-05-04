# Milestone 02 — Simple Model and Performance Evaluation

## About the Final Project

The Final Project is a **group capstone** (groups of four randomly assigned) culminating in a **research poster**. Across the term your group completes four milestone deliverables (M1 → M4); each member also submits a confidential intra-group peer evaluation at the end. The project is worth **35% of your overall course grade**, broken down as:

- **Milestone Deliverables — 40%** of the project grade. Averaged across the four milestones (M1–M4). Graded for clarity, completeness, and timely submission.
- **Peer Evaluation — 20%** of the project grade. Confidential intra-group ratings collected at the end of the course.
- **Instructor / TA Evaluation — 40%** of the project grade. The final research poster, graded against the poster rubric (Milestone 04).

Optional presentation at the **Fall 2026 Purdue Undergraduate Research Conference** is strongly encouraged but **not required**. Professor Moreira is happy to serve as faculty mentor for groups choosing to present. Award-winning prior posters from this course: <https://davi-moreira.github.io/applied_projects.html>. Additional information about Purdue undergraduate research conferences: <https://www.purdue.edu/undergrad-research/conferences/index.php>.

---

## Purpose

This milestone moves the project from "we know what we want to predict" to "we have a working baseline model evaluated honestly with cross-validation." By the end of M2 the group should have:

- A clean, reproducible Pipeline (`ColumnTransformer` + scaler + encoder + simple estimator)
- A defensible split + CV protocol with the test set in the lockbox
- A baseline-results table reporting the metric(s) chosen at M1, with a 95% Student's *t* CI

This baseline is the **floor every later candidate model must beat**. M3's complex model will be judged against it under the CI-overlap rule.

## Components

Your M2 report must address all six sections below.

### 0. Prediction Goal(s)

Restate (and refine, if needed, based on M1 feedback) your prediction goal. Confirm whether the problem is **regression** (continuous target) or **classification** (categorical target).

### 1. Dataset Exploration

#### Dataset Overview
- Number of observations and features
- Brief description of each feature (data type, meaningful range)
- Identification of the response variable (continuous or categorical)

#### Descriptive Statistics & Visualizations

Summary statistics for the response variable (mean, median, SD; class-balance percentages for classification) plus a discussion of notable findings (skew, outliers, class imbalance, missingness patterns).

**Required visualizations** (embed each in the report with axis labels, units where applicable, and a 1–2 sentence caption):

- **Target distribution** — histogram for continuous targets; bar chart of class counts (or percentages) for categorical targets
- **Feature distributions** — histograms or boxplots for at least 2–3 numeric predictors; bar charts for at least 1–2 categorical predictors
- **Correlation heatmap** of the numeric features (use `seaborn.heatmap` with annotated values)
- **At least one bivariate plot** relating a top predictor to the response (scatter plot for regression; boxplot or grouped histogram for classification)

#### Predictor Variables & Response Variable
- Classify each predictor's type
- Discuss expected relationship to the response (domain knowledge or preliminary correlation)

### 2. Feature Engineering

- Create at least **one new feature** (interaction term, derived ratio, temporal feature, domain-knowledge encoding, etc.)
- Provide a clear rationale tied to either domain knowledge or data-driven insight
- Explain how the new feature is expected to improve model performance or interpretability
- Implement the feature **inside the Pipeline** (using `FunctionTransformer` or a custom step) so it is recomputed on each training fold without leakage

### 3. Handling Missing Values

- Identify which features have missing values and the extent
- Choose and **justify** a strategy (median imputation, indicator flag, drop, etc.)
- Compare against at least one alternative approach with reasoning

### 4. Baseline Model Implementation

#### 4a. Model Choice
Pick the simplest reasonable model for your goal:
- **Continuous response:** Linear Regression (or Ridge / Lasso if the feature set is wide)
- **Categorical response:** Logistic Regression

Justify the choice in one short paragraph.

#### 4b. Implementation (Feature Selection + k-Fold CV)
- Combine a feature-selection procedure (correlation-based, stepwise, or feature-importance-based) with **k-fold cross-validation** (k=5 or k=10) inside `Pipeline` + `GridSearchCV`
- Stratified k-fold for classification (preserves the positive-class rate per fold); plain k-fold for regression
- Compare candidate feature subsets on the chosen metric (MAE/RMSE/R² for regression; accuracy/PR-AUC/recall for classification)
- Pick the best baseline by lowest CV error (or highest CV PR-AUC)

#### 4c. Interpretation & Next Steps
- Report final CV score with **95% Student's *t* CI**
- Reflect on whether results suggest further feature engineering, feature selection, or hyperparameter tuning at M3
- Propose realistic future steps grounded in the observed performance

#### 4d. Required Diagnostic Visualizations

Embed each of the following in the report with axis labels, units where applicable, and a 1–2 sentence caption explaining what the figure shows:

- **Bar chart of CV scores across candidate feature subsets** (one bar per candidate), with **error bars showing the 95% Student's *t* CI** — this is the visual evidence behind your "best baseline" choice.
- **For regression** problems:
  - **Predicted-vs-actual scatter plot** with a 45° reference line (`y = x`)
  - **Residual plot** (residuals vs. predicted values) with a horizontal `y = 0` reference line
- **For classification** problems:
  - **Confusion matrix** at the default threshold (use `ConfusionMatrixDisplay`)
  - **ROC curve with AUC annotation** and **Precision–Recall curve with PR-AUC annotation** — lead with PR-AUC when the positive class is rare

---

## Submission Expectations

| Item | Specification |
|---|---|
| **Structured Report (PDF)** | Clearly labeled sections matching the components above; explanations and rationale in paragraph form; visualizations and tables embedded; key code snippets in the appendix or inline |
| **Code Files** | Python script or Jupyter notebook submitted separately. Must run cleanly top-to-bottom on the dataset (a fresh Colab "Runtime → Run all" must succeed) |
| **Submission location** | Brightspace — Module 2, Final Project Milestone 02 |
| **Filename convention** | `group-<NN>_M02_baseline.pdf` and `group-<NN>_M02_baseline.ipynb` |

---

## Grading Rubric (100 points)

| Criterion | Points |
|---|---:|
| **0. Prediction Goal(s)** — Clear, specific goals with rationale tied to dataset/problem context; regression vs. classification confirmed | **5** |
| **1. Dataset Exploration** — Comprehensive overview, meaningful summary statistics, **all four required EDA visualizations present and properly labeled** (target distribution, 2–3 feature distributions, correlation heatmap, ≥1 bivariate plot), insightful discussion | **20** |
| **2. Feature Engineering** — At least one new feature with convincing justification and clear method; implemented inside the Pipeline | **15** |
| **3. Handling Missing Values** — Thorough identification, justified strategy, comparison with at least one alternative | **10** |
| **4. Baseline Model Implementation** | **30** |
| &nbsp;&nbsp;&nbsp;&nbsp;4a. Model Choice (5) | |
| &nbsp;&nbsp;&nbsp;&nbsp;4b. Implementation: feature selection + k-fold CV (12) | |
| &nbsp;&nbsp;&nbsp;&nbsp;4c. Interpretation & Next Steps (8) | |
| &nbsp;&nbsp;&nbsp;&nbsp;4d. Required Diagnostic Visualizations: CV-bar-with-CI plot + residual / predicted-vs-actual (regression) OR confusion matrix + ROC + PR curves (classification) (5) | |
| **5. Report Quality & Clarity** — Well-structured PDF, properly labeled visualizations, logical flow, error-free code | **20** |
| **Total** | **100** |

This rubric grade contributes to the **Milestone Deliverables (40%)** component of the Final Project grade — the average across all four milestones (M1–M4).

---

## Tips and Common Pitfalls

- **Lock the test set now.** All M2 evaluation uses 5-fold CV on the training fold. The held-out test set stays in its lockbox until M4.
- **Any fit step belongs inside the Pipeline.** Imputers, scalers, encoders, feature-engineering transformers — fit only on the training fold, never on the full dataset.
- **Beat both baselines.** A defensible model beats both `DummyRegressor`/`DummyClassifier` AND a segment-aware mean (per-category for regression; per-cohort for classification). If the candidate beats only the dummy, the lift may not be real.
- **Report the CI, not just the mean.** The headline number is `CV mean ± 95% Student's *t* CI` — single-split numbers are not the headline.
- **Watch for leakage.** Recompute target encoders inside each fold, exclude post-decision/future-window features, never `fit` on validation rows.
- **Diagnostic visualizations are required, not optional.** Draft the CV-bar-with-CI plot and the regression / classification diagnostic figures as soon as you have a working baseline — they make problems (heteroscedastic residuals, miscalibrated probabilities, class-confusion patterns) visible that summary numbers hide.

---

**End of Milestone 02 instructions.**
