# MGMT 47400 – Predictive Analytics (3 credits)  
## 4-Week Fully Online Course Plan (Daniels School of Business)  
**Run dates (business days):** Tue **May 18, 2027** → Mon **June 14, 2027** (20 business days)  
**Daily engagement target:** **112.5 minutes per business day** (videos + Colab notebooks + exercises + quizzes + project work)  
**Instruction format:** short recorded **micro-videos (≤ 12 minutes each)** + **hands-on Jupyter Notebooks** opened in **Google Colab**  
**AI support:** students use **Gemini inside Colab** for guided “vibe coding” (draft → verify → document)  
**Course center of gravity:** supervised predictive modeling in Python (ISLP-with-Python style)

---

## Delivery constraints (operational)
- **112.5 minutes per business day** per student (Mon–Fri), inclusive of videos, notebook work, readings, exercises, quizzes, and project work.
- All instructional video segments are **≤ 12 minutes**.
- Every lecture/topic includes at least one **Google Colab-ready notebook**.
- Every day includes at least one **10-minute “pause-and-do” exercise** inside the notebook.

---

## Pedagogical pattern (used consistently)
For each topic/day, content follows a repeating loop:
1. **Concept + demo in notebook**  
2. **Guided practice** with a **10-minute student exercise (“pause-and-do”)**  
3. **Next micro-video begins with solution + common mistakes + extensions**  
4. **Next concept + demo** … and repeat

---

## Course-wide core references (used repeatedly)
- **James, Witten, Hastie, Tibshirani.** *An Introduction to Statistical Learning* (ISLP) + Python labs.
- **Hastie, Tibshirani, Friedman.** *The Elements of Statistical Learning* (ESL).
- **Provost, Fawcett.** *Data Science for Business*.
- **Pedregosa et al.** “Scikit-learn: Machine Learning in Python.” *JMLR*.
- **scikit-learn User Guide** (pipelines, preprocessing, model selection, metrics, inspection).
- **Chip Huyen.** *Designing Machine Learning Systems* (deployment thinking, monitoring).

---

# Weekly structure and project milestones
## Project (single end-to-end applied project; progresses weekly)
- **Week 1 (due Day 5): Proposal + dataset selection**
- **Week 2 (due Day 10): Baseline model + evaluation plan**
- **Week 3 (due Day 15): Improved model + interpretation**
- **Week 4 (due Day 20): Final model + executive-ready deliverable (slide-style narrative + conference video)**

---

# Week 1 (Days 1–5): Foundations, EDA, Splits, Linear Regression, Regularization  
**Project milestone:** Week 1 proposal due **Day 5**

---

## Day 1 — Tue May 18  
### Launchpad: Colab workflow, Gemini vibe-coding, EDA, and splitting correctly  
**Learning objectives**
- Course Syllabus and Logistics
- Operate course workflow in Google Colab (run-all, save-copy, etc.).
- Use Gemini in Colab to accelerate coding while preserving accountability (explain + verify).
- Understand the Predictive Analytics Workflow
- Perform structured EDA (types, missingness, target distribution, leakage sniff test).
- Create train/validation/test splits with reproducible seeds.
- Identify obvious leakage patterns before modeling.

**Micro-videos (total 54 min)**
1. Welcome and Introductions 
  1.1 Instructor
  1.2 Students
2. Course Syllabus and Logistics
  2.1 Course Brightspace Page
  2.2 Course Syllabus
  2.3 Grade
  2.4 Quizzes
  2.5 Course Case competition
  2.6 Final Project
  2.7 AI Policy
3. Concept+demo: Colab setup + course notebook conventions (10)  
4. Introduction to Predictive Analytics
  4.1 Examples
  4.2 Supervised vs Unsupervised Learning Models: we will focus on Supervised models
  4.3 End-to-End Workflow
  4.4 Data Leakage
  4.5 Assessing model accuracy
  4.6 The curse of dimensionality
  4.7 Flexibility vs. Interpretability
  4.8 Bias-Variance Trade-off
5. Guided practice: EDA checklist (what to compute/plot first) (8)  
6. Solution: EDA walkthrough + common plotting/data-type mistakes + extensions (9)  
7. Concept+demo: Train/validation/test and why leakage happens (10)  
8. Guided practice: Implement reproducible splits + sanity checks (8)  
9. Solution: Split validation + leakage red flags + extension: stratified splits (9)

**Notebook(s)**
- File: `01_launchpad_eda_splits.ipynb`  
- Sections:
  - Setup (installs, imports, seeds, display settings)
  - Gemini workflow rules (“ask → verify → document”)
  - Load dataset (course-provided sample)
  - EDA checklist (Section 6 in notebook):
    - 6.1 Data Types Audit — `df.dtypes` and `df.info()` to confirm all features are numeric, identify column count, and verify no unexpected object/string columns
    - 6.2 Missingness Check — per-column missing count and percentage table; confirms California Housing has zero missing values
    - 6.3 Basic Descriptive Statistics — `df.describe()` summary (mean, std, min, quartiles, max) across all features and target; students spot scale differences and outlier-prone columns (AveRooms, AveOccup, Population)
    - 6.4 Target Distribution — side-by-side histogram and box plot of MedHouseVal with mean/median reference lines; reveals right skew and the $500k cap; outputs key statistics (count, mean, median, std, min, max)
    - 6.5 Feature Distributions — 3×3 grid of histograms (one per feature) with mean reference lines; highlights MedInc right skew, HouseAge uniformity, Population heavy tail, and Latitude/Longitude geographic clustering
    - 6.6 Correlation Analysis — annotated heatmap of the full correlation matrix plus sorted correlations with target; confirms MedInc is the strongest predictor (r ≈ 0.69) and surfaces multicollinearity (AveRooms–AveBedrms)
  - Splits (train/val/test) + leakage sniff test
  - Wrap-up: key takeaways + “next-day readiness” cells

**In-notebook exercises (10-minute scope)**
- Pause-and-do (10): Complete the EDA checklist on a provided dataset and summarize 3 findings.  
- Pause-and-do (10): Create train/val/test splits and write 3 leakage risks specific to the dataset.

**Assessments**
- Concept quiz (auto-graded, 5–7 items): EDA, splits, leakage basics  
- Colab readiness check: submit Colab link with all cells executed

**Time budget (112.5 min)**
- Videos 54 + Notebook work 46 + Quiz 7.5 + Reflection 5 = 112.5

**Bibliography**
- ISLP: introductory material + Python lab basics (as assigned)  
- scikit-learn User Guide: cross-validation overview; common pitfalls and recommended practices  
- Kaggle Learn (optional): data leakage + train/test split discipline

---

## Day 2 — Wed May 19  
### Data setup and preprocessing pipelines (the professional way)  
**Learning objectives**
- Audit data types and fix common pandas pitfalls (strings, categories, dates).
- Handle missing values without leaking information.
- Build a preprocessing + model Pipeline with `ColumnTransformer`.
- Separate “fit on train only” logic from evaluation logic.
- Use Gemini to draft pipeline code and then harden it (tests + comments).

**Micro-videos (54 min)**
1. Concept+demo: pandas audit: types, missingness, duplicates (10)  
2. Guided practice: Write a minimal cleaning function (8)  
3. Solution: Cleaning solution + mistakes + extension: unit checks (9)  
4. Concept+demo: Pipelines + ColumnTransformer (numeric/categorical) (10)  
5. Guided practice: Build preprocessing pipeline (impute/encode/scale) (8)  
6. Solution: Pipeline debugging + extension: `get_feature_names_out()` (9)

**Notebook(s)**
- File: `02_preprocessing_pipelines.ipynb`  
- Sections:
  - Setup + dataset load
  - Data audit report function
  - Train/val/test imports from Day 1 pattern
  - Pipeline template (numeric + categorical)
  - Gemini prompt cards for pipeline generation
  - Wrap-up: checklist for “pipeline done right”

**In-notebook exercises (10-minute scope)**
- Pause-and-do (10): Implement `make_data_report(df)` (types, missingness %, unique counts).  
- Pause-and-do (10): Create a full sklearn Pipeline and run one validation score.

**Assessments**
- Concept quiz: pipelines, fit/transform, leakage via preprocessing  
- Notebook checkpoint submission (Colab link)

**Time budget (112.5 min)**
- Videos 54 + Notebook 46 + Quiz 7.5 + Reflection 5 = 112.5

**Bibliography**
- scikit-learn User Guide: pipelines and composite estimators; ColumnTransformer; preprocessing  
- Pedregosa et al. (scikit-learn paper): estimator API conventions  
- ISLP Python labs: preprocessing patterns aligned to regression/classification

---

## Day 3 — Thu May 20  
### Train/validation/test rigor + regression metrics + baseline modeling  
**Learning objectives**
- Choose regression metrics aligned to business loss (MAE vs RMSE).
- Establish a baseline model and interpret it correctly.
- Run holdout evaluation without contaminating the test set.
- Use quick diagnostic plots to spot obvious modeling issues.
- Document evaluation decisions (metric, split, baseline, assumptions).

**Micro-videos (54 min)**
1. Concept+demo: Regression metrics (MAE/RMSE/R²) and when to use each (10)  
2. Guided practice: Compute metrics + baseline model (8)  
3. Solution: Metric interpretation + mistakes + extension: error distribution (9)  
4. Concept+demo: Holdout evaluation workflow + test set “lockbox” (10)  
5. Guided practice: Build baseline + compare to simple linear model (8)  
6. Solution: Comparison table + pitfalls + extension: residual plots (9)

**Notebook(s)**
- File: `03_regression_metrics_baselines.ipynb`  
- Sections:
  - Metrics utilities (`mae`, `rmse`)
  - Baseline predictors (mean/median)
  - Holdout evaluation template
  - Residual plots and error summary table
  - Gemini prompts: “write a clean evaluation function”
  - Wrap-up: “test lockbox” discipline

**In-notebook exercises (10-minute scope)**
- Pause-and-do (10): Write `evaluate_regression(y_true, y_pred)` returning MAE/RMSE/R².  
- Pause-and-do (10): Compare baseline vs linear regression and interpret the delta.

**Assessments**
- Concept quiz: metrics, baselines, test lockbox  
- 3-sentence evaluation note (submitted in LMS)

**Time budget (112.5 min)**
- Videos 54 + Notebook 46 + Quiz 7.5 + Reflection 5 = 112.5

**Bibliography**
- ISLP: Model Assessment and Selection (holdout/validation/test discipline)  
- ESL: test error, training error, bias–variance, evaluation framing  
- scikit-learn User Guide: regression metrics and evaluation patterns

---

## Day 4 — Fri May 21  
### Linear regression that actually works: features, interactions, diagnostics  
**Learning objectives**
- Fit and interpret linear regression in a pipeline.
- Create interaction/polynomial features responsibly.
- Diagnose underfit/overfit using validation results.
- Use residual analysis to spot nonlinearity and heteroskedasticity.
- Translate coefficients into business meaning (with caveats).

**Micro-videos (54 min)**
1. Concept+demo: Linear regression in sklearn + coefficient interpretation (10)  
2. Guided practice: Fit baseline linear model with preprocessing (8)  
3. Solution: Interpretation + mistakes (leakage, scaling, encoding) + extension (9)  
4. Concept+demo: Interactions/polynomials + when they help (10)  
5. Guided practice: Add feature transforms and re-evaluate (8)  
6. Solution: Diagnostics + extension: compare MAE vs RMSE impacts (9)

**Notebook(s)**
- File: `04_linear_features_diagnostics.ipynb`  
- Sections:
  - Pipeline baseline recap
  - Linear regression + coefficient extraction
  - Feature engineering (`PolynomialFeatures`, interactions)
  - Residual diagnostics and “what to try next”
  - Gemini prompts for feature engineering blocks

**In-notebook exercises (10-minute scope)**
- Pause-and-do (10): Add an interaction or polynomial block and measure validation change.  
- Pause-and-do (10): Write a short diagnostic conclusion (what error patterns suggest).

**Assessments**
- Concept quiz: linear regression, features, diagnostics  
- Notebook checkpoint submission (Colab link)

**Time budget (112.5 min)**
- Videos 54 + Notebook 46 + Quiz 7.5 + Reflection 5 = 112.5

**Bibliography**
- ISLP: Linear Regression (interpretation, interactions, diagnostics)  
- ESL: linear model treatment (bias–variance, residual structure)  
- scikit-learn User Guide: LinearRegression, PolynomialFeatures, pipeline patterns

---

## Day 5 — Mon May 24  
### Regularization (Ridge/Lasso) + Project proposal sprint  
**Learning objectives**
- Explain why regularization improves generalization.
- Fit Ridge/Lasso with proper scaling and CV selection.
- Interpret coefficient shrinkage and sparsity.
- Draft a project proposal with a viable dataset + target + metric + split plan.
- Use Gemini to scaffold code and then add guardrails (checks + comments).

**Micro-videos (48 min)**
1. Concept+demo: Ridge vs Lasso vs Elastic Net (intuition) (8)  
2. Guided practice: Standardize + fit Ridge with CV (7)  
3. Solution: CV results + mistakes + extension: coefficient paths (8)  
4. Concept+demo: Lasso for feature selection (what it can/can’t do) (8)  
5. Guided practice: Fit LassoCV + compare to Ridge (7)  
6. Solution: Model comparison + pitfalls + extension: stability discussion (10)

**Notebook(s)**
- File: `05_regularization_project_proposal.ipynb`  
- Sections:
  - Regularization pipeline templates
  - CV selection (`RidgeCV`, `LassoCV`)
  - Comparison table (baseline vs linear vs ridge vs lasso)
  - Project proposal builder (prompted cells)
  - Gemini prompts: “write Ridge/Lasso pipeline + report table”

**In-notebook exercises (10-minute scope)**
- Pause-and-do (10): Run RidgeCV and summarize alpha choice + validation performance.  
- Pause-and-do (10): Run LassoCV and identify top selected features (if any).

**Assessments**
- Concept quiz: regularization + CV  
- **Project Milestone 1 (due): Proposal + dataset selection**
  - 1-page proposal + dataset link + target + metric + split plan + leakage risks

**Time budget (112.5 min)**
- Videos 48 + Notebook 47 + Quiz 7.5 + Project work 10 = 112.5

**Bibliography**
- ISLP: Linear Model Selection and Regularization (ridge/lasso/elastic net)  
- ESL: shrinkage and regularization theory  
- scikit-learn User Guide: Ridge/Lasso/ElasticNet and CV variants

---

# Week 2 (Days 6–10): Classification, Metrics, Resampling, Comparison + Midterm  
**Project milestone:** Week 2 baseline due **Day 10**  
**Midterm:** Day 10 business-case strategy practicum

---

## Day 6 — Tue May 25  
### Logistic regression: probabilities, decision boundaries, and pipelines  
**Learning objectives**
- Fit logistic regression with preprocessing in a pipeline.
- Interpret probabilities vs classes (and why thresholds matter).
- Use regularization in logistic regression for stability.
- Choose an appropriate baseline for classification.
- Document the classification objective and error costs.

**Micro-videos (54 min)**
1. Concept+demo: Logistic regression: log-odds → probabilities (10)  
2. Guided practice: Fit logistic baseline pipeline (8)  
3. Solution: Interpreting output + mistakes + extension: odds ratios (9)  
4. Concept+demo: Regularized logistic regression + why scaling matters (10)  
5. Guided practice: Tune `C` quickly (validation set) (8)  
6. Solution: Comparison + pitfalls + extension: coefficient stability (9)

**Notebook(s)**
- File: `06_logistic_pipelines.ipynb`  
- Sections:
  - Classification baselines
  - Logistic regression pipeline
  - Probability outputs + thresholding intro
  - Gemini prompts for clean pipeline + reporting

**In-notebook exercises (10-minute scope)**
- Pause-and-do (10): Build logistic pipeline and compute validation accuracy + log loss.  
- Pause-and-do (10): Change threshold from 0.5 and observe metric shifts.

**Assessments**
- Concept quiz: logistic regression, probabilities, thresholds  
- Notebook checkpoint submission (Colab link)

**Time budget (112.5 min)**
- Videos 54 + Notebook 46 + Quiz 7.5 + Reflection 5 = 112.5

**Bibliography**
- ISLP: Classification (logistic regression fundamentals)  
- ESL: logistic regression/classification foundations  
- scikit-learn User Guide: LogisticRegression, probability outputs, regularization, pipelines

---

## Day 7 — Wed May 26  
### Classification metrics: confusion matrix, ROC/PR, calibration, and business costs  
**Learning objectives**
- Compute and interpret precision, recall, F1, ROC-AUC, PR-AUC.
- Select thresholds based on business cost tradeoffs.
- Understand calibration and when it matters.
- Handle class imbalance at the evaluation level (metrics first).
- Produce a metrics dashboard table for model comparison.

**Micro-videos (54 min)**
1. Concept+demo: Confusion matrix + precision/recall tradeoffs (10)  
2. Guided practice: Compute full metric set from predicted probabilities (8)  
3. Solution: Common metric mistakes + extension: PR curves for imbalance (9)  
4. Concept+demo: Thresholding via cost (expected cost framework) (10)  
5. Guided practice: Choose an “optimal” threshold for a given cost matrix (8)  
6. Solution: Cost-based thresholding + pitfalls + extension: calibration curves (9)

**Notebook(s)**
- File: `07_classification_metrics_thresholding.ipynb`  
- Sections:
  - Metric functions + ROC/PR plotting
  - Threshold sweep table
  - Cost-based threshold selection
  - Optional: calibration plot preview

**In-notebook exercises (10-minute scope)**
- Pause-and-do (10): Build a threshold sweep and pick a threshold by business cost.  
- Pause-and-do (10): Explain why accuracy fails under imbalance (with evidence).

**Assessments**
- Concept quiz: metrics, ROC/PR, calibration concepts  
- Short deliverable: threshold recommendation (1 paragraph)

**Time budget (112.5 min)**
- Videos 54 + Notebook 46 + Quiz 7.5 + Reflection 5 = 112.5

**Bibliography**
- Fawcett: “An introduction to ROC analysis”  
- Saito & Rehmsmeier: PR curves under class imbalance  
- scikit-learn User Guide: classification metrics + ROC/PR tooling  
- Optional calibration: Niculescu-Mizil & Caruana; Zadrozny & Elkan

---

## Day 8 — Thu May 27  
### Resampling and CV: how to compare models without fooling yourself  
**Learning objectives**
- Run k-fold cross-validation for classification and regression.
- Use stratified CV for classification.
- Understand variance of performance estimates (why one split is fragile).
- Compare models using consistent CV and a single primary metric.
- Build a reusable CV evaluation function (project-ready).

**Micro-videos (54 min)**
1. Concept+demo: Why CV exists (variance, stability, fair comparison) (10)  
2. Guided practice: Implement StratifiedKFold + cross_validate (8)  
3. Solution: CV summary + pitfalls + extension: repeated CV (9)  
4. Concept+demo: Model comparison under CV (what is “fair”) (10)  
5. Guided practice: Compare two pipelines with identical CV (8)  
6. Solution: Comparison table + extension: repeated runs for stability (9)

**Notebook(s)**
- File: `08_cross_validation_model_comparison.ipynb`  
- Sections:
  - CV utilities (`cross_validate`, scoring)
  - StratifiedKFold template
  - Comparison report function (mean, std, runtime)
  - Gemini prompts for reusable `compare_models()`

**In-notebook exercises (10-minute scope)**
- Pause-and-do (10): Write `cv_report(model, X, y, cv, scoring)` returning mean/std.  
- Pause-and-do (10): Compare logistic vs regularized logistic under the same CV.

**Assessments**
- Concept quiz: CV logic, stratification, comparison discipline  
- Notebook checkpoint submission

**Time budget (112.5 min)**
- Videos 54 + Notebook 46 + Quiz 7.5 + Reflection 5 = 112.5

**Bibliography**
- ISLP: Model Assessment and Selection (k-fold CV, resampling concepts)  
- ESL: resampling theory and selection bias  
- scikit-learn User Guide: cross-validation utilities and scoring

---

## Day 9 — Fri May 28  
### Feature engineering + model selection workflow (and Project baseline build)  
**Learning objectives**
- Engineer features with pipelines without leakage.
- Use `GridSearchCV` / `RandomizedSearchCV` for systematic tuning.
- Define a project-grade evaluation plan (metric + split/CV + baseline + reporting).
- Produce a baseline model notebook that can be extended.
- Use Gemini to draft search grids and then simplify them.

**Micro-videos (48 min)**
1. Concept+demo: Feature engineering inside pipelines (safe patterns) (8)  
2. Guided practice: Add engineered features and re-run CV (7)  
3. Solution: Results + mistakes + extension: ablation mindset (8)  
4. Concept+demo: Intro to GridSearchCV (what it really does) (8)  
5. Guided practice: Run a small grid and collect best params (7)  
6. Solution: Tuning pitfalls + extension: runtime controls (10)

**Notebook(s)**
- File: `09_tuning_feature_engineering_project_baseline.ipynb`  
- Sections:
  - Pipeline + feature blocks
  - Minimal GridSearch template
  - Reporting: baseline vs tuned model table
  - Project baseline notebook scaffold (copy into project repo)

**In-notebook exercises (10-minute scope)**
- Pause-and-do (10): Build a small grid (2–3 params) and report best CV score.  
- Pause-and-do (10): Create a baseline report table suitable for the project.

**Assessments**
- Concept quiz: pipelines + tuning fundamentals  
- Project checkpoint: draft baseline notebook link

**Time budget (112.5 min)**
- Videos 48 + Notebook 47 + Quiz 7.5 + Project work 10 = 112.5

**Bibliography**
- ISLP Python labs: feature engineering examples aligned to course datasets  
- scikit-learn User Guide: grid search; randomized search; pipeline parameter tuning  
- Provost & Fawcett: evaluation and business framing of predictive tasks

---

## Day 10 — Mon May 31  
### Midterm: Business-case predictive strategy practicum + Project baseline submission  
**Learning objectives**
- Translate business cases into predictive tasks (target, unit, horizon, KPI).
- Select split strategy and metrics aligned to case and cost structure.
- Identify leakage risks and data availability constraints.
- Propose a modeling shortlist and an evaluation plan.
- Deliver a baseline model + evaluation plan for the course project.

**Micro-videos (30 min; 6×5 min)**
1. Case 1 briefing + what a “good plan” looks like (5)  
2. Guided practice: Case 1 plan build instructions (5)  
3. Debrief: Case 1 rubric + common mistakes + extensions (5)  
4. Case 2 briefing + framing templates (5)  
5. Guided practice: Case 2 (and optional Case 3) execution checklist (5)  
6. Debrief: scoring rubric + pitfalls + “how to earn full credit” (5)

**Notebook(s)**
- File: `10_midterm_casebook.ipynb`  
- Sections:
  - Integrity + allowed resources + Gemini usage boundaries (explain/verify)
  - Case 1 prompt + structured response cells
  - Case 2 prompt + structured response cells
  - Optional mini-case 3
  - Submission checklist (self-audit)

**In-notebook exercises (10-minute scope)**
- Pause-and-do (10): Case 1 plan (split + metric + leakage risks + model shortlist).  
- Pause-and-do (10): Case 2 evaluation plan + error-cost logic.  
- Pause-and-do (10): Mini-case strategy under constraints.

**Assessments**
- **Midterm submission (graded):** completed notebook (strategy + minimal prototype code where requested)  
- **Project Milestone 2 (due): Baseline model + evaluation plan**
  - baseline pipeline + metric + split/CV design + baseline report table

**Time budget (112.5 min)**
- Videos 30 + Midterm notebook work 60 + Project baseline finalization 15 + Concept check 7.5 = 112.5

**Bibliography**
- Provost & Fawcett: end-to-end predictive modeling process and business framing  
- ISLP: assessment/selection + classification/regression chapters as reference  
- scikit-learn User Guide: common pitfalls (especially leakage and improper evaluation)

---

# Week 3 (Days 11–15): Trees, Ensembles, Tuning, Interpretation  
**Project milestone:** Week 3 improved model due **Day 15**

---

## Day 11 — Tue June 1  
### Decision trees: interpretable models with sharp edges  
**Learning objectives**
- Fit decision trees for regression/classification.
- Control complexity (depth, min samples) to manage overfitting.
- Interpret tree structure and failure modes.
- Compare tree vs linear/logistic baselines under CV.
- Document “when a tree is the right tool.”

**Micro-videos (54 min)**
1. Concept+demo: Trees intuition + key hyperparameters (10)  
2. Guided practice: Fit a tree + visualize + baseline compare (8)  
3. Solution: Overfitting patterns + mistakes + extension: cost-complexity pruning (9)  
4. Concept+demo: Tree evaluation under CV + stability concerns (10)  
5. Guided practice: Small grid over depth/min_samples (8)  
6. Solution: Tuning result + extension: sensitivity analysis (9)

**Notebook(s)**
- File: `11_decision_trees.ipynb`  
- Sections:
  - Tree fit + visualization
  - Hyperparameter effects (depth sweep)
  - CV comparison table
  - Gemini prompts: “generate a clean depth sweep block”

**In-notebook exercises (10-minute scope)**
- Pause-and-do (10): Run a depth sweep and choose depth based on CV.  
- Pause-and-do (10): Write 3 observed tree failure modes (with evidence).

**Assessments**
- Concept quiz: tree mechanics + overfitting  
- Notebook checkpoint submission

**Time budget (112.5 min)**
- Videos 54 + Notebook 46 + Quiz 7.5 + Reflection 5 = 112.5

**Bibliography**
- ISLP: Tree-Based Methods (trees, pruning)  
- ESL: CART foundations and complexity control  
- scikit-learn User Guide: DecisionTreeClassifier/Regressor parameters and inspection tools

---

## Day 12 — Wed June 2  
### Random forests: bagging, OOB intuition, and feature importance  
**Learning objectives**
- Explain bagging and why forests reduce variance.
- Train a random forest and tune the most impactful knobs.
- Use permutation importance responsibly.
- Compare forest vs tree vs linear/logistic baselines.
- Produce project-ready model comparison tables.

**Micro-videos (54 min)**
1. Concept+demo: Bagging → random forests (why it works) (10)  
2. Guided practice: Fit a forest + baseline compare (8)  
3. Solution: Mistakes + extension: OOB vs CV discussion (9)  
4. Concept+demo: Permutation importance (what it means / doesn’t) (10)  
5. Guided practice: Compute importance + sanity checks (8)  
6. Solution: Interpretation pitfalls + extension: grouped features (9)

**Notebook(s)**
- File: `12_random_forests_importance.ipynb`  
- Sections:
  - Forest training + CV comparison
  - Permutation importance + plot
  - Reporting template (model table + narrative bullets)
  - Gemini prompts: “importance + report block”

**In-notebook exercises (10-minute scope)**
- Pause-and-do (10): Tune `n_estimators` and `max_features` minimally and report effects.  
- Pause-and-do (10): Compute permutation importance and write 3 interpretation bullets.

**Assessments**
- Concept quiz: bagging/forests + importance  
- Notebook checkpoint submission

**Time budget (112.5 min)**
- Videos 54 + Notebook 46 + Quiz 7.5 + Reflection 5 = 112.5

**Bibliography**
- Breiman: “Random Forests”  
- ISLP: Tree-Based Methods (bagging/forests)  
- scikit-learn User Guide: RandomForest estimators; permutation importance and caveats

---

## Day 13 — Thu June 3  
### Gradient boosting: performance with discipline (and leakage avoidance)  
**Learning objectives**
- Explain boosting vs bagging at a high level.
- Train a gradient boosting model with sensible defaults.
- Tune learning rate / depth / estimators with runtime controls.
- Compare boosted model vs forest under consistent CV.
- Identify and control overfitting in boosting.

**Micro-videos (54 min)**
1. Concept+demo: Boosting intuition + bias/variance lens (10)  
2. Guided practice: Fit a baseline boosting model (8)  
3. Solution: Common pitfalls + extension: learning rate tradeoff (9)  
4. Concept+demo: Tuning boosting (small, smart grids) (10)  
5. Guided practice: Run a constrained randomized search (8)  
6. Solution: Result interpretation + extension: stability notes (9)

**Notebook(s)**
- File: `13_gradient_boosting.ipynb`  
- Sections:
  - Baseline GBM fit
  - Constrained tuning template
  - Comparison report (forest vs GBM)
  - Gemini prompts: constrained RandomizedSearchCV with guardrails

**In-notebook exercises (10-minute scope)**
- Pause-and-do (10): Train baseline GBM and compare against RF under CV.  
- Pause-and-do (10): Run constrained tuning and report best params + score.

**Assessments**
- Concept quiz: boosting, tuning tradeoffs  
- Notebook checkpoint submission

**Time budget (112.5 min)**
- Videos 54 + Notebook 46 + Quiz 7.5 + Reflection 5 = 112.5

**Bibliography**
- Friedman: “Greedy Function Approximation: A Gradient Boosting Machine”  
- ISLP: Tree-Based Methods (boosting overview)  
- scikit-learn User Guide: gradient boosting estimators and tuning guidance

---

## Day 14 — Fri June 4  
### Model selection and comparison: making the call like a professional  
**Learning objectives**
- Build a standardized model comparison workflow (same CV, same metric).
- Use multiple metrics without “metric shopping.”
- Select a champion model and justify it (performance, stability, interpretability, cost).
- Create a reproducible experiment log table.
- Prepare project improved-model plan for submission.

**Micro-videos (54 min)**
1. Concept+demo: Comparison protocol (what must be held constant) (10)  
2. Guided practice: Build a comparison harness (3 models, 1 function) (8)  
3. Solution: Harness review + mistakes + extension: runtime tracking (9)  
4. Concept+demo: Selecting a champion (beyond top score) (10)  
5. Guided practice: Write a decision memo from results (8)  
6. Solution: Decision memo example + extension: robustness checks (9)

**Notebook(s)**
- File: `14_model_selection_protocol.ipynb`  
- Sections:
  - Comparison harness (pipelines list → CV scores table)
  - Multi-metric reporting (primary + supporting metrics)
  - Champion selection memo scaffold
  - Gemini prompts: “generate experiment log table”

**In-notebook exercises (10-minute scope)**
- Pause-and-do (10): Implement the comparison harness for 3 candidate models.  
- Pause-and-do (10): Write a champion selection memo (5 bullets + 1 risk).

**Assessments**
- Concept quiz: selection protocol + robustness  
- Notebook checkpoint submission

**Time budget (112.5 min)**
- Videos 54 + Notebook 46 + Quiz 7.5 + Reflection 5 = 112.5

**Bibliography**
- ISLP: Model Assessment and Selection (protocols for fair comparison)  
- ESL: selection bias and repeated peeking hazards  
- scikit-learn User Guide: model evaluation + parameter tuning best practices

---

## Day 15 — Mon June 7  
### Interpretation: feature importance + partial dependence + project improved model delivery  
**Learning objectives**
- Generate model interpretation artifacts (permutation importance, PDP/ICE).
- Conduct error analysis to find systematic failure segments.
- Communicate model behavior honestly (limits, caveats, instability).
- Deliver a project improved model with interpretation and error analysis.
- Use Gemini to draft explanation text, then tighten it to evidence.

**Micro-videos (48 min)**
1. Concept+demo: Interpretation toolkit overview (importance vs PDP) (8)  
2. Guided practice: Compute permutation importance for your champion (7)  
3. Solution: Interpretation pitfalls + extension: correlated features (8)  
4. Concept+demo: Partial dependence + what it can mislead (8)  
5. Guided practice: Create PDP/ICE + do segment error analysis (7)  
6. Solution: Error analysis patterns + extension: calibration check (10)

**Notebook(s)**
- File: `15_interpretation_error_analysis_project.ipynb`  
- Sections:
  - Importance + PDP/ICE
  - Segment-level error table (by key categorical / quantile bins)
  - Interpretation narrative template (evidence-based)
  - Project Milestone 3 scaffold

**In-notebook exercises (10-minute scope)**
- Pause-and-do (10): Create permutation importance and write 3 evidence-based bullets.  
- Pause-and-do (10): Run segment error analysis and identify one failure segment.

**Assessments**
- Concept quiz: interpretation + PDP caveats  
- **Project Milestone 3 (due): Improved model + interpretation**
  - updated comparison, champion choice, importance + PDP/ICE, error segment findings

**Time budget (112.5 min)**
- Videos 48 + Notebook 47 + Quiz 7.5 + Project work 10 = 112.5

**Bibliography**
- scikit-learn User Guide: inspection tools (permutation importance, partial dependence)  
- Molnar (optional): *Interpretable Machine Learning* (global/local methods and caveats)  
- ISLP: interpretation discussions across linear and tree-based methods

---

# Week 4 (Days 16–20): Error Analysis, Fairness/Ethics, Deployment Thinking, Executive Narrative, Final Project  
**Project milestone:** Week 4 final deliverable due **Day 20**

---

## Day 16 — Tue June 8  
### Error analysis to decisions: thresholds, calibration, and KPI alignment  
**Learning objectives**
- Translate model outputs into business decisions (thresholds, costs, constraints).
- Evaluate calibration and when to calibrate probabilities.
- Compare models by decision impact, not only by AUC/accuracy.
- Produce a threshold/decision recommendation.
- Document risks and assumptions explicitly.

**Micro-videos (54 min)**
1. Concept+demo: From prediction to action (thresholds and costs) (10)  
2. Guided practice: Threshold sweep with expected cost (8)  
3. Solution: Choosing thresholds + mistakes + extension: sensitivity analysis (9)  
4. Concept+demo: Calibration intuition + reliability plots (10)  
5. Guided practice: Calibrate and compare decision impact (8)  
6. Solution: Calibration pitfalls + extension: decision policy reporting (9)

**Notebook(s)**
- File: `16_decision_thresholds_calibration.ipynb`  
- Sections:
  - Cost matrix + expected cost computation
  - Threshold sweep dashboard
  - Calibration plot + optional calibration model
  - Decision policy summary block (ready for slides)

**In-notebook exercises (10-minute scope)**
- Pause-and-do (10): Select a threshold that minimizes expected cost and justify it.  
- Pause-and-do (10): Check calibration and decide whether calibration is needed.

**Assessments**
- Concept quiz: thresholds, calibration, decision impact  
- Short deliverable: decision policy paragraph (for project slides)

**Time budget (112.5 min)**
- Videos 54 + Notebook 46 + Quiz 7.5 + Reflection 5 = 112.5

**Bibliography**
- Provost & Fawcett: decision-making with predictions and cost alignment  
- scikit-learn User Guide: thresholding and calibration tooling  
- Optional calibration: Niculescu-Mizil & Caruana; Zadrozny & Elkan

---

## Day 17 — Wed June 9  
### Fairness and ethics basics: responsible predictive analytics (minimum viable rigor)  
**Learning objectives**
- Identify fairness risks and ethical failure modes in predictive systems.
- Compute basic group fairness diagnostics (when sensitive attributes exist).
- Use slicing to detect performance disparities across segments.
- Write a model card-style limitations and responsible-use section.
- Apply responsible AI framing to the course project deliverable.

**Micro-videos (54 min)**
1. Concept+demo: Fairness vocabulary (disparity, harm, proxies, feedback loops) (10)  
2. Guided practice: Set up group slicing evaluation (8)  
3. Solution: Slicing report + mistakes + extension: intersectional slices (9)  
4. Concept+demo: Fairness metrics (selection rate, TPR/FPR gaps) + caution (10)  
5. Guided practice: Compute basic fairness diagnostics (8)  
6. Solution: Interpretation + what not to claim + extension: mitigation options (9)

**Notebook(s)**
- File: `17_fairness_slicing_model_cards.ipynb`  
- Sections:
  - Slice-based performance table
  - Optional fairness metrics (dataset permitting)
  - Model card template: intended use, limitations, risks, monitoring
  - Gemini prompts for drafting text + evidence-tightening checklist

**In-notebook exercises (10-minute scope)**
- Pause-and-do (10): Create slice performance table and highlight one disparity (if any).  
- Pause-and-do (10): Draft a model card limitations section (6–8 lines).

**Assessments**
- Concept quiz: fairness basics, responsible communication  
- Upload model card draft text (project-ready)

**Time budget (112.5 min)**
- Videos 54 + Notebook 46 + Quiz 7.5 + Reflection 5 = 112.5

**Bibliography**
- Barocas, Hardt, Narayanan: *Fairness and Machine Learning*  
- Hardt, Price, Srebro: Equality of Opportunity  
- Mitchell et al.: Model Cards for Model Reporting  
- Optional: Chouldechova (fair prediction with disparate impact)

---

## Day 18 — Thu June 10  
### Deployment thinking: reproducibility, monitoring, drift, and “don’t ship a notebook”  
**Learning objectives**
- Package a model pipeline reproducibly (single function, fixed preprocessing).
- Save/load model artifacts and ensure consistent inference.
- Define monitoring signals (data drift, performance drift, calibration drift).
- Create a minimal production checklist and risk log.
- Prepare the project notebook for executive-facing reproducibility.

**Micro-videos (54 min)**
1. Concept+demo: Reproducible pipelines (fit once, run anywhere) (10)  
2. Guided practice: Refactor notebook into functions + config block (8)  
3. Solution: Refactor review + extension: experiment config (9)  
4. Concept+demo: Monitoring + drift (what to watch and why) (10)  
5. Guided practice: Create monitoring checklist + drift proxies (8)  
6. Solution: Monitoring plan + mistakes + extension: governance (9)

**Notebook(s)**
- File: `18_reproducibility_monitoring.ipynb`  
- Sections:
  - Refactor into `train()` / `predict()` / `evaluate()`
  - Save/load via joblib
  - Monitoring plan template (tables)
  - “Ready-to-share” notebook hygiene checklist

**In-notebook exercises (10-minute scope)**
- Pause-and-do (10): Implement `train_model(config)` returning pipeline + metrics.  
- Pause-and-do (10): Draft a monitoring plan with 5–8 signals and owners.

**Assessments**
- Concept quiz: reproducibility + drift  
- Notebook checkpoint submission

**Time budget (112.5 min)**
- Videos 54 + Notebook 46 + Quiz 7.5 + Reflection 5 = 112.5

**Bibliography**
- Chip Huyen: *Designing Machine Learning Systems*  
- Optional: *Machine Learning Design Patterns* (Lakshmanan et al.)  
- Optional: *Dataset Shift in Machine Learning* (Quionero-Candela et al.)  
- Rabanser, Günnemann, Lipton: dataset shift detection (Failing Loudly)  
- scikit-learn User Guide: model persistence and reproducible pipelines

---

## Day 19 — Fri June 11  
### Executive narrative: slide-style story + conference video plan (project studio)  
**Learning objectives**
- Convert technical work into an executive-ready slide narrative.
- Build a clear “problem → approach → results → recommendation → risks” flow.
- Create credible visuals (comparison table, key plots, decision policy).
- Script a short conference video (tight, evidence-based, non-technical).
- Finalize the project deliverable package.

**Micro-videos (42 min)**
1. Concept+demo: Executive narrative structure (1-slide-per-idea) (7)  
2. Guided practice: Storyboard your deck (10-slide target) (6)  
3. Solution: Storyboard example + mistakes + extension: stakeholder tailoring (7)  
4. Concept+demo: What visuals to include (and what to avoid) (7)  
5. Guided practice: Draft speaker script + timing plan (6)  
6. Solution: Script tightening + extension: Q&A slide and limitations (9)

**Notebook(s)**
- File: `19_project_narrative_video_studio.ipynb`  
- Sections:
  - Slide outline template (markdown → slides)
  - Required visuals checklist
  - Script template (time-coded)
  - Gemini prompts: tighten script; convert findings to executive bullets

**In-notebook exercises (10-minute scope)**
- Pause-and-do (10): Create a 10-slide outline (titles + 2 bullets each).  
- Pause-and-do (10): Write a 2–3 minute script aligned to the outline.

**Assessments**
- Concept quiz: executive communication and evidence discipline  
- Project checkpoint: draft slide outline + script (links)

**Time budget (112.5 min)**
- Videos 42 + Notebook 45 + Quiz 7.5 + Project studio 18 = 112.5

**Bibliography**
- Cole Nussbaumer Knaflic: *Storytelling with Data*  
- Barbara Minto: *The Pyramid Principle*  
- Provost & Fawcett: communicating results to stakeholders (framing and impact)

---

## Day 20 — Mon June 14  
### Final delivery: project package submission + peer review + course closeout  
**Learning objectives**
- Deliver a complete end-to-end predictive analytics package.
- Produce an executive-ready deck and a conference-style video.
- Demonstrate reproducibility (run-all notebook, documented choices).
- Evaluate peers’ work using a structured rubric and provide actionable feedback.
- Write a concise postmortem: what worked, what didn’t, what you’d do next.

**Micro-videos (30 min; 6×5 min)**
1. Final submission checklist (what graders check first) (5)  
2. Guided practice: Run-all reproducibility audit (5)  
3. Solution: Common submission failures + prevention (5)  
4. Peer review rubric (how to be useful, not nice) (5)  
5. Guided practice: High-signal feedback in 5 minutes (5)  
6. Solution: Example peer review + extension: next-iteration roadmap (5)

**Notebook(s)**
- File: `20_final_submission_peer_review.ipynb`  
- Sections:
  - Final self-audit checklist (run-all, outputs, links)
  - Submission links + artifact manifest (notebook, deck, video)
  - Peer review form (rubric + comment prompts)
  - Postmortem prompts (8–10 lines)

**In-notebook exercises (10-minute scope)**
- Pause-and-do (10): Run-all audit and fix one reproducibility issue (real or simulated).  
- Pause-and-do (10): Complete one peer review with rubric scores + 3 actionable edits.

**Assessments**
- **Project Milestone 4 (due): Final model + executive-ready deliverable**
  - Final run-all notebook + model card/limitations + monitoring plan  
  - Slide narrative (deck or slide markdown)  
  - Conference-style video  
- Peer review submission (graded for quality)  
- Final concept quiz (course wrap)

**Time budget (112.5 min)**
- Videos 30 + Final submission notebook work 55 + Peer review 20 + Wrap quiz 7.5 = 112.5

**Bibliography**
- Mitchell et al.: Model Cards (responsible reporting alignment)  
- Chip Huyen: deployment checklists and monitoring as product handoff  
- Storytelling with Data / Pyramid Principle: narrative polish and reviewer-friendly structure

---

# Project bibliography (applies across all milestones)
- Provost & Fawcett: *Data Science for Business* (problem framing, value, evaluation)
- ISLP + Python labs (modeling, resampling, interpretation foundations)
- scikit-learn User Guide (pipelines, tuning, metrics, inspection)
- Mitchell et al. (Model Cards) + Barocas/Hardt/Narayanan (Fairness and ML) for limitations, risks, and responsible-use language
- Chip Huyen: *Designing Machine Learning Systems* (monitoring and deployment thinking)

