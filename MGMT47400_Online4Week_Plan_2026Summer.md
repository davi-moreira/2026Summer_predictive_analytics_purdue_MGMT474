# MGMT 47400 – Predictive Analytics (3 credits)  
## 4-Week Fully Online Course Plan (Daniels School of Business)  
**Run dates (business days):** Mon **May 18, 2026** → Fri **June 12, 2026** (20 business days)  
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

# Weekly structure, project milestones, and case competition

## Kaggle Case Competition (individual or pairs)
- **Competition:** Summer 2026 MGMT47400 Case Competition: Bank Churn
- **Task:** Predict the probability that a bank customer will churn (`Exited` = 1)
- **Metric:** AUC-ROC
- **Platform:** Kaggle (private class competition, max 5 submissions per day)
- **Deadline:** Fri June 12, 2026 at 11:59 PM (both Kaggle final submission and Brightspace code submission)
- **Brightspace deliverable:** Submit the complete code for your best-performing model. The code must be fully replicable, allowing the instructor and TA to reproduce the same results and performance metrics. Include all necessary steps: data preprocessing, feature engineering, model training, evaluation, and generation of the submission file.

## Project (single end-to-end applied project; **groups of four randomly assigned members**; progresses weekly)

Canonical reference: [`_final_project/2026Summer/final_project_milestone_reference.md`](_final_project/2026Summer/final_project_milestone_reference.md)

- **Week 1 (due Day 5): Initial Project Proposal** (prediction goal + motivation + data overview + preliminary methods + expected contributions; 1–2 pages). Detail: [`milestone_01_proposal.md`](_final_project/2026Summer/milestone_01_proposal.md)
- **Week 2 (due Day 10): Simple Model + Performance Evaluation** (dataset exploration + feature engineering + missing-value handling + baseline pipeline with k-fold CV). Detail: [`milestone_02_baseline_model.md`](_final_project/2026Summer/milestone_02_baseline_model.md)
- **Week 3 (due Day 15): More Complex Model + Hyperparameter Tuning + Draft Abstract** (~250 words). Detail: [`milestone_03_complex_model_and_abstract.md`](_final_project/2026Summer/milestone_03_complex_model_and_abstract.md)
- **Week 4 (due Day 20): Final Research Poster + intra-group Peer Evaluation** (single PDF named `<group-number>.pdf`; optional Fall 2026 Purdue Undergraduate Research Conference presentation strongly encouraged). Detail: [`milestone_04_final_poster.md`](_final_project/2026Summer/milestone_04_final_poster.md)

## Grading

| Assessment | Weight |
|---|---:|
| Participation | 10% |
| Daily Concept Quizzes | 15% |
| Midterm (Business Case Practicum) | 20% |
| Kaggle Case Competition | 20% |
| Final Project + Milestones | 35% |

**Kaggle Case Competition (20%):**
- At least one Kaggle submission: 30% of competition grade (6% of total)
- Leaderboard ranking: 70% of competition grade (14% of total)

**Final Project + Milestones (35%):**
- Milestone Deliverables (M1–M4): 40% of project grade (14% of total)
- Peer Evaluation (intra-group, confidential): 20% of project grade (7% of total)
- Instructor / TA Evaluation (final research poster): 40% of project grade (14% of total)

---

## Notebook Sequence Rationale

The 20 notebooks follow a deliberate pedagogical progression: each notebook builds exactly one conceptual layer, assumes only what prior notebooks have taught, and prepares exactly what the next notebook needs. The sequence is organized into four weekly arcs, each culminating in a project milestone that forces integration of that week's skills.

### Sequencing Table

| NB | Title | Why It Exists | Why This Position |
|----|-------|---------------|-------------------|
| 00 | Launchpad: Course Setup | Pre-course orientation — orients students to the platform (Colab, Gemini) and course logistics (syllabus, grading, daily workflow) so nb01 can focus purely on analytics content. | Day 0 (pre-course); no predecessor. Students cannot engage with any technical content until they understand the platform and AI assistant policy. |
| 01 | EDA & Splits | Conceptual foundation — introduces predictive analytics (Y = f(X) + ε), the EDA checklist, and the data workflow (60/20/20 splitting, leakage prevention) that every subsequent notebook depends on. | Follows nb00 (platform ready). Students cannot preprocess, model, or evaluate anything until they understand leakage and data splitting. |
| 02 | Preprocessing Pipelines | Operationalizes leakage prevention from nb01 by teaching Pipeline + ColumnTransformer — the tool that makes safe preprocessing automatic and reproducible. | nb01 provides the vocabulary (split, leakage, EDA); nb02 gives the tool that enforces it. nb03 assumes the pipeline is a solved problem. |
| 03 | Regression Metrics & Baselines | Teaches formal regression metrics (MAE, RMSE, R²) and baseline models, giving every future comparison a meaningful performance floor. | nb02 solves preprocessing; nb03 shifts focus to evaluation. nb04 needs metrics to measure whether feature engineering helps. |
| 04 | Linear Features & Diagnostics | Teaches feature engineering (interactions, polynomials) and residual diagnostics — revealing the accuracy vs. complexity tradeoff and exposing overfitting risk. | nb03 provides the evaluation framework; without it, students would engineer features blindly. nb04 creates the overfitting problem that nb05 solves. |
| 05 | Regularization (Ridge/Lasso) | Introduces regularization as the direct solution to nb04's overfitting problem. Closes the Week 1 regression arc and hosts the project proposal milestone. | nb04 creates the problem (polynomial explosion, unstable coefficients); nb05 delivers the solution. Completes the regression toolkit before the Week 2 pivot to classification. |
| 06 | Logistic Regression & Pipelines | Marks the transition from regression to classification, teaching predicted probabilities, threshold sensitivity, and pipeline reuse in a classification context. | nb05 introduces regularization via alpha; nb06 applies the same idea via C in classification, reusing the Pipeline pattern for a seamless transition. nb07 needs probability foundations. |
| 07 | Classification Metrics & Thresholding | Builds the complete classification evaluation toolkit — precision, recall, F1, ROC/PR curves, and cost-based threshold selection. Calibration is deferred to nb16 where it naturally attaches to tree-based (often miscalibrated) classifiers. | nb06 introduces probabilities and confusion matrices informally; nb07 formalizes them. nb08 needs metric vocabulary to choose a `scoring` parameter for CV. |
| 08 | Cross-Validation & Model Comparison | Teaches reliable, low-variance performance estimation through k-fold CV, replacing the fragile single train/val split with a systematic evaluation framework. | nb07 provides the metrics nb08 passes as `scoring`. nb09 embeds CV inside grid search; students must understand standalone CV first. |
| 09 | Hyperparameter Tuning + Feature Engineering + Leakage Detection | Single-file three-section notebook with an opening **toolkit-closer banner** (cell 1, before the Learning Objectives). Section A turns nb08's CV ritual into `GridSearchCV` / `RandomizedSearchCV`, reading `cv_results_` through the CI-overlap rule (includes a one-paragraph `C`-parameter primer in Section 1.4). Section B introduces `ColumnTransformer` on a synthetic TechCorp Talent Analytics case (first dataset in the course with real categorical columns, including a high-cardinality `manager_id`), adds `FunctionTransformer` for domain features, and stages two leakage case studies (target-encoding in the main demo, `SelectKBest`-outside-pipeline in the pause-and-do). **Section C — Toolkit Recap** consolidates the full mid-course toolkit (concepts, workflow, sklearn primitives, decision rules) into a one-page reference. | nb08 teaches standalone CV; nb09 embeds it inside grid search. nb10 (midterm) requires the full pipeline template from nb09 — Section C's recap is the natural reference for the casebook's strategic-reasoning prompts. The leakage case studies become the prerequisite for nb13's "leaky features dominate boosting" callout. |
| 10 | Midterm Casebook | Week 2 capstone — tests strategic reasoning (target, metric, split, leakage risks) across business cases. Hosts the project baseline milestone. Includes a **one-page cheat-sheet appendix** with decision tables for metric/scaler/stratify choice, Ridge vs Lasso tie-breakers, CI-overlap rule, and leakage checklist. | nb09 completes the toolkit; nb10 tests whether students can wield it strategically. Creates a natural pause before the Week 3 tree-based methods arc. |
| 11 | Decision Trees | Introduces the first non-linear model family (CART), teaching the bias-variance tradeoff concretely through depth sweeps and overfitting demonstrations. Includes a class-imbalance section with `class_weight='balanced'` on artificially down-sampled data and an explicit anti-SMOTE warning. | nb10 consolidates Weeks 1–2; students enter nb11 with solid evaluation skills and can focus on tree mechanics. nb12 solves the single tree's high-variance problem. |
| 12 | Random Forests & Importance | Solves the single tree's instability through bagging + random feature subsets. Introduces permutation importance and OOB scores. Opens with a **four-method feature-importance reconciliation table** (coefficient magnitude / impurity / permutation / PDP) that becomes the course-wide reference through nb15. | nb11 proves single trees overfit; nb12's motivation ("average many unstable trees") only makes sense after experiencing that instability. nb13 needs bagging as a contrast for boosting. |
| 13 | Gradient Boosting | Completes the ensemble trilogy — sequential error correction that often achieves the highest tabular accuracy but requires careful tuning discipline. Closes with a **"leaky features dominate the top" callout** connecting nb09's leakage case studies to boosting's amplification effect on any leaked feature. | nb12 establishes the parallel ensemble baseline (bagging reduces variance); nb13 contrasts with sequential approach (boosting reduces bias). nb14 needs the full candidate roster. |
| 14 | Model Selection Protocol + Test Set Opening Ceremony | Replaces informal "pick the highest number" comparison with a structured, fair, reproducible protocol — identical CV folds, primary metric. Explicitly opens the locked test set exactly once, computes Student's *t* 95% CI on the champion's CV scores, and delivers an INSIDE / ABOVE / BELOW verdict using nb08's vocabulary — the payoff for eight notebooks of locking discipline. | nb13 completes the candidate pool (logistic, tree, RF, GBM); a formal protocol would be premature without all candidates. nb15 interprets the selected champion. |
| 15 | Interpretation & Error Analysis (Project Improved Model) | Answers "what is the champion learning and where does it fail?" via permutation importance, PDP/ICE, and segment-level error analysis. Opens with an explicit cross-reference to nb12's four-method importance table, positioning PDP/ICE as the "shape" complement to nb12's three "rank" methods. Hosts the improved model milestone. | nb14 selects the champion; interpretation is only meaningful after commitment to one model. nb16 uses error analysis to motivate threshold adjustments. |
| 16 | Probability Calibration for Decision Quality | Pivots from nb07's threshold-tuning content (now a 5-minute refresh) to calibration as the main focus: reliability diagrams, Brier score, `CalibratedClassifierCV` with isotonic vs. sigmoid, and a concrete demonstration on a Random Forest (which is typically miscalibrated). Explains when calibration matters (action decisions) and when it does not (ranking decisions — AUC is invariant under calibration). | nb15 reveals failure segments; nb16 asks whether the champion's probabilities are trustworthy enough to inform business decisions. nb17 needs calibration-aware thresholds to analyze fairness implications. |
| 17 | Fairness & Model Cards | Teaches that excellent aggregate metrics can still harm specific groups. Introduces slice-based evaluation, fairness diagnostics, and model card documentation. | nb16 teaches threshold setting; nb17 asks whether that threshold is fair across groups. nb18 needs fairness signals for its monitoring plan. |
| 18 | Reproducibility & Monitoring | Transitions from "works in a notebook" to "can be saved, loaded, verified, and monitored" through function refactoring, joblib serialization, monitoring plans, and a **Kaggle submission mechanics section** (load saved pipeline → predict on held-out CSV → produce `submission.csv` with exact column names). | nb17 establishes the ethical layer; nb18 adds the operational layer and the last-mile Kaggle glue needed for the Day 20 competition deadline. Together they form the pre-deployment checklist. nb19 needs artifacts and vocabulary for the executive narrative. |
| 19 | Elements of Data Communication & Poster Design | Walks the **six principles** of data communication (context, visualization, less-is-more / data-ink ratio, hierarchy, beauty, story) and applies them to the **eleven-section research-poster architecture** of the Purdue Undergraduate Research Conference template. Includes a chart-audit exercise on a project figure and an outline-plus-abstract drafting exercise for the M4 poster. | nb18 provides reproducible artifacts and the headline numbers (CV-CI, locked-test verdict) that the poster has to communicate; without them, the design lecture would lack a payload. nb20 requires the poster outline + abstract drafted here. |
| 20 | Final Submission & Peer Review | Capstone — self-audit, submit complete deliverable package, peer review using structured rubric, and postmortem reflection. | nb19 develops deliverables; nb20 audits and submits them. Closes the course arc from nb01's first data split to a fully reviewed and reflected-upon submission. |

### Weekly Arc Dependencies

```
Pre-course — ORIENTATION
  00 Launchpad/Setup
  (Platform fluency)

Week 1 — REGRESSION ARC
  01 EDA/Splits → 02 Pipelines → 03 Metrics/Baselines → 04 Features/Diagnostics → 05 Regularization
  (Foundation)    (Tool)         (Measurement)          (Improvement)             (Control + Proposal)

Week 2 — CLASSIFICATION ARC
  06 LogReg → 07 Classification Metrics → 08 Cross-Validation → 09 Tuning+FE+Leakage → 10 Midterm
  (New task)   (New metrics)               (Reliable comparison) (Integration + leak detection) (Assessment + Baseline + Cheat Sheet)

Week 3 — ENSEMBLES ARC
  11 Trees → 12 Random Forests → 13 Gradient Boosting → 14 Selection + Test Set Ceremony → 15 Interpretation
  (Non-linear + class_weight) (Bagging + Importance Table) (Boosting + Leakage Callout) (Fair protocol + open the test set)  (Explain + Improved Model)

Week 4 — PRODUCTION ARC
  16 Calibration → 17 Fairness → 18 Deployment + Kaggle Submission → 19 Narrative → 20 Final Submission
  (Trustworthy probabilities) (Ethics) (Operations + last-mile submission) (Communication) (Audit + Review + Reflection)
```

Each week follows the same pattern: introduce a new capability, build evaluation skills, practice integration, then deliver a milestone. The dependency arrows within each week are strict — no notebook can be skipped without breaking the next one's assumptions.

---

# Week 1 (Days 1–5): Foundations, EDA, Splits, Linear Regression, Regularization  
**Project milestone:** Week 1 proposal due **Day 5**

---

## Day 1 — Mon May 18  
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
- File: `nb01_launchpad_eda_splits.ipynb`  
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

## Day 2 — Tue May 19  
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
- File: `nb02_preprocessing_pipelines.ipynb`  
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
- Participation: notebook submission with completed exercises (Colab link)

**Time budget (112.5 min)**
- Videos 54 + Notebook 46 + Quiz 7.5 + Reflection 5 = 112.5

**Bibliography**
- scikit-learn User Guide: pipelines and composite estimators; ColumnTransformer; preprocessing  
- Pedregosa et al. (scikit-learn paper): estimator API conventions  
- ISLP Python labs: preprocessing patterns aligned to regression/classification

---

## Day 3 — Wed May 20  
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
- File: `nb03_regression_metrics_baselines.ipynb`  
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

## Day 4 — Thu May 21  
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
- File: `nb04_linear_features_diagnostics.ipynb`  
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
- Participation: notebook submission with completed exercises (Colab link)

**Time budget (112.5 min)**
- Videos 54 + Notebook 46 + Quiz 7.5 + Reflection 5 = 112.5

**Bibliography**
- ISLP: Linear Regression (interpretation, interactions, diagnostics)  
- ESL: linear model treatment (bias–variance, residual structure)  
- scikit-learn User Guide: LinearRegression, PolynomialFeatures, pipeline patterns

---

## Day 5 — Fri May 22  
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
- File: `nb05_regularization_project_proposal.ipynb`  
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
- **Project Milestone 1 (due): Initial Project Proposal**
  - 1–2 pages: prediction goal + motivation + data overview + preliminary methods + expected contributions
  - Detail: [`_final_project/2026Summer/milestone_01_proposal.md`](_final_project/2026Summer/milestone_01_proposal.md)

**Time budget (async: 112.5 min)**
- Videos 48 + Notebook 47 + Quiz 7.5 + Project work 10 = 112.5

**Synchronous session plan (112.5 min, recorded)**
Pre-recorded micro-videos are available for students to watch before or after the session.

| Block | Duration | Content |
|-------|----------|---------|
| Welcome + Week 1 Recap | 10 min | Review Days 1-4 key concepts, address common questions from async work |
| Live Recap & Demo: Regularization | 15 min | Condensed highlights from videos + live Colab demo reinforcing key ideas |
| PAUSE-AND-DO (live) | 20 min | Students run RidgeCV/LassoCV with instructor available for help |
| Break | 5 min | |
| Project Discussion | 25 min | Milestone 1 review (proposals due today), dataset selection tips, Milestone 2 preview and expectations |
| Kaggle Competition Launch | 20 min | Join competition walkthrough, explore data, submission format demo, pair formation |
| Course Q&A + Quiz | 17.5 min | Week 1 doubts, logistics, concept quiz |

**Bibliography**
- ISLP: Linear Model Selection and Regularization (ridge/lasso/elastic net)  
- ESL: shrinkage and regularization theory  
- scikit-learn User Guide: Ridge/Lasso/ElasticNet and CV variants

---

# Week 2 (Days 6–10): Classification, Metrics, Resampling, Comparison + Midterm  
**Project milestone:** Week 2 baseline due **Day 10**  
**Midterm:** Day 10 business-case strategy practicum

---

## Day 6 — Mon May 25  
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
- File: `nb06_logistic_pipelines.ipynb`  
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
- Participation: notebook submission with completed exercises (Colab link)

**Time budget (112.5 min)**
- Videos 54 + Notebook 46 + Quiz 7.5 + Reflection 5 = 112.5

**Bibliography**
- ISLP: Classification (logistic regression fundamentals)  
- ESL: logistic regression/classification foundations  
- scikit-learn User Guide: LogisticRegression, probability outputs, regularization, pipelines

---

## Day 7 — Tue May 26  
### Classification metrics: confusion matrix, ROC/PR, and business costs  
**Learning objectives**
- Compute and interpret precision, recall, F1, ROC-AUC, PR-AUC.
- Select thresholds based on business cost tradeoffs.
- Handle class imbalance at the evaluation level (metrics first).
- Produce a metrics dashboard table for model comparison.

*(Calibration is deferred to nb16 — Decision Thresholds & Calibration — where students have already met classifiers, such as random forests and gradient boosting, that can actually be miscalibrated. Logistic regression is natively well-calibrated by its loss function, so covering calibration in Week 2 has no natural pain point to anchor it.)*

**Micro-videos (54 min)**
1. Concept+demo: Confusion matrix + precision/recall tradeoffs (10)  
2. Guided practice: Compute full metric set from predicted probabilities (8)  
3. Solution: Common metric mistakes + extension: PR curves for imbalance (9)  
4. Concept+demo: Thresholding via cost (expected cost framework) (10)  
5. Guided practice: Choose an “optimal” threshold for a given cost matrix (8)  
6. Solution: Cost-based thresholding + pitfalls + extension: metrics dashboard as a reusable evaluation artifact (9)

**Notebook(s)**
- File: `nb07_classification_metrics_thresholding.ipynb`  
- Sections:
  - Question-first metric framework (Precision / Recall / F1 / Accuracy paired with the business question each answers)
  - ROC curve and AUC
  - PR curve and Average Precision
  - Threshold sweep + cost-based threshold selection
  - Accuracy paradox under extreme imbalance (95/5 synthetic dataset)

**In-notebook exercises (10-minute scope)**
- Pause-and-do (10): Build a threshold sweep and pick a threshold by business cost.  
- Pause-and-do (10): Explain why accuracy fails under imbalance (with evidence).

**Assessments**
- Concept quiz: metrics, ROC/PR, cost-based thresholding concepts  
- Short deliverable: threshold recommendation (1 paragraph)

**Time budget (112.5 min)**
- Videos 54 + Notebook 46 + Quiz 7.5 + Reflection 5 = 112.5

**Bibliography**
- Fawcett: “An introduction to ROC analysis”  
- Saito & Rehmsmeier: PR curves under class imbalance  
- scikit-learn User Guide: classification metrics + ROC/PR tooling
- (Calibration bibliography — Niculescu-Mizil & Caruana; Zadrozny & Elkan — moves to nb16's reading list.)

---

## Day 8 — Wed May 27  
### Resampling and CV: mean, SD, and 95% CI for both business cases  
**Learning objectives**
- Write the k-fold CV estimator for both regression (MSE) and classification (misclassification / score-based).
- Run 5-fold CV on the California Housing regression case and the breast cancer classification case, reporting mean, standard deviation, and a 95% confidence interval every time.
- Plot per-fold CV scores with the mean and CI, and compare against a single validation-set score using a second bar plot.
- Interpret whether a single validation score was lucky, unlucky, or representative based on whether it falls inside the CV 95% CI.
- Use the 95% CI overlap rule to decide whether one model (or hyperparameter choice) is convincingly better than another on the same task.

**Micro-videos (54 min)**
1. Concept+demo: Why one split is fragile — distribution of scores, not a single number (10)  
2. Concept+demo: The k-fold CV estimator for regression and classification (equations + stratification) (10)  
3. Guided practice: Implement k-fold CV with mean, SD, and Student's-t 95% CI on California Housing (Ridge) (8)  
4. Solution: Interpret the per-fold bar plot + single-split vs. CV comparison plot (9)  
5. Guided practice: Repeat the recipe with StratifiedKFold on the breast cancer data (LogReg, ROC-AUC) (8)  
6. Solution: Interpret the classification comparison plot + extension: Ridge vs. OLS CI-overlap test (9)

**Notebook(s)**
- File: `nb08_cross_validation_model_comparison.ipynb`  
- Sections:
  - Why CV exists (k-fold estimator for regression and classification, plus mean/SD/95% CI formulas)
  - K-fold CV for regression — California Housing (per-fold plot + single-split vs. CV comparison + interpretation)
  - Stratified k-fold CV for classification — Breast Cancer (same recipe)
  - Pause-and-do 1: Ridge vs. OLS CI-overlap test on California Housing
  - Pause-and-do 2: LogReg (C=1.0) vs. LogReg (C=0.01) CI-overlap test on Breast Cancer

**In-notebook exercises (10-minute scope)**
- Pause-and-do (10): Ridge (α=1.0) vs. plain OLS on California Housing — run 5-fold CV for both and decide whether their 95% CIs overlap; use that to defend or reject regularization to the CFO.
- Pause-and-do (10): LogReg (C=1.0) vs. LogReg (C=0.01) on Breast Cancer — run 5-fold stratified CV for both and decide whether their 95% CIs overlap; use that to judge whether regularization strength meaningfully moves MedScreen's ROC-AUC (previewing nb09's GridSearchCV).

**Assessments**
- Concept quiz: CV estimator, stratification, confidence-interval reporting  
- Participation: notebook submission with completed exercise

**Time budget (112.5 min)**
- Videos 54 + Notebook 46 + Quiz 7.5 + Reflection 5 = 112.5

**Bibliography**
- ISLP: Model Assessment and Selection (k-fold CV equations 5.3 and 5.4)  
- ESL: resampling theory and selection bias  
- scikit-learn User Guide: cross-validation utilities and scoring

---

## Day 9 — Thu May 28  
### Hyperparameter tuning + feature engineering + leakage detection  
**Learning objectives**
- Run `GridSearchCV` and `RandomizedSearchCV` on known models and read `cv_results_` as a table of nb08-style CV runs.
- Apply the 95% CI overlap rule from nb08 to pick the simplest model among the top candidates in `cv_results_`.
- Build a `ColumnTransformer` that handles both categorical and numeric features inside a single `Pipeline`.
- Use `FunctionTransformer` to embed domain feature engineering inside the pipeline without leakage.
- Detect a data-leakage bug in a provided pipeline by comparing CV scores before and after the fix.
- Explain why every feature-engineering step must live inside the pipeline that `cross_val_score` or `GridSearchCV` evaluates.

**Micro-videos (60 min)**
1. Concept+demo: From one CV run to a grid — GridSearchCV intuition (7)  
2. Guided practice: GridSearchCV on Ridge + reading `cv_results_` (8)  
3. Solution: CI-overlap rule on the top rows of `cv_results_` + RandomizedSearchCV for large grids (8)  
4. Concept+demo: TechCorp Talent Analytics case — `ColumnTransformer` on real categorical columns + `handle_unknown='ignore'` (8)  
5. Guided practice: Build the TechCorp pipeline end-to-end + `FunctionTransformer` for domain ratios (7)  
6. Concept+demo: The leakage trap — target encoding on full data inflates CV, the Kaggle classic (8)  
7. Solution: Fix and re-run, observe the score drop to reality (7)  
8. Solution: PAUSE-AND-DO 2 walkthrough — SelectKBest outside pipeline, same leak pattern different flavor (7)

**Notebook(s)**
- File: `nb09_tuning_feature_engineering_project_baseline_student.ipynb`  
- Structure: a single file with an opening **toolkit-closer banner** (cell 1, before the Learning Objectives — flags nb09 as the last "new tools" notebook of the mid-course arc) plus three big sections
  - **Section A — Grid search as nb08 × a grid**: `GridSearchCV` on Ridge α-grid (California Housing) + `RandomizedSearchCV` on LogReg `C` distribution (Breast Cancer, with a one-paragraph `C`-parameter primer); CI-overlap rule applied to `cv_results_` top rows; champion selection pattern. PAUSE-AND-DO 1: `GridSearchCV` on LogReg `C` grid with CI-overlap verdict.
  - **Section B — Feature engineering + leakage detection + categoricals**: TechCorp Talent Analytics synthetic business case (2,000 employees, 5 numeric + 3 low-card categorical + `manager_id` high-cardinality + 30 HRIS noise metrics); leak-free baseline with `ColumnTransformer` + `OneHotEncoder(handle_unknown='ignore')`; domain feature via `FunctionTransformer`; intern's dramatic target-encoding leak and its fix. PAUSE-AND-DO 2: detect and fix a SelectKBest-outside-pipeline leak.
  - **Section C — Toolkit Recap — What You Hold After nb01–nb09**: a one-page consolidated reference for the full mid-course toolkit. Four subsections — concepts (bias–variance, overfitting/underfitting, curse of dimensionality, leakage, regression vs classification), workflow (EDA → split → pipeline → evaluate → CI-overlap), tools (sklearn-primitives table by layer with notebook anchors), and decision rules (when to scale, when to stratify, metric choice from cost asymmetry, Ridge vs Lasso, what `C` means, CI-overlap rule, leakage rule). Closes the notebook before the wrap-up.

**In-notebook exercises (10-minute scope)**
- Pause-and-do (10): `GridSearchCV` on LogisticRegression `C` grid (MedScreen); apply CI-overlap rule to pick a simpler champion than `best_params_`.
- Pause-and-do (10): Find and fix the SelectKBest-outside-pipeline leak in a provided snippet; compare leaky vs leak-free CV means.

**Assessments**
- Concept quiz: grid search, CI-overlap ranking, `ColumnTransformer`, leakage  
- Participation: notebook submission with completed exercises

**Time budget (112.5 min)**
- Videos 60 + Notebook 40 + Quiz 7.5 + Project work 5 = 112.5

**Bibliography**
- ISLP: Resampling Methods (grid search built on top of 5-fold CV)  
- scikit-learn User Guide: grid search, randomized search, `ColumnTransformer`, common pitfalls  
- Kaufman, Rosset, Perlich (2012): *Leakage in Data Mining — Formulation, Detection, and Avoidance*  
- Provost & Fawcett: leakage and evaluation discipline in business framing

---

## Day 10 — Fri May 29  
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
- File: `nb10_midterm_casebook_student.ipynb`  
- Sections:
  - Integrity + allowed resources + Gemini usage boundaries (explain/verify)
  - Case 1 prompt + structured response cells
  - Case 2 prompt + structured response cells
  - Optional mini-case 3
  - **Midterm Cheat Sheet appendix** (decision tables for metric choice, scaler choice, stratify yes/no, Ridge vs Lasso, CI-overlap rule, leakage checklist — copied from nb01–nb09 into one reference card)
  - Submission checklist (self-audit)

**In-notebook exercises (10-minute scope)**
- Pause-and-do (10): Case 1 plan (split + metric + leakage risks + model shortlist).  
- Pause-and-do (10): Case 2 evaluation plan + error-cost logic.  
- Pause-and-do (10): Mini-case strategy under constraints.

**Assessments**
- **Midterm submission (graded):** completed notebook (strategy + minimal prototype code where requested)  
- **Project Milestone 2 (due): Simple Model + Performance Evaluation**
  - dataset exploration + feature engineering + missing-value handling + baseline pipeline (Linear/Logistic) + feature selection inside k-fold CV + baseline report with 95% CI
  - Detail: [`_final_project/2026Summer/milestone_02_baseline_model.md`](_final_project/2026Summer/milestone_02_baseline_model.md)

**Time budget (async: 112.5 min)**
- Videos 30 + Midterm notebook work 60 + Project baseline finalization 15 + Concept check 7.5 = 112.5

**Synchronous session plan (112.5 min, recorded)**
Pre-recorded micro-videos are available for students to watch before or after the session.

| Block | Duration | Content |
|-------|----------|---------|
| Week 2 Recap + Midterm Instructions | 10 min | Review Days 6-9, explain midterm format, allowed resources, Gemini boundaries |
| Midterm: Business Case Practicum | 50 min | Students work through cases live (instructor available for clarification only) |
| Break | 5 min | |
| Midterm Debrief | 10 min | Common strategies, pitfalls, what good answers look like (after submission) |
| Project Discussion | 20 min | Milestone 2 review (baseline due today), Milestone 3 preview, common modeling issues |
| Competition Check-in | 10 min | Leaderboard review, strategy tips (students now have classification + CV + tuning toolkit) |
| Course Q&A | 7.5 min | Week 2 review, Week 3 tree-based methods preview |

**Bibliography**
- Provost & Fawcett: end-to-end predictive modeling process and business framing  
- ISLP: assessment/selection + classification/regression chapters as reference  
- scikit-learn User Guide: common pitfalls (especially leakage and improper evaluation)

---

# Week 3 (Days 11–15): Trees, Ensembles, Tuning, Interpretation  
**Project milestone:** Week 3 improved model due **Day 15**

---

## Day 11 — Mon June 1  
### Decision trees: interpretable models with sharp edges  
**Learning objectives**
- Fit decision trees for regression/classification.
- Control complexity (depth, min samples) to manage overfitting.
- Interpret tree structure and failure modes.
- Compare tree vs linear/logistic baselines under CV.
- Handle class imbalance with `class_weight='balanced'` as the first-resort tool; understand why SMOTE is not the default.
- Document “when a tree is the right tool.”

**Micro-videos (54 min)**
1. Concept+demo: Trees intuition + key hyperparameters (10)  
2. Guided practice: Fit a tree + visualize + baseline compare (8)  
3. Solution: Overfitting patterns + mistakes + extension: cost-complexity pruning (9)  
4. Concept+demo: Tree evaluation under CV + stability concerns (10)  
5. Guided practice: Imbalanced classes — `class_weight='balanced'` on a down-sampled screening task, anti-SMOTE warning (8)  
6. Solution: Tuning result + extension: sensitivity analysis (9)

**Notebook(s)**
- File: `nb11_decision_trees_student.ipynb`  
- Sections:
  - Tree fit + visualization
  - Hyperparameter effects (depth sweep)
  - CV comparison table
  - Imbalanced classes — `class_weight='balanced'` as the first-resort tool (artificially imbalanced Breast Cancer demo, anti-SMOTE warning)
  - Gemini prompts: “generate a clean depth sweep block”

**In-notebook exercises (10-minute scope)**
- Pause-and-do (10): Run a depth sweep and choose depth based on CV.  
- Pause-and-do (10): Write 3 observed tree failure modes (with evidence).

**Assessments**
- Concept quiz: tree mechanics + overfitting  
- Participation: notebook submission with completed exercises

**Time budget (112.5 min)**
- Videos 54 + Notebook 46 + Quiz 7.5 + Reflection 5 = 112.5

**Bibliography**
- ISLP: Tree-Based Methods (trees, pruning)  
- ESL: CART foundations and complexity control  
- scikit-learn User Guide: DecisionTreeClassifier/Regressor parameters and inspection tools

---

## Day 12 — Tue June 2  
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
- File: `nb12_random_forests_importance_student.ipynb`  
- Sections:
  - **Prelude — Four things we call "feature importance"** (coefficient magnitude / impurity / permutation / PDP) with a reconciliation table that is referenced for the rest of the course
  - Forest training + CV comparison
  - Permutation importance + plot
  - Reporting template (model table + narrative bullets)
  - Gemini prompts: “importance + report block”

**In-notebook exercises (10-minute scope)**
- Pause-and-do (10): Tune `n_estimators` and `max_features` minimally and report effects.  
- Pause-and-do (10): Compute permutation importance and write 3 interpretation bullets.

**Assessments**
- Concept quiz: bagging/forests + importance  
- Participation: notebook submission with completed exercises

**Time budget (112.5 min)**
- Videos 54 + Notebook 46 + Quiz 7.5 + Reflection 5 = 112.5

**Bibliography**
- Breiman: “Random Forests”  
- ISLP: Tree-Based Methods (bagging/forests)  
- scikit-learn User Guide: RandomForest estimators; permutation importance and caveats

---

## Day 13 — Wed June 3  
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
- File: `nb13_gradient_boosting_student.ipynb`  
- Sections:
  - Baseline GBM fit
  - Constrained tuning template
  - Comparison report (forest vs GBM)
  - **A warning for boosting — leaky features dominate the top** (callout section tying nb09's leakage case study to GBM's sequential fitting + three red flags + debugging recipe)
  - Gemini prompts: constrained RandomizedSearchCV with guardrails

**In-notebook exercises (10-minute scope)**
- Pause-and-do (10): Train baseline GBM and compare against RF under CV.  
- Pause-and-do (10): Run constrained tuning and report best params + score.

**Assessments**
- Concept quiz: boosting, tuning tradeoffs  
- Participation: notebook submission with completed exercises

**Time budget (112.5 min)**
- Videos 54 + Notebook 46 + Quiz 7.5 + Reflection 5 = 112.5

**Bibliography**
- Friedman: “Greedy Function Approximation: A Gradient Boosting Machine”  
- ISLP: Tree-Based Methods (boosting overview)  
- scikit-learn User Guide: gradient boosting estimators and tuning guidance

---

## Day 14 — Thu June 4  
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
- File: `nb14_model_selection_protocol_student.ipynb`  
- Sections:
  - Comparison harness (pipelines list → CV scores table)
  - Multi-metric reporting (primary + supporting metrics)
  - Champion selection memo scaffold
  - **Opening the locked test set — the ceremony** (nb08-style Student's t 95% CI on champion CV scores, INSIDE / ABOVE / BELOW verdict for the single test score, payoff for the whole course's locking discipline)
  - Gemini prompts: “generate experiment log table”

**In-notebook exercises (10-minute scope)**
- Pause-and-do (10): Implement the comparison harness for 3 candidate models.  
- Pause-and-do (10): Write a champion selection memo (5 bullets + 1 risk).

**Assessments**
- Concept quiz: selection protocol + robustness  
- Participation: notebook submission with completed exercises

**Time budget (112.5 min)**
- Videos 54 + Notebook 46 + Quiz 7.5 + Reflection 5 = 112.5

**Bibliography**
- ISLP: Model Assessment and Selection (protocols for fair comparison)  
- ESL: selection bias and repeated peeking hazards  
- scikit-learn User Guide: model evaluation + parameter tuning best practices

---

## Day 15 — Fri June 5  
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
- File: `nb15_interpretation_error_analysis_project.ipynb`  
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
- **Project Milestone 3 (due): More Complex Model + Hyperparameter Tuning + Draft Abstract**
  - complex-model implementation, GridSearch/RandomSearch CV, CI-overlap comparison vs. M2 baseline, importance + PDP/ICE, error segment findings, ~250-word draft abstract
  - Detail: [`_final_project/2026Summer/milestone_03_complex_model_and_abstract.md`](_final_project/2026Summer/milestone_03_complex_model_and_abstract.md)

**Time budget (async: 112.5 min)**
- Videos 48 + Notebook 47 + Quiz 7.5 + Project work 10 = 112.5

**Synchronous session plan (112.5 min, recorded)**
Pre-recorded micro-videos are available for students to watch before or after the session.

| Block | Duration | Content |
|-------|----------|---------|
| Week 3 Recap | 10 min | Review Days 11-14 (trees, forests, boosting, model selection) |
| Live Recap & Demo: Interpretation | 15 min | Condensed highlights + live Colab demo of importance & PDP on a real model |
| PAUSE-AND-DO (live) | 20 min | Students interpret their champion model with instructor guidance |
| Break | 5 min | |
| Project Discussion | 25 min | Milestone 3 review (improved model + draft abstract due today), Milestone 4 preview (final research-poster requirements + intra-group peer-evaluation form, Purdue Undergraduate Research Conference poster format) |
| Competition Strategy | 10 min | Advanced tips: ensembles, feature engineering, leaderboard update |
| Course Q&A + Quiz | 17.5 min | Week 3 review, Week 4 preview, video recording guidance, concept quiz |

**Bibliography**
- scikit-learn User Guide: inspection tools (permutation importance, partial dependence)  
- Molnar (optional): *Interpretable Machine Learning* (global/local methods and caveats)  
- ISLP: interpretation discussions across linear and tree-based methods

---

# Week 4 (Days 16–20): Error Analysis, Fairness/Ethics, Deployment Thinking, Executive Narrative, Final Project  
**Project milestone:** Week 4 final deliverable due **Day 20**  
**Kaggle Case Competition:** Final submission deadline **Day 20 (Fri June 12, 11:59 PM)**

---

## Day 16 — Mon June 8  
### Probability calibration for decision quality  
**Learning objectives**
- Diagnose whether a classifier's probabilities are trustworthy using reliability diagrams and the Brier score.
- Apply post-hoc calibration with `CalibratedClassifierCV` (isotonic vs. sigmoid/Platt) and measure the improvement.
- Explain why tree-based ensembles are often miscalibrated and why well-regularized linear models usually are not.
- Recognize when calibration matters for a decision (action decisions) and when it does not (ranking decisions — AUC is invariant under calibration).
- Run a short cost-based threshold refresh from nb07, applied on top of calibrated probabilities.

**Micro-videos (54 min)**
1. Concept+demo: Discrimination vs. calibration — when a "70% probability" actually means 70% (10)  
2. Guided practice: nb07 threshold-tuning 5-minute refresh on the screening cost matrix (8)  
3. Concept+demo: Reliability diagrams + Brier score as the calibration metric (10)  
4. Guided practice: `CalibratedClassifierCV` with isotonic vs. sigmoid, three-way reliability overlay (8)  
5. Concept+demo: Why tree ensembles need calibration and linear models usually don't (9)  
6. Solution: Picking the calibrator + bridge to decision policy and Kaggle probabilistic scoring (9)

**Notebook(s)**
- File: `nb16_decision_thresholds_calibration_student.ipynb`  
- Sections:
  - **nb07 threshold + cost refresh** (compact 5-minute section, references nb07 for deep context)
  - **Calibration diagnostics** — reliability diagram + Brier score on a Random Forest from nb12
  - **Calibration fixes** — `CalibratedClassifierCV` with isotonic vs. sigmoid, three-way reliability overlay + Brier winner
  - Decision policy summary block (ready for slides)
  - Sensitivity analysis on the FN cost

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

## Day 17 — Tue June 9  
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
- File: `nb17_fairness_slicing_model_cards.ipynb`  
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

## Day 18 — Wed June 10  
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
- File: `nb18_reproducibility_monitoring_student.ipynb`  
- Sections:
  - Refactor into `train()` / `predict()` / `evaluate()`
  - Save/load via joblib
  - Monitoring plan template (tables)
  - **From trained pipeline to Kaggle submission** (load the saved pipeline → read a held-out CSV → produce `submission.csv` with correct column names; the last-mile mechanics for the Bank Churn competition on Day 20)
  - “Ready-to-share” notebook hygiene checklist

**In-notebook exercises (10-minute scope)**
- Pause-and-do (10): Implement `train_model(config)` returning pipeline + metrics.  
- Pause-and-do (10): Draft a monitoring plan with 5–8 signals and owners.

**Assessments**
- Concept quiz: reproducibility + drift  
- Participation: notebook submission with completed exercises

**Time budget (112.5 min)**
- Videos 54 + Notebook 46 + Quiz 7.5 + Reflection 5 = 112.5

**Bibliography**
- Chip Huyen: *Designing Machine Learning Systems*  
- Optional: *Machine Learning Design Patterns* (Lakshmanan et al.)  
- Optional: *Dataset Shift in Machine Learning* (Quionero-Candela et al.)  
- Rabanser, Günnemann, Lipton: dataset shift detection (Failing Loudly)  
- scikit-learn User Guide: model persistence and reproducible pipelines

---

## Day 19 — Thu June 11  
### Elements of data communication and poster design: six principles applied to the eleven-section research-poster architecture  
**Learning objectives**
- Apply the **six principles of data communication** (context, visualization, less-is-more / data-ink ratio, hierarchy, beauty, story) to a project figure.
- Diagnose common chart failures (misleading scales, dual axes, pie-chart abuse) and rebuild the same data into a clearer view.
- Plan the **layout, typography, and visual hierarchy** of a research-conference poster aimed at a non-expert audience.
- Draft the **eleven-section poster outline** and the **120–150-word abstract** for the M4 final-poster submission.

**Micro-videos (42 min)**
1. Concept+demo: Forest-and-trees framing + the six-principles overview (6)  
2. Concept+demo: Context, visualization-derives-from-data, common chart failures (8)  
3. Concept+demo: Less-is-more — data-ink ratio + the eight-step cleanup walk-through (8)  
4. Concept+demo: Hierarchy + beauty — accent colors, emphasis, "telling your story" sequence (7)  
5. Guided practice: Eleven-section poster outline + visual-hierarchy planning on the URC template (7)  
6. Solution: Poster-outline example + abstract-paragraph rewrite + extension: presenting at URC (6)

**Notebook(s)**
- File: `nb19_data_communication_poster.ipynb`  
- Sections:
  - The six principles of data communication (worked examples + chart-failure gallery)
  - Data-ink ratio cleanup walk-through (eight panels)
  - Hierarchy + beauty + telling-your-story sequences (six-panel and nine-panel walk-throughs)
  - Poster design: template, rubric, visual hierarchy, layout, eleven-section content map
  - Crafting a clear narrative + research-design flow + presentation tips
  - Gemini prompts: chart-audit; abstract-paragraph rewrite

**In-notebook exercises**
- PAUSE-AND-DO 1 (8 min): Audit one project figure against the six principles; produce a three-bullet rebuild plan.  
- PAUSE-AND-DO 2 (15 min): Draft the eleven-section poster outline + a 120–150-word abstract.

**Assessments**
- Concept quiz: data communication principles + poster section architecture  
- Project checkpoint: draft poster outline + abstract paragraph (M4 input)

**Time budget (112.5 min)**
- Videos 42 + Notebook 45 + Quiz 7.5 + Project studio 18 = 112.5

**Bibliography**
- Edward Tufte: *The Visual Display of Quantitative Information* (data-ink ratio)  
- Kieran Healy: *Data Visualization — A Practical Introduction*  
- Cole Nussbaumer Knaflic: *Storytelling with Data*  
- Kastellec & Leoni: "Using Graphs Instead of Tables in Political Science"  
- Purdue Undergraduate Research Conference poster rubric and template

---

## Day 20 — Fri June 12  
### Final delivery: project package submission + peer review + course closeout  
**Learning objectives**
- Deliver a complete end-to-end predictive analytics package.
- Produce a final research poster (single PDF named `<group-number>.pdf`) following the Purdue Undergraduate Research Conference template, plus the supporting run-all notebook.
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
- File: `nb20_final_submission_peer_review.ipynb`  
- Sections:
  - Final self-audit checklist (run-all, outputs, links)
  - Submission links + artifact manifest (notebook, deck, video)
  - Peer review form (rubric + comment prompts)
  - Postmortem prompts (8–10 lines)

**In-notebook exercises (10-minute scope)**
- Pause-and-do (10): Run-all audit and fix one reproducibility issue (real or simulated).  
- Pause-and-do (10): Complete one peer review with rubric scores + 3 actionable edits.

**Assessments**
- **Project Milestone 4 (due): Final Research Poster + intra-group Peer Evaluation**
  - Single PDF poster named `<group-number>.pdf` (e.g., `01.pdf`, `17.pdf`) following the Brightspace template (Purdue Undergraduate Research Conference poster format)
  - Each group member submits an individual confidential peer-evaluation form rating the other three teammates
  - Optional Fall 2026 conference presentation strongly encouraged (faculty mentorship available — email Prof. Moreira before M3)
  - Detail: [`_final_project/2026Summer/milestone_04_final_poster.md`](_final_project/2026Summer/milestone_04_final_poster.md)
- **Kaggle Case Competition deadline (11:59 PM):** Final Kaggle submission + Brightspace code submission (complete, replicable code for best-performing model)

**Time budget (async: 112.5 min)**
- Videos 30 + Final submission notebook work 55 + Competition code packaging 20 + Postmortem 7.5 = 112.5

**Synchronous session plan (112.5 min, recorded)**
Pre-recorded micro-videos are available for students to watch before or after the session.

| Block | Duration | Content |
|-------|----------|---------|
| Final Submission Workshop | 25 min | Live run-all audit demo, reproducibility checklist, Brightspace submission walkthrough |
| Break | 5 min | |
| Student Presentations | 40 min | Selected teams/individuals present projects (~5 min each, 6-8 presentations) |
| Competition Leaderboard Reveal | 15 min | Final rankings, top performers share their approaches |
| Course Wrap-up | 20 min | Key takeaways, career applications, postmortem discussion, course evaluations |
| Final Q&A | 7.5 min | Last questions |

**Note:** Milestone 4 + Kaggle competition deadline at 11:59 PM (after the lecture).

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

