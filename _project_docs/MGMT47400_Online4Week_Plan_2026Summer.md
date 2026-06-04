# QM47400 – Predictive Analytics (3 credits)  
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
- **Competition:** Summer 2026 QM47400 Case Competition: Bank Churn
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
| 11 | Decision Trees (paired clf + reg) | Introduces the first non-linear model family (CART), running **both** Wisconsin breast cancer classification and California Housing regression in parallel under the `_clf` / `_reg` namespace convention. Teaches the bias-variance tradeoff concretely through paired depth sweeps and overfitting demonstrations on both spines, plus tree-vs-linear comparisons (LogReg + Ridge). Two PAUSE-AND-DO exercises — one per spine — apply the one-standard-error rule for depth selection. **§7 verdicts:** classification keeps `LogReg(C=1.0)` (tree CI ends below LogReg's CI lower bound — no displacement, CI-clear); regression switches to `DecisionTreeRegressor(max_depth=5)` under the **dominance tiebreaker** — the tree's CV mean (0.613) AND its CV CI lower bound (0.604) both sit above OLS's (0.586, 0.554), even with the upper-end CI overlap, so the tree wins as a modest winner. OLS stays on as the close-second runner-up; the depth-5 tree becomes the regression floor that nb12's RF must displace. | nb10 consolidates Weeks 1–2; students enter nb11 with solid evaluation skills and can focus on tree mechanics. nb12 solves the single tree's high-variance problem and inherits the dual-spine pattern. |
| 12 | Random Forests & Importance (paired clf + reg) | Solves the single tree's instability through bagging + random feature subsets, on **both** cases. Paired sections throughout: single-tree vs forest (§4, depth-pinned to nb11 to isolate the bagging effect), **joint 3D `n_estimators × max_features × max_depth` grid** (§5, 27 cells per case displayed as 3 rows × 2 cols of heatmaps — rows are depths {3, 5, None} including nb11's per-case pick, columns are the two cases; red rectangle marks largest-mean-smallest-CI cell per case), OOB vs CV diagnostic, **four-method feature-importance reconciliation heatmap** (linear coefficient / impurity / permutation / drop-column) — the course-wide reference table that nb15 lifts forward. §5 lands on `(n_estimators=50, max_features='sqrt', max_depth=3)` for clf — **nb11's depth survives** joint tuning (all 27 cells tie under CI-overlap; parsimony picks the simplest cell tied with the red rectangle) — and `(n_estimators=50, max_features=0.5, max_depth=None)` for reg — **nb11's depth does NOT survive** (the depth-3 and depth-5 cells are CI-clear below the depth-None cells on California Housing; parsimony picks the smallest-n cell within the depth-None tied group). Closes with a comprehensive comparison plot showing single tree, forest, and the **Week-2 reference** (LogReg / OLS) on a CV-CI dot plot per case. | nb11 proves single trees overfit on both cases; nb12's variance-reduction motivation lands only after experiencing that instability. nb13 needs bagging as a contrast for boosting. |
| 13 | Gradient Boosting (paired clf + reg) | Completes the ensemble trilogy on **both** cases — sequential error correction with **joint 4D tuning grid** `learning_rate × n_estimators × max_features × max_depth` (54 cells per case across 12 panels, showing CV mean + 95% CI half-width annotated per cell, red rectangle marking the largest-mean-smallest-CI cell per case) that includes nb12's per-case RF shipped `(max_features, max_depth)` pairs as candidates. The 4D grid lands on **the same config for both cases**: `(learning_rate=0.2, n_estimators=200, max_features=0.5, max_depth=3)` — a clean symmetry. nb12's clf RF pick `(mf='sqrt', md=3)` transfers cleanly to GBM (statistically tied with the red rectangle); nb12's reg RF pick `(mf=0.5, md=None)` does NOT transfer (unlimited depth overfits in boosting unless lr is small). The five-candidate comparison plot has the tuned GBM beating nb12's RF on regression CV R² (0.812 vs 0.804), CV RMSE (0.499 vs 0.509), CV MAE (essentially tied), AND CV CI lower bound (0.798 vs 0.794) — the GBM's lower-CI dominance breaks the upper-end overlap, so the GBM is the **modest winner** (not a CI-clear blowout). Closes with the **"leaky features dominate the top" callout** connecting nb09's leakage case studies to boosting's amplification effect. | nb12 establishes the parallel ensemble baseline (bagging reduces variance); nb13 contrasts with sequential approach (boosting reduces bias). nb14 needs the full candidate roster. |
| 14 | Model Selection Protocol + TWO Ceremonies + Post-Deployment Monitoring (paired clf + reg) | Replaces informal "pick the highest number" comparison with a structured, fair, reproducible protocol on **both** cases — identical CV folds, declared primary metric, and a complete roster anchored to nb11/nb12/nb13's actual ship picks. **7 classification candidates** (Dummy, LogReg(C=1.0), LogReg L1, Tree(d=3), RF `(n=50, sqrt, d=3)` — nb12 ship, GBM default, GBM tuned `(lr=0.2, n=200, mf=0.5, d=3)` — nb13 ship). **8 regression candidates** (Dummy, OLS, Ridge, Lasso, Tree(d=5), RF `(n=50, mf=0.5)` — nb12 ship, GBM default, GBM tuned `(lr=0.2, n=200, mf=0.5, d=3)` — nb13 ship). Champion selection memo template + Student's *t* 95% CI. **Two parallel test-set ceremonies** — one for `X_test_clf`, one for `X_test_reg`, each opened exactly ONCE — with INSIDE / ABOVE / BELOW money plot per case. Champions named: **LogReg(C=1.0) ships clf** (tied with the tuned GBM and the RF under CI-overlap, parsimony picks the simplest linear baseline); **tuned GBM (lr=0.2, n=200, mf=0.5, d=3) ships reg** — modest winner over the RF on mean R², mean RMSE, mean MAE (≈ tied), AND CV CI lower bound (0.798 vs 0.794). After §5's selection, the tuned GBM is retrained on the full train+val pool (16,512 rows) before the locked test set opens once for the final out-of-sample estimate. **§6.3 verdict playbook** spells out what to do for each INSIDE / ABOVE / BELOW outcome — INSIDE ships the model with CV CI + test point + retrain triggers recorded; small-drift ABOVE ships with a documented gap (Step 1's train+val refit); large-drift ABOVE (>2 CI widths) investigates before deploying (easy test-set draw, opposite-direction CV contamination, or train-only data quality issue); BELOW stops deployment, diagnoses the pipeline (leakage / distribution shift / insufficient signal), and the **test-set-is-spent rule** requires a NEW held-out test set before re-opening the ceremony. **Singleness rule callout** explicitly clarifies: two ceremonies in this notebook is a pedagogical demo; your project gets ONE ceremony. **§7 post-deployment monitoring** turns the one-shot ceremony into an ongoing discipline: **Silent decay** (data drift / concept drift / population shift) vs **Front-door arrivals** (announced changes — new feature, regulatory shift); **Three R's escalation ladder** (Re-evaluate cheap → Re-train medium → Rebuild expensive); case-specific monitoring checklists (MedScreen: reliability + Brier + recall + KS on top-3 + population composition; HomeValue: MAE/RMSE + errors by price tier + errors by location + KS on top features + major outside events); coded **KS drift demo** with the rule that alert thresholds live on the KS statistic, not the p-value (PSI / chi-squared / Wasserstein / MMD flagged as other detectors worth researching); one-page **Monitoring Card** every deployed model gets. | nb13 completes the candidate pool on both cases; nb15 turns the committed champion into a complete M3 walkthrough (baseline replication with 95% CI, tuning, CI-overlap pick + final-training step with `random_state` locked, draft abstract). |
| 15 | Final Project Milestone 03 Walkthrough — Complex Model + Hyperparameter Tuning + Draft Abstract | **Markdown-only walkthrough** (no code, no PAUSE-AND-DO) pointing students at `milestone_03_complex_model_and_abstract.md`. Same shape as nb05 §6/§7 and nb10's milestone sections — a companion read for groups working on M3. Sections map directly to the M3 rubric: §0 Prediction Goals, §1a Baseline Replication (M2) with 95% CI, §1b Complex Model + Tuning + CI-overlap rule + final-training step on the full training fold (with `random_state` locked so M4 reproduces the same fitted Pipeline), §1c Required Visualizations (hyperparameter-search plot, model-comparison bar with 95% CI error bars, feature importance, regression OR classification diagnostics), §2 Draft Abstract (\~250 words, six required elements), Course Case Competition (Kaggle Bank Churn) alignment with the same M3 workflow, and Tips and Common Pitfalls. **No Interpretation, Calibration, or Decision-Quality content** — those are out of M3's scope for the 2026 Summer cohort. | nb14 commits the champion; nb15 is the markdown companion for the M3 milestone deliverable. nb16 (time series) opens the next analytical thread. |
| 16 | Time-Series Forecasting | Introduces forecasting as a structurally distinct supervised problem — never shuffle, walk-forward CV via `TimeSeriesSplit`, lag features (`lag1` / `lag12`), and three baselines (naive, seasonal-naive, lag-feature linear regression) compared on identical CV folds. Closes with a one-shot opening of the locked test window mirroring nb14's protocol. | nb15 closes the static-classification arc; nb16 widens the lens to temporal data the operations team will see in real business series. nb17 needs the cross-validated forecast as a candidate poster figure. |
| 17 | Data Communication & Poster Design (formerly nb19) | Walks the **six principles** of data communication (context, visualization, less-is-more / data-ink ratio, hierarchy, beauty, story) and applies them to the **eleven-section research-poster architecture** of the Purdue Undergraduate Research Conference template. Includes a chart-audit exercise on a project figure and an outline-plus-abstract drafting exercise for the M4 poster. | nb15 + nb16 supply the headline numbers (CV-CI, calibration, locked-test verdict, forecast comparison) that the poster has to communicate; without them, the design lecture would lack a payload. nb18 takes the poster outline into competition-pipeline mode. |
| 18 | Competition Workflow & Kaggle Submission | End-to-end production pipeline for the Bank Churn case competition: load → EDA snapshot → `ColumnTransformer` preprocessor → baseline + improved model on identical CV folds → refactor into `train_pipeline` / `predict_pipeline` → `joblib` save/load → generate `submission.csv` with exact column names. The Kaggle-submission demo is the only authorized non-nb14 use of the locked Kaggle test file (no labels = production prediction, not model evaluation). | nb15-nb17 supply the champion model and the poster narrative; nb18 packages them into a leaderboard submission and a portable artifact. nb19 widens the lens to deep learning as a horizon topic. |
| 19 | Deep Learning | Focused, business-undergrad-friendly notebook built around a hands-on PyTorch experience (7 sections): (1) history + drivers, (2) PyTorch vs. TensorFlow, (3) **Neural Networks intuition** — the four 3Blue1Brown chapters embedded as inline-playable `YouTubeVideo` cells, (4) **end-to-end PyTorch training on FashionMNIST** following the official Quickstart (load → build → train/test loop → save → load → predict, the centerpiece), (5) **when-to-use** rubric + an honest neural-net vs. gradient-boosting tie on tabular data, (6) **special topic + hands-on lab: Large Language Models (LLMs)** — concept (next-token prediction, tokens/context/temperature/hallucination, usage modes) then a Hugging Face `transformers` lab (sentiment + zero-shot ticket routing, no API key) and an optional `ask_llm` hosted-API helper (Purdue/OpenAI/Anthropic, key-gated) running a structured-JSON ticket-triage task, (7) wrap-up. The ISLP conceptual sections (single-layer math, fitting, CNN, document classification, RNN, time series) were dropped to keep the focus on the hands-on lab; a tight three-shapes recap (MLP/CNN/RNN) is folded into the when-to-use section. | nb18 ships the gradient-boosted tabular champion; nb19 widens the lens so analysts can answer the "what about AI?" question with evidence and rubric, and gives hands-on first contact with training a network in PyTorch. |
| 20 | Course End and Reflection | Capstone walkthrough (markdown only, nb15 milestone-walkthrough style): the **Milestone 04 research poster** (required components, visualizations, 100-pt rubric) + the **one-shot locked-test ceremony** (INSIDE/ABOVE/BELOW verdict against the M3 CV CI), the **Kaggle case-competition final submission** (`NN_kaggle_code.ipynb`, `Group NN` naming, don't chase the public leaderboard), and the **two course closeouts**: the confidential intra-group **peer evaluation** and the required course-end **reflection survey** (\~10–15 min). | nb19 closes the awareness arc; nb20 closes the course. Reflection survey output seeds the next-summer redesign. |

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

Week 4 — DELIVERY ARC
  15 Interpretation+Calibration → 16 Time Series → 17 Communication → 18 Competition Workflow → 19 Deep Learning → 20 Course End + Reflection
  (Trust + decision policy) (Walk-forward CV) (Poster design) (Kaggle submission pipeline) (Awareness + when-to-use) (Submit + Peer Review + Reflection Survey)
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
### Decision trees: interpretable models with sharp edges (paired clf + reg)  
**Learning objectives**
- Fit `DecisionTreeClassifier` (Wisconsin breast cancer) and `DecisionTreeRegressor` (California Housing) using the `_clf` / `_reg` namespace convention.
- Visualize each tree with `plot_tree` and interpret the splits in stakeholder language (clinician-readable for clf, defensible-in-court for reg).
- Diagnose overfitting via paired train-vs-CV curves (5-fold CV) — same pattern, different scales — on both spines.
- Compare each tree to its linear analogue (LogReg for clf, Ridge for reg) under identical CV folds.
- Choose `max_depth` using the one-standard-error rule on both spines.

**Micro-videos (54 min)**
1. Concept+demo: Tree intuition — axis-aligned splits visualized step-by-step on synthetic 2D data (10)  
2. Guided practice: Fit + visualize a classification tree on breast cancer; same pattern for a regression tree on California housing (8)  
3. Solution: Predicted-vs-actual + step-function visuals for the regression tree; why trees can't extrapolate (9)  
4. Concept+demo: Paired overfitting curves — same shape on both spines (10)  
5. Guided practice: Tree vs linear analogue (LogReg / Ridge) under identical CV folds (8)  
6. Solution: Depth selection by the one-standard-error rule + bridge to ensemble methods (nb12, nb13) (9)

**Notebook(s)**
- File: `nb11_decision_trees_student.ipynb`  
- Sections:
  - Decision tree intuition (synthetic 2D dataset showing axis-aligned splits at depths 1, 2, 3, 6)
  - Load both datasets — locked test sets, paired CV splitters
  - Classification tree on breast cancer — `plot_tree` + train-vs-CV scores
  - Regression tree on California Housing — `plot_tree` + predicted-vs-actual + step-function visualization
  - Paired overfitting problem (depth sweep on both, side-by-side panels with the `plot_train_val_curve` helper)
  - Tree vs linear analogue — paired CV-bar comparison (Tree vs LogReg, Tree vs Ridge)
  - Gemini prompts: depth sweep + one-SE rule selector

**In-notebook exercises**
- PAUSE-AND-DO Exercise 1 (clf, 5 min): Tune classification tree depth via 5-fold CV ROC-AUC; apply the one-SE rule.
- PAUSE-AND-DO Exercise 2 (reg, 5 min): Tune regression tree depth via 5-fold CV R²; convert best CV-RMSE to USD; apply the one-SE rule.

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
### Random forests: bagging, OOB intuition, and feature importance (paired clf + reg)  
**Learning objectives**
- Explain bagging + random feature subsets on both spines (Wisconsin breast cancer and California Housing) and quantify the variance-reduction payoff under identical CV folds.
- Fit `RandomForestClassifier` and `RandomForestRegressor`; demonstrate CV-score lift over the nb11 single tree and the Week-2 reference (LogReg / OLS).
- Tune `n_estimators`, `max_features`, and `max_depth` **jointly** via a 3D grid (3×3×3 = 27 cells) that includes nb11's per-case depth picks (3 clf / 5 reg) alongside unlimited depth; apply the CI-overlap rule with parsimony as the tiebreaker on both cases — and observe whether nb11's depth survives joint tuning per case.
- Read the OOB score as a free, non-redundant validation signal; know when to trust it vs CV.
- Build the four-method feature-importance reconciliation heatmap (linear coef / MDI / permutation / drop-column) that nb15 uses for interpretation.

**Micro-videos (54 min)**
1. Concept+demo: Bagging intuition — bootstrap KDE plot, why forests reduce variance (10)  
2. Guided practice: Single tree vs forest on both spines, paired CV-CI dot plot (8)  
3. Solution: joint 3D `n_estimators × max_features × max_depth` grid (CV mean + 95% CI per cell, 27 cells × 2 cases = 6 heatmaps) + extension: OOB vs CV diagnostic (9)  
4. Concept+demo: Four importance methods explained — what each one measures (10)  
5. Guided practice: Build the four-method reconciliation heatmap on both spines (8)  
6. Solution: Interpretation pitfalls + comprehensive Tree-vs-Forest-vs-Week-2-reference comparison (9)

**Notebook(s)**
- File: `nb12_random_forests_importance_student.ipynb`  
- Sections:
  - Setup with five plot helpers (`plot_train_val_curve`, `plot_predicted_vs_actual`, `plot_cv_ci`, `plot_importance_bars`, `plot_importance_heatmap`) and the two Week-2 reference pipelines
  - Bagging intuition (bootstrap KDE plot)
  - Load both datasets — locked test sets, paired CV splitters
  - Single tree vs forest — paired CV-CI dot plot
  - Tuning `n_estimators × max_features × max_depth` — joint 3D grid (27 cells per case, displayed as 3×2 panels) with CV mean + 95% Student's *t* CI half-width annotated per cell (red rectangle marks the largest-mean-smallest-CI cell per case); nb11's per-case depth picks are included in the grid as candidates; same annotation approach as nb13 §6 extended to a third dimension
  - OOB vs CV — paired diagnostic curves
  - **Four-method feature-importance reconciliation heatmap** — paired across both spines
  - Permutation importance with error bars — paired horizontal bars
  - Comprehensive comparison: Single Tree vs Forest vs Week-2 reference on a CV-CI dot plot
  - Gemini prompts: tuning grid + importance heatmap

**In-notebook exercises**
- PAUSE-AND-DO Exercise 1 (clf, 5 min): Tune `RandomForestClassifier` over a 3×3 (n_estimators, max_features) grid; apply the one-SE rule.
- PAUSE-AND-DO Exercise 2 (reg, 5 min): Tune `RandomForestRegressor` over a 3×3 grid; convert best CV-RMSE to USD; apply the one-SE rule.

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
### Gradient boosting: performance with discipline (paired clf + reg)  
**Learning objectives**
- Explain bias-reducing sequential boosting vs variance-reducing parallel bagging on both cases.
- Fit `GradientBoostingClassifier` and `GradientBoostingRegressor`; benchmark against the nb12 random forest and the Week-2 reference.
- Tune `learning_rate × n_estimators` jointly via a 3×3 heatmap with CV mean + 95% CI half-width annotated per cell; pick the cell by the largest-mean-smallest-CI rule.
- Recognize that **default GBM with sklearn defaults does not always beat RF** — the §6 `(lr=0.2, n=200)` pick is what lifts GBM into a CI-tie on California Housing.
- Apply the leakage callout: GBM amplifies leaky features more aggressively than any other course algorithm.

**Micro-videos (~45 min)**
1. Concept+demo: Boosting vs bagging — sequential vs parallel schematic + bias/variance contrast (10)  
2. Guided practice: Default GBM on both cases + comparison vs RF + Week-2 reference (8)  
3. Solution: Learning rate trade-off + extension: why slow learning + many trees beats fast learning (9)  
4. Concept+demo: Joint tuning heatmap with CV mean ± 95% CI annotated per cell (9)  
5. Guided practice: Tune GBM 3×3 grid on both cases (8)  
6. Solution: Final comparison + leaky-features-dominate-boosting warning (9)

**Notebook(s)**
- File: `nb13_gradient_boosting_student.ipynb`  
- Sections:
  - Setup with helpers + Week-2 references
  - Load both datasets
  - Boosting-vs-bagging schematic
  - Baseline GBM on both cases (paired CV-CI)
  - Learning rate trade-off (paired log-scale curves)
  - n_estimators × learning_rate joint heatmap with CV mean + 95% CI half-width per cell (paired)
  - Final comparison: 5 candidates per case — Reference, Tree, RF, default GBM, tuned GBM ((lr=0.2, n=200) on regression)
  - **Leaky-features warning** with debugging recipe
  - Gemini prompts: joint tuning grid

**In-notebook exercises**
- PAUSE-AND-DO Exercise 1 (clf, 5 min): Tune `GradientBoostingClassifier` over a 3×3 (n_estimators, learning_rate) grid; apply the one-SE rule.
- PAUSE-AND-DO Exercise 2 (reg, 5 min): Tune `GradientBoostingRegressor` over a 3×3 grid; convert best CV-RMSE to USD; apply the one-SE rule.

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
### Model selection ceremony + post-deployment monitoring — paired clf + reg  
**Learning objectives**
- Build a standardized model comparison workflow (same CV, same metric, same fold splits) on both spines, evaluating the **full reference lineage from nb01–nb13** (7 clf + 8 reg candidates).
- Use multiple metrics (primary + 2 supporting) without "metric shopping."
- Apply the **CI-overlap rule** — the top model earns displacement only if its 95% CI is clearly above the runner-up's — and the **dominance tiebreaker** for modest winners when CIs overlap at the upper end but every fold-level metric (mean, lower-CI bound, worst-case fold) points the same direction.
- Write a 5-part **champion selection memo** in stakeholder language for each spine.
- Open the locked test set **exactly once per spine** and pronounce an INSIDE / ABOVE / BELOW verdict against the champion's CV CI.
- Read the **§6.3 verdict playbook** — INSIDE → ship; small-drift ABOVE → ship with documented gap; large-drift ABOVE → investigate; BELOW → STOP + diagnose + test-set-is-spent rule.
- Internalize the **singleness rule**: two demo cases × one ceremony each = one ceremony per project.
- Build a **post-deployment monitoring plan** — distinguish **Silent decay** (data drift / concept drift / population shift) from **Front-door arrivals**; apply the **Three R's escalation ladder** (Re-evaluate → Re-train → Rebuild); draft a case-specific monitoring checklist (cadence, alert threshold, escalation step per row); run a coded **KS drift detector** (alert on KS statistic, not on the p-value); document the deployed model on a one-page **Monitoring Card**.

**Micro-videos (54 min)**
1. Concept+demo: Selection protocol — what must be held constant; CI-overlap rule + dominance tiebreaker (10)  
2. Guided practice: 7-clf / 8-reg comparison harness + CV-CI dot plot per case (8)  
3. Solution: Champion selection memo template + edge cases (parsimony tie-break vs modest winner) (9)  
4. Concept+demo: The test-set ceremony — protocol, money plot, INSIDE/ABOVE/BELOW + §6.3 verdict playbook (10)  
5. Guided practice: Walk both ceremonies (clf + reg) and read the verdicts (8)  
6. Concept+demo: §7 monitoring — Silent decay vs Front-door arrivals, Three R's, MedScreen + HomeValue checklists, KS demo, Monitoring Card + bridge to nb15 (9)

**Notebook(s)**
- File: `nb14_model_selection_protocol_student.ipynb`  
- Sections:
  - Setup with `compare_models_comprehensive` harness + verdict helper
  - The model selection problem (concept)
  - Load both datasets — locked test sets
  - Define the full candidate roster per spine: **7 clf** (Dummy, LogReg(C=1.0), LogReg L1, Tree(d=3), RF(n=50, sqrt, d=3) — nb12 ship, default GBM, tuned GBM lr=0.2/n=200/mf=0.5/d=3 — nb13 ship) and **8 reg** (Dummy, OLS, Ridge α=1.0, Lasso α=0.01, Tree(d=5), RF(n=50, mf=0.5) — nb12 ship, default GBM, tuned GBM lr=0.2/n=200/mf=0.5/d=3 — nb13 ship)
  - Multi-metric reporting — paired CV-CI dot plot with primary + supporting metrics (post-hoc metric computation, no `neg_*` scoring strings)
  - **§6.1 Classification Ceremony** — `X_test_clf` opens once; money plot with INSIDE/ABOVE/BELOW; smoke-test verdict is often BELOW on the ~114-sample test set (CV CI near ceiling, pedagogical case for unfavorable verdicts)
  - **§6.2 Regression Ceremony** — `X_test_reg` opens once; money plot in USD-RMSE units; tuned GBM is the §5 modest winner under the dominance tiebreaker, with RF as the close-second runner-up
  - **§6.3 Reading the Verdict — INSIDE/ABOVE/BELOW playbook** + test-set-is-spent rule + singleness-rule reinforcement
  - **§7.1 Three R's escalation ladder** — Re-evaluate → Re-train → Rebuild
  - **§7.2 MedScreen monitoring checklist** — reliability + Brier (LogReg gives calibrated probabilities), recall at the operating threshold, KS drift on top-3 features, population composition
  - **§7.3 HomeValue monitoring checklist** — MAE/RMSE on last month's appraisals, errors by price tier, errors by location grid, KS drift on `MedInc`/`AveOccup`/`Population`, major outside events
  - **§7.4 KS drift demo in code** — `scipy.stats.ks_2samp` on one feature per case across no-shift / modest-shift / bigger-shift scenarios; alert threshold lives on the KS statistic, not the p-value; PSI / chi-squared / Wasserstein / MMD flagged as other detectors
  - **§7.5 One-page Monitoring Card** — which model is running, what the test set said, what we're tracking, who acts
  - §8 Wrap-up + bridge to nb15

**In-notebook exercises**
- PAUSE-AND-DO Exercise 1 (clf, ~7 min): Apply the CI-overlap rule + parsimony in code; set the clf champion variables; write the State Health Department memo with an operational revisit trigger.
- PAUSE-AND-DO Exercise 2 (reg, ~7 min): Apply the CI-overlap rule + dominance tiebreaker in code; set the reg champion variables; write the HomeValue Analytics memo, converting R² to USD-RMSE.

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
### Interpretation, calibration, and decision quality (paired clf + reg, project improved model delivery)
**Learning objectives**
- Refit the committed champions from nb14 on both spines (LogReg for clf, tuned GBM for reg) and explain via permutation importance + PDP.
- Run error analysis on each spine: confusion matrix + per-segment metrics for clf; residual plot + RMSE-by-quintile for reg.
- (Classification track only) Diagnose calibration with reliability diagrams + Brier score; fix miscalibration with `CalibratedClassifierCV`.
- (Classification track only) Tune a decision threshold under an explicit cost matrix; run an FN-cost sensitivity sweep.
- Write the M3 milestone scaffold (problem-type-agnostic) — interpretation findings + error analysis + decision quality.

**Micro-videos (50 min)**
1. Concept+demo: Interpretation toolkit overview — permutation importance + PDP on both spines (8)
2. Concept+demo: Error analysis differentiated per spine (confusion vs residual; heatmap vs quintile bars) (8)
3. Guided practice: Refit champions + paired permutation + paired PDP (7)
4. Concept+demo: Calibration + threshold + cost (clf-only) + parallel framing for reg (residual diagnostic) (9)
5. Guided practice: Reliability diagram + threshold sweep under FN:FP cost matrix (9)
6. Solution: M3 scaffold filled in for one clf and one reg case + bridge to Day 16 (9)

**Notebook(s)**
- File: `nb15_interpretation_calibration_project_student.ipynb`
- Sections:
  - Setup with helpers + Week-2 references
  - Load both datasets + refit committed champions (LogReg + tuned GBM)
  - Permutation importance — paired
  - PDP — paired (top-3 features per spine)
  - Error analysis — clf (confusion + per-segment heatmap), reg (residual + RMSE-by-quintile)
  - Decision quality (clf only): reliability diagram + Brier + threshold/cost + FN-cost sensitivity sweep
  - PAUSE-AND-DO 1 (clf): segment analysis findings
  - PAUSE-AND-DO 2 (reg): residual diagnosis findings
  - M3 scaffold (problem-type-agnostic)

**In-notebook exercises**
- PAUSE-AND-DO Exercise 1 (clf, 5 min): Three findings on the per-segment heatmap and confusion matrix for the State Health Department's screening tool.
- PAUSE-AND-DO Exercise 2 (reg, 5 min): Three findings on the residual plot and RMSE-by-quintile for HomeValue Analytics' price-prediction model, including an escalation-threshold recommendation.

**Assessments**
- Concept quiz: interpretation + calibration + decision policy
- **Project Milestone 3 (due): More Complex Model + Hyperparameter Tuning + Draft Abstract** — see [`milestone_03_complex_model_and_abstract.md`](_final_project/2026Summer/milestone_03_complex_model_and_abstract.md).

**Time budget (async: 112.5 min)**
- Videos 50 + Notebook 45 + Quiz 7.5 + M3 work 10 = 112.5

**Bibliography**
- scikit-learn User Guide: inspection tools (permutation importance, partial dependence) and probability calibration.
- Provost & Fawcett: cost-aware decision making with predictions.
- Niculescu-Mizil & Caruana; Zadrozny & Elkan: foundational calibration references.

---

# Week 4 (Days 16–20): Time Series, Communication, Competition Workflow, Deep Learning, Course End
**Project milestone:** Week 4 final deliverable (M4 Poster + Peer Evaluation) due **Day 20**
**Kaggle Case Competition:** Final submission deadline **Day 20 (Fri June 12, 11:59 PM)**
**Reflection survey:** Required for course completion, due **Day 20 (Fri June 12, 11:59 PM)**

> **Restructure note (April 2026):** Week 4 was redesigned to add Time Series (Day 16) and Deep Learning (Day 19) as core topics for business analytics careers. Calibration was merged into Day 15. Standalone fairness and reproducibility days were retired; the load-bearing pieces (model card limitations, joblib persistence, monitoring vocabulary) were folded into nb15 (decision-policy + limitations) and nb18 (competition workflow + pipeline persistence) respectively.

---

## Day 16 — Mon June 8
### Time-series forecasting: walk-forward CV, lag features, and baseline models
**Learning objectives**
- Distinguish a forecasting problem from a generic supervised-learning problem and choose the right evaluation protocol.
- Build a time-respecting train/test split where the test window is the most recent slice of history.
- Run **walk-forward cross-validation** with `TimeSeriesSplit` instead of k-fold.
- Engineer **lag features** (`lag1`, `lag12`) and fit a linear regression that respects temporal order.
- Compare three baselines (naive, seasonal-naive, linear-with-lags) using identical CV folds.
- Open the locked test window in a one-shot evaluation ceremony, mirroring nb14's protocol.

**Micro-videos (50 min)**
1. Concept+demo: Forecasting vs. generic supervised — why row order changes everything (9)
2. Guided practice: Time plot + STL intuition on a monthly demand series (8)
3. Concept+demo: Walk-forward CV with `TimeSeriesSplit` (9)
4. Guided practice: Build `lag1` / `lag12` features and CV a linear baseline (8)
5. Solution: Three-baseline comparison (naive, seasonal-naive, linear) on identical folds (8)
6. Solution: Open the locked test window — one-shot evaluation ceremony + bridge to nb17 (8)

**Notebook(s)**
- File: `nb16_time_series_forecasting_student.ipynb`
- Sections:
  - Synthetic 60-month DemandCo series (trend + annual seasonality + noise)
  - Time-respecting train/test split (last 12 months held out)
  - `TimeSeriesSplit(5)` walk-forward CV visualization
  - Lag features (`lag1`, `lag12`) construction and discussion of dropped early rows
  - Three-baseline comparison with Student's *t* 95% CIs
  - Locked-test-window opening ceremony with INSIDE / ABOVE / BELOW verdict

**In-notebook exercises (10-minute scope)**
- PAUSE-AND-DO 1: Cross-validate a linear regression on `[lag1, lag12]` using `TimeSeriesSplit(5)` and report the mean MAE.
- PAUSE-AND-DO 2: Add a `month_of_year` calendar feature and rerun the comparison; decide whether it beats the 2-lag baseline.

**Assessments**
- Concept quiz: forecasting vocabulary, walk-forward CV, lag-feature engineering

**Time budget (112.5 min)**
- Videos 50 + Notebook 45 + Quiz 7.5 + Reflection 10 = 112.5

**Bibliography**
- Hyndman & Athanasopoulos: *Forecasting: Principles and Practice* — [otexts.com/fpp3](https://otexts.com/fpp3/).
- scikit-learn User Guide: `TimeSeriesSplit` and time-series cross-validation.
- Course slides: `lecture_slides/08_time_series/`.

---

## Day 17 — Tue June 9
### Data communication and poster design: six principles applied to the eleven-section research-poster architecture
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
- File: `nb17_data_communication_poster_student.ipynb`
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
- Edward Tufte: *The Visual Display of Quantitative Information* (data-ink ratio).
- Cole Nussbaumer Knaflic: *Storytelling with Data*.
- Kieran Healy: *Data Visualization — A Practical Introduction*.
- Purdue Undergraduate Research Conference poster rubric and template.

---

## Day 18 — Wed June 10
### Competition workflow: end-to-end pipeline from notebook to Kaggle submission
**Learning objectives**
- Walk the Bank Churn case competition end-to-end through the eight workflow steps the rubric expects.
- Refactor modeling code into reusable functions (`build_pipeline`, `train_pipeline`, `predict_pipeline`).
- Save and reload the trained pipeline with `joblib`.
- Generate a `submission.csv` file in the exact column format the Kaggle leaderboard requires.
- Run the full pipeline on the locked Kaggle test set once at the end (the Day 20 final-submission ceremony).

**Micro-videos (50 min)**
1. Concept+demo: End-to-end workflow overview — the eight rubric steps (8)
2. Guided practice: `ColumnTransformer` preprocessor + baseline logistic regression (9)
3. Guided practice: Improved model (gradient boosting) + CV CI comparison (9)
4. Concept+demo: Refactor to `train_pipeline` / `predict_pipeline` functions (8)
5. Concept+demo: `joblib.dump` / `joblib.load` for portable artifacts (8)
6. Solution: Generate `submission.csv` with exact column names + leaderboard checklist (8)

**Notebook(s)**
- File: `nb18_competition_workflow_student.ipynb`
- Sections:
  - Load competition data (`train.csv` + unlabeled `test.csv`) and sanity checks
  - EDA snapshot (class balance, missingness, correlations)
  - `ColumnTransformer` preprocessor (numeric + categorical)
  - Baseline logistic regression + CV
  - Improved gradient-boosting model + CI-overlap rule
  - Refactor into `train_pipeline` / `predict_pipeline`
  - `joblib` save and reload
  - Generate `submission.csv` with exact Kaggle column names

**In-notebook exercises (10-minute scope)**
- PAUSE-AND-DO 1: Pick the competition champion based on CV CIs and justify in three sentences.
- PAUSE-AND-DO 2: Generate `submission.csv` and confirm the five-item leaderboard checklist passes.

**Assessments**
- Concept quiz: pipeline persistence, column-name contracts, covariate-shift diagnostics
- Participation: notebook submission with completed exercises + Kaggle leaderboard upload

**Time budget (112.5 min)**
- Videos 50 + Notebook 45 + Quiz 7.5 + Submission staging 10 = 112.5

**Bibliography**
- Chip Huyen: *Designing Machine Learning Systems* (chapters on serving and pipelines).
- scikit-learn User Guide: model persistence with `joblib`.
- Course case-competition starter pack: `_course_case_competition/2026Summer/`.

---

## Day 19 — Thu June 11
### Deep learning (awareness, hands-on PyTorch, when-to-use, one tabular demo, and a hands-on LLM lab)
**Learning objectives**
- Explain the historical arc that took neural networks from the 1980s rebrand into deep learning's 2010 resurgence and the three drivers (compute, data, frameworks).
- Describe what a single neuron, a single hidden layer, and a multi-layer perceptron compute, in plain language.
- Distinguish three deep-learning structural inventions — fully-connected MLPs, CNNs, and RNNs/Transformers — and name one problem class each is designed for.
- Decide whether deep learning is the right tool for a given business problem using a four-question rubric.
- Run a single MLP classifier on a familiar tabular dataset and compare it honestly to gradient boosting from nb13.

**Micro-videos (45 min)**
1. Concept+demo: Historical arc — 1980s rebrand to 2010 resurgence (8)
2. Concept+demo: PyTorch vs. TensorFlow — what the frameworks add over `numpy` (7)
3. Concept+demo: Single neuron → single layer → MLP (8)
4. Concept+demo: CNNs for images + RNNs/Transformers for sequences (8)
5. Guided practice: `MLPClassifier` vs. gradient boosting on tabular data (7)
6. Solution: Four-question rubric for "is deep learning right for this problem?" (7)

**Notebook(s)**
- File: `nb19_deep_learning_student.ipynb`
- Sections (focused 7-section structure, business-undergrad-friendly; figures used: `10_1_1-1.png` pioneers, `pytorch_vs_tensorflow.png`, `cifar100.png`, `2_7-1.png`):
  - **1. Deep Learning** — historical arc + three drivers + pioneers (with the three AI-visionary interviews as inline `YouTubeVideo` cells)
  - **2. PyTorch vs. TensorFlow** — what a framework gives you (tensors, autograd, GPU, layers), explained in business terms
  - **3. Neural Networks — build your intuition** — the four 3Blue1Brown *Deep Learning* chapters (what is a NN, gradient descent, backprop intuitively, backprop calculus), each embedded as an inline-playable `YouTubeVideo` cell with a text-link fallback
  - **⭐ 4. PyTorch hands-on, end-to-end on FashionMNIST** (the centerpiece; follows the official Quickstart = data + buildmodel + optimization + saveloadrun): load data + `DataLoader` + visualize grid → build `NeuralNetwork` (+ layer-by-layer + untrained preview) → train/test loop across epochs (CrossEntropyLoss + SGD) → save `state_dict` → load → predict (single image + random-photo grid colored by correctness); core tutorial pages embedded via `IFrame`
  - **5. When to Use Deep Learning** — three-shapes recap (MLP/CNN/RNN), flexibility-vs-interpretability, four-question rubric, and an honest neural-net vs. gradient-boosting tie on tabular data
  - **6. Large Language Models — concept + hands-on lab** — four intuition videos (clickable thumbnails) plus: 6.1 what an LLM is (next-token prediction, tokens, context window, temperature, hallucination), 6.2 the three ways analysts use LLMs + prompt rules, **6.3 a hands-on Hugging Face `transformers` lab** (a sentiment classifier and a zero-shot support-ticket router, run on the free Colab CPU with no API key), and **6.4 an optional `ask_llm` hosted-API helper** wrapping Purdue GenAI Studio (free, Purdue login), OpenAI (`gpt-4o-mini`), and Anthropic (`claude-haiku-4-5`) behind one `PROVIDER` toggle (plain `requests`, key-gated), then a **structured-JSON ticket-triage task built on 6.3b** (the model returns department + urgency + sentiment + summary + a drafted reply, parsed into a DataFrame) with markdown on how to adjust the schema, categories, model, and temperature
  - **7. Wrap-Up** — key takeaways + a scripted evidence-based answer for the VP
- **Note:** the ISLP conceptual sections (single-layer math, fitting, CNN, document classification, RNN, time-series forecasting) were dropped to keep the focus on the §4 hands-on experience; all YouTube videos render via `IPython.display.YouTubeVideo` (code cells) so they play inline in Colab rather than being stripped from markdown.

**In-notebook exercises (10-minute scope)**
- PAUSE-AND-DO 1: Apply the four-question rubric to the Bank Churn case competition and the DemandCo time-series forecast; produce one verdict each.
- PAUSE-AND-DO 2: Add a deeper MLP (`hidden_layer_sizes=(128,64,32)`) to the comparison; write three sentences answering "should we be using deep learning?" for the VP of Strategy.

**Assessments**
- Concept quiz: deep-learning vocabulary, structural inventions, when-to-use rubric

**Time budget (112.5 min)**
- Videos 45 + Notebook 50 + Quiz 7.5 + Reflection 10 = 112.5

**Bibliography**
- ISLP, Chapter 10: Deep Learning.
- PyTorch tutorials: <https://pytorch.org/tutorials/beginner/basics/intro.html>.
- Goodfellow, Bengio, Courville: *Deep Learning* — <https://www.deeplearningbook.org/>.
- 3Blue1Brown's neural-network video series.

---

## Day 20 — Fri June 12
### Course end and reflection: project package submission + peer review + reflection survey
**Learning objectives**
- Deliver a complete end-to-end predictive analytics package (M4 poster + Kaggle submission).
- Demonstrate reproducibility (run-all notebook, documented choices).
- Evaluate peers' work using a structured rubric and provide actionable feedback.
- Write a concise postmortem: what worked, what did not, what to do next.
- Submit the **course-end reflection survey** seeding next-summer's redesign.

**Micro-videos (30 min; 6×5 min)**
1. Final submission checklist (what graders check first) (5)
2. Guided practice: Run-all reproducibility audit (5)
3. Solution: Common submission failures + prevention (5)
4. Peer review rubric (how to be useful, not nice) (5)
5. Guided practice: High-signal feedback in 5 minutes (5)
6. Solution: Reflection survey walk-through + course-end Q&A (5)

**Notebook(s)**
- File: `nb20_final_submission_peer_review_student.ipynb`
- Sections:
  - Final self-audit checklist (run-all, outputs, links)
  - Submission links + artifact manifest (notebook, poster, joblib)
  - Peer review form (rubric + comment prompts)
  - Postmortem prompts (8–10 lines)
  - **Reflection survey** intro + pre-survey reflection seeds (10–15 min on Brightspace)

**In-notebook exercises (10-minute scope)**
- PAUSE-AND-DO: Run-all audit and fix one reproducibility issue (real or simulated).
- PAUSE-AND-DO: Complete one peer review with rubric scores + 3 actionable edits.
- PAUSE-AND-DO (5 min): Pre-survey reflection seeds before opening the survey link.

**Assessments**
- **Project Milestone 4 (due): Final Research Poster + intra-group Peer Evaluation**
  - Single PDF poster named `<group-number>.pdf` (e.g., `01.pdf`, `17.pdf`) — Brightspace template (Purdue URC poster format).
  - Each group member submits an individual confidential peer-evaluation form rating the other three teammates.
- **Kaggle Case Competition: Final submission** — leaderboard locks 11:59 PM.
- **Course-end Reflection Survey (required for completion)** — Brightspace link.
- Optional Fall 2026 conference presentation strongly encouraged (faculty mentorship available).

**Time budget (112.5 min)**
- Videos 30 + Notebook 45 + Quiz 7.5 + Submission/Survey 30 = 112.5

**Bibliography**
- Course rubric for M4 final poster.
- Purdue Undergraduate Research Conference: poster guidelines and dress code.

# Project bibliography (applies across all milestones)
- Provost & Fawcett: *Data Science for Business* (problem framing, value, evaluation)
- ISLP + Python labs (modeling, resampling, interpretation foundations)
- scikit-learn User Guide (pipelines, tuning, metrics, inspection)
- Mitchell et al. (Model Cards) + Barocas/Hardt/Narayanan (Fairness and ML) for limitations, risks, and responsible-use language
- Chip Huyen: *Designing Machine Learning Systems* (monitoring and deployment thinking)

