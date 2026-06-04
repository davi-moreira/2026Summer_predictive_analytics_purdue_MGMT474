# Implementation Plan: 2026 Summer Predictive Analytics Course Migration

## Overview
Transform the 2025F semester-based course into a 2026 Summer 4-week intensive format (20 business days) while maintaining GitHub Pages deployment and git best practices.

## Context Summary

### Current State
- **Repository:** `/Users/dcordeir/Dropbox/academic/cursos/cursos-davi/predictive_analytics/2026Summer_predictive_analytics_purdue_MGMT474`
- **Old format:** 16-week semester with Quarto slides in `/lecture_slides/`
- **New format:** 4-week intensive with Jupyter notebooks in `/notebooks/`
- **Status:** 17/20 notebooks created (Days 1-17 complete, Days 18-20 missing)
- **Course plan:** `MGMT47400_Online4Week_Plan_2026Summer.md` (complete, 956 lines)
- **GitHub Pages:** Uses Quarto → `docs/` → GitHub Pages
- **Old course URL:** https://davi-moreira.github.io/2025F_predictive_analytics_purdue_MGMT474/

### What Needs to Be Done
1. Complete missing notebooks (Days 18-20)
2. Update Quarto website configuration for new 20-day structure
3. Create new schedule page mapping to 20 business days
4. Update syllabus for 4-week format
5. Initialize local git repository
6. Connect to GitHub remote
7. Configure for GitHub Pages deployment
8. Create conversation log for future resumption

---

## Notebook Sequence and Content Justification

Each notebook builds exactly one conceptual layer, assumes only what prior notebooks have taught, and prepares exactly what the next notebook needs. The table below summarizes the rationale; full speaking prompts and cell-level detail are in each `video_guides/NN_video_lecture_guide.md` (Sections 1–3).

### Sequencing Map

| NB | Title | Key Libraries/Tools | Depends On | Prepares For | Why This Position |
|----|-------|---------------------|------------|--------------|-------------------|
| 00 | Launchpad: Course Setup | Google Colab, Google Gemini | — (first) | 01 (platform ready), all subsequent | Pre-course: Colab navigation, Gemini Ask→Verify→Document, course structure, environment verification |
| 01 | EDA & Splits | pandas, numpy, matplotlib, seaborn, train_test_split | 00 (platform ready) | 02 (pipeline), all subsequent | Foundation: statistical learning framework, EDA workflow, 60/20/20 split, leakage vocabulary |
| 02 | Preprocessing Pipelines | Pipeline, ColumnTransformer, SimpleImputer, OneHotEncoder, StandardScaler | 01 (split, leakage) | 03 (metrics assume pipeline solved) | Operationalizes leakage prevention with the tool that makes safe preprocessing automatic |
| 03 | Regression Metrics & Baselines | mean_absolute_error, mean_squared_error, r2_score, DummyRegressor | 02 (pipeline) | 04 (needs metrics to measure feature engineering impact) | Teaches how to measure model quality before attempting to improve it |
| 04 | Linear Features & Diagnostics | LinearRegression, PolynomialFeatures, make_pipeline | 03 (evaluation framework) | 05 (creates overfitting problem Ridge/Lasso solves) | Feature engineering + residual analysis; exposes polynomial overfitting |
| 05 | Regularization (Ridge/Lasso) | Ridge, Lasso, RidgeCV, LassoCV, ElasticNet | 04 (overfitting problem) | 06 (complete regression toolkit before classification pivot) | Direct solution to nb04's overfitting; closes Week 1 regression arc + project proposal |
| 06 | Logistic Regression & Pipelines | LogisticRegression, accuracy_score, log_loss | 05 (regularization via alpha → C) | 07 (needs probability foundations) | Regression → classification pivot; reuses Pipeline pattern in new context |
| 07 | Classification Metrics | confusion_matrix, precision/recall/f1, roc_curve, precision_recall_curve | 06 (probabilities, confusion matrix) | 08 (needs metric vocabulary for CV scoring) | Complete classification evaluation toolkit; cost-based threshold selection |
| 08 | Cross-Validation | cross_val_score, StratifiedKFold, cross_validate | 07 (metrics for scoring param) | 09 (CV embedded inside grid search) | Reliable model comparison replacing fragile single split |
| 09 | Tuning + Feature Eng. + Leakage Detection | GridSearchCV, RandomizedSearchCV, ColumnTransformer, FunctionTransformer, OneHotEncoder, SelectKBest | 08 (standalone CV) | 10 (midterm needs full pipeline template), 13 (leakage callout bridges here) | Three-section single file with opening toolkit-closer banner (cell 1, before LO): (A) grid search as nb08 × a grid with CI-overlap ranking + `C`-parameter primer; (B) TechCorp synthetic business case with real categoricals + target-encoding leak + SelectKBest leak detection; (C) Toolkit Recap — one-page reference consolidating concepts, workflow, sklearn primitives, decision rules across nb01–nb09 |
| 10 | Midterm Casebook + Cheat Sheet | — (strategy + reference card, no new libraries) | 01–09 (full toolkit) | 11 (pause before tree-based arc) | Strategic assessment; tests reasoning, not mechanics; **one-page cheat-sheet appendix** for open-during-midterm reference; project baseline due |
| 11 | Decision Trees (paired clf + reg) | DecisionTreeClassifier on breast cancer + DecisionTreeRegressor on California Housing, plot_tree, paired depth sweep with the plot_train_val_curve helper, predicted-vs-actual scatter | 10 (evaluation skills consolidated) | 12 (high-variance problem motivates forests; dual-spine pattern continues) | First non-linear model on both spines; concrete bias-variance demonstration via paired depth sweep; one-SE-rule depth selection on each spine; `_clf` / `_reg` namespace convention introduced here and used through nb15. **§7 verdicts:** classification keeps `LogReg(C=1.0)` (CI-clear win); regression switches to `DecisionTreeRegressor(max_depth=5)` under the **dominance tiebreaker** — tree CV mean 0.613 AND lower-CI 0.604 both above OLS's 0.586 and 0.554 → modest winner. OLS becomes the close-second runner-up; the tree is the new regression floor for nb12 |
| 12 | Random Forests + Importance (paired clf + reg) | RandomForestClassifier on breast cancer + RandomForestRegressor on California Housing; **joint 3D `(n_estimators × max_features × max_depth)` tuning grid** that **includes nb11's per-case `max_depth` picks (3 clf / 5 reg) as candidates alongside `None`**; CV mean + 95% CI half-width annotated per cell (red rectangle marks the largest-mean-smallest-CI cell per case); permutation_importance, OOB scoring on both, drop-column refit; four-method importance heatmap; CV-CI dot plot with Week-2 references as the floor | 11 (tree instability on both cases) | 13 (bagging baseline for boosting contrast), 15 (importance heatmap + PDP for interpretation) | Parallel ensemble solving single-tree variance on both cases; **four-method feature-importance reconciliation heatmap** (linear coef / MDI / permutation / drop-column); §5 joint 3D grid lands on `(n_estimators=50, max_features='sqrt', max_depth=3)` for clf (**nb11's depth survives** — all 27 cells tie under CI-overlap, parsimony picks the simplest) and `(n_estimators=50, max_features=0.5, max_depth=None)` for reg (**nb11's depth does NOT survive** — depth-3 and depth-5 cells are CI-clear below depth-None on California Housing); comprehensive comparison plot benchmarks single tree, forest, and Week-2 reference |
| 13 | Gradient Boosting (paired clf + reg) | GradientBoostingClassifier on breast cancer + GradientBoostingRegressor on California Housing, **joint 4D tuning grid** `n_estimators × learning_rate × max_features × max_depth` (54 cells per case across 12 panels, CV mean + 95% CI half-width annotated per cell, red rectangle marks the largest-mean-smallest-CI cell per case) that includes nb12's per-case RF shipped `(max_features, max_depth)` pairs as candidates | 12 (bagging baseline on both cases) | 14 (needs full candidate roster: Reference, Tree, RF, default GBM, tuned GBM per case) | Sequential ensemble on both cases; bias reduction vs variance reduction contrast; the 4D grid lands on **the same config for both cases** — `(learning_rate=0.2, n_estimators=200, max_features=0.5, max_depth=3)`; nb12's clf RF pick `(mf='sqrt', md=3)` transfers cleanly; nb12's reg RF pick `(mf=0.5, md=None)` does NOT transfer (unlimited depth overfits in boosting); tuned GBM ships regression as a **modest winner**: higher mean R² (0.8117 vs 0.8038), lower mean RMSE, ≈-tied MAE, AND higher CV CI lower bound (0.7983 vs 0.7937) than nb12's RF — the dominance tiebreaker breaks the upper-end CI overlap in the GBM's favor (the RF stays as the close-second runner-up); **leaky-features-dominate-boosting callout** referencing nb09 |
| 14 | Model Selection Protocol + Two Ceremonies + Post-Deployment Monitoring (paired clf + reg) | comparison harness (`compare_models_comprehensive`, post-hoc multi-metric CV using nb03/nb06/nb07 metric functions — no `neg_*` scoring strings), Student's *t* 95% CI helper, verdict helper, money plot for INSIDE/ABOVE/BELOW, `scipy.stats.ks_2samp` for the §7.4 drift-detection demo | 13 (full candidate pool on both cases), 08 (CI vocabulary) | 15 (champion committed for downstream tuning + deployment walkthrough) | Formal, fair, reproducible comparison on **both** cases; **7 classification candidates** (Dummy, LogReg(C=1.0), LogReg L1, Tree(d=3), RF(n=50, sqrt, d=3) — nb12 ship, GBM default, GBM tuned (lr=0.2, n=200, mf=0.5, d=3) — nb13 ship) and **8 regression candidates** (Dummy, OLS, Ridge, Lasso, Tree(d=5), RF(n=50, mf=0.5) — nb12 ship, GBM default, GBM tuned (lr=0.2, n=200, mf=0.5, d=3) — nb13 ship); **two parallel test-set ceremonies** with explicit singleness-rule callout (one ceremony per project, two in this notebook are pedagogical demos); **§6.3 verdict playbook** (INSIDE → ship + record CI / test point / retrain triggers; small-drift ABOVE → ship with documented gap from Step 1's train+val refit; large-drift ABOVE → investigate before deploying; BELOW → STOP, diagnose pipeline, test-set-is-spent rule requires a NEW test set before re-opening); champions: LogReg(C=1.0) clf (CI-overlap with tree ensembles → parsimony picks the linear baseline), **tuned GBM (lr=0.2, n=200, mf=0.5, d=3) reg** (modest winner over the RF on mean R², RMSE, MAE, AND CV CI lower bound — same pick as nb13 §7); **§7 post-deployment monitoring** — Silent decay (data drift / concept drift / population shift) vs Front-door arrivals; Three R's escalation ladder (Re-evaluate → Re-train → Rebuild); case-specific monitoring checklists (MedScreen: reliability + Brier, recall, KS drift on top-3 features, population composition; HomeValue: MAE/RMSE, errors by price tier, errors by location grid, KS drift on top features, major outside events); coded KS demo with alert on KS statistic (not p-value); one-page Monitoring Card; `audit_cv_first.py` exception list updated for both ceremony cells |
| 15 | Final Project Milestone 03 Walkthrough — Complex Model + Hyperparameter Tuning + Draft Abstract | **Markdown-only walkthrough** (no code, no PAUSE-AND-DO) — companion read for the `milestone_03_complex_model_and_abstract.md` rubric | 14 (champion committed), 09 (hyperparameter tuning), 08 (CI-overlap rule) | 16 (time-series CV splitter pivot) | Same shape as nb05 §6/§7 and nb10's milestone sections. Sections map directly to the M3 rubric: §0 Prediction Goals, §1a Baseline Replication (M2) with 95% CI, §1b Complex Model + Tuning + CI-overlap rule + final-training step on the full training fold (with random_state locked so M4 reproduces the same fitted Pipeline), §1c Required Visualizations, §2 Draft Abstract (~250 words), Course Case Competition (Kaggle Bank Churn) alignment, and Tips and Common Pitfalls. **No Interpretation, Calibration, or Decision-Quality content.** |
| 16 | Time-Series Forecasting | TimeSeriesSplit, lag features (`pd.shift`), LinearRegression | 15 (closes static-classification arc), 08 (CV CIs), 14 (locked-test ceremony pattern) | 17 (forecast as a candidate poster figure) | Forecasting ≠ generic supervised: never shuffle, walk-forward CV, lag features, three-baseline comparison (naive / seasonal-naive / linear-with-lags) on identical CV folds, one-shot opening of the locked test window. |
| 17 | Data Communication & Poster Design (formerly nb19) | — (markdown/narrative, no new libraries) | 15+16 (headline numbers, CV-CI, calibration, locked-test verdict, forecast comparison) | 18 (poster outline before competition packaging), 20 (poster + abstract feed M4) | Six principles applied to the eleven-section URC poster architecture; chart audit + outline + 120–150-word abstract drafted in studio. |
| 18 | Competition Workflow & Kaggle Submission | ColumnTransformer, Pipeline, GradientBoostingClassifier, joblib, pandas.to_csv | 17 (poster outline locked) | 19 (gradient-boosted tabular champion as the comparison anchor for DL) | End-to-end production pipeline for the Bank Churn case competition: `train_pipeline` / `predict_pipeline` refactor → `joblib` save/load → `submission.csv` with exact Kaggle column names. The Kaggle test set is unlabeled — `predict_proba(X_test)` is production prediction, not model evaluation (audit_cv_first.py exception). |
| 19 | Deep Learning | PyTorch (`torch`/`torchvision`) for the §4 FashionMNIST lab; `sklearn.neural_network.MLPClassifier`/`MLPRegressor` for the §5 comparison against nb14's champions; **Hugging Face `transformers`** for the §6 LLM lab (sentiment + zero-shot pipelines); `requests` for the optional Purdue GenAI Studio API call; figures from `notebooks/figures/` | 18 (gradient-boosted tabular champion) | 20 (course-end horizon module) | Awareness + hands-on module: historical arc, frameworks, MLP / CNN / RNN / Transformer structural inventions, **end-to-end PyTorch training on FashionMNIST**, four-question rubric for "is DL right for this problem?", an honest deep-learning-vs-nb14-champion comparison on both business cases (Breast Cancer classification and California Housing regression) with cross-validation confidence-interval plots and per-case verdicts, and a **special topic + hands-on lab on Large Language Models (LLMs)** — run a sentiment classifier and a zero-shot ticket router with no API key, then optionally call a hosted model via Purdue GenAI Studio. Designed for business-undergrad audience. |
| 20 | Course End and Reflection | — (audit + review + survey link) | 19 (awareness arc closed) | — (last) | Self-audit, M4 poster + Kaggle submission + intra-group peer evaluation, postmortem, **course-end Reflection Survey** (10–15 min on Brightspace, required for completion). |

### Weekly Arc Pattern

Each week follows: introduce capability → build evaluation skills → practice integration → deliver milestone.

- **Week 1 (Regression):** EDA/Splits → Pipelines → Metrics → Features → Regularization → *Proposal + Kaggle Launch (Day 5)*
- **Week 2 (Classification):** LogReg → Metrics → CV → Tuning + Leakage → *Midterm + Baseline + Cheat Sheet + Kaggle Check-in (Day 10)*
- **Week 3 (Ensembles):** Trees + class_weight → Forests + Importance Table → Boosting + Leak Callout → Selection + Test Set Ceremony → *M3 Walkthrough — Complex Model + Tuning + Abstract (Day 15)*
- **Week 4 (Production):** Calibration → Fairness → Deployment + Kaggle Submission → Narrative → *Final Submission + Kaggle Leaderboard Reveal (Day 20)*

> **Cross-reference:** For full speaking prompts, cell references, and timestamps, see `video_guides/NN_video_lecture_guide.md` Sections 1–3 (Why exists, Why after N-1, Why before N+1).

---

## Phase 1: Complete Notebook Creation (Days 18-20)

### Files to Create

#### 1.1 Day 18: `notebooks/nb18_reproducibility_monitoring.ipynb`
**Source:** MGMT47400_Online4Week_Plan_2026Summer.md lines 812-856
**Structure:**
- Header with Colab badge
- Learning objectives (5 items: packaging, saving/loading, monitoring, checklists, reproducibility)
- Section 1: Setup (imports: joblib, Pipeline, StandardScaler)
- Section 2: Refactor notebook into functions
  - `train_model(config)` → returns pipeline + metrics dict
  - `predict(model, X)` → returns predictions
  - `evaluate(model, X, y)` → returns metrics
- Section 3: Save/load model artifacts
  - Use joblib to save pipeline
  - Load and verify reproducibility
- Section 4: Monitoring plan template
  - Data drift signals (feature distribution changes)
  - Performance drift (metric degradation)
  - Calibration drift
  - Table with Signal | Threshold | Owner | Action
- Section 5: "Ready-to-share" notebook hygiene checklist
- Section 6: PAUSE-AND-DO exercises (2 exercises)
  - Exercise 1: Implement `train_model(config)` returning pipeline + metrics (10 min)
  - Exercise 2: Draft monitoring plan with 5-8 signals and owners (10 min)
- Section 7: Wrap-up (key takeaways, critical rules)
- Bibliography (Chip Huyen, scikit-learn User Guide, Dataset Shift papers)

#### 1.2 Day 19: `notebooks/nb19_data_communication_poster.ipynb`
**Source:** MGMT47400_Online4Week_Plan_2026Summer.md (Day 19 block); mirrors `lecture_slides/09_data_communication_poster/09_data_communication_poster.qmd`
**Structure:**
- Header with Colab badge
- Learning objectives (5 items: apply six principles, diagnose chart failures, raise data-ink ratio, plan poster layout, draft outline + abstract)
- Why-This-Matters cell (named stakeholder: URC faculty mentor)
- Section 1: The Forest and the Trees (`floresta.jpg`)
- Section 2: Six Principles overview table
- Section 3: Context Matters (`contexto-add.png`, `contexto-obs.png`)
- Section 4: Visualization Derives From Data (table-vs-plot, scale fail, dual/triple axes, pie-chart abuse, graph galleries)
- Section 5: Less Is More — eight-step data-ink-ratio cleanup walk-through (`limpeza-1` → `limpeza-8`)
- Section 6: Hierarchy Among Data (count-the-3s, accent-color highlighting)
- Section 7: Beauty Counts (emphasis with size + color)
- Section 8: Telling Your Story — five-pass annotation walk-through (`final-1`→`final-5`) and nine-pass time-series walk-through (`hist-1`→`hist-9`)
- **PAUSE-AND-DO Exercise 1 (8 min):** Audit one project figure against the six principles; produce a three-bullet rebuild plan
- Section 9: Why a Poster Presentation? (`project-history.jpg`)
- Section 10: Designing Objectives (`paper-abstract.jpg`)
- Section 11: Template & Rubric (URC poster template + course rubric pointer)
- Sections 12–15: Visual hierarchy, layout & design, content organization (eleven sections), predictive-analytics-specific design
- Section 16: Crafting a Clear Narrative (`how-to-write-good-02.png`, `thesis_word_count_02.gif`)
- Section 17: Research Design Flow (`research_design_flow.jpg`)
- Sections 18–22: Effective figures and tables, results & interpretation, conclusion & future work, final touches, presenting at URC
- **PAUSE-AND-DO Exercise 2 (15 min):** Eleven-section poster outline + 120–150-word abstract draft
- Section 23: Additional Material (Flowing Data, Information is Beautiful, The Functional Art, FT COVID coverage)
- Section 24: Wrap-Up + bridge to nb20
- Submission Instructions
- Bibliography (Tufte, Healy, Knaflic, Kastellec & Leoni, URC rubric)

#### 1.3 Day 20: `notebooks/nb20_final_submission_peer_review.ipynb`
**Source:** `_final_project/2026Summer/milestone_04_final_poster.md` + `_course_case_competition/2026Summer/course_case_competition_instructions.md`
**Structure (markdown-only milestone walkthrough, nb15 style — no code cells; instructor == student):**
- Header with Colab badge
- Learning objectives (5 items: M4 poster + final notebook, the one-shot test ceremony, the 100-pt rubric, the Kaggle final submission, the two individual closeouts)
- Why This Matters (delivery day; the one authorized opening of the locked test set)
- Section 1: About the Final Project (35% of course grade — 40% milestones / 20% peer eval / 40% instructor-TA poster; optional PURC)
- Section 2: What to Submit at M4 (`NN.pdf` poster + `NN_final.ipynb`; -10 filename penalty)
- Section 3: The Poster — Required Components (10 components, Intro→Methods→Results→Conclusions)
- Section 4: Required Visualizations (model-comparison-with-CI, test-set verdict, feature importance, regression/classification diagnostics)
- Section 5: The One-Shot Test-Set Evaluation (reload M3 `champion_pipeline.joblib` → open test once → INSIDE/ABOVE/BELOW verdict → report)
- Section 6: The M4 Poster Rubric (100 pts across 7 criteria + -10 filename penalty)
- Section 7: Course Case Competition — Final Kaggle Submission (ROC-AUC, `Group NN` naming, 5 submissions/day, trust the CV CI not the public leaderboard)
- Section 8: Course Closeout — confidential peer evaluation (20%) + required reflection survey (4 areas)
- Section 9: Tips and Common Pitfalls
- Participation Assignment Submission Instructions (+ closeout checklist; terminal — no Next Step notebook)
- Bibliography (Provost & Fawcett; ISLP; scikit-learn common pitfalls)

**Pattern Consistency:**
Follows the milestone-walkthrough template established by nb05/nb10/nb15:
- Colab badge in header
- Markdown only (no code cells, no PAUSE-AND-DO exercises); instructor and student files are identical
- Walks the milestone rubric in grading order, with point values surfaced
- Why-This-Matters → numbered rubric sections → Tips → Submission Instructions → Bibliography → Thank you
- Docstrings for any functions
- Blockquotes for critical rules
- ✓ Checkmarks for confirmations

---

## Phase 2: Update Quarto Website Structure

### 2.1 Update `_quarto.yml`
**File:** `_quarto.yml`
**Changes:**
- Keep existing structure (website, docs output)
- Update title to: "QM47400: Predictive Analytics (Summer 2026 - 4-Week Intensive)"
- Update GitHub link to new repo name (TBD)
- Keep sidebar structure (Home, Syllabus, Schedule and Material)

### 2.2 Update `index.qmd` (Homepage)
**File:** `index.qmd`
**Changes:**
- Update course dates: "May 18 - June 12, 2026 (20 business days)"
- Update format description: "4-week fully online intensive"
- Update course structure:
  - 20 business days (Mon-Fri)
  - 112.5 minutes per day engagement
  - Micro-videos (≤12 min each) + Google Colab notebooks
  - Single capstone project with 4 weekly milestones
- Update instructor info if needed
- Link to new GitHub repo

### 2.3 Create New `schedule.qmd`
**File:** `schedule.qmd`
**Strategy:** Replace 16-week semester schedule with 20-day intensive schedule
**Structure:**
- Introduction: 20 business days, May 18 - June 12, 2026
- Weekly breakdown with daily rows
- Columns:
  - **Day** (1-20)
  - **Date** (May 18 - June 12, business days only)
  - **Topic** (from course plan)
  - **Videos** (micro-video count + total time)
  - **Notebook** (link to Colab-ready notebook on GitHub)
  - **Quiz/Assessment** (auto-graded quiz or milestone)
  - **Materials** (bibliography references)

**Table structure:**
```markdown
| Day | Date | Topic | Videos | Notebook | Assessment | Materials |
|-----|------|-------|--------|----------|------------|-----------|
| 0 | Pre-course | Launchpad: Course Setup, Colab Orientation | 2 videos (10 min) | [00_launchpad](link) | Colab Readiness Check | Colab docs |
| 1 | Mon May 18 | PA Fundamentals, EDA, Splits | 5 videos (48 min) | [01_eda_splits](link) | Concept Quiz | ISLP Ch2, sklearn |
| 2 | Tue May 19 | Preprocessing Pipelines | 6 videos (54 min) | [02_preprocessing](link) | Concept Quiz | sklearn Pipelines |
...
```

**Project milestones highlighted (groups of four randomly assigned members; canonical reference at `_final_project/2026Summer/final_project_milestone_reference.md`):**
- Day 5: M1 Initial Project Proposal due
- Day 10: M2 Simple Model + Performance Evaluation + Midterm due
- Day 15: M3 More Complex Model + Tuning + Draft Abstract due
- Day 20: M4 Final Research Poster + intra-group Peer Evaluation due (`<group-number>.pdf` per Purdue Undergraduate Research Conference poster format; optional Fall 2026 conference presentation)

### 2.4 Update `syllabus.qmd`
**File:** `syllabus.qmd`
**Changes:**
- Update course title: "QM47400 - Predictive Analytics (Summer 2026, 4-Week Online Intensive)"
- Update dates: May 18 - June 12, 2026
- Update course description:
  - 20 business days
  - 112.5 minutes daily engagement
  - Micro-videos + Colab notebooks
  - Single capstone project
- Update grading breakdown:
  - Daily quizzes: 20% (20 quizzes × 1%)
  - Midterm (Day 10): 15%
  - Project milestones:
    - Proposal (Day 5): 5%
    - Baseline (Day 10): 10%
    - Improved model (Day 15): 15%
    - Final deliverable (Day 20): 25%
  - Peer review (Day 20): 10%
- Update calendar/schedule section to reference 20-day structure
- Update textbooks (same: ISLP, ESL, Provost & Fawcett)
- Update technology requirements:
  - Google Colab (primary platform)
  - Google Gemini (AI assistance)
  - Brightspace LMS
- Add daily engagement expectations section

---

## Phase 3: Git Repository Setup

### 3.1 Initialize Local Git Repository
**Location:** Current directory
**Steps:**
1. Check if `.git/` already exists (it does based on git status output)
2. If exists, clean up current state
3. Create comprehensive `.gitignore`

### 3.2 Create/Update `.gitignore`
**File:** `.gitignore`
**Content:**
```gitignore
# R/RStudio
.Rproj.user
.Rhistory
.RData
.Ruserdata
*.Rproj

# Quarto
/.quarto/
_freeze/

# Python
__pycache__/
*.py[cod]
*$py.class
.ipynb_checkpoints/
*.pyc
.venv/
venv/
env/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Admin (keep out of public repo)
_adm_stuff/
_grades/
_students_contact/
_tas/
_accomodation/
_course_eval/
_reflection_*/

# Large files (exclude from repo, link instead)
*.zip
*.mp4
*.mp3
*.m4a
*.mov

# Keep docs for GitHub Pages
!docs/

# Temporary
*.tmp
*.bak
.scratch/
```

### 3.3 Initial Commit Structure
**Strategy:** Atomic commits for different components

**Commit sequence:**
1. `docs: Add course plan for 2026 Summer intensive`
   - Add `MGMT47400_Online4Week_Plan_2026Summer.md`

2. `feat: Add Days 1-17 notebooks for 4-week intensive`
   - Add all 17 existing notebooks in `notebooks/`

3. `feat: Add Days 18-20 notebooks (reproducibility, narrative, final)`
   - Add three newly created notebooks

4. `docs: Update Quarto website for 2026 Summer format`
   - Update `_quarto.yml`, `index.qmd`, `schedule.qmd`, `syllabus.qmd`

5. `chore: Add .gitignore for project`
   - Add `.gitignore`

6. `build: Render Quarto site for GitHub Pages`
   - Render site with `quarto render`
   - Commit updated `docs/` directory

**Commit message format:**
```
<type>: <subject>

<body>

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

---

## Phase 4: GitHub Remote Connection

### 4.1 Create GitHub Repository
**Method:** Use `gh` CLI (GitHub CLI)
**Steps:**
1. Check if `gh` is authenticated: `gh auth status`
2. Create new repo: `gh repo create 2026Summer_predictive_analytics_purdue_MGMT474 --public --description "QM47400 Predictive Analytics - 4-Week Summer Intensive (2026)" --source=. --remote=origin`
3. Verify remote: `git remote -v`

**Repository settings:**
- **Visibility:** Public
- **Description:** "QM47400 - Predictive Analytics | 4-Week Online Intensive | Purdue Daniels School of Business | Python/scikit-learn | Colab notebooks"
- **Topics:** `predictive-analytics`, `machine-learning`, `python`, `jupyter-notebook`, `scikit-learn`, `data-science`, `mba-course`, `colab`
- **Homepage:** (Will be set after GitHub Pages is configured)

### 4.2 Push to Remote
**Steps:**
1. `git branch -M main` (ensure main branch)
2. `git push -u origin main` (push with upstream tracking)

---

## Phase 5: GitHub Pages Configuration

### 5.1 Repository Settings
**Method:** Use `gh` CLI or web interface
**Configuration:**
- **Source:** Deploy from `docs/` directory on `main` branch
- **Custom domain:** None (use default `*.github.io`)
- **HTTPS:** Enforce HTTPS

**CLI command:**
```bash
gh repo edit --enable-pages --pages-branch main --pages-path docs
```

### 5.2 Verify Deployment
**Expected URL:** `https://davi-moreira.github.io/2026Summer_predictive_analytics_purdue_MGMT474/`

**Verification steps:**
1. Check GitHub Pages build status: `gh run list --workflow pages-build-deployment`
2. Wait for deployment (usually 1-2 minutes)
3. Visit URL and verify:
   - Homepage loads
   - Sidebar navigation works
   - Schedule page shows 20-day structure
   - Notebook links work (should point to GitHub raw notebooks)

### 5.3 Update Repository Homepage
**Command:**
```bash
gh repo edit --homepage "https://davi-moreira.github.io/2026Summer_predictive_analytics_purdue_MGMT474/"
```

---

## Phase 6: Documentation and Conversation Log

### 6.1 Create `CONVERSATION_LOG.md`
**File:** `CONVERSATION_LOG.md` (in repository root)
**Purpose:** Track all work done and decisions made for future session resumption

**Structure:**
```markdown
# Conversation Log: 2026 Summer Course Development

## Session 1: January 27, 2026

### Objective
Transform 2025F semester-based course into 2026 Summer 4-week intensive format (20 business days)

### Context
- **User:** Professor Davi Moreira
- **Course:** QM47400 - Predictive Analytics
- **Institution:** Purdue Daniels School of Business
- **Old format:** 16-week semester (2025 Fall)
- **New format:** 4-week intensive (May 18 - June 12, 2026, 20 business days)

### Work Completed

#### 1. Notebooks Created
- **Days 1-17:** Already existed (created earlier in session)
- **Days 18-20:** Created new notebooks:
  - `nb18_reproducibility_monitoring.ipynb` - Deployment thinking, packaging, monitoring
  - `nb19_data_communication_poster.ipynb` - Six principles of data communication + eleven-section poster architecture (URC template)
  - `nb20_final_submission_peer_review.ipynb` - M4 poster + one-shot test ceremony, Kaggle final submission, peer evaluation, reflection survey

#### 2. Quarto Website Updated
- `_quarto.yml` - Updated title and configuration
- `index.qmd` - Updated for 4-week intensive format
- `schedule.qmd` - Complete rewrite for 20-day structure
- `syllabus.qmd` - Updated grading, calendar, format

#### 3. Git Repository
- Initialized local repository
- Created comprehensive `.gitignore`
- Made atomic commits for each component

#### 4. GitHub Connection
- Created remote repository
- Pushed all commits to main branch
- Configured GitHub Pages

### Key Decisions

1. **Notebook organization:** Keep flat structure in `/notebooks/` (01-20), don't nest by week
2. **Website deployment:** Continue using Quarto → docs/ → GitHub Pages (same as 2025F)
3. **Git strategy:** Atomic commits by component, maintain clean history
4. **Admin materials:** Keep `_adm_stuff/` but exclude from git
5. **Large files:** Link to external storage (Google Drive), don't commit to git

### File Structure
```
Repository Root
├── notebooks/                         # 20 Jupyter notebooks (Days 1-20)
├── docs/                              # GitHub Pages output (compiled by Quarto)
├── images/                            # Course logo and assets
├── _quarto.yml                        # Quarto configuration
├── index.qmd                          # Homepage
├── schedule.qmd                       # 20-day schedule table
├── syllabus.qmd                       # Course syllabus
├── styles.css                         # Custom styling
├── MGMT47400_Online4Week_Plan_2026Summer.md  # Master course plan
├── CONVERSATION_LOG.md                # This file
├── .gitignore                         # Git ignore rules
└── 2026Summer_predictive_analytics_purdue_MGMT474.Rproj  # RStudio project file
```
```

### 6.2 Create `README.md`
**File:** `README.md` (in repository root)
**Purpose:** Public-facing repository documentation

---

## Critical Files Summary

### Files to Create (New)
1. `notebooks/nb18_reproducibility_monitoring.ipynb`
2. `notebooks/nb19_data_communication_poster.ipynb`
3. `notebooks/nb20_final_submission_peer_review.ipynb`
4. `CONVERSATION_LOG.md`
5. `README.md`
6. `.gitignore` (update existing or create)

### Files to Update (Existing)
1. `_quarto.yml` (minimal - update title and GitHub link)
2. `index.qmd` (update course dates, format, structure)
3. `schedule.qmd` (complete rewrite for 20-day structure)
4. `syllabus.qmd` (update grading, calendar, format)

### Files to Keep (No Changes)
1. All 17 existing notebooks (Days 1-17)
2. `MGMT47400_Online4Week_Plan_2026Summer.md` (master plan)
3. `styles.css` (custom CSS)
4. `images/` directory (logos, assets)
5. `2026Summer_predictive_analytics_purdue_MGMT474.Rproj`

---

## Success Criteria

### Must Have (Blocking)
- [ ] All 20 notebooks exist and have consistent structure
- [ ] Git repository initialized with clean history
- [ ] GitHub remote connected and pushed
- [ ] GitHub Pages deployed and accessible
- [ ] Schedule page shows all 20 days with correct dates
- [ ] Syllabus updated for 4-week format
- [ ] All Colab badges work in notebooks

### Should Have (High Priority)
- [ ] Conversation log created for future resumption
- [ ] README.md with comprehensive documentation
- [ ] All notebook links tested in Colab
- [ ] Website responsive on mobile

### Nice to Have (Future Work)
- [ ] Auto-graded quizzes created in Brightspace
- [ ] Micro-videos recorded and linked
- [ ] Sample project deliverable created
- [ ] All notebooks tested end-to-end in Colab

---

## Implementation Order

1. **Complete Notebooks** (Phase 1) - Days 18-20
2. **Update Quarto Website** (Phase 2) - _quarto.yml, index, schedule, syllabus
3. **Git Setup** (Phase 3) - .gitignore, commits
4. **GitHub Connection** (Phase 4) - create repo, push
5. **GitHub Pages** (Phase 5) - configure, verify
6. **Documentation** (Phase 6) - conversation log, README
7. **Verification** - Test everything works

---

## End of Plan

This plan provides a complete roadmap for migrating the 2025F semester course to the 2026 Summer 4-week intensive format while maintaining git best practices and GitHub Pages deployment.
