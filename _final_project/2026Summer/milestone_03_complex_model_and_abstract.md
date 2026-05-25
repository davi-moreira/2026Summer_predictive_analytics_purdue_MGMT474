# Final Project Milestone 03 — More Complex Model + Hyperparameter Tuning + Draft Abstract

## About the Final Project

The Final Project is a **group capstone** (groups of four randomly assigned) culminating in a **research poster**. Across the term your group completes four milestone deliverables (M1 → M4); each member also submits a confidential intra-group peer evaluation at the end. The project is worth **35% of your overall course grade**, broken down as:

- **Milestone Deliverables — 40%** of the project grade. Averaged across the four milestones (M1–M4). Graded for clarity, completeness, and timely submission.
- **Peer Evaluation — 20%** of the project grade. Confidential intra-group ratings collected at the end of the course.
- **Instructor / TA Evaluation — 40%** of the project grade. The final research poster, graded against the poster rubric (Milestone 04).

Optional presentation at the **Fall 2026 Purdue Undergraduate Research Conference** is strongly encouraged but **not required**. Professor Moreira is happy to serve as faculty mentor for groups choosing to present. Award-winning prior posters from this course: <https://davi-moreira.github.io/applied_projects.html>. Additional information about Purdue undergraduate research conferences: <https://www.purdue.edu/undergrad-research/conferences/index.php>.

---

## What to Submit on Brightspace

Submit **two files** per group on Brightspace by the posted deadline. One designated group member uploads on behalf of the whole group.

| # | File | Description |
|---|---|---|
| 1 | **`NN_complex_model.pdf`** *(e.g., for group 03, `03_complex_model.pdf`)* | Structured PDF report including the **draft abstract (~250 words)** as the opening section and **all required visualizations** — hyperparameter-search plot, model-comparison bar chart with 95% CI error bars, feature importance, plus regression diagnostics (predicted-vs-actual + residual) OR classification diagnostics (confusion matrix + ROC + PR curves) — embedded and captioned. |
| 2 | **`NN_complex_model.ipynb`** *(e.g., for group 03, `03_complex_model.ipynb`)* | Jupyter notebook with the runnable code. The notebook must execute top-to-bottom from a fresh `Runtime → Run all` with `random_state` locked everywhere, so M4 can re-execute the same modeling cell to reproduce the exact fitted Pipeline before opening the locked test set. |

Detailed format requirements (file types, exact Brightspace location) are in the **Submission Expectations** section near the bottom of this document.

---

## Purpose

Two threads come together at M3.

**Thread 1 — Modeling.** Move past the M2 baseline: introduce a more flexible model family (Random Forest, Gradient Boosting, SVM, or similar), tune hyperparameters with cross-validated GridSearch/RandomSearch, and decide whether the gains over the M2 baseline are real (CIs disjoint) or illusory (CIs overlap).

**Thread 2 — Communication.** Draft the **250-word project abstract** that will become the headline panel of the M4 poster. The abstract pins down the prediction problem, methodology, key findings, and broader implications in language an academic-and-industry audience can read in 90 seconds.

By the end of M3 the group should have: a champion model with a defensible CV CI, a clear interpretation of why it improves (or doesn't improve) on the baseline, and a polished draft abstract.

> **Supporting notebooks (nb11–nb15) cover both project tracks symmetrically.** Every algorithm taught in Week 3 is paired across classification (Wisconsin breast cancer / State Health Department example) and regression (California Housing / HomeValue Analytics example) under the `_clf` / `_reg` namespace convention. Each notebook benchmarks new models against the **Week-2 reference** floor (`LogReg(C=1.0)` for clf; `OLS` for reg) using the CI-overlap rule. Bring whichever spine matches your project type into your M3 work; the supporting notebooks demonstrate both.

## Components

### 0. Prediction Goal(s)

Restate the prediction goal(s). Explain why they matter in the context of your dataset. Confirm regression vs. classification.

### 1. Modeling Approach

#### 1a. Baseline Model (replicated from M2 — short summary)
- Model choice and one-paragraph justification
- Feature-selection method + k-fold CV (k=5 or 10)
- Headline metric with 95% CI
- Brief reflection on baseline strengths/limitations

#### 1b. More Complex Model — Implementation & Tuning
- **Model choice.** Pick a more complex family suited to the prediction goal (Random Forest, Gradient Boosting, SVM, …). Justify how it may capture patterns the baseline cannot (non-linearity, interactions, etc.).
- **Hyperparameter tuning with cross-validation.** Define a grid (or randomized search distribution) of hyperparameter values (tree depth, learning rate, number of estimators, regularization parameters, …). Use **5- or 10-fold CV inside `GridSearchCV` / `RandomizedSearchCV`** to compare configurations. Report results as a table or plot of CV metric across configurations.
- **Model selection & final comparison.** Identify the best hyperparameter combination by CV performance. Compare the tuned complex model against the M2 baseline with both CV CIs displayed. Apply the **CI-overlap rule**:
  - If CIs are **disjoint**, the complex model's gain is statistically distinguishable — adopt it as the champion.
  - If CIs **overlap**, prefer the simpler baseline (interpretability tiebreaker) unless an operational argument justifies the complex model.
- **Final-training step.** After selecting the champion via the CI-overlap rule, **retrain the chosen Pipeline on the full training fold (train + validation rows together)**. Lock `random_state` everywhere so the fitted Pipeline reproduces cleanly from a fresh `Runtime → Run all` — M4 will re-execute your modeling cell to recover the same Pipeline before opening the locked test set, so the model that scored the M3 CV CI is the one that touches the test set.

#### 1c. Required Visualizations

Embed each of the following in the report with axis labels, units where applicable, a clear legend, and a 1–2 sentence caption explaining what the figure shows:

- **Hyperparameter-search plot.** CV metric vs. hyperparameter value(s) — line plot for a 1-D grid, heatmap for a 2-D grid, or boxplot of per-fold scores across configurations. The selected best-hyperparameter point must be visually marked.
- **Model-comparison bar chart.** Two bars (M2 baseline vs. M3 complex champion), each showing the **CV mean with error bars representing the 95% Student's *t* CI**. This figure is the visual evidence behind your CI-overlap-rule decision.
- **Feature importance / coefficient plot.** Horizontal bar chart of the top features for the champion model — signed coefficients for linear/logistic; permutation importance or impurity importance for tree-based models.
- **For regression** problems:
  - **Predicted-vs-actual scatter plot** with a 45° reference line (`y = x`)
  - **Residual plot** (residuals vs. predicted) with a horizontal `y = 0` reference line
  - **RMSE-by-quintile bar chart** (in stakeholder units, e.g. USD) — optional but recommended when the prediction error has direct stakeholder-language interpretation
- **For classification** problems:
  - **Confusion matrix** at the chosen operating threshold (`ConfusionMatrixDisplay`)
  - **ROC curve with AUC annotation** and **Precision–Recall curve with PR-AUC annotation**
  - **Reliability diagram + Brier score** — optional but recommended if the model's probabilities feed a downstream decision threshold

### 2. Draft Abstract (~250 words)

Submit a polished draft abstract that will become the headline of your M4 poster.

**Required elements (in the order they appear):**

1. **Project title.** Concise and informative. *If the project uses a synthetic generated dataset, the title must say so explicitly.*
2. **Prediction problem (framed as a question with a "?").** Example: *"Can six-month customer-churn risk be predicted from transaction history and engagement metrics?"*
3. **Prediction goal and motivation.** One or two sentences: what you predict and why it matters.
4. **Methodology and tools.** A summary of the data preparation, analytical methods (baseline + complex model + tuning + CV protocol), and tools used (sklearn, etc.).
5. **Key findings / expected contributions.** A brief overview of preliminary findings or anticipated contributions of the analysis.
6. **Broader implications.** A statement on how the project informs business practice or contributes to the broader field of predictive analytics.

The abstract is the lead paragraph of the M4 poster. Treat it like a press release for your project: every sentence pulls weight.

---

## Submission Expectations

| Item | Specification |
|---|---|
| **Structured Report (PDF)** | Clearly labeled sections matching the components above; visualizations and tables embedded; key code snippets in the appendix or inline; the **draft abstract** as the report's opening section before the methodology |
| **Code Files** | Python script or Jupyter notebook (must run cleanly top-to-bottom) |
| **Submission location** | Brightspace |
| **Filename convention** | `NN_complex_model.pdf` and `NN_complex_model.ipynb` (e.g. group 03, `03_complex_model.pdf` and `03_complex_model.ipynb`) |

---

## Grading Rubric (100 points)

The rubric uses four performance levels per criterion. The point range for each level is scaled to the criterion's maximum:

- **Exemplary** ≥ 90% of the criterion's max
- **Proficient** 70–89% of the criterion's max
- **Developing** 50–69% of the criterion's max
- **Beginning** 0–49% of the criterion's max

| Criterion | Exemplary | Proficient | Developing | Beginning |
|---|---|---|---|---|
| **0. Prediction Goal(s)** (5 pts) | Goal clearly restated; regression-vs-classification confirmed; rationale tied to dataset context. | Goal restated; framing correct; rationale brief. | Goal restated but vague; framing inconsistent. | Goal missing or framed incorrectly. |
| **1a. Baseline Model** (20 pts) | M2 baseline replicated with model choice justified; feature selection + k-fold CV correctly implemented; CV score with **95% CI** reported; brief reflection on strengths/limits. | Baseline present with minor gaps in justification, implementation, or CI reporting. | Baseline implementation incomplete; CV details missing. | No baseline OR baseline incorrectly implemented. |
| **1b. Model Choice & justification** (8 pts) | More complex family (Random Forest, Gradient Boosting, SVM, …) chosen with clear rationale for why it can capture patterns the baseline cannot (non-linearity, interactions). | Complex model chosen; justification adequate but generalized. | Complex model chosen but justification weak or missing the baseline-comparison framing. | Model inappropriate, unstated, or trivially identical to the baseline. |
| **1b. Hyperparameter Tuning & Cross-Validation** (12 pts) | Defined grid (or randomized search distribution) of hyperparameter values; 5- or 10-fold CV inside `GridSearchCV` / `RandomizedSearchCV`; results shown as table or plot of CV metric across configurations; rationale for k value clear. | Tuning performed; minor gaps in grid documentation, plot, or k rationale. | Tuning incomplete; CV usage unclear; selection criteria undocumented. | No tuning OR test set used during the search (leakage). |
| **1b. Model Selection + Final-Training step** (10 pts) | CI-overlap rule applied correctly to choose the champion; champion **retrained on the full training fold**; `random_state` locked and the modeling cell reproduces the same fitted Pipeline from a fresh `Runtime → Run all` (so M4 recovers the same model). | Champion identified via CI-overlap; final-training step partially documented. | Champion identified but final-training step missing OR reproducibility not demonstrated. | No model-selection rationale OR no final-training step. |
| **1b. Comparison vs. baseline narrative** (5 pts) | Clear narrative comparing the M3 complex model to the M2 baseline with both CIs visible; honest assessment of whether the complexity is worth it. | Comparison present; narrative brief. | Comparison limited; CIs not displayed side-by-side. | No comparison OR misleading interpretation. |
| **1c. Required Visualizations** (20 pts) | All required figures present (hyperparameter-search plot, model-comparison bar chart with **95% CI error bars**, feature importance, **plus** regression diagnostics — predicted-vs-actual + residual — OR classification diagnostics — confusion matrix + ROC + PR curves); each labeled, captioned, and integrated into the narrative. | One required figure missing OR labeling/captioning gaps on one or two figures. | Two required figures missing OR multiple figures unlabeled/uncaptioned. | Three or more required figures missing OR all unlabeled. |
| **2. Draft Abstract (~250 words)** (15 pts) | All six elements present (title, prediction question with "?", motivation, methodology, key findings, broader implications); polished and accessible to an academic-and-industry audience; word count ~250. | Most elements present; minor gaps in polish or length. | Two or more elements missing or weakly developed; word count off. | Abstract missing OR fundamentally incomplete. |
| **3. Report Quality & Clarity** (5 pts) | Well-structured PDF; visualizations labeled; logical flow; error-free runnable code. | Generally clear with minor formatting issues. | Structure unclear; some labels missing. | Disorganized; code fails to run. |

**Total: 100 points** (the §1b More Complex Model block — Model Choice + Tuning + Selection/Final-Training + Comparison narrative — is worth **35 points** combined).

### Penalties

| Issue | Deduction |
|---|---|
| Filename does not follow the `NN_complex_model.pdf` / `NN_complex_model.ipynb` convention (e.g., for group 03, `03_complex_model.pdf` and `03_complex_model.ipynb`) | **−10 points** |

This rubric grade contributes to the **Milestone Deliverables (40%)** component of the Final Project grade — the average across all four milestones (M1–M4).

---

## Tips and Common Pitfalls

- **Tune on the training fold only.** GridSearchCV uses the training fold's CV; the held-out test set stays in the lockbox until M4.
- **Don't peek at the test set during the search.** Reporting "best test MAE at alpha=10" after evaluating all alphas on the test set is a textbook leakage failure.
- **Show the search.** A hyperparameter table or a plot of CV metric vs. hyperparameter value is the easiest way to demonstrate the search was systematic.
- **Honor the CI-overlap rule.** If the complex model's CI overlaps the baseline's CI, the simpler model wins — that's not a failure, that's a finding ("more complexity didn't help here, and we have evidence of that").
- **Write the abstract last.** Draft modeling section first; abstract synthesizes after the numbers settle.
- **The abstract IS the poster lead.** Polish it. Read it aloud. Three rounds of revision is the floor, not the ceiling.
- **Lock `random_state` everywhere so M3 → M4 reproduces cleanly.** M4's test-set ceremony will re-execute your M3 notebook's modeling cell to recover the fitted Pipeline, then open the locked test set ONCE. The model that scored the M3 CV CI must be the model that touches the test set — locking the seed is what makes that contract work without storing a binary artifact.
- **Required visualizations are part of the rubric.** The hyperparameter-search plot, the model-comparison bar chart with 95% CI error bars, the feature-importance plot, and the regression / classification diagnostic figures are graded under §1c (20 points). Drafting them while the modeling section is still open catches issues early.

---

**End of Milestone 03 instructions.**
