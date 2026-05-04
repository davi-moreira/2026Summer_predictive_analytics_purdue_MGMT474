# Milestone 03 — More Complex Model + Hyperparameter Tuning + Draft Abstract

## About the Final Project

The Final Project is a **group capstone** (groups of four randomly assigned) culminating in a **research poster**. Across the term your group completes four milestone deliverables (M1 → M4); each member also submits a confidential intra-group peer evaluation at the end. The project is worth **35% of your overall course grade**, broken down as:

- **Milestone Deliverables — 40%** of the project grade. Averaged across the four milestones (M1–M4). Graded for clarity, completeness, and timely submission.
- **Peer Evaluation — 20%** of the project grade. Confidential intra-group ratings collected at the end of the course.
- **Instructor / TA Evaluation — 40%** of the project grade. The final research poster, graded against the poster rubric (Milestone 04).

Optional presentation at the **Fall 2026 Purdue Undergraduate Research Conference** is strongly encouraged but **not required**. Professor Moreira is happy to serve as faculty mentor for groups choosing to present. Award-winning prior posters from this course: <https://davi-moreira.github.io/applied_projects.html>. Additional information about Purdue undergraduate research conferences: <https://www.purdue.edu/undergrad-research/conferences/index.php>.

---

## Purpose

Two threads come together at M3.

**Thread 1 — Modeling.** Move past the M2 baseline: introduce a more flexible model family (Random Forest, Gradient Boosting, SVM, or similar), tune hyperparameters with cross-validated GridSearch/RandomSearch, and decide whether the gains over the M2 baseline are real (CIs disjoint) or illusory (CIs overlap).

**Thread 2 — Communication.** Draft the **250-word project abstract** that will become the headline panel of the M4 poster. The abstract pins down the prediction problem, methodology, key findings, and broader implications in language an academic-and-industry audience can read in 90 seconds.

By the end of M3 the group should have: a champion model with a defensible CV CI, a clear interpretation of why it improves (or doesn't improve) on the baseline, and a polished draft abstract.

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
- **Final-training step.** After selecting the champion via the CI-overlap rule, **retrain the chosen Pipeline on the full training fold (train + validation rows together)** and save the fitted pipeline as `champion_pipeline.joblib` plus a `CONFIG.json` recording features, hyperparameters, and the selection date. The saved artifacts must be reproducible from a fresh "Run All".

#### 1c. Required Visualizations

Embed each of the following in the report with axis labels, units where applicable, a clear legend, and a 1–2 sentence caption explaining what the figure shows:

- **Hyperparameter-search plot.** CV metric vs. hyperparameter value(s) — line plot for a 1-D grid, heatmap for a 2-D grid, or boxplot of per-fold scores across configurations. The selected best-hyperparameter point must be visually marked.
- **Model-comparison bar chart.** Two bars (M2 baseline vs. M3 complex champion), each showing the **CV mean with error bars representing the 95% Student's *t* CI**. This figure is the visual evidence behind your CI-overlap-rule decision.
- **Feature importance / coefficient plot.** Horizontal bar chart of the top features for the champion model — signed coefficients for linear/logistic; permutation importance or impurity importance for tree-based models.
- **For regression** problems:
  - **Predicted-vs-actual scatter plot** with a 45° reference line (`y = x`)
  - **Residual plot** (residuals vs. predicted) with a horizontal `y = 0` reference line
- **For classification** problems:
  - **Confusion matrix** at the chosen operating threshold (`ConfusionMatrixDisplay`)
  - **ROC curve with AUC annotation** and **Precision–Recall curve with PR-AUC annotation**

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
| **Submission location** | Brightspace — Module 3, Final Project Milestone 03 |
| **Filename convention** | `group-<NN>_M03_complex_model.pdf` and `group-<NN>_M03_complex_model.ipynb` |

---

## Grading Rubric (100 points)

| Criterion | Points |
|---|---:|
| **0. Prediction Goal(s)** — Clearly stated, connected to dataset context, regression vs. classification confirmed | **5** |
| **1a. Baseline Model** — Model choice & justification; implementation with feature selection + k-fold CV; interpretation | **20** |
| **1b. More Complex Model — Implementation & Tuning** | **35** |
| &nbsp;&nbsp;&nbsp;&nbsp;Model Choice & justification (8) | |
| &nbsp;&nbsp;&nbsp;&nbsp;Hyperparameter Tuning & Cross-Validation (12) | |
| &nbsp;&nbsp;&nbsp;&nbsp;Model Selection (CI-overlap rule) + Final-Training step + saved `champion_pipeline.joblib` (10) | |
| &nbsp;&nbsp;&nbsp;&nbsp;Comparison vs. baseline narrative (5) | |
| **1c. Required Visualizations** — Hyperparameter-search plot, model-comparison bar chart with 95% CI error bars, feature importance plot, plus regression diagnostics (predicted-vs-actual + residual) OR classification diagnostics (confusion matrix + ROC + PR curves). All figures labeled and captioned. | **20** |
| **2. Draft Abstract (~250 words)** — title, prediction question with "?", motivation, methodology, key findings, broader implications | **15** |
| **3. Report Quality & Clarity** — Well-structured PDF, labeled visualizations, logical flow, error-free code | **5** |
| **Total** | **100** |

This rubric grade contributes to the **Milestone Deliverables (40%)** component of the Final Project grade — the average across all four milestones (M1–M4).

---

## Tips and Common Pitfalls

- **Tune on the training fold only.** GridSearchCV uses the training fold's CV; the held-out test set stays in the lockbox until M4.
- **Don't peek at the test set during the search.** Reporting "best test MAE at alpha=10" after evaluating all alphas on the test set is a textbook leakage failure.
- **Show the search.** A hyperparameter table or a plot of CV metric vs. hyperparameter value is the easiest way to demonstrate the search was systematic.
- **Honor the CI-overlap rule.** If the complex model's CI overlaps the baseline's CI, the simpler model wins — that's not a failure, that's a finding ("more complexity didn't help here, and we have evidence of that").
- **Write the abstract last.** Draft modeling section first; abstract synthesizes after the numbers settle.
- **The abstract IS the poster lead.** Polish it. Read it aloud. Three rounds of revision is the floor, not the ceiling.
- **Save the `champion_pipeline.joblib` AT M3.** M4's test-set ceremony loads this exact saved artifact and evaluates it once. Don't refit silently between M3 and M4 — the model that scored the M3 CV CI must be the model that touches the test set.
- **Required visualizations are part of the rubric.** The hyperparameter-search plot, the model-comparison bar chart with 95% CI error bars, the feature-importance plot, and the regression / classification diagnostic figures are graded under §1c (20 points). Drafting them while the modeling section is still open catches issues early.

---

**End of Milestone 03 instructions.**
