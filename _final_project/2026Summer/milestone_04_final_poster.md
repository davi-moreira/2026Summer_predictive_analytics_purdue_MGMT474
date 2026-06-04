# Final Project Milestone 04 — Final Poster

## About the Final Project

The Final Project is a **group capstone** (groups of four randomly assigned) culminating in a **research poster**. Across the term your group completes four milestone deliverables (M1 → M4); each member also submits a confidential intra-group peer evaluation at the end. The project is worth **35% of your overall course grade**, broken down as:

- **Milestone Deliverables — 40%** of the project grade. Averaged across the four milestones (M1–M4). Graded for clarity, completeness, and timely submission.
- **Peer Evaluation — 20%** of the project grade. Confidential intra-group ratings collected through a separate Brightspace assignment at the end of the course.
- **Instructor / TA Evaluation — 40%** of the project grade. The final research poster, graded against the poster rubric.

Milestone 04 contributes to **two** components of the project grade: the poster is the **40% Instructor/TA Evaluation**, and on-time, complete submission of M4's deliverables also contributes to the **40% Milestone Deliverables** average alongside M1–M3.

Optional presentation at the **Fall 2026 Purdue Undergraduate Research Conference** is strongly encouraged but **not required**. Professor Moreira is happy to serve as faculty mentor for groups choosing to present. Award-winning prior posters from this course: <https://davi-moreira.github.io/applied_projects.html>. Additional information about Purdue undergraduate research conferences: <https://www.purdue.edu/undergrad-research/conferences/index.php>.

---

## What to Submit on Brightspace

Submit **two files** per group on Brightspace by the posted deadline. One designated group member uploads on behalf of the whole group.

| # | File | Description |
|---|---|---|
| 1 | **`NN.pdf`** *(e.g., Group 01 submits `01.pdf`; Group 17 submits `17.pdf`)* | Single PDF poster, Purdue Undergraduate Research Conference format. **Do not include the section number in the filename** — this convention is what allows the instructor to print the posters for free. |
| 2 | **`NN_final.ipynb`** *(e.g., for group 03, `03_final.ipynb`)* | The final Jupyter notebook with the runnable code. Must complete a fresh **"Runtime → Run All"** in Colab without errors. |

Detailed format requirements (poster dimensions, attached template, notebook content, exact Brightspace location) are in the **Poster — Components → Format** sections below. The poster rubric is embedded in the **Grading — Poster Rubric** section of this document.

---

## Purpose

M4 is the capstone deliverable — the public-facing research poster that synthesizes the entire project arc into a single readable artifact, plus the final Jupyter notebook that backs every claim on the poster. The instructor/TA evaluates your group's poster against the rubric in the **Grading — Poster Rubric** section of this document, which drives the **40% Instructor/TA Evaluation** component of the project grade; on-time, complete submission of both files also contributes to the **40% Milestone Deliverables** average across M1–M4.

## Poster — Components

A standout poster reads in roughly five minutes. Yours must include:

1. **Project Title.** Concise and informative. *If the project uses a synthetic generated dataset, the title must say so.*
2. **Group members + section.**
3. **Prediction problem.** Framed as a clear question with a question mark — e.g., *"Can six-month customer-churn risk be predicted from transaction history?"*
4. **Motivation and significance.** Why this prediction matters.
5. **Data overview.** Source, size, key variables, response variable type.
6. **Methodology.**
   - Preprocessing pipeline (`ColumnTransformer` + `StandardScaler` + `OneHotEncoder`)
   - Feature engineering moves (with the leakage-safe pattern)
   - Baseline model (M2) and more complex model (M3)
   - Cross-validation protocol (5- or 10-fold; stratified for classification)
   - Hyperparameter tuning (Grid/Random search)
7. **Results — including the one-shot test-set evaluation (REQUIRED).**
   - Report the **CV mean ± 95% Student's *t* CI** for the M3 champion (carried over from M3).
   - **Open the locked test set EXACTLY ONCE** to compute the champion's test-set metric (use the `champion_pipeline.joblib` saved at M3, retrained on the full training fold). Report the test-set metric on the poster.
   - State the **INSIDE / ABOVE / BELOW verdict** of the test-set metric against the M3 CV CI (INSIDE = test number falls within the CV CI; ABOVE = above the upper bound, i.e. *worse* in error metrics; BELOW = below the lower bound, i.e. *better* than CV expected).
   - If the verdict is ABOVE the CV CI's upper bound (the model performs worse on test than CV predicted), include a one-sentence diagnosis and a "deploy / pause / re-train" recommendation. Do **not** open the test set a second time to "recover" the number — that would invalidate the lockbox discipline.
8. **Interpretation and insights.** Feature importance or coefficient summary; what the model says about the business question.
9. **Limitations + next steps.** Honest about what the data can and cannot support.
10. **References.** Citations for any datasets, papers, or external resources used in the analysis. (The submitted **`NN_final.ipynb`** is the canonical reproducibility artifact — no separate code-repository link is required on the poster.)

### Required Visualizations on the Poster

A great poster carries 4–6 high-information figures, not 12 mediocre ones. Each figure must have axis labels, units where applicable, a clear legend, and a 1-sentence caption that names the takeaway. The following figures are **required** (omitting any of them costs rubric points under the Visual Design and Results dimensions):

- **Model-comparison bar chart** showing the M2 baseline, the M3 complex champion, and (if it improves on M3) any further-tuned variant — each as a bar with **error bars representing the 95% Student's *t* CI**. This figure visually justifies the CI-overlap-rule decision.
- **Test-set verdict figure.** A point or short bar showing the test-set metric overlaid on the M3 CV CI bar — visualises the INSIDE / ABOVE / BELOW verdict from §7 above.
- **Feature importance / coefficient plot** for the champion model — horizontal bar chart of the top features.
- **For regression:** a **predicted-vs-actual scatter** with the 45° reference line, OR a **residual plot** with `y = 0` reference.
- **For classification:** a **confusion matrix** at the chosen operating threshold AND a **ROC curve with AUC annotation** OR a **Precision–Recall curve with PR-AUC annotation** (lead with PR if the positive class is rare).

If a figure tells the same story as one already on the poster, replace it — every figure should add information.

### Format — Poster (`NN.pdf`)

| Item | Specification |
|---|---|
| **File type** | Single PDF |
| **Dimensions** | Per the **poster template attached to this Brightspace assignment** (standard Purdue Undergraduate Research Conference format) |
| **Filename convention** | `NN.pdf` where `NN` is your assigned group number — e.g., Group 01 submits `01.pdf`; Group 17 submits `17.pdf`. **Do not include the section number.** Following this convention is what allows the instructor to print the posters for free. |
| **Template** | Attached to this Brightspace assignment by the instructor (download before you begin drafting) |
| **Rubric** | Embedded in this document — see the **Grading — Poster Rubric** section below. Use it for self-check before submission. |
| **Examples** | Award-winning prior posters: <https://davi-moreira.github.io/applied_projects.html> |

### Format — Final Notebook (`NN_final.ipynb`)

| Item | Specification |
|---|---|
| **File type** | Jupyter notebook (`.ipynb`) |
| **Filename convention** | `NN_final.ipynb` (e.g., for group 03, `03_final.ipynb`) |
| **Required content** | The **one-shot test-set evaluation** with the INSIDE / ABOVE / BELOW verdict; all code that produces the figures shown on the poster |
| **Reproducibility** | Must complete a fresh **"Runtime → Run All"** in Colab without errors |

**Submission location for both files:** Brightspace.

## Optional: Presenting at the Summer/Fall Conference

Groups choosing to present at the **Summer/Fall Purdue Undergraduate Research Conference** are encouraged to email Professor Moreira early so mentorship and conference-specific guidance can be arranged.

- Conference details: <https://www.purdue.edu/undergrad-research/conferences/index.php>
- Presenting is **optional** and has no impact on the course grade.

---

## Grading — Poster Rubric

### How Milestone 04 Appears in the Brightspace Gradebook

There are **two separate gradebook entries** for your final-project work at this milestone:

- **Milestone 04 (under "Milestone Deliverables")** — graded on **completion / submission**. You receive credit for this entry by submitting both required files (`NN.pdf` and `NN_final.ipynb`) on time and complete. This entry contributes to the **40% Milestone Deliverables** average alongside M1–M3.
- **Final Submission (under "Instructor / TA Evaluation")** — your poster is graded against the **Predictive Analytics Poster Rubric** below (100-point scale). This grade is reflected in the **Instructor/TA Evaluation** category of the Brightspace gradebook and counts as the **40% Instructor/TA Evaluation** component of your Final Project grade.

> **The instructor will attach the poster template to this Brightspace assignment.** Download it before you begin drafting and use it as the starting layout for your poster. The poster rubric itself is embedded directly in this assignment document (see the table immediately below) — use it to self-check your draft before submission.

### Predictive Analytics Poster Rubric (Poster-Only Assessment)

| # | Criterion | Points | Exemplary (Full Points) | Competent (Partial Points) | Needs Improvement (Minimal Points) |
|---|---|---:|---|---|---|
| 1 | Prediction Goal(s) | 5 | **5 pts:** Clear, specific predictive objective(s); response type (regression/classification) explicitly identified and justified; strong business/research rationale. | **3–4 pts:** Goal stated; response type identified but weakly justified; limited link to context. | **0–2 pts:** Goal unclear/missing; response type unspecified; no rationale. |
| 2 | Dataset Overview | 8 | **7–8 pts:** Reports #obs/#features; types for key features; response variable identified/typed; data provenance/window stated. | **4–6 pts:** Size and response identified; partial feature typing; minimal provenance. | **0–3 pts:** Missing size/response ID; little/no feature typing or context. |
| 3 | Exploratory Data Analysis (EDA) | 12 | **10–12 pts:** Appropriate summaries (response + key predictors); ≥2 relevant plots; interprets skew/outliers/imbalance/seasonality; connects insights to modeling choices and metric selection. | **6–9 pts:** Basic stats and ≥1 plot; some interpretation but limited linkage to modeling. | **0–5 pts:** Incomplete stats; weak/irrelevant visuals; minimal/no interpretation. |
| 4 | Data Preparation (Feature Engineering, Missing Data, etc.) | 20 | **17–20 pts:** ≥1 non-trivial new feature (interactions, lags/rolling stats, domain-driven ratios, text-derived signals) with clear rationale; quantifies missingness by feature; justified strategy (drop/impute/advanced) with leakage-safe implementation; demonstrates impact via ablation/CV deltas or sensitivity to imputation choices. | **10–16 pts:** New feature with partial rationale; identifies missingness and applies a reasonable method; limited evidence of impact or sensitivity analysis. | **0–9 pts:** No new/trivial feature; missingness unreported or mishandled; unjustified/inappropriate method; no evidence of value. |
| 5 | Baseline Model (Choice, CV, Interpretation) | 20 | **17–20 pts:** Appropriate simple baseline (e.g., Linear/Logistic Regression); pipeline is leakage-safe; k=5/10 CV with correct metric (RMSE/MAE/Accuracy/F1/AUC); tracks mean/variance; summarizes implications and limitations; proposes concrete next steps. | **10–16 pts:** Reasonable model; CV present with partial integration/details; metric adequate; interpretation limited or generic. | **0–9 pts:** Inappropriate/unspecified baseline; no/incorrect CV; unclear metric; raw numbers only; no reflection. |
| 6 | More Complex Model (Choice, Tuning, Comparison) | 25 | **22–25 pts:** Model aligned to data/goal (RF/GBM/XGBoost/SVM/regularized GLM/NN); hyperparameter search (grid/random/Bayesian) inside proper CV; leakage controls (scaling/encoding inside folds); comparative results (tables/plots) with variability and selection criterion; rigorous comparison to baseline with practical interpretation (effect sizes, trade-offs), plus error/confusion analysis where relevant. | **13–21 pts:** Plausible model; some tuning with partial documentation; CV present but thin; basic comparison to baseline; limited practical takeaways. | **0–12 pts:** Misaligned/unspecified model; superficial/absent tuning; incorrect CV; opaque selection; no clear comparison or interpretation. |
| 7 | Report Quality & Visual Communication | 10 | **9–10 pts:** Clear, scannable poster; logical flow (Intro→Methods→Results→Conclusions); legible, labeled figures/tables (titles, axes, units, legends); consistent typography/layout; code refs/appendix where appropriate; minimal grammar issues. | **6–8 pts:** Generally readable; minor layout/labeling gaps; some figures under-explained or styling inconsistencies. | **0–5 pts:** Disorganized; illegible/unlabeled visuals; weak linkage between text and graphics; notable grammar/formatting issues. |

**Total: 100 points**

### Penalties

| Issue | Deduction |
|---|---|
| Filename does not follow the `NN.pdf` (poster) / `NN_final.ipynb` (notebook) convention (e.g., for group 03, `03.pdf` and `03_final.ipynb`) | **−10 points** |

---

## Tips and Common Pitfalls

- **Posters are read at a distance.** Headings should be legible from 5 feet away; body text from 3 feet.
- **One figure per claim.** A great poster has 4–6 high-information figures, not 12 mediocre ones — but the four required figures (model-comparison-with-CI, test-set-verdict, feature-importance, plus diagnostic) are non-negotiable.
- **Show the CV CI.** Replace point-estimate bar charts with bars that show the 95% confidence interval. The CI carries information; the point estimate alone does not.
- **Open the test set ONCE.** The champion was selected on CV at M3. The test-set number is computed exactly once with the saved `champion_pipeline.joblib` and reported with the INSIDE / ABOVE / BELOW verdict. Multiple test-set evaluations would invalidate the lockbox discipline that has carried through M1 → M2 → M3 — and reviewers can usually tell from the code history.
- **The abstract IS the lead paragraph.** Don't paraphrase your M3 abstract into something fluffier — use it.
- **The submitted `NN_final.ipynb` must be reproducible.** A submission whose notebook fails on a fresh **"Runtime → Run All"** loses the 10% reproducibility dimension. The notebook must produce all the figures shown on the poster.
- **Caption every figure.** A figure with axis labels but no caption forces the audience to guess the takeaway. One sentence per figure is the floor.
- **Filename: `<group-number>.pdf` only.** No section number. No "final_v3_REAL_v2.pdf". The instructor's print pipeline depends on this convention.
- **Submit BOTH files: the poster `NN.pdf` and the final notebook `NN_final.ipynb`.** Missing either costs points.

---

**End of Milestone 04 instructions.**
