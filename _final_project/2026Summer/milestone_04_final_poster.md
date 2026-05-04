# Milestone 04 — Final Research Poster + Peer Evaluation

## About the Final Project

The Final Project is a **group capstone** (groups of four randomly assigned) culminating in a **research poster**. Across the term your group completes four milestone deliverables (M1 → M4); each member also submits a confidential intra-group peer evaluation at the end. The project is worth **35% of your overall course grade**, broken down as:

- **Milestone Deliverables — 40%** of the project grade. Averaged across the four milestones (M1–M4). Graded for clarity, completeness, and timely submission.
- **Peer Evaluation — 20%** of the project grade. Confidential intra-group ratings collected at the end of the course (see Track B below).
- **Instructor / TA Evaluation — 40%** of the project grade. The final research poster, graded against the poster rubric (see Track A below).

**Milestone 04 is the milestone that drives all three components of the project grade.** The poster is the **40% Instructor/TA Evaluation**; each member's peer-evaluation submission is the **20% Peer Evaluation**; and on-time, complete submission of M4's deliverables also contributes to the **40% Milestone Deliverables** average alongside M1–M3.

Optional presentation at the **Fall 2026 Purdue Undergraduate Research Conference** is strongly encouraged but **not required**. Professor Moreira is happy to serve as faculty mentor for groups choosing to present. Award-winning prior posters from this course: <https://davi-moreira.github.io/applied_projects.html>. Additional information about Purdue undergraduate research conferences: <https://www.purdue.edu/undergrad-research/conferences/index.php>.

---

## Purpose

M4 is the capstone deliverable — the public-facing artifact that synthesizes the entire project arc into a single readable poster. It is also the milestone where:

- The instructor/TA evaluates your group's poster against the rubric (this drives the **40%** Instructor/TA Evaluation component of the project grade).
- Each member submits a confidential **peer evaluation** of teammates (this drives the **20%** Peer Evaluation component).

There are therefore **two submission tracks** at M4: (1) the group poster and (2) each member's individual peer evaluation form.

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
10. **References.** Including a link to the supporting code repository or Colab notebook (so the rubric's reproducibility dimension can be checked).

### Required Visualizations on the Poster

A great poster carries 4–6 high-information figures, not 12 mediocre ones. Each figure must have axis labels, units where applicable, a clear legend, and a 1-sentence caption that names the takeaway. The following figures are **required** (omitting any of them costs rubric points under the Visual Design and Results dimensions):

- **Model-comparison bar chart** showing the M2 baseline, the M3 complex champion, and (if it improves on M3) any further-tuned variant — each as a bar with **error bars representing the 95% Student's *t* CI**. This figure visually justifies the CI-overlap-rule decision.
- **Test-set verdict figure.** A point or short bar showing the test-set metric overlaid on the M3 CV CI bar — visualises the INSIDE / ABOVE / BELOW verdict from §7 above.
- **Feature importance / coefficient plot** for the champion model — horizontal bar chart of the top features.
- **For regression:** a **predicted-vs-actual scatter** with the 45° reference line, OR a **residual plot** with `y = 0` reference.
- **For classification:** a **confusion matrix** at the chosen operating threshold AND a **ROC curve with AUC annotation** OR a **Precision–Recall curve with PR-AUC annotation** (lead with PR if the positive class is rare).

If a figure tells the same story as one already on the poster, replace it — every figure should add information.

### Format

| Item | Specification |
|---|---|
| **File type** | Single PDF |
| **Dimensions** | Per the **poster template attached to this Brightspace assignment** (standard Purdue Undergraduate Research Conference format) |
| **Filename convention** | `<NN>.pdf` where `NN` is your assigned group number — e.g., Group 01 submits `01.pdf`; Group 17 submits `17.pdf`. **Do not include the section number.** Following this convention is what allows the instructor to print the posters for free. |
| **Template** | Attached to this Brightspace assignment by the instructor (download before you begin drafting) |
| **Rubric** | Attached to this Brightspace assignment by the instructor (use it for self-check before submission) |
| **Examples** | Award-winning prior posters: <https://davi-moreira.github.io/applied_projects.html> |
| **Submission location** | Brightspace — Module 4, Final Project Milestone 04 — Poster |

## Peer Evaluation — Components

Each group member submits **one peer-evaluation form** through Brightspace by the M4 deadline. The form rates the **other three teammates only** (no self-evaluation).

### What's on the form

For each of your three teammates:

| Dimension | Scale |
|---|---|
| Commitment to the project | 1 (insufficient) – 5 (exemplary) |
| Technical contribution | 1 – 5 |
| Communication and responsiveness | 1 – 5 |
| Dependability and follow-through | 1 – 5 |
| Fairness of work distribution | 1 – 5 |
| **One specific strength** | Free text |
| **One specific area for improvement** | Free text |

Plus one overall comment about how the group functioned as a team.

### Confidentiality

Only the instructor (and TA) see individual evaluations. Aggregated, anonymized feedback may be returned to the team.

### How the 20% is computed

Each member's peer-evaluation score is the **average of the three ratings the member receives** (across the five dimensions × three raters), with light moderation by the instructor when ratings appear strategically inflated or deflated. Two members of the same group can therefore receive different peer-evaluation scores.

**Submission location:** Brightspace — Module 4, Peer Evaluation form.

> **Failure to submit a peer evaluation reduces the submitter's own peer-evaluation score (not the teammates').**

---

## Optional: Presenting at the Fall 2026 Conference

Groups choosing to present at the **Fall 2026 Purdue Undergraduate Research Conference** are encouraged to email Professor Moreira early (ideally before M3) so mentorship and conference-specific guidance can be arranged.

- Conference details: <https://www.purdue.edu/undergrad-research/conferences/index.php>
- Presenting is **optional** and has no impact on the course grade.

---

## Grading — Poster Template and Rubric (Attached Separately)

> **The instructor will attach two documents to this Brightspace assignment:**
>
> 1. A **poster template** (the starting layout you must use, sized for the Purdue Undergraduate Research Conference poster format).
> 2. A **detailed rubric document** specifying the criteria, weights, and scoring levels that will be used to grade your final poster submission.
>
> Download both documents from Brightspace before you begin drafting. Use the template as the basis for your poster file and the rubric to self-check your draft *before* submission. The rubric is the authoritative grading document — the dimensions listed in the table below are indicative of the rubric's structure, but the attached rubric supersedes anything written here in case of any difference.

The poster grade (100-point scale, mapped from the attached rubric) counts as the **40% Instructor/TA Evaluation** component of your Final Project grade.

**Indicative high-level dimensions** (the attached rubric document is authoritative):

| Dimension | Indicative Weight |
|---|---:|
| **Prediction problem framing & significance** | 15% |
| **Methodology** (preprocessing, feature engineering, modeling, CV, tuning) | 25% |
| **Results & interpretation, including the one-shot test-set evaluation and INSIDE / ABOVE / BELOW verdict** | 25% |
| **Required visualizations present and well-executed** (model-comparison bar chart with 95% CI, test-set verdict figure, feature importance, regression diagnostics or classification diagnostics) | 15% |
| **Visual design & clarity** (information density, figure quality, readability at 5 ft) | 10% |
| **Reproducibility** (link to code, runnable notebook, saved `champion_pipeline.joblib`, reproducible figures) | 10% |
| **Total** | **100%** |

The peer-evaluation portion (Track B above) is graded separately and counts as the **20% Peer Evaluation** component of the Final Project grade — computed per member from the average of the three teammate ratings each member receives, with light instructor moderation when ratings appear strategically inflated or deflated.

---

## Tips and Common Pitfalls

- **Posters are read at a distance.** Headings should be legible from 5 feet away; body text from 3 feet.
- **One figure per claim.** A great poster has 4–6 high-information figures, not 12 mediocre ones — but the four required figures (model-comparison-with-CI, test-set-verdict, feature-importance, plus diagnostic) are non-negotiable.
- **Show the CV CI.** Replace point-estimate bar charts with bars that show the 95% confidence interval. The CI carries information; the point estimate alone does not.
- **Open the test set ONCE.** The champion was selected on CV at M3. The test-set number is computed exactly once with the saved `champion_pipeline.joblib` and reported with the INSIDE / ABOVE / BELOW verdict. Multiple test-set evaluations would invalidate the lockbox discipline that has carried through M1 → M2 → M3 — and reviewers can usually tell from the code history.
- **The abstract IS the lead paragraph.** Don't paraphrase your M3 abstract into something fluffier — use it.
- **Reproducibility link is required.** A poster that does not link to runnable code loses the 10% reproducibility dimension. The link must reach a notebook that runs cleanly top-to-bottom and the saved `champion_pipeline.joblib` from M3.
- **Caption every figure.** A figure with axis labels but no caption forces the audience to guess the takeaway. One sentence per figure is the floor.
- **Filename: `<group-number>.pdf` only.** No section number. No "final_v3_REAL_v2.pdf". The instructor's print pipeline depends on this convention.
- **Submit BOTH the poster (group) AND your peer evaluation (individual).** Missing either one costs points.

---

**End of Milestone 04 instructions.**
