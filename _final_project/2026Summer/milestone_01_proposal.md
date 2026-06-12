# Milestone 01 — Initial Project Proposal

## About the Final Project

The Final Project is a **group capstone** (groups of four randomly assigned) culminating in a **research poster**. Across the term your group completes four milestone deliverables (M1 → M4); each member also submits a confidential intra-group peer evaluation at the end. The project is worth **35% of your overall course grade**, broken down as:

- **Milestone Deliverables — 40%** of the project grade. Averaged across the four milestones (M1–M4). Graded for clarity, completeness, and timely submission.
- **Peer Evaluation — 20%** of the project grade. Confidential intra-group ratings collected at the end of the course.
- **Instructor / TA Evaluation — 40%** of the project grade. The final research poster, graded against the poster rubric (Milestone 04).

Optional presentation at the **Fall 2026 Purdue Undergraduate Research Conference** is strongly encouraged but **not required**. Professor Moreira is happy to serve as faculty mentor for groups choosing to present. Award-winning prior posters from this course: <https://davi-moreira.github.io/applied_projects.html>. Additional information about Purdue undergraduate research conferences: <https://www.purdue.edu/undergrad-research/conferences/index.php>.

---

## What to Submit on Brightspace

Submit **one file** per group on Brightspace by the posted deadline. One designated group member uploads on behalf of the whole group.

| # | File | Description |
|---|---|---|
| 1 | **`NN_proposal.pdf`** *(e.g., for group 03, `03_proposal.pdf`)* | The 1–2 page Initial Project Proposal covering all five components below — Prediction Goal, Motivation & Significance, Data Overview, Preliminary Methods, Expected Contributions. |

Detailed format requirements (length, style, exact Brightspace location) are in the **Submission Expectations** section near the bottom of this document.

---

## Purpose

This milestone is your group's opportunity to clearly articulate the core of your predictive analytics project. Rather than formulating a traditional research question, you will define a **prediction goal** that outlines the specific outcome you aim to forecast using your data. This proposal sets the foundation for every subsequent milestone — the baseline you'll build at M2, the more complex model and abstract at M3, and the final poster at M4.

## Components

Your proposal must address all five components below.

### 1. Prediction Goal

Provide a clear and concise statement of the outcome you intend to predict.

> **Example:** *"Predict the likelihood of a customer churning within the next six months based on their transaction history and engagement metrics."*

### 2. Project Motivation and Significance

Explain why this prediction goal is important. Describe its relevance to current business challenges or industry trends, and the potential impact of accurate predictions.

### 3. Data Overview

Offer a high-level description of the dataset(s) you plan to use. Include:
- The data source(s) (link or full citation)
- Key variables of interest (target + main predictors)
- A brief discussion of the data's suitability for predictive analytics (size, granularity, recency, ethical/privacy considerations)
- **Whether the dataset is real-world or synthetic/generated.** Real-world datasets are strongly preferred. **Synthetic datasets are discouraged and incur a −10-point penalty**.

#### Free Database Sources for Predictive Analytics

To support your project, you may consider using one of the following free database sources:

- [Kaggle Datasets](https://www.kaggle.com/datasets) — A vast repository covering diverse domains including business, healthcare, and finance.
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php) — A well-established source of datasets suitable for predictive analytics.
- [Google Dataset Search](https://datasetsearch.research.google.com/) — A search engine that helps you locate datasets available across the web.
- [Data.gov](https://www.data.gov/) — The U.S. Government's open data portal with datasets on topics ranging from economics to healthcare.
- [World Bank Open Data](https://data.worldbank.org/) — Comprehensive global development data useful for economic and social predictive models.
- [Quandl](https://www.quandl.com/) — Offers a variety of financial, economic, and alternative datasets, with many available for free.
- [Introduction to Statistical Learning](https://www.statlearning.com/resources-python) — Our course textbook has many valuable datasets you can use.

### 4. Preliminary Methods

Outline the predictive analytics techniques you plan to explore — drawn from the course's nb01–nb09 toolkit:
- Regression vs. classification framing (which one and why)
- Candidate model families (e.g., linear/logistic regression, Ridge/Lasso, tree-based methods you'll meet later in the course)
- Split design (60/20/20 stratified for classification, random for regression — `random_state=474`)
- Evaluation metric(s) tied to the prediction goal

### 5. Expected Contributions / Outcomes

Summarize what your group expects to achieve. What insights or benefits might your predictions offer to business decision-making?

---

## Submission Expectations

| Item | Specification |
|---|---|
| **Length** | 1–2 pages |
| **Format** | PDF; clear headings and bullet points |
| **Style** | Clear, academic, accessible; free of grammatical errors |
| **Submission location** | Brightspace |
| **Filename convention** | `NN_proposal.pdf` (e.g., for group 03 `03_proposal.pdf`) |

---

## Grading Rubric (50 points)

| Criterion | Exemplary (9–10) | Proficient (7–8) | Developing (5–6) | Beginning (0–4) |
|---|---|---|---|---|
| **Prediction Goal** | Clearly defines a specific, measurable prediction goal directly linked to business applications. | States a clear goal, though some details (e.g., measurability) could be more precise. | Mentions a goal but it is vague or only partially aligned with business challenges. | Goal is unclear, missing, or unrelated to a predictive-analytics context. |
| **Motivation & Significance** | Compelling rationale that convincingly explains why the goal matters, with strong ties to current business or industry trends. | Adequate explanation, though rationale may not be deeply developed. | Some motivation but lacks depth or clear connection to practical business impact. | Minimal or no explanation of importance. |
| **Data Overview** | Detailed description: source, key variables, and a clear explanation of why the data suits the prediction goal. | Most key elements present; some details or justifications partially explained. | Basic description with limited detail or justification. | Incomplete, unclear, or missing critical aspects. |
| **Preliminary Methods** | Clearly outlines techniques, demonstrates strong alignment with the prediction goal. | Methods outlined with reasonable clarity; some elements general or not fully connected. | Methods mentioned without sufficient detail or clarity. | Poorly described, irrelevant, or absent. |
| **Expected Contributions & Organization** | Clear statement of anticipated contributions, strong relevance to business decision-making; document very well organized and error-free. | Adequate clarity and organization; minor errors do not detract significantly. | Vague contributions; document organization needs improvement. | Outcomes missing or poorly defined; document disorganized. |

**Total: 50 points** (5 criteria × 10 points each).

### Penalties

| Issue | Deduction |
|---|---|
| Filename does not follow the `NN_proposal.pdf` convention (e.g., for group 03, `03_proposal.pdf`) | **−10 points** |
| Project uses a **synthetic / generated dataset** instead of a real-world dataset (real-world data is strongly preferred — see §3 Data Overview) | **−10 points** |

This rubric grade contributes to the **Milestone Deliverables (40%)** component of the Final Project grade — the average across all four milestones (M1–M4).

---

## Tips and Common Pitfalls

- **Pick the dataset before drafting.** A proposal written without verifying the dataset is the most common cause of M2 rework.
- **Frame the prediction goal with a number, not a vibe.** "Predict customer churn" is too vague; "predict the probability that a customer with ≥6 months of tenure will churn within the next 90 days" is something you can build a model for.
- **Mark the metric explicitly.** Pick MAE/RMSE/R² for regression or PR-AUC/recall on the minority class for imbalanced classification (default: PR-AUC + recall when the positive class is the rare-and-important outcome).
- **Acknowledge ethical/privacy considerations** if the dataset includes any personally identifiable, protected, or sensitive attributes — even one sentence shows you've thought about it.
- **One proposal per group.** Decide the lead author and have the others review and sign off before submission.

---

**End of Milestone 01 instructions.**
