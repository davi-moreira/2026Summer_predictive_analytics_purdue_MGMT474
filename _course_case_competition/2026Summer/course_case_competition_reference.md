# Course Case Competition — Reference (Single Source of Truth)

**Course:** QM47400 — Predictive Analytics
**Term:** Summer 2026
**Section:** Y01 (online)
**Instructor:** Professor Davi Moreira
**Competition title:** Summer 2026 QM47400 Case Competition — Bank Churn

> **This file is the canonical reference for everything related to the Course Case Competition.** The syllabus, schedule, planning documents, the Brightspace assignment instructions, and the supporting workflow notebook (`churn_exit_prediction_workflow_student.ipynb`) must be kept consistent with what is written here. Update this file first whenever competition rules change.

---

## 1. Overview

Welcome to the **Summer 2026 QM47400 Case Competition: Bank Churn**. This challenge is adapted from Kaggle's *Playground Series (Season 4, Episode 1)* using AI-generated synthetic data and is integrated into the coursework this term.

Your objective is to build predictive models that determine whether a bank customer will close their account (**churn**) or remain.

- **Reference (public Kaggle Playground):** <https://www.kaggle.com/competitions/playground-series-s4e1/overview>
- **QM47400 (Summer 2026) competition page:** <https://www.kaggle.com/competitions/summer-2026-mgmt-47400-case-competition-bank-churn>
- **Invitation link** (private competition; required to access training/test data and the leaderboard): provided on the course **Brightspace** page.

---

## 2. Goal and Evaluation Metric

For each customer in the **test dataset**, predict the probability that they will churn (`Exited = 1`).

- Predictions must be **probabilities between 0 and 1**.
- Higher probabilities indicate higher likelihood of churn.

**Evaluation metric: Area Under the ROC Curve (AUC-ROC).**

- Measures how well the model ranks positive (churn) vs. negative (non-churn) cases.
- Perfect classifier: **AUC = 1.0**; random guessing: **AUC = 0.5**.

The public leaderboard is computed on a fraction of the test set during the competition; the **private leaderboard** (used for final grading) is revealed at the deadline and is computed on the held-out remainder of the test set.

---

## 3. Team Logistics

| Item | Detail |
|---|---|
| **Team size** | Up to **4 members** |
| **Team formation** | Randomly assigned by the instructor at the start of the course (Brightspace groups). Same membership as the Final Project group. |
| **Team name on Kaggle** | Must match your Brightspace group exactly, in the format **`Group XX – Y01`** (e.g., *Group 03 – Y01*). Use a **leading zero** for single-digit group numbers; the section is **Y01** for Summer 2026. |
| **Team changes** | Not permitted after group assignments are finalized except for documented attendance issues, at instructor discretion. |

> **Why the naming convention matters.** The instructor and TA use the team name to map Kaggle leaderboard standings back to Brightspace groups for grading. **Misnamed teams cannot be linked to the gradebook automatically** and will be subject to a participation-grade deduction until corrected.

---

## 4. Assessment Structure

The Course Case Competition is worth **20% of the overall course grade**. Within the competition, the three components below combine to 100%.

| Component | Weight (of competition) | Weight (of course) | What it covers |
|---|---:|---:|---|
| **Team Participation** | 20% | 4.0% | At least one valid team submission to Kaggle. Ensures every team is actively engaged in the competition. |
| **Leaderboard Performance** | 60% | 12.0% | The team's final standing on the **private** Kaggle leaderboard at the close of the competition, plus an evaluation of the submitted code for replicability. |
| **Peer Evaluation** | 20% | 4.0% | Confidential intra-team rating of teammates' contributions. Adjusts each member's grade to reflect balanced participation and collaborative effectiveness. |
| **Total** | **100%** | **20.0%** | |

The **Peer Evaluation** for the competition is collected together with the Final Project peer evaluation in a single Brightspace form (see Section 11).

---

## 5. Files Provided (on the Kaggle competition page)

1. **`train.csv`** — training dataset.
   - Columns include customer demographics, account information, and behaviors.
   - `Exited` is the binary target variable (**1 = churned, 0 = retained**).
2. **`test.csv`** — test dataset.
   - Same structure as `train.csv` but **without** the target variable.
   - Your job is to predict `Exited` probabilities for these records.
3. **`sample_submission.csv`** — template showing the required submission format.
   - Two columns: `id` and `Exited`.
   - Example:

     ```
     id,Exited
     165034,0.9
     165035,0.1
     165036,0.5
     ```

---

## 6. Dataset Description

- The data is **synthetic** but derived from the original *Bank Customer Churn Prediction Dataset*.
- It was generated using a deep-learning model to mimic realistic distributions.
- The synthetic distribution is **similar but not identical** to the original.
- You are encouraged to explore the original dataset as an additional resource.

---

## 7. How to Participate — Step by Step

### 7.1 Join the competition on Kaggle

1. **Create a Kaggle account** if you don't already have one. Use a name and email the instructor will recognize (your Purdue identity is preferred).
2. Open the **invitation link** posted on Brightspace and click **"Accept"** to join the private competition.
3. Read and **accept the competition rules and terms** when prompted.
4. After accepting, the competition will appear in your "My Competitions" list and you'll be able to download `train.csv`, `test.csv`, and `sample_submission.csv` from the **Data** tab.

### 7.2 Form your team on Kaggle

The simplest path is to form your team **before any teammate makes a submission**.

#### Option A — Designate one member to create the team (recommended)

1. Each teammate joins the competition individually (Section 7.1).
2. **One designated member** (the "team captain") opens the competition page and clicks the **"Team"** tab.
3. In the **"Team Name"** field, enter **`Group XX – Y01`** using the exact format and the leading zero for single-digit group numbers (e.g., `Group 03 – Y01`). Click **Save**.
4. In the **"Invite teammate"** field, enter each teammate's **Kaggle username** (or the email associated with their Kaggle account) and click **Invite**. Kaggle sends an invitation to that user.
5. Each invited teammate accepts by going to the same competition's **"Team"** tab and clicking **"Accept"** on the pending invitation.
6. Once everyone has accepted, the team appears on the leaderboard with the chosen name and shares one team standing.

#### Option B — Merge two existing teams (use only if some teammates have already submitted)

If two or more teammates each independently submitted before forming the team, Kaggle requires a **team merge** rather than a simple invite.

1. The two team captains coordinate. From the **"Team"** tab, one team clicks **"Request Merge"** and enters the other team's name.
2. The **other** team's captain accepts the merge request from the same tab.
3. After the merge succeeds, **set the merged team's name to `Group XX – Y01`**.
4. **Caution — merges share submissions.** When teams merge, the combined team's daily-submission count for that day equals the **sum of submissions already used by both teams**. If both teams submitted twice today, the merged team has used 4 of its 5 submissions for the day.

#### Important rules and limits

- After teaming up, the **four members share 5 submissions per day** as a single team — not 5 each.
- Once you accept a merge or invite, you **cannot leave the team without instructor approval**; a leaver still counts against the team's submission allowance for that day.
- **Form your team well before the final submission deadline.** Kaggle disables team merges at a host-configured cutoff before competition close — verify the exact cutoff on the competition's **"Team"** tab and form teams as early as possible.
- The team name must remain **`Group XX – Y01`** at the time of the deadline so the instructor can map standings to Brightspace groups for grading.

### 7.3 Build your model

- Explore `train.csv` to understand the features.
- Preprocess the data inside a `Pipeline` — encode categorical variables, scale numeric features, and handle missing values without leaking information from the test set.
- Train candidate models (logistic regression, decision trees, random forests, gradient boosting, neural networks).
- Evaluate locally using **AUC-ROC under cross-validation** — the same CV-first protocol the course adopts from `nb08` onward (k-fold + Student's *t* 95% CI).
- Select the model with the most defensible **cross-validated** AUC for your final submission, not the one that happens to score highest on the public leaderboard. Public-leaderboard overfitting is real and will hurt you on the private leaderboard.

The instructor-provided notebook `_course_case_competition/2026Summer/churn_exit_prediction_workflow_student.ipynb` walks the team through the full pipeline end-to-end on this dataset.

### 7.4 Prepare the submission file

- Generate predictions for **every row** in `test.csv`.
- Save the output as a `.csv` file with **two columns** in this exact order:
  - `id` — from the test set
  - `Exited` — predicted probability of churn, in `[0, 1]`
- Use the `sample_submission.csv` format to avoid submission errors. UTF-8 encoded, no extra columns, no quotes around numeric values.

### 7.5 Submit on Kaggle

1. From the competition page, click **"Submit Predictions"**.
2. Upload your `.csv` file (drag-and-drop or click "Browse").
3. Add a short description of the submission (model + key hyperparameters) so you can track experiments.
4. Kaggle automatically scores the submission and updates the public leaderboard within seconds.
5. You may submit multiple times — **maximum 5 per day per team**. The leaderboard displays the best public score; final grading uses the **private leaderboard** revealed at the close of the competition.

---

## 8. Daily Submission Limits and Deadlines

| Item | Detail |
|---|---|
| **Daily submission limit per team** | **5 per day** (Kaggle enforces this) |
| **Competition open** | **Mon May 18, 2026** |
| **Final submission deadline** | **Fri June 12, 2026 at 11:59 PM EST** (Kaggle competition close) |
| **Team formation cutoff** | Verified on the Kaggle competition's **Team** tab (host-configured before close). **Form teams as early as possible.** |
| **Brightspace code submission deadline** | **Fri June 12, 2026 at 11:59 PM EST** — same as Kaggle |

---

## 9. Deliverables

1. **Final Kaggle submission** — submitted via the Kaggle competition portal by the deadline.
2. **Python code on Brightspace** — submitted by the same deadline. The code must be **fully replicable**, allowing the instructor and TA to reproduce the same predictions and AUC. Include all steps:
   - Data loading and inspection
   - Preprocessing (categorical encoding, scaling, missing-value handling)
   - Feature engineering
   - Model training (pipelines + CV-first evaluation)
   - Hyperparameter tuning (if used)
   - Final model fit and prediction generation
   - Writing the submission `.csv` with the correct format

   The submitted notebook should run **top-to-bottom in Google Colab** without manual edits given the original Kaggle data files.
3. **Class share-out (optional, instructor-invited).** Top teams may be invited to share modeling decisions and results during the closing Friday synchronous session (Day 20, **Fri June 12, 2026**, 10:30 am – 12:00 pm EST).

---

## 10. Quick Checklist

Use this as a runtime checklist throughout the competition.

- [ ] **Step 1 — Join the competition.** Click the invitation link on Brightspace; accept the rules and terms.
- [ ] **Step 2 — Form your team.** Open the **Team** tab. Invite teammates (or accept invites). Rename the team to **`Group XX – Y01`** exactly.
- [ ] **Step 3 — Explore the data.**
  - `train.csv` → training dataset (target = `Exited`)
  - `test.csv` → test dataset (no target — predict it)
  - `sample_submission.csv` → submission format
- [ ] **Step 4 — Build your model.** Train candidates (logistic regression, decision trees, random forests, gradient boosting, neural networks); evaluate locally with **cross-validated AUC**.
- [ ] **Step 5 — Create the submission file.** Two columns (`id`, `Exited`); save as `.csv`.
- [ ] **Step 6 — Submit on Kaggle.** Upload the file; the **best score counts** toward the leaderboard.
- [ ] **Step 7 — Final deliverables.** Final Kaggle submission (by deadline) + Python code on Brightspace (by deadline).

**Tip.** Start early, test different models, and keep a short experiment log (model name, hyperparameters, CV AUC, public AUC) so you don't lose track of what you tried. *Good luck — may the best churn predictor win!*

---

## 11. Peer Evaluation — Logistics

The peer evaluation for the Course Case Competition is administered with the Final Project peer evaluation **in a single Brightspace form** (described in [`_final_project/2026Summer/final_project_milestone_reference.md`](../../_final_project/2026Summer/final_project_milestone_reference.md), Section 6).

Mechanics:

- Each member evaluates the other **three teammates only** (no self-evaluation).
- Confidential — only the instructor (and TA) sees individual evaluations; aggregated feedback may be returned to the team.
- Submitted via a Brightspace form (link provided in the relevant Brightspace module). Form structure:
  - Numerical rating (1–5) on each of: commitment, technical contribution, communication, dependability, fairness of work distribution
  - Free-text field for one specific strength and one specific area for improvement per teammate
  - One overall comment about how the team functioned across **both** the competition and the Final Project
- **Failure to submit a peer evaluation reduces the submitter's own peer-evaluation score** (not the teammates').

The 20% peer-evaluation component is calculated **per member** from the average of the three ratings each member receives, with light moderation by the instructor when ratings appear strategically inflated or deflated.

---

## 12. Resources and Cross-References

- **This folder:** `_course_case_competition/2026Summer/`
  - `course_case_competition_reference.md` (this file — canonical reference)
  - `churn_exit_prediction_workflow_student.ipynb` (worked end-to-end pipeline notebook)
  - `churn_exit_prediction_workflow_instructor.ipynb` (instructor copy; gitignored)
  - `reference/Course Case Competition Guidelines.docx` (legacy long-form reference; superseded by this .md but retained for archive)
  - `reference/quick_check_list.docx` (legacy quick-check list; superseded by Section 10 above)
- **Public Kaggle Playground (origin of the data):** <https://www.kaggle.com/competitions/playground-series-s4e1/overview>
- **QM47400 (Summer 2026) competition:** <https://www.kaggle.com/competitions/summer-2026-mgmt-47400-case-competition-bank-churn>
- **Final Project canonical reference:** [`_final_project/2026Summer/final_project_milestone_reference.md`](../../_final_project/2026Summer/final_project_milestone_reference.md)
- **Course syllabus + schedule:** `syllabus.qmd`, `schedule.qmd` (rendered to the course website's `docs/`)

---

## 13. Key Notes (carried forward from the original guidelines)

- Use the provided `sample_submission.csv` format to avoid submission errors.
- Teams must follow the **exact group naming convention** (`Group XX – Y01`).
- The competition is **graded** as part of your final course grade (20%).

---

## 14. Change Log

Maintain this section when the structure changes.

| Date | Change | Authored by |
|---|---|---|
| 2026-05-05 | Initial canonical version: 20/60/20 assessment (Team Participation / Leaderboard / Peer Evaluation), step-by-step participation flow with detailed Kaggle team-formation instructions (Option A invite, Option B merge), quick checklist, deliverables, peer-evaluation alignment with the Final Project. Compressed `Course Case Competition Guidelines.docx` + `quick_check_list.docx` (Fall 2025 long-form references) into the Summer 2026 canonical reference. | Davi Moreira (with Claude assistance) |

---

**End of canonical reference.**
