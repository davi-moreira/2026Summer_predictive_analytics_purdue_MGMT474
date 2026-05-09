# Course Case Competition — Bank Churn

## About the Course Case Competition

The Course Case Competition is a **team competition** hosted on Kaggle. Across the term your group (the same Brightspace group as your Final Project team) builds predictive models for the **Case Competition: Bank Churn** and submits to a private Kaggle leaderboard. Check the course syllabus for grade details.

---

## What to Submit on Brightspace

Submit the following on Brightspace.

| # | File | Description |
|---|---|---|
| 1 | **`NN_kaggle_code.ipynb`** *(e.g., for group 03, `03_kaggle_code.ipynb`)* | The fully replicable Colab notebook for your **best-performing model** — data loading, preprocessing, feature engineering, model training, hyperparameter tuning, final fit, prediction generation, and writing the submission `.csv`. Must run top-to-bottom in Google Colab without manual edits given the original Kaggle data files. |

The **final Kaggle submission** itself is uploaded directly to Kaggle (not Brightspace) — see Section "How to Participate" below.

The **peer evaluation** is collected via a separate Brightspace form, alongside the Final Project peer evaluation, in the last Brightspace module of the course.

---

## The Challenge

**Goal.** For each customer in the test dataset, predict the probability that they will churn (`Exited = 1`).

- Predictions must be probabilities in `[0, 1]`.
- Higher probabilities indicate a higher likelihood of churn.

**Evaluation metric: Area Under the ROC Curve (AUC-ROC).**

- Measures how well your model ranks positive (churn) vs. negative (non-churn) cases.
- Perfect classifier: AUC = 1.0; random guessing: AUC = 0.5.
- The **public leaderboard** scores a fraction of the test set during the competition; the **private leaderboard** (used for final grading) is revealed at the close.

**Dataset.** The data is **synthetic** but derived from the original *Bank Customer Churn Prediction Dataset*, generated using a deep-learning model to mimic realistic distributions. The synthetic distribution is similar but not identical to the original. You are encouraged to explore the original dataset as an additional resource.

**Origin (public Kaggle Playground):** <https://www.kaggle.com/competitions/playground-series-s4e1/overview>

---

## Files Provided (on the Kaggle competition page)

1. **`train.csv`** — training dataset; columns include customer demographics, account information, and behaviors. `Exited` is the binary target (1 = churned, 0 = retained).
2. **`test.csv`** — test dataset; same structure as `train.csv` but **without** the target. Predict `Exited` for these records.
3. **`sample_submission.csv`** — template showing the required submission format. Two columns: `id` and `Exited`. Example:

   ```
   id,Exited
   165034,0.9
   165035,0.1
   165036,0.5
   ```

---

## How to Participate — Step by Step

### Step 1 — Join the competition on Kaggle

1. **Create a Kaggle account** if you don't already have one. Use a name and email the instructor will recognize (your Purdue identity is preferred).
2. Open the **invitation link** posted on Brightspace and click **"Accept"** to join the private competition.
3. Read and **accept the competition rules and terms** when prompted.
4. After accepting, the competition will appear in your "My Competitions" list and you'll be able to download `train.csv`, `test.csv`, and `sample_submission.csv` from the **Data** tab.

### Step 2 — Form your team on Kaggle

Form your team **before any teammate makes a submission** — it is much simpler that way. **Your team name on Kaggle must match your Brightspace group exactly, in the format `Group NN`** (e.g., `Group 03`).

> **Why the naming convention matters.** The instructor and TA use the team name to map Kaggle leaderboard standings back to Brightspace groups for grading. **Misnamed teams cannot be linked to the gradebook automatically** and will be subject to a participation-grade deduction until corrected.

#### Option A — Designate one teammate to create the team (recommended)

1. Each teammate joins the competition individually (Step 1).
2. **One designated member** (the "team captain") opens the competition page and clicks the **"Team"** tab.
3. In the **"Team Name"** field, enter **`Group NN`** using the exact format and the leading zero (e.g., `Group 03`). Click **Save**.
4. In the **"Invite teammate"** field, enter each teammate's **Kaggle username** (or the email associated with their Kaggle account) and click **Invite**. Kaggle sends an invitation that the teammate must accept from their own account.
5. Each invited teammate accepts by going to the same competition's **"Team"** tab and clicking **"Accept"** on the pending invitation.
6. Once everyone has accepted, the team appears on the leaderboard with the chosen name.

#### Option B — Merge two existing teams (use only if some teammates have already submitted)

If two or more teammates each independently submitted before forming the team, Kaggle requires a **team merge** rather than a simple invite.

1. The two team captains coordinate. From the **"Team"** tab, one team clicks **"Request Merge"** and enters the other team's name.
2. The other team's captain accepts the merge request from the same tab.
3. After the merge succeeds, **set the merged team's name to `Group NN`**.
4. **Caution — merges share submissions.** When teams merge, the combined team's daily-submission count for that day equals the **sum of submissions already used by both teams**. If both teams submitted twice today, the merged team has used 4 of its 5 submissions for the day.

#### Important rules and limits

- After teaming up, the **four members share 5 submissions per day** as a single team — not 5 each.
- Once you accept a merge or invite, you **cannot leave the team without instructor approval**; a leaver still counts against the team's submission allowance for that day.
- **Form your team well before the final submission deadline.** Kaggle disables team merges at a host-configured cutoff before competition close — verify the exact cutoff on the competition's **"Team"** tab and form teams as early as possible.
- The team name must remain **`Group NN`** at the time of the deadline so the instructor can map standings to Brightspace groups for grading.

### Step 3 — Build your model

- Explore `train.csv` to understand the features.
- Preprocess the data inside a `Pipeline` — encode categorical variables, scale numeric features, and handle missing values without leaking information from the test set.
- Train candidate models (logistic regression, decision trees, random forests, gradient boosting, neural networks).
- Evaluate locally using **AUC-ROC under cross-validation**.
- Select the model with the most defensible **cross-validated** AUC for your final submission, not the one that happens to score highest on the public leaderboard. **Public-leaderboard overfitting is real and will hurt you on the private leaderboard.**

> **Worked starter pipeline (attached):** `churn_exit_prediction_workflow_student.ipynb` is attached to this assignment and walks through the full pipeline end-to-end on this dataset.

### Step 4 — Prepare the submission file

- Generate predictions for **every row** in `test.csv`.
- Save the output as a `.csv` file with **two columns** in this exact order:
  - `id` — from the test set
  - `Exited` — predicted probability of churn, in `[0, 1]`
- Use the `sample_submission.csv` format to avoid submission errors. UTF-8 encoded, no extra columns, no quotes around numeric values.

### Step 5 — Submit on Kaggle

1. From the competition page, click **"Submit Predictions"**.
2. Upload your `.csv` file (drag-and-drop or click "Browse").
3. Add a short description of the submission (model + key hyperparameters) so you can track experiments.
4. Kaggle automatically scores the submission and updates the public leaderboard within seconds.
5. You may submit multiple times — **maximum 5 per day per team**. The leaderboard displays the best public score; final grading uses the **private leaderboard** revealed at the close.

---

## Tips and Common Pitfalls

- **Form the team early — and name it correctly.** The single most common participation-grade hit comes from a misnamed team that the gradebook can't auto-link. Set the name to `Group NN` (with the leading zero) **before** anyone submits.
- **Don't chase the public leaderboard.** The private leaderboard scores the held-out remainder of the test set. Models that overfit the public split routinely drop 10+ ranks on the private leaderboard. **Trust your cross-validated AUC.**
- **Plan your daily submissions.** A team of four shares **5 submissions per day**, not 5 each. Pick the experiments that will actually *teach you something* — don't burn all 5 on small variations of the same model.
- **Build a `Pipeline`, not a script of steps.** Preprocessing must live inside the `Pipeline` so it gets refit fold-by-fold during CV and can be re-applied at submission time. Manual `.fit_transform` on the full data leaks information from the test split.
- **Keep an experiment log.** Record (model name, hyperparameters, CV AUC, public AUC) for each submission. When the deadline approaches, you'll know which model to ship — and the log makes the Brightspace notebook write-up trivial.
- **Verify the submission format before uploading.** Two columns (`id`, `Exited`), no quotes around numbers, UTF-8, every row of `test.csv` covered. Format errors don't count toward your daily quota — but they do waste real time.
- **One Brightspace upload per team.** Decide who uploads `NN_kaggle_code.ipynb` and have the others verify it runs cleanly in Colab from a fresh runtime before the deadline.
- **Cite any AI tool you used.** Per the syllabus AI Policy, include a short note at the end of the notebook acknowledging AI use (model + how it was used).

---

*Good luck — may the best churn predictor win!*

— *Prof. Moreira*

**End of Course Case Competition instructions.**
