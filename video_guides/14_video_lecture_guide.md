# Video Lecture Guide: Notebook 14 — Model Selection and Comparison - Making the Call Like a Professional

## At a Glance

This guide covers:

1. **Why NB14 exists** — It formalizes the model selection process that students have been doing informally since NB03, replacing ad-hoc "pick the highest number" with a systematic, documented, reproducible protocol

2. **Why it must follow NB13** — NB13 completes the model-building toolkit (Gradient Boosting), giving students their full roster of candidate models. Without all candidates in hand, a formal comparison protocol would be premature

3. **Why it must precede NB15** — NB15 interprets and error-analyzes the champion model. Interpretation is only meaningful once you have committed to a specific model; running PDP/ICE on every candidate would be wasteful and confusing

4. **Why each library/tool is used:**
   - `cross_validate` — evaluates multiple metrics in a single pass over the same CV folds
   - `StratifiedKFold` — guarantees identical folds across all candidate models
   - `make_scorer` — available for custom business metrics, though built-in strings suffice here
   - `time` — captures wall-clock training cost as a selection criterion
   - `classification_report` — provides per-class precision/recall for the final test evaluation

5. **Common student questions** with answers (e.g., why not just pick the highest AUC? why document selection criteria before training?)

6. **Connection to the full course arc** — how the comparison harness from NB14 produces the champion that NB15 interprets, NB16 calibrates, NB17 audits for fairness, and NB18 deploys

7. **Suggested video structure** — both single-video (~12 min) and three-video (~4 min each) formats with speaking prompts, timestamps, and notebook cell references

---

## Purpose of This Document

This guide is your **lecture companion** for recording the video(s) that accompany `14_model_selection_protocol.ipynb`. It explains the *why* behind every design choice in the notebook — why it exists, why it sits between notebooks 13 and 15, why it uses specific evaluation infrastructure, and why the champion selection memo matters. Use it as speaking notes, not as a script: the goal is to internalize the reasoning so you can explain it naturally on camera.

---

## 1. Why This Notebook Exists

By notebook 13, students have built five distinct model families: Logistic Regression (L1 and L2), Decision Trees, Random Forests, and Gradient Boosting. They have compared models informally in individual notebooks — training a model, looking at its CV score, and mentally comparing it to what they remember from last time.

Notebook 14 replaces that informal mental comparison with a **structured, fair, reproducible protocol**. It answers the question every student has after building multiple models: **"I have five trained models — how do I formally choose one?"**

The answer is: define your evaluation criteria before training, use the same cross-validation folds for all models, rank on a primary metric with supporting metrics to break ties, document the selection in a memo, and only then touch the test set — once.

This notebook exists because model selection is where most real-world projects go wrong. Without a protocol, teams fall into "metric shopping" (trying metrics until their favorite model wins), "split shopping" (re-splitting data until results look good), or "test set peeking" (evaluating on test repeatedly during selection). Each of these practices produces overly optimistic performance estimates that collapse in production. The protocol taught here makes those mistakes structurally difficult.

---

## 2. Why It Must Come After Notebook 13

Notebook 13 introduces Gradient Boosting — the final model family in the course's algorithm toolkit. It also builds a preliminary five-model comparison table at the end (Logistic Regression, Decision Tree, Random Forest, GBM default, GBM tuned). That comparison is informal: it uses `cross_val_score` one model at a time, potentially with different metrics or approaches.

| NB13 Concept | How NB14 Uses It |
|---|---|
| **Full candidate roster** | NB14 defines all five candidates upfront and evaluates them through a single comparison harness. Without NB13, the GBM candidates would be missing from the lineup. |
| **Hyperparameter tuning** | NB13 teaches RandomizedSearchCV for GBM. NB14 assumes tuning is complete and evaluates the *best configuration* of each model. Students know the difference between tuning (NB13) and selection (NB14). |
| **Informal comparison table** | NB13 ends with a comparison that motivates the question "how do we make this formal?" NB14 answers it with a reusable harness function and a structured memo. |

**In short:** NB13 gives the final candidate. NB14 gives the protocol for choosing among all candidates.

If you tried to teach the selection protocol before NB13, students would not have enough candidates to make the exercise meaningful. If you skipped NB14 entirely, students would pick models based on gut feeling, memory, or whichever number they happened to see last — exactly the anti-patterns that cause production failures.

---

## 3. Why It Must Come Before Notebook 15

Notebook 15 introduces **model interpretation** (permutation importance, PDP/ICE plots) and **error analysis** (residuals, segment-level errors). Its learning objectives assume students have already:

- Selected a champion model through a documented process
- Committed to that model for interpretation
- Understood that interpretation follows selection, not the other way around

NB15's entire workflow depends on having a single champion to analyze. Running permutation importance on five models would produce five sets of feature rankings, five sets of PDP curves, and five error analyses — overwhelming the student and diluting the interpretive narrative.

The deliberate sequence is:

```
NB13: Build all candidate models (Gradient Boosting completes the roster)
            |
NB14: Compare all candidates fairly, select champion, document decision
            |
NB15: Interpret and error-analyze the champion model
```

Each notebook adds exactly one conceptual layer. By NB15, the selection is a solved problem — students can focus entirely on understanding *how* and *where* the champion model succeeds and fails.

---

## 4. Why These Specific Libraries and Tools

### 4.1 `sklearn.model_selection.cross_validate`

**What it does:** Evaluates a model using cross-validation and returns scores for multiple metrics simultaneously, including both train and test scores and timing information.

**Why we use it (instead of `cross_val_score`):**

1. **Multi-metric evaluation.** `cross_val_score` computes one metric at a time. If you want AUC, accuracy, and F1, you need three separate calls — each of which may use different internal CV folds unless you pass the same `cv` object. `cross_validate` computes all three in a single pass over the same folds, guaranteeing consistency.

2. **Train scores.** With `return_train_score=True`, `cross_validate` reports both training and validation performance for each fold. The gap between the two is the overfitting signal. `cross_val_score` does not offer this.

3. **Timing.** `cross_validate` records `fit_time` and `score_time` per fold. Combined with `time.time()` for wall-clock measurement, this gives a complete picture of computational cost.

**What to emphasize on camera:** `cross_validate` is the professional-grade version of `cross_val_score`. It does more work per call but guarantees that every metric sees the exact same data split. That consistency is what makes the comparison fair.

### 4.2 `sklearn.model_selection.StratifiedKFold`

**What it does:** Splits data into K folds while preserving the class distribution in each fold.

**Why we use it:**

- **Fairness of comparison.** By creating one `StratifiedKFold` object and passing it to every `cross_validate` call, we guarantee that Model A and Model E see exactly the same training and validation samples in every fold. Without this, one model might get an "easy" fold that another model never sees, biasing the comparison.

- **Reproducibility.** The `random_state=RANDOM_SEED` parameter locks the fold assignments. Anyone who runs the notebook with the same seed gets the same folds, the same scores, and the same champion.

- **Class balance.** The breast cancer dataset is roughly 63/37. Without stratification, a random fold might end up with 55/45 or 70/30, causing metric instability — especially for precision and recall.

**What to emphasize on camera:** The single most important line in this notebook is `cv = StratifiedKFold(...)`. That one object is shared across all five models. Change it, and you break the fairness of the comparison.

### 4.3 `time` (Standard Library)

**What it does:** Provides wall-clock timing via `time.time()`.

**Why we use it:**

- **Training cost as a selection criterion.** Two models with identical AUC are not equivalent if one takes 0.1 seconds and the other takes 30 seconds. On large datasets, the difference between Logistic Regression and Gradient Boosting can be minutes versus hours. Timing captures this.

- **Stakeholder communication.** When presenting to a business audience, "this model takes 10 seconds to retrain" versus "this model takes 3 hours to retrain" changes the conversation about deployment frequency, monitoring cadence, and infrastructure cost.

**What to emphasize on camera:** Time is a first-class selection criterion, not an afterthought. A 0.002 AUC improvement that costs 100x more compute is rarely worth it in production.

### 4.4 `sklearn.metrics.classification_report`

**What it does:** Produces a per-class precision, recall, F1, and support table for a set of predictions.

**Why we use it (in the final test evaluation):**

- **Class-level detail.** Overall accuracy can hide poor performance on the minority class. The classification report breaks down precision and recall for both malignant and benign cases, revealing whether the champion model is equally good at detecting both.

- **Final checkpoint.** The classification report is the last piece of evidence before the model is "signed off." If recall on the malignant class (the medically critical one) is suspiciously low, the team should reconsider the champion choice.

**What to emphasize on camera:** The classification report is for the test set only, and it is produced exactly once. If you find yourself running it multiple times with different models, you are no longer doing fair selection — you are optimizing on the test set.

### 4.5 `Pipeline` (from `sklearn.pipeline`)

**What it does:** Chains preprocessing and model into a single estimator.

**Why we use it here:**

- **Leakage prevention inside CV.** When `cross_validate` calls `pipeline.fit(X_fold_train, y_fold_train)`, the `StandardScaler` inside the pipeline fits only on that fold's training portion. Without the pipeline, students would need to manually scale before CV, leaking validation fold statistics into the scaler.

- **Apples-to-apples comparison.** Logistic Regression needs scaling; tree models do not. By wrapping LogReg in a pipeline with `StandardScaler` and leaving tree models unwrapped, each model gets exactly the preprocessing it needs — and the comparison measures model quality, not preprocessing accidents.

**What to emphasize on camera:** The pipeline is not new — students built it in NB02. What is new is using it *inside* a comparison harness to ensure that preprocessing differences do not contaminate the model comparison.

---

## 5. Key Concepts to Explain on Camera

### 5.1 The Fair Comparison Protocol

This is the central idea of the notebook. A fair comparison requires four commitments, all made *before* seeing any results:

1. **Same CV object** for all models — identical fold assignments.
2. **Same primary metric** — one number that determines the winner.
3. **Supporting metrics** — additional evidence to break ties or flag problems.
4. **Test set touched once** — after the champion is selected, not during selection.

Breaking any of these commitments introduces selection bias. The most insidious violation is #4: if you evaluate on the test set, adjust a model, and re-evaluate, the test score is no longer an unbiased estimate of generalization — it has become another validation score.

**Analogy for students:** Think of model selection like a cooking competition. The judges (CV folds) taste every dish (model) using the same spoon (metrics). The winner is announced based on the judging, not on a secret taste test (test set). If the chef gets to tweak the recipe after seeing the judges' scores and then re-submit, the competition is rigged.

### 5.2 Metric Shopping and Why It Is Dangerous

Metric shopping is the practice of trying multiple metrics until you find one where your preferred model wins. It is the model-selection equivalent of p-hacking in statistics.

Example: Model A has ROC-AUC 0.985, Model B has ROC-AUC 0.990. You prefer Model A because it is simpler. So you check accuracy — Model B still wins. You check F1 — Model B still wins. You check precision at recall=0.95 — finally, Model A wins by 0.001. You report that metric.

The protocol prevents this by requiring you to declare the primary metric before training. Supporting metrics are used to *understand* the champion, not to *select* it.

**What to emphasize on camera:** Declare your primary metric in writing before running any model. If you change it afterward, document why — and be suspicious of yourself.

### 5.3 The Champion Selection Memo

The memo is not just a teaching exercise — it is a real-world artifact. In industry, model selection decisions are reviewed by managers, compliance teams, and sometimes regulators. A memo that says "we picked Gradient Boosting because it had the highest AUC" is insufficient. A memo that says "we picked Gradient Boosting because it achieved ROC-AUC 0.993 +/- 0.004, 0.003 above the runner-up, with an overfit gap of 0.005 and CV time under 2 seconds; the primary risk is reduced interpretability compared to Logistic Regression" — that is a defensible decision.

**What to emphasize on camera:** The memo forces you to articulate *why*, not just *what*. If you cannot write a convincing justification, you probably should not select that model.

### 5.4 The Experiment Log as Institutional Memory

The experiment log captures every model tried, its parameters, CV scores, and metadata (date, dataset, seed). Over the course of a project, this log answers questions that would otherwise require re-running experiments: "Did we already try L1 with C=0.01?" or "What was our best AUC three weeks ago?"

In production teams, experiment logs are stored in tools like MLflow, Weights & Biases, or even a shared spreadsheet. The format matters less than the discipline of recording every experiment.

**What to emphasize on camera:** The experiment log is cheap to maintain and expensive to lack. Write it once after each experiment, and you will never waste time re-running something you already tried.

---

## 6. What Students Should Take Away

After watching your video and completing this notebook, students should be able to answer these questions confidently:

1. **"Why can't I just pick the model with the highest test set score?"** — Because evaluating multiple models on the test set and picking the best one is a form of selection bias. The test score of the winner is optimistically biased because you chose it *because* it scored highest. Use CV for selection, test for final validation.

2. **"What is a comparison harness?"** — A function that evaluates all candidate models using the same CV folds, the same metrics, and the same timing methodology. It guarantees a fair comparison by construction.

3. **"Why declare the primary metric before training?"** — To prevent metric shopping. If you choose the metric after seeing results, you can always find a metric where your preferred model wins. Pre-registration eliminates this bias.

4. **"What goes in a champion selection memo?"** — The champion name, primary metric with mean and std, margin over runner-up, supporting metrics, overfit gap, training time, justification, and risks/limitations.

5. **"When do we touch the test set?"** — Exactly once, after the champion is selected. The test evaluation confirms that the CV estimate generalizes. If CV and test disagree dramatically, investigate before deploying.

---

## 7. Common Student Questions (Anticipate These)

**Q: "If all five models have nearly identical AUC, does the protocol still matter?"**
A: Especially then. When margins are tight, the protocol prevents you from making an arbitrary choice and later rationalizing it. With tight margins, the supporting criteria (stability, training time, interpretability) become the real differentiators. The memo forces you to articulate which of those criteria matters most for your deployment context.

**Q: "Why use ROC-AUC as the primary metric instead of accuracy?"**
A: ROC-AUC measures the model's ability to rank positive cases above negative cases across all thresholds. Accuracy depends on the default 0.50 threshold, which may not be optimal for your business problem. In NB16, students will learn to optimize the threshold for business costs — and AUC-ranked models are generally better starting points for threshold tuning.

**Q: "Why not use the test set for cross-validation? We would get more data."**
A: The test set exists to provide an unbiased estimate of real-world performance. If you include it in CV, you lose that unbiased estimate. There is no way to "un-peek" at data. The 20% cost of reserving a test set is the price of trustworthy evaluation.

**Q: "What if the test score is much higher than the CV score?"**
A: This usually means the test set happened to contain "easier" samples. It is less concerning than the reverse (test << CV), but it should still be noted. If the test score falls within two standard deviations of the CV mean, the discrepancy is within expected sampling variability.

**Q: "Should I always pick the most complex model?"**
A: No. Complexity is a cost, not a benefit. A Logistic Regression that scores 0.985 in 0.05 seconds is often preferable to a Gradient Boosting model that scores 0.990 in 2 seconds. The simpler model is faster to train, easier to interpret, cheaper to deploy, and less prone to overfitting. Pick the simplest model that meets your performance requirements.

---

## 8. Connection to the Broader Course Arc

| Week | Notebooks | What NB14's Protocol Enables |
|---|---|---|
| **Week 1** | 01-05 | Students learn preprocessing and evaluation foundations. NB14 formalizes the evaluation rigor introduced informally in NB03. |
| **Week 2** | 06-10 | Classification metrics and models accumulate. NB14 provides the harness that puts all Week 2 models on equal footing. |
| **Week 3** | 11-15 | Ensemble models complete the candidate roster. NB14 selects the champion; NB15 interprets it. Together they close Week 3. |
| **Week 4** | 16-20 | The champion from NB14 flows into threshold tuning (NB16), fairness audit (NB17), deployment (NB18), and the final project (NB19-20). |

**The protocol is the bridge between model building and model deployment.** Everything before NB14 builds candidates. Everything after NB14 refines, audits, and deploys the champion. Without a rigorous selection step, the downstream work lacks a defensible foundation.

---

## 9. Suggested Video Structure

Below are two recording options. Each segment includes the **speaking prompt** (what to say), **timestamps**, and the **notebook cells** to show on screen.

All cell references use the format `Cell N` where N is the zero-indexed cell number in `14_model_selection_protocol.ipynb`.

---

### Option A: Single Video (~12 minutes)

#### Segment 1 — Opening & The Selection Problem `[0:00–2:00]`

> **Show:** Cell 0 (header) and Cell 1 (learning objectives) on screen.

**Say:**

"Welcome back. Over the past several notebooks, you have built five different model families: Logistic Regression with L1 and L2 regularization, Decision Trees, Random Forests, and Gradient Boosting. You have seen CV scores for each one. But here is the problem: those scores were computed in different notebooks, possibly with different CV setups, and you have been comparing them from memory. That is not a systematic comparison — that is guesswork. Today we fix that. We are going to build a formal model selection protocol: same CV folds for every model, same metrics, a documented champion selection memo, and a single test-set evaluation at the very end. By the time we are done, you will have a reproducible, defensible answer to the question 'which model should we deploy?'"

> **Action:** Scroll through the 5 learning objectives in Cell 1, briefly reading each one aloud.

> **Show:** Cell 4 (The Model Selection Problem). Point to the two columns: common mistakes (wrong) and correct practices (right).

**Say:**

"Let me walk you through the most common mistakes first. Comparing models on different data splits — wrong. Using different metrics for different models — wrong. Choosing based on test set performance — wrong. Metric shopping, which means trying different metrics until your favorite model wins — wrong. The right approach is on the other side: same cross-validation folds, single primary metric, select on CV, validate on test once, and document everything before you start."

---

#### Segment 2 — Setup & Data `[2:00–3:00]`

> **Show:** Cell 2 (setup explanation in Cell 3), then **run** Cell 2 (imports).

**Say:**

"Standard setup with one important new import: `cross_validate` instead of `cross_val_score`. The difference is critical. `cross_val_score` computes one metric at a time. `cross_validate` computes multiple metrics in a single pass and returns training scores and timing information. We also import `time` to measure wall-clock cost, because training time is a real selection criterion."

> **Run** Cell 5 (load data + split).

**Say:**

"We are using the breast cancer dataset — 455 training samples, 114 test samples. The test set is explicitly labeled 'LOCKED until final selection.' We will not touch it until the very end of this notebook, after we have selected our champion."

---

#### Segment 3 — Comparison Harness `[3:00–5:30]`

> **Show:** Cell 7 (harness explanation), then **run** Cell 8 (define `compare_models_comprehensive`).

**Say:**

"The comparison harness is a single function that takes a dictionary of models, a CV splitter, and a list of metrics, then evaluates every model under identical conditions. For each model, it calls `cross_validate` with the same `cv` object, records train and test scores for all metrics, computes the overfitting gap, and tracks timing. The result is a sorted DataFrame with the best model on the primary metric at the top. Writing this as a function guarantees that every model sees the exact same folds and the exact same evaluation logic. No subtle bugs, no inconsistencies."

> **Show:** Cell 10 (candidate models explanation), then **run** Cell 11 (define candidates).

**Say:**

"We define five candidates before seeing any results: Logistic Regression with L2, Logistic with L1 for feature sparsity, a Decision Tree, a Random Forest, and Gradient Boosting. Notice that the logistic models are wrapped in Pipelines with StandardScaler — scaling happens inside each CV fold to prevent leakage. The tree models do not need scaling. Defining all candidates upfront prevents model shopping, just like we prevent metric shopping by declaring the primary metric upfront."

> **Run** Cell 14 (run comparison). Point to the results table.

**Say:**

"And here are the results. Five models, all evaluated on 5-fold stratified CV with ROC-AUC as primary metric. Look at the table: the models are sorted by CV ROC-AUC. The overfit gap column tells you how much each model's training score exceeds its CV score — ensemble models typically show small gaps. The total CV time column shows computational cost. Now we have a single, fair comparison to make our decision from."

---

#### Segment 4 — Multi-Metric Visualization & Champion Memo `[5:30–8:30]`

> **Run** Cell 17 (4-panel visualization).

**Say:**

"This 2-by-2 panel shows four perspectives on the same five models. Top left: ROC-AUC, our primary metric. Top right: accuracy. Bottom left: F1. Bottom right: training time. Look for consistency across metrics. If a model ranks first in AUC but third in F1, that is worth investigating — it might have a threshold calibration issue. The error bars show CV variability. If two models have overlapping error bars, they are not statistically distinguishable on that metric."

> **Show:** Cell 19 (memo explanation), then **run** Cell 20 (champion selection memo).

**Say:**

"Now the champion selection memo. This is a structured document that names the winner, quantifies the margin over the runner-up, and provides four justification points: primary metric advantage, overfitting gap, training time, and fold stability. It also flags risks — if the champion is an ensemble model, we note reduced interpretability. In industry, this memo becomes part of the audit trail. A regulator or compliance officer can trace exactly why this model was chosen over the alternatives. If you cannot write a convincing memo, you probably should not select that model."

> **Mention:** "Pause the video here and complete Exercise 2 — write your own champion selection memo with five evidence bullets and one risk." Point to Cell 22 (PAUSE-AND-DO 2).

---

#### Segment 5 — Final Test Evaluation & Experiment Log `[8:30–10:30]`

> **Show:** Cell 24 (test evaluation explanation), then **run** Cell 25 (final test evaluation).

**Say:**

"Now and only now do we touch the test set. We train the champion on the full training data and evaluate on the 114 held-out test samples. The critical check is the CV-vs-test comparison: if the test ROC-AUC falls within two standard deviations of the CV mean, the model generalizes as expected. The classification report breaks down precision and recall by class. For breast cancer, pay special attention to the malignant class recall — missing a malignant case is medically worse than a false alarm."

> **Run** Cell 28 (experiment log).

**Say:**

"Finally, the experiment log. This table captures every model we tried, its CV scores, the test score for the champion only — because we never evaluated non-champion models on test — and metadata like date, dataset, seed, and fold count. In a real project, you would append this to a persistent CSV or database after every modeling session. Three weeks from now, when someone asks 'did we try L1 with C=0.01?' you can look it up instead of re-running the experiment."

---

#### Segment 6 — Wrap-Up & Closing `[10:30–12:00]`

> **Show:** Cell 30 (Wrap-Up).

**Say:**

"Let me leave you with the three critical rules of model selection. First: same CV folds for all models, always. If you use different folds for different models, the comparison is unfair. Second: select on CV, validate on test — once. The test set is your last line of defense against overfitting to the selection process. Third: document your selection criteria before training. If you decide the primary metric after seeing results, you are metric shopping.

The protocol we built today — harness function, primary metric, champion memo, single test evaluation, experiment log — is what separates a professional model selection process from guesswork. In the next notebook, we will take our champion and ask two critical questions: what is it learning, and where is it failing? That is model interpretation and error analysis. See you there."

> **Show:** Cell 31 (Submission Instructions) briefly, then Cell 32 (Thank You).

---

### Option B: Three Shorter Videos (~4 min each)

---

#### Video 1: "The Fair Comparison Protocol" (~4 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–0:45` | Show header + objectives | Cell 0, Cell 1 | "Welcome back. You have built five model families across the last several notebooks. Today we formalize how to choose between them. Here are our five learning objectives." (Read them briefly.) |
| `0:45–1:30` | Show selection problem | Cell 4 | "Here are the most common model selection mistakes: comparing on different splits, using different metrics, peeking at the test set, and metric shopping. The right approach: same CV folds, one primary metric, document criteria before training, test set touched once." |
| `1:30–2:15` | Run setup + data | Cell 2, Cell 5 | "Standard setup with `cross_validate` instead of `cross_val_score` — it evaluates multiple metrics in one pass. Breast cancer dataset, 455 training samples, 114 test samples locked away until final selection." |
| `2:15–3:30` | Run comparison harness | Cell 8, Cell 11, Cell 14 | "The harness function evaluates every model with the same CV object, same metrics, same timing. Five candidates: two logistic regressions, a decision tree, random forest, and gradient boosting. All defined before seeing any results. The output is a sorted table with the best model on top." |
| `3:30–4:00` | Exercise 1 prompt | Cell 13 | "Pause the video now and complete Exercise 1. Implement the comparison harness for three models and review the results. Come back when you are done." |

---

#### Video 2: "Champion Selection and Multi-Metric Evaluation" (~4 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–0:30` | Recap | — | "Welcome back. We have our five-model comparison table. Now we need to choose a champion and document the decision." |
| `0:30–1:30` | Run visualization | Cell 17 | "This four-panel chart shows ROC-AUC, accuracy, F1, and training time for all five models. Look for consistency: a model that ranks first in AUC but third in F1 may have a calibration issue. Error bars show CV variability — overlapping bars mean the models are not statistically distinguishable." |
| `1:30–3:00` | Run champion memo | Cell 20 | "The champion selection memo names the winner, quantifies the advantage over the runner-up, and provides four justification points plus risks. This is the document a manager or regulator reviews. If the champion is an ensemble, we flag interpretability risk. If training time is high, we flag iteration cost. This is not just an exercise — it is an industry best practice." |
| `3:00–4:00` | Exercise 2 prompt | Cell 22, Cell 23 | "Pause and complete Exercise 2. Write your own champion selection memo with five evidence bullets and one key risk. Be specific — use the actual numbers from the comparison table. Come back when you are done." |

---

#### Video 3: "Test Evaluation, Experiment Log & Wrap-Up" (~4 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–1:00` | Run test evaluation | Cell 24, Cell 25 | "Now and only now do we touch the test set. The champion is trained on the full training data and evaluated once on 114 held-out samples. The critical check: does the test ROC-AUC fall within two standard deviations of the CV mean? If yes, the model generalizes as expected. The classification report gives per-class detail — malignant recall is the clinically important number." |
| `1:00–2:00` | Run experiment log | Cell 27, Cell 28 | "The experiment log captures every model, its scores, and metadata. Notice that only the champion has a test score — the other models were never evaluated on test. This is the institutional memory of the project. In production, you append to this log after every session so you never re-run an experiment you already tried." |
| `2:00–3:00` | Show wrap-up | Cell 30 | "Three rules to remember. One: same CV folds for all models, always. Two: select on CV, validate on test once. Three: document your criteria before training. These three rules are what separate professional model selection from guesswork." |
| `3:00–4:00` | Closing | Cell 31, Cell 32 | "This protocol — harness, primary metric, memo, single test evaluation, experiment log — is the backbone of the model selection process. Next notebook, we interpret and error-analyze the champion: what is it learning, and where does it fail? Submit your notebook to Brightspace and complete the quiz. See you there." |

---

*This guide was created to support video lecture recording for MGMT 47400 — Predictive Analytics, Purdue University.*
