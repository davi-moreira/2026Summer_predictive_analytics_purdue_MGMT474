# Video Lecture Guide: Notebook 07 — Classification Metrics - Confusion Matrix, ROC/PR, Calibration, and Business Costs

## At a Glance

This guide covers:

1. **Why NB07 exists** — It builds the complete classification evaluation toolkit that students need to answer the question "Is my classifier actually good?" — moving beyond accuracy to precision, recall, F1, ROC curves, PR curves, and cost-based threshold selection

2. **Why it must follow NB06** — NB06 introduces predicted probabilities, the 0.5 threshold, and the confusion matrix as a first look. NB07 formalizes these into a complete metrics framework. Without NB06, students would not understand what `predict_proba()` returns or why the four cells of the confusion matrix matter

3. **Why it must precede NB08** — NB08 teaches cross-validation for model comparison. It assumes students already know which metric to use as the scoring parameter (ROC-AUC? F1? PR-AUC?) and why. NB07 provides that metric vocabulary so NB08 can focus purely on the comparison methodology

4. **Why each library/tool is used:**
   - `precision_score`, `recall_score`, `f1_score` — the three core classification metrics derived from the confusion matrix
   - `roc_curve`, `roc_auc_score` — threshold-independent model evaluation via TPR vs. FPR
   - `precision_recall_curve`, `average_precision_score` — the imbalance-aware alternative to ROC
   - `ConfusionMatrixDisplay`, `RocCurveDisplay`, `PrecisionRecallDisplay` — publication-quality visualizations
   - `make_classification` — synthetic data to demonstrate the accuracy paradox under extreme imbalance

5. **Common student questions** with answers (e.g., when to use ROC vs. PR curves?, how to choose a threshold in practice?)

6. **Connection to the full course arc** — how the metrics dashboard from NB07 is reused in NB08 (CV comparison), NB09 (tuning), NB10 (midterm), and the final project

7. **Suggested video structure** — both single-video (~12 min) and three-video (~4-5 min each) formats with speaking prompts, timestamps, and notebook cell references

---

## Purpose of This Document

This guide is your **lecture companion** for recording the video(s) that accompany `07_classification_metrics_thresholding.ipynb`. It explains the *why* behind every metric, visualization, and exercise in the notebook — why precision and recall are introduced before ROC, why cost-based thresholding uses dollar values, and why the accuracy paradox is demonstrated with synthetic data. Use it as speaking notes, not as a script: the goal is to internalize the reasoning so you can explain it naturally on camera.

---

## 1. Why This Notebook Exists

Notebook 06 ends with students knowing that logistic regression produces probabilities, that the 0.5 threshold is adjustable, and that the confusion matrix reveals error types. But NB06 introduces these ideas *informally* — it shows a confusion matrix heatmap and a four-threshold sweep, but it does not formalize the metrics, does not plot ROC or PR curves, and does not connect thresholds to business costs.

Notebook 07 delivers the **complete classification evaluation toolkit**. It answers the question every student has after NB06: **"I have a confusion matrix — now what do I do with it?"** The answer is: compute precision, recall, F1, and specificity from the four cells; plot ROC and PR curves to see performance across all thresholds; compute AUC to summarize threshold-independent quality; and use a cost function to pick the threshold that minimizes business losses.

This notebook is the most metrics-dense in the entire course. By the end, students have a reusable `create_metrics_dashboard()` function that wraps every classification metric into a single call — a function they will use in NB08 for cross-validation, NB09 for tuning reports, and their final project.

---

## 2. Why It Must Come After Notebook 06

Notebook 06 establishes four foundations that notebook 07 *consumes*:

| NB06 Concept | How NB07 Uses It |
|---|---|
| **Predicted probabilities (`predict_proba()`)** | NB07 uses probabilities to plot ROC and PR curves, which require sweeping across all possible thresholds. Without understanding what probabilities are, students could not interpret these curves. |
| **The 0.5 threshold and its arbitrariness** | NB06 demonstrates that changing the threshold changes predictions. NB07 formalizes this into a cost-based optimization: given dollar costs for FP and FN, find the threshold that minimizes total expected loss. |
| **The confusion matrix (first exposure)** | NB06 shows a confusion matrix heatmap and labels the four cells (TP, TN, FP, FN). NB07 uses these four numbers as *inputs* to compute precision, recall, F1, and specificity. The formulas are simple arithmetic on TN/FP/FN/TP. |
| **The accuracy trap (63% baseline)** | NB06 shows that a DummyClassifier scores 63% by always predicting benign. NB07 extends this lesson with a synthetic 95/5 imbalanced dataset where a naive baseline scores 95% — making the accuracy paradox even more dramatic. |

**In short:** NB06 gives the intuition (probabilities, thresholds, confusion matrix); NB07 gives the formalism (metrics, curves, cost functions).

If you tried to teach ROC curves *before* NB06, students would not understand what "sweeping the threshold" means or what probabilities are being thresholded. If you skipped NB07 entirely, students in NB08 would not know which `scoring` parameter to pass to `cross_val_score`.

---

## 3. Why It Must Come Before Notebook 08

Notebook 08 introduces **cross-validation for model comparison**. Its core API is `cross_val_score(model, X, y, cv=..., scoring='roc_auc')`. The `scoring` parameter requires students to choose a metric — and to understand *why* that metric is appropriate for their problem.

NB08's learning objectives assume students already know:

- The difference between ROC-AUC and PR-AUC and when to prefer each
- Why accuracy is insufficient for imbalanced problems
- How to interpret a metrics dashboard with precision, recall, F1, specificity, and AUC
- That model comparison should use a *single primary metric* chosen based on business costs

NB07 builds each of these capabilities. Without it, students in NB08 would blindly choose `scoring='accuracy'` for every comparison — which, as NB07 demonstrates, can select the wrong model.

The deliberate sequence is:

```
NB06: What are probabilities and thresholds? (Intuition)
            |
NB07: How do we measure classification quality? (Metrics + curves + cost optimization)
            |
NB08: How do we compare models fairly? (CV with the right metric)
```

Each notebook adds exactly one conceptual layer.

---

## 4. Why These Specific Libraries and Tools

### 4.1 `sklearn.metrics.precision_score`, `recall_score`, `f1_score`

**What they do:** Compute precision (TP / (TP+FP)), recall (TP / (TP+FN)), and F1 (harmonic mean of precision and recall) from true and predicted labels.

**Why we use them:**

1. **They decompose accuracy into actionable components.** Accuracy treats all errors equally. Precision tells you "when I predicted positive, how often was I right?" Recall tells you "of all actual positives, how many did I find?" These two questions have very different business implications.

2. **They expose the precision-recall tradeoff.** You cannot maximize both simultaneously (unless you have a perfect model). The tradeoff forces students to think about which error type matters more in their specific problem.

3. **F1 provides a single-number summary when you need balance.** The harmonic mean penalizes extreme imbalance between precision and recall, making it more conservative than the arithmetic mean.

**What to emphasize on camera:** Precision answers the question of the *user* (when the model says positive, can I trust it?). Recall answers the question of the *subject* (if I truly am positive, will the model find me?). In medical screening, recall matters most. In spam filtering, precision matters most. The right choice depends on who bears the cost of each error.

### 4.2 `sklearn.metrics.roc_curve` and `roc_auc_score`

**What they do:** `roc_curve` computes the True Positive Rate and False Positive Rate at every possible threshold. `roc_auc_score` computes the area under that curve, a threshold-independent summary of model quality.

**Why we use them:**

- **ROC-AUC is the most widely used classification metric.** When students read papers, Kaggle leaderboards, or industry reports, they will encounter ROC-AUC constantly. They need to know what it means and how to interpret it.
- **It evaluates the model's ranking ability.** An AUC of 0.99 means that if you randomly pick one positive and one negative sample, the model assigns a higher probability to the positive sample 99% of the time. This interpretation is intuitive and powerful.
- **It is threshold-independent.** During model *selection* (choosing between models), you want a metric that captures overall quality, not quality at one specific threshold. ROC-AUC does this.

**What to emphasize on camera:** The diagonal line on the ROC plot represents a random coin flip (AUC = 0.5). A perfect model hugs the top-left corner (AUC = 1.0). The further the curve bows toward the top-left, the better the model separates the two classes.

### 4.3 `sklearn.metrics.precision_recall_curve` and `average_precision_score`

**What they do:** `precision_recall_curve` computes precision and recall at every threshold. `average_precision_score` computes the area under the PR curve (also called Average Precision).

**Why we use them:**

- **PR curves are more informative than ROC curves when classes are imbalanced.** ROC-AUC can look optimistic for imbalanced data because the FPR denominator (TN + FP) is dominated by the large negative class. PR-AUC focuses exclusively on the positive class, exposing weaknesses that ROC hides.
- **The baseline on a PR curve is the class prevalence**, not the diagonal. This makes it immediately obvious whether the model is better than random for the minority class.
- **Industry applications with rare events (fraud, rare disease, equipment failure) rely on PR curves** because the positive class is what matters, and its rarity makes ROC-AUC misleadingly high.

**What to emphasize on camera:** Use ROC-AUC when classes are roughly balanced. Use PR-AUC when classes are imbalanced and you care most about the minority class. In this course, we teach both because students will encounter both in practice.

### 4.4 `ConfusionMatrixDisplay`, `RocCurveDisplay`, `PrecisionRecallDisplay`

**What they do:** Scikit-learn display helpers that produce publication-quality plots with a single function call.

**Why we use them:**

- **Consistency.** Every student's plots look the same, making it easier to compare and grade.
- **Simplicity.** One line of code instead of 10 lines of matplotlib boilerplate.
- **Focus on interpretation, not plotting.** Students should spend their time understanding what the curves mean, not debugging axis labels and color maps.

**What to emphasize on camera:** These display functions are convenience wrappers. Behind the scenes, they call `matplotlib`. In your project, you can use either the display helpers or raw matplotlib — the important thing is that the plots are clear, labeled, and interpretable.

### 4.5 `sklearn.datasets.make_classification`

**What it does:** Generates a synthetic classification dataset with user-specified class proportions, number of informative features, and other properties.

**Why we use it:**

- **Demonstrates the accuracy paradox under extreme imbalance.** The breast cancer dataset is only mildly imbalanced (37/63). To drive the lesson home, we create a 95/5 dataset where a naive baseline scores 95% accuracy while missing every positive case. This extreme example makes the point unforgettable.
- **Controlled experiment.** By specifying exact class weights, we can isolate the effect of imbalance without confounding factors from real data.

**What to emphasize on camera:** This synthetic dataset is a teaching device. In real applications, you will encounter datasets that are even more extreme — credit card fraud (0.1% positive), rare disease screening (0.01% positive), network intrusion detection (< 1% positive). The lesson from this cell applies directly to those scenarios.

---

## 5. Key Concepts to Explain on Camera

### 5.1 Precision vs. Recall — The Fundamental Tradeoff

Precision and recall cannot both be maximized simultaneously (except with a perfect model). When you lower the threshold to catch more positives (higher recall), you inevitably include more false positives (lower precision). When you raise the threshold to reduce false alarms (higher precision), you inevitably miss some true positives (lower recall).

**Analogy for students:** Imagine casting a fishing net. A wide net (low threshold) catches more fish (high recall) but also catches more debris (low precision). A narrow net (high threshold) catches only fish (high precision) but lets some swim away (low recall). The right net size depends on whether you care more about catching every fish or about keeping the catch clean.

### 5.2 ROC Curve — The Model's Operating Menu

The ROC curve shows every possible tradeoff between True Positive Rate and False Positive Rate as you sweep the threshold from 1.0 to 0.0. Each point on the curve represents one threshold setting. The curve as a whole represents the model's *capability* — the set of all operating points you could choose.

**Why AUC matters:** AUC summarizes this entire menu into one number. A model with AUC = 0.99 gives you many excellent operating points to choose from. A model with AUC = 0.70 gives you only mediocre options. AUC tells you about the model's quality; the threshold you choose tells you about your business priorities.

### 5.3 Cost-Based Threshold Selection — The Business Decision

The most important concept in this notebook is that threshold selection is a *business decision*, not a statistical one. Given specific dollar costs for false positives and false negatives, you can compute the expected total cost at each threshold and pick the one that minimizes it. The notebook demonstrates this with $50,000 per false negative (missed cancer) and $1,000 per false positive (unnecessary biopsy), resulting in an optimal threshold well below 0.5.

**Why this matters for the course:** The midterm (NB10) asks students to design threshold strategies for business cases. Students who understand cost-based threshold selection will write stronger responses than those who default to 0.5.

### 5.4 The Accuracy Paradox — Why 95% Accuracy Can Be Useless

The synthetic 95/5 dataset demonstrates that a model which always predicts the majority class achieves 95% accuracy. This is the *accuracy paradox* — a metric that appears excellent while the model has zero ability to detect the class you care about. This example motivates the shift from accuracy to precision, recall, F1, and AUC.

---

## 6. What Students Should Take Away

After watching your video and completing this notebook, students should be able to answer these questions confidently:

1. **"What is precision and when do I care about it?"** — Precision = TP/(TP+FP). It answers: "When the model says positive, how often is it right?" You care about precision when false positives are costly (e.g., spam filter sending real emails to junk, or flagging a legitimate transaction as fraud).

2. **"What is recall and when do I care about it?"** — Recall = TP/(TP+FN). It answers: "Of all actual positives, how many did the model find?" You care about recall when false negatives are costly (e.g., missing a disease, failing to detect a security breach).

3. **"When should I use PR-AUC instead of ROC-AUC?"** — When classes are imbalanced and you care primarily about the positive (minority) class. ROC-AUC can look deceptively high on imbalanced data; PR-AUC provides a more honest assessment.

4. **"How do I choose a threshold in practice?"** — Assign dollar costs to false positives and false negatives, compute expected cost at each threshold using validation data, and select the threshold that minimizes total cost. The optimal threshold depends on the cost ratio, not on mathematical convention.

5. **"Why is 95% accuracy meaningless on a 95/5 imbalanced dataset?"** — Because a model that always predicts the majority class achieves 95% accuracy while having zero recall for the minority class. Accuracy inflates with imbalance and hides total failure on the class you care about.

---

## 7. Common Student Questions (Anticipate These)

**Q: "If ROC-AUC is so popular, why do we also need PR-AUC?"**
A: ROC-AUC measures how well the model separates positives from negatives across all thresholds. But when negatives vastly outnumber positives (e.g., 99:1), even a small FPR translates to a large number of false positives in absolute terms, which ROC-AUC does not penalize. PR-AUC, which ignores true negatives entirely, exposes this problem. Rule of thumb: use ROC-AUC when classes are roughly balanced; use PR-AUC when the positive class is rare and important.

**Q: "Can I just use F1 for everything?"**
A: F1 is the harmonic mean of precision and recall, which equally weights both. If your business problem has asymmetric error costs (e.g., missing cancer is 50x worse than a false alarm), F1 does not capture that asymmetry. In such cases, use cost-based thresholding or a weighted F-beta score (F2 emphasizes recall, F0.5 emphasizes precision).

**Q: "How do I know which errors are more costly?"**
A: Ask the business stakeholder. In the medical example, the cost of a missed cancer (delayed treatment, potential death) vs. unnecessary biopsy (patient anxiety, $1,000 procedure) is clear. In other domains — churn prediction, fraud detection, quality control — the cost structure may require research or estimation. The key insight is that *someone* has to make this judgment; the model cannot make it for you.

**Q: "Why does the optimal threshold end up below 0.5?"**
A: Because the cost of a false negative ($50,000) is 50 times higher than a false positive ($1,000). To minimize total cost, the model should err on the side of caution — flag more cases as positive (lower threshold), accepting more false positives to avoid the much more expensive false negatives. If the cost ratio were reversed, the optimal threshold would be above 0.5.

---

## 8. Connection to the Broader Course Arc

| Week | Notebooks | What NB07's Metrics Enable |
|---|---|---|
| **Week 2** | 08 | NB08 uses ROC-AUC and accuracy as the `scoring` parameter in `cross_val_score` and `cross_validate`. Students choose these metrics *because NB07 taught them what they mean* |
| **Week 2** | 09 | NB09 uses ROC-AUC as the scoring metric for `GridSearchCV` and `RandomizedSearchCV`. The metrics dashboard function from NB07 is adapted into the baseline report table |
| **Week 2** | 10 | NB10 (midterm) asks students to select metrics and design threshold strategies for business cases. Every question references precision, recall, cost-based thresholds, or the accuracy paradox — all from NB07 |
| **Week 3** | 11-15 | Tree-based models are evaluated using the same ROC/PR curves and metrics dashboard. NB07's `create_metrics_dashboard()` function transfers directly |
| **Week 4** | 16-20 | Final project requires a comprehensive metrics report. Students reuse the dashboard pattern from NB07 to present their results |

**NB07 is the evaluation backbone of the course.** Every notebook from NB08 onward assumes students can compute, interpret, and justify classification metrics. NB07 is where that capability is built.

---

## 9. Suggested Video Structure

Below are two recording options. Each segment includes the **speaking prompt** (what to say), **timestamps**, and the **notebook cells** to show on screen.

All cell references use the format `Cell N` where N is the zero-indexed cell number in `07_classification_metrics_thresholding.ipynb`.

---

### Option A: Single Video (~12 minutes)

#### Segment 1 — Opening & Motivation `[0:00–1:30]`

> **Show:** Cell 0 (header) and Cell 1 (learning objectives) on screen.

**Say:**

"In the previous notebook, we built a logistic regression pipeline and got 97% accuracy on the breast cancer dataset. We also saw a confusion matrix and explored how changing the threshold changes predictions. But we left several questions unanswered: What exactly are precision and recall? How do ROC and PR curves work? And most importantly, how do we choose a threshold based on business costs rather than mathematical convention? Today we answer all of these questions. This is the most metrics-dense notebook in the course, and by the end, you will have a complete evaluation toolkit that you will use for every classification task going forward."

> **Action:** Scroll through the 5 learning objectives in Cell 1, briefly reading each one aloud.

---

#### Segment 2 — Setup & Model Training `[1:30–3:00]`

> **Show:** Cell 2 (setup code), then **run** it. Point to imports.

**Say:**

"Notice the imports — this is where the notebook's power lives. We import precision_score, recall_score, f1_score for the core metrics. We import roc_curve, roc_auc_score for the ROC analysis. We import precision_recall_curve and average_precision_score for the PR analysis. And we import the display helpers — ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay — which generate publication-quality plots in one line. These are the tools of classification evaluation."

> **Show:** Cell 4 (data explanation), then **run** Cell 5 (load + train model).

**Say:**

"We load the breast cancer dataset, split 60-20-20 with stratification, and fit the same logistic regression pipeline from the previous notebook. Validation accuracy is about 97%. But today, 97% accuracy is just the starting point. We are going to decompose that number into its constituent parts and understand exactly where the model succeeds and where it fails."

---

#### Segment 3 — Confusion Matrix & Core Metrics `[3:00–5:30]`

> **Show:** Cell 7 (confusion matrix explanation), then **run** Cell 8 (heatmap + counts).

**Say:**

"The confusion matrix — four cells, four stories. True negatives: malignant cases correctly identified. False positives: malignant cases incorrectly called benign — these patients might miss treatment. False negatives: benign cases incorrectly called malignant — these patients get unnecessary follow-up. True positives: benign cases correctly identified. Every classification metric is arithmetic on these four numbers. Let me show you."

> **Show:** Cell 10 (metric formulas), then **run** Cell 11 (compute all metrics + classification report).

**Say:**

"Precision equals TP divided by TP plus FP — 'when I predict positive, how often am I right?' Recall equals TP divided by TP plus FN — 'of all actual positives, how many did I find?' F1 is the harmonic mean of the two — a single number that balances both. Specificity equals TN divided by TN plus FP — 'of all actual negatives, how many did I correctly rule out?' And accuracy is the total correct divided by the total. The classification_report function gives you all of this in one call, broken down by class, with support counts. These numbers are all above 0.95 on this dataset, which is great — but let me show you a scenario where they tell a very different story."

---

#### Segment 4 — ROC and PR Curves `[5:30–8:00]`

> **Show:** Cell 13 (ROC explanation), then **run** Cell 14 (ROC curve plot).

**Say:**

"The ROC curve shows True Positive Rate — which is just recall — on the y-axis, versus False Positive Rate on the x-axis. Each point on this curve represents one threshold setting. The curve sweeps from threshold 1.0 at the bottom-left to threshold 0.0 at the top-right. The dashed diagonal is a random classifier — AUC of 0.5. Our model's curve hugs the top-left corner with an AUC of about 0.99, meaning it achieves near-perfect separation between classes. Think of AUC as answering the question: 'If I pick one random positive and one random negative sample, what is the probability that the model ranks the positive higher?' At 0.99, it almost always gets this right."

> **Show:** Cell 16 (PR curve explanation), then **run** Cell 17 (PR curve plot).

**Say:**

"The precision-recall curve tells a complementary story. Precision on the y-axis, recall on the x-axis. The dashed horizontal line is the class prevalence — about 63% for benign in our dataset. A model that predicts everything as positive would land on this baseline. Our model stays close to the top-right corner, maintaining high precision even at high recall. The average precision score, also called PR-AUC, is about 0.99, confirming the strong performance. Now here is the key insight: when classes are heavily imbalanced — say 99% negative and 1% positive — ROC-AUC can still look high while the model actually performs poorly on the minority class. PR-AUC catches this because it ignores true negatives entirely. Use ROC for balanced data, PR for imbalanced data."

---

#### Segment 5 — Cost-Based Thresholding & Imbalance `[8:00–10:30]`

> **Show:** Cell 19 (Exercise 1 prompt), then **run** Cell 20 (cost-based threshold sweep).

**Say:**

"This is the most important exercise in the notebook. We assign real dollar costs: 50,000 dollars for a false negative — missing a malignant tumor — and 1,000 dollars for a false positive — an unnecessary biopsy. Then we sweep the threshold from 0.1 to 0.9, computing the total expected cost at each point. The result: the optimal threshold is well below 0.5, typically around 0.1 to 0.3. Why? Because the cost of missing cancer is 50 times higher than a false alarm, so the model should err heavily on the side of caution — flagging more cases as potentially malignant, accepting more false positives, to avoid the catastrophic false negatives. The left plot shows the U-shaped cost curve, the right plot shows precision and recall as functions of threshold."

> **Show:** Cell 23 (Exercise 2 prompt), then **run** Cell 24 (imbalanced dataset demonstration).

**Say:**

"Now the accuracy paradox. We create a synthetic dataset with 95% class 0 and 5% class 1. A naive baseline that always predicts class 0 achieves 95% accuracy — while having zero recall for the minority class. It misses every single positive case. The trained model does better, but its class-1 recall may still be disappointing because there are so few positive examples to learn from. This is why you must never trust accuracy alone on imbalanced data. Use precision, recall, F1, and especially PR-AUC."

---

#### Segment 6 — Metrics Dashboard & Closing `[10:30–12:00]`

> **Show:** Cell 27 (dashboard explanation), then **run** Cell 28 (create_metrics_dashboard function).

**Say:**

"We wrap everything into a reusable function: create_metrics_dashboard. Pass it true labels, predicted labels, and predicted probabilities, and it returns a dictionary with every metric we have discussed — accuracy, precision, recall, F1, specificity, ROC-AUC, PR-AUC, and the raw confusion matrix counts. This function is project-ready. In your course project, you will call this function for every candidate model and collect the results into a comparison table. No more forgetting a metric or computing it inconsistently."

> **Show:** Cell 30 (Wrap-Up).

**Say:**

"Three rules to remember. First: never trust accuracy alone — the confusion matrix reveals where the model fails. Second: choose thresholds based on business costs, not defaults — the optimal threshold depends on how much each error type costs your organization. Third: with imbalanced data, use PR curves over ROC curves — they give a more honest picture of performance on the class that matters. In the next notebook, we learn cross-validation — how to compare models fairly by evaluating them on multiple data splits using the metrics we learned today. See you there."

> **Show:** Cell 31 (Submission Instructions) briefly, then Cell 32 (Thank You).

---

### Option B: Three Shorter Videos (~4-5 minutes each)

---

#### Video 1: "The Confusion Matrix and Core Metrics" (~5 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–0:45` | Show header + objectives | Cell 0, Cell 1 | "Last notebook we built a logistic regression pipeline and got 97% accuracy. Today we take that number apart. We are going to learn precision, recall, F1, ROC curves, PR curves, and cost-based threshold selection. This is the complete classification evaluation toolkit." (Read objectives briefly.) |
| `0:45–1:30` | Run setup + train model | Cell 2, Cell 3, Cell 4, Cell 5, Cell 6 | "Same breast cancer dataset, same pipeline. We train the logistic regression and extract both hard predictions and probabilities. Validation accuracy about 97%. Now let us see what that 97% actually means." |
| `1:30–3:00` | Confusion matrix | Cell 7, Cell 8, Cell 9 | "Four cells: true negatives, false positives, false negatives, true positives. Each cell tells a different story. In our medical context, a false positive means an unnecessary biopsy — stressful but not dangerous. A false negative means a missed cancer — potentially fatal. These four numbers are the atoms of classification evaluation. Every metric is arithmetic on them." |
| `3:00–4:30` | Core metrics | Cell 10, Cell 11, Cell 12 | "Precision: TP over TP plus FP — when I predict positive, how often am I right? Recall: TP over TP plus FN — of all actual positives, how many did I find? F1: the harmonic mean that balances both. Specificity: TN over TN plus FP — how well do I rule out negatives? The classification report gives you all of these by class. Notice precision and recall can differ significantly even when accuracy is high." |
| `4:30–5:00` | Transition | — | "These metrics evaluate the model at one threshold. But what about all possible thresholds? That is what ROC and PR curves show us. Let us look at those next." |

---

#### Video 2: "ROC Curves, PR Curves, and Cost-Based Thresholds" (~5 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–0:30` | Recap | — | "Welcome back. We have the core metrics from the confusion matrix. Now we zoom out and look at model performance across all thresholds, not just the default 0.5." |
| `0:30–1:30` | ROC curve | Cell 13, Cell 14, Cell 15 | "The ROC curve: True Positive Rate on the y-axis, False Positive Rate on the x-axis. Each point is one threshold. The diagonal is random guessing, AUC 0.5. Our model hugs the top-left corner, AUC about 0.99. Think of AUC as: if I pick one positive and one negative sample at random, what is the probability the model ranks the positive higher? At 0.99, it almost always does." |
| `1:30–2:30` | PR curve | Cell 16, Cell 17, Cell 18 | "The PR curve: precision on y, recall on x. The baseline is the class prevalence — about 63% here. Our model stays near the top-right, AP about 0.99. Key insight: when classes are imbalanced, ROC can look great while PR exposes problems. If your positive class is rare — fraud, disease, equipment failure — use PR-AUC." |
| `2:30–4:30` | Cost-based thresholding | Cell 19, Cell 20, Cell 21 | "Now the business decision. False negative costs 50,000 dollars — missed cancer. False positive costs 1,000 dollars — unnecessary biopsy. We sweep thresholds, compute total cost at each one, and find the minimum. The optimal threshold lands well below 0.5 — around 0.1 to 0.3 — because the model should err on the side of catching cancer, even if it means more false alarms. The cost ratio drives the threshold, not convention." |
| `4:30–5:00` | Exercise prompt | Cell 22 | "Pause and answer the questions about cost-based thresholding. Why is the optimal threshold below 0.5? What would happen if you changed the cost ratio? Come back when you are done." |

---

#### Video 3: "The Accuracy Paradox, Dashboard, and Wrap-Up" (~4 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–1:30` | Accuracy paradox | Cell 23, Cell 24, Cell 25 | "Let us see why accuracy fails. We create a synthetic dataset with 95% class 0 and 5% class 1. A naive baseline that always predicts class 0 achieves 95% accuracy — while having zero recall for the minority class. It misses every positive case. This is the accuracy paradox. On real imbalanced problems — fraud detection, rare disease, network intrusion — accuracy is not just insufficient, it is actively misleading. Always check precision, recall, and AUC." |
| `1:30–2:30` | Metrics dashboard | Cell 27, Cell 28, Cell 29 | "We wrap everything into a reusable function: create_metrics_dashboard. One call, every metric, structured output. Accuracy, precision, recall, F1, specificity, ROC-AUC, PR-AUC, plus the raw confusion matrix counts. This function goes directly into your project. Every model you evaluate should be measured with the same dashboard." |
| `2:30–4:00` | Wrap-up + closing | Cell 30, Cell 31, Cell 32 | "Three rules. One: never trust accuracy alone. Two: choose thresholds based on business costs. Three: use PR curves for imbalanced data. This metrics toolkit is the evaluation backbone of the entire course. In the next notebook, we learn cross-validation — how to get stable, reliable estimates of these metrics by evaluating across multiple data splits. Submit your notebook to Brightspace and complete the quiz. See you there." |

---

*This guide was created to support video lecture recording for MGMT 47400 — Predictive Analytics, Purdue University.*
