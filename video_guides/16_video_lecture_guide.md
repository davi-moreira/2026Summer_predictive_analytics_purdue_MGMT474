# Video Lecture Guide: Notebook 16 — Error Analysis to Decisions - Thresholds, Calibration, and KPI Alignment

## At a Glance

This guide covers:

1. **Why NB16 exists** — It bridges the gap between model predictions and business decisions by teaching students that a classifier's output is a probability, not an action, and that translating probabilities into actions requires a cost matrix, a threshold, and calibrated scores

2. **Why it must follow NB15** — NB15 reveals *where* the model fails through error analysis. NB16 uses that understanding to set thresholds that minimize the business cost of those failures. Without knowing the failure segments, threshold tuning is blind optimization

3. **Why it must precede NB17** — NB17 audits the model for fairness across demographic groups. Threshold choices directly affect group-level outcomes (e.g., a lower threshold increases the selection rate for all groups, but not equally). Students need to understand threshold mechanics before analyzing their fairness implications

4. **Why each library/tool is used:**
   - `make_classification` — generates a synthetic dataset with controllable class imbalance and noise
   - `calibration_curve` — diagnoses whether predicted probabilities match observed frequencies
   - `CalibratedClassifierCV` — post-hoc recalibration using isotonic regression
   - `confusion_matrix` — computes TP/FP/FN/TN counts at each threshold for cost calculation
   - Cost-based threshold sweep — translates confusion matrix counts into dollar values

5. **Common student questions** with answers (e.g., why not just use 0.50? does calibration change AUC?)

6. **Connection to the full course arc** — how the threshold/calibration skills from NB16 connect to fairness auditing (NB17), deployment monitoring (NB18), and the final project

7. **Suggested video structure** — both single-video (~12 min) and three-video (~4 min each) formats with speaking prompts, timestamps, and notebook cell references

---

## Purpose of This Document

This guide is your **lecture companion** for recording the video(s) that accompany `16_decision_thresholds_calibration.ipynb`. It explains the *why* behind every design choice — why we use cost matrices instead of accuracy, why calibration matters for probability-based decisions, and why sensitivity analysis is essential before committing to a threshold. Use it as speaking notes, not as a script: the goal is to internalize the reasoning so you can explain it naturally on camera.

---

## 1. Why This Notebook Exists

Up to this point, students have been evaluating classifiers using metrics like AUC, accuracy, precision, and recall. These metrics summarize model quality but do not tell you *what to do* with the model's output. A classifier that outputs P(positive) = 0.65 — should the business act on that? The answer depends on the cost of acting versus the cost of not acting, and that is a business decision, not a statistical one.

Notebook 16 teaches students to:

1. **Define a cost matrix** that maps each confusion-matrix outcome (TP, FP, FN, TN) to a dollar value.
2. **Sweep thresholds** from low to high, computing the expected business value at each one.
3. **Select the threshold that maximizes business value**, not accuracy.
4. **Diagnose calibration** — are the model's probability estimates trustworthy?
5. **Apply recalibration** when probabilities are systematically biased.
6. **Run sensitivity analysis** to verify the threshold is robust to cost-estimation uncertainty.

This notebook exists because the default 0.50 threshold is almost never optimal in practice. In medical screening, missing a positive case (FN) is catastrophic, so you lower the threshold to catch more positives at the cost of more false alarms. In fraud detection, investigating a false positive is expensive, so you raise the threshold to minimize wasted investigations. The right threshold depends on the business context, and this notebook teaches students to find it.

---

## 2. Why It Must Come After Notebook 15

Notebook 15 provides two critical foundations that NB16 builds on:

| NB15 Concept | How NB16 Uses It |
|---|---|
| **Error analysis by segment** | NB15 reveals that the model fails more in certain segments (e.g., high-income properties, specific geographic areas). NB16's threshold tuning is most impactful when you understand *where* the errors concentrate. A threshold change affects all predictions — but the impact on total cost depends on which errors are most expensive, and that depends on the error distribution NB15 characterized. |
| **Feature importance and PDP** | NB15 shows which features drive predictions and how the predicted value responds to feature changes. NB16 extends this by asking: given those predictions, at what point should the business *act*? Understanding the prediction surface (NB15) is a prerequisite for setting decision boundaries (NB16). |
| **Honest communication** | NB15 teaches students to disclose model limitations. NB16 adds decision-level disclosure: "we set the threshold at 0.35, which catches 95% of positives but generates 40% false alarms, because the cost of missing a positive ($150) vastly exceeds the cost of a false alarm ($30)." |

**In short:** NB15 answers "where does the model fail?" NB16 answers "given those failures, how should we decide?"

If you tried to teach thresholds before error analysis, students would not understand *why* the optimal threshold is not 0.50. If you skipped NB16, students would deploy classifiers with the default threshold and unknowingly accept suboptimal business outcomes.

---

## 3. Why It Must Come Before Notebook 17

Notebook 17 introduces **fairness metrics** (demographic parity, equal opportunity, equalized odds) and **model cards**. Its learning objectives assume students understand:

- That the classification threshold directly determines the selection rate, TPR, and FPR for each group
- That changing the threshold changes all group-level metrics simultaneously
- That probability calibration affects whether threshold-based policies are reliable

NB17's fairness analysis evaluates *at a specific threshold* — the threshold chosen in NB16. If the threshold is set to maximize business value for the overall population, it may not be equally fair to all subgroups. NB17 reveals and measures those disparities. But students can only understand that analysis if they first understand how thresholds work mechanically.

The deliberate sequence is:

```
NB15: Interpret the champion + Find failure segments
            |
NB16: Set the optimal business threshold + Calibrate probabilities
            |
NB17: Audit the threshold for fairness across demographic groups
```

Each notebook adds one layer of decision refinement. By NB17, students understand thresholds mechanically (NB16) and can analyze whether those thresholds produce equitable outcomes across groups.

---

## 4. Why These Specific Libraries and Tools

### 4.1 `sklearn.datasets.make_classification`

**What it does:** Generates a synthetic binary classification dataset with controllable parameters: number of features, class weights, label noise (`flip_y`), and the number of informative versus redundant features.

**Why we use it (instead of the breast cancer dataset):**

1. **Controllable class imbalance.** We set `weights=[0.7, 0.3]` to create a 70/30 imbalance. This is realistic — most business problems have unbalanced classes (more non-fraud than fraud, more healthy than sick). On a balanced dataset, threshold tuning has less impact and the lesson is weaker.

2. **Controllable noise.** The `flip_y=0.05` parameter adds 5% label noise, simulating real-world data where some labels are incorrect. This noise means the model's probabilities should not be perfectly calibrated, motivating the calibration section.

3. **Larger sample size.** With 5,000 samples (versus 569 for breast cancer), the calibration curves are smoother and the threshold sweep is more stable. Students see cleaner patterns that are easier to interpret.

**What to emphasize on camera:** We use synthetic data here on purpose. It lets us control the class imbalance and noise level so the threshold and calibration lessons are clear. In your project, you will apply the same techniques to real data.

### 4.2 `sklearn.calibration.calibration_curve`

**What it does:** Bins predicted probabilities and computes the actual fraction of positives in each bin. The result is a reliability diagram: predicted probability on the x-axis, observed frequency on the y-axis.

**Why we use it:**

1. **Diagnoses trustworthiness.** If the model says P(positive) = 0.80 for a group of 100 samples, and only 60 of those are actually positive, the model is overconfident. The reliability diagram reveals this pattern visually: points below the diagonal mean overconfidence, points above mean underconfidence.

2. **Motivates calibration.** You cannot make cost-based decisions using probabilities that do not correspond to real frequencies. If P = 0.80 actually means P = 0.60, your expected-value calculation is off by 33%. The calibration curve tells you whether recalibration is needed.

**What to emphasize on camera:** Random Forests are known for having poorly calibrated probabilities — they tend to push scores toward 0 and 1 (overconfident at the extremes). The calibration curve will typically show this pattern. Logistic Regression, by contrast, is generally well-calibrated because it directly models log-odds.

### 4.3 `sklearn.calibration.CalibratedClassifierCV`

**What it does:** Wraps a fitted classifier and applies a post-hoc monotone transformation (Platt scaling or isotonic regression) that maps raw scores to better-calibrated probabilities.

**Why we use it:**

1. **Post-hoc correction.** You do not need to retrain the model. The calibrator learns a mapping from raw scores to calibrated probabilities on a held-out set. This is fast and does not change the model's ranking of instances (AUC stays the same).

2. **Isotonic regression** is the default choice because it is non-parametric and can correct complex miscalibration patterns. Platt scaling (sigmoid) is better when the miscalibration is simple (a monotone S-curve), but isotonic is safer in general.

3. **Production relevance.** In deployed systems, recalibration is a standard post-processing step. Teaching it here prepares students for real-world model pipelines where the raw classifier and the calibration layer are separate components.

**What to emphasize on camera:** Calibration does not change the model's ranking — a sample that was ranked #1 before calibration is still ranked #1 after. What changes is the *scale* of the scores. After calibration, a predicted probability of 0.80 should mean roughly 80% of such samples are actually positive.

### 4.4 Cost Matrix and `confusion_matrix`

**What it does:** The cost matrix assigns a dollar value to each outcome: TP = +$100, FP = -$30, FN = -$150, TN = $0. The `confusion_matrix` function counts TP, FP, FN, TN at each threshold. Multiplying counts by costs gives the expected value.

**Why we use it:**

1. **Translates metrics into dollars.** AUC and F1 are abstract. "This threshold generates $8,500 in expected value per 1,000 cases" is concrete and actionable for business stakeholders.

2. **Reveals cost asymmetry.** In our setup, FN costs 5x more than FP ($150 vs $30). This asymmetry pushes the optimal threshold below 0.50 because the cost of missing a positive overwhelms the cost of a false alarm.

3. **Foundation for sensitivity analysis.** By parameterizing the cost matrix, we can vary individual costs and see how the optimal threshold shifts. This robustness check is essential when cost estimates are uncertain.

**What to emphasize on camera:** The cost matrix is the bridge between data science and business strategy. If you do not know the costs, ask the business. If they do not know either, run sensitivity analysis over a plausible range.

### 4.5 Sensitivity Analysis

**What it does:** Varies the FN cost from $100 to $250 in steps and records the optimal threshold and expected value at each point.

**Why we use it:**

- **Cost estimates are never exact.** A business that says "a missed detection costs $150" is giving you an estimate, not a law of physics. Sensitivity analysis tests whether your recommended threshold is robust to estimation error.

- **If the threshold barely moves** as FN cost doubles, the decision is robust and you can recommend it confidently. **If the threshold swings dramatically**, you need better cost estimates before committing.

**What to emphasize on camera:** Sensitivity analysis is what turns a model recommendation into a trustworthy business policy. Without it, you are betting the deployment on a single cost estimate that might be wrong.

---

## 5. Key Concepts to Explain on Camera

### 5.1 The Default 0.50 Threshold Is Almost Never Optimal

Most students (and many practitioners) accept the 0.50 threshold without question. Scikit-learn's `.predict()` method uses 0.50 by default, which reinforces the assumption. But 0.50 is only optimal when:

- The classes are perfectly balanced (50/50)
- The costs of false positives and false negatives are equal
- You care about accuracy and nothing else

In practice, none of these conditions hold. Classes are imbalanced, costs are asymmetric, and businesses care about dollars, not accuracy. The threshold sweep demonstrates this concretely: the optimal threshold in this notebook is typically 0.25-0.40, well below 0.50, because FN costs ($150) dominate FP costs ($30).

**Analogy for students:** A fire alarm with a threshold of 0.50 means "sound the alarm only if there is a 50% chance of fire." That is absurdly conservative — you want the alarm to go off at 10% or 20% because the cost of missing a real fire (FN = burned building) vastly exceeds the cost of a false alarm (FP = unnecessary evacuation). The same logic applies to any model where FN costs exceed FP costs.

### 5.2 Calibration vs. Discrimination

Discrimination is the model's ability to rank positive cases above negative cases — measured by AUC. Calibration is whether the predicted probabilities match observed frequencies. A model can have perfect discrimination (AUC = 1.0) but terrible calibration (says 0.90 when the true rate is 0.60), and vice versa.

Why this distinction matters:

- **If you only use the model for ranking** (e.g., "show me the top 100 highest-risk cases"), calibration does not matter — only AUC matters.
- **If you use the model for probability-based decisions** (e.g., "act when P > threshold"), calibration is critical because the expected-value calculation depends on the probabilities being accurate.

The calibration section of this notebook shows students how to diagnose and fix miscalibration without changing the model's ranking ability.

### 5.3 The Threshold-Value Curve Has a Single Peak

The expected-value-vs-threshold curve is typically an inverted U: low thresholds catch too many false positives (costly alarms), high thresholds miss too many positives (costly omissions), and the peak is where the marginal cost of an additional FP exactly equals the marginal cost of an additional FN. This single-peaked shape means there is a clear optimal point.

If the curve is flat near the peak, the exact threshold does not matter much — any value in the flat region gives similar business value. If the curve is sharply peaked, the threshold must be precise and sensitivity analysis is essential.

### 5.4 Communicating in Multiple Units

The decision policy summary in the notebook reports three views of the same decision: dollar value (for finance), confusion-matrix counts (for operations), and precision/recall/F1 (for data science). Tailoring the communication to the audience is a professional skill that students need to practice.

---

## 6. What Students Should Take Away

After watching your video and completing this notebook, students should be able to answer these questions confidently:

1. **"Why is the default 0.50 threshold usually wrong?"** — Because it assumes balanced classes, equal costs, and accuracy as the only objective. In practice, costs are asymmetric, classes are imbalanced, and business value matters more than accuracy. The optimal threshold depends on the cost matrix.

2. **"What does a calibration curve tell me?"** — Whether the model's predicted probabilities correspond to actual frequencies. Points on the diagonal mean perfect calibration. Points below the diagonal mean the model is overconfident. Points above mean underconfident. Miscalibrated probabilities lead to miscalculated expected values.

3. **"Does calibration change AUC?"** — No. Calibration remaps scores monotonically, preserving the ranking of instances. AUC depends only on ranking, so it stays the same. What changes is the scale of the scores — after calibration, a predicted 0.80 actually means about 80% of such cases are positive.

4. **"How do I set the threshold for my project?"** — Define a cost matrix with your stakeholder, sweep thresholds on the validation set computing expected value at each point, pick the threshold that maximizes value, and run sensitivity analysis on the cost assumptions. Present the recommendation with dollar values, counts, and rates.

5. **"What if I don't know the exact costs?"** — Run sensitivity analysis. Vary the uncertain costs over a plausible range and see how the optimal threshold shifts. If it barely moves, your decision is robust. If it swings wildly, invest in getting better cost estimates before deploying.

---

## 7. Common Student Questions (Anticipate These)

**Q: "If the FN cost is much higher than the FP cost, why not just set the threshold to zero and classify everyone as positive?"**
A: Because false positives still have a cost, even if it is smaller. At threshold = 0, every sample is classified positive — you catch all true positives but also generate the maximum number of false alarms. The total cost of all those FPs can exceed the total cost of the FNs you would have avoided at a slightly higher threshold. The optimal threshold balances these opposing forces.

**Q: "Why use isotonic regression for calibration instead of Platt scaling?"**
A: Isotonic regression is non-parametric and can correct any monotone miscalibration pattern, including the step-like patterns that Random Forests produce. Platt scaling (logistic sigmoid) assumes the miscalibration follows a specific S-curve shape, which is a good assumption for SVMs and neural networks but often too rigid for tree ensembles. For general use, isotonic is the safer default.

**Q: "Why do we fit the calibrator on the validation set instead of the training set?"**
A: Because the calibrator needs to learn the mapping between predicted probabilities and actual outcomes. If you fit it on the training set, the predicted probabilities are overly optimistic (the model memorizes training data), and the calibrator learns the wrong mapping. Using the validation set gives realistic probability estimates that generalize to new data.

**Q: "Can I use different thresholds for different customer segments?"**
A: Yes, and that is sometimes the right approach — it is called segment-specific threshold tuning. If the cost of a false negative is $500 for high-value customers but only $50 for low-value customers, different thresholds are justified. However, segment-specific thresholds add complexity to deployment and monitoring, and they may raise fairness concerns (NB17).

**Q: "How often should we re-evaluate the threshold in production?"**
A: Whenever the cost assumptions change, the class distribution shifts, or the model is retrained. In practice, quarterly re-evaluation is common for stable business contexts. For fast-changing domains (e.g., fraud detection), monthly or even weekly re-evaluation may be necessary.

---

## 8. Connection to the Broader Course Arc

| Week | Notebooks | What NB16's Threshold/Calibration Skills Enable |
|---|---|---|
| **Week 1** | 01-05 | Regression metrics (NB03) measure prediction quality. NB16 extends this to classification by showing that metrics alone do not determine the optimal action. |
| **Week 2** | 06-10 | Classification metrics (NB07) — precision, recall, ROC-AUC — describe model quality at various thresholds. NB16 shows how to *choose* the threshold using business costs. |
| **Week 3** | 11-15 | NB14 selects the champion; NB15 interprets and error-analyzes it. NB16 takes the champion's probability outputs and converts them into actionable decisions. |
| **Week 4** | 16-20 | NB16 opens Week 4 with a decision-centric lens. NB17 audits the threshold for fairness. NB18 deploys the model with the chosen threshold. NB19-20 apply threshold tuning in the final project. |

**Thresholds and calibration are the bridge between model quality and business impact.** Every metric before NB16 described how good the model is. NB16 is where students learn to ask "given this model, what should the business *do*?" That question carries through the rest of the course.

---

## 9. Suggested Video Structure

Below are two recording options. Each segment includes the **speaking prompt** (what to say), **timestamps**, and the **notebook cells** to show on screen.

All cell references use the format `Cell N` where N is the zero-indexed cell number in `16_decision_thresholds_calibration.ipynb`.

---

### Option A: Single Video (~12 minutes)

#### Segment 1 — Opening & The Decision Problem `[0:00–2:00]`

> **Show:** Cell 0 (header) and Cell 1 (learning objectives) on screen.

**Say:**

"Welcome back. In the last two notebooks we selected a champion model and interpreted it — we know what it learns and where it fails. But we have not yet answered the most important question for a business: given a prediction, what should we *do*? A classifier says the probability of fraud is 0.65. Do we block the transaction? Investigate it? Ignore it? That decision depends not on the model's AUC, but on the cost of acting versus the cost of not acting. Today we learn to translate model outputs into business decisions using cost matrices, threshold optimization, and probability calibration. By the end, you will be able to recommend a specific threshold, justify it in dollar terms, and test whether your recommendation is robust."

> **Action:** Scroll through the 5 learning objectives in Cell 1, briefly reading each one aloud.

---

#### Segment 2 — Setup & Data `[2:00–3:00]`

> **Run** Cell 3 (imports). Point to `calibration_curve` and `CalibratedClassifierCV`.

**Say:**

"Two new imports: `calibration_curve` for diagnosis — are the probabilities trustworthy? And `CalibratedClassifierCV` for correction — if they are not, fix them. We also use `make_classification` to generate synthetic data with a 70-30 class imbalance and 5 percent label noise."

> **Run** Cell 6 (generate data + split).

**Say:**

"Five thousand samples, split 60-20-20. About 3,000 training, 1,000 validation, 1,000 test. The class distribution shows the 70-30 imbalance — about 700 positives and 300 negatives in the validation set. This imbalance is realistic and it is exactly why the default 0.50 threshold will not be optimal."

> **Run** Cell 8 (train Random Forest + ROC-AUC).

**Say:**

"Random Forest trained, ROC-AUC on validation is strong. But AUC tells us the model can *rank* cases well — it does not tell us *where to draw the line*. That is what the cost matrix will determine."

---

#### Segment 3 — Cost Matrix & Threshold Sweep `[3:00–6:30]`

> **Show:** Cell 10 (cost matrix explanation), then **run** Cell 11 (define cost matrix + expected value function).

**Say:**

"Here is the cost matrix. True positive: we gain $100 — correctly catching a positive case. False positive: we lose $30 — wasted investigation. False negative: we lose $150 — we missed a positive case. True negative: zero — no action needed. Two things to notice. First, missing a positive costs five times more than a false alarm. This asymmetry will push the optimal threshold below 0.50 because we want to avoid the expensive misses even if it means more false alarms. Second, the `compute_expected_value` function simply multiplies counts by costs and sums. Simple arithmetic, but powerful because it converts abstract confusion-matrix counts into dollars."

> **Show:** Cell 13 (threshold sweep explanation), then **run** Cell 14 (sweep thresholds).

**Say:**

"We sweep thresholds from 0.10 to 0.85 in steps of 0.05. At each threshold, we classify the validation set, compute the confusion matrix, and multiply by the cost matrix to get total expected value. The results table shows each threshold alongside its dollar value and the four confusion-matrix counts. Look at the bottom: the optimal threshold is typically around 0.25 to 0.40 — well below 0.50. Why? Because the FN cost of $150 dominates. The model would rather generate extra false alarms at $30 each than miss a positive case at $150."

> **Run** Cell 16 (threshold visualization).

**Say:**

"The left panel shows the expected-value curve — it rises, peaks at our optimal threshold marked by the red dashed line, and then falls as we become too conservative and start missing positives. The right panel shows the trade-off directly: as the threshold goes up, true positives and false positives both drop, but false negatives climb. The optimal threshold is where the cost savings from fewer FPs are exactly offset by the rising cost of FNs."

> **Mention:** "Pause the video here and complete Exercise 1 — select a threshold and justify it with business reasoning." Point to Cell 18 (PAUSE-AND-DO 1).

---

#### Segment 4 — Calibration `[6:30–9:00]`

> **Show:** Cell 20 (calibration intro), then **run** Cell 21 (calibration curve + probability histogram).

**Say:**

"Now a critical question: are the model's probabilities trustworthy? If the model says 0.80, does that really mean 80 percent of such cases are positive? The calibration curve answers this. The diagonal is perfect calibration — where we want to be. Each dot shows the actual positive rate in a probability bin. For Random Forests, you will typically see the dots veering below the diagonal at high probabilities — the model is overconfident. The right panel shows probability distributions for each class. Well-separated histograms confirm good discrimination. But good discrimination does not mean good calibration."

> **Run** Cell 24 (apply isotonic calibration + comparison plot).

**Say:**

"We fix the calibration with isotonic regression. The `CalibratedClassifierCV` wraps the trained model and learns a mapping from raw scores to calibrated probabilities. The comparison plot shows before and after: the calibrated curve — the squares — hugs the diagonal more tightly than the original circles. Crucially, this does not change the model's AUC. It only rescales the scores so they mean what they claim to mean. After calibration, a predicted 0.80 actually corresponds to about 80 percent true positives."

> **Mention:** "Pause the video here and complete Exercise 2 — assess whether calibration is needed and justify your recommendation." Point to Cell 26 (PAUSE-AND-DO 2).

---

#### Segment 5 — Decision Policy & Sensitivity Analysis `[9:00–11:00]`

> **Run** Cell 29 (decision policy summary).

**Say:**

"Here is the final recommendation in three formats. First, dollars: the optimal threshold generates a total expected value of X dollars, which is Y dollars per case. Second, counts: Z true positives, W false positives, at this threshold. Third, rates: the classification report shows precision, recall, and F1 for both classes. Presenting in all three formats makes the recommendation accessible to different audiences — finance cares about dollars, operations cares about workload, and data science cares about rates."

> **Run** Cell 32 (sensitivity analysis).

**Say:**

"But how confident are we in the $150 FN cost? What if it is really $100, or $250? The sensitivity analysis varies the FN cost and records how the optimal threshold shifts. If the threshold barely changes — say it moves from 0.30 to 0.35 as FN cost doubles — the recommendation is robust. If it jumps from 0.20 to 0.50, you need better cost estimates before deploying. On this dataset, the threshold is usually quite stable, which means our recommendation is defensible even with imperfect cost knowledge."

---

#### Segment 6 — Wrap-Up & Closing `[11:00–12:00]`

> **Show:** Cell 34 (Wrap-Up).

**Say:**

"Let me leave you with three takeaways. First: model performance metrics do not pay the bills — business value does. Always optimize for the cost matrix, not for accuracy. Second: check calibration before using probabilities for decisions. Miscalibrated scores lead to miscalculated expected values. Third: never present a single threshold without sensitivity analysis. Cost assumptions are estimates, not laws of physics.

This notebook marks a shift in the course. Everything before today was about building and evaluating models. From today forward, it is about using models responsibly — for decisions, for fairness, and for deployment. In the next notebook, we audit the model for fairness across demographic groups, compute group-level disparities, and draft a model card. See you there."

> **Show:** Cell 35 (Submission Instructions) briefly, then Cell 36 (Thank You).

---

### Option B: Three Shorter Videos (~4 min each)

---

#### Video 1: "From Prediction to Decision: Cost Matrices and Thresholds" (~5 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–0:45` | Show header + objectives | Cell 0, Cell 1 | "Welcome back. We selected and interpreted our champion model. Today we answer the question that matters most to the business: given a prediction, what should we do? Here are our five learning objectives." (Read them briefly.) |
| `0:45–1:30` | Run setup + data | Cell 3, Cell 6, Cell 8 | "Synthetic dataset, 5,000 samples, 70-30 class imbalance, 60-20-20 split. Random Forest trained with strong ROC-AUC. But AUC does not tell us where to draw the decision line — the cost matrix does." |
| `1:30–2:30` | Define cost matrix | Cell 10, Cell 11 | "Cost matrix: TP gains $100, FP costs $30, FN costs $150, TN costs nothing. FN costs five times more than FP. This asymmetry will push the optimal threshold below 0.50 because missing a positive is far more expensive than a false alarm." |
| `2:30–4:15` | Run threshold sweep | Cell 13, Cell 14, Cell 16 | "We sweep thresholds from 0.10 to 0.85 and compute total expected value at each point. The optimal threshold is around 0.30 — well below the default 0.50. The left chart shows the inverted-U value curve. The right chart shows the trade-off: as threshold rises, we catch fewer positives but also trigger fewer alarms. The peak is where these costs balance." |
| `4:15–5:00` | Exercise 1 prompt | Cell 18 | "Pause the video now and complete Exercise 1. Select a threshold, justify it with business reasoning, and describe the trade-offs. Come back when you are done." |

---

#### Video 2: "Calibration: Are Your Probabilities Trustworthy?" (~4 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–0:30` | Recap | — | "Welcome back. We found the optimal threshold based on costs. But that analysis assumes the predicted probabilities are accurate. What if a predicted 0.80 really means only 0.60? Our expected-value calculation would be off by 33 percent. Let's check." |
| `0:30–1:30` | Run calibration curve | Cell 20, Cell 21 | "The calibration curve plots predicted probability versus observed frequency. Diagonal means perfect. Our Random Forest veers below the diagonal at high probabilities — it is overconfident. The probability histogram shows well-separated distributions, meaning good discrimination, but discrimination is not calibration." |
| `1:30–3:00` | Apply calibration | Cell 23, Cell 24 | "Isotonic regression wraps the trained model and remaps raw scores to calibrated probabilities. The before-and-after comparison shows the calibrated curve hugging the diagonal more closely. AUC stays the same — calibration only changes the scale, not the ranking. After calibration, a predicted 0.80 means approximately 80 percent of such cases are truly positive." |
| `3:00–4:00` | Exercise 2 prompt | Cell 26, Cell 27 | "Pause and complete Exercise 2. Assess the calibration quality, decide whether to use calibrated probabilities, and justify your recommendation. Come back when you are done." |

---

#### Video 3: "Decision Policy, Sensitivity Analysis & Wrap-Up" (~4 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–1:00` | Run decision summary | Cell 28, Cell 29 | "The decision policy in three formats: dollars for the CFO, confusion-matrix counts for operations, classification report for data science. All describe the same threshold — they just speak different languages. Always present recommendations in the units your audience cares about." |
| `1:00–2:30` | Run sensitivity analysis | Cell 31, Cell 32 | "We vary the FN cost from $100 to $250 and see how the threshold moves. If it barely shifts, the recommendation is robust — you can deploy confidently even with imperfect cost estimates. If it swings, invest in better cost measurement before committing. On this dataset, the threshold is stable, which is good news." |
| `2:30–3:30` | Show wrap-up | Cell 34 | "Three rules. One: optimize for business value, not accuracy. Two: check calibration before using probabilities for decisions. Three: never present a threshold without sensitivity analysis." |
| `3:30–4:00` | Closing | Cell 35, Cell 36 | "Everything before today was about building models. From today forward, it is about using models responsibly. Next notebook: fairness metrics, group-level disparities, and model cards. Submit your notebook to Brightspace and complete the quiz. See you there." |

---

*This guide was created to support video lecture recording for MGMT 47400 — Predictive Analytics, Purdue University.*
