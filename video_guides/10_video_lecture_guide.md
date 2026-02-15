# Video Lecture Guide: Notebook 10 — Midterm - Business-Case Predictive Strategy Practicum + Project Baseline Submission

## At a Glance

This guide covers:

1. **Why NB10 exists** — It is the Week 2 capstone that forces students to apply everything from NB01-NB09 to realistic business scenarios *without writing code*, testing whether they truly understand the strategic reasoning behind predictive analytics

2. **Why it must follow NB09** — NB09 completes the full toolkit (pipelines, metrics, cross-validation, tuning, feature engineering). NB10 tests whether students can wield that toolkit strategically, not just mechanically

3. **Why it must precede NB11** — NB11 opens the tree-based methods arc (Week 3). The midterm provides a natural pause point for reflection before introducing an entirely new model family. Students also submit Project Milestone 2 (baseline model), which requires the tuning and evaluation skills from NB06-NB09

4. **Why this notebook is different:**
   - No new scikit-learn tools are taught
   - The focus is on *design thinking*: problem framing, metric selection, leakage identification, and evaluation planning
   - Cases involve real-world business contexts (churn prediction, loan default, medical diagnosis)
   - The project milestone component bridges the case analysis to students' own datasets

5. **Common student questions** with answers (e.g., why is this not a coding exam?, how do I choose the right metric for a business problem?)

6. **Connection to the full course arc** — NB10 is the checkpoint that ensures students internalized the reasoning from Weeks 1-2 before the complexity ramps up in Weeks 3-4

7. **Suggested video structure** — both single-video (~12 min) and three-video (~4 min each) formats with speaking prompts, timestamps, and notebook cell references

---

## Purpose of This Document

This guide is your **lecture companion** for recording the video(s) that accompany `10_midterm_casebook.ipynb`. Unlike every other video guide in this course, this one does not walk through code — because the notebook itself contains almost no code. Instead, the video should orient students to the *strategic thinking* the midterm requires: how to translate a business problem into a predictive analytics plan, how to select metrics aligned with cost structures, how to spot leakage, and how to design an evaluation protocol. Use it as speaking notes, not as a script: the goal is to internalize the reasoning so you can explain it naturally on camera.

---

## 1. Why This Notebook Exists

Notebooks 01-09 teach the *mechanics* of predictive analytics: splitting, preprocessing, modeling, evaluating, tuning. Students can execute each step, but can they *design* a complete predictive analytics strategy from a blank page? NB10 answers that question.

The midterm casebook presents two main business cases (customer churn, loan default) and one optional mini-case (medical diagnosis). Each case provides a business context, available data, cost structure, and constraints. Students must produce a structured plan — not code — that covers problem framing, metric selection, split strategy, leakage risks, model shortlist, and threshold logic.

This is the most important assessment in the first half of the course because it tests *understanding*, not just *execution*. A student who can run `Pipeline.fit()` but cannot explain why ROC-AUC is more appropriate than accuracy for an imbalanced churn dataset has not actually learned predictive analytics.

---

## 2. Why It Must Come After Notebook 09

NB10 is the culmination of Weeks 1-2. Every case question maps directly to a concept taught in a prior notebook:

| NB10 Case Question | Source Notebook |
|---|---|
| **"What is the prediction target, unit, and horizon?"** | NB01 (problem framing, EDA) |
| **"How should you split the data?"** | NB01 (train/val/test), NB08 (cross-validation) |
| **"What preprocessing is needed?"** | NB02 (pipelines), NB04 (feature engineering) |
| **"Which metric aligns with business costs?"** | NB03 (regression metrics), NB07 (classification metrics, thresholding) |
| **"What are the leakage risks?"** | NB01 (leakage introduction), NB02 (pipeline safety), NB08 (CV leakage) |
| **"Which models would you try?"** | NB05 (regularization), NB06 (logistic regression), NB09 (model selection) |
| **"How would you set the decision threshold?"** | NB07 (precision-recall tradeoff, threshold tuning) |
| **"What would you monitor in production?"** | NB08 (CV as generalization estimate), NB09 (evaluation plans) |

**In short:** NB10 is the exam for NB01-NB09. If you tried to administer it before NB09, students would lack the vocabulary and tools to answer Case 2 (expected value calculations, constraint handling, class imbalance strategies). If you placed it after NB11, students would conflate tree-based methods into their answers even though the cases do not require them.

The Project Milestone 2 component also requires the tuning and evaluation workflow from NB09. Students must submit a baseline model with a proper pipeline, evaluation on validation data, and a comparison table — all skills that crystallized in NB09.

---

## 3. Why It Must Come Before Notebook 11

NB11 introduces decision trees — the first model family in a completely new paradigm (tree-based methods). The shift from linear/logistic models to tree-based models is the largest conceptual jump in the course. Placing the midterm here creates three benefits:

1. **Natural pause point.** Students consolidate Weeks 1-2 knowledge before absorbing new material. Without this pause, the continuous stream of new models would overwhelm working memory.

2. **Project baseline checkpoint.** The Project Milestone 2 submission ensures students have a working pipeline and evaluation plan for their own dataset before Week 3 adds more modeling options. Students who fall behind here will struggle to keep up with the rapid-fire NB11-NB13 tree ensemble sequence.

3. **Clean conceptual boundary.** Week 2 ends with strategic thinking (NB10); Week 3 begins with a concrete new algorithm (NB11). This boundary prevents students from confusing "which model to use" (a strategic question) with "how does this model work" (a technical question).

The sequence is:

```
NB09: Tuning + feature engineering + project baseline scaffold
            |
NB10: Midterm — prove you can DESIGN a strategy (no new tools)
            |
NB11: Decision Trees — first tree-based method (new paradigm)
```

---

## 4. Why These Specific Cases and Components

### 4.1 Case 1: Customer Churn Prediction (StreamFlix)

**What it tests:** End-to-end problem framing for a binary classification task with asymmetric costs.

**Why this case:**

1. **Universally relatable.** Every student has used a streaming service. The churn scenario requires no domain expertise beyond common sense, so the focus stays on analytics methodology rather than industry knowledge.

2. **Asymmetric costs make metric selection non-trivial.** Acquiring a new customer costs $50; retention costs $10. This cost structure means false negatives (missing a churner) are 5x more expensive than false positives (sending an unnecessary retention offer). Students must connect this asymmetry to metric choice (precision-recall focus, not accuracy) and threshold selection (bias toward recall).

3. **Leakage risks are realistic and subtle.** Features like "account status after churn" or "last login = 30 days ago" are obvious leaks, but features like "support tickets in the last 7 days" require more careful reasoning — is this a predictor of churn, or a consequence of already deciding to churn?

**What to emphasize on camera:** The goal is not the "right answer" — it is a *justified* answer. A student who says "use accuracy" with a thoughtful cost analysis showing that costs are approximately symmetric would score better than a student who says "use F1" with no justification.

### 4.2 Case 2: Loan Default Prediction

**What it tests:** Evaluation planning under regulatory constraints and extreme class imbalance.

**Why this case:**

1. **Regulatory constraint adds realism.** The 3% maximum default rate is not just a performance target — it is a hard constraint that the model must satisfy regardless of other metrics. Students must explain how to enforce this constraint (e.g., threshold calibration on the validation set, monitoring in production).

2. **Expected value calculation is concrete.** With a $500 profit per good loan and a $7,000 loss per default, students can compute the expected value of each decision (approve/reject) as a function of the model's predicted probability. This connects abstract metrics to dollar amounts.

3. **Class imbalance is extreme.** At 1-2% default rate, accuracy of 98% is trivially achievable by predicting "no default" for every applicant. Students must recognize this trap and propose appropriate strategies (stratified sampling, class weights, PR-AUC instead of ROC-AUC).

### 4.3 Mini-Case 3: Medical Diagnosis (Optional Bonus)

**What it tests:** Metric selection when false negatives are life-threatening.

**Why this case:** It pushes the cost asymmetry to an extreme (missing a disease = life-threatening vs. unnecessary test = $5,000). The 0.1% prevalence rate (100 positives in 100,000 patients) makes this the most severe imbalance scenario in the course. Students who attempt it demonstrate mastery.

### 4.4 Project Milestone 2: Baseline Model + Evaluation Plan

**What it tests:** Can students apply the case analysis skills to their own dataset?

**Why it is here:** The milestone forces students to produce a concrete artifact — a working baseline pipeline, a comparison table, and a documented evaluation plan — before Week 3 introduces new model families. This ensures that students are not just passively absorbing lecture material but actively building their project.

---

## 5. Key Concepts to Explain on Camera

### 5.1 Problem Framing: Target, Unit, Horizon

The first three questions in every case (prediction target, prediction unit, prediction horizon) are not technical — they are *strategic*. Getting them wrong means building the wrong model entirely.

- **Prediction target:** What label are we trying to predict? (e.g., "binary: churned within 30 days or not")
- **Prediction unit:** What is one row in the dataset? (e.g., "one customer-month observation")
- **Prediction horizon:** How far ahead are we predicting? (e.g., "30 days from the observation date")

**Analogy for students:** Problem framing is like choosing what game you are playing before picking a strategy. If you define the target incorrectly (e.g., predicting *why* customers churn instead of *who* will churn), no amount of modeling sophistication will produce a useful result.

### 5.2 Metric Selection Driven by Cost Structure

This is the central skill tested by the midterm. The key insight: **the right metric is the one that aligns with the business cost structure**, not the one that produces the highest number.

For Case 1 (churn): Missing a churner costs $50 (replacement acquisition), while sending an unnecessary retention offer costs $10. Since FN >> FP, the metric should emphasize recall (catching churners) more than precision.

For Case 2 (loans): A default costs $7,000, while a successful loan earns $500. Since the loss-to-profit ratio is 14:1, the metric should be heavily weighted toward avoiding false negatives (approving bad loans).

**What to emphasize on camera:** Walk through the expected-value calculation explicitly. Show that a model with 95% accuracy but 50% recall on defaulters would approve many bad loans, each costing $7,000. The "accurate" model is actually terrible from a business perspective.

### 5.3 Leakage Identification as a Professional Skill

Leakage questions require students to think about *when* data is generated relative to the prediction point. A feature that is computed *after* the event you are predicting is a leak.

**Examples for Case 1:**
- "account_status_after_churn" — obviously leaks the outcome
- "last_login = 30 days ago" — this might be the churn itself, not a predictor of churn
- "support_tickets_last_7_days" — borderline; could be a leading indicator or a consequence

**What to emphasize on camera:** Leakage is the most common and most dangerous mistake in production predictive analytics. It inflates validation scores dramatically, giving false confidence, and then the model fails completely in deployment when leaked features are not available at prediction time.

### 5.4 The Midterm Is a Design Exercise, Not a Coding Exercise

Students may be anxious because the midterm looks different from every other notebook. Reassure them: the setup cell imports familiar libraries, but the actual work is *writing* — structured markdown responses that demonstrate analytical thinking. The few code cells (setup, optional cost calculations) are scaffolding, not the assessment itself.

---

## 6. What Students Should Take Away

After watching your video and completing this notebook, students should be able to answer these questions confidently:

1. **"How do I translate a business problem into a prediction target, unit, and horizon?"** — By asking: what decision does the model inform? What entity are we predicting about? How far in advance do we need the prediction?

2. **"How do I choose a metric when business costs are asymmetric?"** — By computing the cost of each type of error (FP, FN, TP, TN) and selecting the metric that most closely reflects the total expected cost.

3. **"How do I spot leakage in a feature set?"** — By asking: would this feature be available at prediction time? Was it generated before or after the event I am predicting?

4. **"Why is accuracy misleading for imbalanced classes?"** — Because a model that always predicts the majority class achieves high accuracy without learning anything about the minority class.

5. **"What does a complete evaluation plan include?"** — A primary metric with business justification, a split/CV strategy, leakage prevention measures, a baseline model, and documented next steps.

---

## 7. Common Student Questions (Anticipate These)

**Q: "Why is the midterm mostly writing, not coding?"**
A: Because the hardest part of predictive analytics is not fitting a model — scikit-learn does that in one line. The hardest part is *deciding what to fit, how to evaluate it, and whether to trust the results*. That requires analytical reasoning, which is best demonstrated through structured writing.

**Q: "Is there one right answer for each case?"**
A: No. Multiple metric choices, split strategies, and model shortlists can be correct — as long as they are *justified*. A student who chooses recall with a clear cost-based argument scores as well as a student who chooses PR-AUC with a different but equally valid argument. The rubric rewards reasoning, not specific answers.

**Q: "Can I use Gemini to help with the cases?"**
A: You can use Gemini to explain concepts, generate code scaffolds for optional calculations, and debug errors. You cannot ask Gemini to solve an entire case for you. The key test: can you explain and defend every sentence in your response? If not, it is not your work.

**Q: "For Case 2, how do I enforce the 3% default rate constraint?"**
A: Set the classification threshold so that the predicted default rate on the validation set is at or below 3%. This means adjusting the probability cutoff: only approve loans where the model's predicted probability of default is below some threshold t, where t is chosen so the overall approval pool has a default rate under 3%.

**Q: "What models should I put in my shortlist?"**
A: At this point in the course, your toolkit includes: linear/logistic regression, Ridge, Lasso, and possibly a simple classifier like KNN. List 2-3 from this set and justify each choice based on the data characteristics (numeric vs. categorical features, linearity assumptions, interpretability requirements). Do not list models you have not learned yet.

**Q: "For the project milestone, what counts as a 'baseline'?"**
A: A baseline is the simplest reasonable model that establishes a performance floor. For regression, this might be predicting the mean. For classification, it might be always predicting the majority class, or a simple logistic regression with default hyperparameters. The point is to have a reference number that you can improve upon.

---

## 8. Connection to the Broader Course Arc

| Week | Notebooks | How NB10 Connects |
|---|---|---|
| **Week 1** | 01-05 | NB10 tests mastery of regression fundamentals: splitting, pipelines, metrics, feature engineering, regularization |
| **Week 2** | 06-10 | NB10 is the capstone: classification metrics, thresholding, CV, tuning, and now strategic design |
| **Week 3** | 11-15 | NB10's evaluation plans and project baselines become the scaffolding that students extend with tree-based models |
| **Week 4** | 16-20 | NB10's cost-based metric reasoning returns in NB16 (calibration), NB17 (fairness), and the final project |

**NB10 is the bridge between learning tools and using them wisely.** Everything before it teaches "how"; everything after it assumes students know "why." The midterm ensures that assumption is justified.

---

## 9. Suggested Video Structure

Below are two recording options. Each segment includes the **speaking prompt** (what to say), **timestamps**, and the **notebook cells** to show on screen.

All cell references use the format `Cell N` where N is the zero-indexed cell number in `10_midterm_casebook.ipynb`.

---

### Option A: Single Video (~12 minutes)

#### Segment 1 — Opening & Exam Overview `[0:00–2:00]`

> **Show:** Cell 0 (header) and Cell 1 (learning objectives) on screen.

**Say:**

"Welcome to the midterm. This notebook is different from everything you have done so far. There is almost no coding. Instead, you will be given two business cases — a streaming service with a churn problem, and a fintech company with a loan default problem — and you will design a complete predictive analytics strategy for each one. That means defining the prediction target, choosing the right metric based on business costs, identifying leakage risks, proposing a model shortlist, and designing an evaluation plan. This is the most important skill in predictive analytics: not fitting a model, but deciding *what* to fit and *how* to evaluate it."

> **Scroll** through the 5 learning objectives in Cell 1, briefly reading each one aloud.

> **Show:** Cell 2 (academic integrity instructions).

**Say:**

"A quick note on academic integrity. You may use your course notebooks, scikit-learn documentation, the course textbooks, and Gemini for code scaffolds and concept explanations. You may not communicate with other students during the exam, and you may not ask Gemini to solve entire case problems for you. The test is whether *you* can reason through the problem. If you cannot explain and defend every sentence in your response, it is not your work."

---

#### Segment 2 — Case 1: Customer Churn `[2:00–5:00]`

> **Show:** Cell 5 (Case 1 business context) on screen.

**Say:**

"Here is Case 1. StreamFlix is a video streaming service with a 5 percent monthly churn rate. Acquiring a new customer costs 50 dollars; retaining an existing customer with a discount offer costs 10 dollars. The data includes demographics, usage patterns, account history, and a binary churn label. Your job is to design a predictive modeling plan. Let me walk you through what good answers look like."

> **Scroll slowly** through the eight task items in Cell 5.

**Say:**

"Task 1 asks for the prediction target. Be specific: we are predicting a binary outcome — whether a customer will churn in the next 30 days or not. Task 2 asks for the prediction unit. One row is one customer observed at a specific point in time. Task 3 is the prediction horizon: 30 days, because that is the business decision window — the retention team needs lead time to send an offer."

"Task 4 — primary metric — is the most important question. Think about costs. A false negative means we miss a churner. That costs us 50 dollars because we have to acquire a replacement. A false positive means we send a 10-dollar retention offer to someone who was not going to churn anyway. Since missing a churner is five times more expensive, we want a metric that penalizes false negatives heavily. That points toward recall-oriented metrics — precision-recall AUC, or F-beta with beta greater than 1."

> **Show:** Cell 6 (PAUSE-AND-DO prompt) and Cell 7 (response template).

**Say:**

"Task 6 asks you to list three specific leakage risks. Here is what I mean by specific: do not write 'data leakage might happen.' Instead, write something like 'including account-status-after-churn as a feature would leak the outcome because it is recorded after the churn event.' Think about which features in the dataset would not be available at the time you need to make the prediction."

---

#### Segment 3 — Case 2: Loan Default `[5:00–8:00]`

> **Show:** Cell 8 (Case 2 business context) on screen.

**Say:**

"Case 2 is a fintech lending platform. Average loan is 10,000 dollars, profit per successful loan is 500 dollars, loss per default is 7,000 dollars. There is also a regulatory constraint: the default rate must stay below 3 percent. This case adds two layers of complexity beyond Case 1: first, the costs are much more asymmetric — the loss-to-profit ratio is 14 to 1. Second, the regulatory constraint is a hard boundary, not just an optimization target."

"For Task 1, think about expected value. If you approve a good applicant, you earn 500 dollars. If you approve a bad applicant, you lose 7,000 dollars. If you reject a good applicant, you lose the 500 dollars of potential profit. If you reject a bad applicant, you save 7,000 dollars. Write these numbers down explicitly — the rubric awards points for showing the calculation."

> **Show:** Cell 10 (Case 2 response template). Point to the expected value calculation section.

**Say:**

"Task 2 asks how to enforce the 3 percent default rate constraint. The answer is threshold calibration. After training your model, you adjust the classification threshold on the validation set so that the predicted default rate among approved loans stays below 3 percent. This means you will reject more borderline applicants, which reduces profit but keeps you in regulatory compliance."

"Task 3 — class imbalance. If only 1 to 2 percent of applicants default, accuracy is useless. A model that predicts 'no default' for everyone gets 98 percent accuracy and approves every bad loan. You need strategies: stratified cross-validation, class weights in the model, oversampling the minority class, or using PR-AUC instead of ROC-AUC as your metric."

---

#### Segment 4 — Project Milestone 2 `[8:00–10:00]`

> **Show:** Cell 13 (Project Milestone 2 requirements).

**Say:**

"The second part of this notebook is your Project Milestone 2 submission. This has three components. First, a baseline pipeline: your project dataset loaded, split into train, validation, and test, preprocessed with a proper pipeline, and fitted with at least one simple model. Second, a baseline report table comparing your baseline model to at least one improved model, with multiple metrics. Third, an evaluation plan documenting your primary metric with justification, your split or CV design, your leakage prevention measures, and your next steps."

"This milestone is your proof that you can apply everything from the first two weeks to your own data. If your pipeline runs without errors, your evaluation is on the validation set and not the test set, and your report table is clear and complete, you are in good shape."

> **Show:** Cell 14 (grading rubric). Briefly highlight the point distribution.

**Say:**

"The grading rubric is transparent. Case 1 is worth 35 points: 10 for problem framing, 10 for metric selection, 5 for leakage awareness, and 10 for modeling plan. Case 2 is also 35 points: 10 for expected value logic, 10 for constraint handling, 10 for imbalance handling, and 5 for monitoring plan. The project baseline is 30 points: 10 for code quality, 10 for evaluation rigor, and 10 for documentation."

---

#### Segment 5 — Tips for Success & Closing `[10:00–12:00]`

> **Show:** Cell 15 (common mistakes) and Cell 16 (how to earn full credit).

**Say:**

"Let me walk through the most common mistakes I see on midterms like this. First, generic answers. Do not write 'use cross-validation' without explaining why a time-based split might be more appropriate for churn data where temporal patterns matter. Second, ignoring business costs. If you choose accuracy as your metric for a 5 percent churn rate, you have not connected the metric to the business problem. Third, vague leakage risks. Name specific features and explain *why* they would leak."

> **Show:** Cell 16 (excellent vs. poor answer example). Read both examples aloud.

**Say:**

"Look at this comparison. The poor answer says 'I would use accuracy because it is a good metric.' The excellent answer says 'I would use PR-AUC as primary metric with a focus on recall at 20 percent precision,' then gives four numbered reasons connecting to the cost structure, the class imbalance, and the operating point on the precision-recall curve. That is the level of specificity and reasoning I am looking for."

"One final reminder: the setup cell at the top imports pandas, numpy, scikit-learn, and other familiar tools. You are welcome to write code to support your analysis — for example, computing expected costs or simulating a threshold sweep. But the core deliverable is your written analysis in the markdown cells. Good luck."

> **Show:** Cell 17 (submission instructions), then Cell 18 (Thank You).

---

### Option B: Three Shorter Videos (~4 min each)

---

#### Video 1: "Midterm Overview and Case 1 Walkthrough" (~5 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–0:45` | Show header + objectives | Cell 0, Cell 1 | "Welcome to the midterm. This notebook is different: almost no coding. You will design predictive analytics strategies for two business cases, then submit your project baseline. Here are the five learning objectives." (Read them briefly.) |
| `0:45–1:30` | Academic integrity + setup | Cell 2, Cell 3 | "Quick note on allowed resources: course notebooks, documentation, textbooks, and Gemini for scaffolding — but not for solving entire cases. The setup cell imports familiar libraries. You may write optional code, but the core work is written analysis." |
| `1:30–3:30` | Case 1 walkthrough | Cell 5 | "Case 1: StreamFlix churn prediction. Five percent monthly churn, 50 dollars to acquire a customer, 10 dollars for a retention offer. Think about what this cost structure means for metric selection. Missing a churner costs five times more than a false alarm. That points toward recall-focused metrics, not accuracy. For leakage, think about *when* each feature is generated relative to the churn event. A feature computed after the customer already churned is a leak." |
| `3:30–4:30` | PAUSE-AND-DO + response template | Cell 6, Cell 7 | "The response template has eight sections. Fill in every one. Be specific: name features, compute costs, justify every choice. The rubric rewards reasoning, not a particular 'right answer.' Multiple metric choices can earn full credit if they are properly justified. Pause and work through Case 1 now." |

---

#### Video 2: "Case 2: Loan Default and Regulatory Constraints" (~4 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–0:30` | Recap | — | "Welcome back. Case 1 tested churn prediction with asymmetric costs. Case 2 ramps up the stakes: a fintech lending platform where bad loans cost 7,000 dollars each, good loans earn 500, and regulators require a default rate below 3 percent." |
| `0:30–2:00` | Case 2 context + expected value | Cell 8 | "Start with the expected value calculation. If you approve a good applicant, you earn 500 dollars. Approve a bad applicant, you lose 7,000. The loss-to-profit ratio is 14 to 1. That means every default wipes out the profit from 14 good loans. Your metric must heavily penalize false negatives — approving bad loans. To enforce the 3 percent constraint, you calibrate your threshold on the validation set: only approve loans where predicted default probability is below a cutoff that keeps the pool's overall default rate under 3 percent." |
| `2:00–3:00` | Class imbalance discussion | Cell 8 | "With only 1-2 percent defaults, accuracy is a trap. A model that approves everyone gets 98 percent accuracy and loses thousands on every default. Strategies include stratified CV to preserve class balance in every fold, class weights in the loss function to penalize missed defaults more heavily, and PR-AUC as the primary metric because it focuses on the minority class unlike ROC-AUC which can look optimistic under extreme imbalance." |
| `3:00–4:00` | Mini-Case 3 + exercise prompt | Cell 11, Cell 12 | "Mini-Case 3 is optional bonus: a rare disease at 0.1 percent prevalence where false negatives are life-threatening. If you want to attempt it, think about which metric maximizes recall even at the cost of many false positives. Pause and work through Case 2 now, then attempt Case 3 if time allows." |

---

#### Video 3: "Project Milestone 2 and Exam Tips" (~4 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–1:00` | Project Milestone 2 | Cell 13 | "The second part of this notebook is Project Milestone 2. Three deliverables: a baseline pipeline that runs without errors, a comparison table with at least two models and multiple metrics, and a documented evaluation plan with your primary metric, split design, leakage prevention, and next steps. This is your proof that you can apply the first two weeks to your own data." |
| `1:00–2:00` | Grading rubric | Cell 14 | "The rubric: Case 1 is 35 points, Case 2 is 35 points, project baseline is 30 points. Within each case, points are distributed across problem framing, metric selection, leakage awareness, and modeling plan. For the project, code quality, evaluation rigor, and documentation each count for 10 points." |
| `2:00–3:00` | Common mistakes | Cell 15 | "Avoid these mistakes. Generic answers that could apply to any problem. Ignoring business costs when choosing metrics. Vague leakage risks like 'data leakage might happen' instead of naming specific features. Fitting on the full dataset before splitting. Looking at the test set during development. And not including a baseline for comparison." |
| `3:00–4:00` | Excellent vs. poor answers + closing | Cell 16, Cell 17 | "Look at the example comparison in the notebook. The poor answer says 'use accuracy because it is good.' The excellent answer gives four specific, cost-justified reasons for choosing PR-AUC with a recall focus. That level of specificity is what earns full credit. Submit your notebook and complete the quiz in Brightspace. Next time we start a brand new model family: decision trees. Good luck on the midterm." |

---

*This guide was created to support video lecture recording for MGMT 47400 — Predictive Analytics, Purdue University.*
