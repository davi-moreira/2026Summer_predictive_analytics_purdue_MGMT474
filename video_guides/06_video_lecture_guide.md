# Video Lecture Guide: Notebook 06 — Logistic Regression - Probabilities, Decision Boundaries, and Pipelines

## At a Glance

This guide covers:

1. **Why NB06 exists** — It marks the transition from regression (Week 1) to classification (Week 2), teaching students how to predict categories instead of continuous values using logistic regression

2. **Why it must follow NB05** — NB05 introduces regularization (Ridge/Lasso) and the concept that penalty parameters control model complexity. NB06 applies the same idea (the C parameter) in a classification context, making the transition seamless

3. **Why it must precede NB07** — NB07 dives deep into classification metrics (precision, recall, ROC curves, PR curves). It assumes students already understand predicted probabilities, the default 0.5 threshold, and the confusion matrix. NB06 builds all three foundations

4. **Why each library/tool is used:**
   - `LogisticRegression` — maps linear combinations to probabilities via the sigmoid function
   - `DummyClassifier` — establishes the naive baseline that any real model must beat
   - `StandardScaler` — essential for logistic regression convergence and regularization fairness
   - `confusion_matrix` + `classification_report` — introduces the 2x2 error taxonomy that drives all of Week 2
   - `load_breast_cancer` — the course's standard classification dataset for Weeks 2-3

5. **Common student questions** with answers (e.g., why not just use linear regression for 0/1 targets?, why does the default threshold matter?)

6. **Connection to the full course arc** — how logistic regression becomes the baseline classifier reused in NB07 (metrics), NB08 (cross-validation), NB09 (tuning), and the course project

7. **Suggested video structure** — both single-video (~12 min) and three-video (~4-5 min each) formats with speaking prompts, timestamps, and notebook cell references

---

## Purpose of This Document

This guide is your **lecture companion** for recording the video(s) that accompany `06_logistic_pipelines.ipynb`. It explains the *why* behind every design choice in the notebook — why it exists, why it sits between notebooks 05 and 07, why it uses specific libraries, and why concepts like predicted probabilities and threshold sensitivity are introduced before the full metrics toolkit. Use it as speaking notes, not as a script: the goal is to internalize the reasoning so you can explain it naturally on camera.

---

## 1. Why This Notebook Exists

Notebook 05 ends with a complete regression toolkit: students can fit linear, Ridge, and Lasso models inside pipelines, tune alpha via cross-validation, and interpret coefficient shrinkage. But every dataset so far has had a continuous target variable (house prices). Students have never predicted a *category*.

Notebook 06 answers the question every student has at the start of Week 2: **"What changes when the target is a class label instead of a number?"** The answer is: almost everything downstream changes (the loss function, the output interpretation, the evaluation metrics), but the pipeline architecture stays the same. This continuity is deliberate — it shows students that the `Pipeline` + `ColumnTransformer` pattern they learned in Week 1 transfers directly to classification.

The notebook introduces logistic regression as the classification analogue of linear regression: same linear combination of features, but passed through the sigmoid function to produce a probability between 0 and 1. Students learn three critical ideas: (1) the model outputs *probabilities*, not just class labels; (2) a *threshold* converts probabilities to labels; and (3) the default threshold of 0.5 is not always the right choice. These ideas are the foundation for the entire classification arc in Week 2.

---

## 2. Why It Must Come After Notebook 05

Notebook 05 establishes three concepts that notebook 06 *consumes*:

| NB05 Concept | How NB06 Uses It |
|---|---|
| **Regularization (Ridge/Lasso alpha)** | NB06 introduces the `C` parameter in `LogisticRegression`, which is the *inverse* of alpha. Students who understand "higher alpha = stronger penalty" can immediately grasp "lower C = stronger penalty." The conceptual bridge is direct. |
| **Pipeline with StandardScaler** | NB05 shows that scaling is essential for regularized models because the penalty depends on coefficient magnitudes. NB06 reuses the exact same `Pipeline([StandardScaler(), model])` pattern, now with `LogisticRegression` instead of `Ridge`. Students do not need to learn a new workflow. |
| **Model comparison tables** | NB05 ends with a comparison table (OLS vs. Ridge vs. Lasso). NB06 builds a similar comparison (baseline vs. logistic regression at different C values), reinforcing the habit of structured model evaluation. |

**In short:** NB05 gives the regularization vocabulary and the pipeline habit; NB06 applies both to a new problem type (classification).

If you tried to teach classification *before* regularization, students would not understand why `LogisticRegression(C=0.01)` behaves differently from `LogisticRegression(C=100)`. If you skipped NB06 entirely and jumped to NB07's metrics, students would not understand what predicted probabilities are or why the 0.5 threshold is a design choice rather than a law of nature.

---

## 3. Why It Must Come Before Notebook 07

Notebook 07 introduces **classification metrics**: precision, recall, F1, ROC curves, PR curves, and cost-based threshold selection. Its learning objectives assume students already know:

- What `predict_proba()` returns and how it differs from `predict()`
- What a confusion matrix looks like and what the four cells (TP, TN, FP, FN) mean
- That the 0.5 threshold is adjustable and that changing it reshapes the prediction profile
- That accuracy can be misleading with imbalanced classes (the "most frequent" baseline)

NB06 builds each of these foundations explicitly. The threshold sweep in Section 5, the confusion matrix heatmap in Section 7, and the baseline comparison in Section 2 are all designed to prepare students for NB07's deeper treatment.

The deliberate sequence is:

```
NB05: How do we penalize complexity? (Regularization)
            |
NB06: How do we predict categories? (Logistic regression + probabilities + thresholds)
            |
NB07: How do we measure classification quality? (Precision, recall, ROC, PR, cost-based thresholds)
```

Each notebook adds exactly one conceptual layer. By NB07, the logistic regression pipeline is a solved problem — students can focus entirely on what the metrics mean.

---

## 4. Why These Specific Libraries and Tools

### 4.1 `sklearn.linear_model.LogisticRegression`

**What it does:** Fits a linear model that maps features to log-odds, then applies the sigmoid function to produce class probabilities. With the default `penalty='l2'`, it applies Ridge-style regularization controlled by the `C` parameter (inverse regularization strength).

**Why we use it:**

1. **Natural bridge from regression to classification.** Students already understand linear combinations from Weeks 1's regression models. Logistic regression adds one new concept — the sigmoid — and everything else (features, coefficients, pipeline) stays the same. This minimizes cognitive load at the transition point.

2. **Probability output.** Unlike decision trees or SVMs (in their default configuration), logistic regression naturally outputs calibrated probabilities via `predict_proba()`. This makes it the ideal vehicle for teaching the distinction between probabilities and class labels, which is foundational for threshold tuning in NB07.

3. **Built-in regularization.** The `C` parameter connects directly to NB05's regularization lesson. Students see the same bias-variance tradeoff, now in a classification context: small C underfits (too much penalty), large C overfits (too little penalty).

**What to emphasize on camera:** Logistic regression is not just a "beginner model" — it is the most widely deployed classification algorithm in industry (credit scoring, ad click prediction, medical screening). Its interpretability (coefficients have meaning) and probability calibration make it the default choice for many business applications.

### 4.2 `sklearn.dummy.DummyClassifier`

**What it does:** Produces predictions using a simple rule (most frequent class, stratified random, constant value) without examining the features at all.

**Why we use it:**

- **Establishes the accuracy floor.** With 63% benign samples in the breast cancer dataset, a model that always predicts "benign" achieves 63% accuracy. Any real classifier must beat this. Without the baseline, students might think 65% accuracy is good — when in fact it barely outperforms guessing.
- **Demonstrates the accuracy trap.** The DummyClassifier scores 63% while missing *every single malignant case*. This concrete example motivates the need for precision, recall, and the confusion matrix.

**What to emphasize on camera:** The baseline is not just a formality — it is a *sanity check*. If your trained model cannot beat the majority-class baseline, something is wrong with your features, your pipeline, or your data. In industry, failing to establish a baseline is one of the most common mistakes in ML projects.

### 4.3 `sklearn.preprocessing.StandardScaler`

**What it does:** Centers features to mean 0 and scales to unit variance, using statistics computed from the training data only.

**Why we use it:**

- **Logistic regression with L2 penalty is scale-sensitive.** The C parameter penalizes the sum of squared coefficients. If features are on different scales (e.g., one ranges 0-1 and another 0-10,000), the penalty unfairly targets the large-scale feature. Scaling removes this distortion.
- **Improves convergence.** The `lbfgs` solver (default in scikit-learn) converges faster when features are on similar scales. Without scaling, students may encounter convergence warnings with `max_iter=1000`.

**What to emphasize on camera:** This is the same scaler from NB02, used for the same reason. The pipeline template works for both regression and classification — the only thing that changes is the model at the end.

### 4.4 `sklearn.metrics.confusion_matrix` and `classification_report`

**What it does:** `confusion_matrix` returns the 2x2 matrix of TP, TN, FP, FN counts. `classification_report` computes precision, recall, F1, and support for each class.

**Why we use it:**

- **Introduces the error taxonomy.** The four cells of the confusion matrix are the atoms of classification evaluation. Every metric in NB07 (precision, recall, F1, ROC-AUC, PR-AUC) is arithmetic on these four numbers. Learning to read the matrix now makes NB07's formulas intuitive.
- **Connects errors to business costs.** In the breast cancer context, a false negative (missed malignant case) is catastrophically different from a false positive (unnecessary biopsy). The confusion matrix makes this asymmetry visible.

**What to emphasize on camera:** The confusion matrix is not a metric — it is a *fact sheet* about your model's behavior. Metrics are summaries of the matrix. NB07 will show how different summaries (precision vs. recall vs. F1) prioritize different cells, and how the right choice depends on the business problem.

### 4.5 `sklearn.datasets.load_breast_cancer`

**What it does:** Loads the Breast Cancer Wisconsin dataset: 569 samples, 30 numeric features, binary target (malignant=0, benign=1).

**Why we use it:**

- **Well-understood domain.** Students instantly grasp the stakes: missing a malignant tumor is life-threatening; an unnecessary biopsy is inconvenient but not dangerous. This asymmetry makes threshold tuning and cost-based evaluation viscerally meaningful.
- **Mild class imbalance (37/63).** The dataset is unbalanced enough to demonstrate the accuracy trap (63% baseline), but not so extreme that standard algorithms fail. This sets up NB07's deeper treatment of imbalance.
- **All numeric features.** No categorical encoding needed, so the pipeline stays simple and the focus remains on classification concepts rather than preprocessing mechanics.

**What to emphasize on camera:** We will use this dataset for NB06 through NB09. By the time students reach NB10 (midterm), they should know the breast cancer dataset inside and out — its features, its class balance, its easy separability — so they can focus on the business-case analysis rather than data wrangling.

---

## 5. Key Concepts to Explain on Camera

### 5.1 The Sigmoid Function — Linear Regression's Classification Upgrade

The core idea of logistic regression is deceptively simple: take the same linear combination from linear regression (z = b0 + b1*x1 + ... + bp*xp), but instead of outputting z directly, pass it through the sigmoid function: P(y=1) = 1/(1 + e^(-z)). This squashes any real number into the range [0, 1], giving us a valid probability.

**Analogy for students:** Think of the sigmoid as a "confidence meter." When the linear combination z is large and positive, the model is very confident the sample belongs to class 1 (probability near 1.0). When z is large and negative, it is very confident about class 0 (probability near 0.0). When z is near zero, the model is uncertain (probability near 0.5). The sigmoid smoothly translates "how strong is the evidence" into "how confident am I."

### 5.2 Probabilities vs. Classes — The Most Important Distinction in Classification

Students must understand that logistic regression produces *two* outputs: `predict()` gives hard class labels (0 or 1), and `predict_proba()` gives continuous probabilities. The hard labels are derived from the probabilities by applying a threshold (default 0.5).

**Why this matters:** Two predictions can have the same class label but very different probabilities. A patient with P(malignant) = 0.49 and one with P(malignant) = 0.01 both get predicted as "benign" at the default threshold, but a doctor would treat these cases very differently. Probabilities carry information that hard labels discard.

### 5.3 The 0.5 Threshold Is a Design Choice, Not a Law

The default threshold of 0.5 is mathematically convenient (it corresponds to z=0 on the sigmoid), but it assumes that both types of errors are equally costly. In medical diagnosis, missing a malignant tumor (false negative) is far worse than ordering an unnecessary biopsy (false positive). By lowering the threshold, you catch more true positives at the cost of more false positives — a tradeoff that should be driven by business costs, not mathematical convention.

**Analogy for students:** Imagine a fire alarm. Setting the threshold low means it goes off at the slightest smoke (high recall, many false alarms). Setting it high means it only triggers for major fires (high precision, but you might miss a small fire that grows). The right sensitivity depends on the consequences of each error.

### 5.4 Baselines Set the Floor

The DummyClassifier that always predicts "benign" achieves 63% accuracy. This number is not a model's achievement — it is a *property of the dataset*. Any classifier that scores near 63% has learned nothing. Students must internalize that accuracy is only meaningful relative to the baseline, and that with imbalanced classes, the baseline can be deceptively high.

---

## 6. What Students Should Take Away

After watching your video and completing this notebook, students should be able to answer these questions confidently:

1. **"What does the sigmoid function do?"** — It transforms a linear combination of features into a probability between 0 and 1, allowing us to use linear methods for classification.

2. **"What is the difference between `.predict()` and `.predict_proba()`?"** — `.predict()` returns hard class labels (0 or 1) by applying a threshold to the probabilities. `.predict_proba()` returns the raw probabilities, which carry more information and allow threshold tuning.

3. **"Why can a model with 63% accuracy be worthless?"** — Because on this dataset, always predicting the majority class achieves 63%. The model has learned nothing about the data; it is just reflecting the class distribution.

4. **"What does the C parameter control in logistic regression?"** — C is the inverse regularization strength. Small C means strong penalty (simpler model, coefficients pushed toward zero). Large C means weak penalty (more complex model, coefficients can be large). It is the classification analogue of alpha in Ridge/Lasso.

5. **"Why should I look at the confusion matrix instead of just accuracy?"** — Because the confusion matrix reveals *where* the model makes mistakes. Two models can have the same accuracy but very different error profiles (e.g., one misses all malignant cases, the other catches most of them). The right error profile depends on business costs.

---

## 7. Common Student Questions (Anticipate These)

**Q: "Why not just use linear regression to predict 0 and 1?"**
A: Linear regression outputs unbounded values — it can predict -0.3 or 1.7, which are not valid probabilities. The sigmoid function guarantees the output stays between 0 and 1. Also, the squared-error loss that linear regression minimizes is not well-suited for binary outcomes; logistic regression uses log loss (cross-entropy), which penalizes confident wrong predictions much more heavily.

**Q: "If the model already gets 97% accuracy, why do we need to worry about thresholds?"**
A: Because accuracy does not tell you *which* errors the model makes. In a medical context, the 3% of errors could all be missed malignant cases (false negatives), which is catastrophic. Adjusting the threshold lets you control the balance between false positives and false negatives to match the real-world cost structure.

**Q: "Is C the same as alpha in Ridge regression?"**
A: They control the same concept (regularization strength) but are inversely related. In Ridge, higher alpha = stronger penalty. In logistic regression, higher C = *weaker* penalty (C = 1/alpha). This is a scikit-learn convention: logistic regression follows the liblinear tradition where C is the inverse of the regularization parameter.

**Q: "Why does the notebook use `stratify=y` in `train_test_split`?"**
A: Stratification ensures that each split (train, validation, test) has the same proportion of malignant and benign samples as the full dataset (~37%/63%). Without stratification, a small split might end up with 50% malignant or 25% malignant purely by chance, distorting both training and evaluation.

**Q: "Why start with breast cancer? It seems like a morbid choice."**
A: The dataset is chosen precisely because the stakes are immediately obvious. When students see "missed malignant tumor" in the confusion matrix, they intuitively understand why accuracy alone is insufficient and why threshold tuning matters. The emotional weight of the domain reinforces the technical lesson.

---

## 8. Connection to the Broader Course Arc

| Week | Notebooks | What NB06's Concepts Enable |
|---|---|---|
| **Week 2** | 07-09 | NB06 provides the logistic regression pipeline reused in NB07 (metrics deep dive), NB08 (cross-validation comparison), NB09 (GridSearchCV tuning). The probability output enables ROC/PR curves in NB07 and the C parameter becomes a grid search target in NB09 |
| **Week 2** | 10 | NB10 (midterm) asks students to design classification strategies for business cases. NB06's lessons on baselines, probabilities, thresholds, and confusion matrices are directly tested |
| **Week 3** | 11-15 | Tree-based classifiers (Decision Tree, Random Forest, Gradient Boosting) plug into the same pipeline template from NB06. The evaluation patterns (confusion matrix, threshold sweep) transfer directly |
| **Week 4** | 16-20 | The logistic regression baseline from NB06 serves as the "simple model" benchmark in the final project. Students compare their advanced models against it |

**NB06 is the gateway to the classification half of the course.** Every classification notebook from NB07 onward assumes students can build a logistic regression pipeline, generate probabilities, and read a confusion matrix. NB06 is where all three skills are established for the first time.

---

## 9. Suggested Video Structure

Below are two recording options. Each segment includes the **speaking prompt** (what to say), **timestamps**, and the **notebook cells** to show on screen.

All cell references use the format `Cell N` where N is the zero-indexed cell number in `06_logistic_pipelines.ipynb`.

---

### Option A: Single Video (~12 minutes)

#### Segment 1 — Opening & Motivation `[0:00–1:30]`

> **Show:** Cell 0 (header) and Cell 1 (learning objectives) on screen.

**Say:**

"Welcome to Week 2. For the past five notebooks, everything we predicted was a number — house prices, continuous values. Starting today, we are predicting categories. Is this tumor malignant or benign? Will this customer churn or stay? Will this loan default or not? Classification is where predictive analytics meets most business decisions, because most business actions are binary: approve or deny, intervene or wait, flag or pass. The tool we start with is logistic regression — and you will see that it is built on the same linear model you already know, with one crucial addition: the sigmoid function that turns a number into a probability. Let me walk through the five learning objectives."

> **Action:** Scroll through the 5 learning objectives in Cell 1, briefly reading each one aloud.

---

#### Segment 2 — Setup & Dataset `[1:30–3:30]`

> **Show:** Cell 2 (setup code), then **run** it. Point to `Setup complete!` output.

**Say:**

"Our setup looks familiar — pandas, numpy, matplotlib, seaborn, scikit-learn. The new imports are classification-specific: `LogisticRegression` for the model, `DummyClassifier` for baselines, and `accuracy_score`, `log_loss`, `confusion_matrix`, and `classification_report` for evaluation. We keep `RANDOM_SEED = 474` for reproducibility."

> **Show:** Cell 4 (dataset explanation), then **run** Cell 5 (load + split).

**Say:**

"We switch from California Housing to the Breast Cancer Wisconsin dataset — 569 samples, 30 numeric features measuring properties of cell nuclei, and a binary target: malignant or benign. Notice the class distribution: about 37% malignant, 63% benign. This matters — a model that always predicts 'benign' would already get 63% accuracy without learning anything. Our splits are stratified to preserve this ratio: 341 training, 114 validation, 114 test. Test set is locked as always."

---

#### Segment 3 — Baseline & Why Accuracy Is Dangerous `[3:30–5:00]`

> **Show:** Cell 7 (baseline explanation), then **run** Cell 8 (DummyClassifier).

**Say:**

"Before we build any real model, we establish a baseline. The DummyClassifier with `strategy='most_frequent'` always predicts the majority class — benign. Its accuracy is about 63%. That sounds okay until you realize it misses every single malignant case. Every cancer patient would be told they are fine. This is why accuracy alone is a dangerous metric for imbalanced datasets. The 63% is our floor — any real model must beat it, and more importantly, it must beat it on the metrics that matter, not just accuracy. We will build the full metrics toolkit in the next notebook."

---

#### Segment 4 — Sigmoid & Logistic Regression Pipeline `[5:00–8:00]`

> **Show:** Cell 10 (sigmoid math explanation), then **run** Cell 11 (sigmoid visualization).

**Say:**

"Here is the core idea. Logistic regression takes the same linear combination you know from linear regression — beta-zero plus beta-one times x-one, and so on — and passes it through the sigmoid function. The sigmoid squashes any real number into the range zero to one, giving us a valid probability. When the linear combination is large and positive, the probability is near 1 — the model is confident about class 1. When it is large and negative, the probability is near 0 — confident about class 0. Near zero, the model is uncertain. The red dashed line at 0.5 is the default decision threshold."

> **Show:** Cell 13 (pipeline explanation), then **run** Cell 14 (fit pipeline + scores).

**Say:**

"Now we build the pipeline — same pattern as Week 1. StandardScaler first, then LogisticRegression with our random seed and max_iter of 1000 to ensure convergence. One call to fit, and we get both predictions and probabilities. The results: train accuracy about 97%, validation accuracy about 97% — a huge improvement over the 63% baseline. Log loss is around 0.08 to 0.12, which means the probability estimates are well-calibrated. But remember: accuracy alone does not tell us where the errors are. Two models with 97% accuracy could have very different error profiles."

> **Show:** Cell 16 (Exercise 1 prompt). Then **run** Cell 17 (sample predictions table).

**Say:**

"Look at this table. For each validation sample, we see the true label, the predicted class, and the probability for each class. Notice that probabilities always sum to 1.0. Some predictions are highly confident — probability 0.99 — while others are close to 0.5, meaning the model is uncertain. The predicted class is just the argmax of the two probabilities. Pause the video here and complete Exercise 1 — examine these probabilities and explain the difference between predict and predict_proba."

---

#### Segment 5 — Thresholds & Regularization `[8:00–10:30]`

> **Show:** Cell 20 (thresholding explanation), then **run** Cell 21 (threshold sweep).

**Say:**

"Here is a critical insight. We test four thresholds: 0.3, 0.5, 0.7, and 0.9. As the threshold increases, the model becomes more conservative about predicting benign — the number of positive predictions drops. At threshold 0.9, the model only predicts benign when it is very confident, which means more malignant predictions but also more false alarms. At threshold 0.3, almost everything is predicted benign, which catches more benign cases but misses some malignant ones. The default 0.5 is not magic — it is arbitrary. In the next notebook, we will learn how to choose the threshold based on business costs."

> **Show:** Cell 25 (regularization explanation), then **run** Cell 26 (C parameter sweep).

**Say:**

"Logistic regression in scikit-learn uses L2 regularization by default, controlled by the C parameter. Remember Ridge regression from last week? C is the inverse of alpha — lower C means stronger regularization. We sweep C from 0.01 to 100. At very low C, the model is too constrained and accuracy dips slightly. At moderate C values, accuracy peaks and plateaus. The train-val gap stays small across all values, which tells us this dataset does not suffer from severe overfitting. The best C is identified automatically."

> **Mention:** "Pause the video here and complete Exercise 2 — change thresholds and observe how the prediction distribution shifts." Point to Cell 23 (PAUSE-AND-DO 2).

---

#### Segment 6 — Confusion Matrix, Wrap-Up & Closing `[10:30–12:00]`

> **Show:** Cell 28 (confusion matrix explanation), then **run** Cell 29 (confusion matrix heatmap).

**Say:**

"The confusion matrix is the foundation of everything in the next notebook. Four cells: true negatives, false positives, false negatives, true positives. In our medical context, false positives mean a benign patient gets an unnecessary biopsy — stressful but not dangerous. False negatives mean a malignant patient is told they are fine — potentially fatal. This asymmetry is why accuracy is not enough. You need to know which cell contains your errors and whether that pattern is acceptable for your business problem."

> **Show:** Cell 31 (Wrap-Up).

**Say:**

"Let me leave you with three critical rules from today. First: always look at probabilities, not just classes. The probability 0.99 and the probability 0.51 both produce the same class label, but they represent very different levels of confidence. Second: accuracy is not enough — the confusion matrix reveals where your model fails. Third: the 0.5 threshold should be tuned to your business costs, not accepted as a default. In the next notebook, we go deep on classification metrics — precision, recall, F1, ROC curves, and precision-recall curves. We will learn how to choose thresholds systematically and how to evaluate models when classes are imbalanced. See you there."

> **Show:** Cell 32 (Submission Instructions) briefly, then Cell 33 (Thank You).

---

### Option B: Three Shorter Videos (~4-5 minutes each)

---

#### Video 1: "From Regression to Classification — Baselines and Sigmoid" (~5 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–0:45` | Show header + objectives | Cell 0, Cell 1 | "Welcome to Week 2. We are switching from regression to classification. Instead of predicting a number like house price, we are now predicting a category — malignant or benign, churn or stay, default or repay. Here are our five learning objectives for today." (Read them briefly.) |
| `0:45–1:30` | Run setup | Cell 2, Cell 3 | "Standard setup with classification-specific imports: LogisticRegression, DummyClassifier, accuracy_score, log_loss, confusion_matrix, classification_report. We also import load_breast_cancer — this dataset will be our standard classification benchmark for the next four notebooks. Random seed 474." |
| `1:30–2:30` | Load + split | Cell 4, Cell 5, Cell 6 | "The Breast Cancer Wisconsin dataset: 569 samples, 30 numeric features describing cell nuclei, binary target — malignant is 0, benign is 1. Class balance is 37% malignant, 63% benign. We split 60-20-20 with stratification to preserve this ratio in every partition. Test set stays locked." |
| `2:30–3:30` | Baseline | Cell 7, Cell 8, Cell 9 | "DummyClassifier with most_frequent strategy always predicts benign. Accuracy: 63%. Sounds reasonable until you realize it misses every malignant case. This is the accuracy trap — when classes are unbalanced, a naive strategy can score respectably while being clinically useless. Our floor is 63%. Any real model must beat it, and must beat it on metrics that capture error quality, not just error quantity." |
| `3:30–5:00` | Sigmoid visualization | Cell 10, Cell 11, Cell 12 | "Logistic regression takes the linear combination from linear regression and passes it through the sigmoid function — this S-shaped curve that maps any real number to a probability between 0 and 1. When z is large positive, probability near 1. Large negative, near 0. Near zero, the model is uncertain at 0.5. The red line marks the default threshold. Everything above 0.5 gets predicted as class 1, everything below as class 0. But as we will see, this threshold is adjustable." |

---

#### Video 2: "Pipeline, Probabilities, and Thresholds" (~5 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–0:30` | Recap | — | "Welcome back. We have our breast cancer dataset split and our baseline established at 63% accuracy. Now we build a real logistic regression pipeline and explore its probability outputs." |
| `0:30–1:30` | Build and fit pipeline | Cell 13, Cell 14, Cell 15 | "Same pipeline pattern as Week 1: StandardScaler followed by LogisticRegression, all inside a Pipeline object. Scaling matters because logistic regression's L2 penalty depends on coefficient magnitudes. We fit on training data and get both hard predictions via predict and probabilities via predict_proba. Train accuracy about 97%, validation about 97%. Log loss around 0.08 to 0.12, meaning the probabilities are well-calibrated. That is a 34-point jump over our 63% baseline." |
| `1:30–2:30` | Examine predictions table | Cell 16, Cell 17, Cell 18 | "This table is the key insight. For ten validation samples, we see the true label, the predicted class, and the probability for each class. The probabilities always sum to 1.0. Some predictions are confident — 0.99 for class 1 — while others are near 0.5, uncertain. The predicted class is just the argmax. But here is the crucial point: a prediction of 0.51 and a prediction of 0.99 both map to class 1, yet they represent very different confidence levels. Probabilities carry information that hard labels discard." |
| `2:30–4:00` | Threshold sweep | Cell 20, Cell 21, Cell 22 | "Now we change the threshold from 0.5 and watch what happens. At threshold 0.3, almost everything is predicted positive — high recall, many false positives. At 0.7, the model becomes conservative — fewer positive predictions, more false negatives. At 0.9, it only predicts positive when it is nearly certain. The accuracy changes at each threshold, and the prediction distribution shifts dramatically. The default 0.5 is not optimal — it assumes both error types are equally costly. In the next notebook, we will learn how to pick thresholds using business costs." |
| `4:00–5:00` | Exercise prompts | Cell 23, Cell 24 | "Pause the video and complete Exercise 2. Change the thresholds and observe how the number of positive and negative predictions shifts. Think about when you would want a threshold other than 0.5. Write your observations in the markdown cells. Come back when you are done." |

---

#### Video 3: "Regularization, Confusion Matrix, and Wrap-Up" (~4 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–1:00` | Regularization (C parameter) | Cell 25, Cell 26, Cell 27 | "Logistic regression uses L2 regularization by default, controlled by C. Remember Ridge from last week? C is the inverse of alpha. Lower C equals stronger regularization. We test C from 0.01 to 100. At very low C, accuracy dips slightly — the model is over-constrained. At moderate C, performance peaks. The train-val gap stays small, so overfitting is not a major concern here. The best C is printed at the bottom." |
| `1:00–2:30` | Confusion matrix | Cell 28, Cell 29, Cell 30 | "The confusion matrix is a 2-by-2 table of outcomes: true negatives, false positives, false negatives, true positives. In our breast cancer context, a false positive means a benign patient gets an unnecessary biopsy — stressful but survivable. A false negative means a malignant patient is sent home undiagnosed — potentially fatal. This asymmetry is the whole reason accuracy is not enough. The confusion matrix shows you exactly where your model makes mistakes and lets you decide if that error profile is acceptable." |
| `2:30–4:00` | Wrap-up + closing | Cell 31, Cell 32, Cell 33 | "Three rules from today. One: always examine probabilities, not just class labels. Two: the confusion matrix reveals error quality, not just error quantity. Three: the 0.5 threshold is a default, not a recommendation — tune it to match your business costs. This logistic regression pipeline is the starting point for everything in Week 2. In the next notebook, we go deep on classification metrics: precision, recall, F1, ROC curves, and precision-recall curves. We will learn to evaluate models properly and choose thresholds based on what errors actually cost. Submit your notebook to Brightspace and complete the quiz. See you there." |

---

*This guide was created to support video lecture recording for MGMT 47400 — Predictive Analytics, Purdue University.*
