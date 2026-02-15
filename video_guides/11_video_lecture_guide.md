# Video Lecture Guide: Notebook 11 — Decision Trees - Interpretable Models with Sharp Edges

## At a Glance

This guide covers:

1. **Why NB11 exists** — It introduces the first non-linear model family (CART decision trees) and teaches students to see the bias-variance tradeoff concretely through tree depth control

2. **Why it must follow NB10** — NB10 (midterm) consolidates Weeks 1-2 knowledge. NB11 opens Week 3 with an entirely new paradigm: tree-based models that require no preprocessing, handle non-linearity natively, and are visually interpretable — but are prone to severe overfitting

3. **Why it must precede NB12** — NB12 (Random Forests) solves the single tree's high-variance problem through bagging. Without understanding *why* a single tree is unstable, students cannot appreciate *why* averaging many trees helps

4. **Why each library/tool is used:**
   - `DecisionTreeClassifier` / `DecisionTreeRegressor` — the CART algorithm for classification and regression
   - `plot_tree` — visual rendering of the learned tree structure (the key interpretability advantage)
   - `StratifiedKFold` + `cross_val_score` — depth selection via cross-validation instead of a single train/test split
   - `load_breast_cancer` — classification demo dataset (30 features, binary target)
   - `fetch_california_housing` — regression demo dataset (same data from Weeks 1-2, now modeled with trees)

5. **Common student questions** with answers (e.g., why no scaling?, why not just use the deepest tree?, do trees work for regression too?)

6. **Connection to the full course arc** — decision trees are the building blocks of Random Forests (NB12) and Gradient Boosting (NB13); understanding single-tree behavior is prerequisite to understanding ensembles

7. **Suggested video structure** — both single-video (~12 min) and three-video (~4 min each) formats with speaking prompts, timestamps, and notebook cell references

---

## Purpose of This Document

This guide is your **lecture companion** for recording the video(s) that accompany `11_decision_trees.ipynb`. It explains the *why* behind every design choice in the notebook — why trees are introduced now, why depth control is the central lesson, why we compare trees to linear models, and why the tree's weaknesses motivate the ensemble methods in the next two notebooks. Use it as speaking notes, not as a script: the goal is to internalize the reasoning so you can explain it naturally on camera.

---

## 1. Why This Notebook Exists

Weeks 1-2 taught students to build, evaluate, and tune linear and logistic regression models. These models assume a linear relationship between features and target. But many real-world relationships are non-linear: a customer's churn risk might spike above a certain usage threshold, not decline linearly with usage hours.

Decision trees are the natural next step because they are the simplest non-linear model. A tree makes predictions by asking a sequence of yes/no questions about feature values (e.g., "Is income > $50k?"), partitioning the data into increasingly homogeneous regions. No assumptions about linearity, no need for feature scaling, and the result is a diagram that anyone — including non-technical stakeholders — can read.

But trees have a critical flaw: without depth constraints, they memorize the training data perfectly and fail on new data. This notebook makes that flaw viscerally clear through a depth sweep that shows training accuracy climbing to 100% while test accuracy peaks and then degrades. Understanding this failure mode is essential preparation for NB12 and NB13, which fix it through ensemble methods.

---

## 2. Why It Must Come After Notebook 10

Notebook 10 is the midterm — a strategic design exercise that consolidates Weeks 1-2. NB11 opens a new chapter (Week 3: tree-based methods) that builds on but fundamentally differs from everything before.

| NB10 Contribution | How NB11 Uses It |
|---|---|
| **Strategic thinking (problem framing, metric selection)** | NB11 uses the same breast cancer and California Housing datasets. Students already know the prediction targets, appropriate metrics, and evaluation protocols. This frees cognitive bandwidth for the new concept: tree structure. |
| **Evaluation discipline (CV, no test peeking)** | NB11 uses `StratifiedKFold` cross-validation for depth selection. Students already understand why CV is superior to a single split, so the focus stays on interpreting the depth sweep results. |
| **Project baseline (Milestone 2)** | Students have a working pipeline from NB09/NB10. NB11 gives them a new model family to add to their comparison table in future notebooks. |

**In short:** NB10 clears the conceptual decks. Students enter NB11 with a solid foundation in evaluation methodology and can focus entirely on learning how trees work, how they fail, and how to control them.

If NB11 came before the midterm, students would not have consolidated their evaluation skills and would be trying to learn tree mechanics and evaluation discipline simultaneously — too much new material at once.

---

## 3. Why It Must Come Before Notebook 12

Notebook 12 introduces Random Forests — ensembles of many decision trees trained on bootstrap samples with random feature subsets. The entire motivation for Random Forests rests on three properties of single trees that NB11 demonstrates:

1. **High variance (instability).** NB11's depth sweep and CV results show that small changes in data produce large changes in tree structure and predictions. NB12 solves this by averaging many unstable trees.

2. **Overfitting without depth control.** NB11's unlimited-depth tree achieves 100% training accuracy but poor test accuracy. NB12 shows that forests can use deep, overfit trees *on purpose* because averaging cancels out the individual errors.

3. **Interpretability through visualization.** NB11's `plot_tree` visualization teaches students to read tree diagrams. NB12 leverages this when explaining how each tree in the forest makes its own set of splits, and why feature importance aggregated across hundreds of trees is more reliable than a single tree's splits.

The deliberate sequence is:

```
NB11: Single tree — how it works, how it fails (high variance, overfitting)
            |
NB12: Random Forest — fix variance by averaging many trees (bagging)
            |
NB13: Gradient Boosting — fix bias by sequentially correcting errors
```

Each notebook solves a problem identified in the previous one. Without NB11, students would not understand what problem NB12 is solving.

---

## 4. Why These Specific Libraries and Tools

### 4.1 `sklearn.tree.DecisionTreeClassifier` and `DecisionTreeRegressor`

**What they do:** Implement the CART (Classification and Regression Trees) algorithm. Classification trees split to maximize Gini impurity reduction; regression trees split to minimize mean squared error within each region.

**Why we use them:**

1. **Unified API.** Both classifiers and regressors follow scikit-learn's `.fit()` / `.predict()` / `.score()` interface, so students apply the same workflow they learned with linear models. The only conceptual difference is how splits are evaluated (Gini vs. MSE).

2. **Built-in complexity control.** Parameters like `max_depth`, `min_samples_split`, and `min_samples_leaf` give students direct control over the bias-variance tradeoff. This makes the tradeoff *tangible* rather than abstract.

3. **Foundation for ensembles.** `RandomForestClassifier` and `GradientBoostingClassifier` in NB12 and NB13 are built on top of these same tree estimators. Understanding the base learner is prerequisite to understanding the ensemble.

**What to emphasize on camera:** Trees are *greedy* — they pick the locally best split at each node without looking ahead. This means a globally better tree might exist that the algorithm never finds. Greedy splitting is fast but suboptimal, which is another reason ensembles help: different trees find different greedy paths, and averaging them approximates a better solution.

### 4.2 `sklearn.tree.plot_tree`

**What it does:** Renders the learned tree as a matplotlib figure, showing split rules, impurity values, sample counts, and class distributions at every node.

**Why we use it:**

1. **Interpretability is the killer feature.** The single greatest advantage of decision trees over neural networks, SVMs, or even logistic regression with many features is that you can *draw the decision logic*. A doctor, loan officer, or marketing manager can follow each path from root to leaf and understand exactly why a prediction was made.

2. **Debugging tool.** If a tree makes an obviously wrong prediction, you can trace the path and identify which split went wrong. This is impossible with black-box models.

3. **Depth control motivation.** Visualizing a depth-3 tree (readable) versus a depth-20 tree (incomprehensible) makes the case for limiting depth far more effectively than any abstract argument.

**What to emphasize on camera:** Walk through one complete root-to-leaf path. Read the split rule, explain the Gini impurity, count the samples, and show how the class distribution shifts at each level. This hands-on walkthrough is more valuable than any textbook definition of "information gain."

### 4.3 `sklearn.model_selection.StratifiedKFold` + `cross_val_score`

**What they do:** `StratifiedKFold` creates CV folds that preserve class proportions. `cross_val_score` evaluates a model across all folds and returns per-fold scores.

**Why we use them:**

- **Depth selection.** The most important hyperparameter for a decision tree is `max_depth`. Instead of picking a depth by intuition, we sweep depths 2 through 15 and select the one with the best average CV ROC-AUC. This is the same methodology students used for regularization strength in NB05, now applied to tree complexity.

- **Stability assessment.** The standard deviation of CV scores across folds measures how much the tree's performance varies with different data subsets. Trees typically have higher CV variance than logistic regression — a fact that motivates bagging in NB12.

**What to emphasize on camera:** The error-bar plot of CV ROC-AUC vs. depth is the key diagnostic. Point out that error bars get wider at higher depths, showing that deeper trees are more sensitive to which data lands in which fold.

### 4.4 `sklearn.datasets.load_breast_cancer`

**What it is:** Wisconsin Breast Cancer dataset — 569 samples, 30 numeric features (cell nucleus measurements), binary target (malignant vs. benign).

**Why we use it:**

- **Continuity.** The same dataset was used in NB06 (logistic regression) through NB09 (tuning). Students can directly compare: "My logistic regression got 0.98 ROC-AUC; my decision tree gets 0.97 — is the interpretability worth the small performance drop?"

- **Well-separated classes.** The breast cancer dataset is linearly separable enough that logistic regression performs well, but it also contains non-linear patterns that trees can exploit. This makes the head-to-head comparison informative rather than one-sided.

### 4.5 `sklearn.datasets.fetch_california_housing`

**What it is:** California Housing dataset — 20,640 samples, 8 numeric features, continuous target (median house value).

**Why we use it:**

- **Regression demonstration.** After classification trees, students need to see that the same algorithm works for regression (minimizing MSE instead of Gini). Using the California Housing data from Weeks 1-2 provides a familiar regression benchmark.

- **Larger dataset.** With 20,000+ samples, the overfitting pattern is even more dramatic: an unrestricted regression tree can have thousands of leaves, each memorizing a small neighborhood of the training data.

---

## 5. Key Concepts to Explain on Camera

### 5.1 The CART Algorithm: Recursive Binary Splitting

The core algorithm is simple: at each node, consider every feature and every possible threshold. For each candidate split, compute the impurity (Gini for classification, MSE for regression) of the two resulting child nodes. Pick the split that reduces impurity the most. Repeat until a stopping criterion is met.

**Analogy for students:** Imagine sorting a pile of mixed red and blue marbles. You pick up one marble at a time and ask, "Should I put this in the left bucket or the right bucket?" The best question is the one that makes each bucket as pure (all red or all blue) as possible. Decision trees do this automatically for every feature and every threshold.

### 5.2 Overfitting as Memorization

An unrestricted tree splits until every leaf contains a single sample (or until all samples in a leaf have the same target value). This means the tree has memorized every training example. On new data, these hyper-specific rules fail because they capture noise, not signal.

**Key visualization:** The depth sweep plot (Cell 11). Training accuracy climbs monotonically to 1.0 as depth increases. Test accuracy peaks at moderate depth (3-5) and then declines. The overfit gap (bar chart) grows with depth. This is the single most important plot in the notebook.

**What to emphasize on camera:** Point to the unlimited-depth bar. Say: "This tree has perfect training accuracy and the worst test accuracy in the table. It has memorized the training data — every quirk, every outlier, every noise pattern. On new data, those memorized patterns are wrong. This is why we *always* limit tree depth."

### 5.3 Depth as the Bias-Variance Knob

- **Shallow tree (depth 1-2):** High bias, low variance. The tree underfits — it captures only the single most important split and misses subtler patterns.
- **Moderate tree (depth 3-5):** Good balance. The tree captures the main patterns without memorizing noise.
- **Deep tree (depth 10+):** Low bias, high variance. The tree captures everything, including noise. Each fold of CV may produce a different-looking tree.

**What to emphasize on camera:** Depth for trees is analogous to regularization strength for linear models. In NB05, students learned that high C (weak regularization) overfits, while low C (strong regularization) underfits. Here, high depth overfits and low depth underfits. The principle is the same — controlling model complexity — but the mechanism is different.

### 5.4 Tree vs. Linear Model: When Each Wins

The head-to-head comparison (Cell 17) shows decision tree vs. logistic regression under the same CV. On the breast cancer dataset, they perform similarly because the classes are nearly linearly separable. But trees win when:

- Relationships are non-linear (e.g., risk spikes above a threshold, not linearly)
- Feature interactions matter (trees capture them automatically; logistic regression requires manual interaction terms)
- Interpretability to non-technical stakeholders is required

And logistic regression wins when:

- The relationship is approximately linear
- The dataset is small (trees are more unstable with fewer samples)
- Smooth probability estimates are needed (tree probabilities are step functions)

### 5.5 Regression Trees Follow the Same Logic

Section 5 (Cell 21-22) switches to the California Housing dataset to show that regression trees work identically: split to minimize MSE, predict the mean target value in each leaf. The depth sweep shows the same overfitting pattern. A regression tree at moderate depth matches or slightly beats Ridge regression (R-squared ~0.60-0.70), but an unrestricted tree dramatically overfits.

---

## 6. What Students Should Take Away

After watching your video and completing this notebook, students should be able to answer these questions confidently:

1. **"How does a decision tree make predictions?"** — It asks a sequence of yes/no questions about feature values, following a path from root to leaf. At the leaf, it predicts the majority class (classification) or the mean target (regression).

2. **"Why do unrestricted trees overfit?"** — Because they keep splitting until every leaf contains a single sample, memorizing the training data including noise. On new data, these memorized patterns fail.

3. **"How do I choose the right tree depth?"** — Use cross-validation. Sweep candidate depths, compute CV performance at each depth, and pick the depth with the highest mean CV score. Prefer simpler (shallower) when scores are tied.

4. **"When should I use a tree instead of logistic regression?"** — When interpretability is critical, relationships are non-linear, or you need a quick baseline. Avoid trees when the dataset is small, you need smooth boundaries, or you need to extrapolate beyond the training range.

5. **"Why are single trees called 'building blocks for ensembles'?"** — Because their high variance can be reduced by averaging many trees (Random Forests) or their residual errors can be corrected by sequential trees (Gradient Boosting). The next two notebooks build on exactly this idea.

---

## 7. Common Student Questions (Anticipate These)

**Q: "Why don't we need to scale features before fitting a tree?"**
A: Because trees make decisions based on thresholds (e.g., "Is income > $50k?"), not on distances or dot products. Multiplying a feature by 1,000 changes the threshold value but not the split's quality. This is a major practical advantage over logistic regression and SVMs, which are sensitive to feature scale.

**Q: "If unlimited trees get 100% training accuracy, why not just use them?"**
A: Because 100% training accuracy means the tree has memorized the data, including noise and outliers. On new data — which is what matters in production — those memorized patterns are wrong. The goal is not to fit the training data perfectly, but to generalize to unseen data.

**Q: "Can trees handle categorical features directly?"**
A: In theory, yes — CART can split on categorical values. In scikit-learn's implementation, you must encode categoricals as numbers first (e.g., one-hot or ordinal encoding). But trees handle encoded categoricals more naturally than linear models because they do not assume a linear relationship between the encoded values and the target.

**Q: "What is Gini impurity?"**
A: Gini impurity measures how "mixed" a node is. A node with all samples from one class has Gini = 0 (perfectly pure). A node with a 50/50 split has Gini = 0.5 (maximum impurity for two classes). The tree seeks splits that create the purest child nodes. You do not need to memorize the formula — the intuition is "how mixed is this group?"

**Q: "Why does the tree's first split use 'worst radius'?"**
A: Because the CART algorithm is greedy: it picks the split that most reduces impurity at each step. "Worst radius" (the largest cell nucleus radius in the image) is the single most discriminative feature for separating malignant from benign tumors. The tree discovers this automatically from the data.

**Q: "Are regression trees just classification trees with a different criterion?"**
A: Essentially, yes. The algorithm is identical: recursive binary splitting. The only difference is the split criterion (MSE instead of Gini) and the leaf prediction (mean target value instead of majority class). Everything else — depth control, overfitting behavior, visualization — works the same way.

---

## 8. Connection to the Broader Course Arc

| Week | Notebooks | How NB11 Connects |
|---|---|---|
| **Week 1** | 01-05 | NB11 reuses California Housing (regression trees) and the pipeline/evaluation patterns from NB02-NB03. Trees require no scaling, simplifying the pipeline. |
| **Week 2** | 06-10 | NB11 reuses Breast Cancer dataset and CV methodology from NB06-NB09. The tree-vs-logistic comparison directly extends the model comparison skills from NB09. |
| **Week 3** | 11-15 | **NB11 is the foundation.** NB12 fixes tree variance with bagging. NB13 fixes bias with boosting. NB14 formalizes the comparison protocol. NB15 interprets the champion model. |
| **Week 4** | 16-20 | Tree-based models (especially GBM from NB13) often become the champion in students' projects. NB16-NB17 interpret and audit these champions. |

**The tree is the atom of ensemble learning.** Every Random Forest, every Gradient Boosting model, every XGBoost pipeline is built from decision trees. NB11 teaches students to understand this fundamental building block before combining it in NB12 and NB13.

---

## 9. Suggested Video Structure

Below are two recording options. Each segment includes the **speaking prompt** (what to say), **timestamps**, and the **notebook cells** to show on screen.

All cell references use the format `Cell N` where N is the zero-indexed cell number in `11_decision_trees.ipynb`.

---

### Option A: Single Video (~12 minutes)

#### Segment 1 — Opening & Tree Intuition `[0:00–2:00]`

> **Show:** Cell 0 (header) and Cell 1 (learning objectives) on screen.

**Say:**

"Welcome to Week 3. Up until now, every model we have used — linear regression, Ridge, Lasso, logistic regression — assumes a linear relationship between features and target. Today we break that assumption. Decision trees are non-linear models that make predictions by asking a sequence of yes-or-no questions about your features: Is income above 50,000? Is age above 30? Each question splits the data into two groups, and the tree keeps splitting until it reaches a prediction. The result is a flowchart that anyone can read — your manager, your client, a regulator. That interpretability is the tree's greatest strength. But it comes with a sharp edge: without constraints, trees memorize the training data perfectly and fail on anything new."

> **Scroll** through the 5 learning objectives in Cell 1, briefly reading each one aloud.

> **Show:** Cell 4 (tree intuition markdown). Point to the algorithm steps and the example decision path.

**Say:**

"Here is the CART algorithm in four steps. Start with all data at the root. Find the best feature and threshold to split on — 'best' means the split that creates the most homogeneous child groups. Create two child nodes. Repeat recursively until a stopping criterion is met. The key hyperparameters are max-depth, min-samples-split, and min-samples-leaf. These are your complexity controls, and getting them right is the central skill of this notebook."

---

#### Segment 2 — Classification Tree: Fit and Visualize `[2:00–5:00]`

> **Show:** Cell 5 (classification tree explanation), then **run** Cell 6 (fit tree, print scores).

**Say:**

"We start with the breast cancer dataset — 569 samples, 30 features, binary target: malignant or benign. Same data from notebooks 6 through 9, so you already know the prediction task. We fit a decision tree with max-depth 3, meaning the tree can ask at most three sequential questions. Look at the scores: training accuracy around 97-98 percent, test accuracy close behind, overfit gap near zero. A three-question flowchart is already competitive with the logistic regression you built in notebook 6."

> **Run** Cell 8 (plot_tree visualization). **Zoom in** on the root node.

**Say:**

"Now the payoff. This is the entire model in one picture. Every box is a decision node. The root node asks: is worst radius less than or equal to 16.8? If yes, go left; if no, go right. Each box shows four things: the split rule, the Gini impurity, the number of samples, and the class distribution. Follow any path from root to leaf and you have a plain-English explanation: if worst radius is above 16.8 and worst concave points is above 0.14, predict malignant. Try doing that with a logistic regression that has 30 coefficients — it is possible, but far less intuitive."

> **Point to** a leaf node with Gini = 0. Explain: "Gini equals zero means this leaf is perfectly pure — every sample in it belongs to one class."

---

#### Segment 3 — The Overfitting Problem: Depth Sweep `[5:00–8:00]`

> **Show:** Cell 10 (overfitting explanation), then **run** Cell 11 (depth sweep + plots).

**Say:**

"Now the critical lesson. What happens if we remove the depth constraint? We sweep depths from 1 to unlimited and plot training accuracy versus test accuracy. Watch the left panel: training accuracy — the blue line — climbs steadily and hits 1.0 at unlimited depth. The tree has memorized every single training sample. But test accuracy — the orange line — peaks at moderate depth and then stagnates or declines. The right panel shows the overfit gap: the difference between training and test accuracy grows as depth increases."

> **Point to** the unlimited-depth bar in the gap chart.

**Say:**

"At unlimited depth, the tree has dozens of leaves with just a handful of samples each. Those tiny leaves have memorized idiosyncratic patterns — noise, outliers, one-off data points — that do not generalize. This is the single most important takeaway: never use an unrestricted tree in production. Always control depth, and always select it with cross-validation."

> **Run** Cell 14 (CV depth sweep with error bars).

**Say:**

"Here is the right way to choose depth. We sweep depths 2 through 15 using 5-fold stratified cross-validation with ROC-AUC as the metric. The error-bar plot shows mean plus or minus one standard deviation at each depth. The best depth — marked by the red line — is typically around 3 to 5. Notice that the error bars get wider at higher depths, confirming that deeper trees are more variable across folds. When two depths have similar means, always prefer the shallower one — it is more interpretable and more stable."

> **Mention:** "Pause the video here and complete Exercise 1 — run the depth sweep and choose your optimal depth." Point to Cell 13.

---

#### Segment 4 — Tree vs. Linear Model + Regression Trees `[8:00–10:30]`

> **Show:** Cell 16 (tree vs. linear explanation), then **run** Cell 17 (comparison table).

**Say:**

"Now a fair head-to-head comparison: our best decision tree versus logistic regression, both evaluated under the same 5-fold CV. On the breast cancer dataset, they perform similarly — both above 0.97 ROC-AUC — because the classes are nearly linearly separable. But compare the standard deviations: the tree's CV scores vary more across folds than logistic regression's. That higher variance is a fundamental property of trees, and it is exactly what Random Forests will fix in the next notebook."

> **Show:** Cell 21 (regression tree explanation), then **run** Cell 22 (regression depth sweep).

**Say:**

"Quick demonstration: regression trees work identically, except they minimize mean squared error instead of Gini impurity, and leaf predictions are averages instead of majority votes. We use the California Housing dataset from Week 1. Same overfitting story: at moderate depth, the regression tree matches Ridge regression at around 0.60-0.70 R-squared. At unlimited depth, training R-squared hits 1.0 but test R-squared drops dramatically. Same lesson, same solution: control depth."

> **Mention:** "Pause the video here and complete Exercise 2 — document three tree failure modes with evidence from the experiments." Point to Cell 19.

---

#### Segment 5 — When to Use Trees & Wrap-Up `[10:30–12:00]`

> **Show:** Cell 24 (strengths and weaknesses list).

**Say:**

"Let me summarize when to reach for a decision tree. Strengths: interpretable, handles non-linear relationships, no feature scaling needed, handles mixed data types, captures interactions automatically, and fast to train. Weaknesses: high variance — small data changes produce different trees — easy to overfit, poor extrapolation beyond the training range, biased toward features with many unique values, and step-function boundaries that are not smooth."

"Use a tree when interpretability is critical, when you need a quick baseline, or when you know relationships are highly non-linear. Avoid trees when your data is small, when you need smooth decision boundaries, or when you need to extrapolate."

> **Show:** Cell 25 (wrap-up).

**Say:**

"Three critical rules to remember. One: never use unrestricted trees in production. Two: always tune depth with cross-validation. Three: trees are building blocks for ensembles. That third point is the bridge to the next notebook. A single tree is unstable — change a few data points and you get a completely different tree. Random Forests solve this by training hundreds of trees on different bootstrap samples and averaging their predictions. The variance goes down, the accuracy goes up, and we keep most of the tree's advantages. See you in notebook 12."

> **Show:** Cell 26 (submission instructions), then Cell 27 (Thank You).

---

### Option B: Three Shorter Videos (~4 min each)

---

#### Video 1: "Tree Intuition and Classification Example" (~4 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–0:45` | Show header + objectives | Cell 0, Cell 1 | "Welcome to Week 3. We are leaving linear models behind and entering the world of tree-based methods. Decision trees make predictions by asking yes-or-no questions about your features — like a flowchart. Here are the five learning objectives." (Read them briefly.) |
| `0:45–1:15` | Setup | Cell 2, Cell 3 | "Standard setup with two new imports: DecisionTreeClassifier and plot_tree. plot_tree is the visualization function that makes trees interpretable. Random seed 474 for reproducibility." |
| `1:15–2:15` | Fit classification tree | Cell 5, Cell 6 | "Breast cancer dataset, 569 samples, 30 features. We fit a depth-3 tree. Training accuracy about 0.97, test accuracy close behind. A tree with only three levels of questions is already competitive with logistic regression." |
| `2:15–3:45` | Visualize tree | Cell 8, Cell 9 | "Here is the full model in one picture. Root node: worst radius less than 16.8? Each box shows the split rule, Gini impurity, sample count, and class distribution. Follow any path from root to leaf for a plain-English decision rule. This transparency is the tree's killer feature — you can hand this to a domain expert and they can audit every prediction." |
| `3:45–4:00` | Transition | — | "But there is a catch. What happens if we let the tree grow without limits? That is the next video." |

---

#### Video 2: "The Overfitting Problem and Depth Selection" (~5 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–0:30` | Recap | — | "We saw that a depth-3 tree performs well. But what if we remove all constraints and let the tree grow to any depth? Let us find out." |
| `0:30–2:00` | Depth sweep | Cell 10, Cell 11 | "The depth sweep tests depths from 1 to unlimited. Left panel: training accuracy hits 1.0 at unlimited depth — perfect memorization. Test accuracy peaks around depth 3-5 and then plateaus or declines. Right panel: the overfit gap grows dramatically. At unlimited depth, the tree has so many leaves that each one memorizes a handful of samples. Those memorized patterns are noise, not signal." |
| `2:00–3:30` | CV depth selection | Cell 13, Cell 14 | "The right way to choose depth: 5-fold stratified CV with ROC-AUC. The error-bar plot shows performance and variability at each depth. Best depth is around 3-5. Notice the error bars widen at higher depths — deeper trees are more sensitive to which data lands in which fold. When two depths have similar means, prefer the shallower one." |
| `3:30–4:15` | Exercise 1 prompt | Cell 13 | "Pause and complete Exercise 1: run the depth sweep with CV and choose your optimal depth. Document why you chose it." |
| `4:15–5:00` | Key takeaway | — | "Two rules: never use an unrestricted tree, and always select depth with cross-validation. In the next video, we compare trees to linear models and look at regression trees." |

---

#### Video 3: "Tree vs. Linear, Regression Trees, and Wrap-Up" (~4 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–1:00` | Tree vs. logistic | Cell 16, Cell 17 | "Head-to-head comparison: best decision tree versus logistic regression under the same 5-fold CV. On breast cancer, both achieve above 0.97 ROC-AUC. But the tree's standard deviation is higher — its performance varies more across folds. That higher variance is a fundamental tree property. Random Forests in the next notebook fix it by averaging many trees." |
| `1:00–2:00` | Regression trees | Cell 21, Cell 22 | "Regression trees work identically: minimize MSE instead of Gini, predict leaf averages instead of majority class. California Housing dataset: moderate-depth tree matches Ridge at R-squared 0.60-0.70. Unlimited depth: training R-squared 1.0, test R-squared collapses. Same lesson as classification." |
| `2:00–2:45` | Exercise 2 + failure modes | Cell 19, Cell 20 | "Pause and complete Exercise 2: document three failure modes — overfitting from the depth sweep, instability from CV variance, and poor extrapolation from the regression example. These three weaknesses motivate the ensemble methods in the next two notebooks." |
| `2:45–4:00` | When to use + closing | Cell 24, Cell 25, Cell 26 | "Use trees when interpretability matters, relationships are non-linear, or you need a quick baseline. Avoid trees on small datasets, when you need smooth boundaries, or when extrapolation matters. Three rules: never unrestricted, always CV-tune depth, and remember that trees are building blocks for ensembles. Next notebook: Random Forests — we fix the variance problem by averaging hundreds of trees. Submit and complete the quiz." |

---

*This guide was created to support video lecture recording for MGMT 47400 — Predictive Analytics, Purdue University.*
