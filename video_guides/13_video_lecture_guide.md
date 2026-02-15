# Video Lecture Guide: Notebook 13 — Gradient Boosting - Performance with Discipline (and Leakage Avoidance)

## At a Glance

This guide covers:

1. **Why NB13 exists** — It completes the tree-based ensemble trilogy by teaching the most powerful (and most tuning-sensitive) ensemble method: Gradient Boosting, where trees are trained sequentially to correct each other's errors

2. **Why it must follow NB12** — NB12 established the parallel ensemble baseline (Random Forests). NB13's sequential approach is a conceptual contrast: bagging reduces variance, boosting reduces bias. Without NB12, students cannot appreciate what boosting does differently or why it often wins

3. **Why it must precede NB14** — NB14 formalizes the model selection protocol. By the time students reach NB14, they need a full roster of candidates: logistic regression, decision tree, Random Forest, and Gradient Boosting. NB13 completes that roster

4. **Why each library/tool is used:**
   - `GradientBoostingClassifier` — sklearn's sequential ensemble that fits trees to residuals
   - `RandomizedSearchCV` — efficient hyperparameter search for the 5+ knobs that GBM exposes
   - `staged_predict_proba` — iteration-by-iteration performance tracking for overfitting detection
   - `scipy.stats.randint` / `uniform` — define continuous and integer distributions for random search

5. **Common student questions** with answers (e.g., why keep trees shallow?, what is the learning rate?, how is this different from Random Forests?)

6. **Connection to the full course arc** — Gradient Boosting is typically the strongest model on tabular data and often becomes the champion in students' final projects

7. **Suggested video structure** — both single-video (~12 min) and three-video (~4-5 min each) formats with speaking prompts, timestamps, and notebook cell references

---

## Purpose of This Document

This guide is your **lecture companion** for recording the video(s) that accompany `13_gradient_boosting.ipynb`. It explains the *why* behind every design choice in the notebook — why boosting trains trees sequentially, why the learning rate is the most important hyperparameter, why trees must be kept shallow, and why overfitting monitoring is non-negotiable. Use it as speaking notes, not as a script: the goal is to internalize the reasoning so you can explain it naturally on camera.

---

## 1. Why This Notebook Exists

NB11 introduced decision trees and their weaknesses. NB12 showed how Random Forests fix the variance problem by averaging many independent trees. But what if the problem is not variance but *bias* — the model is not flexible enough to capture the underlying signal?

Gradient Boosting addresses this by building trees *sequentially*, where each new tree specifically corrects the mistakes (residuals) of the current ensemble. Instead of averaging independent models, boosting *accumulates* corrections. The result is a model that progressively focuses on the hardest-to-predict samples, often achieving the highest accuracy on structured tabular data.

But this power comes with a cost: Gradient Boosting is more sensitive to hyperparameters than Random Forests. The learning rate, number of trees, tree depth, and subsampling fraction all interact in non-obvious ways. Training too many iterations without monitoring leads to overfitting — unlike Random Forests, which are relatively immune to this failure mode. This notebook teaches students to harness boosting's power while maintaining the discipline to avoid its pitfalls.

---

## 2. Why It Must Come After Notebook 12

NB13 requires three concepts that NB12 establishes:

| NB12 Concept | How NB13 Uses It |
|---|---|
| **Parallel ensemble (bagging).** NB12 trained 100 trees independently on bootstrap samples and averaged them. | NB13 introduces the *contrast*: sequential ensemble where each tree depends on all previous trees. The opening section (Cell 4) explicitly compares "Bagging = parallel, reduces variance" vs. "Boosting = sequential, reduces bias." This comparison only works if students already understand bagging. |
| **Random Forest as performance baseline.** NB12 established the forest's CV ROC-AUC on the breast cancer dataset (~0.99). | NB13 compares GBM directly to RF under the same CV (Cell 8). The question "Does boosting beat the forest?" is the central empirical test. Without NB12's baseline, this comparison has no anchor. |
| **Feature importance and OOB scores.** NB12 taught permutation importance and the concept of "free validation." | NB13 uses the same permutation importance method (via NB14) and introduces a related concept: staged predictions for overfitting monitoring. The idea that "the training procedure itself can provide validation signals" connects OOB (NB12) to early stopping (NB13). |

**In short:** NB12 provides the baseline, the vocabulary (bagging vs. boosting), and the evaluation framework. NB13 builds on all three to introduce a more powerful but more complex method.

If NB13 came before NB12, students would have no baseline for comparison, no understanding of what "parallel ensemble" means to contrast with "sequential ensemble," and no appreciation for why the extra tuning complexity of GBM is worth the effort.

---

## 3. Why It Must Come Before Notebook 14

NB14 introduces the **model selection protocol** — a formalized workflow for comparing all candidate models under the same CV, selecting a champion, writing a selection memo, and evaluating on the test set exactly once. NB14 needs a complete roster of candidates:

1. Logistic Regression (from NB06)
2. Decision Tree (from NB11)
3. Random Forest (from NB12)
4. **Gradient Boosting (from NB13)**

Without NB13, the candidate pool would be incomplete. More importantly, Gradient Boosting is often the *strongest* candidate on tabular data, so omitting it would deprive students of their most likely champion.

NB14's selection protocol also assumes students understand the tradeoffs between model families:

- Logistic Regression: fast, interpretable, but linear
- Decision Tree: interpretable, but high variance
- Random Forest: stable, minimal tuning, but slower and less interpretable
- **Gradient Boosting: highest accuracy, but requires careful tuning and overfitting monitoring**

These tradeoffs are taught across NB11-NB13. NB14 formalizes them into a decision framework.

The deliberate sequence is:

```
NB11: Decision Tree — building block (interpretable, unstable)
            |
NB12: Random Forest — parallel ensemble (stable, minimal tuning)
            |
NB13: Gradient Boosting — sequential ensemble (powerful, tuning-sensitive)
            |
NB14: Model Selection — pick the champion from the full roster
```

---

## 4. Why These Specific Libraries and Tools

### 4.1 `sklearn.ensemble.GradientBoostingClassifier`

**What it does:** Builds an additive model of shallow decision trees trained sequentially. At each iteration, a new tree is fitted to the negative gradient of the loss function (pseudo-residuals) and added to the ensemble with a small weight (the learning rate).

**Why we use it:**

1. **Scikit-learn consistency.** Students have been using scikit-learn's `.fit()` / `.predict()` / `.score()` API since NB01. GBM follows the same interface, so the only new concepts are the hyperparameters and the sequential training logic, not the code structure.

2. **Staged predictions.** The `.staged_predict_proba()` method returns predictions at every iteration from 1 to `n_estimators`, enabling the overfitting visualization in Cell 24. This feature is not available in XGBoost or LightGBM's native APIs without extra configuration.

3. **Deterministic given a seed.** Unlike some random forest implementations where parallel execution introduces non-determinism, scikit-learn's GBM is fully deterministic given `random_state=474`. This ensures reproducibility for student exercises.

**What to emphasize on camera:** GBM is conceptually simple: start with a constant prediction (e.g., the mean), compute errors, fit a tree to those errors, add it to the model with a small weight, and repeat. The power comes from the sequential error correction — each tree specializes in the samples that previous trees got wrong.

### 4.2 `sklearn.model_selection.RandomizedSearchCV`

**What it does:** Samples random combinations of hyperparameters from specified distributions, evaluates each combination with cross-validation, and returns the best.

**Why we use it here (not GridSearchCV):**

- GBM has at least 5 hyperparameters that matter: `n_estimators`, `learning_rate`, `max_depth`, `min_samples_split`, and `subsample`. A full grid search over 5 parameters with even 3 values each would require 3^5 = 243 combinations, each evaluated across 5 CV folds — over 1,200 model fits. On larger datasets, this is prohibitively slow.
- `RandomizedSearchCV` with 20 iterations provides a good exploration of the space with only 100 model fits (20 combinations x 5 folds).
- The search spaces use continuous distributions (`uniform` for learning rate and subsample) and integer distributions (`randint` for n_estimators, max_depth, min_samples_split), giving finer granularity than a discrete grid.

**What to emphasize on camera:** The parameter distributions in Cell 19 are *constrained*: learning rate between 0.01 and 0.20, max_depth between 3 and 7, subsample between 0.7 and 1.0. These ranges encode domain knowledge — we know from the learning-rate sweep earlier that values outside these ranges are unlikely to be optimal. Constraining the search space is a professional skill.

### 4.3 `staged_predict_proba` for Overfitting Detection

**What it does:** Returns a generator that yields the ensemble's predictions after each boosting iteration (after tree 1, after trees 1+2, after trees 1+2+3, etc.).

**Why we use it:**

1. **Overfitting visualization.** By plotting training and validation ROC-AUC at every iteration, students see exactly where the validation curve peaks and starts to decline. This is the most important diagnostic for boosting.

2. **Motivation for early stopping.** The iteration where the validation curve peaks is the optimal number of trees. Training beyond that point hurts generalization. In production, you would use `n_iter_no_change` (sklearn) or `early_stopping_rounds` (XGBoost) to halt training automatically.

3. **Contrast with Random Forests.** In NB12, more trees was always better (or at worst neutral). In NB13, more trees can actively hurt. This contrast highlights the fundamental difference between bagging (averaging independent trees) and boosting (accumulating dependent corrections).

**What to emphasize on camera:** The training curve (blue) climbs to 1.0 and stays there. The validation curve (orange) peaks and then either plateaus or gently declines. The gap between them is the overfitting signal. Say: "In Random Forests, adding more trees is free insurance. In Gradient Boosting, adding more trees past the validation peak is like studying for an exam by memorizing the practice test — you get perfect scores on practice but worse scores on the real exam."

### 4.4 `scipy.stats.randint` and `uniform`

**What they do:** Define integer and continuous probability distributions for `RandomizedSearchCV` to sample from.

**Why we use them:**

- `randint(100, 300)` generates random integers between 100 and 299, giving uniform coverage of the n_estimators range.
- `uniform(0.01, 0.19)` generates random floats between 0.01 and 0.20, allowing fine-grained exploration of the learning rate.
- These distributions are more efficient than discrete grids because they can probe values that a grid would miss (e.g., learning_rate = 0.073).

---

## 5. Key Concepts to Explain on Camera

### 5.1 Boosting vs. Bagging: The Core Distinction

This is the single most important conceptual point in the notebook. Frame it as a contrast:

| | Bagging (Random Forest) | Boosting (Gradient Boosting) |
|---|---|---|
| **Strategy** | Train many trees independently | Train trees sequentially |
| **Each tree sees** | A random bootstrap sample | The residuals (errors) of the current ensemble |
| **What it reduces** | Variance | Bias (and some variance) |
| **Tree depth** | Can be deep (low bias, high variance per tree) | Should be shallow (each tree is a "weak learner") |
| **Parallelizable** | Yes (independent trees) | No (each tree depends on previous) |
| **Overfitting risk** | Low (more trees rarely hurts) | Higher (more trees can overfit) |

**Analogy for students:** Random Forest is like polling 100 independent experts and taking the average opinion. Gradient Boosting is like hiring a coach who watches you practice, identifies your mistakes, and designs drills specifically for your weaknesses. The coach approach is more targeted and can make you better faster, but if the coach over-corrects for random bad plays, you end up worse.

### 5.2 The Learning Rate: The Most Important Hyperparameter

The learning rate (eta or `learning_rate`) controls how much each tree contributes to the ensemble. A learning rate of 0.1 means each tree's predictions are scaled by 0.1 before being added.

- **High learning rate (0.3-0.5):** Each tree contributes a lot. You need fewer trees, but the model may overshoot the optimal solution and overfit.
- **Low learning rate (0.01-0.05):** Each tree contributes a little. You need more trees, training takes longer, but the optimization path is smoother and the final model is often better.

**Rule of thumb:** Set the learning rate as low as your compute budget allows, then increase `n_estimators` until validation performance plateaus.

**What to emphasize on camera:** Walk through the learning rate sweep (Cell 11). At `learning_rate=0.01` with only 100 trees, the model underfits — not enough trees to compensate for tiny steps. At `learning_rate=0.5`, the model overshoots. The sweet spot is around 0.05-0.1 with 100 trees. Then show Cell 14 where lower learning rate with more trees (500 trees at 0.01) often produces the best result.

### 5.3 Why Keep Trees Shallow in Boosting

In Random Forests, each tree is typically deep (sometimes unrestricted) because the averaging cancels out individual overfitting. In Gradient Boosting, each tree should be shallow (depth 3-5) because:

1. **Each tree is a "weak learner."** The boosting algorithm assumes each base model is only slightly better than random. Shallow trees satisfy this assumption. Deep trees would be "strong learners" that might overfit the residuals from the first few iterations.

2. **Shallow trees prevent overfitting.** Since trees are added sequentially and each one corrects the previous errors, deep trees can over-correct, fitting noise in the residuals rather than signal.

3. **Shallow trees are faster.** With 100-500 trees in a sequential pipeline, each tree must be fast to train. Depth 3 means at most 8 leaves per tree — very fast.

### 5.4 Overfitting Monitoring Is Non-Negotiable

The staged-prediction plot (Cell 24) is the most important diagnostic in this notebook. It shows:

1. **Training ROC-AUC** climbing steadily toward 1.0 as more trees are added.
2. **Validation ROC-AUC** rising, peaking, and then either plateauing or declining.
3. **The optimal number of trees** at the validation peak.

**What to emphasize on camera:** Unlike Random Forests, where more trees is always safe, in Gradient Boosting, more trees past the validation peak is *actively harmful*. Every additional tree after the peak is fitting noise, not signal. In production, you would use early stopping to halt training automatically when validation performance stops improving.

### 5.5 The Cumulative Model Comparison

Section 7 (Cell 27) puts five models side by side: Logistic Regression, Decision Tree, Random Forest, default GBM, and tuned GBM. This is the most comprehensive comparison table in the course so far and previews the formal protocol in NB14.

**What to emphasize on camera:** The champion is identified at the bottom. On the breast cancer dataset, the margins between the top models are often razor-thin (< 0.01 ROC-AUC). This raises the question: is the extra complexity of tuned GBM worth 0.003 more ROC-AUC than a simple logistic regression? The answer depends on the deployment context — which is exactly what NB14 formalizes.

---

## 6. What Students Should Take Away

After watching your video and completing this notebook, students should be able to answer these questions confidently:

1. **"How is Gradient Boosting different from Random Forests?"** — Random Forests train trees in parallel on bootstrap samples and average them (reduces variance). Gradient Boosting trains trees sequentially, each one correcting the previous ensemble's errors (reduces bias). Boosting often achieves higher accuracy but requires more careful tuning.

2. **"What is the most important hyperparameter in Gradient Boosting?"** — The learning rate. Lower values (0.01-0.05) produce better results but require more trees. The rule of thumb: set the learning rate as low as your compute budget allows, then increase n_estimators until validation performance plateaus.

3. **"Why should boosting trees be shallow?"** — Because each tree is meant to be a weak learner that makes a small correction. Deep trees would over-correct, fitting noise in the residuals. Depth 3-5 is typical.

4. **"How do I know when to stop adding trees?"** — Monitor the validation curve. When it peaks and starts declining (or plateaus), stop. In production, use early stopping parameters (`n_iter_no_change` in sklearn, `early_stopping_rounds` in XGBoost).

5. **"When should I use GBM over Random Forests?"** — When you need maximum accuracy on tabular data and have the time/skill to tune hyperparameters carefully. If you want a strong model with minimal tuning, Random Forests are the safer choice. If you are willing to invest in hyperparameter search, GBM usually wins.

---

## 7. Common Student Questions (Anticipate These)

**Q: "Why is it called 'gradient' boosting?"**
A: Because each tree is fitted to the negative gradient of the loss function, which equals the residuals for squared-error loss. Think of it as gradient descent in function space: instead of updating parameter values, you update the model by adding a new tree that points in the direction of steepest loss reduction.

**Q: "If lower learning rate is always better, why not use 0.001?"**
A: Because you would need thousands of trees to compensate, which is slow. There is a practical tradeoff between accuracy and compute time. A learning rate of 0.01-0.05 with 200-500 trees is a good balance for most datasets.

**Q: "Why does GBM overfit but Random Forest does not?"**
A: Random Forest trees are independent — each one's errors are partially random, and averaging cancels random errors. GBM trees are dependent — each one specifically targets the previous ensemble's errors. After many iterations, the only remaining "errors" are noise, so the new trees start fitting noise. This is why monitoring the validation curve is essential for GBM but not critical for RF.

**Q: "What is the `subsample` parameter?"**
A: It controls the fraction of training samples used to fit each tree. A value of 0.8 means each tree sees 80% of the training data (randomly sampled without replacement). This adds a bagging-like randomness to boosting, which can reduce overfitting. It is sometimes called "stochastic gradient boosting."

**Q: "Should I use sklearn's GBM or XGBoost?"**
A: For this course, sklearn's `GradientBoostingClassifier` is sufficient and keeps the API consistent with everything else. In production, XGBoost and LightGBM are faster and have more features (built-in early stopping, GPU support, better handling of categorical features). The concepts — learning rate, tree depth, overfitting monitoring — are identical.

**Q: "The tuned GBM only beats the default by 0.002. Is that worth the tuning effort?"**
A: On this small dataset, probably not. But on larger datasets with more complex patterns, the difference can be much larger. More importantly, the tuning *process* (constrained search, CV evaluation, overfitting monitoring) is a skill that transfers to every future project.

---

## 8. Connection to the Broader Course Arc

| Week | Notebooks | How NB13 Connects |
|---|---|---|
| **Week 1** | 01-05 | NB13 reuses the pipeline and evaluation patterns from NB02-NB05. Gradient Boosting does not need feature scaling (tree-based), but the discipline of CV evaluation and leakage prevention is unchanged. |
| **Week 2** | 06-10 | NB13 continues with the breast cancer dataset from NB06-NB09, enabling direct comparison to logistic regression. The midterm (NB10) tested strategic metric selection — NB13 applies that by using ROC-AUC consistently. |
| **Week 3** | 11-15 | **NB13 completes the tree-based trilogy.** NB11 = single tree (building block). NB12 = parallel ensemble (Random Forest). NB13 = sequential ensemble (Gradient Boosting). NB14 = choose the champion. NB15 = interpret the champion. |
| **Week 4** | 16-20 | GBM is often the champion in students' final projects. NB16 (calibration) and NB17 (fairness) audit the champion's predictions. NB18 (reproducibility) serializes it for deployment. |

**Gradient Boosting is typically the strongest model on structured tabular data.** In Kaggle competitions, GBM variants (XGBoost, LightGBM, CatBoost) dominate the leaderboards. In this course, students who master NB13 have the most powerful tool in their arsenal for the final project.

---

## 9. Suggested Video Structure

Below are two recording options. Each segment includes the **speaking prompt** (what to say), **timestamps**, and the **notebook cells** to show on screen.

All cell references use the format `Cell N` where N is the zero-indexed cell number in `13_gradient_boosting.ipynb`.

---

### Option A: Single Video (~12 minutes)

#### Segment 1 — Opening & Boosting vs. Bagging `[0:00–2:30]`

> **Show:** Cell 0 (header) and Cell 1 (learning objectives) on screen.

**Say:**

"In the last two notebooks, we built single decision trees and then fixed their instability with Random Forests. Forests work by training many trees in parallel and averaging their predictions — a strategy called bagging. Today we learn a fundamentally different approach: Gradient Boosting. Instead of training trees in parallel, boosting trains them *sequentially*. Each new tree looks at the mistakes of the current ensemble and specifically tries to correct them. It is like having a tutor who watches your practice sessions, identifies your weak spots, and designs exercises targeting exactly those weaknesses."

> **Scroll** through the 5 learning objectives in Cell 1, briefly reading each one aloud.

> **Show:** Cell 4 (boosting vs. bagging comparison).

**Say:**

"Let me make the contrast explicit. Bagging: train many trees independently on bootstrap samples, average predictions. Reduces variance. Trees can be deep. Parallelizable. Boosting: train trees sequentially, each one correcting previous errors. Reduces bias. Trees should be shallow. Not parallelizable. The algorithm is simple: start with an initial prediction — the mean. Compute residuals. Fit a shallow tree to predict those residuals. Add it to the model with a small weight — the learning rate. Repeat. Each tree chips away at the remaining errors. After 100 or 200 iterations, the accumulated model is remarkably accurate."

---

#### Segment 2 — Baseline GBM vs. Random Forest `[2:30–5:00]`

> **Run** Cell 5 (load data).

**Say:**

"Same breast cancer dataset, same split: 398 training, 171 test, stratified. Same data as notebooks 11 and 12, so we can compare directly."

> **Run** Cell 8 (default GBM vs. RF comparison).

**Say:**

"Here is the first head-to-head: default Gradient Boosting versus 100-tree Random Forest, both under 5-fold stratified CV. Default GBM uses 100 trees, learning rate 0.1, max depth 3. Look at the numbers: GBM typically matches or slightly beats the forest — around 0.99 ROC-AUC on this dataset. The advantage line at the bottom shows the GBM lift. On a more complex dataset, this gap would be larger. But even out of the box, with no tuning at all, GBM is competitive with a forest."

> **Run** Cell 11 (learning rate sweep + plot).

**Say:**

"Now the most important hyperparameter: the learning rate. This sweep tests five values from 0.01 to 0.5, all with 100 trees. At 0.01, the model underfits — 100 tiny steps are not enough to reach the optimal solution. At 0.5, each step is too large, and the model may overshoot. The sweet spot is around 0.05 to 0.1. But here is the key insight: the optimal learning rate depends on the number of trees. If I give the model 500 trees instead of 100, a learning rate of 0.01 becomes viable — and often produces the best final model."

> **Run** Cell 14 (learning rate + n_estimators grid).

**Say:**

"This table confirms the tradeoff. Four configurations: 50 trees at 0.1, 100 trees at 0.1, 200 trees at 0.05, and 500 trees at 0.01. The best performer is typically the low-learning-rate, high-tree configuration. The product of learning rate times number of trees is sometimes called the 'effective budget' — configurations with similar budgets tend to perform similarly."

> **Mention:** "Pause the video here and complete Exercise 1 — compare GBM to RF and analyze when you would choose each." Point to Cell 16.

---

#### Segment 3 — Tuning with RandomizedSearchCV `[5:00–7:30]`

> **Show:** Cell 18 (hyperparameter descriptions), then **run** Cell 19 (RandomizedSearchCV).

**Say:**

"Gradient Boosting has five hyperparameters that matter: n_estimators, learning_rate, max_depth, min_samples_split, and subsample. A full grid search would be too expensive, so we use RandomizedSearchCV with 20 random trials. Notice the constrained search ranges: learning rate between 0.01 and 0.20, max depth between 3 and 7, subsample between 0.7 and 1.0. These ranges encode what we already learned from the sweeps — values outside these ranges are unlikely to help."

> **Point to** the best parameters and best CV score in the output.

**Say:**

"The best parameters typically settle around learning rate 0.05-0.15, depth 3-5, and 150-250 trees. The best CV score should be at or slightly above the default GBM's score. The test score confirms the CV estimate — a small gap means the model generalizes well. In practice, 20 random trials is a starting point. For a final production model, you might run 50-100 trials or follow up with a focused grid search around the best region."

> **Mention:** "Pause the video here and complete Exercise 2 — document your tuning strategy and results." Point to Cell 21.

---

#### Segment 4 — Overfitting Monitoring `[7:30–10:00]`

> **Run** Cell 24 (staged predictions: training vs. validation curves).

**Say:**

"This is the most important plot in the notebook. We train a GBM with 500 trees and learning rate 0.05, then track ROC-AUC on both training data and a held-out validation slice at every single iteration. The blue training curve climbs steadily toward 1.0. The orange validation curve rises, peaks — and then either plateaus or gently declines. The gap between the two curves is the overfitting signal."

> **Point to** the optimal number of trees in the printed output.

**Say:**

"The optimal number of trees is printed at the bottom — wherever the validation curve peaks. Everything after that point is fitting noise, not signal. This is the fundamental difference from Random Forests: in forests, more trees is always safe. In boosting, more trees past the validation peak is actively harmful. In production, you would use early stopping — halt training when the validation score has not improved for a set number of rounds. scikit-learn supports this with the n_iter_no_change parameter."

---

#### Segment 5 — Final Comparison & Wrap-Up `[10:00–12:00]`

> **Run** Cell 27 (five-model comparison table).

**Say:**

"Final comparison: five models under the same 5-fold CV. Logistic Regression, Decision Tree, Random Forest, default GBM, and tuned GBM. The table is sorted by CV ROC-AUC. Tuned GBM typically leads, followed closely by default GBM and Random Forest, then Logistic Regression, then the single tree. But look at the margins — on this dataset, the top three are within 0.01 of each other. Is the extra complexity of tuned GBM worth 0.003 more ROC-AUC? That depends on the deployment context, which is exactly what we formalize in the next notebook."

> **Show:** Cell 29 (wrap-up).

**Say:**

"Five takeaways. One: boosting trains trees sequentially, each correcting previous errors. Two: the learning rate is the most important knob — lower is better if you can afford more trees. Three: keep trees shallow — depth 3 to 5 — because each tree should be a weak learner. Four: monitor overfitting by tracking the validation curve at every iteration. Five: GBM often beats Random Forests on accuracy, but requires more careful tuning."

"Three rules to remember. Lower learning rate plus more trees equals better but slower. Keep trees shallow for boosting. And monitor overfitting closely — always. Next notebook, we formalize the model selection protocol: how to compare all these candidates fairly, choose a champion, write a selection memo, and evaluate on the test set exactly once. See you there."

> **Show:** Cell 30 (submission instructions), then Cell 31 (Thank You).

---

### Option B: Three Shorter Videos (~4-5 min each)

---

#### Video 1: "Boosting Intuition and Baseline Comparison" (~5 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–0:45` | Show header + objectives | Cell 0, Cell 1 | "We have been building toward this: the most powerful model for tabular data. Gradient Boosting trains trees sequentially, each one correcting the mistakes of the previous ensemble. Here are the five learning objectives." (Read them briefly.) |
| `0:45–1:45` | Boosting vs. bagging | Cell 4 | "Key contrast: Random Forests train trees in parallel and average them — variance reduction. Gradient Boosting trains trees sequentially, each one fitting residuals — bias reduction. The algorithm: start with the mean, compute errors, fit a shallow tree to errors, add it with a small weight, repeat. Each tree chips away at remaining errors." |
| `1:45–2:30` | Load data | Cell 5, Cell 6 | "Same breast cancer dataset, same split as notebooks 11 and 12: 398 training, 171 test, 30 features. Direct comparison to all previous models." |
| `2:30–3:30` | Default GBM vs. RF | Cell 7, Cell 8 | "Default GBM: 100 trees, learning rate 0.1, depth 3. Compare to 100-tree Random Forest under the same 5-fold CV. GBM typically matches or slightly beats the forest at around 0.99 ROC-AUC. Out of the box, no tuning needed, already competitive." |
| `3:30–5:00` | Learning rate sweep | Cell 10, Cell 11 | "The most important hyperparameter: learning rate. Five values from 0.01 to 0.5, all with 100 trees. Too low — underfits because 100 tiny steps are not enough. Too high — overshoots. Sweet spot around 0.05-0.1. But the optimal rate depends on the number of trees: more trees allow lower rates, which usually produce better final models." |

---

#### Video 2: "Tuning and Overfitting Monitoring" (~5 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–0:30` | Recap | — | "We established that default GBM is competitive with Random Forests. Now we tune it properly and learn the critical discipline of overfitting monitoring." |
| `0:30–1:00` | Learning rate + n_estimators grid | Cell 13, Cell 14 | "Four configurations: more trees with lower learning rate typically wins but takes longer. The product of learning rate times trees is the effective budget — similar budgets give similar performance." |
| `1:00–2:30` | RandomizedSearchCV | Cell 18, Cell 19 | "Five hyperparameters: n_estimators, learning_rate, max_depth, min_samples_split, subsample. Too many for grid search, so we use randomized search with 20 trials. Constrained ranges encode what we learned from the sweeps. Best parameters: typically learning rate 0.05-0.15, depth 3-5, 150-250 trees." |
| `2:30–4:30` | Staged predictions + overfitting plot | Cell 23, Cell 24 | "The most important plot: training and validation ROC-AUC at every iteration from 1 to 500 trees. Training curve climbs to 1.0. Validation curve peaks and then plateaus or declines. The peak is your optimal number of trees. Everything after it is fitting noise. This is the key difference from forests: in boosting, more trees can hurt. In production, use early stopping to halt at the peak automatically." |
| `4:30–5:00` | Exercise 2 prompt | Cell 21, Cell 22 | "Pause and complete Exercise 2: document your best parameters, compare to the baseline, and explain what you would try next with more compute." |

---

#### Video 3: "Final Comparison, Rules, and Bridge to Model Selection" (~4 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–1:30` | Five-model comparison | Cell 26, Cell 27 | "Final comparison: five models — Logistic Regression, Decision Tree, Random Forest, default GBM, tuned GBM — all under the same 5-fold CV. Tuned GBM typically leads, but the top three are within 0.01 ROC-AUC. The champion is printed at the bottom. Is the extra tuning complexity worth 0.003 more? That depends on the deployment context." |
| `1:30–2:30` | Wrap-up: five takeaways | Cell 29 | "Five takeaways: boosting is sequential, learning rate is the key knob, keep trees shallow, monitor overfitting always, and GBM usually beats RF but needs more tuning. Three rules: lower learning rate plus more trees equals better, keep depth 3-5, and never skip validation monitoring." |
| `2:30–4:00` | Bridge to NB14 + closing | Cell 29, Cell 30 | "We now have four model families in our toolkit: logistic regression, decision tree, Random Forest, and Gradient Boosting. The question is: how do you systematically choose between them? Not by gut feeling, not by trying metrics until one model wins, but by a formal protocol. Next notebook formalizes this: same CV folds for all models, define criteria before training, select on CV, validate on test exactly once, and write a champion selection memo. Submit your notebook and complete the Brightspace quiz. See you in notebook 14." |

---

*This guide was created to support video lecture recording for MGMT 47400 — Predictive Analytics, Purdue University.*
