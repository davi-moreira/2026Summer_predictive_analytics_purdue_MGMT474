# Video Lecture Guide: Notebook 12 — Random Forests - Bagging, OOB Intuition, and Feature Importance

## At a Glance

This guide covers:

1. **Why NB12 exists** — It solves the single tree's high-variance problem (demonstrated in NB11) by averaging many decorrelated trees, introducing students to the first ensemble method in the course

2. **Why it must follow NB11** — NB11 proved that single trees are unstable and overfit without depth control. NB12's entire motivation — "average many unstable trees to get a stable prediction" — only makes sense once students have experienced that instability firsthand

3. **Why it must precede NB13** — NB13 (Gradient Boosting) is a *sequential* ensemble that corrects errors iteratively. Understanding the *parallel* ensemble (Random Forest) first gives students a baseline for comparison and establishes the distinction between variance reduction (bagging) and bias reduction (boosting)

4. **Why each library/tool is used:**
   - `RandomForestClassifier` — the bagging + random feature selection ensemble
   - `permutation_importance` — model-agnostic feature importance, more reliable than built-in Gini importance
   - `oob_score` — "free" cross-validation from out-of-bag samples
   - `load_breast_cancer` — continuity with NB11 for fair model comparison

5. **Common student questions** with answers (e.g., why not just use more trees?, why is permutation importance better than Gini?, what is out-of-bag?)

6. **Connection to the full course arc** — Random Forests are the first ensemble, preparing students for Gradient Boosting (NB13) and the model selection protocol (NB14)

7. **Suggested video structure** — both single-video (~12 min) and three-video (~4-5 min each) formats with speaking prompts, timestamps, and notebook cell references

---

## Purpose of This Document

This guide is your **lecture companion** for recording the video(s) that accompany `12_random_forests_importance.ipynb`. It explains the *why* behind every design choice in the notebook — why bagging reduces variance, why random feature subsets decorrelate trees, why permutation importance should replace Gini importance in production, and why OOB scores approximate cross-validation for free. Use it as speaking notes, not as a script: the goal is to internalize the reasoning so you can explain it naturally on camera.

---

## 1. Why This Notebook Exists

NB11 ended with three identified weaknesses of single decision trees: high variance, overfitting, and instability. Students were told: "Trees are building blocks for ensembles." NB12 delivers on that promise by showing how to turn an army of mediocre, unstable trees into a powerful, stable ensemble.

The Random Forest algorithm has two key innovations:

1. **Bagging (Bootstrap Aggregating):** Train each tree on a different bootstrap sample (random sample with replacement) of the training data. This means each tree sees slightly different data, so they make different errors. Averaging their predictions cancels out individual errors, reducing variance.

2. **Random Feature Subsets:** At each split, only consider a random subset of features (typically sqrt(n_features) for classification). This prevents a single strong feature from dominating every tree's root split, further decorrelating the trees and amplifying the variance reduction.

This notebook also introduces two critical production skills: **permutation importance** (a reliable way to rank feature contributions) and **OOB scores** (a free proxy for cross-validation). These skills are immediately applicable to students' course projects.

---

## 2. Why It Must Come After Notebook 11

NB12 requires three concepts that NB11 establishes:

| NB11 Concept | How NB12 Uses It |
|---|---|
| **Single trees are high-variance.** NB11's depth sweep showed that small changes in training data produce very different tree structures. | NB12 opens by repeating this instability claim and then demonstrates that averaging 100 trees produces both higher mean performance and lower variance across CV folds. Without NB11's evidence, the claim "trees are unstable" would be unsubstantiated. |
| **Overfitting without depth control.** NB11 showed unrestricted trees memorize training data. | NB12 explains that Random Forests can use *deep* trees (even unrestricted) because averaging many overfit trees cancels out individual memorization. This counterintuitive idea — deliberately overfit each tree but average them — only makes sense after seeing what overfitting looks like in NB11. |
| **Tree visualization and split logic.** NB11 taught students to read tree diagrams. | NB12 uses this understanding when explaining how different trees in the forest make different splits due to bootstrap samples and random feature subsets. Students can visualize *why* the trees are different, even though the algorithm is the same. |

**In short:** NB11 identifies the disease (high variance); NB12 provides the cure (bagging + random features). Teaching the cure before the disease would be backward.

---

## 3. Why It Must Come Before Notebook 13

NB13 introduces Gradient Boosting — a fundamentally different ensemble strategy that builds trees *sequentially*, with each tree correcting the errors of the previous ones. The distinction between bagging (parallel, variance reduction) and boosting (sequential, bias reduction) is one of the most important conceptual contrasts in machine learning.

NB12 must come first for three reasons:

1. **Conceptual progression.** Bagging is simpler to understand: "train many trees independently and average." Boosting is more complex: "train trees sequentially, each one fitting the residuals of the previous model." Students need the simpler concept first.

2. **Performance baseline.** NB12 establishes the Random Forest's performance on the breast cancer dataset (CV ROC-AUC ~0.99). NB13 then asks: "Can Gradient Boosting do better?" Without NB12's baseline, the comparison would be meaningless.

3. **Feature importance foundation.** NB12 teaches both built-in (Gini) and permutation importance. NB13 uses the same importance tools without re-explaining them. NB14's model selection protocol assumes students can already produce importance rankings.

The deliberate sequence is:

```
NB11: Single tree — unstable, overfits (identifies the problem)
            |
NB12: Random Forest — parallel ensemble, variance reduction (first solution)
            |
NB13: Gradient Boosting — sequential ensemble, bias reduction (better solution, more tuning)
            |
NB14: Model Selection — pick the best from all candidates
```

---

## 4. Why These Specific Libraries and Tools

### 4.1 `sklearn.ensemble.RandomForestClassifier`

**What it does:** Trains an ensemble of decision trees using bagging with random feature subsets at each split. Final predictions are majority votes (classification) or averages (regression).

**Why we use it:**

1. **Direct successor to NB11.** The `RandomForestClassifier` is literally a collection of `DecisionTreeClassifier` objects. Students can trace the connection: the same base learner they studied in NB11, now combined into an ensemble.

2. **Few hyperparameters that matter.** The two most impactful knobs are `n_estimators` (number of trees) and `max_features` (features per split). This simplicity makes tuning fast and accessible, unlike Gradient Boosting (NB13) which has a trickier learning-rate / n-estimators coupling.

3. **Parallelizable.** The `n_jobs=-1` parameter trains all trees simultaneously across CPU cores. This is a practical advantage over Gradient Boosting, which is inherently sequential.

**What to emphasize on camera:** Random Forests are the "safe default" for tabular data. They rarely overfit badly (unlike single trees), require minimal tuning (unlike GBM), and deliver strong performance on most datasets. If you only learn one ensemble method, make it Random Forests.

### 4.2 `sklearn.inspection.permutation_importance`

**What it does:** For each feature, randomly shuffles its values, measures the drop in model performance, and reports the mean drop as the feature's importance. Repeated multiple times for statistical stability.

**Why we use it:**

1. **Model-agnostic.** Permutation importance works for any model — trees, forests, logistic regression, neural networks. Students learn one importance method that transfers across all model families.

2. **Computed on held-out data.** Unlike built-in Gini importance (which reflects training-time split frequency), permutation importance measures genuine predictive contribution on test data. This makes it more trustworthy for feature selection and reporting.

3. **Corrects Gini importance biases.** Gini importance is biased toward high-cardinality continuous features and double-counts correlated features. The notebook demonstrates this by showing side-by-side rankings where correlated features (e.g., `worst radius` and `worst perimeter`) swap positions between the two methods.

**What to emphasize on camera:** In your course project, always report permutation importance, not built-in importance. Built-in importance is fine for quick screening, but permutation importance is what goes in your final report and stakeholder presentations.

### 4.3 Out-of-Bag (OOB) Score via `oob_score=True`

**What it does:** Each tree in the forest is trained on a bootstrap sample (~63% of training data). The remaining ~37% ("out-of-bag" samples) are used to estimate that tree's generalization error. Aggregating OOB predictions across all trees produces a performance estimate without needing a separate validation set or cross-validation.

**Why we use it:**

1. **Free validation.** OOB score is computed during training with zero extra cost. Cross-validation requires training 5 separate forests (5x the compute). For initial hyperparameter screening, OOB is faster.

2. **Approximates CV well.** The notebook demonstrates that OOB accuracy and 5-fold CV accuracy agree within ~0.005-0.01. This empirical validation gives students confidence to use OOB for quick iterations.

3. **Conceptual bridge.** OOB introduces the idea that "built-in validation from the training procedure" is possible — a concept that reappears in Gradient Boosting's early stopping (NB13).

**What to emphasize on camera:** OOB is great for quick screening but not a complete replacement for CV. Always use full cross-validation for final model comparison (as in NB14), because OOB only works for bagging-based models, while CV works for any model.

### 4.4 `sklearn.datasets.load_breast_cancer`

**Why we continue with this dataset:** Direct comparison to NB11's single tree and NB06-NB09's logistic regression. Students can add the forest's numbers to their cumulative model comparison table, building toward NB14's formal model selection.

---

## 5. Key Concepts to Explain on Camera

### 5.1 Why Averaging Reduces Variance

This is the mathematical heart of bagging. If you have N independent estimators each with variance sigma-squared, their average has variance sigma-squared / N. Trees are not perfectly independent, but random feature subsets make them less correlated, so averaging still reduces variance substantially.

**Analogy for students:** Imagine asking 100 people to estimate the number of jelly beans in a jar. Each person's guess is noisy, but the *average* of 100 guesses is remarkably close to the true number. This is the "wisdom of crowds" effect, and it is exactly what Random Forests exploit.

### 5.2 Bootstrap Sampling + Random Feature Subsets = Decorrelated Trees

Bootstrap sampling alone (pure bagging) helps, but if one feature is very strong, every tree will use it at the root, making the trees correlated. Averaging correlated estimators does not reduce variance as much as averaging independent ones.

Random feature subsets fix this: at each split, only `sqrt(n_features)` candidates are considered. Sometimes the strong feature is not in the subset, forcing the tree to use alternative features. This creates more diverse trees, which average more effectively.

**What to emphasize on camera:** Show the `max_features` sweep (Cell 14). When `max_features=None` (all features), every tree is similar (high correlation, less variance reduction). When `max_features='sqrt'`, trees are diverse (low correlation, more variance reduction). The sweet spot is usually `sqrt` for classification, confirming the default.

### 5.3 n_estimators: More Trees = Better (But Diminishing Returns)

The n_estimators sweep (Cell 11) shows that performance improves rapidly from 10 to 50 trees, then plateaus around 100-200. Adding more trees always helps (or at worst does nothing), but the marginal gain shrinks.

**Rule of thumb:** Use 100-200 trees for experimentation, 500+ for a final production model. The only cost of more trees is compute time, which scales linearly with `n_estimators`.

### 5.4 Gini Importance vs. Permutation Importance

This is one of the most practically important lessons in the notebook. Students will be tempted to use `rf.feature_importances_` (Gini importance) because it is easy — it is a one-liner. But Gini importance has two known flaws:

1. **Biased toward continuous/high-cardinality features.** Features with more unique values offer more split candidates, so they appear in more splits and get higher importance scores, even if they are not genuinely predictive.

2. **Double-counts correlated features.** If `worst radius` and `worst perimeter` are correlated (r > 0.99), both get high Gini importance even though they carry essentially the same information.

Permutation importance avoids both problems because it measures the actual performance drop when a feature is shuffled, computed on held-out data.

**What to emphasize on camera:** Show the side-by-side comparison (Cell 24). Point out specific features whose rankings change between the two methods. This concrete evidence is more convincing than an abstract explanation.

### 5.5 The Comprehensive Model Comparison Table

Section 7 (Cell 29) compares four models under the same CV: Logistic Regression, Decision Tree, Random Forest (100 trees), and Random Forest (200 trees). This comparison table extends the one from NB11 and prepares for NB14's formal model selection protocol.

**What to emphasize on camera:** The table has four columns that matter: Model, CV Mean, CV Std, and Test Score. A model with both high mean and low std dominates. If two models have similar means, the one with lower std (more stable) is usually preferable.

---

## 6. What Students Should Take Away

After watching your video and completing this notebook, students should be able to answer these questions confidently:

1. **"Why do Random Forests outperform single trees?"** — Because averaging many trees reduces variance. Each tree sees different data (bootstrap) and different features (random subsets), making errors partially independent. When you average partially independent errors, the noise cancels out.

2. **"What are the most important hyperparameters to tune?"** — `n_estimators` (more is better until diminishing returns, usually 100-200 is enough) and `max_features` (controls tree diversity; `sqrt` is a strong default for classification).

3. **"What is the OOB score and when should I use it?"** — The OOB score is a free generalization estimate from the ~37% of data each tree does not see during training. Use it for quick hyperparameter screening; use full CV for final model comparison.

4. **"Why should I use permutation importance instead of built-in importance?"** — Permutation importance is computed on held-out data, is model-agnostic, and is not biased toward high-cardinality or correlated features. Built-in Gini importance is fast but can be misleading.

5. **"Do Random Forests overfit?"** — Rarely. Adding more trees does not cause overfitting (unlike adding depth to a single tree). The main risk is *underfitting* — using too few trees or too little depth. However, forests are not immune to overfitting entirely; it can happen with very high `max_depth` on small datasets.

---

## 7. Common Student Questions (Anticipate These)

**Q: "If more trees is always better, why not use 10,000 trees?"**
A: Because of diminishing returns and compute cost. Going from 100 to 200 trees might gain 0.002 ROC-AUC. Going from 200 to 10,000 might gain 0.0001 but take 50x longer. In practice, the performance plateaus well before the compute becomes unreasonable.

**Q: "Why does bootstrap sampling use ~63% of the data?"**
A: When you draw N samples with replacement from N items, each item has a (1 - 1/N)^N probability of never being drawn. As N gets large, this approaches 1/e, which is about 36.8%. So each bootstrap sample contains roughly 63.2% of the unique training samples. The remaining 36.8% are the out-of-bag samples.

**Q: "Can I use Random Forests for regression?"**
A: Yes. `RandomForestRegressor` works identically — each tree is a regression tree (minimizing MSE), and the forest averages their predictions. The hyperparameter tuning logic is the same, except the default `max_features` is `n_features/3` instead of `sqrt(n_features)`.

**Q: "Why does the notebook use `n_jobs=-1`?"**
A: This tells scikit-learn to use all available CPU cores for training. Since each tree in the forest is independent, they can be trained in parallel. This is a practical advantage over Gradient Boosting, which must train trees sequentially.

**Q: "When should I use a Random Forest instead of Gradient Boosting?"**
A: Random Forests are the safer default: they require less tuning, are harder to overfit, and train faster (parallel vs. sequential). Gradient Boosting *can* achieve higher accuracy but requires careful tuning of learning rate, number of trees, and depth. Use Random Forests first as your ensemble baseline, then try Gradient Boosting if you need more performance.

**Q: "Why does permutation importance have a standard deviation?"**
A: Because the shuffling is random. Each feature is shuffled multiple times (default `n_repeats=10`), and the performance drop varies slightly each time. The standard deviation quantifies this variability. A feature with high mean importance but also high std should be interpreted cautiously.

---

## 8. Connection to the Broader Course Arc

| Week | Notebooks | How NB12 Connects |
|---|---|---|
| **Week 1** | 01-05 | NB12's pipeline-free approach contrasts with NB02's pipeline emphasis. Trees and forests do not need scaling, simplifying the workflow. But the evaluation discipline (CV, train/val separation) from NB02-NB05 carries forward unchanged. |
| **Week 2** | 06-10 | The breast cancer dataset and StratifiedKFold from NB06-NB09 are reused. The model comparison table from NB12 extends the cumulative comparison started in NB09. |
| **Week 3** | 11-15 | **NB12 is the middle of the tree-based trilogy.** NB11 provides the building block (single tree). NB12 provides the parallel ensemble (forest). NB13 provides the sequential ensemble (boosting). NB14 compares them all. NB15 interprets the winner. |
| **Week 4** | 16-20 | Permutation importance from NB12 is reused in NB15 (interpretation), NB17 (fairness), and the final project. Students who master it here will use it throughout Week 4. |

**Random Forests are the course's "reliable workhorse."** They rarely win the final model competition (Gradient Boosting usually edges them out), but they provide the ensemble baseline that makes every other comparison meaningful.

---

## 9. Suggested Video Structure

Below are two recording options. Each segment includes the **speaking prompt** (what to say), **timestamps**, and the **notebook cells** to show on screen.

All cell references use the format `Cell N` where N is the zero-indexed cell number in `12_random_forests_importance.ipynb`.

---

### Option A: Single Video (~12 minutes)

#### Segment 1 — Opening & Bagging Intuition `[0:00–2:30]`

> **Show:** Cell 0 (header) and Cell 1 (learning objectives) on screen.

**Say:**

"In the last notebook, we built single decision trees and discovered three problems: they are unstable — small changes in data produce completely different trees. They overfit easily — an unrestricted tree memorizes every training sample. And their performance varies a lot across cross-validation folds. Today we fix all three problems with one idea: instead of building one tree, we build a hundred trees and average their predictions. This is the Random Forest."

> **Scroll** through the 5 learning objectives in Cell 1, briefly reading each one aloud.

> **Show:** Cell 4 (bagging explanation markdown).

**Say:**

"The Random Forest algorithm has two ingredients. First: bagging — short for bootstrap aggregating. We create 100 bootstrap samples of our training data — each one is a random sample with replacement, containing about 63 percent of the unique training points. We train one tree on each sample. Since each tree sees slightly different data, they make slightly different errors. When we average their predictions, the errors partially cancel out. Second: at each split, we only consider a random subset of features — typically the square root of the total number. This prevents one dominant feature from appearing in every tree's root, making the trees more diverse. More diversity means the errors are more independent, which means they cancel more effectively when averaged."

---

#### Segment 2 — Single Tree vs. Forest + n_estimators `[2:30–5:30]`

> **Run** Cell 5 (load data, print shapes).

**Say:**

"Same breast cancer dataset as notebook 11. 398 training, 171 test, 30 features. Same stratified split. This means we can directly compare the forest's numbers to the single tree's numbers from last time."

> **Run** Cell 8 (single tree vs. forest comparison + box plot).

**Say:**

"Here is the core result. The single depth-5 tree scores about 0.95-0.97 ROC-AUC with a standard deviation around 0.02 across folds. The 100-tree forest scores about 0.98-0.99 with a standard deviation around 0.01. Two improvements in one: higher mean *and* lower variance. The box plot makes it visual — the forest's box is both higher and narrower. That is what variance reduction looks like in practice."

> **Run** Cell 11 (n_estimators sweep + plot).

**Say:**

"How many trees do we need? This sweep goes from 10 to 300. Performance improves rapidly from 10 to 50 trees, then plateaus around 100-200. Going from 200 to 300 adds less than 0.001 to the mean ROC-AUC. Rule of thumb: use 100-200 trees for experimentation, bump to 500 for a final model. The only cost of more trees is compute time."

> **Mention:** "Pause the video here and complete Exercise 1 — tune n_estimators and max_features." Point to Cell 13.

---

#### Segment 3 — OOB Score + Feature Importance `[5:30–9:00]`

> **Show:** Cell 17 (OOB explanation), then **run** Cell 18 (OOB vs. CV comparison).

**Say:**

"Here is a bonus feature of Random Forests: the out-of-bag score. Each tree is trained on about 63 percent of the data. The other 37 percent — the out-of-bag samples — serve as a built-in validation set for that tree. Aggregate the OOB predictions across all 100 trees and you get a generalization estimate for free, without needing separate cross-validation. Look at the numbers: OOB accuracy and 5-fold CV accuracy agree within about 0.005. OOB is great for quick screening; save full CV for final comparisons."

> **Run** Cell 21 (built-in Gini importance + bar chart).

**Say:**

"Now feature importance. The built-in method — Gini importance — ranks features by how much they reduce impurity across all splits in all trees. 'Worst concave points' and 'worst radius' typically dominate. But there is a catch: Gini importance is biased toward continuous features with many unique values, and it double-counts correlated features. 'Worst radius' and 'worst perimeter' are correlated at r greater than 0.99 — they carry the same information, but both get high Gini importance."

> **Run** Cell 24 (permutation importance + comparison table).

**Say:**

"Permutation importance fixes these problems. For each feature, we randomly shuffle its values and measure how much ROC-AUC drops. If shuffling a feature barely affects performance, it is not important. We do this on the test set — not the training set — so we measure genuine predictive contribution, not memorized patterns. Look at the comparison table: some features swap positions between the two methods. In your project reports, always use permutation importance. It is more reliable, more defensible, and works for any model, not just forests."

> **Mention:** "Pause the video here and complete Exercise 2 — write three interpretation bullets about the importance rankings." Point to Cell 26.

---

#### Segment 4 — Model Comparison & Wrap-Up `[9:00–12:00]`

> **Run** Cell 29 (comprehensive model comparison table + bar chart).

**Say:**

"Final comparison: four models under the same 5-fold CV. Logistic Regression, single Decision Tree, Random Forest with 100 trees, Random Forest with 200 trees. The forests lead, followed closely by Logistic Regression, with the single tree trailing. On this dataset the margin between forest and logistic regression is small — both above 0.97 — because the data is nearly linearly separable. The real payoff of forests comes on datasets with non-linear patterns and feature interactions, which your project data likely has."

> **Show:** Cell 31 (wrap-up section). Read the key takeaways.

**Say:**

"Five takeaways. One: bagging reduces variance by averaging many trees. Two: random feature subsets decorrelate the trees, amplifying the variance reduction. Three: n_estimators and max_features are the two most important knobs. Four: OOB score gives you free validation. Five: use permutation importance for production, not built-in Gini importance."

"Three critical rules to remember. More trees is almost always better — with diminishing returns after 100-500. Use permutation importance for production. And Random Forests rarely overfit badly, but they can underfit if you use too few trees or too much feature restriction."

"Next notebook: Gradient Boosting. Instead of training trees in parallel and averaging them — which is what forests do — Gradient Boosting trains trees *sequentially*, where each new tree corrects the mistakes of the previous ones. It is more powerful but more sensitive to tuning. See you there."

> **Show:** Cell 32 (submission instructions), then Cell 33 (Thank You).

---

### Option B: Three Shorter Videos (~4-5 min each)

---

#### Video 1: "From Single Tree to Forest" (~5 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–0:45` | Show header + objectives | Cell 0, Cell 1 | "Last notebook, we found that single decision trees are unstable and overfit easily. Today we fix both problems with one idea: train a hundred trees on different data samples and average their predictions. This is the Random Forest. Here are the five learning objectives." (Read them briefly.) |
| `0:45–1:30` | Bagging intuition | Cell 4 | "Two ingredients: bagging and random feature subsets. Bagging creates bootstrap samples — random sampling with replacement — so each tree sees different data. Random feature subsets force different trees to use different features at each split. Together, they create diverse trees whose errors partially cancel when averaged. Think of it as the wisdom of crowds applied to decision trees." |
| `1:30–2:30` | Load data | Cell 5, Cell 6 | "Same breast cancer dataset, same split as notebook 11. 398 training, 171 test, 30 features. We can compare forest numbers directly to single-tree numbers from last time." |
| `2:30–4:00` | Tree vs. forest comparison | Cell 7, Cell 8 | "The core result: single depth-5 tree scores about 0.95-0.97 ROC-AUC with high variance across folds. The 100-tree forest scores 0.98-0.99 with low variance. Higher mean, lower spread. The box plot makes it visual: the forest's box is both higher and narrower. That is variance reduction in action." |
| `4:00–5:00` | n_estimators sweep | Cell 11, Cell 12 | "How many trees? 10 to 300 sweep. Performance plateaus around 100-200 trees. Adding more always helps, but diminishing returns mean the effort is not worth it past 200-300 for most datasets. Use 100-200 for experiments, 500 for production." |

---

#### Video 2: "OOB Score and Feature Importance" (~5 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–0:30` | Recap | — | "We have a 100-tree forest that outperforms a single tree. Now two production skills: estimating performance without cross-validation, and ranking which features matter most." |
| `0:30–1:30` | OOB score | Cell 17, Cell 18 | "Each tree trains on 63 percent of the data. The remaining 37 percent — out-of-bag samples — provide a free validation estimate. OOB accuracy agrees with 5-fold CV within about 0.005. Use OOB for quick screening, CV for final comparisons. It costs nothing extra because it is computed during training." |
| `1:30–3:00` | Built-in Gini importance | Cell 20, Cell 21 | "Built-in feature importance ranks features by Gini impurity reduction. 'Worst' features dominate because they capture extreme cell nucleus measurements. But Gini importance has two flaws: it favors continuous features with many split points, and it double-counts correlated features. 'Worst radius' and 'worst perimeter' are nearly perfectly correlated, yet both rank high." |
| `3:00–4:30` | Permutation importance | Cell 23, Cell 24 | "Permutation importance fixes both flaws. Shuffle each feature on the test set, measure how much ROC-AUC drops. Features whose shuffling barely affects performance are not important. Compare the rankings: some features swap positions. Always use permutation importance in your project reports. It is more reliable, works for any model, and is computed on held-out data." |
| `4:30–5:00` | Exercise 2 prompt | Cell 26, Cell 27 | "Pause and complete Exercise 2: write three interpretation bullets. Which features matter most? How do the two methods differ? What business insight does this provide?" |

---

#### Video 3: "Model Comparison and Wrap-Up" (~4 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–1:30` | Comprehensive comparison | Cell 28, Cell 29 | "Final comparison: Logistic Regression, Decision Tree, Random Forest 100 trees, Random Forest 200 trees — all under the same 5-fold CV. Forests lead in ROC-AUC with the lowest standard deviation. On this dataset the margin over logistic regression is small because the data is linearly separable. On your project data, the gap may be larger." |
| `1:30–2:30` | Wrap-up takeaways | Cell 31 | "Five takeaways. One: bagging reduces variance. Two: random features decorrelate trees. Three: n_estimators and max_features are the key knobs. Four: OOB gives free validation. Five: permutation importance beats Gini importance. Three rules: more trees is better with diminishing returns, use permutation importance for production, and forests rarely overfit." |
| `2:30–4:00` | Bridge to NB13 + closing | Cell 31, Cell 32 | "Random Forests fix the variance problem by averaging parallel trees. But what if the problem is not variance but *bias* — the model is not flexible enough to capture the signal? Gradient Boosting fixes bias by training trees *sequentially*, each one correcting the errors of the previous ones. It is more powerful but more sensitive to tuning. Next notebook, we explore that tradeoff. Submit your notebook and complete the quiz in Brightspace." |

---

*This guide was created to support video lecture recording for MGMT 47400 — Predictive Analytics, Purdue University.*
