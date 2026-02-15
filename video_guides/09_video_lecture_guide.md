# Video Lecture Guide: Notebook 09 — Feature Engineering + Model Selection Workflow (Project Baseline Build)

## At a Glance

This guide covers:

1. **Why NB09 exists** — It brings together everything from Week 2 into a practical workflow: feature engineering inside pipelines, systematic hyperparameter tuning with GridSearchCV and RandomizedSearchCV, and a project baseline report that students can extend for their course project

2. **Why it must follow NB08** — NB08 teaches cross-validation as a standalone evaluation tool. NB09 embeds CV inside grid search — students must understand what `cv=StratifiedKFold(5)` means inside `GridSearchCV` before they can interpret tuning results

3. **Why it must precede NB10** — NB10 is the midterm, which asks students to design evaluation plans and submit a project baseline. NB09 provides the scaffold, the tuning tools, and the reporting template they need for both deliverables

4. **Why each library/tool is used:**
   - `GridSearchCV` — exhaustive search over discrete parameter grids with built-in CV
   - `RandomizedSearchCV` — sampling-based search over continuous distributions for large spaces
   - `PolynomialFeatures` — demonstrates feature engineering (and its dangers) inside pipelines
   - `SelectKBest` — univariate feature selection to manage dimensionality before polynomial expansion
   - `scipy.stats.uniform`, `randint` — parameter distributions for randomized search

5. **Common student questions** with answers (e.g., grid search vs. randomized search?, how do I tune pipeline parameters?)

6. **Connection to the full course arc** — how the baseline report template from NB09 becomes the project deliverable in NB10, the comparison framework for Week 3 ensembles, and the final project report structure

7. **Suggested video structure** — both single-video (~12 min) and three-video (~4-5 min each) formats with speaking prompts, timestamps, and notebook cell references

---

## Purpose of This Document

This guide is your **lecture companion** for recording the video(s) that accompany `09_tuning_feature_engineering_project_baseline.ipynb`. It explains the *why* behind every design choice — why feature engineering must live inside the pipeline, why grid search and randomized search are both necessary, and why the baseline report table is the most project-relevant artifact in the notebook. Use it as speaking notes, not as a script: the goal is to internalize the reasoning so you can explain it naturally on camera.

---

## 1. Why This Notebook Exists

Notebooks 06-08 teach the individual pieces of the classification workflow: logistic regression (NB06), metrics and thresholds (NB07), and cross-validation for comparison (NB08). But none of them answer the question that students face at this point in the course: **"I have a baseline model — how do I systematically improve it?"**

Notebook 09 answers this question with two tools: **feature engineering** (creating new features from existing ones) and **hyperparameter tuning** (searching for the best model configuration). Both tools are wrapped inside the pipeline and evaluated with cross-validation, ensuring no leakage and fair comparison.

The notebook also marks a pedagogical transition. It is the **last teaching notebook before the midterm** (NB10). By the end of NB09, students should be able to:
- Build a baseline model with proper pipelines and CV evaluation
- Engineer features (polynomial, selection) without leakage
- Tune hyperparameters systematically (grid search or randomized search)
- Produce a baseline report table that documents their modeling journey

This is the capstone of Week 2's classification arc: NB06 built the model, NB07 measured it, NB08 stabilized the measurement, and NB09 improves the model and documents the results.

---

## 2. Why It Must Come After Notebook 08

Notebook 08 establishes the cross-validation framework that notebook 09 *embeds inside grid search*:

| NB08 Concept | How NB09 Uses It |
|---|---|
| **`StratifiedKFold` as CV splitter** | NB09 passes `cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)` to both `GridSearchCV` and `RandomizedSearchCV`. This is the same splitter object from NB08 — students recognize it instantly. |
| **`cross_val_score` for evaluation** | NB09 uses `cross_val_score` to evaluate the polynomial features pipeline and compare it against the baseline. This is the same function from NB08, applied to a new context. |
| **Fair comparison rules (same folds, same metrics)** | NB09's baseline report table compares three models evaluated under the same conditions. The fairness principles from NB08 are baked into the workflow. |
| **Reporting mean +/- std** | NB09's GridSearchCV results include `mean_test_score` and `std_test_score` for every parameter combination. Students who understand CV variance from NB08 can immediately interpret these columns. |

**In short:** NB08 teaches CV as a standalone tool; NB09 shows CV embedded inside hyperparameter search.

If you tried to teach GridSearchCV *before* CV, students would not understand what the `cv` parameter does, what `mean_test_score` means, or why the same folds must be used for all parameter combinations. If you skipped NB09 entirely, students would enter the midterm without knowing how to tune models or engineer features.

---

## 3. Why It Must Come Before Notebook 10

Notebook 10 is the **midterm**, which has two components:

1. **Business case analysis** — students design evaluation plans for scenarios like customer churn and loan default. These plans require choosing metrics (from NB07), specifying CV strategies (from NB08), and proposing modeling shortlists with tuning approaches (from NB09).

2. **Project Milestone 2** — students submit a baseline model notebook with proper splits, a baseline model, an improved model, a comparison table, and an evaluation plan. NB09's baseline report table is the direct template for this deliverable.

Without NB09, students would not know how to:
- Tune hyperparameters systematically (they would guess-and-check)
- Engineer features inside pipelines (they would risk leakage)
- Structure a baseline report (they would produce ad hoc results)

The deliberate sequence is:

```
NB08: How do we evaluate models reliably? (CV framework)
            |
NB09: How do we improve models systematically? (Feature engineering + tuning + baseline report)
            |
NB10: Prove you can design and execute an evaluation plan (Midterm + project milestone)
```

NB09 is the final preparation before the midterm. Everything students need for NB10 is established by NB09.

---

## 4. Why These Specific Libraries and Tools

### 4.1 `sklearn.model_selection.GridSearchCV`

**What it does:** Exhaustively evaluates every combination of hyperparameters in a specified grid, using cross-validation to score each combination. After the search, it refits the best parameters on the full training set.

**Why we use it:**

1. **Systematic, not ad hoc.** Students have been manually trying different C values (NB06), comparing a few models (NB08). GridSearchCV automates this process: define the grid, run one command, get the best parameters. No more guess-and-check.

2. **Built-in CV.** Each parameter combination is evaluated using k-fold CV, not a single split. This means the "best" parameters are robust — they perform well across multiple data splits, not just one lucky partition.

3. **Pipeline-aware.** When searching over pipeline parameters, the notation `clf__C` tells GridSearchCV which step and which parameter to tune. This keeps feature engineering and model tuning inside the same pipeline, preventing leakage.

4. **Results accessible via `cv_results_`.** The full results DataFrame shows mean scores, standard deviations, ranks, and timing for every combination. Students can visualize how performance varies with each parameter.

**What to emphasize on camera:** GridSearchCV does three things: search (try all combinations), evaluate (CV each one), and refit (train the best on full data). After calling `.fit()`, the object IS the best model — you can call `.predict()` and `.score()` directly. Also emphasize the computational cost: 4 C values times 2 solvers times 5 folds = 40 model fits. For small grids this is fine; for large grids, use RandomizedSearchCV.

### 4.2 `sklearn.model_selection.RandomizedSearchCV`

**What it does:** Samples a fixed number of parameter combinations from specified distributions (instead of trying all combinations), evaluates each with CV, and returns the best.

**Why we use it:**

- **Handles continuous parameters.** GridSearchCV requires a discrete list of values. RandomizedSearchCV can sample from continuous distributions like `uniform(0, 1)` or integer distributions like `randint(50, 200)`. This is essential for parameters like `max_depth` or `min_samples_split` that have wide, continuous ranges.
- **Scales to large search spaces.** With 4 parameters each having 5 possible values, grid search requires 5^4 = 625 combinations. Randomized search with `n_iter=20` tries only 20 combinations while still exploring the space. Research by Bergstra and Bengio (2012) shows that random search finds near-optimal configurations more efficiently than grid search in high-dimensional parameter spaces.
- **Budget control.** The `n_iter` parameter sets the computational budget. You decide how many combinations to try, regardless of grid size.

**What to emphasize on camera:** Use GridSearchCV when the parameter space is small and discrete (e.g., 2-3 parameters with 3-4 values each). Use RandomizedSearchCV for initial exploration of large or continuous spaces. A common workflow: randomized search first to identify a good region, then grid search to refine within that region.

### 4.3 `sklearn.preprocessing.PolynomialFeatures`

**What it does:** Generates polynomial and interaction features up to a specified degree. For degree 2 with 10 features, it produces 10 original + 10 squared + 45 pairwise interactions = 65 features.

**Why we use it:**

- **Demonstrates feature engineering inside pipelines.** The key lesson is that `PolynomialFeatures` must be inside the pipeline so that it is fitted (statistics learned) only on training data. If you created polynomial features *before* splitting, the interactions would be computed using all data — leakage.
- **Illustrates the dimensionality explosion.** With 10 input features, degree-2 polynomials produce 65 features. With 30 features (full breast cancer), degree 2 would produce 496 features. Students see firsthand why dimensionality control (via `SelectKBest`) is necessary before polynomial expansion.
- **Provides a concrete "does it help?" experiment.** The notebook compares baseline CV scores with polynomial CV scores. Sometimes polynomial features help; sometimes they hurt. Students learn that more features are not always better.

**What to emphasize on camera:** Polynomial features are a double-edged sword. They can capture non-linear relationships that a linear model misses, but they multiply the feature space quadratically and increase overfitting risk. Always compare against the baseline with CV to determine if the added complexity is justified.

### 4.4 `sklearn.feature_selection.SelectKBest`

**What it does:** Selects the top k features based on a univariate statistical test (here, `f_classif` — ANOVA F-statistic for classification).

**Why we use it:**

- **Controls dimensionality.** Before polynomial expansion, we reduce from 30 features to 10 or 20, keeping the polynomial output manageable (65 or 231 features instead of 496).
- **Inside the pipeline = no leakage.** `SelectKBest` computes F-statistics on training data only. If you selected features on the full dataset, you would be using validation/test labels to choose features — leakage.
- **Demonstrates the feature engineering pipeline pattern.** Scale -> Select -> Polynomial -> Scale again -> Model. Students see how multiple preprocessing steps chain together.

**What to emphasize on camera:** Feature selection is *preprocessing*, and like all preprocessing, it must happen inside the pipeline. The features that survive selection might differ if the training data changes (e.g., in a different CV fold), and the pipeline handles this correctly.

### 4.5 `scipy.stats.uniform` and `scipy.stats.randint`

**What they do:** Define continuous (uniform) and discrete (randint) probability distributions for parameter sampling in RandomizedSearchCV.

**Why we use them:**

- **Enable continuous parameter exploration.** Instead of testing C = [0.01, 0.1, 1.0, 10.0], you can sample C uniformly from [0.01, 100]. This covers the entire range without requiring you to guess which discrete values to include.
- **Standard scipy distributions.** Students should recognize these as the same distributions from introductory statistics. The connection between statistical distributions and machine learning tuning reinforces the underlying math.

**What to emphasize on camera:** `randint(50, 200)` generates integers between 50 and 199 uniformly at random. `uniform(0, 1)` generates floats between 0 and 1. These distributions define the *search space* — RandomizedSearchCV samples from them, evaluates each sample with CV, and returns the best.

---

## 5. Key Concepts to Explain on Camera

### 5.1 Feature Engineering Must Live Inside the Pipeline

The most common leakage bug in applied ML is computing features *outside* the pipeline:

```python
# WRONG: leaks information
X_poly = PolynomialFeatures(degree=2).fit_transform(X)  # Sees ALL data
X_train, X_test = train_test_split(X_poly, ...)
```

The correct approach:

```python
# RIGHT: no leakage
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('model', LogisticRegression())
])
pipeline.fit(X_train, y_train)  # Polynomial features fitted on train only
```

**Analogy for students:** Feature engineering is like adding seasoning to a recipe. If you season the entire pot before tasting (fitting on all data), you cannot objectively judge the flavor (evaluate on unseen data). Season the training portion, evaluate on the unseasoned test portion.

### 5.2 GridSearchCV = Nested Loops (Parameter x CV)

Students often struggle to visualize what GridSearchCV does. The mental model is two nested loops:

```
for each parameter combination:
    for each CV fold:
        train model with these parameters on this fold's training data
        evaluate on this fold's validation data
    average the fold scores
select the combination with the best average
refit on all training data with the best parameters
```

The total number of model fits is: (number of combinations) x (number of folds). For a grid of 8 combinations and 5 folds, that is 40 fits. This is important for estimating runtime and deciding whether to use grid search or randomized search.

### 5.3 The Baseline Report Table — A Project Deliverable

The baseline report table at the end of the notebook is not just a summary — it is a **deliverable format** that students will use in their project. It documents:
- Every model tried (with its name)
- The parameters used (default or tuned)
- The validation metric (ROC-AUC)
- The feature count
- A brief note

This table answers the stakeholder question: "What did you try, and what worked best?" It also answers the reproducibility question: "How did you arrive at this model?"

### 5.4 Start Small, Then Refine

The notebook demonstrates a pragmatic tuning workflow:
1. **Baseline** — default parameters, no feature engineering
2. **Feature engineering** — does adding/selecting features help?
3. **Grid search** — small grid (8 combinations) on the best pipeline
4. **Randomized search** — broader exploration of a more complex model (Random Forest)
5. **Report** — compare all candidates in one table

This "start simple, add complexity only if it helps" philosophy prevents students from jumping to complex tuning before establishing whether the baseline is strong enough.

---

## 6. What Students Should Take Away

After watching your video and completing this notebook, students should be able to answer these questions confidently:

1. **"Why must feature engineering be inside the pipeline?"** — Because feature engineering steps like polynomial expansion and feature selection compute statistics from data. If those statistics include validation/test samples, the evaluation is contaminated. The pipeline ensures these steps are fitted only on training data.

2. **"When should I use GridSearchCV vs. RandomizedSearchCV?"** — Use GridSearchCV when the parameter space is small and discrete (e.g., 3 parameters with 3-4 values each = 30-80 combinations). Use RandomizedSearchCV when the space is large, continuous, or high-dimensional (e.g., 4+ parameters with wide ranges). A common workflow is: randomized search first to identify a promising region, then grid search to fine-tune within that region.

3. **"What does `clf__C` mean in the parameter grid?"** — The double underscore separates the pipeline step name (`clf`) from the parameter name (`C`). This notation tells GridSearchCV to tune the `C` parameter of the step named `clf` in the pipeline. It works for any pipeline step and any parameter.

4. **"How do I know if feature engineering helped?"** — Compare the CV scores of the pipeline with and without the new features. If the mean score improves by more than one standard deviation, the improvement is likely real. If it is within the noise, the extra features add complexity without benefit.

5. **"What goes into a project baseline report?"** — A comparison table with model names, parameters, validation metrics, feature counts, and notes. Plus an evaluation plan: primary metric, CV strategy, leakage prevention measures, and next steps. This table is a deliverable, not a debugging tool.

---

## 7. Common Student Questions (Anticipate These)

**Q: "GridSearchCV takes forever. How do I speed it up?"**
A: Three strategies. First, reduce the grid — start with 2-3 values per parameter instead of 10. Second, use `n_jobs=-1` to parallelize across CPU cores. Third, switch to `RandomizedSearchCV` with a fixed `n_iter` budget. A grid of 4x3x3x3 = 108 combinations with 5-fold CV requires 540 fits. Randomized search with n_iter=20 requires only 100 fits.

**Q: "The polynomial pipeline has two scalers. Why?"**
A: The first `StandardScaler` normalizes the original features before `SelectKBest` and polynomial expansion. After creating polynomial features, the new columns (squares and interactions) are on very different scales. The second `StandardScaler` normalizes the expanded feature set before logistic regression, ensuring the regularization penalty treats all features fairly.

**Q: "Should I always try polynomial features?"**
A: No. Polynomial features are most useful when you suspect non-linear relationships that a linear model cannot capture. On the breast cancer dataset, the classes are already well-separated linearly, so polynomials may not help. On datasets with complex boundaries (e.g., spatial data, biological assays), they can be powerful. Always compare against the baseline with CV — never assume more features are better.

**Q: "What if GridSearchCV and my validation set give different answers about the best model?"**
A: The CV score from GridSearchCV is an average over multiple folds of the training data. The validation score is from a single held-out set. Small discrepancies are normal. If they disagree dramatically (e.g., grid search picks C=0.01 but validation prefers C=10), it may indicate that the validation set is not representative, or that the model is sensitive to the data partition. In such cases, trust the CV score — it is more robust.

**Q: "How does this connect to the course project?"**
A: Directly. The baseline report table at the end of this notebook is the template for Project Milestone 2 (due with the midterm). Copy the table format, replace the breast cancer models with your own, and add your evaluation plan. The `GridSearchCV` workflow is how you will tune your project model.

---

## 8. Connection to the Broader Course Arc

| Week | Notebooks | What NB09's Tuning & Reporting Enable |
|---|---|---|
| **Week 2** | 10 | The midterm asks students to design evaluation plans that include tuning strategy. The baseline report table from NB09 is the template for Project Milestone 2 |
| **Week 3** | 11-15 | Tree-based models (Decision Tree, Random Forest, Gradient Boosting) use the same `GridSearchCV` pattern from NB09. Parameters like `max_depth`, `n_estimators`, and `min_samples_split` are tuned using the same workflow |
| **Week 3** | 13-14 | Ensemble methods combine multiple models. Feature engineering pipelines from NB09 are reused to prepare data for stacking and boosting |
| **Week 4** | 16-18 | Model deployment requires a finalized pipeline. The best pipeline from GridSearchCV (NB09) is the artifact that gets serialized with `joblib.dump()` in NB18 |
| **Week 4** | 19-20 | Final project requires a comprehensive model comparison with tuned hyperparameters. The `GridSearchCV` + report table pattern from NB09 is the workflow students follow |

**NB09 is the practical capstone of Week 2.** It combines every skill from the week — logistic regression (NB06), metrics (NB07), cross-validation (NB08) — into a complete model improvement and reporting workflow. The baseline report table becomes a reusable template for the rest of the course.

---

## 9. Suggested Video Structure

Below are two recording options. Each segment includes the **speaking prompt** (what to say), **timestamps**, and the **notebook cells** to show on screen.

All cell references use the format `Cell N` where N is the zero-indexed cell number in `09_tuning_feature_engineering_project_baseline.ipynb`.

---

### Option A: Single Video (~12 minutes)

#### Segment 1 — Opening & Motivation `[0:00–1:30]`

> **Show:** Cell 0 (header) and Cell 1 (learning objectives) on screen.

**Say:**

"This is the capstone notebook for Week 2. Over the past three notebooks, we built a logistic regression classifier, learned to evaluate it with precision, recall, ROC-AUC, and cost-based thresholds, and stabilized our estimates with cross-validation. Today we answer the next question: how do I systematically improve my model? The answer has two parts: feature engineering — creating new features from existing ones — and hyperparameter tuning — searching for the best model configuration. Both must happen inside the pipeline to prevent leakage, and both must be evaluated with cross-validation to ensure the improvement is real. By the end of this notebook, you will have a baseline report table that you can directly adapt for your course project. Let me walk through the five learning objectives."

> **Action:** Scroll through the 5 learning objectives in Cell 1, briefly reading each one aloud.

---

#### Segment 2 — Setup & Baseline `[1:30–3:00]`

> **Show:** Cell 2 (setup code), then **run** it.

**Say:**

"New imports for today: GridSearchCV and RandomizedSearchCV for tuning, PolynomialFeatures for feature engineering, SelectKBest for feature selection, and scipy distributions for randomized search parameter spaces. Everything else is familiar — breast cancer dataset, StandardScaler pipeline, random seed 474."

> **Show:** Cell 4 (baseline explanation), then **run** Cell 5 (load data + baseline).

**Say:**

"Before we try anything fancy, we establish the baseline. Same breast cancer dataset, same 60-20-20 split with stratification, same logistic regression pipeline with default parameters. Baseline validation accuracy: about 97%. This number is our reference point. Every experiment in this notebook must be compared against it. If feature engineering or tuning does not beat 97%, the extra complexity is not justified."

---

#### Segment 3 — Feature Engineering Inside Pipelines `[3:00–5:30]`

> **Show:** Cell 7 (feature engineering patterns), then **run** Cell 8 (SelectKBest pipeline).

**Say:**

"Feature engineering step one: feature selection. We use SelectKBest to keep only the 20 most informative features out of 30, based on the ANOVA F-statistic. Notice this happens inside the pipeline — the F-statistics are computed from training data only. The result: 20 features instead of 30, and the validation accuracy is close to the baseline. This tells us that 10 of the 30 features are not contributing much — Lasso from Week 1 told us the same thing."

> **Show:** Cell 10 (Exercise 1 prompt), then **run** Cell 11 (polynomial features comparison).

**Say:**

"Feature engineering step two: polynomial features. We first reduce to 10 features with SelectKBest, then generate degree-2 polynomials — all squares and pairwise interactions. This expands 10 features to 65. We add a second StandardScaler after the polynomial step because the new features are on different scales. Then we compare with CV: baseline versus polynomial, both scored by ROC-AUC. The improvement may be marginal or nonexistent on this dataset — the classes are already well-separated linearly. But the lesson is the *process*: engineer features inside the pipeline, compare with CV, keep the improvement only if it is meaningful. Notice the feature explosion warning: 10 features became 65. With all 30 features, degree-2 would give us 496. That is why we select first, then expand."

---

#### Segment 4 — GridSearchCV `[5:30–8:30]`

> **Show:** Cell 14 (GridSearchCV explanation), then **run** Cell 15 (grid search execution).

**Say:**

"Now we tune hyperparameters. GridSearchCV takes three inputs: a pipeline, a parameter grid, and a CV splitter. The grid specifies 4 values of C and 2 solvers — 8 combinations total. With 5-fold CV, that is 40 model fits. The double-underscore notation — `clf__C` — tells grid search which step and parameter to tune. After fitting, three results: best parameters, best CV score, and validation score. The best CV score is the average over 5 folds for the winning combination. The validation score is an independent check — if it matches the CV score closely, the search did not overfit to the folds."

> **Run** Cell 17 (results table + C parameter plot).

**Say:**

"The results table shows all 8 combinations ranked by CV score. The line plot shows how C affects ROC-AUC for each solver. The curve typically flattens after C equals 0.1 or 1.0, meaning further reducing regularization does not help. The two solvers — lbfgs and liblinear — produce nearly identical scores, confirming that solver choice is secondary to regularization strength for this problem. This visualization is important: it tells you whether the parameter space is well-explored. If the best score is at the edge of your grid, you should extend the grid in that direction."

---

#### Segment 5 — RandomizedSearchCV `[8:30–10:00]`

> **Show:** Cell 19 (randomized search explanation), then **run** Cell 20 (randomized search on Random Forest).

**Say:**

"Now we tune a more complex model — Random Forest — with RandomizedSearchCV. The parameter space has four dimensions: n_estimators from 50 to 200, max_depth from 3 to 20, min_samples_split from 2 to 20, and min_samples_leaf from 1 to 10. Grid search would require testing every combination — potentially thousands. Randomized search samples 20 random combinations and evaluates each with 5-fold CV — 100 fits total, much more manageable.

The best parameters, best CV score, and validation score are printed. Random Forest may match or slightly exceed logistic regression's performance, but at significantly higher computational cost. Remember the lesson from NB08: if two models perform similarly, prefer the simpler one."

---

#### Segment 6 — Baseline Report, Gemini Tips & Closing `[10:00–12:00]`

> **Show:** Cell 22 (Exercise 2 prompt), then **run** Cell 23 (baseline report table).

**Say:**

"Here is the most project-relevant output: the baseline report table. Three models — logistic regression with defaults, logistic regression tuned by grid search, and Random Forest tuned by randomized search — compared on validation ROC-AUC. Each row documents the model name, the metric, the parameters, the feature count, and a note. The champion model is highlighted. This is the format you will use in your course project. Copy this table, replace the models with your own, and you have your Project Milestone 2 deliverable."

> **Show:** Cell 26 (Gemini prompts), then Cell 27 (Wrap-Up).

**Say:**

"A quick note on using Gemini for tuning. You can ask Gemini to generate parameter grids, optimize search spaces, or debug slow searches. But always verify the suggestions with CV — Gemini does not know your specific data distribution. The prompt examples in the notebook show effective ways to get useful scaffolds from Gemini while maintaining your own critical judgment.

Three rules from today. One: all feature engineering must be inside the pipeline. Two: start with small grids, then refine — do not try to optimize everything at once. Three: document every modeling choice in a baseline report table. This is the last teaching notebook before the midterm. In NB10, you will apply everything from Week 2 — logistic regression, metrics, cross-validation, tuning, and reporting — to business case analyses and your project baseline. Make sure your pipeline template is solid. See you at the midterm."

> **Show:** Cell 28 (Submission Instructions) briefly, then Cell 29 (Thank You).

---

### Option B: Three Shorter Videos (~4-5 minutes each)

---

#### Video 1: "Feature Engineering Inside Pipelines" (~5 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–0:45` | Show header + objectives | Cell 0, Cell 1 | "This is the capstone of Week 2. We have built a classifier, measured it, and stabilized the measurement. Today we systematically improve it through feature engineering and hyperparameter tuning. Here are our five learning objectives — notice the last one: produce a baseline model notebook for your project." (Read objectives briefly.) |
| `0:45–1:30` | Run setup + baseline | Cell 2, Cell 3, Cell 4, Cell 5, Cell 6 | "Breast cancer dataset, 60-20-20 split, logistic regression baseline. Validation accuracy about 97%. This is our reference. Every experiment must beat it or the complexity is not justified." |
| `1:30–3:00` | Feature selection | Cell 7, Cell 8, Cell 9 | "Feature selection with SelectKBest: keep the top 20 features by ANOVA F-statistic, discard the 10 weakest. This happens inside the pipeline — F-statistics computed on training data only. Accuracy stays close to baseline, meaning those 10 features were not contributing much. Feature selection does not always improve accuracy, but it reduces dimensionality, speeds up training, and simplifies the model." |
| `3:00–5:00` | Polynomial features | Cell 10, Cell 11, Cell 12, Cell 13 | "Polynomial features: reduce to 10 features first, then generate degree-2 polynomials — 65 features from 10. We add a second scaler because polynomials produce values on wildly different scales. CV comparison: baseline versus polynomial. On this dataset, the improvement may be negligible because the classes separate linearly. The lesson is the process, not the result. And the warning: 10 features became 65. With all 30, that would be 496. Feature explosion is real — always select before you expand." |

---

#### Video 2: "GridSearchCV and RandomizedSearchCV" (~5 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–0:30` | Recap | — | "Welcome back. We have tried feature engineering and seen that it does not always help dramatically. Now we try the other lever: hyperparameter tuning. The tools are GridSearchCV and RandomizedSearchCV." |
| `0:30–2:00` | GridSearchCV | Cell 14, Cell 15, Cell 16 | "GridSearchCV takes a pipeline, a parameter grid, and a CV splitter. Our grid: 4 values of C times 2 solvers equals 8 combinations, times 5 folds equals 40 fits. The double-underscore notation — clf double-underscore C — addresses the C parameter of the clf step in the pipeline. Results: best parameters, best CV score, validation score. If CV and validation agree, the search was fair." |
| `2:00–3:00` | Examine grid results | Cell 17, Cell 18 | "The results table ranks all 8 combinations. The plot shows C versus ROC-AUC for each solver. Performance flattens after C equals 1.0 — further tuning yields diminishing returns. If the best score were at the edge of your grid, you would extend the search. Here it is in the middle, so the grid was well-designed." |
| `3:00–5:00` | RandomizedSearchCV | Cell 19, Cell 20, Cell 21 | "For Random Forest with four continuous parameters, grid search would require thousands of combinations. RandomizedSearchCV samples 20 random combinations from scipy distributions — randint for integers, uniform for floats. Twenty combinations times 5 folds equals 100 fits, far more manageable than exhaustive search. Best parameters, best CV score, and validation score are reported. Random Forest may match logistic regression's accuracy but takes longer to train. The practical recommendation: use grid search for small discrete grids, randomized search for large or continuous spaces." |

---

#### Video 3: "Baseline Report, Project Prep, and Wrap-Up" (~4 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–1:30` | Baseline report table | Cell 22, Cell 23, Cell 24 | "The baseline report table is the most important output of this notebook. Three models, each documented with name, metric, parameters, feature count, and note. The champion is highlighted. This table is a deliverable — copy it into your project and replace the models with your own. It answers the stakeholder question: what did you try, and what worked best? And it answers the reproducibility question: how did you arrive at this model?" |
| `1:30–2:30` | Gemini prompts + scaffold | Cell 25, Cell 26 | "The Gemini section shows how to use AI assistance for tuning. Ask Gemini to generate parameter grids, optimize search spaces, or debug slow searches. But always verify with CV — Gemini does not know your data distribution. The scaffold in Section 5 lists the six required components of a project baseline: data audit, splits, baseline model, improved model, evaluation report, and documentation. Use this checklist when building your Project Milestone 2." |
| `2:30–4:00` | Wrap-up + closing | Cell 27, Cell 28, Cell 29 | "Three rules. One: all feature engineering inside the pipeline. Two: start with small grids, then refine. Three: document every modeling choice in a baseline report. This is the last teaching notebook before the midterm. NB10 tests everything from Week 2: logistic regression, metrics, cross-validation, tuning, reporting, business case analysis, and your project baseline submission. Make sure your pipeline template is solid, your metrics vocabulary is sharp, and your baseline report is ready. Submit your notebook and complete the quiz. See you at the midterm." |

---

*This guide was created to support video lecture recording for MGMT 47400 — Predictive Analytics, Purdue University.*
