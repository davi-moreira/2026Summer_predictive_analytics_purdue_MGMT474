# Video Lecture Guide: Notebook 02 — Data Setup and Preprocessing Pipelines

## At a Glance

This guide covers:

1. **Why NB02 exists** — It operationalizes the leakage prevention promise from NB01 by teaching the tool (Pipeline) that makes safe preprocessing automatic

2. **Why it must follow NB01** — NB01 provides the vocabulary (split, leakage, EDA); NB02 gives the tool that enforces it. Without NB01, students wouldn't understand *why* fitting only on training data matters

3. **Why it must precede NB03** — NB03 focuses on *evaluating* model quality (metrics, baselines). It assumes the pipeline is a solved problem. Teaching both preprocessing and evaluation in one notebook would be cognitive overload

4. **Why each library/tool is used:**
   - `Pipeline` — leakage prevention by construction, reproducibility, deployment-readiness
   - `ColumnTransformer` — routes numeric/categorical columns through appropriate transformations
   - `StandardScaler` — prepares for regularized models in NB05
   - `SimpleImputer` — anticipates real-world missingness even though the demo dataset is complete
   - `LinearRegression` — deliberately simple baseline so focus stays on pipeline mechanics

5. **Common student questions** with answers (e.g., why not `pd.get_dummies()`?, why scale if LinearRegression doesn't need it?)

6. **Connection to the full course arc** — how the pipeline template from NB02 is reused in all 18 subsequent notebooks

7. **Suggested video structure** — both single-video (~12 min) and three-video (~4-5 min each) formats with speaking prompts, timestamps, and notebook cell references

---

## Purpose of This Document

This guide is your **lecture companion** for recording the video(s) that accompany `02_preprocessing_pipelines.ipynb`. It explains the *why* behind every design choice in the notebook — why it exists, why it sits between notebooks 01 and 03, why it uses specific libraries, and why concepts like `ColumnTransformer` and `Pipeline` are introduced this early. Use it as speaking notes, not as a script: the goal is to internalize the reasoning so you can explain it naturally on camera.

---

## 1. Why This Notebook Exists

Notebook 01 ends with a **proper train/val/test split** and a promise: *"Split first, preprocess second, model third."* Students leave NB01 knowing that they must never let validation or test data influence their preprocessing. But they have not actually *done* any preprocessing yet.

Notebook 02 delivers on that promise. It answers the immediate question every student has after splitting: **"I have three clean DataFrames — now what do I do with them before I can train a model?"**

The answer is: build a **preprocessing pipeline** — a structured, reproducible, leak-proof sequence of transformations that turns raw features into model-ready inputs. This notebook teaches that skill as the *first* hands-on technical task of the course, because every subsequent notebook depends on it.

---

## 2. Why It Must Come After Notebook 01

Notebook 01 establishes three foundational ideas that notebook 02 *consumes*:

| NB01 Concept | How NB02 Uses It |
|---|---|
| **Train/val/test split (60/20/20)** | NB02 loads the same California Housing dataset and immediately splits it. Students already understand *why* the split exists; now they learn *how to respect it* during preprocessing. |
| **Data leakage** | NB01 introduces two types of leakage (target leakage and train-test contamination). NB02 makes the second type concrete: if you compute the scaler's mean on the full dataset instead of the training set, you contaminate your evaluation. The pipeline enforces the correct behavior automatically. |
| **EDA and data types** | NB01 teaches students to inspect data types, missing values, and distributions. NB02's `make_data_report()` function formalizes that EDA into a reusable audit. Students see the same checks they did manually in NB01, now wrapped in a function they can apply to any dataset. |

**In short:** NB01 gives the vocabulary (split, leakage, EDA), NB02 gives the tool (Pipeline) that operationalizes that vocabulary.

If you tried to teach pipelines *before* NB01, students would not understand *why* fitting only on training data matters. If you skipped NB02 entirely, students would manually preprocess their data in NB03 and beyond — which invites copy-paste errors, inconsistent transformations, and silent leakage.

---

## 3. Why It Must Come Before Notebook 03

Notebook 03 introduces **regression metrics and baseline models**. Its learning objectives assume students already know how to:

- Build a `Pipeline` that chains preprocessing with a model
- Call `pipeline.fit(X_train, y_train)` and `pipeline.score(X_val, y_val)`
- Trust that the pipeline handles scaling and imputation without leakage

NB03's focus is on *evaluating* model quality (MAE, RMSE, R², residual plots), not on building the preprocessing machinery. If students had to learn both preprocessing *and* evaluation in the same notebook, the cognitive load would be too high and neither concept would stick.

The deliberate sequence is:

```
NB01: Why do we split? What is leakage?
            ↓
NB02: How do we preprocess safely? (Pipeline + ColumnTransformer)
            ↓
NB03: How do we measure model quality? (Metrics + baselines)
```

Each notebook adds exactly one conceptual layer. By NB03, the pipeline is a solved problem — students can focus entirely on what the numbers mean.

---

## 4. Why These Specific Libraries and Tools

### 4.1 `sklearn.pipeline.Pipeline`

**What it does:** Chains multiple processing steps (transformations + model) into a single object with `.fit()`, `.predict()`, and `.score()` methods.

**Why we use it:**

1. **Leakage prevention by construction.** When you call `pipeline.fit(X_train, y_train)`, every transformer in the chain sees *only* training data. When you call `pipeline.predict(X_val)`, those same transformers apply their *already-learned* parameters to the validation data. Students do not need to remember "fit on train, transform on val" for each step — the pipeline enforces it.

2. **Reproducibility.** The pipeline is a self-contained description of the entire data-to-prediction workflow. You can serialize it with `joblib.dump()`, hand it to a colleague or deploy it to production, and the exact same transformations will be applied. No separate preprocessing scripts, no mismatched scaler parameters.

3. **Simplicity for students.** Instead of writing 5-6 separate fit/transform calls (impute, scale, encode, fit model, predict), students write 2 lines: `pipeline.fit(X_train, y_train)` and `pipeline.score(X_val, y_val)`. This reduces cognitive overhead and lets them focus on *what* to preprocess rather than *how* to wire it together.

**What to emphasize on camera:** The pipeline is not just a convenience — it is a **safety mechanism**. In a real business project, manual preprocessing is where most leakage bugs hide. The pipeline makes those bugs structurally impossible.

### 4.2 `sklearn.compose.ColumnTransformer`

**What it does:** Applies different transformation pipelines to different subsets of columns, then concatenates the results into a single feature matrix.

**Why we use it:**

Real-world datasets almost always mix numeric and categorical columns. Each type needs different treatment:

| Column Type | Typical Steps | Why |
|---|---|---|
| **Numeric** | Impute (median) then Scale (StandardScaler) | Models like linear regression and regularized models are sensitive to feature scale. Median imputation is robust to outliers. |
| **Categorical** | Impute (constant) then Encode (OneHotEncoder) | Models cannot process strings. One-hot encoding creates binary columns that preserve category information without imposing ordinal relationships. |

`ColumnTransformer` routes each column group through its appropriate pipeline and produces a clean numeric matrix. Without it, students would need to manually slice DataFrames, transform each piece, and recombine them — a process that is tedious and error-prone.

**What to emphasize on camera:** Even though the California Housing dataset has only numeric features (so the categorical path is idle), we include the categorical transformer *on purpose*. It teaches the pattern. When students encounter their project datasets (which will have categorical columns), the same code structure works with zero changes. Build the habit now, benefit later.

### 4.3 `sklearn.preprocessing.StandardScaler`

**What it does:** Transforms each feature to have mean = 0 and standard deviation = 1, using statistics computed from the training data.

**Why we use it:**

- **Linear regression** (used in this notebook) is not strictly scale-sensitive, but regularized models introduced in NB05 (Ridge, Lasso) **are**. If one feature ranges from 0 to 10 and another from 0 to 10,000, the regularization penalty unfairly penalizes the smaller-scale feature. Scaling fixes this.
- By introducing scaling in NB02 (before students need it), we build the habit early. When Ridge regression appears in NB05, students already have a scaled pipeline and can focus on the regularization concept rather than debugging scale mismatches.

**What to emphasize on camera:** `StandardScaler` computes mean and std from `X_train` during `.fit()`, then applies those same values to `X_val` and `X_test` during `.transform()`. This is the "fit on train only" principle in action. If you fitted the scaler on the entire dataset, the mean would include information from the validation and test sets — that is leakage.

### 4.4 `sklearn.impute.SimpleImputer`

**What it does:** Replaces missing values (`NaN`) with a computed statistic (mean, median, most frequent) or a constant value.

**Why we use it:**

- **Most real-world datasets have missing data.** The California Housing dataset happens to be complete, but production data almost never is. Teaching students to include an imputer in every pipeline means they do not need to rewrite their code when missingness appears.
- **The imputer is inside the pipeline**, which means it respects the fit-on-train-only rule. If you imputed manually before splitting, you would compute the median from *all* data — leakage.
- **Median strategy** is the default for numeric features because it is robust to outliers. The California Housing data has extreme values in `AveOccup` and `AveRooms`; mean imputation would be pulled toward those outliers.

**What to emphasize on camera:** The imputer does literally nothing in this notebook (zero missing values). That is *the point*. A professional pipeline anticipates problems before they happen. When students' project data has missing values in 6 columns, the same pipeline handles it automatically.

### 4.5 `sklearn.linear_model.LinearRegression`

**What it does:** Fits a linear model: $\hat{y} = \beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p$.

**Why we use it here:**

- **It is the simplest supervised model.** The focus of NB02 is on preprocessing infrastructure, not on model sophistication. Using linear regression keeps the modeling step trivial so students can concentrate on the pipeline mechanics.
- **It provides a baseline R² (~0.60).** This number becomes the benchmark that students will try to beat in NB03 (metrics), NB04 (feature engineering), NB05 (regularization), and beyond.
- **It demonstrates the pipeline in action.** Students see that `full_pipeline.fit(X_train, y_train)` fits the imputer, the scaler, *and* the model in one call. Swapping `LinearRegression()` for any other estimator (tree, SVM, ensemble) requires changing exactly one line.

**What to emphasize on camera:** The model is deliberately underwhelming. An R² of 0.60 means the model explains only 60% of the variance in house prices. That is fine — the goal here is not a good model, it is a *correct* pipeline. We will improve the model in future notebooks.

---

## 5. Key Concepts to Explain on Camera

### 5.1 "Fit on Train Only" — The Golden Rule of Preprocessing

This is the single most important idea in the notebook. Every transformer (imputer, scaler, encoder) has two phases:

- **`.fit()`**: Learn parameters from data (e.g., compute mean and std).
- **`.transform()`**: Apply those learned parameters to data.

The rule: **`.fit()` must only see training data.** Validation and test data are only `.transform()`-ed using the parameters learned from training. The `Pipeline` enforces this automatically, but students need to understand *why* — because when they debug or extend the pipeline, they need to know what would break the rule.

**Analogy for students:** Imagine you are designing a test for a class. You write the questions based on what you taught (training data). You do not look at student answers (validation/test data) before writing the questions. If you peeked at the answers and designed questions around them, the test would not measure real understanding — it would be "leaked."

### 5.2 Why Preprocessing Must Be Inside the Pipeline (Not Outside)

A common beginner mistake:

```python
# WRONG: leaks information
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Sees ALL data
X_train, X_test = train_test_split(X_scaled, ...)
```

The correct approach:

```python
# RIGHT: no leakage
X_train, X_test = train_test_split(X, ...)
pipeline = Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())])
pipeline.fit(X_train, y_train)  # Scaler sees ONLY training data
```

When preprocessing lives inside the pipeline, it is *structurally impossible* to leak. When it lives outside, it is *structurally inevitable*.

### 5.3 The Data Audit as a Professional Habit

The `make_data_report()` function is not just a teaching device — it is a real-world practice. Before touching any data with transformations, a data scientist audits:

- **Data types:** Are strings being treated as numbers? Are dates parsed correctly?
- **Missing values:** How many? Which columns? Is missingness random or systematic?
- **Cardinality:** How many unique values does each column have? High-cardinality categoricals need special handling.

Running this audit on *each split separately* catches schema mismatches. For example, if a rare category appears only in the test set, the one-hot encoder (fitted on training data) would not know about it — `handle_unknown='ignore'` in the `OneHotEncoder` handles this gracefully, but only if you know to look for it.

### 5.4 The Template Nature of This Notebook

NB02 is designed as a **reusable template**. Every future notebook in the course will follow the same pattern:

1. Load data
2. Split (60/20/20)
3. Audit (data report)
4. Build `ColumnTransformer` (numeric + categorical paths)
5. Chain into `Pipeline` (preprocessor + model)
6. Fit on train, evaluate on validation
7. Never touch test until the final evaluation

By teaching this pattern once in NB02 and repeating it in every subsequent notebook, it becomes muscle memory. Students do not need to reinvent the preprocessing workflow each time — they copy the template and change only the model and feature engineering steps.

---

## 6. What Students Should Take Away

After watching your video and completing this notebook, students should be able to answer these questions confidently:

1. **"Why can't I scale the data before splitting?"** — Because the scaler would learn the mean and std from the full dataset, including validation and test rows. That is train-test contamination.

2. **"What is a ColumnTransformer?"** — It routes different column groups through different preprocessing pipelines and concatenates the results. Numeric columns get scaled; categorical columns get encoded.

3. **"Why include an imputer if there are no missing values?"** — Because production data will have missing values, and a robust pipeline handles them automatically. Build the pipeline right once.

4. **"What does an R² of 0.60 mean?"** — The model explains 60% of the variance in house prices. It is a reasonable baseline but not a good final model — we will improve it.

5. **"Why is Pipeline better than manual preprocessing?"** — It prevents leakage by construction, it is reproducible, and it is deployment-ready.

---

## 7. Common Student Questions (Anticipate These)

**Q: "If LinearRegression doesn't need scaling, why do we scale?"**
A: Because the pipeline is a *template*. In NB05, we swap LinearRegression for Ridge/Lasso, which *do* need scaling. By scaling from the start, the swap requires changing one line instead of restructuring the entire pipeline.

**Q: "Why median imputation instead of mean?"**
A: Median is robust to outliers. The dataset has extreme values in AveOccup (e.g., 1,243 people per household — likely a data artifact). Mean imputation would be pulled toward these outliers; median ignores them.

**Q: "What does `handle_unknown='ignore'` do in OneHotEncoder?"**
A: If the validation or test set contains a category that the encoder never saw during training, `'ignore'` produces a row of all zeros instead of raising an error. This is essential for production robustness.

**Q: "Why `remainder='drop'`?"**
A: Any column not explicitly routed through a transformer is discarded. This is a safety net — it prevents accidentally including the target variable or an ID column that slipped into the feature set.

**Q: "Why not just use `pd.get_dummies()` instead of OneHotEncoder?"**
A: `get_dummies()` is a pandas function that works on the *entire DataFrame at once*. It cannot be fitted on training data and applied to validation/test data separately. `OneHotEncoder` inside a pipeline can — and that is the difference between a school exercise and a production system.

---

## 8. Connection to the Broader Course Arc

| Week | Notebooks | What NB02's Pipeline Enables |
|---|---|---|
| **Week 1** | 01-05 | NB02 provides the preprocessing template reused in NB03 (metrics), NB04 (feature engineering), NB05 (regularization) |
| **Week 2** | 06-10 | Same pipeline structure, now with classification models (LogisticRegression replaces LinearRegression) |
| **Week 3** | 11-15 | Tree-based models (DecisionTree, RandomForest, GradientBoosting) plug into the same pipeline |
| **Week 4** | 16-20 | Pipeline is serialized for deployment (NB18), used in final project (NB20) |

**The pipeline is the backbone of the entire course.** Every modeling notebook from NB03 onward follows the same structure: load → split → preprocess (pipeline) → model → evaluate. NB02 is where students learn this structure for the first and most important time.

---

## 9. Suggested Video Structure

Below are two recording options. Each segment includes the **speaking prompt** (what to say), **timestamps**, and the **notebook cells** to show on screen.

All cell references use the format `Cell N` where N is the zero-indexed cell number in `02_preprocessing_pipelines.ipynb`.

---

### Option A: Single Video (~12 minutes)

#### Segment 1 — Opening & Motivation `[0:00–1:30]`

> **Show:** Cell 0 (header) and Cell 1 (learning objectives) on screen.

**Say:**

"Welcome back. In the previous notebook we learned three things: how to split data into train, validation, and test sets; what data leakage is; and why we must split *before* we do anything else to the data. We ended with a promise — split first, preprocess second, model third. Today we deliver on that promise. We are going to build a preprocessing pipeline — a single scikit-learn object that handles imputation, scaling, and encoding, all while guaranteeing that no information from the validation or test sets leaks into our training process. By the end of this notebook, you will have a reusable template that you will use in every single notebook for the rest of this course."

> **Action:** Scroll through the 5 learning objectives in Cell 1, briefly reading each one aloud.

---

#### Segment 2 — Setup & Data Loading `[1:30–3:00]`

> **Show:** Cell 2 (setup explanation), then **run** Cell 3 (imports).

**Say:**

"Let's start with our standard setup. Every notebook in this course begins the same way — we import pandas, numpy, matplotlib, seaborn, and our scikit-learn tools. Notice the new imports here that we did not have in notebook 01: Pipeline, ColumnTransformer, StandardScaler, SimpleImputer, and OneHotEncoder. Each one has a specific job in the preprocessing chain, and I will explain them as we build the pipeline. As always, we lock the random seed to 474 — the course number — so every split and every result is reproducible."

> **Run** Cell 3. Point to the `✓ Setup complete!` output.

> **Show:** Cell 5 (dataset explanation), then **run** Cell 6 (load data).

**Say:**

"We continue with the California Housing dataset from notebook 01 — about 20,000 census block-groups with 8 numeric features and a target variable, median house value. Since you already explored this data in the previous notebook, we can move quickly here."

> **Run** Cell 6. Point to the shape (20640, 9) and the `.head()` preview.

> **Run** Cell 8 (split). Point to the 60/20/20 percentages.

**Say:**

"And here is our 60-20-20 split — same pattern as notebook 01. Training set: about 12,000 samples. Validation: about 4,000. Test: about 4,000. The test set is locked away — we will not touch it until the very end of the course. Everything we do today uses only the training and validation sets."

---

#### Segment 3 — Data Audit `[3:00–5:00]`

> **Show:** Cell 10 (data audit explanation), then **run** Cell 11 (`make_data_report` on training set).

**Say:**

"Before we preprocess anything, we audit the training data. This function checks three things for every column: the data type, how many values are missing, and how many unique values there are. Think of it as a health check-up for your data. On this dataset, the audit is clean — all 8 features are float64, zero missing values, and the unique counts look reasonable. But in your course project, this will not always be the case. You might find string columns that should be numeric, missing values in 30% of a column, or a categorical column with 10,000 unique values. Running this audit first saves you from surprises later."

> **Run** Cell 14 (audit on validation and test sets).

**Say:**

"Now we run the same audit on the validation and test sets. We are looking for consistency — same data types, similar distributions of unique values, no unexpected missingness. If you ever see a column that is float64 in training but object in validation, something went wrong in your data loading. This three-way comparison is a simple but powerful sanity check."

> **Mention:** "Pause the video here and complete Exercise 1 — run the audit on all three splits and document any differences." Point to Cell 13 (PAUSE-AND-DO 1).

---

#### Segment 4 — Building the ColumnTransformer `[5:00–8:00]`

> **Run** Cell 17 (identify column types). Point to the output: 8 numeric, 0 categorical.

**Say:**

"Before we build the pipeline, we need to know what types of columns we have. We use `select_dtypes` to automatically separate numeric and categorical features. This dataset is all numeric, but we write the code to handle both types because your project data will almost certainly have categorical columns — things like neighborhood names, product categories, or yes/no flags."

> **Show:** Cell 19 (pipeline design principles), then **run** Cell 20 (build preprocessor).

**Say:**

"Now the heart of this notebook — the ColumnTransformer. Here is what it does: it takes your feature matrix and routes different columns through different transformation pipelines. Numeric features go through two steps: first, SimpleImputer with median strategy fills any missing values; then StandardScaler centers each feature to mean zero and standard deviation one. Categorical features go through a different path: SimpleImputer fills missing values with the string 'missing', then OneHotEncoder converts each category into binary columns. The ColumnTransformer applies these transformations in parallel and concatenates the results into one clean matrix.

Two things I want you to notice. First, the imputer uses median, not mean. Why? Because median is robust to outliers. If you have a column where most values are between 2 and 5 but one row has a value of 1,200 — which happens in this dataset — the mean would be pulled way up, but the median stays sensible. Second, we include the categorical path even though this dataset has zero categorical columns. That is intentional. When you swap in a different dataset for your project, this same code works without changes. Build the pipeline right once, reuse it everywhere."

> **Show:** Cell 22 (full pipeline explanation), then **run** Cell 23 (fit pipeline + scores).

**Say:**

"Now we chain the ColumnTransformer with a model — LinearRegression — into a single Pipeline object. When I call `pipeline.fit(X_train, y_train)`, three things happen in sequence: the imputer computes medians from the training data, the scaler computes means and standard deviations from the training data, and the linear regression fits its coefficients on the scaled training data. All from one line of code. And when I call `pipeline.score(X_val, y_val)`, the validation data is transformed using the *training* statistics — not its own. That is how the pipeline prevents leakage by construction.

The results: train R-squared is about 0.60, validation R-squared is about 0.62. The gap is tiny — essentially no overfitting. But R-squared of 0.60 means this linear model explains only 60 percent of the variance in house prices. That is a reasonable *baseline*, not a great model. In the next notebooks, we will learn how to measure this properly and then improve it with feature engineering, regularization, and more flexible models."

---

#### Segment 5 — Inspecting the Pipeline `[8:00–9:30]`

> **Run** Cell 26 (get feature names).

**Say:**

"One more important skill — inspecting what comes out of the pipeline. The `get_feature_names_out` method shows every column that exits the ColumnTransformer. Notice the `num__` prefix on each name — that tells you which transformer created it. In this case, 8 features went in and 8 came out, just rescaled. If we had categorical columns, you would see `cat__` prefixed columns — one for each category created by the one-hot encoder. Being able to trace transformed features back to their original names is essential when you start interpreting model coefficients in later notebooks."

> **Mention:** "Pause the video here and complete Exercise 2 — experiment with different imputation strategies and compare validation scores." Point to Cell 28 (PAUSE-AND-DO 2).

---

#### Segment 6 — Checklist, Wrap-Up & Closing `[9:30–12:00]`

> **Show:** Cell 33 (Pipeline Done Right Checklist). Read through the 4 categories.

**Say:**

"Before we wrap up, let me walk through the Pipeline Done Right Checklist. This is your quality control for every pipeline you build in this course. Pre-fitting: did you split before building the pipeline? Did you identify your column types? Fitting: did you fit only on training data? Is the model the last step? Evaluation: did you evaluate on validation, not test? Is the train-val gap reasonable? Code quality: are your parameters explicit? Can you explain every step? Go through this checklist every time."

> **Show:** Cell 34 (Wrap-Up).

**Say:**

"Let me leave you with two critical rules. First: fit only on training data. Any statistics — means, medians, category lists — must come from the training set only. If you compute them on the full dataset, you have leakage, and your evaluation numbers are lying to you. Second: put everything inside the pipeline. If you do preprocessing manually — scaling here, encoding there, fitting the model over here — you will eventually make a mistake. The pipeline makes that mistake structurally impossible.

This pipeline template — load, split, audit, ColumnTransformer, Pipeline, fit, evaluate — is the starting point for every single model in this course. In the next notebook, we will learn how to properly *measure* model quality using regression metrics like MAE and RMSE, and we will establish baseline models that give us a floor to beat. See you there."

> **Show:** Cell 35 (Submission Instructions) briefly, then Cell 36 (Thank You).

---

### Option B: Three Shorter Videos (~4-5 minutes each)

---

#### Video 1: "Why Pipelines? Data Loading and Auditing" (~5 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–0:45` | Show header + objectives | Cell 0, Cell 1 | "Welcome back. In notebook 01 we split our data and learned about leakage. Today we build the tool that makes safe preprocessing automatic — the scikit-learn Pipeline. Here are our five learning objectives." (Read them briefly.) |
| `0:45–1:30` | Run setup | Cell 2, Cell 3 | "Standard setup — same as every notebook. Notice the new imports: Pipeline, ColumnTransformer, StandardScaler, SimpleImputer, OneHotEncoder. Each has a specific role in the preprocessing chain. Random seed 474 for reproducibility." |
| `1:30–2:30` | Load + split | Cell 5, Cell 6, Cell 8 | "California Housing dataset, same as notebook 01. 20,000 rows, 8 numeric features, target is median house value. We split 60-20-20 — training, validation, test. All preprocessing decisions will be based on the training set only." |
| `2:30–4:15` | Data audit | Cell 10, Cell 11, Cell 14 | "Before any transformation, we audit. This function checks data types, missing values, and unique counts for every column. On training: all float64, zero missing. We run it on validation and test too — same schema, no surprises. In your project, this is where you will discover problems: string columns, missing data, weird cardinality." |
| `4:15–5:00` | Exercise 1 prompt | Cell 13 | "Pause the video now and complete Exercise 1. Run the data audit on all three splits. Document any differences you observe. Come back when you are done." |

---

#### Video 2: "Building the Preprocessing Pipeline" (~5 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–0:30` | Recap | — | "Welcome back. We have our data loaded, split, and audited. Now we build the preprocessing pipeline — the core skill of this notebook." |
| `0:30–1:30` | Identify column types | Cell 16, Cell 17 | "First, we identify which columns are numeric and which are categorical. We use `select_dtypes` so the code adapts automatically to any dataset. Here: 8 numeric, 0 categorical. Your project will likely have both." |
| `1:30–3:30` | Build ColumnTransformer | Cell 19, Cell 20 | "The ColumnTransformer routes each column group through its own mini-pipeline. Numeric columns: impute with median, then scale to mean-zero and unit-variance. Categorical columns: impute with a constant string, then one-hot encode. Why median instead of mean? Robust to outliers. Why include the categorical path when we have no categorical columns? Because this is a template — when you use a different dataset, it just works." |
| `3:30–5:00` | Full pipeline + scores | Cell 22, Cell 23 | "Now we chain the ColumnTransformer with LinearRegression into one Pipeline object. One call to fit — imputer, scaler, and model all train on the training data only. One call to score — validation data is transformed with training statistics, then predicted. Train R-squared about 0.60, validation about 0.62. No overfitting, but only 60 percent of variance explained. That is our baseline — we will beat it in future notebooks." |

---

#### Video 3: "Pipeline Inspection, Exercises & Wrap-Up" (~4 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–1:00` | Inspect pipeline | Cell 25, Cell 26 | "Let's look inside the pipeline. `get_feature_names_out` shows every column that exits the ColumnTransformer. The `num__` prefix traces each feature back to its transformer. 8 in, 8 out — just rescaled. When you have categorical columns, you will see `cat__` prefixed one-hot columns here too." |
| `1:00–2:00` | Exercise 2 prompt + results | Cell 28, Cell 29, Cell 30 | "Pause and complete Exercise 2 — swap median imputation for mean and compare scores. You will find the scores are identical. Why? Because there are no missing values, so the imputer is a no-op either way. The lesson: we include the imputer not for *this* dataset, but for the day your data has gaps." |
| `2:00–3:00` | Checklist | Cell 33 | "Before you submit, run through the Pipeline Done Right Checklist. Four categories: pre-fitting, fitting, evaluation, code quality. Did you split before building? Did you fit only on train? Did you evaluate on validation? Can you explain every step? This checklist applies to every notebook going forward." |
| `3:00–4:00` | Wrap-Up + closing | Cell 34, Cell 35 | "Two rules to remember. One: fit only on training data. Two: put everything inside the pipeline. This template — load, split, audit, ColumnTransformer, Pipeline, fit, evaluate — is the backbone of the entire course. Next notebook, we learn how to properly measure model quality with regression metrics and baseline models. Submit your notebook to Brightspace and complete the quiz. See you there." |

---

*This guide was created to support video lecture recording for MGMT 47400 — Predictive Analytics, Purdue University.*
