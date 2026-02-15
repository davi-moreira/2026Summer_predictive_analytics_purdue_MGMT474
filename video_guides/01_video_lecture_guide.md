# Video Lecture Guide: Notebook 01 — Launchpad: Welcome, Predictive Analytics Fundamentals, and Data Workflow

## At a Glance

This guide covers:

1. **Why NB01 exists** — It is the course launchpad: it orients students to the tools (Google Colab, Gemini), introduces the conceptual foundations of predictive analytics (statistical learning, leakage, bias-variance), and establishes the data workflow (EDA + train/val/test splitting) that every subsequent notebook depends on

2. **Why it is the first notebook** — Students cannot preprocess, model, or evaluate anything until they understand *what* predictive analytics is, *how* to use the course platform, and *why* they must split data before touching it. NB01 provides all three prerequisites in a single session

3. **Why it must come before NB02** — NB02 assumes students can load data, perform basic EDA, create a 60/20/20 split, and explain what data leakage is. Without NB01, the pipeline mechanics in NB02 would be disconnected from their purpose

4. **Why each library/tool is used:**
   - `pandas` / `numpy` — data manipulation and numeric computation
   - `matplotlib` / `seaborn` — visualization for EDA
   - `sklearn.model_selection.train_test_split` — reproducible 60/20/20 splitting
   - `sklearn.datasets.fetch_california_housing` — clean, real-world regression dataset with known properties
   - Google Colab — zero-setup cloud environment
   - Google Gemini — AI coding assistant with the "Ask, Verify, Document" discipline

5. **Key concepts to explain on camera** — the statistical learning framework, data leakage, bias-variance trade-off, EDA checklist, and train/val/test splitting

6. **Common student questions** with answers (e.g., why three splits instead of two? why seed 474? why not just use training error?)

7. **Connection to the full course arc** — how the vocabulary, workflow, and dataset from NB01 are reused in all 19 subsequent notebooks

8. **Suggested video structure** — both a multi-segment single-session format (~5 videos, 10-12 min each) and a shorter-video alternative with speaking prompts, timestamps, and notebook cell references

---

## Purpose of This Document

This guide is your **lecture companion** for recording the video(s) that accompany `01_launchpad_eda_splits.ipynb`. It explains the *why* behind every design choice in the notebook — why the course begins here, why these specific concepts are introduced first, why the California Housing dataset, and why EDA and splitting precede any modeling. Use it as speaking notes, not as a script: the goal is to internalize the reasoning so you can explain it naturally on camera.

NB01 is the largest notebook in the course (57 cells). It covers more ground than any other single notebook because it must simultaneously orient students to the platform, the theory, and the workflow. The video structure section accounts for this by recommending multiple videos rather than a single 12-minute recording.

---

## 1. Why This Notebook Exists

Notebook 01 is the **foundation layer** for the entire course. Before students can build a preprocessing pipeline (NB02), evaluate a model (NB03), engineer features (NB04), or tune hyperparameters (NB05-onward), they need three things:

1. **Platform fluency.** They need to know how to open a notebook in Google Colab, run cells, save their work, and use Gemini responsibly. Without this, they will spend future notebooks fighting the tooling instead of learning the content.

2. **Conceptual vocabulary.** They need to understand what predictive analytics is (Y = f(X) + epsilon), the difference between supervised and unsupervised learning, what data leakage means, why training error is misleading, and how bias and variance compete. This vocabulary is referenced in every subsequent notebook without re-explanation.

3. **Data workflow.** They need to load a dataset, perform exploratory data analysis (data types, missingness, distributions, correlations), and create a proper train/validation/test split. This exact sequence — load, EDA, split — is the opening act of every modeling notebook in the course.

NB01 delivers all three in a single, carefully sequenced session. It ends with a promise: *"Split first, preprocess second, model third."* NB02 picks up that promise and teaches students how to preprocess safely.

---

## 2. Why It Is the First Notebook

The ordering is deliberate. NB01 must be first because:

| What NB01 Provides | Why It Cannot Wait |
|---|---|
| **Colab setup and Gemini orientation** | If students cannot run code or understand the AI assistant policy, they cannot engage with any notebook. This must be Day 1 content. |
| **Statistical learning framework (Y = f(X) + epsilon)** | Every model in the course is an attempt to estimate f(X). Without this framing, students would see each model as an isolated technique rather than a specific strategy for the same underlying problem. |
| **Data leakage definition** | Leakage prevention drives the entire course architecture (split before preprocess, fit on train only, lock the test set). If students do not understand leakage in NB01, the pipeline design in NB02 makes no sense. |
| **Bias-variance trade-off** | This concept explains *why* we validate, *why* we regularize, *why* we compare simple and complex models. It is referenced in every modeling notebook from NB03 onward. |
| **EDA checklist** | Students need a systematic approach to understanding data before transforming it. NB02's data audit function formalizes what NB01 teaches manually. |
| **Train/val/test splitting** | Every notebook from NB02 onward begins with a split. Students must understand *why* three sets, *why* 60/20/20, and *why* the test set is a lockbox. |

In short: NB01 is the launchpad. Every concept it introduces is consumed by at least one — and usually many — subsequent notebooks. Skipping or deferring any of these topics would create gaps that compound as the course progresses.

---

## 3. Why It Must Come Before Notebook 02

Notebook 02 introduces **preprocessing pipelines** using `Pipeline` and `ColumnTransformer`. Its learning objectives assume students already know how to:

- Open and run a notebook in Google Colab
- Load the California Housing dataset and understand its features
- Explain what data leakage is and why it is dangerous
- Create a 60/20/20 train/val/test split using `train_test_split`
- Perform basic EDA (data types, missing values, distributions)

Every one of these skills is taught in NB01 and consumed in NB02 without re-explanation. The deliberate sequence is:

```
NB01: What is predictive analytics? How do we set up our tools?
      What is leakage? How do we split data? How do we explore data?
            |
NB02: How do we preprocess data safely? (Pipeline + ColumnTransformer)
            |
NB03: How do we measure model quality? (Metrics + baselines)
```

If NB01 were skipped, students in NB02 would not understand:
- *Why* the pipeline fits only on training data (because they would not know what leakage is)
- *Why* the data is split 60/20/20 (because they would not know the purpose of validation vs. test sets)
- *Why* the data audit matters (because they would not have done manual EDA first)

NB01 provides the *why*; NB02 provides the *how*.

---

## 4. Why These Specific Libraries and Tools

### 4.1 Google Colab

**What it does:** Provides a free, cloud-based Jupyter notebook environment with pre-installed Python libraries, GPU access, and Google Drive integration.

**Why we use it:**

1. **Zero setup.** Students do not need to install Python, manage virtual environments, or troubleshoot OS-specific package issues. They open a browser and start coding. This eliminates the most common barrier to entry in data science courses.

2. **Consistent environment.** Every student runs the same Python version with the same library versions. When the instructor's output shows R-squared = 0.604, every student sees the same number. No "it works on my machine" problems.

3. **Built-in AI assistant.** Google Gemini is integrated directly into the Colab interface, making the "Ask, Verify, Document" pattern natural rather than requiring students to context-switch to a separate tool.

**What to emphasize on camera:** Colab is not just a convenience — it is a pedagogical choice. By removing setup friction, we spend 100% of class time on predictive analytics instead of debugging installation errors. Show students how to open the notebook via the Colab badge, how to run cells (Shift+Enter), how to save to Drive, and how to restart the runtime when things go wrong.

### 4.2 Google Gemini (AI Coding Assistant)

**What it does:** An AI assistant embedded in Google Colab that can generate code, explain errors, and answer conceptual questions.

**Why we use it:**

1. **Realistic preparation.** In industry, data scientists use AI assistants daily. Teaching students to use Gemini responsibly prepares them for how work actually gets done.

2. **Scaffolded learning.** Students can ask Gemini to draft boilerplate code (imports, standard plots) so they can focus on understanding the logic rather than memorizing syntax.

3. **Accountability through documentation.** The "Ask, Verify, Document" pattern teaches students that AI-generated code must be understood, tested, and annotated before submission. This builds critical thinking habits.

**What to emphasize on camera:** Gemini is a tool, not a tutor. If a student submits Gemini-generated code without understanding it, they have learned nothing. The three-step pattern — Ask (get a draft), Verify (run it and check the output), Document (add your own comments explaining why the code works) — is the discipline that separates responsible AI use from copy-paste.

### 4.3 `pandas` and `numpy`

**What they do:** `pandas` provides DataFrames for tabular data manipulation (loading, filtering, grouping, summarizing). `numpy` provides array operations and mathematical functions.

**Why we use them:**

- They are the standard data manipulation stack in Python. Every data science job posting expects fluency in pandas.
- `pandas` makes EDA natural: `.info()`, `.describe()`, `.isnull().sum()`, `.corr()` — each method maps directly to a step in the EDA checklist.
- `numpy` provides `np.random.seed()` for reproducibility and efficient numeric operations.

**What to emphasize on camera:** In this notebook, pandas is used for loading data, inspecting data types, computing missing value counts, and generating descriptive statistics. These are not one-off tasks — they are the EDA checklist that students will repeat in every notebook.

### 4.4 `matplotlib` and `seaborn`

**What they do:** `matplotlib` is the foundational plotting library. `seaborn` builds on matplotlib with statistical visualization defaults (better colors, easier heatmaps, distribution plots).

**Why we use them:**

- Visualization is how students *see* patterns in data. The histogram of MedHouseVal reveals the right-skew and the cap at 5.0. The correlation heatmap reveals that MedInc dominates. The scatter plot of Income vs. House Value makes Y = f(X) + epsilon tangible.
- `seaborn.set_style('whitegrid')` provides clean, publication-quality defaults so students do not waste time on formatting.

**What to emphasize on camera:** Every plot in this notebook exists to answer a specific question. The target distribution histogram answers: "Is the target normally distributed?" The feature distribution grid answers: "Do any features have extreme outliers?" The correlation heatmap answers: "Which features are most related to the target?" Teach students to *read* visualizations, not just generate them.

### 4.5 `sklearn.model_selection.train_test_split`

**What it does:** Splits arrays or DataFrames into random subsets, with control over the split ratio and random seed.

**Why we use it:**

1. **Reproducibility.** Setting `random_state=474` ensures the same rows end up in train, validation, and test every time the notebook runs. Students get identical results, and debugging becomes possible.

2. **Two-stage splitting.** The notebook uses a two-call pattern: first split 80/20 (temp/test), then split the temp set 75/25 (train/val). This produces the 60/20/20 ratio. Students learn that `test_size=0.25` applied to the 80% temp set yields 0.25 * 0.80 = 0.20 of the original data.

3. **Foundation for NB02.** Once students have `X_train`, `X_val`, `X_test`, `y_train`, `y_val`, `y_test`, they are ready to pass these directly into the preprocessing pipeline in the next notebook.

**What to emphasize on camera:** The test set is a **lockbox**. Students create it in NB01 and do not touch it again until the final evaluation. All model development, hyperparameter tuning, and feature engineering use only the train and validation sets. This discipline prevents overfitting to the test set and ensures a fair final evaluation.

### 4.6 `sklearn.datasets.fetch_california_housing`

**What it does:** Loads the California Housing dataset — 20,640 census block-groups with 8 numeric features and 1 continuous target (median house value in $100,000s).

**Why we use it:**

1. **Right size and complexity.** Large enough to demonstrate real patterns (20K rows, 8 features), small enough to run instantly on Colab without GPU. No data download, no file management — one function call.

2. **Pure regression.** The target is continuous, which aligns with the course's first-week focus on regression. Classification comes in Week 2.

3. **Realistic imperfections.** The data has outliers (AveOccup with extreme values), a capped target (MedHouseVal maxes at 5.0), and strong multicollinearity (AveRooms and AveBedrms at r = 0.85). These imperfections create natural teaching moments throughout the course.

4. **Reused across notebooks.** The same dataset appears in NB01 through NB05 (and beyond), so students build cumulative familiarity rather than learning a new dataset every day.

**What to emphasize on camera:** This dataset was chosen for its pedagogical properties, not because it is the most exciting. Housing prices are intuitive — every student understands that income, location, and house size affect price. This intuitive grounding lets students focus on methodology rather than domain knowledge.

---

## 5. Key Concepts to Explain on Camera

### 5.1 The Statistical Learning Framework: Y = f(X) + epsilon

This is the conceptual backbone of the entire course. Every model students will build — linear regression, decision trees, random forests, gradient boosting, neural networks — is an attempt to estimate the unknown function f(X).

**What to say:** "The equation Y = f(X) + epsilon is deceptively simple, but it frames everything we do. Y is what we want to predict — house prices, spam labels, customer churn. X is the information we use to predict — income, email text, purchase history. f is the true relationship between X and Y that we are trying to learn. And epsilon is noise — randomness we can never eliminate no matter how good our model is."

**Analogy for students:** "Think of f as the signal and epsilon as the static. A good radio (model) captures the signal clearly. A bad radio distorts it. But even the best radio in the world cannot eliminate the static — that is the irreducible error."

### 5.2 Data Leakage: Why It Matters More Than Any Algorithm

NB01 introduces two types of leakage — target leakage and train-test contamination. This concept is arguably the single most important idea in the course because it determines whether evaluation metrics can be trusted.

**What to say:** "Data leakage is the silent killer of predictive models. Your model will show excellent performance during development — high accuracy, low error — and you will deploy it thinking it works. Then in production, it collapses. Why? Because during development, information from the future or from the test set leaked into your training process. The model was cheating without you knowing."

**Analogy for students:** "Imagine studying for an exam using the answer key. You will score 100% on that exam, but you have not actually learned anything. When you face a different exam on the same material, you fail. That is leakage — artificially inflated performance that does not generalize."

### 5.3 The Bias-Variance Trade-off

This is the theoretical foundation for model selection, regularization, cross-validation, and ensemble methods — topics that span Weeks 1 through 4.

**What to say:** "Every model makes errors for one of two reasons. Bias: the model is too simple and misses real patterns — that is underfitting. Variance: the model is too complex and memorizes noise — that is overfitting. The trade-off is that reducing one tends to increase the other. Our job as data scientists is to find the sweet spot where total error — bias squared plus variance plus irreducible noise — is minimized."

**Analogy for students:** "Bias is like a clock that is consistently 10 minutes fast — it is wrong in a predictable, systematic way. Variance is like a clock that is sometimes 5 minutes fast and sometimes 5 minutes slow — it is unreliable. Ideally, you want a clock that is both accurate (low bias) and consistent (low variance)."

### 5.4 EDA as a Professional Habit

The EDA section (Sections 6.1-6.6 of the notebook) is not just a checklist exercise — it is professional practice. Students should internalize the habit of auditing data types, checking for missing values, examining distributions, and inspecting correlations *before* any modeling.

**What to say:** "Before you build any model, you need to know your data. What types are the columns? Are any values missing? Are the distributions skewed? Are any features highly correlated with each other? These questions are not optional — skipping them leads to models that silently fail. In this notebook, we go through the full checklist manually. In notebook 02, we will automate parts of it with a reusable function."

### 5.5 Train/Validation/Test Splitting: The Three-Way Discipline

The 60/20/20 split is not arbitrary — it reflects a deliberate balance between having enough data to train, enough to tune, and enough to evaluate.

**What to say:** "We split into three sets, not two, because we need two separate safeguards. The validation set lets us tune our model — try different hyperparameters, compare different algorithms — without contaminating our final evaluation. The test set is the lockbox — we touch it exactly once, at the very end, to get an honest estimate of how well our model generalizes. If we used only train and test, every time we adjusted our model based on test performance, we would be leaking test information into our modeling decisions."

---

## 6. What Students Should Take Away

After watching your videos and completing this notebook, students should be able to answer these questions confidently:

1. **"What is predictive analytics?"** — It is the practice of using historical data to learn patterns (Y = f(X) + epsilon) and make predictions about future or unknown outcomes. The two main types are regression (continuous target) and classification (categorical target).

2. **"What is data leakage and why does it matter?"** — Leakage occurs when information from outside the training set contaminates the model. It causes evaluation metrics to be misleadingly optimistic, leading to models that fail in production. The two types are target leakage (a feature encodes the outcome) and train-test contamination (test data influences training).

3. **"Why do we split into three sets instead of two?"** — Training data fits the model, validation data tunes the model and compares alternatives, test data provides a final unbiased evaluation. If we used only two sets, every model adjustment based on test performance would leak test information into our decisions.

4. **"What is the bias-variance trade-off?"** — Test error decomposes into bias-squared (underfitting — model too simple), variance (overfitting — model too complex), and irreducible noise. Increasing model complexity reduces bias but increases variance. The goal is to find the sweet spot that minimizes total test error.

5. **"What should I check during EDA before building any model?"** — Data types (are columns numeric or categorical?), missing values (how many and where?), target distribution (skewed? capped?), feature distributions (outliers?), and correlations (which features predict the target? which are redundant?).

---

## 7. Common Student Questions (Anticipate These)

**Q: "Why do we use RANDOM_SEED = 474 instead of the common 42?"**
A: The seed value 474 corresponds to the course number (MGMT 474). Any fixed integer would work — the important thing is that the seed is constant so results are reproducible. We chose 474 as a memorable, course-specific convention.

**Q: "Why 60/20/20 and not 70/15/15 or 80/10/10?"**
A: 60/20/20 balances three needs: enough training data to learn patterns (~12,400 samples), enough validation data for reliable hyperparameter tuning (~4,100 samples), and enough test data for a trustworthy final evaluation (~4,100 samples). With 20,640 total rows, 60/20/20 is generous for all three purposes. Smaller datasets might warrant different ratios, but for this course we standardize on 60/20/20 for consistency.

**Q: "The MedHouseVal target is capped at 5.0. Won't that cause problems?"**
A: Yes, and that is intentional as a teaching point. The cap means the model can never learn the true value for homes worth more than $500,000. This introduces ceiling bias — a real-world data quality issue. You will see this manifest as a cluster of points at 5.0 in the target distribution histogram. In a production setting, you might remove capped values or treat them separately.

**Q: "Why do we use `test_size=0.25` in the second split instead of `test_size=0.20`?"**
A: Because the second split operates on the *remaining* 80% of the data (after the test set was removed). 25% of 80% = 20% of the original data. So the validation set is 20% of the full dataset, as intended. This two-step approach is the standard pattern for creating three sets from `train_test_split`, which natively supports only binary splits.

**Q: "Why does the notebook spend so much time on theory before any coding?"**
A: Because predictive analytics is not a coding exercise — it is a thinking exercise. The code is a tool for implementing ideas. If you do not understand what data leakage is, no amount of scikit-learn expertise will save you from building a model that cheats. The theory in Sections 4.1-4.9 provides the mental framework that makes the hands-on coding in Sections 5-7 meaningful.

**Q: "What is the difference between supervised and unsupervised learning?"**
A: In supervised learning, you have labeled data — both the features X and the outcome Y — and your goal is to learn the mapping from X to Y. In unsupervised learning, you have only features X (no labels) and your goal is to discover structure (clusters, patterns, anomalies). This course focuses on supervised learning because most business prediction problems have a known target variable.

**Q: "AveRooms and AveBedrms have a correlation of 0.85. Should I drop one?"**
A: Not necessarily. High correlation means the features carry redundant information, but dropping a feature is a modeling decision that depends on the algorithm. Linear regression may suffer from inflated coefficient variance (multicollinearity), but tree-based models handle correlated features naturally. In NB05, regularized models (Ridge, LASSO) will address multicollinearity automatically. For now, keep both features and note the issue.

---

## 8. Connection to the Broader Course Arc

| Week | Notebooks | What NB01's Foundations Enable |
|---|---|---|
| **Week 1** | 01-05 | NB01's split workflow is reused in NB02-05. Leakage awareness drives pipeline design (NB02). Bias-variance trade-off motivates regularization (NB05). EDA checklist becomes a reusable function (NB02). |
| **Week 2** | 06-10 | Statistical learning framework extends to classification (Y is categorical). Leakage concepts apply identically to classification metrics (accuracy, AUC). Colab fluency enables faster engagement with new model types. |
| **Week 3** | 11-15 | Bias-variance trade-off explains why trees overfit (high variance) and why ensembles (Random Forest, Gradient Boosting) reduce variance. Flexibility-interpretability spectrum guides model selection discussions. |
| **Week 4** | 16-20 | Full predictive modeling workflow (9 steps introduced in NB01 Section 4.4) is executed end-to-end in the final project. Students deploy models they first conceptualized in NB01. |

**NB01 is the conceptual spine of the entire course.** Every modeling decision students make — which algorithm to use, how to tune it, how to evaluate it, how to deploy it — connects back to a concept first introduced in this notebook.

---

## 9. Suggested Video Structure

Below are two recording options. NB01 is substantially larger than other notebooks (57 cells covering welcome, logistics, theory, EDA, and splitting), so the recommended approach is **Option B: five shorter videos** of 8-12 minutes each. Option A provides a condensed three-video alternative.

All cell references use the format `Cell N` where N is the zero-indexed cell number in `01_launchpad_eda_splits.ipynb`.

---

### Option A: Three Videos (~10-12 minutes each)

#### Video 1: "Welcome, Tools, and the Statistical Learning Framework" (~12 min)

##### Segment 1 — Course Welcome & Instructor Introduction `[0:00-2:00]`

> **Show:** Cell 0 (header with course logo and Colab badge) and Cell 1 (instructor introduction).

**Say:**

"Welcome to MGMT 474 Predictive Analytics. I am Professor Davi Moreira, and this is our first notebook — the launchpad for the entire course. Over the next four weeks, you will learn to build, evaluate, and deploy predictive models using Python and scikit-learn. Every concept, every line of code, every modeling decision builds on what we establish today. This notebook covers a lot of ground — tools, theory, and hands-on practice — because we need a solid foundation before we can move fast in the notebooks that follow."

> **Show:** Cell 2 (student introduction PAUSE-AND-DO). Briefly mention the participation assignment.

**Say:**

"Before we dive in, take a moment for the student introduction exercise. This is your number zero participation assignment — check Brightspace for details. Introduce yourself: your background, your experience with Python, and what you hope to learn. When you are done, come back and we will set up our tools."

##### Segment 2 — Course Logistics `[2:00-3:00]`

> **Show:** Cell 3 (course syllabus and logistics).

**Say:**

"Let me highlight the key logistics. This is a 4-week intensive course with 20 business days. Each day, you will watch micro-videos — no longer than 12 minutes each — and work through a Google Colab notebook with guided exercises. Grading includes participation, quizzes, homework, a midterm, and a final project that you build incrementally over the course. The official syllabus with full details is on Brightspace and the course website. The important thing to know right now: this course is hands-on from the very first day. You learn predictive analytics by *doing* predictive analytics."

##### Segment 3 — Google Colab Setup `[3:00-5:30]`

> **Show:** Cell 4 (Why Google Colab), then Cell 5 (Colab navigation and workflow).

**Say:**

"We use Google Colab as our primary platform. Why? Zero installation. No Python setup, no package management, no environment conflicts. You open a browser, click the Colab badge, and you are coding. Colab also gives us free GPU access for later notebooks, cloud storage through Google Drive, and — importantly — Google Gemini integration for AI-assisted coding."

> **Show:** Cell 5 on screen. Point to the keyboard shortcuts.

**Say:**

"Here are the essential shortcuts you need to know. Shift plus Enter runs a cell and moves to the next one — this is the one you will use most. Control plus Enter, or Command plus Enter on Mac, runs the cell but stays in place. Use Runtime, Run All to execute the entire notebook top to bottom, which you should do before submitting any assignment to make sure everything works."

> **Show:** Cell 6 (notebook conventions).

**Say:**

"Every notebook in this course follows the same conventions. The first code cell is always a setup cell with imports, random seed, and display settings. Random seed is always 474 — our course number — so every student gets identical results. Exercises are marked with the PAUSE-AND-DO label. Critical rules appear in blockquotes with warning symbols. And when code runs successfully, you will see a checkmark confirmation."

> **Show:** Cell 7 (Gemini: Ask, Verify, Document pattern).

**Say:**

"Now, an important topic: using Google Gemini responsibly. Gemini is an AI coding assistant built into Colab. It can write code, explain errors, and answer questions. But here is the rule — and this is non-negotiable — you are responsible for every line of code you submit. The pattern is Ask, Verify, Document. Ask Gemini to draft code. Verify that the code runs correctly and that you understand what it does. Document by adding your own comments explaining the logic in your own words. If you cannot explain what a line does, you do not understand it, and you should not submit it."

> **Show:** Cell 8 (setup test introduction), then **run** Cell 9 (setup test code).

**Say:**

"Let us test your environment. Run this cell — you should see Python and library versions, a simple sine wave plot, and a setup confirmation. If you see any errors, use Gemini to help debug. The plot is just a sanity check — if it renders, your environment is working."

> **Run** Cell 9. Point to the `Setup complete!` output and the test plot.

##### Segment 4 — What Is Predictive Analytics? `[5:30-8:00]`

> **Show:** Cell 10 (Section 4.1: What is Predictive Analytics?).

**Say:**

"Now let us get into the substance. Predictive analytics is the practice of extracting patterns from historical data to make predictions about future or unknown outcomes. It is everywhere — spam filters that protect your inbox, recommendation engines that suggest what to watch next, credit scoring models that decide whether you get a loan, medical diagnosis tools that detect disease from imaging. The common thread: all of these take historical data where the outcome is known, learn patterns, and apply those patterns to new cases where the outcome is unknown."

> **Show:** Cell 11 (Section 4.2: Statistical Learning Framework, Y = f(X) + epsilon).

**Say:**

"At the core of predictive analytics is this equation: Y equals f of X plus epsilon. Y is what you want to predict — the target variable. X is the information you use to predict — the features. f is the true relationship between X and Y, which is unknown and is what we are trying to learn from data. And epsilon is irreducible noise — randomness that no model, no matter how sophisticated, can eliminate. Think of f as the signal and epsilon as the static. A good model captures the signal. A bad model either distorts the signal with bias or amplifies the static with variance."

> **Show:** Cell 12 (scatter plots: MedHouseVal vs. MedInc, HouseAge, AveRooms). **Run** the cell.

**Say:**

"Here is the equation in action, using the California Housing dataset. Each plot shows a different slice of Y equals f of X plus epsilon. Look at the leftmost panel — Median Income versus House Value. There is a clear upward trend: higher income neighborhoods have higher house values. That trend is f of X. But the points are scattered around the trend — that scatter is epsilon. Now look at the middle panel, House Age — the relationship is much weaker. And Average Rooms on the right — there is a weak positive relationship but lots of noise. This tells us that Median Income will likely be the strongest predictor in our models."

> **Run** Cell 14 (zoomed scatter with linear fit line). Point to the trend line.

**Say:**

"This zoomed-in view shows Median Income versus House Value with a linear trend line. The line captures the general direction of the relationship, but it misses the curvature — notice how the relationship flattens above income of about 10. This tells us that a simple linear model might underfit this data. More flexible models might capture that curvature better. We will explore this trade-off throughout the course."

##### Segment 5 — Supervised vs. Unsupervised and the Workflow `[8:00-10:00]`

> **Show:** Cell 15 (Section 4.3: Supervised vs. Unsupervised Learning).

**Say:**

"Machine learning tasks fall into two broad categories. Supervised learning — which is the focus of this course — means you have labeled data. You know both the features X and the outcome Y for your training observations. The goal is to learn the mapping from X to Y so you can predict Y for new observations. There are two types: regression, where Y is a continuous number like house price or sales forecast, and classification, where Y is a category like spam versus legitimate or approve versus deny. Unsupervised learning, in contrast, means you have only features — no labels. The goal is to discover structure in the data, like clusters of similar customers. We will not cover unsupervised learning in depth, but it is useful to understand where it fits in the landscape."

> **Show:** Cell 16 (Section 4.4: End-to-End Predictive Modeling Workflow).

**Say:**

"Here is the big picture — the end-to-end predictive modeling workflow. There are nine core steps, from defining the business problem to deploying and monitoring the model. Each step maps to specific notebooks in this course. Today we cover steps one through four: define the problem, collect and load data, explore the data with EDA, and split into train, validation, and test sets. In notebook 02, we tackle step five: preprocessing. In notebook 03, step six: evaluation. And so on through the course, building one layer at a time. By the end, you will have executed this entire workflow on your final project dataset."

##### Segment 6 — Closing Video 1 `[10:00-12:00]`

> **Show:** Cell 17 (Section 4.5: Data Leakage preview — do not go deep, just introduce).

**Say:**

"Before we move on, let me plant one critical idea that we will explore in detail in the next video: data leakage. Leakage is when information that would not be available at prediction time sneaks into your model during training. It makes your evaluation metrics look incredible — and then your model fails catastrophically in production. There are two types: target leakage, where a feature encodes the outcome you are trying to predict, and train-test contamination, where information from the test set influences your training process. Keep this concept in mind — it drives almost every design decision in this course, starting with how we split data in the next video."

---

#### Video 2: "Foundational Concepts: Leakage, Bias-Variance, and Model Selection" (~12 min)

##### Segment 1 — Data Leakage Deep Dive `[0:00-3:00]`

> **Show:** Cell 17 (Section 4.5: Data Leakage — full content).

**Say:**

"Welcome back. In this video, we cover the foundational concepts that guide every modeling decision in this course. Let us start with data leakage — the silent killer of predictive models. Leakage occurs when your model gets access to information during training that it would not have when making real predictions. The danger is that your model will show excellent performance on your validation data, you will deploy it, and then it will collapse because the leaked information is not available in production."

**Say:**

"There are two types. Type one: target leakage. This happens when a predictor variable contains information that is derived from or influenced by the target variable. For example, if you are predicting whether a patient will be hospitalized and one of your features is 'total hospital charges' — that feature exists *because* the patient was hospitalized. It perfectly predicts the outcome, but it would not be available at the time you need to make the prediction. Type two: train-test contamination. This happens when information from the validation or test sets influences your training process. The most common form is preprocessing leakage — for example, computing the mean of a feature on the entire dataset before splitting, so the scaler's parameters include information from the test set. In notebook 02, we will build a Pipeline that makes this type of leakage structurally impossible."

##### Segment 2 — Train vs. Test Error `[3:00-5:00]`

> **Show:** Cell 18 (Section 4.6: Assessing Model Accuracy). Point to the MSE formula and the figure showing overfitting.

**Say:**

"How do we know if our model is any good? The naive approach is to compute the prediction error on the training data. But training error is misleadingly optimistic. As model complexity increases, training error always decreases — even if the model is overfitting. Look at this figure: the flexible spline fits every single training point perfectly, giving a training error of zero. But it is memorizing noise, not learning patterns. On new data, it would perform terribly."

**Say:**

"The better approach: evaluate on held-out data that the model has never seen. That is what the validation and test sets are for. Training error tells you how well the model *memorizes*. Validation and test error tell you how well the model *generalizes*. This distinction is the reason we split our data into three sets, and we will enforce it rigorously throughout the course."

##### Segment 3 — Curse of Dimensionality `[5:00-6:30]`

> **Show:** Cell 19 (Section 4.7: Curse of Dimensionality). Point to the figures.

**Say:**

"Here is a common misconception: more features should mean better predictions. In reality, adding features without adding proportionally more data can hurt performance. This is the curse of dimensionality. In one dimension, to capture 10 percent of the data you need a small interval. In two dimensions, you need a much larger region. As dimensions increase, data becomes sparse, and finding similar observations becomes exponentially harder. The practical implication: do not throw every available feature into your model. Feature selection and dimensionality reduction are essential skills, and we will cover them later in the course."

##### Segment 4 — Flexibility vs. Interpretability `[6:30-8:00]`

> **Show:** Cell 20 (Section 4.8: Flexibility vs. Interpretability). Point to the spectrum figure.

**Say:**

"Should we use simple models or complex models? It depends on your goals. On one end of the spectrum, linear regression gives you coefficients that directly quantify feature importance — easy to explain to stakeholders. On the other end, neural networks can capture incredibly complex patterns, but they are black boxes — hard to explain why they made a specific prediction. In business contexts, interpretability often matters as much as accuracy. If you cannot explain to a regulator or a VP why your model denied a loan, the model is not useful no matter how accurate it is. This course covers models across the entire spectrum, and we will discuss when to use each."

##### Segment 5 — The Bias-Variance Trade-off `[8:00-12:00]`

> **Show:** Cell 21 (Section 4.9: Bias-Variance Trade-off). This is the longest and most important subsection.

**Say:**

"Now the most important concept in predictive modeling — the bias-variance trade-off. Test error decomposes into three components: bias squared, variance, and irreducible error."

> Point to the mathematical formula.

**Say:**

"Bias is the error from wrong assumptions. If you use a straight line to model a curved relationship, the line will systematically miss the curvature. That is high bias — underfitting. The model is too rigid."

> Point to the linear model figure (high bias).

**Say:**

"Variance is the error from sensitivity to training data. If you use a very flexible model — say a high-degree polynomial — it will fit the training data perfectly, but if you train it on a slightly different sample, it will give wildly different predictions. That is high variance — overfitting. The model is too flexible."

> Point to the flexible spline figure (high variance).

**Say:**

"The trade-off: as you increase model complexity, bias decreases — you fit the training data better. But variance increases — you become more sensitive to the specific training sample. The sweet spot is where the total test error is minimized."

> Point to the U-shaped test error curve figure.

**Say:**

"Look at this figure carefully. The gray curve is training error — it always goes down as complexity increases. The red curve is test error — it goes down initially as the model captures real patterns, reaches a minimum, then goes up as overfitting kicks in. That minimum is where we want to be. Every tool we will learn — cross-validation, regularization, ensemble methods, early stopping — is a strategy for finding and staying at that minimum."

> Point to the decomposition figure (bias squared + variance + irreducible error).

**Say:**

"This decomposition figure shows it most clearly. Blue is bias squared — decreasing with complexity. Orange is variance — increasing with complexity. The dotted line is irreducible error — constant, the floor we can never go below. Their sum — the red curve — has a U shape. Our job is to find the bottom of that U."

**Say:**

"Every modeling decision in this course connects back to this trade-off. Why do we validate? To estimate where we are on the U-curve. Why do we regularize? To move left on the curve when we have overfit. Why do we use ensemble methods? To reduce variance without increasing bias. Keep this framework in your head — it is the lens through which you should evaluate every model you build."

---

#### Video 3: "Hands-On EDA and Train/Val/Test Splitting" (~12 min)

##### Segment 1 — Business Case and Data Loading `[0:00-2:30]`

> **Show:** Cell 22 (Section 5: Hands-On Practice introduction and business case).

**Say:**

"Welcome back. We have covered the theory — now let us apply it. Here is our business case: you are a data analyst at a California real estate investment firm. The firm identifies undervalued properties by comparing predicted market values to listing prices. Currently, manual appraisals cost $500 to $1,000 per property and take weeks. You want a predictive model that estimates median house values instantly from census data."

> **Run** Cell 23 (load California Housing dataset). Point to the shape and column list.

**Say:**

"We load the California Housing dataset from scikit-learn — 20,640 rows and 9 columns. Eight features and one target. The target is MedHouseVal — median house value in units of $100,000. The features include median income, house age, average rooms, population, latitude, and longitude. Notice that the code also saves the data to a CSV file — this demonstrates data persistence, a practical skill for real projects."

> **Show:** Cell 24 (identify features and target), then **run** Cell 25 (dataset overview).

**Say:**

"Before any analysis, we explicitly identify our target variable and feature set. The target is MedHouseVal — what we want to predict. The eight remaining columns are features — what we use to predict. This seems obvious, but in real projects, misidentifying the target or accidentally including it as a feature is a common source of leakage. Always be explicit about which column is Y and which columns are X."

##### Segment 2 — EDA Checklist: Types, Missingness, Descriptive Stats `[2:30-5:30]`

> **Run** Cell 27 (data types audit). Point to the output.

**Say:**

"Step one of EDA: check data types. All 9 columns are float64 — this is ideal for scikit-learn, which expects numeric inputs. In your project datasets, you will likely find object columns — strings that represent categories — which need encoding. The `df.info()` call also confirms 20,640 non-null entries in every column, meaning no missing values. Let us verify that explicitly."

> **Run** Cell 30 (missing values check).

**Say:**

"Step two: check for missing values. Zero missing values across all columns. This is unusual — most real-world datasets have missing data. We will still learn to handle missingness in notebook 02, because your project data almost certainly will have gaps. But for this dataset, we can proceed knowing the data is complete."

> **Run** Cell 33 (descriptive statistics).

**Say:**

"Step three: descriptive statistics. The `describe` table reveals several important patterns. First, the scale differences are enormous — MedInc ranges from about 0.5 to 15, while Population ranges from 3 to over 35,000. This tells us we will need to scale features before using models that are sensitive to scale, like linear regression with regularization. Second, look at AveOccup — the max is 1,243 people per household. That is clearly a data artifact, not a real value. Outliers like this are common in real data and need to be handled carefully. Third, the target MedHouseVal has a maximum of exactly 5.001 — the values are capped at 5.0, meaning any house worth more than $500,000 is recorded as $500,000. That cap will limit our model's ability to predict at the high end."

##### Segment 3 — EDA Checklist: Target and Feature Distributions `[5:30-8:00]`

> **Run** Cell 36 (target distribution: histogram and box plot).

**Say:**

"Step four: visualize the target distribution. Two key observations. The histogram on the left shows a right-skewed distribution — most house values cluster between $100,000 and $250,000, with a long tail toward higher values. There is also a spike at $500,000 — that is the cap we noticed in the descriptive statistics. About 800 observations are piled up at that ceiling. The box plot on the right confirms the skew and shows outlier whiskers. For modeling, this skew means our predictions will be less accurate for expensive homes."

> **Run** Cell 39 (feature distributions: 3x3 histogram grid).

**Say:**

"Step five: visualize feature distributions. Each panel shows one feature. MedInc is right-skewed with a long tail — a few very wealthy neighborhoods pull the distribution right. HouseAge has a spike at the maximum — census data likely caps house age. Population and AveOccup have extreme outliers — those are the data artifacts we saw in the descriptive statistics. Latitude and Longitude show bimodal patterns that correspond to the two major urban areas in California: the San Francisco Bay Area and the Los Angeles basin."

> **Run** Cell 42 (correlation heatmap).

**Say:**

"Step six: correlation analysis. The heatmap reveals the linear relationships between all variables. The strongest predictor of house value is MedInc with a correlation of about 0.69 — income and house price move together, as economic intuition suggests. AveRooms and AveBedrms are highly correlated with each other at 0.85 — they carry largely redundant information, which we call multicollinearity. No feature has a suspiciously high correlation with the target — no leakage red flags from this analysis. The bottom line: MedInc will likely dominate any linear model, and we will need non-linear methods to extract predictive signal from the other features."

> **Mention:** "Pause the video here and complete Exercise 1." Point to Cell 50 (PAUSE-AND-DO 1).

##### Segment 4 — Train/Val/Test Splitting `[8:00-10:30]`

> **Show:** Cell 44 (Section 7: Why Three Splits?).

**Say:**

"Now the critical step that connects everything we have learned. We split the data into three sets: train, validation, and test. The training set — 60 percent — is where the model learns patterns. The validation set — 20 percent — is where we tune hyperparameters and compare models. The test set — 20 percent — is the lockbox. We touch it exactly once, at the very end, to get an honest estimate of generalization performance."

> **Run** Cell 45 (split code). Point to the two-step split and the percentages.

**Say:**

"The split uses scikit-learn's train_test_split with two calls. First, we split 80-20 to separate the test set. Then we split the remaining 80 percent into 75-25, which gives us 60 percent train and 20 percent validation of the original data. The random state is 474 — our course seed — so every student gets the same split. Notice the output: about 12,400 training samples, 4,100 validation, and 4,100 test. Each set is large enough to be representative."

> **Run** Cell 47 (target distribution across splits). Point to the three histograms.

**Say:**

"Sanity check: do the three splits have similar target distributions? Yes — the three histograms look nearly identical. Same shape, same skew, same cap at 5.0, similar means and medians. This is what we want — a random split should produce statistically similar subsets. If one split looked dramatically different, something would be wrong with our splitting process."

> **Show:** Cell 48 (leakage sniff test explanation), then **run** Cell 49 (leakage checklist code).

**Say:**

"Finally, we run a leakage sniff test. This checklist confirms that we split before any preprocessing, that there is no overlap between the splits, that no feature suspiciously encodes the target. All checks pass. In your projects, this is a step you should never skip."

##### Segment 5 — Exercise 2, Wrap-Up, and Closing `[10:30-12:00]`

> **Mention:** "Pause the video here and complete Exercise 2." Point to Cell 52 (PAUSE-AND-DO 2).

**Say:**

"Pause the video and complete Exercise 2 — identify three potential leakage risks for real estate prediction. Think about whether each feature would be available at the time you need to make a prediction. Come back when you are done."

> **Show:** Cell 54 (Wrap-Up: Key Takeaways). Read through the 10 takeaways.

**Say:**

"Let me leave you with the key takeaways. Today we covered a lot of ground — from Colab setup and Gemini usage, to the statistical learning framework, to data leakage, bias-variance, EDA, and splitting. But three ideas matter most. First: Y equals f of X plus epsilon — every model is an attempt to estimate f while managing epsilon. Second: the bias-variance trade-off — every modeling decision balances underfitting and overfitting. Third: split first, preprocess second, model third — this order prevents leakage and ensures valid evaluation."

> **Show:** Cell 55 (submission instructions) briefly.

**Say:**

"To submit, complete both exercises, run all cells to make sure everything works, save the notebook, and upload to Brightspace. In the next notebook, we build our first preprocessing pipeline — the tool that makes safe preprocessing automatic. See you there."

> **Show:** Cell 56 (Thank You).

---

### Option B: Five Shorter Videos (~8-12 minutes each)

---

#### Video 1: "Welcome, Course Setup, and Google Colab" (~8 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `[0:00-1:30]` | Show header + instructor intro | Cell 0, Cell 1 | "Welcome to MGMT 474 Predictive Analytics. I am Professor Davi Moreira. This is our first notebook — the launchpad for the entire course. Over four weeks, you will learn to build, evaluate, and deploy predictive models. Every concept builds on what we establish today." |
| `[1:30-2:30]` | Student introductions | Cell 2 | "Pause here and complete the student introduction — this is your number zero participation assignment. Introduce yourself, your background, your experience with data, and what you hope to learn. Check Brightspace for details." |
| `[2:30-3:30]` | Course logistics | Cell 3 | "Key logistics: 4-week intensive, 20 business days, micro-videos plus Colab notebooks each day. Grading includes participation, quizzes, homework, a midterm, and a final project you build incrementally. Full syllabus is on Brightspace and the course website." |
| `[3:30-5:00]` | Why Colab + navigation | Cell 4, Cell 5 | "We use Google Colab — zero installation, consistent environment, free GPU access. Open the notebook via the Colab badge. Essential shortcut: Shift plus Enter to run a cell. Use Runtime, Run All to test the full notebook before submitting." |
| `[5:00-6:00]` | Notebook conventions | Cell 6 | "Every notebook follows the same pattern. Setup cell first, random seed 474 for reproducibility, PAUSE-AND-DO exercises for practice, blockquotes for critical rules, checkmarks for confirmations." |
| `[6:00-7:00]` | Gemini: Ask, Verify, Document | Cell 7 | "Google Gemini is your AI coding assistant. The rule: Ask, Verify, Document. Ask Gemini to draft code. Verify it runs and you understand it. Document with your own comments. If you cannot explain a line of code, do not submit it." |
| `[7:00-8:00]` | Run setup test | Cell 8, Cell 9 | "Let us test your environment. Run this cell — you should see library versions, a sine wave plot, and a setup confirmation. If anything fails, use Gemini to debug. The plot is a sanity check — if it renders, you are good to go." |

---

#### Video 2: "What Is Predictive Analytics? The Statistical Learning Framework" (~10 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `[0:00-2:00]` | What is predictive analytics? | Cell 10 | "Predictive analytics extracts patterns from historical data to predict future outcomes. Spam filters, recommendation engines, credit scoring, medical diagnosis — all use the same principle: learn from labeled data, predict on new data. The equation at the heart of it all: Y equals f of X plus epsilon." |
| `[2:00-4:00]` | Statistical learning framework | Cell 11 | "Y is what we predict — house prices, spam labels. X is what we use to predict — income, email text. f is the unknown relationship we are trying to learn. And epsilon is noise — randomness we can never eliminate. Think of f as the signal and epsilon as the static. A good model captures the signal without amplifying the static." |
| `[4:00-5:30]` | Run scatter plots: Y = f(X) + epsilon | Cell 12, Cell 13 | "Here is the equation in action. Three scatter plots, three features. Median Income shows a clear upward trend — that is f of X. The scatter around the trend — that is epsilon. House Age and Average Rooms show weaker relationships. Median Income will likely dominate our models." |
| `[5:30-6:30]` | Run zoomed Income vs. House Value plot | Cell 14 | "Zooming in: the linear trend line captures the general direction, but it misses the curvature. The relationship flattens at higher incomes. A simple linear model might underfit here. We will explore more flexible models throughout the course." |
| `[6:30-8:00]` | Supervised vs. unsupervised | Cell 15 | "Two categories of machine learning. Supervised: you have labeled data — features and outcomes — and you learn the mapping. Two subtypes: regression for continuous targets, classification for categorical targets. Unsupervised: you have only features, no labels, and you discover structure like clusters. This course focuses on supervised learning." |
| `[8:00-10:00]` | End-to-end workflow | Cell 16 | "The big picture: nine steps from business problem to deployed model. Today we cover steps one through four — define the problem, load data, explore with EDA, and split. Notebook 02 covers preprocessing, notebook 03 covers evaluation, and so on. By week 4, you will have executed this entire workflow on your final project." |

---

#### Video 3: "Critical Concepts: Leakage, Bias-Variance, and Model Selection" (~12 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `[0:00-2:30]` | Data leakage | Cell 17 | "Data leakage is the silent killer of predictive models. It makes your evaluation look amazing and your production model fail. Two types: target leakage — a feature encodes the outcome, like using hospital charges to predict hospitalization. Train-test contamination — information from test data influences training, like computing a scaler's mean on the full dataset before splitting. In notebook 02, we build a Pipeline that makes contamination structurally impossible." |
| `[2:30-4:30]` | Train vs. test error | Cell 18 | "How do we know if a model is good? Not by training error — that always decreases with complexity, even when overfitting. Look at this figure: the flexible curve fits every training point perfectly but would fail on new data. The right approach: evaluate on held-out data. Training error measures memorization. Validation and test error measure generalization. This is why we split." |
| `[4:30-6:00]` | Curse of dimensionality | Cell 19 | "More features should mean better predictions, right? Not necessarily. In high dimensions, data becomes sparse. To capture 10 percent of data in 1D, you need a small interval. In 2D, a much larger region. In 100 dimensions, you would need nearly the entire space. The lesson: do not throw every available feature into your model. Feature selection matters." |
| `[6:00-7:30]` | Flexibility vs. interpretability | Cell 20 | "Simple models like linear regression give you interpretable coefficients — easy to explain to stakeholders. Complex models like neural networks capture subtle patterns but are black boxes. In business, interpretability often matters as much as accuracy. If you cannot explain to a regulator why your model denied a loan, the model is not useful. This course covers models across the entire spectrum." |
| `[7:30-12:00]` | Bias-variance trade-off | Cell 21 | "The most important concept in predictive modeling. Test error equals bias squared plus variance plus irreducible error. Bias: the model is too simple and misses real patterns — underfitting. Variance: the model is too complex and memorizes noise — overfitting. As complexity increases, bias decreases but variance increases. The sweet spot is the minimum of test error — the bottom of the U-shaped curve. Every tool in this course — cross-validation, regularization, ensemble methods — is a strategy for finding that sweet spot. Look at the decomposition figure: blue bias decreasing, orange variance increasing, their sum forming the U-shape. That U-curve is the conceptual backbone of every modeling decision we will make." |

---

#### Video 4: "Hands-On EDA: California Housing Dataset" (~10 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `[0:00-1:30]` | Business case + data loading | Cell 22, Cell 23 | "Time for hands-on practice. Business case: you are a data analyst at a California real estate investment firm predicting house values to identify undervalued properties. We load the California Housing dataset — 20,640 rows, 8 features, 1 target. The target is MedHouseVal in units of $100,000." |
| `[1:30-2:30]` | Identify features and target | Cell 24, Cell 25 | "Before any analysis, explicitly identify your target and features. Target: MedHouseVal. Features: everything else — income, house age, rooms, bedrooms, population, occupancy, latitude, longitude. Misidentifying the target is a common source of leakage. Always be explicit." |
| `[2:30-4:00]` | Data types + missingness | Cell 26, Cell 27, Cell 30 | "EDA step one: data types. All float64 — ideal for scikit-learn. Step two: missing values. Zero across all columns. Unusual for real-world data, but your project datasets will likely have gaps. We will handle that in notebook 02." |
| `[4:00-5:30]` | Descriptive statistics | Cell 32, Cell 33 | "Step three: descriptive statistics. Three key observations. Scale differences are enormous — income ranges to 15, population ranges to 35,000. We will need scaling. AveOccup has a maximum of 1,243 people per household — a data artifact. And the target is capped at 5.0 — any house above $500,000 is recorded as 500,000." |
| `[5:30-7:00]` | Target distribution | Cell 35, Cell 36 | "Step four: target distribution. Right-skewed with a spike at the $500,000 cap — about 800 observations are piled at the ceiling. The box plot confirms the skew and outliers. Our predictions will be less accurate for expensive homes because of this cap." |
| `[7:00-8:30]` | Feature distributions | Cell 38, Cell 39 | "Step five: feature distributions. Income is right-skewed. House age has a spike at the max — likely a data cap. Population and AveOccup have extreme outliers. Latitude and Longitude show bimodal patterns — the two major California metro areas: San Francisco Bay and Los Angeles." |
| `[8:30-10:00]` | Correlation analysis | Cell 41, Cell 42, Cell 43 | "Step six: correlations. MedInc is the strongest predictor at r equals 0.69. AveRooms and AveBedrms are correlated with each other at 0.85 — redundant information, multicollinearity. No suspiciously high correlation with the target — no obvious leakage. Bottom line: income dominates, and we will need non-linear methods to extract signal from the other features." |

---

#### Video 5: "Train/Val/Test Splitting, Exercises, and Wrap-Up" (~10 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `[0:00-1:30]` | Why three splits? | Cell 44 | "The final critical step: splitting. Three sets, not two. Training set — 60 percent — fits the model. Validation set — 20 percent — tunes hyperparameters and compares models. Test set — 20 percent — lockbox for final evaluation. Why not just two? Because every time you adjust your model based on test performance, you leak test information into your decisions." |
| `[1:30-3:00]` | Run the split | Cell 45 | "Two calls to train_test_split. First, 80-20 to carve out the test set. Then 75-25 on the remaining 80 percent to create train and validation. 25 percent of 80 percent equals 20 percent of the original. Random state 474. Result: about 12,400 train, 4,100 validation, 4,100 test. Each large enough to be representative." |
| `[3:00-4:30]` | Split sanity checks | Cell 46, Cell 47 | "Sanity check: do the splits have similar target distributions? Three histograms — nearly identical shapes, similar means and medians. A random split produces statistically similar subsets. If one looked dramatically different, something would be wrong." |
| `[4:30-5:30]` | Leakage sniff test | Cell 48, Cell 49 | "Leakage sniff test: we confirm we split before preprocessing, no overlap between splits, no suspicious features encoding the target. All checks pass. Never skip this step in your projects." |
| `[5:30-6:30]` | PAUSE-AND-DO Exercise 1 | Cell 50, Cell 51 | "Pause the video and complete Exercise 1. Review all the EDA outputs above and summarize three key findings — unusual distributions, strong correlations, potential data quality issues. Write your answers in the cell provided." |
| `[6:30-7:30]` | PAUSE-AND-DO Exercise 2 | Cell 52, Cell 53 | "Now complete Exercise 2. Identify three potential leakage risks for real estate prediction. Think about whether each feature would be available at prediction time. Would a feature like 'days on market' be available before the house is listed? That is the kind of question to ask." |
| `[7:30-9:00]` | Wrap-up: key takeaways | Cell 54 | "Ten takeaways from today. But three ideas matter most. One: Y equals f of X plus epsilon — every model estimates f while managing noise. Two: the bias-variance trade-off — every decision balances underfitting and overfitting. Three: split first, preprocess second, model third — this order prevents leakage. These three ideas will guide every notebook that follows." |
| `[9:00-10:00]` | Submission + closing | Cell 55, Cell 56 | "To submit: complete both exercises, run all cells, save the notebook, upload to Brightspace. Next notebook, we build our first preprocessing pipeline — the tool that makes safe preprocessing automatic. The promise from today — split first, preprocess second, model third — notebook 02 delivers on that promise. See you there." |

---

*This guide was created to support video lecture recording for MGMT 47400 — Predictive Analytics, Purdue University.*
