# Video Lecture Guide: Notebook 05 — Regularization (Ridge/Lasso) + Project Proposal Sprint

## At a Glance

This guide covers:

1. **Why NB05 exists** — It introduces regularization as the direct solution to the overfitting problem exposed in NB04, and it marks the Week 1 project milestone (project proposal due)

2. **Why it must follow NB04** — NB04 showed that polynomial features improve accuracy but widen the overfit gap. NB05 provides the tool (regularization) that controls that gap. Without NB04's demonstration of the problem, the solution would feel abstract

3. **Why it must precede NB06** — NB05 closes out the regression arc of the course (Week 1). NB06 pivots to classification (logistic regression). Students need a complete mental model of the regression workflow — pipeline, metrics, features, regularization — before switching to a new problem type

4. **Why each library/tool is used:**
   - `Ridge` and `RidgeCV` — L2 penalty that shrinks all coefficients uniformly, with built-in cross-validation for automatic alpha selection
   - `Lasso` and `LassoCV` — L1 penalty that can zero out coefficients entirely, performing automatic feature selection
   - `ElasticNet` — mentioned as the L1+L2 hybrid for completeness
   - Coefficient comparison bar chart — makes shrinkage and sparsity visually concrete
   - Project proposal template — structures the Week 1 deliverable

5. **Common student questions** with answers (e.g., Ridge vs Lasso, what alpha means, why scaling is required for regularization)

6. **Connection to the full course arc** — how regularization concepts from NB05 carry into logistic regression (Week 2), tree-based models (Week 3), and the final project

7. **Suggested video structure** — both single-video (~12 min) and three-video (~4-5 min each) formats with speaking prompts, timestamps, and notebook cell references

---

## Purpose of This Document

This guide is your **lecture companion** for recording the video(s) that accompany `05_regularization_project_proposal.ipynb`. It explains the *why* behind every design choice — why regularization is introduced on Day 5, why Ridge and Lasso are taught together, why cross-validated alpha selection is emphasized, and why the project proposal sprint is bundled into this notebook. Use it as speaking notes, not as a script: the goal is to internalize the reasoning so you can explain it naturally on camera.

---

## 1. Why This Notebook Exists

Notebook 04 ends with a dilemma: polynomial features of degree 2 raise R^2 from about 0.60 to approximately 0.65, but the overfit gap widens and the 44 coefficients become large and unstable. Students leave NB04 knowing the *problem* — too many features lead to overfitting — but not the *solution*.

Notebook 05 delivers the solution: **regularization**. By adding a penalty term to the loss function that discourages large coefficients, Ridge and Lasso regression stabilize the model and control overfitting without discarding features manually. Ridge shrinks all coefficients uniformly toward zero; Lasso can drive some coefficients to exactly zero, performing automatic feature selection.

This notebook also carries a second responsibility: it is the **Week 1 capstone**. The project proposal is due on Day 5, so the second half of the notebook provides a structured proposal template covering dataset description, predictive task, evaluation plan, leakage risk assessment, and success criteria. This dual purpose — technical content plus project milestone — reflects the course's design: every week ends with both a modeling skill and a project deliverable.

---

## 2. Why It Must Come After Notebook 04

Notebook 04 establishes three experiences that notebook 05 *consumes*:

| NB04 Concept | How NB05 Uses It |
|---|---|
| **Polynomial feature explosion (8 to 44 features)** | NB05 opens by recalling that adding features improved R^2 but widened the overfit gap. Regularization is presented as the direct remedy: keep the polynomial features, let Ridge/Lasso manage the coefficients. |
| **Unstable coefficient magnitudes** | NB04's coefficient bar chart showed that OLS assigns large weights to correlated polynomial terms. NB05's grouped bar chart (OLS vs Ridge vs Lasso) shows those same coefficients after shrinkage — visibly smaller and more stable. |
| **Overfit gap as a diagnostic** | NB04 taught students to compute (train R^2 - val R^2) and flag models with large gaps. NB05 asks students to compare overfit gaps across OLS, Ridge, and Lasso, confirming that regularization reduces the gap. |

**In short:** NB04 creates the problem (overfitting from feature expansion), NB05 delivers the solution (regularized regression). The two notebooks form a deliberate problem-solution pair.

If you tried to teach regularization *before* NB04, students would not understand why anyone would want to shrink coefficients. The penalty term would feel like arbitrary math with no practical motivation. If you skipped NB05 entirely, students would have no tool to control complexity when they build polynomial or high-dimensional models in their projects.

---

## 3. Why It Must Come Before Notebook 06

Notebook 06 pivots the course from **regression** to **classification**. It introduces logistic regression, probability thresholds, and confusion matrices using the Breast Cancer Wisconsin dataset. Its learning objectives assume students already understand:

- That regularization controls overfitting by penalizing coefficient magnitude
- That cross-validation can automatically select the regularization strength
- That scaling is a prerequisite for regularization to work fairly
- That the pipeline pattern (preprocessing + model) applies to any estimator

NB05 completes the regression toolkit. After NB05, students have seen the full regression arc: load data, split, preprocess (NB02), evaluate with metrics and baselines (NB03), engineer features (NB04), and control complexity with regularization (NB05). This complete mental model transfers to classification: NB06 follows the identical pattern but with a different model class (`LogisticRegression`), different metrics (accuracy, log loss), and a different dataset.

The deliberate sequence is:

```
NB04: How do we improve features? (Polynomial expansion + diagnostics)
            |     [Problem: overfitting]
            |
NB05: How do we control complexity? (Ridge, Lasso, CV alpha selection)
            |     [Solution: regularization + Week 1 project milestone]
            |
NB06: New problem type: Classification (Logistic regression + pipelines)
            [Students transfer the complete regression workflow to a new domain]
```

NB05 is the capstone of the regression arc. It must be fully absorbed before classification begins.

---

## 4. Why These Specific Libraries and Tools

### 4.1 `sklearn.linear_model.Ridge` and `sklearn.linear_model.RidgeCV`

**What it does:** Ridge regression adds an L2 penalty to the OLS loss function: Loss = RSS + alpha * sum(beta_j^2). The penalty shrinks all coefficients toward zero but never sets any to exactly zero. `RidgeCV` wraps this in k-fold cross-validation, automatically selecting the alpha that minimizes CV error from a user-supplied grid.

**Why we use it:**

1. **It addresses the exact problem from NB04.** Polynomial features created correlated columns with inflated coefficients. Ridge penalizes large coefficients, stabilizing them without discarding features. Students see the direct connection: NB04 showed the problem, NB05 shows the fix.

2. **It is the simplest regularizer.** The L2 penalty has a closed-form solution (unlike Lasso), making it computationally fast and numerically stable. This makes it the natural starting point before introducing Lasso's more aggressive L1 penalty.

3. **`RidgeCV` teaches automated hyperparameter selection.** Instead of manually guessing alpha, students learn to let cross-validation choose for them. This pattern — define a grid, let CV search — generalizes to every hyperparameter in every model.

**What to emphasize on camera:** Ridge never zeros out a feature. It shrinks all coefficients uniformly. If you believe every feature has at least some relevance, Ridge is appropriate. If you suspect many features are irrelevant, you need Lasso.

### 4.2 `sklearn.linear_model.Lasso` and `sklearn.linear_model.LassoCV`

**What it does:** Lasso regression adds an L1 penalty: Loss = RSS + alpha * sum(|beta_j|). Unlike L2, the L1 penalty can shrink coefficients to exactly zero, effectively performing automatic feature selection. `LassoCV` performs cross-validated alpha selection.

**Why we use it:**

1. **It introduces sparsity.** Setting coefficients to zero means Lasso selects which features matter and which do not. This is powerful in high-dimensional settings where many features are noise.

2. **It complements Ridge.** Teaching both side by side highlights the L1-vs-L2 distinction: shrinkage (Ridge) vs. shrinkage + selection (Lasso). The grouped bar chart makes this visually obvious.

3. **It previews real-world feature selection.** In students' projects, datasets may have dozens of features. Lasso gives them an automatic way to identify which features drive prediction and which can be safely ignored.

**What to emphasize on camera:** Lasso's zero-coefficient property is its unique advantage. But it can be unstable when features are highly correlated: it may arbitrarily pick one correlated feature and zero out the others. When you have groups of correlated features, consider ElasticNet or Ridge instead.

### 4.3 `sklearn.linear_model.ElasticNet`

**What it does:** Combines L1 and L2 penalties: Loss = RSS + alpha * [l1_ratio * sum(|beta_j|) + (1-l1_ratio) * sum(beta_j^2)]. The `l1_ratio` parameter controls the balance.

**Why we mention it:**

- **Completeness.** Students should know the full regularization family: Ridge (pure L2), Lasso (pure L1), and ElasticNet (hybrid). This prevents the misconception that Ridge and Lasso are the only options.
- **Practical cases.** ElasticNet is useful when you want some feature selection (L1) but also need stability with correlated features (L2). It appears in the markdown but is not extensively coded in this notebook.

**What to emphasize on camera:** ElasticNet is a compromise. If you are unsure whether to use Ridge or Lasso, ElasticNet lets you use both. The `l1_ratio` parameter tunes the blend.

### 4.4 Coefficient Comparison Bar Chart (grouped bars: OLS vs Ridge vs Lasso)

**What it does:** Displays three bars per feature, side by side, showing how each regularization method changed the coefficients relative to unregularized OLS.

**Why we use it:**

- **It makes shrinkage tangible.** Students can see that Ridge bars are shorter than OLS bars (uniform shrinkage) and that Lasso bars may be zero for some features (sparsity).
- **It builds on NB04's single-bar chart.** NB04 showed one set of coefficients. NB05 shows three sets in the same plot, teaching comparative visualization.
- **It communicates to non-technical audiences.** A stakeholder who cannot parse equations can immediately see "Ridge makes everything smaller, Lasso makes some things disappear."

**What to emphasize on camera:** Point to a feature where Lasso's bar is zero but Ridge's is not. Explain: "Lasso decided this feature is not worth keeping. Ridge kept it but made it small. This is the L1-vs-L2 difference in one picture."

### 4.5 Project Proposal Template

**What it does:** Provides a structured six-section template for the Week 1 project milestone: dataset description, predictive task, evaluation plan, leakage risk assessment, initial concerns, and success criteria.

**Why we include it:**

- **Week 1 is ending.** The proposal is due on Day 5 (this notebook). Students need a clear structure so they can draft their proposal in the time remaining.
- **It reinforces course concepts.** The template asks for a metric rationale (NB03), a split strategy (NB01-02), a baseline approach (NB03), and a leakage risk assessment (NB01). Each section connects to something students have learned.
- **It prevents late-stage project disasters.** Students who choose an inappropriate dataset or target now — before investing weeks of work — can course-correct early.

**What to emphasize on camera:** The proposal is not busy work. It forces you to answer the hardest question in any project: "What exactly am I predicting, for whom, and how will I know if my model is good enough?" Getting this right now saves weeks of wasted effort.

---

## 5. Key Concepts to Explain on Camera

### 5.1 Regularization as "Adding a Tax on Complexity"

The most accessible explanation: OLS tries to minimize prediction error with no constraint on coefficient size. Regularization adds a "tax" on large coefficients. The model must now balance two objectives: fit the data well (low RSS) and keep coefficients small (low penalty). The alpha parameter controls how heavy the tax is.

**Analogy for students:** Imagine you are packing for a trip with a weight limit on your luggage (the penalty). Without the limit, you would bring everything. With a strict limit, you bring only the essentials. Ridge is like a weight limit that makes you pack lighter versions of everything. Lasso is like a weight limit that makes you leave some items behind entirely.

### 5.2 Why Scaling Is Mandatory for Regularization

The penalty term sums coefficient magnitudes (L1) or squared magnitudes (L2). If features have different scales — income in $10,000s vs. latitude in degrees — their coefficients will naturally have different magnitudes. The penalty would unfairly penalize the coefficient on the large-scale feature. Scaling all features to the same range (mean 0, std 1) ensures the penalty treats all features equally.

**Analogy for students:** If you weigh luggage in kilograms for some items and pounds for others, the weight limit is unfair. Convert everything to the same unit first. StandardScaler is the unit converter.

### 5.3 Alpha Selection via Cross-Validation

Alpha is the single most important hyperparameter in regularized regression. Too small: essentially no regularization (reverts to OLS). Too large: all coefficients shrink to zero (reverts to the mean predictor). `RidgeCV` and `LassoCV` search a grid of alpha values using k-fold cross-validation and return the one that minimizes CV error.

**Key point for students:** You never hand-pick alpha. You let the data tell you. The CV procedure tests each candidate alpha on held-out folds and picks the winner. This is your first encounter with automated hyperparameter tuning — a pattern you will use with every model in this course.

### 5.4 Shrinkage vs. Sparsity — The L1/L2 Distinction

- **Ridge (L2):** Shrinks all coefficients by a constant factor. No coefficient reaches exactly zero. All features remain in the model. Good when you believe every feature contributes at least a little.
- **Lasso (L1):** Shrinks coefficients and can set some to exactly zero. Performs automatic feature selection. Good when you believe many features are noise.

The geometric intuition (diamond vs. circle constraint regions) is optional at this level, but the practical distinction is essential: Ridge keeps everything, Lasso discards.

### 5.5 Project Framing as a Discipline

The project proposal section is not a detour from the technical content — it is the most important applied skill in the course. Choosing the wrong dataset, target, or metric at the start leads to weeks of wasted effort. The template forces students to articulate their prediction task, justify their metric, and identify leakage risks *before* writing any code.

---

## 6. What Students Should Take Away

After watching your video and completing this notebook, students should be able to answer these questions confidently:

1. **"Why does regularization improve generalization?"** — Because it penalizes large coefficients, preventing the model from fitting noise in the training data. The penalty forces simpler models that transfer better to unseen data.

2. **"What is the difference between Ridge and Lasso?"** — Ridge shrinks all coefficients toward zero but never eliminates them. Lasso can set coefficients to exactly zero, performing feature selection. Use Ridge when all features are potentially relevant; use Lasso when many features may be irrelevant.

3. **"Why must I scale features before regularization?"** — Because the penalty treats all coefficients equally. If features have different scales, the penalty would unfairly target the feature with the larger-scale coefficient, regardless of its true importance.

4. **"How do I choose alpha?"** — Use cross-validation (RidgeCV, LassoCV). Define a grid of candidate alpha values, let k-fold CV evaluate each one, and select the alpha with the lowest CV error. Never hand-pick alpha.

5. **"Why is my project proposal important?"** — Because it forces you to define exactly what you are predicting, for whom, with what data, and how you will evaluate success. A clear proposal prevents you from spending weeks on a doomed project.

---

## 7. Common Student Questions (Anticipate These)

**Q: "Ridge and OLS give nearly identical R^2 on this dataset. Why bother with Ridge?"**
A: On this clean 8-feature dataset, regularization has little room to improve because OLS already has a small overfit gap. Ridge's value becomes dramatic when you add polynomial features (44 columns) or use high-dimensional datasets (hundreds of features). Think of Ridge as insurance: it costs almost nothing when not needed but prevents catastrophe when it is.

**Q: "Lasso kept all 8 features — I thought it was supposed to zero some out?"**
A: Lasso zeros out features when alpha is large enough. The CV-selected alpha on this dataset was small, meaning light regularization was sufficient. With more features (e.g., the 44 polynomial features from NB04), Lasso would almost certainly zero out several. Try feeding the polynomial-expanded data into LassoCV and you will see sparsity.

**Q: "What does the alpha value actually mean? How do I interpret alpha = 1.5?"**
A: Alpha controls the strength of the penalty. Alpha = 0 means no penalty (reverts to OLS). Larger alpha means more shrinkage. The specific number (1.5 vs. 3.0 vs. 10.0) has no intrinsic meaning — it depends on the scale of your data and the number of features. That is why we let CV choose it automatically rather than interpreting the number directly.

**Q: "When should I use ElasticNet instead of Ridge or Lasso?"**
A: When you have groups of correlated features. Lasso tends to pick one feature from a correlated group and zero out the others, which can be arbitrary. ElasticNet's L2 component encourages the model to share the weight among correlated features while the L1 component still provides some sparsity. If you are unsure, start with Ridge or Lasso; try ElasticNet if the results are unstable.

**Q: "For my project proposal, my dataset has 50 features. Should I use Lasso to select features first?"**
A: That is a reasonable approach. Lasso can help you identify which features matter. But do not commit to the selected features before exploring them. Run the full pipeline, inspect which features Lasso keeps, verify they make domain sense, and iterate. Feature selection is a tool, not a final answer.

---

## 8. Connection to the Broader Course Arc

| Week | Notebooks | What NB05's Regularization & Project Proposal Enable |
|---|---|---|
| **Week 1** | 01-05 | NB05 closes the regression arc. Students now have the complete workflow: split (NB01), preprocess (NB02), evaluate (NB03), engineer features (NB04), regularize (NB05). This arc is the reference model for all future modeling cycles. The project proposal sets the foundation for all subsequent milestones. |
| **Week 2** | 06-10 | Logistic regression (NB06) uses the same regularization concepts: scikit-learn's `LogisticRegression` applies L2 regularization by default (controlled by parameter C, the inverse of alpha). Students who understand Ridge from NB05 can immediately transfer the concept to classification. |
| **Week 3** | 11-15 | Tree-based models have their own complexity controls (max_depth, min_samples_leaf) that serve the same purpose as alpha. Students trained on the regularization mindset from NB05 recognize these hyperparameters as different implementations of the same "tax on complexity" principle. |
| **Week 4** | 16-20 | The final project requires students to justify their regularization choices. NB05's coefficient comparison chart becomes a template for the project report. The project proposal from Day 5 has evolved through three milestones into the final deliverable. |

**NB05 is the Week 1 capstone.** It completes the regression toolkit and launches the semester-long project. Every subsequent week builds on both.

---

## 9. Suggested Video Structure

Below are two recording options. Each segment includes the **speaking prompt** (what to say), **timestamps**, and the **notebook cells** to show on screen.

All cell references use the format `Cell N` where N is the zero-indexed cell number in `05_regularization_project_proposal.ipynb`.

---

### Option A: Single Video (~12 minutes)

#### Segment 1 — Opening & Motivation `[0:00–1:30]`

> **Show:** Cell 0 (header) and Cell 1 (learning objectives) on screen.

**Say:**

"Welcome back. In the last notebook we tried polynomial features and saw a frustrating tradeoff: R-squared went up, but the overfit gap widened. We had 44 features, many of them correlated and noisy, and the model started memorizing training data. Today we fix that problem. Regularization adds a penalty on coefficient size, forcing the model to keep things simple even when it has many features to work with. We will cover Ridge regression, which shrinks all coefficients uniformly, and Lasso regression, which can zero out irrelevant features entirely. Both use cross-validation to pick the right penalty strength automatically.

This is also the last day of Week 1, so the second part of the notebook walks through your project proposal — due today. By the end, you will have the complete regression toolkit: preprocessing, metrics, feature engineering, and regularization. Everything you need to start your project."

> **Action:** Scroll through the 5 learning objectives in Cell 1, briefly reading each one aloud.

---

#### Segment 2 — Setup & Data `[1:30–2:30]`

> **Show:** Cell 2 (setup), then **run** Cell 2 (imports).

**Say:**

"Standard setup. New imports: Ridge, RidgeCV, Lasso, LassoCV, and ElasticNet from sklearn.linear_model. The CV variants include built-in cross-validation — they will search over a grid of alpha values and pick the best one for us. Same California Housing dataset, same 60-20-20 split, same seed 474."

> **Run** Cell 2. Point to `Setup complete!`.

> **Run** Cell 5 (load data + split). Point to the sizes.

**Say:**

"12,384 training, 4,128 validation, 4,128 test. Same as every notebook. The splits have not changed, so any performance difference we see is due to regularization, not data."

---

#### Segment 3 — Why Regularization? `[2:30–4:00]`

> **Show:** Cell 7 (regularization explanation markdown).

**Say:**

"Here is the key idea. Without regularization, linear regression has one goal: minimize prediction error on the training data. It does not care how large the coefficients get. If making a coefficient 10,000 reduces training error by a fraction, OLS will do it. But large coefficients mean the model is fitting noise — small random patterns in the training data that will not repeat in new data. That is overfitting.

Regularization adds a second goal: keep the coefficients small. The model now has to balance two objectives — fit the data well *and* keep things simple. The alpha parameter controls the balance. Small alpha: mostly care about fit, light penalty. Large alpha: mostly care about simplicity, heavy penalty. Zero alpha: no penalty at all, identical to OLS.

Ridge uses an L2 penalty — the sum of squared coefficients. It shrinks everything toward zero but never reaches zero. Think of it as turning down the volume on every feature equally. Lasso uses an L1 penalty — the sum of absolute values. It can push coefficients all the way to zero, effectively removing features. Think of it as turning off some speakers entirely."

---

#### Segment 4 — Ridge Regression with CV `[4:00–6:00]`

> **Show:** Cell 8 (Ridge explanation), then **run** Cell 9 (RidgeCV pipeline).

**Say:**

"We build a pipeline with StandardScaler and RidgeCV. The scaler is critical — regularization penalizes coefficient magnitude, and if features have different scales, the penalty is unfair. After scaling, all features are on the same footing.

RidgeCV takes a grid of 50 alpha values log-spaced from 0.001 to 1,000 and runs 5-fold cross-validation on each. The best alpha is printed here. The train and validation R-squared values should be very close to plain OLS, because this 8-feature dataset does not have a severe overfitting problem. Ridge's real power shows up when you have 44 polynomial features or hundreds of project features."

> **Mention:** "Pause the video here and complete Exercise 1 — interpret the chosen alpha and compare Ridge to the baseline." Point to Cell 11 (PAUSE-AND-DO 1).

---

#### Segment 5 — Lasso Regression with CV `[6:00–8:00]`

> **Show:** Cell 13 (Lasso explanation), then **run** Cell 14 (LassoCV pipeline).

**Say:**

"Now Lasso. Same pipeline pattern: StandardScaler plus LassoCV. Same alpha grid, same 5-fold CV. We set max_iter to 10,000 because Lasso's optimization can require more iterations to converge, especially at high alpha values.

The key output to watch is not just R-squared — it is the coefficient table."

> **Run** Cell 17 (Lasso coefficients + feature selection summary).

**Say:**

"Look at this table. Each feature has a Lasso coefficient, and at the bottom we count how many are non-zero. If Lasso zeroed out any features, it is telling you those features do not improve prediction enough to justify their complexity. Compare this to the OLS coefficients from notebook 04 — same features, but some are now smaller or gone. This is automatic feature selection.

On this 8-feature dataset, Lasso may keep all features because the CV-selected alpha is light. But feed it the 44 polynomial features from notebook 04, and it will start zeroing out the noisy interaction terms. That is where Lasso's sparsity really shines."

> **Mention:** "Pause the video and complete Exercise 2 — identify the top selected features and compare to Ridge." Point to Cell 16 (PAUSE-AND-DO 2).

---

#### Segment 6 — Comparison & Coefficient Visualization `[8:00–10:00]`

> **Run** Cell 21 (model comparison table: OLS vs Ridge vs Lasso).

**Say:**

"Here is the full comparison. Three models: Linear Regression with no penalty, Ridge, and Lasso. On this dataset, all three achieve similar validation R-squared — around 0.60. That is expected: with only 8 features and 12,000 samples, overfitting is minimal. The differences are in the details: alpha values, coefficient magnitudes, and the number of retained features."

> **Run** Cell 24 (grouped bar chart: OLS vs Ridge vs Lasso coefficients).

**Say:**

"This chart is the most important visualization in the notebook. Three bars per feature. Blue is OLS, orange is Ridge, green is Lasso. For most features, the Ridge bars are slightly shorter — that is uniform shrinkage. Some Lasso bars may be noticeably shorter or completely absent — that is sparsity. The visual difference between Ridge and Lasso is *the* takeaway: Ridge dims everything equally, Lasso turns off the weak signals.

When you report regularization results in your project, use a chart like this. It communicates the L1-vs-L2 story instantly to any audience."

---

#### Segment 7 — Project Proposal & Closing `[10:00–12:00]`

> **Show:** Cell 26 (project proposal builder introduction).

**Say:**

"Switching gears. This is the last day of Week 1, and your project proposal is due. The template in this notebook gives you six sections to fill out: dataset description, predictive task, evaluation plan, leakage risk assessment, initial concerns, and success criteria. Each section connects to something we have covered."

> **Show:** Cell 27 (project proposal template). Skim through the six sections.

**Say:**

"Dataset description: where does your data come from, how many samples and features? Predictive task: what are you predicting, what is one row, is it regression or classification? Evaluation plan: which metric, which split strategy, what baseline? Leakage risk assessment: list three ways information could leak from the future into your features, and how you will prevent each one. Initial concerns: data quality, missing values, class imbalance. Success criteria: what R-squared or MAE would make this model useful?

This is not busy work. The proposal forces you to answer the hardest question in any project before you write a single line of model code: what exactly am I predicting, for whom, and how will I know if it works?"

> **Show:** Cell 28 (Wrap-Up). Read the three critical rules.

**Say:**

"Three rules to close Week 1. One: always scale features before regularization. Two: use cross-validation to tune alpha — never hand-pick it. Three: start project planning early — dataset choice matters more than algorithm choice. The complete regression toolkit is now in your hands: pipeline, metrics, features, regularization. In Week 2, we take all of these ideas and apply them to classification. Logistic regression, confusion matrices, ROC curves — same pipeline pattern, new problem type. Submit your notebook and your project proposal to Brightspace. See you in notebook 06."

> **Show:** Cell 29 (Submission Instructions) briefly, then Cell 30 (Thank You).

---

### Option B: Three Shorter Videos (~4-5 minutes each)

---

#### Video 1: "Why Regularization? Ridge Regression" (~5 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–0:45` | Show header + objectives | Cell 0, Cell 1 | "Welcome back. Last notebook, polynomial features improved accuracy but increased overfitting. Today we fix that with regularization — a penalty on coefficient size that forces simpler models. Plus, your project proposal is due today. Here are our five learning objectives." (Read them briefly.) |
| `0:45–1:30` | Run setup + load data | Cell 2, Cell 5 | "New imports: Ridge, RidgeCV, Lasso, LassoCV, ElasticNet. Same California Housing data, same split, same seed. Any performance change is purely from regularization." |
| `1:30–3:00` | Explain regularization | Cell 7 | "OLS minimizes prediction error with no constraint on coefficient size. Regularization adds a penalty. Ridge uses L2: sum of squared coefficients. Shrinks everything toward zero but never to zero. Lasso uses L1: sum of absolute values. Can push coefficients to exactly zero — feature selection. Alpha controls penalty strength. The analogy: packing for a trip with a weight limit. Ridge makes you pack lighter. Lasso makes you leave some items behind." |
| `3:00–4:30` | Run RidgeCV | Cell 8, Cell 9 | "Pipeline: StandardScaler plus RidgeCV. Scaling is mandatory — without it, the penalty unfairly targets large-scale features. RidgeCV tests 50 alpha values with 5-fold CV. Best alpha is printed. Train and validation R-squared are very close to OLS on this 8-feature dataset. Ridge's real power shows on high-dimensional problems like the 44 polynomial features." |
| `4:30–5:00` | Exercise 1 prompt | Cell 11 | "Pause and complete Exercise 1. Interpret the chosen alpha. Compare Ridge to unregularized linear regression. Explain why Ridge might improve generalization. Come back when you are done." |

---

#### Video 2: "Lasso, Feature Selection, and Model Comparison" (~5 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–0:30` | Recap | — | "Welcome back. We have Ridge fitted. Now Lasso — the regularizer that can zero out features entirely." |
| `0:30–1:30` | Run LassoCV | Cell 13, Cell 14 | "Same pipeline pattern: StandardScaler plus LassoCV. Same alpha grid, 5-fold CV, max_iter 10,000 for convergence. R-squared values are similar to Ridge. The difference is in the coefficients." |
| `1:30–2:30` | Lasso coefficients + selection | Cell 17 | "The coefficient table shows which features Lasso kept and which it zeroed out. On this 8-feature dataset, Lasso likely keeps everything because the selected alpha is light. Feed it 44 polynomial features and it would start discarding noisy terms. That is the L1 advantage: automatic feature selection." |
| `2:30–3:30` | Exercise 2 prompt | Cell 16 | "Pause and complete Exercise 2. Count non-zero features. Identify the top predictors. Compare Lasso's coefficients to Ridge's. Which features did Lasso shrink the most?" |
| `3:30–4:30` | Model comparison table | Cell 20, Cell 21 | "Three models side by side: OLS, Ridge, Lasso. Similar validation R-squared on this clean dataset. The real differences: Ridge keeps all features with reduced magnitudes, Lasso may zero some out. When models perform similarly, prefer the simpler one." |
| `4:30–5:00` | Coefficient bar chart | Cell 24 | "The grouped bar chart makes the L1-vs-L2 story visual. Blue OLS bars are largest. Orange Ridge bars are slightly shorter — uniform shrinkage. Green Lasso bars may be absent for weak features — sparsity. Use this chart format in your project report." |

---

#### Video 3: "Project Proposal Sprint and Week 1 Wrap-Up" (~4 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–0:30` | Transition to project | — | "Technical content is done. Now the project. Your Week 1 proposal is due today. The template in this notebook gives you everything you need." |
| `0:30–2:00` | Walk through proposal template | Cell 26, Cell 27 | "Six sections. Dataset description: source, size, domain context. Predictive task: target variable, prediction unit, regression or classification, business question. Evaluation plan: metric, rationale, split strategy, baseline. Leakage risk assessment: three potential leakage sources and how you prevent them. Initial concerns: data quality, missing values, imbalance. Success criteria: what performance would be useful versus excellent. Each section connects to a skill from this week: metrics from notebook 03, leakage from notebook 01, baselines from notebook 03." |
| `2:00–3:00` | Emphasize importance | — | "This is not busy work. The proposal forces you to answer the hardest question in any project: what exactly am I predicting, for whom, and how will I know if it works? Getting this right now saves weeks of wasted effort. A bad dataset or a wrong metric discovered in Week 3 means starting over. A good proposal means everything from here builds on a solid foundation." |
| `3:00–4:00` | Wrap-Up + closing | Cell 28, Cell 29, Cell 30 | "Week 1 is complete. You now have the full regression toolkit: splitting, preprocessing, metrics, baselines, feature engineering, and regularization. Three rules: scale before regularizing, use CV to tune alpha, start project planning early. Week 2 begins classification — logistic regression, probabilities, confusion matrices. Same pipeline pattern, new problem type. Submit your notebook and your project proposal to Brightspace. See you in notebook 06." |

---

*This guide was created to support video lecture recording for MGMT 47400 — Predictive Analytics, Purdue University.*
