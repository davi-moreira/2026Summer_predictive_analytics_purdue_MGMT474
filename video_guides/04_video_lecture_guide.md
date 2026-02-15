# Video Lecture Guide: Notebook 04 — Linear Features & Diagnostics

## At a Glance

This guide covers:

1. **Why NB04 exists** — It teaches students how to *improve* a linear model through feature engineering (interactions, polynomial features) and how to *diagnose* model problems using residual analysis and coefficient interpretation

2. **Why it must follow NB03** — NB03 provides the evaluation framework (MAE, RMSE, R^2, baselines). Without it, students would engineer features blindly, with no way to measure whether their changes actually improved the model

3. **Why it must precede NB05** — NB04 demonstrates the overfitting risk of polynomial feature expansion (8 features exploding to 44). NB05 introduces regularization (Ridge, Lasso) as the direct solution to this problem. Without seeing the problem first, students would not understand why regularization matters

4. **Why each library/tool is used:**
   - `PolynomialFeatures` — generates interaction terms and squared features automatically, demonstrating the accuracy-complexity tradeoff
   - `LinearRegression` (with StandardScaler in a Pipeline) — reused from NB02-03 so students can isolate the effect of feature engineering
   - Coefficient bar charts — make feature importance tangible and visual after standardized scaling
   - Residual comparison plots — side-by-side diagnostics that reveal whether polynomial features actually fix model deficiencies

5. **Common student questions** with answers (e.g., why scale before interpreting coefficients, when polynomial features help vs hurt, what heteroskedasticity means)

6. **Connection to the full course arc** — how feature engineering and diagnostics from NB04 inform regularization (NB05), tree-based models (Week 3), and the final project

7. **Suggested video structure** — both single-video (~12 min) and three-video (~4 min each) formats with speaking prompts, timestamps, and notebook cell references

---

## Purpose of This Document

This guide is your **lecture companion** for recording the video(s) that accompany `04_linear_features_diagnostics.ipynb`. It explains the *why* behind every design choice in the notebook — why it introduces feature engineering at this point in the course, why it uses specific tools, and why the complexity-vs-accuracy tradeoff is the central theme. Use it as speaking notes, not as a script: the goal is to internalize the reasoning so you can explain it naturally on camera.

---

## 1. Why This Notebook Exists

Notebooks 02 and 03 leave students with a baseline linear model (R^2 approximately 0.60) and the tools to evaluate it. The natural next question is: **"How do I make it better?"**

Notebook 04 answers this question using the *simplest* improvement strategy available within the linear model family: adding more informative features. Rather than switching to a more complex model class (trees, ensembles), NB04 shows that a linear model's performance ceiling is determined by the features it sees. By engineering interactions and polynomial terms, students push R^2 higher — but at a cost: the feature count explodes from 8 to 44, the overfit gap widens, and the model becomes harder to interpret.

This tradeoff — accuracy vs. complexity — is the central lesson. It sets up NB05 perfectly: regularization is the technique that controls the complexity side of the equation, letting students keep the polynomial features while reining in overfitting.

---

## 2. Why It Must Come After Notebook 03

Notebook 03 establishes four capabilities that notebook 04 *consumes*:

| NB03 Concept | How NB04 Uses It |
|---|---|
| **MAE, RMSE, R^2 metrics** | NB04 computes these same metrics for three models (baseline linear, manual interaction, polynomial) and compares them in a single table. Students already know what the numbers mean; NB04 asks them to *compare* numbers across models. |
| **Baseline comparison habit** | NB04 starts by fitting the same StandardScaler + LinearRegression pipeline from NB03 and recording its R^2 as the "number to beat." This is the baseline habit in action. |
| **Residual analysis** | NB04 generates side-by-side residual plots for the baseline and polynomial models. Students already know how to read residual plots from NB03; NB04 asks them to *compare* two sets of residuals. |
| **Overfit gap monitoring** | NB04 explicitly computes (train R^2 - val R^2) for each model and flags the polynomial model's larger gap. Students learned to check this gap in NB03. |

**In short:** NB03 gives the evaluation vocabulary, NB04 gives the first opportunity to *use* it in a model improvement context.

If you tried to teach feature engineering *before* NB03, students would not know how to measure whether their features helped. If you skipped NB04 entirely and went straight to regularization in NB05, students would not understand *why* regularization is needed — they would never have seen the overfitting problem that polynomial features create.

---

## 3. Why It Must Come Before Notebook 05

Notebook 05 introduces **regularization** (Ridge, Lasso, ElasticNet) plus cross-validated alpha selection. Its learning objectives assume students already understand:

- That adding polynomial features can improve R^2 but also increases overfitting risk
- That the overfit gap (train R^2 minus val R^2) is the primary diagnostic for overfitting
- That coefficient magnitudes become unstable and hard to interpret with many correlated features
- That the model comparison table (baseline vs. improved) is the standard evaluation format

NB04 teaches all four concepts experientially. When NB05 introduces Ridge and Lasso, students immediately recognize the *problem* being solved: "I have 44 polynomial features, some of which are noisy. Ridge shrinks all the coefficients to stabilize them. Lasso zeros out the irrelevant ones." Without NB04's demonstration of the problem, the regularization solution in NB05 would feel unmotivated and abstract.

The deliberate sequence is:

```
NB03: How do we measure model quality? (Metrics + baselines)
            |
NB04: How do we improve the model? (Feature engineering + diagnostics)
            |     [Problem: overfitting from too many features]
            |
NB05: How do we control complexity? (Regularization as the solution)
```

---

## 4. Why These Specific Libraries and Tools

### 4.1 `sklearn.preprocessing.PolynomialFeatures`

**What it does:** Given a feature matrix with p columns, `PolynomialFeatures(degree=2, include_bias=False)` generates all original features, all pairwise products (interactions), and all squared terms. For 8 original features, this produces 8 + 28 interaction pairs + 8 squares = 44 features.

**Why we use it:**

1. **It demonstrates the accuracy-complexity tradeoff concretely.** Students see R^2 jump (good) but also see the feature count explode by 5.5x (concerning). The overfit gap widens. This is the first time in the course that "more features" creates a visible downside.

2. **It motivates regularization.** The 44 polynomial features include many that are correlated (e.g., x1*x2 is correlated with both x1 and x2). This multicollinearity makes OLS coefficients unstable and large — exactly the problem Ridge regression solves in NB05.

3. **It is automatic.** Students do not need to manually create x1^2, x1*x2, etc. The single transformer generates everything. This teaches a professional pattern: let the library handle the combinatorics, then let regularization handle the selection.

**What to emphasize on camera:** The feature explosion is exponential. Degree 2 on 8 features gives 44. Degree 3 would give 164. Degree 4 would give 494. At some point, you have more features than training samples, and the model collapses. This is the curse of dimensionality in miniature.

### 4.2 `sklearn.linear_model.LinearRegression` (reused from NB02-03)

**What it does:** Fits ordinary least-squares regression: minimizes the sum of squared residuals with no penalty on coefficient magnitude.

**Why we reuse it here:**

- **It isolates the effect of feature engineering.** The model class is unchanged from NB02 and NB03. The *only* difference is the features. This controlled comparison lets students attribute any performance change entirely to the new features, not to a different algorithm.
- **It reveals OLS's weakness.** Without regularization, OLS assigns nonzero weight to every feature, including noisy interactions that hurt generalization. When NB05 introduces Ridge/Lasso, students will see those noisy weights shrink or disappear.

**What to emphasize on camera:** We deliberately keep the model simple to isolate the variable we are changing — the features. Good experimental design in machine learning means changing one thing at a time and measuring the effect.

### 4.3 Coefficient Bar Charts (via matplotlib)

**What they do:** Display the fitted linear regression coefficients as horizontal bars, sorted by absolute magnitude.

**Why we use them:**

- **They make feature importance tangible.** Students can see at a glance that MedInc dominates the prediction and that features like Population or AveBedrms contribute little.
- **They require standardized features.** Raw coefficients are meaningless for comparison when features have different scales (income in $10k vs. latitude in degrees). Standardizing first (via the pipeline's `StandardScaler`) puts all coefficients on the same scale, making the bar lengths directly comparable.
- **They set up the NB05 coefficient comparison.** NB05 will show three grouped bars (OLS, Ridge, Lasso) per feature. NB04 introduces the single-bar version so students are familiar with the visual format before the regularization comparison.

**What to emphasize on camera:** Always scale before interpreting coefficients. And remember: coefficients measure *association*, not *causation*. A large positive coefficient on Latitude does not mean moving north causes higher prices — it reflects geographic patterns.

### 4.4 Residual Comparison Plots (2x2 grid)

**What they do:** Display residual-vs-predicted scatter plots and residual histograms for two models side by side.

**Why we use them:**

- **They answer "did the new features fix the model's weaknesses?"** If the baseline model has a curved residual pattern (indicating missed nonlinearity), the polynomial model's residual plot should show whether the curve disappears.
- **They provide visual evidence for the comparison table.** The numbers (MAE, R^2) say the polynomial model is better. The plots show *how* — reduced systematic bias, tighter scatter, fewer extreme residuals.
- **They build the diagnostic habit.** By NB04, students have now seen residual plots in two consecutive notebooks (NB03 and NB04). Repetition builds fluency.

**What to emphasize on camera:** Do not just look at the scatter. Compare the *shapes*. Does the polynomial model remove the funnel pattern? Does it reduce the right tail in the histogram? If so, the added features addressed a real model deficiency, not just noise.

---

## 5. Key Concepts to Explain on Camera

### 5.1 Feature Engineering as the "Cheapest" Way to Improve a Model

Before switching to a more complex model (trees, neural networks), exhaust what the current model can do with better inputs. Linear regression can only learn linear relationships, but it can learn nonlinear relationships if you *create* nonlinear features for it (squares, interactions, logs).

**Analogy for students:** A chef with basic ingredients can only make simple dishes. But give that same chef spices, sauces, and new techniques (feature engineering), and the simple dishes become much better — without buying a new kitchen (a new model class). Feature engineering is seasoning your data.

### 5.2 The Accuracy-Complexity Tradeoff

Every feature you add is a knob the model can turn. With 8 features, the model has 8 knobs — it cannot fit noise very well, so it generalizes. With 44 features, the model has 44 knobs — enough to start memorizing training-set quirks. The overfit gap (train R^2 minus val R^2) is the thermometer for this tradeoff.

**Analogy for students:** Imagine fitting a curve through 10 points. A straight line (2 parameters) cannot overfit. A degree-9 polynomial (10 parameters) passes through every point perfectly — but it oscillates wildly between them. The same thing happens in higher dimensions with too many features.

### 5.3 Coefficient Interpretation Requires Scaling

Raw linear regression coefficients have units of "change in target per one-unit change in feature." But if MedInc is measured in $10,000 units and AveRooms in rooms-per-household, a one-unit change means very different things. After standardizing (mean 0, std 1), a one-unit change in every feature means "one standard deviation." Now the coefficients are directly comparable: the largest coefficient corresponds to the most influential feature, regardless of its original scale.

### 5.4 Residual Patterns Point to Model Improvements

- **Funnel shape:** Errors grow with prediction level. Consider log-transforming the target or using a model that handles heteroskedasticity.
- **Curved pattern:** The model misses a nonlinear relationship. Add polynomial terms or interactions (which is exactly what NB04 does).
- **Clusters:** Subgroups in the data behave differently. Consider stratified modeling or adding indicator features.
- **Random scatter:** The model has extracted all the signal it can. Improvement requires fundamentally different features or a different model class.

---

## 6. What Students Should Take Away

After watching your video and completing this notebook, students should be able to answer these questions confidently:

1. **"Why should I scale features before interpreting coefficients?"** — Because raw coefficients reflect both the feature's importance *and* its scale. If income is in $10,000 units and rooms are in single digits, the coefficient magnitudes are not comparable. Scaling removes the scale factor, leaving only importance.

2. **"What happens when I add polynomial features of degree 2?"** — The feature count goes from p to p + p*(p-1)/2 + p = p*(p+1)/2 features (excluding bias). For 8 features, that is 44. Each new feature is a square or a product of two original features.

3. **"Why did the overfit gap increase with polynomial features?"** — Because the model now has 44 parameters instead of 8. More parameters means more capacity to fit noise in the training data. The training score goes up more than the validation score, widening the gap.

4. **"How do I know if polynomial features are worth the complexity?"** — Compare the validation R^2 improvement against the overfit gap increase. If validation improves by 0.05 but the gap goes from 0.01 to 0.04, you are getting diminishing returns. Also check the residual plots: if the polynomial model removes a visible pattern, the added complexity is justified.

5. **"What should I try next if the overfit gap is too large?"** — Regularization. Ridge and Lasso (covered in the next notebook) penalize large coefficients, effectively shrinking or eliminating the noisy features that polynomial expansion creates.

---

## 7. Common Student Questions (Anticipate These)

**Q: "Is it better to create features manually or use PolynomialFeatures?"**
A: Manual feature engineering uses domain knowledge (e.g., "income times rooms matters because wealthy neighborhoods have bigger houses"). PolynomialFeatures is brute-force — it creates *every* interaction, including meaningless ones. In practice, start with a few targeted manual interactions based on domain reasoning, then consider PolynomialFeatures if you want to be thorough. Regularization (NB05) will sort out which terms matter.

**Q: "Why not go to degree 3 or 4 for even better performance?"**
A: You could, but the feature count explodes: degree 3 on 8 features gives 164 features, degree 4 gives 494. With only 12,000 training samples, you rapidly approach the point where there are more features than meaningful training signal. The model overfits badly, validation R^2 drops, and you need heavy regularization just to survive. Degree 2 is the sweet spot for most tabular datasets.

**Q: "The coefficient on Latitude is large and negative. Does that mean moving south increases house prices?"**
A: It means that in this dataset, census tracts at lower latitudes (southern California — think Los Angeles, San Diego) tend to have higher prices. But this is *correlation*, not *causation*. You cannot move a house south and expect its price to increase. The coefficient reflects geographic price patterns, not causal effects.

**Q: "My residual plot still shows a funnel shape after polynomial features. Why?"**
A: Polynomial features help with nonlinearity but do not necessarily fix heteroskedasticity. The funnel shape (error variance increasing with prediction level) often requires a different remedy: log-transforming the target variable, or using a model that explicitly handles varying error variance. Alternatively, the California Housing dataset's $500k cap creates an artificial ceiling that no feature engineering can overcome.

**Q: "Why does the comparison table show the interaction model improving only slightly?"**
A: Adding a single interaction (MedInc * AveRooms) adds one feature to the 8 already present. That single term captures one specific joint effect. The improvement is small but targeted. PolynomialFeatures(degree=2) adds 36 new features — including that same interaction — which is why it shows a larger improvement. More features = more potential signal, but also more potential noise.

---

## 8. Connection to the Broader Course Arc

| Week | Notebooks | What NB04's Feature Engineering & Diagnostics Enable |
|---|---|---|
| **Week 1** | 01-05 | NB04's polynomial feature expansion directly motivates NB05's regularization. The overfit gap from 44 features is the *problem*; Ridge/Lasso are the *solution*. Students see cause and effect across two notebooks. |
| **Week 2** | 06-10 | Classification notebooks use the same coefficient-interpretation and diagnostic patterns. The concept of "more features can hurt" carries over to logistic regression, where too many features cause separation problems. |
| **Week 3** | 11-15 | Tree-based models (NB11-14) handle nonlinear relationships *natively* — they do not need polynomial features. This creates a natural comparison: NB04's approach (engineer features, stay linear) vs. NB11's approach (let the model discover nonlinearity). Students with NB04 experience can articulate the tradeoff. |
| **Week 4** | 16-20 | The final project requires students to justify their feature engineering choices. NB04's comparison table format (baseline vs. engineered vs. polynomial) becomes the template for project reporting. The complexity-vs-accuracy analysis from NB04 is directly relevant to the executive deliverable. |

**NB04 is the first "model improvement" notebook in the course.** It teaches the *process* of trying something new, measuring the effect, and deciding whether the complexity is worth it. This process — not any specific technique — is the transferable skill.

---

## 9. Suggested Video Structure

Below are two recording options. Each segment includes the **speaking prompt** (what to say), **timestamps**, and the **notebook cells** to show on screen.

All cell references use the format `Cell N` where N is the zero-indexed cell number in `04_linear_features_diagnostics.ipynb`.

---

### Option A: Single Video (~12 minutes)

#### Segment 1 — Opening & Motivation `[0:00–1:30]`

> **Show:** Cell 0 (header) and Cell 1 (learning objectives) on screen.

**Say:**

"Welcome back. Over the last two notebooks we built a preprocessing pipeline and learned how to evaluate models with MAE, RMSE, and R-squared. Our baseline linear model explains about 60 percent of the variance in house prices. Today we ask: how do we push that number higher? The answer is feature engineering — creating new features from existing ones so the linear model can capture relationships it would otherwise miss. We will create interaction terms, polynomial features, and then use the diagnostic tools from the last notebook to see whether the added complexity is actually worth it. By the end, you will understand the fundamental tradeoff between accuracy and complexity — which sets up next notebook's topic: regularization."

> **Action:** Scroll through the 5 learning objectives in Cell 1, briefly reading each one aloud.

---

#### Segment 2 — Setup, Data, & Baseline `[1:30–3:30]`

> **Show:** Cell 2 (setup), then **run** Cell 2 (imports).

**Say:**

"Standard setup. One new import: PolynomialFeatures from sklearn.preprocessing. This transformer will automatically generate interaction and squared terms for us. Everything else is familiar — pandas, numpy, matplotlib, seaborn, LinearRegression, StandardScaler, Pipeline, and our three metric functions."

> **Run** Cell 2. Point to `Setup complete!`.

> **Run** Cell 5 (load data and split). Point to the sizes.

**Say:**

"Same California Housing dataset, same 60-20-20 split, same random seed 474. These numbers are identical to the previous notebooks, so any performance change we see today is entirely due to feature engineering, not data differences."

> **Run** Cell 8 (baseline linear model).

**Say:**

"Before we add any features, we fit the exact same baseline pipeline from notebooks 02 and 03: StandardScaler plus LinearRegression. Train R-squared about 0.60, validation R-squared about 0.62, overfit gap essentially zero. This is our number to beat. We will come back to this table at the end to see whether our feature engineering efforts were worthwhile."

---

#### Segment 3 — Coefficient Interpretation `[3:30–5:30]`

> **Show:** Cell 10 (coefficient interpretation explanation).

**Say:**

"Before adding features, let us understand what the current model has learned. These are the linear regression coefficients, extracted after StandardScaler has normalized all features to mean zero and standard deviation one. Because everything is on the same scale, the coefficient magnitudes are directly comparable. The bar chart makes this visual."

> **Run** Cell 11 (extract coefficients + bar chart).

**Say:**

"MedInc — median income — dominates. Its coefficient is far larger than any other feature. This makes intuitive sense: neighborhood income is the strongest predictor of house prices. Latitude and Longitude have sizable negative and positive coefficients, reflecting the geographic price gradient — coastal southern California is expensive. Features like AveBedrms and Population are smaller, contributing less to the prediction.

Two important caveats. First, these coefficients measure association, not causation. A large coefficient on Latitude does not mean moving south causes higher prices. Second, multicollinearity — correlations between features — can make individual coefficients unstable. When two features are highly correlated, the model can split the effect between them arbitrarily. We will see regularization address this in notebook 05."

---

#### Segment 4 — Feature Engineering `[5:30–8:00]`

> **Show:** Cell 13 (interactions explanation), then **run** Cell 14 (manual interaction: MedInc * AveRooms).

**Say:**

"Now let us engineer some new features. First, a manual interaction. We multiply MedInc by AveRooms to create a new column that captures the combined effect of income and house size. The intuition: a high-income neighborhood with large houses might command disproportionately high prices — more than either factor alone would predict. After adding this one feature, we fit a new pipeline and compare to the baseline. The improvement in validation R-squared is small — maybe a few thousandths. One feature cannot change everything, but the direction is right."

> **Show:** Cell 16 (PAUSE-AND-DO 1 explanation), then **run** Cell 17 (PolynomialFeatures degree=2).

**Say:**

"Now the brute-force approach: PolynomialFeatures with degree 2. Instead of guessing which interactions matter, we let scikit-learn generate all of them: every squared term and every pairwise product. Look at the numbers: 8 original features become 44. That is a 5.5x explosion. And the results: validation R-squared jumps noticeably — polynomial features capture nonlinear relationships that the plain linear model missed. But check the overfit gap. It is wider now. The model has 44 knobs instead of 8, and it is starting to fit training-set noise.

This is the central tradeoff of the notebook. More features buy you accuracy but cost you complexity and generalization. How do you get the accuracy without the overfitting? That is exactly what regularization — Ridge and Lasso in notebook 05 — is designed to solve."

> **Mention:** "Pause the video here and complete Exercise 1 — run PolynomialFeatures and compare the validation scores and overfit gaps." Point to Cell 16 (PAUSE-AND-DO 1).

---

#### Segment 5 — Residual Diagnostics `[8:00–10:30]`

> **Show:** Cell 20 (residual diagnostics explanation), then **run** Cell 21 (2x2 residual comparison).

**Say:**

"Let us see if the polynomial features actually fixed the model's weaknesses or just reduced the average error. Top row: residuals versus predicted values for the baseline model on the left and the polynomial model on the right. Compare the shapes. The baseline may show a slight curve or funnel pattern. Does the polynomial model flatten it? Bottom row: residual histograms. Are the polynomial residuals more tightly centered around zero? Is the right tail — those expensive houses the model under-predicts — smaller?

The printed MAE and RMSE values quantify the improvement. But the *visual* comparison is what tells you whether the improvement is real structural improvement or just memorization. If the polynomial model's residuals look more random — less patterned — than the baseline's, the added features are capturing genuine nonlinear signal."

---

#### Segment 6 — Comparison Table, Wrap-Up & Closing `[10:30–12:00]`

> **Run** Cell 26 (final comparison table: baseline vs interaction vs polynomial).

**Say:**

"Here is the full comparison table. Three models: baseline linear with 8 features, the manual interaction model with 9, and the polynomial model with 44. The polynomial model wins on validation R-squared, but look at the cost: 5.5 times more features and a larger overfit gap. The interaction model gives a small improvement with almost no added complexity. Which one should you choose? That depends on the context. If interpretability and simplicity matter, the interaction model is better. If pure accuracy matters and you are willing to accept the complexity, the polynomial model is better — but you should regularize it.

And that is exactly what we will do in the next notebook. Ridge regression will shrink those 44 coefficients, keeping the useful ones and dampening the noisy ones. Lasso will go further and set some of them to exactly zero, performing automatic feature selection."

> **Show:** Cell 28 (Wrap-Up). Read the three critical rules.

**Say:**

"Three rules from today. One: scale features before interpreting coefficients. Two: polynomial features explode exponentially — use them sparingly and always check the overfit gap. Three: residual plots never lie. They show you whether your features fixed a real model deficiency or just added noise. Next notebook: regularization with Ridge and Lasso — the solution to the complexity problem we just created. See you there."

> **Show:** Cell 29 (Submission Instructions) briefly, then Cell 30 (Thank You).

---

### Option B: Three Shorter Videos (~4 minutes each)

---

#### Video 1: "Baseline and Coefficient Interpretation" (~4 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–0:45` | Show header + objectives | Cell 0, Cell 1 | "Welcome back. Our linear model explains about 60 percent of house price variance. Today we make it better with feature engineering — interactions and polynomials — and we learn to diagnose whether the improvement is real. Here are our five learning objectives." (Read them briefly.) |
| `0:45–1:30` | Run setup + load data | Cell 2, Cell 5 | "Standard setup. New import: PolynomialFeatures. Same California Housing data, same 60-20-20 split, same seed 474. Any performance change we see is purely from feature engineering." |
| `1:30–2:30` | Fit baseline + scores | Cell 8 | "Same StandardScaler plus LinearRegression pipeline. Train R-squared about 0.60, validation about 0.62, overfit gap near zero. This is our benchmark — every new model must beat these numbers." |
| `2:30–4:00` | Coefficients + bar chart | Cell 10, Cell 11 | "After standardizing, we extract coefficients and plot them. MedInc dominates — income is the strongest predictor. Latitude and Longitude capture geographic price patterns. Two caveats: these are associations, not causes, and multicollinearity can make individual coefficients unstable. We will address that with regularization in notebook 05." |

---

#### Video 2: "Feature Engineering — Interactions and Polynomials" (~5 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–0:30` | Recap | — | "Welcome back. We have our baseline R-squared of about 0.60. Now let us engineer features to push it higher." |
| `0:30–1:30` | Manual interaction | Cell 13, Cell 14 | "First, a targeted manual interaction: MedInc times AveRooms. High-income neighborhoods with large houses might command premium prices beyond what either feature alone predicts. One new column, one new pipeline fit, small improvement in validation R-squared. Targeted feature engineering based on domain knowledge." |
| `1:30–3:30` | Polynomial features | Cell 16, Cell 17 | "Now the brute-force approach: PolynomialFeatures degree 2. Eight features become 44. Every square and every pairwise product. Validation R-squared jumps — the polynomial terms capture nonlinear patterns the plain linear model missed. But the overfit gap is wider. Forty-four knobs give the model room to memorize training noise. This is the accuracy-complexity tradeoff in action." |
| `3:30–4:00` | Exercise 1 prompt | Cell 16 | "Pause and complete Exercise 1. Add polynomial features, compare validation scores, and examine the overfit gap. Ask yourself: is the R-squared improvement worth 5.5 times more features? Come back when you are done." |
| `4:00–5:00` | Analysis discussion | Cell 19 | "Write your analysis. Did polynomial features improve validation score? Is the train-val gap larger? Is the added complexity worth the improvement? These are the questions you should ask every time you engineer features." |

---

#### Video 3: "Residual Diagnostics and Model Comparison" (~4 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–1:30` | Residual comparison | Cell 20, Cell 21 | "Numbers say the polynomial model is better. But *how* is it better? Residual plots answer this. Top row: residuals versus predicted. Compare the baseline on the left to the polynomial on the right. Does the polynomial model reduce the funnel shape? Does it flatten the curve? Bottom row: are the residual histograms tighter? The MAE and RMSE numbers below quantify the improvement. But the visual comparison tells you whether the model genuinely fixed a structural problem or just reduced the average error." |
| `1:30–2:00` | Exercise 2 prompt | Cell 23 | "Pause and complete Exercise 2 — write a diagnostic conclusion. Describe the residual patterns, identify heteroskedasticity, compare the two models, and suggest what you would try next." |
| `2:00–3:00` | Final comparison table | Cell 25, Cell 26 | "The full comparison: baseline 8 features, interaction 9 features, polynomial 44 features. Polynomial wins on R-squared but has the largest overfit gap. The interaction model gives modest improvement with almost no added complexity. Always weigh accuracy gains against complexity costs. In the next notebook, regularization lets us keep the polynomial features while controlling the overfitting." |
| `3:00–4:00` | Wrap-Up + closing | Cell 28, Cell 29, Cell 30 | "Three rules. One: scale before interpreting coefficients. Two: polynomial features explode — use sparingly, check the overfit gap. Three: residual plots never lie. Next notebook: Ridge and Lasso regularization — the direct solution to the complexity problem we created today. Submit your notebook and complete the quiz. See you in notebook 05." |

---

*This guide was created to support video lecture recording for MGMT 47400 — Predictive Analytics, Purdue University.*
