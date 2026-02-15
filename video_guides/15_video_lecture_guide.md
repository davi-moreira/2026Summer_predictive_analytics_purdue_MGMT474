# Video Lecture Guide: Notebook 15 — Interpretation - Feature Importance + Partial Dependence + Project Improved Model Delivery

## At a Glance

This guide covers:

1. **Why NB15 exists** — It answers the question that immediately follows model selection: "We chose a champion — now what is it actually learning, and where does it fail?" Without interpretation and error analysis, a deployed model is a black box that stakeholders cannot trust, audit, or improve

2. **Why it must follow NB14** — NB14 selects the champion model through a fair, documented protocol. Interpretation is only meaningful after you have committed to a specific model; running PDP/ICE on every candidate would waste time and confuse the narrative

3. **Why it must precede NB16** — NB16 teaches decision thresholds and probability calibration, which require understanding where the model's predictions are reliable and where they are not. Error analysis from NB15 identifies the failure segments that motivate threshold adjustments in NB16

4. **Why each library/tool is used:**
   - `permutation_importance` — model-agnostic feature importance computed on validation data, not training data
   - `PartialDependenceDisplay` — visualizes marginal effect of each feature on predictions (PDP and ICE)
   - `RandomForestRegressor` — serves as the champion model, chosen because it is non-linear (interesting PDP curves) yet stable (reliable importance estimates)
   - Residual analysis + segment errors — finds systematic failure modes the aggregate metrics hide

5. **Common student questions** with answers (e.g., why compute importance on validation instead of training? why does the PDP flatten at high income?)

6. **Connection to the full course arc** — how the interpretation artifacts from NB15 feed into threshold tuning (NB16), fairness auditing (NB17), model cards (NB17), and the final project deliverables (NB19-20)

7. **Suggested video structure** — both single-video (~12 min) and three-video (~4 min each) formats with speaking prompts, timestamps, and notebook cell references

---

## Purpose of This Document

This guide is your **lecture companion** for recording the video(s) that accompany `15_interpretation_error_analysis_project.ipynb`. It explains the *why* behind every design choice in the notebook — why interpretation comes after selection, why we use permutation importance instead of tree-based importance, why ICE plots matter beyond PDP, and why error analysis is essential for honest model communication. Use it as speaking notes, not as a script: the goal is to internalize the reasoning so you can explain it naturally on camera.

---

## 1. Why This Notebook Exists

Notebook 14 ends with a champion model and a selection memo. The champion has strong CV scores and a consistent test evaluation. But those scores answer *how well* the model performs — they do not answer *what the model learned*, *which features drive its predictions*, or *where it fails systematically*.

Notebook 15 answers all three questions. It introduces three interpretation tools (permutation importance, PDP, ICE) and one diagnostic methodology (error analysis by segment). Together, these tools transform a black-box model into something stakeholders can understand, trust, challenge, and improve.

This notebook exists because:

- **Stakeholders demand explanations.** A model that says "house price = $350,000" is useless to a loan officer unless they also know *why* — is it driven by income, location, house size? Permutation importance and PDP provide those answers.

- **Regulators require transparency.** In regulated industries (finance, healthcare, insurance), deploying a model without an interpretation narrative is a compliance risk. The model card template in NB17 depends on the interpretation artifacts produced here.

- **Error analysis drives improvement.** Knowing that the model has MAE of 0.33 overall is less useful than knowing it has MAE of 0.20 for low-income areas but MAE of 0.55 for high-income areas. The segment analysis reveals where to invest improvement effort.

---

## 2. Why It Must Come After Notebook 14

Notebook 14 establishes the champion model through a fair, documented selection protocol. NB15 consumes that selection:

| NB14 Concept | How NB15 Uses It |
|---|---|
| **Champion model selection** | NB15 interprets the champion specifically. Running permutation importance, PDP, and ICE on five different models would produce five sets of conflicting results. Commitment to one model enables a focused, coherent interpretation narrative. |
| **Selection memo with justification** | NB15 extends the memo with interpretation evidence. The selection memo says "we chose Random Forest because it scored highest." NB15 adds "and here is what it learned: income is the primary driver, geography modulates the effect, and it struggles with luxury homes." |
| **Test set discipline** | NB14 teaches that the test set is touched once. NB15 computes all interpretation artifacts on the validation set, reinforcing that discipline. Computing importance on training data would measure memorization, not generalization. |

**In short:** NB14 answers "which model?" NB15 answers "what does it know, and where does it fail?"

If you tried to teach interpretation before selection, students would not know *which* model to interpret. If you skipped NB15 entirely, students would deploy a model they cannot explain — which is both professionally irresponsible and often illegal in regulated industries.

---

## 3. Why It Must Come Before Notebook 16

Notebook 16 introduces **decision thresholds** and **probability calibration**. Its learning objectives assume students understand:

- Which features drive the model's predictions (from permutation importance)
- Where the model's predictions are reliable versus unreliable (from error analysis)
- That aggregate metrics hide important sub-population differences (from segment analysis)

NB16's threshold tuning makes business sense only when you know *which segments* will be affected by threshold changes. If the model fails systematically on high-income properties, lowering the threshold will not fix that — it will just increase false positives everywhere. The error analysis from NB15 tells you where threshold tuning can help and where it cannot.

The deliberate sequence is:

```
NB14: Select the champion model (fair comparison, documented memo)
            |
NB15: Interpret it (what does it learn?) + Error analysis (where does it fail?)
            |
NB16: Tune the threshold for business costs + Calibrate probabilities
```

Each notebook adds one layer of post-selection refinement. By NB16, students know what the model learned and where it fails — they can make informed decisions about thresholds and calibration.

---

## 4. Why These Specific Libraries and Tools

### 4.1 `sklearn.inspection.permutation_importance`

**What it does:** Measures feature importance by randomly shuffling each feature one at a time and recording how much the model's performance degrades. Features whose shuffling causes large degradation are important; features whose shuffling has no effect are unimportant.

**Why we use it (instead of tree-based `.feature_importances_`):**

1. **Model-agnostic.** Permutation importance works with any estimator — Random Forest, Gradient Boosting, Logistic Regression, SVM. Tree-based `.feature_importances_` only works with tree models. Teaching the model-agnostic version means students can use the same code regardless of which model won the selection in NB14.

2. **Computed on validation data.** The `scoring` parameter evaluates degradation on validation predictions, not training predictions. This measures genuine predictive value, not memorization. Tree-based importance is computed during training and can overweight features that are useful for fitting noise.

3. **Includes uncertainty.** With `n_repeats=10`, each feature is shuffled 10 times, producing a mean and standard deviation. Features with high std relative to their mean are unreliably estimated. Tree-based importance provides no such uncertainty measure.

**What to emphasize on camera:** Permutation importance answers "which features does the model *actually rely on* for its predictions on unseen data?" It is the feature importance method you should use by default. Tree-based importance is a useful diagnostic but should not be your primary reporting tool.

### 4.2 `sklearn.inspection.PartialDependenceDisplay`

**What it does:** Visualizes how the model's prediction changes as a single feature varies, while all other features are held at their observed values. It produces two types of plots: PDP (average effect across all samples) and ICE (effect for each individual sample).

**Why we use it:**

1. **PDP answers "what is the average effect of income on predicted house value?"** This is the most common stakeholder question. A PDP curve that rises steeply from income 0-5 and flattens above 8 communicates the relationship intuitively.

2. **ICE answers "does this effect vary across individuals?"** If all ICE curves follow the same trend, the effect is uniform. If they fan out or cross, the feature interacts with other features — its effect depends on context. This is information PDP alone cannot reveal.

3. **Works with any scikit-learn estimator.** The `from_estimator` API accepts any fitted model, making it model-agnostic like permutation importance.

**What to emphasize on camera:** PDP shows the average story. ICE shows whether that average is representative. Always check ICE before making policy recommendations based on PDP — a flat average can hide dramatic individual-level variation.

### 4.3 `RandomForestRegressor` as the Champion Model

**What it does:** An ensemble of decision trees trained on bootstrap samples with random feature subsets at each split.

**Why we use it for interpretation (rather than Gradient Boosting or Linear Regression):**

- **Non-linear.** Linear regression produces flat PDP lines (the effect of each feature is constant). Random Forest produces curves, steps, and interactions that make PDP and ICE plots visually interesting and pedagogically rich.

- **Stable.** Random Forest's bagging mechanism reduces variance, making permutation importance estimates more reliable than they would be for a single decision tree. The standard deviations across permutation repeats are typically small.

- **Reasonable performance.** With 100 trees and max_depth=10, the Random Forest achieves R-squared around 0.80 on California Housing — good enough to be worth interpreting, but not so perfect that all features appear equally important.

**What to emphasize on camera:** We chose Random Forest for demonstration because it balances performance, non-linearity, and stability. In your project, you will interpret whatever champion model NB14 selected.

### 4.4 Residual Analysis and Segment Error Analysis

**What it does:** Computes prediction residuals (actual minus predicted) and then analyzes them by sub-population segments — typically quartiles of the most important feature.

**Why we use it:**

- **Aggregate metrics hide failures.** An overall MAE of 0.33 sounds acceptable, but if the MAE for high-income properties is 0.55 and for low-income properties is 0.15, the model is twice as bad for expensive homes. Segment analysis reveals this.

- **Motivates targeted improvement.** Once you know *where* the model fails, you can invest effort precisely: add features for that segment, train a specialized sub-model, or flag those predictions for human review.

- **Connects to fairness.** Segment analysis by income is the same methodology as segment analysis by demographic group (NB17). Teaching it here on a "neutral" dimension prepares students to apply it on sensitive dimensions in the fairness notebook.

**What to emphasize on camera:** Error analysis is not optional polish — it is a core professional skill. Every model has failure modes. The question is whether you find them before deployment or your users find them for you.

---

## 5. Key Concepts to Explain on Camera

### 5.1 Permutation Importance on Validation Data, Not Training Data

This is the single most important methodological point in the notebook. Computing importance on training data measures what the model memorized — including noise. Computing importance on validation data measures what the model learned that generalizes.

For a model that overfits (e.g., an unconstrained decision tree), the difference is dramatic: training importance shows every feature as important (because the tree memorized the training set), while validation importance reveals that many features contribute nothing to generalization.

**Analogy for students:** Imagine you are testing whether a student understands algebra. If you re-use the exact homework problems (training data), every student looks like a genius. If you use new problems (validation data), you find out who actually learned algebra versus who just memorized the answers.

### 5.2 PDP Shows the Average — ICE Shows the Variation

A PDP curve for median income might show a smooth upward trend. But what if that trend only applies to coastal properties? For inland properties, the income-price relationship might be flatter. The PDP average of a steep coastal curve and a flat inland curve looks like a moderately steep curve — hiding the fact that two very different relationships are at work.

ICE plots reveal this by showing one line per sample. If the lines bundle tightly, the average is representative. If they fan out, the average is misleading. Wide ICE spread is a signal of interaction effects that merit further investigation.

**What to emphasize on camera:** Never report a PDP curve without checking the ICE spread. A narrow bundle gives you confidence in the average; a wide fan tells you the average is hiding something important.

### 5.3 Error Analysis as a Professional Habit

Students are accustomed to reporting a single MAE or R-squared. Error analysis trains them to ask "for whom?" and "under what conditions?" This habit transfers directly to the fairness analysis in NB17, where the segments become demographic groups instead of income quartiles.

The error analysis workflow is:
1. Compute residuals for all validation samples.
2. Plot residuals vs. predicted values (look for funnel shapes, clusters).
3. Split by the most important feature into quartiles.
4. Compute per-segment MAE, median error, and std.
5. Identify the worst segment and hypothesize why.

**What to emphasize on camera:** The segment with the highest error is your model's Achilles heel. Every model has one. Finding it is not a failure — it is a sign of rigor.

### 5.4 Honest Communication: Strengths, Limitations, and Recommendations

The interpretation narrative template in the notebook has three sections: strengths, limitations, and recommendations. Students often want to skip limitations because they feel it makes their model look bad. The opposite is true: honest disclosure of limitations makes the analyst look trustworthy.

A model card that says "this model works perfectly" is a red flag. A model card that says "this model achieves R-squared 0.80 overall but struggles with high-income properties where MAE exceeds $50,000; we recommend human review for properties predicted above $400,000" — that builds trust and prevents downstream surprises.

---

## 6. What Students Should Take Away

After watching your video and completing this notebook, students should be able to answer these questions confidently:

1. **"Why compute permutation importance on validation data instead of training data?"** — Because training importance measures memorization, not generalization. A model that overfits shows every feature as important on training data, even noise features. Validation importance reveals which features actually help predict unseen data.

2. **"What is the difference between PDP and ICE?"** — PDP shows the average effect of a feature across all samples. ICE shows the effect for each individual sample. If ICE curves are tightly bundled, the PDP average is representative. If they spread wide, the feature interacts with other features and the average is misleading.

3. **"What does a funnel shape in the residual plot mean?"** — Heteroscedasticity — the model's errors grow as the predicted value increases. On California Housing, this typically means the model is less accurate for expensive homes than cheap homes.

4. **"How do I identify a model's failure segment?"** — Split the validation set by quartiles of the most important feature, compute MAE for each quartile, and compare. The quartile with the highest MAE is the failure segment. Investigate why — is it data scarcity? Target capping? Missing features?

5. **"Why include a limitations section in the interpretation narrative?"** — Because honest disclosure builds trust and prevents surprises. Stakeholders who learn about a model's weaknesses from the analyst are grateful; stakeholders who learn about them from production failures are angry.

---

## 7. Common Student Questions (Anticipate These)

**Q: "Why does the PDP for median income flatten at high values?"**
A: Two reasons. First, the target variable `MedHouseVal` is capped at 5.0 ($500,000) in the California Housing dataset, so the model cannot predict above that ceiling. Second, there are very few training samples with income above 10, so the Random Forest has limited data to learn the relationship in that range. The flattening reflects both a data limitation and a target limitation.

**Q: "If permutation importance says a feature is unimportant, should I remove it?"**
A: Not necessarily. An unimportant feature in a Random Forest might become important in a different model or after feature engineering. Permutation importance is model-specific: it tells you what *this* model relies on, not what is objectively predictive. However, if a feature consistently shows zero importance across multiple models, removing it simplifies the model with no performance loss.

**Q: "Why use Random Forest instead of Gradient Boosting for interpretation?"**
A: For this notebook, Random Forest is chosen for pedagogical clarity — its PDP curves are smooth and its importance estimates are stable. In your project, you should interpret whatever model NB14 selected as champion. The interpretation tools (permutation importance, PDP, ICE) work identically with any scikit-learn estimator.

**Q: "The error analysis shows that the model is bad for expensive homes. Should I retrain with more data?"**
A: More data in the high-price range would help, but the fundamental issue is the target cap at $500,000. No amount of data can teach the model to predict above a ceiling that exists in the labels. The practical recommendations are: (1) acknowledge the limitation, (2) add features that differentiate within the capped range (e.g., proximity to coast), and (3) flag predictions near the cap for human review.

**Q: "How do I know if ICE spread is 'too wide'?"**
A: There is no universal threshold. Compare the ICE spread to the PDP range. If the PDP shows a 0.5-unit increase from low to high income, and the ICE curves span 2.0 units at a single income level, the individual variation dwarfs the average effect — the PDP is misleading. If the ICE spread is 0.1 units around a 0.5-unit PDP range, the average is trustworthy.

---

## 8. Connection to the Broader Course Arc

| Week | Notebooks | What NB15's Interpretation Enables |
|---|---|---|
| **Week 1** | 01-05 | Students learned coefficients in linear regression. NB15 generalizes interpretation to non-linear models where coefficients do not exist. |
| **Week 2** | 06-10 | Classification metrics were introduced. NB15's segment error analysis extends them by asking "metrics for whom?" |
| **Week 3** | 11-15 | NB14 selects the champion. NB15 interprets and error-analyzes it, completing the Week 3 narrative: build, select, understand. |
| **Week 4** | 16-20 | NB15's error analysis motivates NB16's threshold tuning (optimize for failure segments). NB15's interpretation feeds into NB17's model card (document what the model learned and its limitations). NB15's segment analysis previews NB17's fairness slicing (same methodology, different dimensions). |

**Interpretation is the bridge between model building and responsible deployment.** Without NB15, students would deploy models they cannot explain, cannot audit, and cannot improve. With NB15, they have the tools to communicate honestly about what the model does, where it fails, and what should be done about it.

---

## 9. Suggested Video Structure

Below are two recording options. Each segment includes the **speaking prompt** (what to say), **timestamps**, and the **notebook cells** to show on screen.

All cell references use the format `Cell N` where N is the zero-indexed cell number in `15_interpretation_error_analysis_project.ipynb`.

---

### Option A: Single Video (~12 minutes)

#### Segment 1 — Opening & Motivation `[0:00–1:30]`

> **Show:** Cell 0 (header) and Cell 1 (learning objectives) on screen.

**Say:**

"Welcome back. In the previous notebook we selected a champion model through a fair, documented protocol. We know it has strong metrics. But we do not yet know *what* it learned. Which features drive its predictions? How does the predicted house value change as income increases? And critically, where does the model fail — are there segments where the error is two or three times the average? Today we answer all three questions. We are going to compute permutation feature importance, create partial dependence and ICE plots, and conduct a segment-level error analysis. By the end, you will be able to explain your model to a stakeholder, not just report its score."

> **Action:** Scroll through the 5 learning objectives in Cell 1, briefly reading each one aloud.

---

#### Segment 2 — Setup & Champion Model `[1:30–3:00]`

> **Show:** Cell 2 (setup explanation), then **run** Cell 3 (imports).

**Say:**

"Our setup includes two new imports from scikit-learn's inspection module: `permutation_importance` and `PartialDependenceDisplay`. Both are model-agnostic — they work with any estimator, not just Random Forests. We are also back to the California Housing regression dataset because continuous predictions make residual analysis and PDP curves more informative than binary classification."

> **Run** Cell 6 (load data + split). Point to the train/val/test sizes.

**Say:**

"Standard 60-20-20 split. About 12,000 training samples, 4,000 validation, 4,000 test. All interpretation will happen on the validation set — we never compute importance on training data because that would measure memorization, not generalization."

> **Run** Cell 8 (train Random Forest + metrics).

**Say:**

"Our champion model: a Random Forest with 100 trees and max depth 10. It achieves R-squared about 0.80 on validation, meaning it explains 80 percent of the variance in house prices. MAE is about 0.33, which is 33,000 dollars. That is our baseline — the number we will try to understand through interpretation and break down through error analysis."

---

#### Segment 3 — Permutation Importance `[3:00–5:30]`

> **Show:** Cell 10 (section header), then **run** Cell 11 (compute permutation importance).

**Say:**

"Permutation importance works by shuffling one feature at a time and measuring how much the model's MAE degrades. If shuffling median income causes MAE to increase by 0.40, that feature is critical. If shuffling population barely changes MAE, that feature is unimportant. The key: we compute this on the validation set with 10 repeats, giving us both a mean importance and a standard deviation.

Look at the results. Median income dominates — shuffling it increases MAE by about 0.30 to 0.50. That makes economic sense: income is the strongest predictor of housing prices. Latitude and longitude rank second and third, capturing the geographic price premium of coastal California. Features like population and average bedrooms are near zero — the model barely relies on them."

> **Run** Cell 14 (importance bar chart). Point to the error bars.

**Say:**

"The bar chart is the primary deliverable for stakeholders who ask 'what drives the model?' A domain expert can look at this and immediately verify whether it makes sense. Income and location driving prices — intuitive. If population were the top feature, that would be suspicious and worth investigating. The error bars show how stable each estimate is across the 10 shuffle repeats."

---

#### Segment 4 — PDP and ICE `[5:30–7:30]`

> **Run** Cell 19 (PDP for top 4 features).

**Say:**

"Partial dependence plots show the average predicted house value as each feature varies. For median income, the PDP rises steeply from 0 to 5 and then flattens. The flattening has two causes: the target is capped at 500,000 dollars, and there are few training samples with very high income. For latitude and longitude, the curves capture the geographic price gradient — certain coordinates correspond to expensive coastal areas."

> **Run** Cell 22 (ICE plot for top feature).

**Say:**

"Now ICE — individual conditional expectation. The thin lines show the effect for each individual sample. Most follow the same upward trend, but notice the spread. Some curves are steeper, some are flatter. That variation tells us the effect of income on price depends on other features — a home in a coastal area gets a bigger price boost from income than a home in a rural area. If the ICE curves were perfectly bundled, the PDP average would tell the whole story. The spread means the average is hiding real heterogeneity."

> **Mention:** "Pause the video here and complete Exercise 1 — review the permutation importance results and write three evidence-based interpretation bullets." Point to Cell 16 (PAUSE-AND-DO 1).

---

#### Segment 5 — Error Analysis `[7:30–10:00]`

> **Run** Cell 25 (residual plot + histogram).

**Say:**

"Now the diagnostic side. The left panel plots residuals against predicted values. See the funnel shape? Errors are small for low-value homes and grow for high-value homes — classic heteroscedasticity. The right panel shows the residual distribution: roughly symmetric and centered near zero, meaning no systematic bias overall. But that average hides segment-level differences."

> **Run** Cell 28 (segment error analysis table).

**Say:**

"Here is where error analysis gets powerful. We split the validation set by income quartiles and compute MAE for each. The lowest income quartile has MAE around 0.15 to 0.20 — the model is quite accurate for affordable homes. The highest income quartile has MAE around 0.40 to 0.60 — two to three times worse. This is the model's Achilles heel: it struggles with luxury homes."

> **Run** Cell 30 (segment error box plot).

**Say:**

"The box plot makes the disparity visual. The Q4 box is taller, higher, and has more outliers. This is the chart you would show a business stakeholder: instead of one MAE number, you can say 'our model predicts affordable homes to within 20,000 dollars but luxury homes only to within 50,000 dollars. We should not use this model for high-end valuations without additional improvements.'"

> **Mention:** "Pause the video here and complete Exercise 2 — identify the worst segment and propose a hypothesis for why it fails." Point to Cell 32 (PAUSE-AND-DO 2).

---

#### Segment 6 — Narrative Template & Wrap-Up `[10:00–12:00]`

> **Show:** Cells 34-39 (interpretation narrative template: strengths, limitations, recommendations).

**Say:**

"The last piece is the interpretation narrative — a structured report with three sections. Strengths: what the model does well, backed by evidence from importance and PDP. Limitations: where it fails, backed by evidence from error analysis. Recommendations: what to do about the failures — additional features, human review for flagged segments, monitoring over time. This template connects directly to the model card you will build in notebook 17. Getting the evidence right here makes the model card almost write itself."

> **Show:** Cell 42 (Wrap-Up).

**Say:**

"Let me leave you with two critical takeaways. First: always compute importance on validation data, not training data. Training importance measures memorization; validation importance measures generalization. Second: every model has failure segments. The question is not whether they exist but whether you find them before deployment. The error analysis methodology you learned today — quartile splits, per-segment MAE, box plots — is the same methodology you will use for fairness auditing in notebook 17, except there the segments will be demographic groups instead of income quartiles.

Next notebook, we move from understanding the model to using it for business decisions. We will learn to tune classification thresholds based on cost matrices and calibrate probability estimates. See you there."

> **Show:** Cell 43 (Submission Instructions) briefly, then Cell 44 (Thank You).

---

### Option B: Three Shorter Videos (~4 min each)

---

#### Video 1: "Feature Importance and Partial Dependence" (~5 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–0:45` | Show header + objectives | Cell 0, Cell 1 | "Welcome back. We selected a champion model last time. Today we find out what it actually learned. Here are our five learning objectives." (Read them briefly.) |
| `0:45–1:30` | Run setup + data + model | Cell 3, Cell 6, Cell 8 | "California Housing dataset, 60-20-20 split. Random Forest champion with R-squared about 0.80 and MAE about 0.33 on validation. Now we interpret it." |
| `1:30–3:00` | Run permutation importance | Cell 11, Cell 14 | "Permutation importance shuffles each feature and measures how much performance degrades. Computed on validation data, not training data. Median income dominates. Latitude and longitude capture geography. Population contributes almost nothing. The bar chart is what you show a stakeholder who asks 'what drives the model?'" |
| `3:00–4:30` | Run PDP + ICE | Cell 19, Cell 22 | "PDP shows the average effect of each feature. For income: steep rise, then flattening due to the target cap. ICE shows individual-level variation. The spread of ICE curves tells you whether the average is trustworthy or hiding interactions. Wide spread means the feature's effect depends on context." |
| `4:30–5:00` | Exercise 1 prompt | Cell 16 | "Pause the video now and complete Exercise 1. Write three evidence-based interpretation bullets using the importance and PDP results. Come back when you are done." |

---

#### Video 2: "Error Analysis — Finding Where the Model Fails" (~4 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–0:30` | Recap | — | "Welcome back. We know what the model learned. Now we find out where it fails." |
| `0:30–1:30` | Run residual analysis | Cell 25 | "Residual plot: funnel shape, errors grow with predicted value. Histogram: symmetric, centered near zero. No systematic bias overall, but the funnel tells us something is wrong at the high end." |
| `1:30–3:00` | Run segment analysis | Cell 28, Cell 30 | "We split by income quartiles and compute MAE for each. Q1: about 0.20 — accurate. Q4: about 0.50 — two to three times worse. The box plot makes this visual. Q4's box is taller, higher, and has more outliers. This is the model's primary failure mode: luxury homes." |
| `3:00–4:00` | Exercise 2 prompt | Cell 32, Cell 33 | "Pause and complete Exercise 2. Identify the worst segment, propose a hypothesis for why it fails, and suggest one improvement. Come back when you are done." |

---

#### Video 3: "Interpretation Narrative and Wrap-Up" (~3 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–1:00` | Show narrative template | Cell 34, Cell 35, Cell 37, Cell 39 | "The interpretation narrative has three sections: strengths, limitations, and recommendations. Strengths: R-squared 0.80, income is the top driver. Limitations: MAE doubles for high-income properties, target cap creates a ceiling. Recommendations: human review for high-value predictions, additional features for the luxury segment." |
| `1:00–2:00` | Show project milestone 3 | Cell 40, Cell 41 | "For your project, Milestone 3 requires all the artifacts we built today: importance plot, PDP/ICE for top features, segment error analysis, and an evidence-based interpretation narrative. The checklist in the notebook tells you exactly what to include." |
| `2:00–3:00` | Wrap-up + closing | Cell 42, Cell 43, Cell 44 | "Two rules to remember. One: compute importance on validation data, not training data. Two: every model has failure segments — find them before deployment. Next notebook, we translate model outputs into business decisions using thresholds and calibration. Submit your notebook to Brightspace and complete the quiz. See you there." |

---

*This guide was created to support video lecture recording for MGMT 47400 — Predictive Analytics, Purdue University.*
