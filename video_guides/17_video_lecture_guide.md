# Video Lecture Guide: Notebook 17 — Fairness and Ethics Basics - Responsible Predictive Analytics

## At a Glance

This guide covers:

1. **Why NB17 exists** — It teaches students that a model can have excellent aggregate metrics and still harm specific groups. Fairness auditing, slice-based evaluation, and model cards are professional obligations, not optional extras

2. **Why it must follow NB16** — NB16 teaches thresholds and calibration. Threshold choices directly determine selection rates, TPR, and FPR for each group. Students need to understand how thresholds work mechanically before analyzing whether those thresholds produce equitable outcomes across demographic groups

3. **Why it must precede NB18** — NB18 covers reproducibility, deployment, and monitoring. A model that is deployed without a fairness audit and a model card is a liability. NB17 produces the fairness assessment and model card that NB18 packages into the deployment artifact

4. **Why each library/tool is used:**
   - `make_classification` — generates synthetic data with a group attribute to simulate real-world sensitive attributes
   - `confusion_matrix` — computes per-group TP/FP/FN/TN for fairness metrics (selection rate, TPR, FPR)
   - Slice-based evaluation — computes accuracy, precision, recall, F1 for each group separately
   - Model card template — structured documentation following Mitchell et al. (2019)

5. **Common student questions** with answers (e.g., why can we not achieve all fairness metrics simultaneously? why exclude the group attribute from training?)

6. **Connection to the full course arc** — how fairness skills from NB17 connect to deployment (NB18), the executive narrative (NB19), and the final project (NB20)

7. **Suggested video structure** — both single-video (~12 min) and three-video (~4 min each) formats with speaking prompts, timestamps, and notebook cell references

---

## Purpose of This Document

This guide is your **lecture companion** for recording the video(s) that accompany `17_fairness_slicing_model_cards.ipynb`. It explains the *why* behind every design choice — why fairness matters even for models that score well on aggregate, why different fairness metrics conflict, why model cards are a professional standard, and why this topic closes the analytical arc before deployment begins. Use it as speaking notes, not as a script: the goal is to internalize the reasoning so you can explain it naturally on camera.

---

## 1. Why This Notebook Exists

By notebook 16, students have built a champion model, interpreted its behavior, set a business-optimal threshold, and calibrated its probabilities. The model is analytically complete. But it is not yet responsibly complete.

Notebook 17 asks a question that aggregate metrics cannot answer: **"Does this model treat all groups of people fairly?"** A model with 95% overall accuracy might have 98% accuracy for one group and 88% for another. A model that maximizes business value might systematically miss positive cases in a minority subgroup. These disparities are invisible in aggregate reporting and can cause real harm — denied loans, missed diagnoses, unfair hiring decisions.

This notebook exists because:

- **Fairness is a professional obligation.** The EU AI Act, US algorithmic accountability proposals, and industry standards (Google's Model Cards, Microsoft's Fairlearn) all require that deployed AI systems be audited for group-level disparities. Students entering the workforce must know how to conduct these audits.

- **Aggregate metrics hide disparities.** NB15 showed this for income segments (the model fails on luxury homes). NB17 extends the same methodology to demographic groups, where the stakes are higher because the disparities affect people, not just prediction accuracy.

- **Model cards are becoming standard.** A model card is a structured document that accompanies a deployed model and discloses its training data, performance metrics, limitations, and ethical considerations. NB17 provides the template and walks students through filling it out.

- **Fairness metrics conflict.** Students need to understand that you cannot simultaneously achieve demographic parity, equal opportunity, and equalized odds (except in trivial cases). This impossibility result, due to Chouldechova (2017) and others, means that fairness requires *choosing which criterion matters most* — a decision that depends on the deployment context.

---

## 2. Why It Must Come After Notebook 16

Notebook 16 establishes the decision-making framework that NB17 audits:

| NB16 Concept | How NB17 Uses It |
|---|---|
| **Threshold selection** | NB16 picks a threshold that maximizes overall business value. NB17 asks: does this threshold produce equitable outcomes across groups? A threshold optimized for the population average may be systematically worse for a minority group. |
| **Cost matrix** | NB16 defines TP/FP/FN/TN costs. NB17 extends this by noting that those costs may differ across groups — a false negative for a high-risk patient subgroup may be more consequential than for a low-risk subgroup. |
| **Calibration** | NB16 calibrates probabilities for the overall population. NB17 asks: are probabilities equally well-calibrated for all groups? A model that is calibrated overall but miscalibrated for a specific subgroup will produce biased decisions for that subgroup. |
| **Sensitivity analysis** | NB16 tests threshold robustness to cost assumptions. NB17 tests threshold robustness to group membership — the same threshold may produce very different selection rates for different groups. |

**In short:** NB16 teaches *how* to set a threshold. NB17 teaches *whether* that threshold is fair.

If you tried to teach fairness before thresholds, students would not understand how group-level selection rates, TPR, and FPR are mechanically produced by the threshold choice. If you skipped NB17 entirely, students would deploy models without ever checking whether they discriminate — which is both professionally irresponsible and increasingly illegal.

---

## 3. Why It Must Come Before Notebook 18

Notebook 18 covers **reproducibility, monitoring, and deployment readiness**. Its learning objectives assume students have:

- Audited the model for group-level performance disparities
- Written a model card documenting limitations and ethical considerations
- Understood that deployment creates feedback loops that can amplify bias

NB18's deployment checklist includes "model card completed" and "fairness audit documented" as prerequisites. Without NB17, those checklist items would be blank. More importantly, NB18's monitoring plan must include fairness signals (e.g., tracking selection rates by group over time), which only makes sense if students understand what those signals measure.

The deliberate sequence is:

```
NB16: Set the business-optimal threshold + Calibrate probabilities
            |
NB17: Audit the threshold for fairness + Write the model card
            |
NB18: Package for deployment + Define monitoring plan (including fairness monitoring)
```

Each notebook adds one layer of deployment readiness. By NB18, the model has been optimized (NB16), audited (NB17), and is ready to be packaged and monitored.

---

## 4. Why These Specific Libraries and Tools

### 4.1 `sklearn.datasets.make_classification` with Synthetic Group Attribute

**What it does:** Generates a classification dataset. After generation, a synthetic `group` column is added using `np.random.choice` with 60/40 proportions.

**Why we use a synthetic group (instead of real demographic data):**

1. **Ethical data handling.** Real demographic datasets carry privacy risks and historical biases that are complex to navigate in a classroom setting. Synthetic data lets students practice the methodology without handling sensitive personal information.

2. **Controllable setup.** By generating the group independently from the features, we create a scenario where the model has no access to group membership during training. Any performance disparity that emerges must come from correlations between the features and the group — exactly the proxy-variable phenomenon students need to understand.

3. **Reproducibility.** With `RANDOM_SEED = 474`, the group assignments are identical every time. Students can compare results across runs and across each other's notebooks.

**What to emphasize on camera:** The group is excluded from training features. The model never "sees" group membership. Yet it may still produce different outcomes for different groups because some features act as proxies for group membership. This is the central lesson of algorithmic fairness: exclusion is not the same as independence.

### 4.2 `sklearn.metrics.confusion_matrix` (Per-Group)

**What it does:** Computes TP, FP, FN, TN counts for each group separately, enabling the calculation of per-group selection rates, TPR, and FPR.

**Why we use it:**

1. **Selection rate (demographic parity).** The fraction of each group predicted positive. If Group A is selected at 40% and Group B at 25%, there is a selection rate disparity. This metric does not depend on ground-truth labels — it only depends on the model's predictions.

2. **True positive rate (equal opportunity).** The fraction of actual positives correctly identified in each group. If the model catches 95% of positives in Group A but only 85% in Group B, qualified individuals in Group B are systematically disadvantaged.

3. **False positive rate (part of equalized odds).** The fraction of actual negatives incorrectly flagged in each group. If Group B receives more false alarms, its members bear a disproportionate burden of unnecessary interventions.

**What to emphasize on camera:** These three metrics capture different aspects of fairness and they cannot all be satisfied simultaneously (except in trivial cases). Choosing which metric to prioritize is a *values* decision, not a technical one.

### 4.3 Slice-Based Evaluation

**What it does:** Computes accuracy, precision, recall, and F1 for each group separately and displays them side by side.

**Why we use it:**

- **Same methodology as NB15.** In NB15, students sliced by income quartiles to find error patterns. Here, they slice by group membership. The code is nearly identical — the conceptual leap is that the segments now represent people, not property values.

- **Visual comparison.** The grouped bar chart immediately reveals any metric where the bars differ meaningfully between groups. Stakeholders can see the disparity at a glance.

- **Connects to model cards.** The "Performance by Group" section of the model card template requires exactly these numbers. Computing them here populates the model card in the next section.

**What to emphasize on camera:** Slice-based evaluation is not a fairness-specific technique — it is the same error analysis methodology from NB15. What makes it a fairness analysis is the choice of slicing dimension: demographic groups instead of feature quartiles.

### 4.4 Model Card Template

**What it does:** Provides a structured Markdown template following Mitchell et al. (2019): model details, intended use, training data, evaluation data, metrics (overall and by group), limitations, ethical considerations, and recommendations.

**Why we use it:**

1. **Industry standard.** Google, Microsoft, Hugging Face, and other organizations publish model cards with their released models. Students who know how to write one are more employable and more responsible practitioners.

2. **Forced disclosure.** The template has sections for "Out-of-Scope Uses," "Limitations," and "Ethical Considerations" that require honest self-assessment. Students cannot skip these sections without leaving visible gaps in the card.

3. **Connects interpretation to deployment.** The model card aggregates artifacts from multiple notebooks: performance metrics (NB14), interpretation (NB15), threshold recommendation (NB16), and fairness audit (NB17). It is the final pre-deployment document.

**What to emphasize on camera:** A model card is not optional documentation — it is a professional deliverable. If you deploy a model without one, you are asking stakeholders to trust a black box. The card makes your model's capabilities, limitations, and risks transparent.

---

## 5. Key Concepts to Explain on Camera

### 5.1 Fairness Metrics Conflict — You Must Choose

The three most common fairness criteria — demographic parity, equal opportunity, and equalized odds — are mathematically incompatible except when the base rates are equal across groups or when the model is perfect.

- **Demographic parity** says both groups should be selected at the same rate: P(Y_hat=1 | Group=A) = P(Y_hat=1 | Group=B).
- **Equal opportunity** says both groups should have the same true positive rate: P(Y_hat=1 | Y=1, Group=A) = P(Y_hat=1 | Y=1, Group=B).
- **Equalized odds** says both groups should have the same TPR *and* FPR.

If the base rates differ (e.g., 50% of Group A is positive but only 30% of Group B is positive), achieving demographic parity requires selecting some Group B negatives — which worsens FPR for Group B. Achieving equal opportunity allows different selection rates — which violates demographic parity.

**What to emphasize on camera:** This is not a technical failure; it is a mathematical impossibility. The choice of which fairness metric to prioritize is a values decision that depends on the deployment context. In hiring, you might prioritize equal opportunity (give qualified candidates equal chances). In lending, you might prioritize equalized odds (equal FPR to avoid discriminatory denials). There is no universally "correct" answer.

### 5.2 Excluding the Sensitive Attribute Is Not Enough

A common misconception: "If we do not include race/gender/age in the model, the model cannot discriminate." This is false. Features like zip code, education level, income, and name correlate with sensitive attributes. These are **proxy variables** — they carry information about group membership even when the sensitive attribute is excluded.

In the notebook, the group is randomly assigned and not correlated with the features, so proxy effects are minimal. But in real-world data, proxies are pervasive. The lesson is that fairness requires active auditing, not passive exclusion.

**Analogy for students:** Imagine a hiring model that does not use gender but does use "years of work experience." If women in the training data have systematically fewer years of experience (due to career gaps for caregiving), the model will penalize women without ever "seeing" their gender. The proxy carries the bias.

### 5.3 Feedback Loops Amplify Bias

When a model's predictions influence future data, a feedback loop forms. Example: a crime prediction model sends more police to high-crime neighborhoods. More police presence leads to more arrests. More arrests increase the "crime rate" in the data. The next model iteration reinforces the pattern.

This dynamic means that a model deployed with a small fairness disparity can produce a large disparity over time. NB17 introduces this concept; NB18 addresses it through monitoring.

**What to emphasize on camera:** Fairness is not a one-time audit — it is an ongoing monitoring responsibility. Even a model that is fair at deployment can become unfair through feedback loops. That is why NB18's monitoring plan includes fairness signals.

### 5.4 Model Cards as Professional Communication

The model card combines technical performance data with ethical assessment into a single document. It is designed for multiple audiences: data scientists (who read the metrics), business stakeholders (who read the intended use and limitations), compliance teams (who read the ethical considerations), and future developers (who read the recommendations and update schedule).

**What to emphasize on camera:** Writing a model card is not busywork — it is the deliverable that accompanies every deployed model in responsible AI organizations. Getting comfortable writing one now will serve you in every data science role you hold.

---

## 6. What Students Should Take Away

After watching your video and completing this notebook, students should be able to answer these questions confidently:

1. **"Why can an overall-accurate model still be unfair?"** — Because aggregate metrics average over all groups. A model with 95% overall accuracy might have 98% accuracy for one group and 88% for another. The 88% group experiences worse predictions, worse decisions, and potential harm — even though the overall metric looks fine.

2. **"What is the difference between demographic parity and equal opportunity?"** — Demographic parity requires equal selection rates across groups (same fraction predicted positive). Equal opportunity requires equal true positive rates (same fraction of actual positives correctly identified). When base rates differ, these two criteria conflict: satisfying one necessarily violates the other.

3. **"Why does excluding the sensitive attribute not prevent discrimination?"** — Because other features (zip code, income, education) can serve as proxies for the sensitive attribute. The model learns the correlations in the data, and if those correlations reflect historical bias, the model reproduces that bias — regardless of whether the sensitive attribute is an explicit input.

4. **"What is a model card and why does it matter?"** — A model card is a structured document that accompanies a deployed model, disclosing its training data, performance metrics (overall and by group), limitations, ethical considerations, and usage recommendations. It makes the model transparent and accountable. It is required by many organizations and increasingly by regulation.

5. **"What should I do if I find a fairness disparity?"** — Document it honestly in the model card. Investigate the cause (is it proxy features? base rate differences? sample size imbalance?). Discuss with stakeholders which fairness criterion matters most for the deployment context. Consider mitigations: threshold adjustment per group, resampling, feature removal, or human-in-the-loop review for the affected group. Monitor the disparity over time.

---

## 7. Common Student Questions (Anticipate These)

**Q: "If the group is randomly assigned in this notebook, why would we see any performance disparity?"**
A: Because random assignment with finite data produces small sample-level differences in feature distributions. These differences are noise, not systematic discrimination. In real-world data, the disparities are much larger because group membership is correlated with features through historical and structural factors. The notebook uses random groups to teach the methodology safely; your project should apply it to real groups.

**Q: "Can't we just use different thresholds for different groups to equalize the metrics?"**
A: Technically yes — this is called group-specific threshold tuning. It can equalize TPR or FPR across groups. However, it raises its own ethical questions: is it fair to use a lower bar for one group? In some contexts (affirmative action, accessibility accommodations) the answer is yes. In others (medical diagnosis, criminal justice) it is more contentious. The point is that this is a policy decision, not a technical one.

**Q: "Which fairness metric is the 'right' one?"**
A: There is no universally correct fairness metric — Chouldechova (2017) proved that several common metrics cannot all be satisfied simultaneously unless the model is perfect or the base rates are equal. The right metric depends on the deployment context: what kind of harm are you most concerned about? Equal opportunity is appropriate when you want to ensure qualified individuals are treated equally. Demographic parity is appropriate when you want to ensure equal representation in outcomes. Equalized odds is appropriate when you want to equalize both benefits and burdens.

**Q: "Does this only apply to protected characteristics like race and gender?"**
A: No. Fairness analysis can be applied to any dimension you care about: age groups, geographic regions, income levels, language spoken, disability status. NB15's segment analysis by income quartile is structurally identical to NB17's fairness analysis by demographic group. The methodology is the same; the ethical weight depends on the dimension.

**Q: "What if we find no disparity — can we skip the model card?"**
A: No. The absence of disparity is itself a finding that should be documented — along with the groups tested, the metrics computed, and the limitations of the analysis (e.g., "we tested two groups; other groupings may reveal disparities"). A clean fairness audit is valuable evidence that the model is ready for deployment.

---

## 8. Connection to the Broader Course Arc

| Week | Notebooks | What NB17's Fairness Skills Enable |
|---|---|---|
| **Week 1** | 01-05 | Students learned to split data and prevent leakage. NB17 extends the "protect what matters" principle from protecting the test set to protecting affected populations. |
| **Week 2** | 06-10 | Classification metrics were introduced. NB17 applies the same metrics (precision, recall, F1) at the group level, showing that aggregate reporting is insufficient. |
| **Week 3** | 11-15 | NB15 introduced segment error analysis (by income). NB17 extends the same technique to demographic groups, where the stakes are ethical, not just statistical. |
| **Week 4** | 16-20 | NB16 sets the threshold; NB17 audits it for fairness. NB18 deploys the model with the fairness audit as part of the artifact package. NB19 uses the model card in the executive narrative. NB20 includes fairness assessment in the final project rubric. |

**Fairness is the ethical capstone of the analytical pipeline.** NB14-16 ensure the model is accurate, interpretable, and cost-optimized. NB17 ensures it is responsible. Together, they produce a model that a business can deploy with confidence — knowing not just how well it performs, but for whom it performs well and where it might cause harm.

---

## 9. Suggested Video Structure

Below are two recording options. Each segment includes the **speaking prompt** (what to say), **timestamps**, and the **notebook cells** to show on screen.

All cell references use the format `Cell N` where N is the zero-indexed cell number in `17_fairness_slicing_model_cards.ipynb`.

---

### Option A: Single Video (~12 minutes)

#### Segment 1 — Opening & Why Fairness Matters `[0:00–2:30]`

> **Show:** Cell 0 (header) and Cell 1 (learning objectives) on screen.

**Say:**

"Welcome back. Over the last few notebooks we have built a champion model, interpreted its behavior, set a cost-optimal threshold, and calibrated its probabilities. The model is analytically complete. But is it responsibly complete? Today we ask a question that aggregate metrics cannot answer: does this model treat all groups of people fairly? A model with 95 percent overall accuracy might have 98 percent accuracy for one group and 88 percent for another. That 10-point gap is invisible in the aggregate number, but it is very real to the people in the 88 percent group. Today we learn to detect those gaps, measure them, and document them honestly."

> **Action:** Scroll through the 5 learning objectives in Cell 1, briefly reading each one aloud.

> **Show:** Cell 5 (fairness vocabulary). Walk through the key concepts.

**Say:**

"Let me define four terms you need to know. First, disparity: a systematic difference in outcomes across groups. Second, harm: allocation harm means withholding opportunities from a group; quality-of-service harm means giving worse predictions to a group. Third, proxy variables: features that correlate with sensitive attributes even when the attribute is excluded. Zip code correlates with race. Years of experience correlates with gender. The model does not need to see the sensitive attribute directly to discriminate — proxies carry the bias. Fourth, feedback loops: when a model's decisions influence future data, small biases can amplify over time.

Now the fairness metrics. Demographic parity: equal selection rates across groups. Equal opportunity: equal true positive rates. Equalized odds: equal TPR and FPR. Here is the critical insight: these metrics conflict with each other. Unless the model is perfect or both groups have identical base rates, you cannot satisfy all three simultaneously. That means fairness requires choosing which criterion matters most — and that is a values decision, not a technical one."

---

#### Segment 2 — Data & Model Setup `[2:30–4:00]`

> **Run** Cell 7 (generate data with group attribute).

**Say:**

"We generate 2,000 samples with a synthetic group attribute: 60 percent Group A, 40 percent Group B. The group column is added after feature generation and is deliberately not used in training. This mirrors real-world practice where you have a sensitive attribute in your data but exclude it from the model. The cross-tabulation shows the target distribution by group — check whether the base rates differ between groups. If they do, that difference will drive some of the fairness trade-offs."

> **Run** Cell 10 (train model without group + overall metrics).

**Say:**

"We train a Random Forest on the 15 numeric features only — the model never sees group membership. Overall performance: accuracy, precision, recall, and F1 on the test set. These numbers set the baseline. They look good. But the question is: do they look equally good for both groups?"

---

#### Segment 3 — Slice-Based Evaluation `[4:00–7:00]`

> **Run** Cell 13 (performance by group table + disparities).

**Say:**

"Here is where we break the aggregate. We compute accuracy, precision, recall, and F1 separately for Group A and Group B. The disparity section below shows the signed gap for each metric. Pay close attention to the recall gap — it tells you whether qualified individuals in one group are more likely to be missed. Even a 3-point gap in recall, when applied to thousands of cases, can mean hundreds of people in one group are denied opportunities that they deserve.

Are these gaps statistically significant? On synthetic data with random group assignment, the gaps are mostly noise — small and inconsistent. On real data with correlated groups, the gaps are often larger and systematic."

> **Run** Cell 15 (performance bar chart).

**Say:**

"The grouped bar chart makes the disparity visual. Look for any metric where the bars are noticeably different heights. On this dataset, the bars should be close because the group is randomly assigned. But imagine a real-world scenario where Group B's recall bar is visibly shorter — that means the model misses more positive cases in Group B. That chart is what you would show a stakeholder to explain why fairness monitoring matters."

> **Mention:** "Pause the video here and complete Exercise 1 — identify the largest performance gap and propose a hypothesis for why it exists." Point to Cell 17 (PAUSE-AND-DO 1).

---

#### Segment 4 — Fairness Metrics `[7:00–9:00]`

> **Run** Cell 20 (selection rate, TPR, FPR by group + gaps).

**Say:**

"Now the formal fairness metrics. Selection rate: the fraction of each group predicted positive — this is demographic parity. TPR: the fraction of actual positives correctly caught in each group — this is equal opportunity. FPR: the fraction of actual negatives incorrectly flagged. The gaps are printed below.

Let me interpret these. A positive selection rate gap means Group A is selected more often. A positive TPR gap means Group A's positives are caught more often — Group B's qualified individuals are more likely to be missed. A positive FPR gap means Group A's negatives are flagged more often — Group A bears more false alarm burden.

The three gaps often point in different directions. This is the impossibility result in action: you cannot equalize selection rate, TPR, and FPR simultaneously. The question is which gap matters most for your deployment context."

---

#### Segment 5 — Model Card `[9:00–11:00]`

> **Show:** Cell 22 (model card explanation), then **scroll** through Cell 23 (model card template).

**Say:**

"The model card is the deliverable that ties everything together. It has sections for model details, intended use, training data, evaluation data, overall and group-level metrics, limitations, ethical considerations, and recommendations. Let me highlight the sections you cannot skip.

Intended Use tells the reader what the model is designed for — and critically, what it is *not* designed for. Out-of-scope uses are essential because someone will inevitably try to repurpose the model for a context it was never validated on.

Limitations must be specific and evidence-based. Do not write 'the model has some limitations.' Write 'the model achieves 88 percent recall for Group B compared to 95 percent for Group A; predictions for Group B should receive additional human review.'

Ethical Considerations requires you to name potential harms. Allocation harm: who might be denied opportunities? Quality-of-service harm: who might receive worse predictions? Feedback loops: how might the model's decisions influence future data? Naming these harms is the first step toward mitigating them."

> **Mention:** "Pause the video here and complete Exercise 2 — draft a model card limitations section for your project model, 6 to 8 lines, specific and evidence-based." Point to Cell 24 (PAUSE-AND-DO 2).

---

#### Segment 6 — Wrap-Up & Closing `[11:00–12:00]`

> **Show:** Cell 26 (Wrap-Up).

**Say:**

"Let me leave you with three takeaways. First: aggregate metrics hide group-level disparities. Always evaluate performance by relevant demographic groups before deployment. Second: fairness metrics conflict — you must choose which criterion matters most, and that choice depends on the deployment context, not on math. Third: document everything in a model card. Perfect fairness is impossible, but honest documentation of limitations is essential.

This notebook marks a turning point. Everything before today was about building an accurate, efficient model. Today we asked whether that model is fair and responsible. From here, we move to deployment: packaging the model with its artifacts, including this fairness audit and model card, and defining monitoring plans that track fairness over time. See you in the next notebook."

> **Show:** Cell 27 (Submission Instructions) briefly, then Cell 28 (Thank You).

---

### Option B: Three Shorter Videos (~4 min each)

---

#### Video 1: "Why Fairness Matters and Slice-Based Evaluation" (~5 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–0:45` | Show header + objectives | Cell 0, Cell 1 | "Welcome back. Our model is analytically complete — selected, interpreted, threshold-optimized. Today we ask: is it fair? Here are our five learning objectives." (Read them briefly.) |
| `0:45–1:45` | Show fairness vocabulary | Cell 5 | "Four terms: disparity, harm, proxy variables, and feedback loops. And three fairness metrics: demographic parity, equal opportunity, equalized odds. Critical insight: these metrics conflict — you cannot satisfy all three simultaneously. Choosing which one matters most is a values decision." |
| `1:45–2:30` | Run data + model | Cell 7, Cell 10 | "Synthetic data, 2,000 samples, two groups, group excluded from training. Overall performance looks strong. The question is: does it look equally strong for both groups?" |
| `2:30–4:15` | Run slice evaluation | Cell 13, Cell 15 | "We compute accuracy, precision, recall, and F1 for each group separately. The disparity table shows signed gaps. The bar chart makes them visual. On synthetic data, gaps are small noise. On real data, they would be larger and systematic. Even a 3-point recall gap affects hundreds of people at scale." |
| `4:15–5:00` | Exercise 1 prompt | Cell 17 | "Pause the video now and complete Exercise 1. Identify the largest performance gap between groups and propose a hypothesis for why it exists. Come back when you are done." |

---

#### Video 2: "Fairness Metrics and the Impossibility Trade-Off" (~4 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–0:30` | Recap | — | "Welcome back. We found performance differences between groups. Now we formalize those differences into standard fairness metrics." |
| `0:30–2:00` | Run fairness metrics | Cell 19, Cell 20 | "Three metrics per group: selection rate, TPR, and FPR. Selection rate measures demographic parity — are both groups selected at equal rates? TPR measures equal opportunity — do qualified individuals have equal chances? FPR measures burden — are false alarms distributed equally? The gaps are printed below. They often point in different directions: equalizing selection rate may widen the TPR gap, and vice versa." |
| `2:00–3:30` | Explain impossibility | — (use Cell 5 as reference) | "This is the impossibility result. Unless both groups have identical base rates or the model is perfect, you cannot equalize selection rate, TPR, and FPR simultaneously. This means fairness is a choice, not a computation. In hiring, you might prioritize equal opportunity. In lending, you might prioritize equalized odds. In marketing, you might prioritize demographic parity. Document which criterion you chose and why." |
| `3:30–4:00` | Transition | — | "Now that we have the metrics, we need to communicate them. That is what the model card does." |

---

#### Video 3: "Model Cards, Documentation & Wrap-Up" (~4 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–1:30` | Walk through model card | Cell 22, Cell 23 | "The model card follows Mitchell et al. 2019. Sections: model details, intended use and out-of-scope uses, training data, evaluation data, overall and per-group metrics, limitations, ethical considerations, and recommendations. Three sections you cannot skip: Out-of-Scope Uses — name the contexts where the model should not be used. Limitations — be specific: 'recall drops to 88% for Group B.' Ethical Considerations — name potential allocation and quality-of-service harms." |
| `1:30–2:30` | Exercise 2 prompt | Cell 24, Cell 25 | "Pause and complete Exercise 2. Draft a limitations section for your project model — 6 to 8 lines, specific and evidence-based. Include data limitations, model limitations, and ethical considerations. Come back when you are done." |
| `2:30–3:30` | Show wrap-up | Cell 26 | "Three rules. One: always evaluate performance by demographic group, not just in aggregate. Two: fairness metrics conflict — choose which one matters and document why. Three: write a model card for every deployed model. Perfect fairness is impossible, but honest documentation is essential." |
| `3:30–4:00` | Closing | Cell 27, Cell 28 | "Everything before today built an accurate model. Today we asked whether it is responsible. Next notebook: deployment — packaging the model with its artifacts, defining monitoring plans, and preparing for production. Submit your notebook to Brightspace and complete the quiz. See you there." |

---

*This guide was created to support video lecture recording for MGMT 47400 — Predictive Analytics, Purdue University.*
