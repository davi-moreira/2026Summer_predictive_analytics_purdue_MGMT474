# Video Lecture Guide: Notebook 18 — Deployment Thinking: Reproducibility, Monitoring, and Don't Ship a Notebook

## At a Glance

This guide covers:

1. **Why NB18 exists** — It teaches students to transition from "my model works in a notebook" to "my model can be saved, loaded, verified, and monitored in production" — the bridge between analytical correctness and operational reliability

2. **Why it must follow NB17** — NB17 introduces fairness, slicing, and model cards — the *ethical* layer of responsible deployment. NB18 adds the *operational* layer: reproducibility, serialization, and monitoring. Together they form the complete pre-deployment checklist

3. **Why it must precede NB19** — NB19 asks students to build an executive narrative and a conference video. That narrative is credible only if the student can demonstrate that the model is reproducible and monitorable. NB18 gives them the artifacts and the language to make that claim

4. **Why each library/tool is used:**
   - `joblib` — serializes the entire fitted pipeline (preprocessing + model) into a single file for reproducible inference
   - `Pipeline` — ensures preprocessing travels with the model so that a loaded artifact applies the same transformations
   - `json` — externalizes configuration and metrics into human-readable, version-controllable files
   - `LogisticRegression` — deliberately simple model so focus stays on deployment mechanics, not model complexity
   - `datetime` — timestamps every run for audit trails

5. **Common student questions** with answers (e.g., why not pickle?, what if the package version changes?, how often should I retrain?)

6. **Connection to the full course arc** — how NB18 transforms the pipeline template from NB02 into a deployment-ready artifact and prepares students for the communication and submission tasks of NB19 and NB20

7. **Suggested video structure** — both single-video (~12 min) and three-video (~4 min each) formats with speaking prompts, timestamps, and notebook cell references

---

## Purpose of This Document

This guide is your **lecture companion** for recording the video(s) that accompany `18_reproducibility_monitoring.ipynb`. It explains the *why* behind every design choice in the notebook — why it exists, why it sits between notebooks 17 and 19, why it uses specific libraries, and why concepts like model serialization and drift monitoring are introduced at this point in the course. Use it as speaking notes, not as a script: the goal is to internalize the reasoning so you can explain it naturally on camera.

---

## 1. Why This Notebook Exists

By notebook 17, students have built, tuned, evaluated, and even audited their models for fairness. They have a working pipeline, solid metrics, and a model card. But everything lives inside a Jupyter notebook — a fragile, session-dependent environment where a single kernel restart wipes the trained model from memory.

Notebook 18 answers the question every student should be asking at this point: **"My model works — now how do I make sure it keeps working after I close this notebook?"**

The answer has three parts:

1. **Refactor into functions.** Move from loose cells to `train_model()`, `predict()`, and `evaluate()` functions with explicit configuration. This separates *what* from *how* and makes the workflow callable, testable, and auditable.

2. **Save and load artifacts.** Use `joblib` to serialize the entire fitted pipeline — including the scaler's learned parameters — into a single file. Save the configuration and metrics alongside it in JSON. Then prove reproducibility by loading the artifact and verifying that predictions match the original.

3. **Plan for monitoring.** Define the signals (data drift, performance drift, calibration drift) that tell you when the model is decaying. Build a monitoring plan with thresholds, frequencies, and owners. This is where the notebook shifts from "data science" to "data engineering" — and it is where most students have never been asked to think.

This notebook is deliberately *not* about deploying to a cloud service or building an API. That would require infrastructure beyond the scope of a 4-week course. Instead, it focuses on the **thinking** that must happen before deployment: reproducibility, artifact management, and monitoring design. These are the skills that separate a class project from a production system.

---

## 2. Why It Must Come After Notebook 17

Notebook 17 establishes the **ethical and documentation** layer of responsible deployment:

| NB17 Concept | How NB18 Uses It |
|---|---|
| **Fairness metrics (TPR gap, selection rate)** | NB18's monitoring plan includes fairness metrics as ongoing signals. Students have already computed these once; now they learn to track them over time. |
| **Sliced evaluation** | NB18's drift detection extends slicing to a temporal dimension — checking whether performance on each segment degrades over time, not just at training time. |
| **Model cards** | NB17 teaches students to document a model's intended use, limitations, and ethical considerations. NB18 adds the *operational* documentation: configuration files, artifact manifests, and production-readiness checklists. Together, they form a complete model dossier. |
| **Responsible AI mindset** | NB17 establishes that "shipping a model" is a responsibility, not just a task. NB18 operationalizes that responsibility with concrete tools: saved artifacts, monitoring plans, and pre-deployment checklists. |

**In short:** NB17 asks "Is this model fair and well-documented?" NB18 asks "Is this model reproducible, persistent, and monitorable?" Both questions must be answered before the model is ready for the executive narrative in NB19.

If you tried to teach monitoring *before* NB17, students would not understand why fairness metrics belong in the monitoring plan. If you skipped NB18 entirely, students would present a model in NB19 that they cannot prove is reproducible — undermining the credibility of their entire narrative.

---

## 3. Why It Must Come Before Notebook 19

Notebook 19 asks students to build an **executive narrative** — a slide deck and a conference video that communicate their project's value to a non-technical audience. That narrative must include:

- A reproducibility claim: "You can rerun this analysis and get the same results."
- A deployment plan: "Here is how we would put this model into production."
- A monitoring strategy: "Here is how we would know if the model stops working."

NB18 provides the artifacts and vocabulary to make each of these claims credible:

```
NB17: Is the model fair? Is it documented? (Model card, fairness audit)
            |
NB18: Is the model reproducible? Is it monitorable? (joblib, CONFIG, monitoring plan)
            |
NB19: Can I tell a compelling, credible story about this model? (Slides, video, narrative)
```

Without NB18, the executive narrative in NB19 would be aspirational rather than evidence-based. Students would say "we could deploy this" without having demonstrated *how*. NB18 gives them the receipts.

---

## 4. Why These Specific Libraries and Tools

### 4.1 `joblib`

**What it does:** Serializes Python objects — including fitted scikit-learn pipelines — to disk as binary files. `joblib.dump(pipeline, 'model.joblib')` saves; `joblib.load('model.joblib')` restores.

**Why we use it:**

1. **Pipeline completeness.** `joblib` saves the *entire* fitted pipeline, including every transformer's learned parameters (the scaler's mean and variance, the imputer's computed medians). When you load the pipeline, it applies the exact same preprocessing to new data. If you saved only the model coefficients, you would need to recreate the preprocessing separately — a common source of inference bugs.

2. **scikit-learn's recommended approach.** The scikit-learn documentation explicitly recommends `joblib` over `pickle` for objects containing large numpy arrays, because `joblib` compresses arrays more efficiently. This is the tool students will encounter in industry documentation.

3. **Simplicity.** Two lines of code: `joblib.dump(pipeline, path)` to save, `joblib.load(path)` to load. No configuration, no serialization boilerplate.

**What to emphasize on camera:** The critical test is not saving — it is *loading and verifying*. The notebook loads the pipeline, runs predictions on the same test data, and checks that results match the in-memory pipeline exactly. If they do not match, something was fitted outside the pipeline and therefore not saved. This is the most common serialization bug.

### 4.2 `json` (for configuration and metrics)

**What it does:** Reads and writes JSON files — human-readable, version-controllable text files for storing dictionaries.

**Why we use it:**

- **Configuration externalization.** The `CONFIG` dictionary holds all hyperparameters, file paths, split ratios, and metadata in one place. Saving it as JSON means anyone can inspect the exact settings used to train the model without reading the code.
- **Metrics logging.** Saving training, validation, and test metrics as JSON creates an audit trail. When someone asks "what was the validation F1 for this model?", the answer is in a file, not in a notebook cell that may have been re-run with different settings.
- **Human readability.** Unlike binary formats (joblib, pickle), JSON can be opened in any text editor, diffed in git, and reviewed in a code review.

**What to emphasize on camera:** The combination of `joblib` (for the pipeline) and `json` (for the config and metrics) gives you a complete, portable model package. You can hand these three files to a colleague, and they can reproduce your predictions without access to your notebook or your Python environment.

### 4.3 `sklearn.pipeline.Pipeline` (revisited from NB02)

**What it does:** Chains preprocessing and modeling into a single object.

**Why it reappears here:**

In NB02, the Pipeline was introduced as a leakage-prevention tool. In NB18, we reveal its second purpose: **deployment readiness**. A pipeline is the *unit of deployment* — the thing you save, load, and call `.predict()` on. If preprocessing lives outside the pipeline, it cannot be serialized with `joblib`, and the saved model is incomplete.

**What to emphasize on camera:** This is the payoff of the decision we made in NB02. Students who have been building pipelines for 16 notebooks now see why that discipline matters: the pipeline is what makes `joblib.dump()` work correctly. No pipeline, no reproducible deployment.

### 4.4 `sklearn.linear_model.LogisticRegression`

**What it does:** Fits a logistic regression classifier.

**Why we use it here:**

The model is deliberately simple. NB18 is about deployment *mechanics*, not model *performance*. Using logistic regression keeps the training fast, the artifact small, and the focus on serialization and monitoring rather than on model tuning. Students already know how to build complex models from NB08-NB16; here they learn how to ship one.

### 4.5 `datetime`

**What it does:** Provides timestamps for recording when training, saving, and loading occur.

**Why we use it:**

Timestamps create an audit trail. When a monitoring alert fires six months after deployment, the team needs to know *when* the model was trained, *when* it was last evaluated, and *when* the config was last changed. Including timestamps in the setup cell and in the saved metadata is a small habit with large operational value.

---

## 5. Key Concepts to Explain on Camera

### 5.1 "Notebooks Are for Exploration. Functions Are for Production."

This is the central mantra of NB18. Loose cells in a notebook are fine for EDA and prototyping, but they are fragile: execution order matters, variables can be overwritten silently, and there is no interface contract. Wrapping training logic in `train_model(X_train, y_train, config)` — with explicit inputs, outputs, and a docstring — is the first step toward code that someone else (or your future self) can trust.

**Analogy for students:** A notebook cell is like a sticky note — useful for jotting ideas, but you would not hand it to a colleague and say "run this in production." A function with a clear signature is like a recipe: someone who has never seen your kitchen can follow it and get the same result.

### 5.2 The Three-File Artifact Package

A production-ready model is not one file — it is three:

| File | Contents | Format |
|---|---|---|
| `model_pipeline.joblib` | Fitted pipeline (preprocessing + model) | Binary |
| `model_config.json` | Hyperparameters, split ratios, seed, paths | JSON |
| `training_metrics.json` | Accuracy, precision, recall, F1, AUC for all splits | JSON |

Together, these files answer three questions: *What does the model do?* (pipeline), *How was it configured?* (config), and *How well does it perform?* (metrics). If any file is missing, the deployment is incomplete.

### 5.3 The Reproducibility Verification Test

Saving a model is necessary but not sufficient. You must also **verify** that the loaded model produces identical predictions. The notebook does this explicitly:

```python
original_preds = pipeline.predict(X_test[:5])
loaded_preds = loaded_pipeline.predict(X_test[:5])
assert np.array_equal(original_preds, loaded_preds)
```

If this assertion fails, something was fitted outside the pipeline — typically a scaler or an encoder that was applied manually before the pipeline step. This is the most common serialization bug, and catching it here prevents silent inference errors in production.

### 5.4 The Three Types of Drift

Models decay because the world changes. NB18 introduces three drift types:

1. **Data drift (covariate shift):** Feature distributions change. Example: average transaction amount increases after a price hike. Detection: Population Stability Index (PSI), Kolmogorov-Smirnov tests on feature distributions.

2. **Performance drift (concept drift):** The relationship between features and target changes. Example: customer behavior shifts after a competitor launches a new product. Detection: track accuracy, precision, recall on incoming labeled data.

3. **Calibration drift:** Predicted probabilities become unreliable. Example: the model says "80% likely" but the actual rate is 60%. Detection: compare predicted probabilities to observed frequencies over time.

**What to emphasize on camera:** Data drift is detectable without labels (you only need feature distributions). Performance drift requires labels, which may arrive with a delay. Calibration drift is the subtlest — the model's hard predictions may still be correct, but its probability estimates are off. A complete monitoring plan watches for all three.

### 5.5 The Production Readiness Checklist

The notebook ends with a 12-item checklist spanning Reproducibility, Testing, Monitoring, Documentation, and Security. Some items are checked (pipeline save/load verified, seeds documented, config externalized); others are open (input validation, unit tests, API documentation). The point is not to achieve 100% readiness in a course project — it is to *know what you are missing*. In industry, this checklist gates the transition from development to production.

---

## 6. What Students Should Take Away

After watching your video and completing this notebook, students should be able to answer these questions confidently:

1. **"Why can't I just re-run my notebook when I need predictions?"** — Because notebooks are session-dependent: a kernel restart wipes the trained model. Saving with `joblib` makes the model persistent and portable.

2. **"What is a CONFIG dictionary and why does it matter?"** — It externalizes all settings (hyperparameters, split ratios, paths) into one place. Changing the config changes the experiment without editing code. Saving it as JSON creates an audit trail.

3. **"How do I know my saved model is correct?"** — Load it and compare predictions to the in-memory version. If they match, serialization is correct. If they do not, something was fitted outside the pipeline.

4. **"What is data drift and why should I care?"** — Feature distributions change over time, which can degrade model performance silently. Monitoring drift signals tells you when to retrain before the model fails in production.

5. **"What does a monitoring plan include?"** — Signal name, metric, warning and critical thresholds, check frequency, and responsible owner. It is a contract that says "here is how we will know the model is still working."

---

## 7. Common Student Questions (Anticipate These)

**Q: "Why joblib instead of pickle?"**
A: `joblib` is optimized for objects containing large numpy arrays, which is exactly what a fitted scikit-learn pipeline contains. It compresses these arrays more efficiently than standard `pickle`. scikit-learn's own documentation recommends `joblib` for this reason.

**Q: "What if I upgrade scikit-learn after saving the model?"**
A: This is a real risk. A model saved with scikit-learn 1.3 may not load correctly in scikit-learn 1.5 if internal data structures changed. That is why the config file records the package version, and why production teams pin their dependency versions. For this course, we use Colab's default environment, which is consistent.

**Q: "How often should I retrain the model?"**
A: It depends on how fast your data changes. The monitoring plan defines triggers: when PSI exceeds 0.25, or when accuracy drops below 80% of baseline, it is time to retrain. Some models retrain weekly; others last years. The monitoring plan tells you which case you are in.

**Q: "Do I need all three artifact files for my course project?"**
A: Yes. The pipeline file proves reproducibility. The config file proves transparency. The metrics file proves performance. Together, they are a complete package that demonstrates production thinking — which is what NB18 is about.

**Q: "Why does the production readiness checklist have unchecked items?"**
A: Because a course project is not a production system. The exercise is to *identify* the gaps between your notebook and a production deployment. Knowing what remains undone — input validation, error handling, unit tests — is itself a sign of professional maturity.

**Q: "What is Population Stability Index (PSI)?"**
A: PSI measures how much a feature's distribution has shifted compared to the training distribution. PSI < 0.1 means no significant shift. PSI > 0.25 means the population has changed substantially and the model should be investigated. It is the most common drift detection metric in industry.

---

## 8. Connection to the Broader Course Arc

| Week | Notebooks | How NB18 Connects |
|---|---|---|
| **Week 1** | 01-05 | NB02 introduced the Pipeline as a leakage-prevention tool. NB18 reveals its second purpose: the pipeline is the *unit of deployment* — the thing you serialize, ship, and monitor. |
| **Week 2** | 06-10 | Students trained classification models (logistic regression, trees). NB18 shows how to save and verify those models so they survive a kernel restart. |
| **Week 3** | 11-15 | Students tuned hyperparameters and selected models. NB18 shows how to externalize those hyperparameters into a config file so experiments are auditable. |
| **Week 4** | 16-20 | NB16 introduced stacking/model selection. NB17 added fairness and model cards. NB18 adds reproducibility and monitoring. NB19-20 will use NB18's artifacts as evidence in the executive narrative and final submission. |

**NB18 is the operational bridge between "building a model" (Weeks 1-3) and "communicating and delivering a model" (NB19-20).** Everything built in the first 17 notebooks converges here into a deployable, monitorable, documented artifact.

---

## 9. Suggested Video Structure

Below are two recording options. Each segment includes the **speaking prompt** (what to say), **timestamps**, and the **notebook cells** to show on screen.

All cell references use the format `Cell N` where N is the zero-indexed cell number in `18_reproducibility_monitoring.ipynb`.

---

### Option A: Single Video (~12 minutes)

#### Segment 1 — Opening & Motivation `[0:00–1:30]`

> **Show:** Cell 0 (header) and Cell 1 (learning objectives) on screen.

**Say:**

"Welcome back. Over the past seventeen notebooks, you have built, tuned, evaluated, and even audited your models for fairness. But here is the problem: everything lives inside a Jupyter notebook. If you restart the kernel, your trained model vanishes. If you hand someone the notebook file, they have to re-run every cell to get predictions. That is not how production works. Today we fix that. We are going to learn three things: how to refactor notebook code into clean, reusable functions; how to save a model so it survives a kernel restart and produces identical predictions when loaded; and how to design a monitoring plan that tells you when your model starts to decay. By the end of this notebook, you will have a three-file artifact package — a pipeline, a config, and a metrics log — that is the foundation of a production deployment."

> **Action:** Scroll through the 5 learning objectives in Cell 1, briefly reading each one aloud.

---

#### Segment 2 — Setup & Configuration `[1:30–3:00]`

> **Show:** Cell 1 (setup explanation), then **run** Cell 2 (imports).

**Say:**

"Let us start with our standard setup. Notice a few new imports compared to earlier notebooks: `joblib` for model serialization, `json` for saving configuration files, and `datetime` for timestamping. These are not machine learning libraries — they are *deployment* libraries. They are what turns a notebook into a portable artifact."

> **Run** Cell 2. Point to the `Setup complete!` output and the timestamp.

> **Show:** Cell 3 (refactoring explanation), then **run** Cell 4 (CONFIG dictionary).

**Say:**

"Here is the first big idea: the CONFIG dictionary. Every setting that affects your model — the split ratio, the scaler type, the model type, the hyperparameters, the file paths — lives in this one dictionary. If you want to change the model from logistic regression to random forest, you change one line in the config. If you want to reproduce this exact experiment, you read the config. And when we save it as a JSON file later, anyone can inspect exactly how this model was trained without reading a single line of Python."

> **Run** Cell 4. Point to the pretty-printed JSON output.

---

#### Segment 3 — Data, Splits, and Function Refactoring `[3:00–5:30]`

> **Show:** Cell 5 (data loading explanation), then **run** Cell 6 (load breast cancer dataset).

**Say:**

"We use scikit-learn's breast cancer dataset for this demo — 569 samples, 30 features, binary classification. The point is not the dataset; it is the deployment pattern. Everything we do here applies identically to your project data."

> **Run** Cell 6. Point to the dataset shape and target distribution.

> **Run** Cell 8 (train/val/test split). Point to the 60/20/20 percentages.

**Say:**

"Standard 60-20-20 split, stratified by target. Notice that the split parameters come from CONFIG, not from hardcoded values. This is what 'configuration externalization' means in practice."

> **Show:** Cell 9 (train function explanation), then **run** Cell 10 (define `train_model()`).

**Say:**

"Now the refactoring step. Instead of fitting the model in a loose cell, we wrap it in a `train_model` function. This function reads the config, builds a pipeline with scaling and the specified model, fits it, and returns the fitted pipeline. The function has a clear signature: inputs are training data and config, output is a pipeline. Anyone can call this function and get the same result."

> **Run** Cell 12 (define `predict()`), then **run** Cell 14 (define `evaluate()`).

**Say:**

"Same pattern for prediction and evaluation. Each function does one thing. `predict` returns predictions and probabilities. `evaluate` calls `predict` internally, computes metrics, and returns them as a dictionary. Dictionary output — not print statements — is what makes these results saveable."

> **Run** Cell 16 (train and evaluate on all splits). Point to the three metric blocks.

**Say:**

"Now we run the full workflow: train on training data, evaluate on all three splits. Compare train to validation to test. Similar numbers mean no overfitting. These metrics will be saved to a file in the next section."

---

#### Segment 4 — Save, Load, Verify `[5:30–8:00]`

> **Show:** Cell 19 (save artifacts explanation), then **run** Cell 20 (save_model_artifacts function + call).

**Say:**

"Here is the core of reproducibility: saving artifacts. We save three files. First, `model_pipeline.joblib` — this is the entire fitted pipeline, including the scaler's learned mean and variance and the model's learned coefficients. Second, `model_config.json` — the configuration dictionary. Third, `training_metrics.json` — accuracy, precision, recall, F1, and AUC for every split. Together, these three files are a complete, self-contained model package. You can hand them to a colleague who has never seen your notebook, and they can load the pipeline and make predictions."

> **Run** Cell 20. Point to the three confirmation lines.

> **Show:** Cell 21 (load explanation), then **run** Cell 22 (load_model_artifacts + verification).

**Say:**

"But saving is only half the test. The real question is: does the loaded model produce the *same predictions* as the original? We load the pipeline, run it on five test samples, and compare the results to the in-memory pipeline. Look at the output: original predictions and loaded predictions are identical, and the match flag is True. This is the reproducibility verification test. If it ever fails, something was fitted outside the pipeline — a manual scaler, an encoder applied before the pipeline step — and that is a serialization bug you must fix before deployment."

> **Run** Cell 22. Point to the `Predictions match: True` line.

> **Mention:** "Pause the video here and complete Exercise 1 — modify the CONFIG to try a different model type and verify that save/load still works." Point to Cell 17 (PAUSE-AND-DO 1).

---

#### Segment 5 — Monitoring Plan `[8:00–10:30]`

> **Show:** Cell 24 (monitoring introduction and drift types explanation).

**Say:**

"Your model works today. But the world changes. Customer behavior shifts, market conditions evolve, data pipelines break. Without monitoring, you will not know your model has stopped working until someone complains — or worse, until bad decisions have already been made. There are three types of drift to watch for. Data drift: feature distributions change — for example, average transaction amounts increase after a price hike. Performance drift: accuracy degrades because the relationship between features and outcomes has shifted. Calibration drift: the model says eighty percent likely, but the true rate is only sixty percent. A complete monitoring plan watches for all three."

> **Run** Cell 25 (monitoring plan table). Walk through 2-3 rows.

**Say:**

"Here is a monitoring plan template with eight signals. Each row specifies the signal name, what type of drift it detects, the metric to track, warning and critical thresholds, check frequency, and the responsible owner. Look at the thresholds — PSI greater than 0.1 triggers a warning; greater than 0.25 is critical. Accuracy below ninety percent of baseline is a warning; below eighty percent is critical. These are industry conventions. For your project, you should customize the thresholds based on your domain's tolerance for degradation."

> **Mention:** "Pause the video here and complete Exercise 2 — draft a monitoring plan with 5-8 signals customized for your project." Point to Cell 27 (PAUSE-AND-DO 2).

---

#### Segment 6 — Production Checklist, Wrap-Up & Closing `[10:30–12:00]`

> **Show:** Cell 29 (notebook hygiene introduction). Briefly mention the three hygiene categories: technical, communication, reproducibility.

> **Run** Cell 36 (production readiness assessment). Point to the checked vs. unchecked items.

**Say:**

"Before we wrap up, look at this production readiness checklist. Twelve items across five categories: Reproducibility, Testing, Monitoring, Documentation, and Security. Five items are checked — we have done those in this notebook. Seven are still open — input validation, error handling, unit tests, logging infrastructure, model card, API documentation, input sanitization. The goal is not to check every box in a course project. The goal is to *know what you are missing*. In industry, this checklist gates the transition from development to production. Knowing the gaps is a sign of professional maturity."

> **Show:** Cell 39 (Wrap-Up).

**Say:**

"Let me leave you with the key takeaway: deployment is not the end — it is the beginning of model maintenance. Without monitoring and maintenance, even the best model decays. Today you learned three skills: refactoring into functions for reproducibility, saving and loading artifacts with joblib for persistence, and designing a monitoring plan for operational awareness. In the next notebook, we shift from the technical to the communicative. You will build an executive narrative — a slide deck and a video — that tells the story of your model to a non-technical audience. The artifacts you created today — your three-file package, your monitoring plan, your production checklist — become the evidence that makes your narrative credible. See you there."

> **Show:** Cell 40 (Submission Instructions) briefly, then the final Thank You cell.

---

### Option B: Three Shorter Videos (~4 minutes each)

---

#### Video 1: "From Notebooks to Functions: Refactoring for Reproducibility" (~4 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–0:45` | Show header + objectives | Cell 0, Cell 1 | "Welcome back. Your model works in a notebook — but if you restart the kernel, it vanishes. Today we fix that. We will learn to refactor code into functions, save model artifacts, and design a monitoring plan. Here are our five learning objectives." (Read them briefly.) |
| `0:45–1:15` | Run setup | Cell 1 (explanation), Cell 2 | "Standard setup with three new imports: joblib for serialization, json for configuration files, and datetime for timestamps. These are deployment tools, not modeling tools." |
| `1:15–2:00` | Show CONFIG | Cell 3, Cell 4 | "The CONFIG dictionary holds every setting in one place. Split ratios, model type, hyperparameters, file paths — all externalized. Change the config, change the experiment. Save the config, document the experiment. This is configuration externalization." |
| `2:00–2:45` | Load data + split | Cell 5, Cell 6, Cell 8 | "Breast cancer dataset, 569 samples, 30 features. Standard 60-20-20 split. Notice the split parameters come from CONFIG, not hardcoded values." |
| `2:45–3:30` | Define functions | Cell 9, Cell 10, Cell 12, Cell 14 | "We wrap training, prediction, and evaluation into three functions. Each has a clear signature: explicit inputs, explicit outputs, a docstring. This is the first step from exploration to production. A function can be tested, reused, and called from outside the notebook." |
| `3:30–4:00` | Train + evaluate | Cell 16 | "Run the full workflow: train on train, evaluate on all splits. Compare metrics across splits — similar numbers mean no overfitting. These metric dictionaries will be saved to JSON in the next video. Pause and complete Exercise 1 — try a different model type in CONFIG and compare." |

---

#### Video 2: "Save, Load, Verify: The Three-File Artifact Package" (~4 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–0:30` | Recap | — | "Welcome back. We have a trained pipeline and clean metric dictionaries. Now we make them permanent — save them to disk and verify that they load correctly." |
| `0:30–1:30` | Save artifacts | Cell 19, Cell 20 | "Three files. model_pipeline.joblib contains the entire fitted pipeline — preprocessing plus model. model_config.json records every setting. training_metrics.json logs performance on all splits. Together, these are a complete model package. You can hand them to a colleague, and they can reproduce your predictions without your notebook." |
| `1:30–2:30` | Load + verify | Cell 21, Cell 22 | "Here is the critical test. We load the pipeline, run it on five test samples, and compare to the original predictions. Look at the output: predictions match is True. If it ever says False, something was fitted outside the pipeline — a manual scaler, an encoder applied before the pipeline step. That is the most common serialization bug." |
| `2:30–3:30` | Reproducibility checklist | Cell 23 | "Walk through the reproducibility checklist. Pipeline includes all preprocessing? Check. Random seeds fixed? Check. Feature names recorded? Check. Model loads and produces identical predictions? Check. Configuration saved separately? Check. These seven items are the minimum standard for reproducibility." |
| `3:30–4:00` | Transition | — | "You now have a reproducible, persistent model. But models decay. In the next video, we design a monitoring plan to catch that decay before it causes harm." |

---

#### Video 3: "Monitoring, Production Readiness & Wrap-Up" (~4 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–0:45` | Drift introduction | Cell 24 | "Models decay because the world changes. There are three types of drift. Data drift: feature distributions change. Performance drift: accuracy degrades. Calibration drift: predicted probabilities become unreliable. A complete monitoring plan watches for all three." |
| `0:45–2:00` | Monitoring plan table | Cell 25 | "Here is a monitoring plan with eight signals. Prediction volume — is the pipeline still running? Feature availability — are inputs arriving correctly? Feature distribution — has the population shifted? Model accuracy and precision/recall — is performance degrading? Calibration — are probabilities still reliable? Business metric — is the model delivering value? Each signal has a warning threshold, a critical threshold, a check frequency, and a responsible owner." |
| `2:00–2:30` | Exercise 2 prompt | Cell 27 | "Pause and complete Exercise 2 — customize this monitoring plan for your project. Add at least two project-specific signals. Think about what features matter most and what business metrics your stakeholders care about." |
| `2:30–3:15` | Production checklist | Cell 36 | "Final check: the production readiness assessment. Twelve items, five categories. Five are done; seven remain. The point is not to check every box — it is to know what is missing. In industry, this checklist gates deployment. Knowing the gaps is professional maturity." |
| `3:15–4:00` | Wrap-Up + closing | Cell 39 | "Key takeaway: deployment is not the end — it is the beginning of model maintenance. You now have three skills: refactoring for reproducibility, artifact persistence with joblib, and monitoring design. In the next notebook, we shift from the technical to the communicative. You will build an executive narrative that tells your model's story. The artifacts and plans you created today become the evidence that makes that story credible. Submit your notebook to Brightspace and complete the quiz. See you there." |

---

*This guide was created to support video lecture recording for MGMT 47400 — Predictive Analytics, Purdue University.*
