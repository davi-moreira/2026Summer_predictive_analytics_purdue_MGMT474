# Video Lecture Guide: Notebook 20 — Final Delivery: Project Package Submission + Peer Review

## At a Glance

This guide covers:

1. **Why NB20 exists** — It is the capstone notebook of the entire course: students submit a complete deliverable package, review a peer's work using a structured rubric, and write a postmortem reflection. It wraps up 20 days of learning into a single, audited submission event

2. **Why it must follow NB19** — NB19 developed the slide outline, video script, and visual strategy. NB20 assumes those deliverables are *ready* and focuses on quality assurance (self-audit), submission logistics, peer evaluation, and reflection. Without NB19's preparation, students would arrive at NB20 with unfinished deliverables and no time to audit them

3. **Why it is the final notebook** — NB20 is the course's closing bracket. It completes the arc from NB01's first data split to NB20's final submission, peer review, and postmortem. Every skill from the course — technical, analytical, communicative — converges in the deliverable package. The peer review makes students *evaluators* as well as *practitioners*, and the postmortem cements learning through structured reflection

4. **Why each library/tool is used:**
   - `pandas` — builds audit checklists, rubric tables, and deliverable manifests as structured DataFrames
   - `json` — creates the artifact manifest as a machine-readable, version-controllable file
   - `datetime` — timestamps the submission for audit trail purposes
   - No scikit-learn, no matplotlib — this is purely a review, submission, and reflection notebook

5. **Common student questions** with answers (e.g., what if my notebook has errors?, how harsh should peer feedback be?, what if I run out of time?)

6. **Connection to the full course arc** — how NB20 brings the entire 20-notebook journey to a close and what students carry forward

7. **Suggested video structure** — both single-video (~12 min) and three-video (~4 min each) formats with speaking prompts, timestamps, and notebook cell references

---

## Purpose of This Document

This guide is your **lecture companion** for recording the video(s) that accompany `20_final_submission_peer_review.ipynb`. It explains the *why* behind every design choice in the notebook — why it exists as the final notebook, why it combines self-audit with peer review and reflection, and why each checklist and rubric is structured the way it is. Use it as speaking notes, not as a script: the goal is to internalize the reasoning so you can explain it naturally on camera.

---

## 1. Why This Notebook Exists

Twenty notebooks. Four weeks. From a first `train_test_split` in NB01 to a serialized, monitored, narratively packaged predictive analytics project. NB20 is where all of that comes together — and where students prove they can deliver a complete, professional product.

But NB20 is not just a submission portal. It serves three distinct pedagogical purposes:

1. **Self-audit.** Two comprehensive checklists — one technical (20 items), one communicative (15 items) — force students to systematically review their own work before submitting. This is not busywork; it is the professional practice of quality assurance. In industry, a pre-deployment checklist catches the errors that confidence misses.

2. **Peer review.** A structured rubric with five dimensions (Reproducibility, Methodology, Results, Communication, Responsible AI) and four scoring levels teaches students to evaluate analytical work critically and constructively. Reviewing someone else's project deepens understanding of one's own: you notice patterns, habits, and mistakes that are invisible in your own work.

3. **Postmortem reflection.** A nine-section reflection template asks students to articulate what worked, what did not, what they learned, and what they would do differently. Research on metacognition consistently shows that structured reflection improves long-term retention of both technical and process skills.

These three activities — audit, review, reflect — are not afterthoughts. They are the final exam of professional judgment. A student who can build a model, explain it, submit it cleanly, evaluate a peer's model fairly, and reflect honestly on their own process has demonstrated the full range of skills this course was designed to teach.

---

## 2. Why It Must Come After Notebook 19

Notebook 19 was preparation: storyboard, script, visuals, video planning. Notebook 20 is execution: submit, review, reflect.

| NB19 Deliverable | How NB20 Uses It |
|---|---|
| **Slide storyboard (11 slides)** | NB20's artifact manifest expects a finalized slide deck (PDF or PPTX). The storyboard from NB19 should have been transformed into actual slides by now. |
| **Video script (450-540 words)** | NB20's artifact manifest expects a recorded conference video (MP4). The script from NB19 should have been rehearsed and recorded. |
| **Visuals checklist (8 items)** | NB20's communication audit checks that all visualizations have titles, labels, and support the narrative. The checklist from NB19 ensures completeness. |
| **Deliverable package table** | NB20's artifact manifest mirrors NB19's package table but adds validation flags (links tested, permissions verified). NB19 planned the package; NB20 verifies it. |

**In short:** NB19 and NB20 form a preparation-execution pair. Trying to combine them into one notebook would be overwhelming — students need a full day to transform outlines into finished deliverables between the two notebooks.

If you tried to do peer review *before* NB19, students would not have a complete narrative package to review. If you skipped NB20 entirely, students would submit into a vacuum — no structured feedback, no quality assurance, no reflection.

---

## 3. Why It Is the Final Notebook

NB20 is the course's **closing bracket**, and it is designed to bring closure at three levels:

### 3.1 Technical Closure

The self-audit checklist walks students back through every technical skill they learned:

| Checklist Item | Notebook Where Skill Was Introduced |
|---|---|
| Run-all completes without errors | NB01 (first notebook workflow) |
| Train/val/test splits clearly labeled | NB01 (data splitting) |
| No test set leakage | NB02 (Pipeline, leakage prevention) |
| Baseline model included | NB03 (metrics and baselines) |
| Multiple models compared | NB06-NB10 (model development) |
| Hyperparameter tuning documented | NB14 (tuning) |
| Model saved with joblib | NB18 (reproducibility) |
| Configuration saved separately | NB18 (config externalization) |

Walking through this checklist is a retrospective tour of the entire course. Students see how each notebook contributed a specific, necessary skill to their final project.

### 3.2 Communicative Closure

The communication audit covers the narrative and visual skills from NB19:

- Clear title and introduction
- Logical section flow
- Visualizations with titles and labels
- Key findings highlighted
- Bibliography included

### 3.3 Reflective Closure

The postmortem template asks students to look back on the entire 4-week journey:

- **What worked well** — reinforces good habits
- **What did not work** — normalizes struggle and failure as learning
- **Key learnings** — forces articulation of tacit knowledge
- **What I would do differently** — connects experience to future practice
- **Most surprising finding** — highlights the unexpected insights that make data science engaging
- **Advice for future students** — creates a legacy of peer-to-peer wisdom

This reflective component is what distinguishes NB20 from a simple assignment submission page. It transforms the final notebook from a logistical task into a learning event.

### 3.4 Why No New Technical Content

NB20 introduces no new modeling techniques, no new libraries, no new concepts. This is deliberate. The final notebook of an intensive course should consolidate, not introduce. Adding new content would distract from the audit-review-reflect cycle and signal that the course was not finished — undermining students' sense of accomplishment.

---

## 4. Why These Specific Libraries and Tools

### 4.1 `pandas` (for checklists, rubrics, and manifests)

**What it does:** Creates DataFrames for the technical audit (20 items), communication audit (15 items), peer review rubric (5 dimensions), and deliverable manifest (6 items).

**Why we use it here:**

Checklists are more effective as structured tables than as free-form markdown. A DataFrame with columns for Category, Item, How to Check, Status, and Priority forces completeness and enables sorting by priority. Students can filter for CRITICAL items first, then HIGH, then MEDIUM — a triage approach that mirrors how quality assurance works in industry.

**What to emphasize on camera:** The checklist is not decorative. Walking through it item by item, checking boxes, is the professional practice of quality assurance. Skip it, and you will submit a notebook that fails Run-all because of a missing import in cell 3.

### 4.2 `json` (for the artifact manifest)

**What it does:** Saves the artifact manifest — project metadata, six deliverables with format and status, and five validation flags — as a JSON file.

**Why we use it here:**

The JSON manifest is a machine-readable submission receipt. It lists every deliverable, its required format, its submission link, and whether it has been submitted and validated. Saving it as JSON (rather than just printing it) means the student has a persistent record of what was submitted and when — useful if a submission dispute arises.

**What to emphasize on camera:** The manifest is your pre-flight checklist. Before clicking "Submit," open this file, walk through every deliverable, and test every link in an incognito browser window. The most common submission failure is a Colab link with incorrect sharing permissions.

### 4.3 `datetime` (for timestamping)

**What it does:** Prints the submission date at the top of the notebook.

**Why we use it here:**

The timestamp serves as a simple audit trail. When combined with the artifact manifest's `submission_date` field, it documents exactly when the final Run-all was executed — useful for verifying that the notebook was completed before the deadline.

### 4.4 Why No scikit-learn, No matplotlib

**This is deliberate.** NB20 is not a modeling notebook or a visualization notebook. It is an *audit, review, and reflection* notebook. The absence of modeling and plotting libraries signals that the technical work is complete — this is the wrap-up. Students who see scikit-learn in the imports might expect new techniques; the empty imports set the right expectation.

---

## 5. Key Concepts to Explain on Camera

### 5.1 The Self-Audit as Professional Practice

The technical audit has 20 items across six categories: Notebook Execution, Data & Splits, Modeling, Evaluation, Artifacts, and Code Quality. Each item has a priority level:

- **CRITICAL (5 items):** Run-all completes, outputs visible, no test leakage, final test results reported. If any CRITICAL item fails, the submission is fundamentally broken.
- **HIGH (10 items):** Data source documented, baseline included, metrics appropriate, model saved. These affect credibility and rigor.
- **MEDIUM (5 items):** No debug code, imports organized, functions documented. These affect professionalism.

**Analogy for students:** Think of it like a building inspection. CRITICAL items are structural (the foundation must be solid). HIGH items are functional (the plumbing must work). MEDIUM items are cosmetic (the paint should be neat). You would never skip the structural inspection to save time.

### 5.2 The Peer Review Rubric

The rubric has five dimensions with explicit weights:

| Dimension | Weight | What It Measures |
|---|---|---|
| Reproducibility | 20% | Can the notebook be rerun? Are outputs visible? |
| Methodology | 30% | Sound methods, proper validation, no leakage? |
| Results | 20% | Clear findings, well-supported, appropriate comparisons? |
| Communication | 20% | Clear narrative, effective visuals, professional presentation? |
| Responsible AI | 10% | Limitations acknowledged, monitoring plan, risk awareness? |

**Why Methodology is weighted highest:** Because the course is fundamentally about analytical rigor. A beautiful presentation of flawed analysis is worse than a rough presentation of sound analysis. Methodology weight signals that getting the analysis right matters most.

**Why Responsible AI is weighted lowest (but still present):** Because it is the newest skill in the course, introduced only in NB17. Students are still developing their responsible AI instincts. A 10% weight acknowledges the importance of the topic without penalizing students who are still learning to articulate limitations.

### 5.3 Actionable Feedback

The peer review form asks for three specific, actionable improvements — not vague praise or criticism. The template for each improvement is:

- **Specific issue:** What you observed
- **Suggested fix:** Concrete action to take
- **Expected improvement:** How this would help

**Bad feedback:** "Visuals could be better."
**Good feedback:** "The ROC curve on slide 6 is missing axis labels and an AUC annotation — adding those would strengthen credibility with the audience."

The difference is specificity and actionability. Good feedback tells the recipient exactly what to do and why it matters.

### 5.4 The Postmortem as Metacognition

The postmortem template has nine sections, but three are most important:

1. **What I would do differently** — This is where students connect experience to future practice. "I would start with simpler models and iterate" is a lesson that sticks because it comes from lived frustration, not a textbook.

2. **Most surprising finding** — This question surfaces the unexpected insights that make data science engaging. It also reveals whether the student engaged deeply with their data or just followed a recipe.

3. **Advice for future students** — This creates a legacy of peer wisdom. Reading advice from students who just completed the course is more persuasive than any instructor guidance.

### 5.5 The Course Arc in Retrospect

NB20's wrap-up section summarizes the entire course in four paragraphs:

- **Week 1 (Foundations):** Colab, EDA, preprocessing, pipelines, baselines, validation
- **Week 2 (Model Development):** Linear models, trees, ensembles, tuning, metrics
- **Week 3 (Advanced Topics):** Feature engineering, interpretation, imbalanced data, calibration
- **Week 4 (Production and Communication):** Reproducibility, deployment, narrative, peer review

This summary serves as a map students can return to when they need to recall where a specific skill was taught. It also provides a sense of accomplishment — four weeks is short, and seeing the full scope of what was covered reinforces that the course was comprehensive.

---

## 6. What Students Should Take Away

After watching your video and completing this notebook, students should be able to answer these questions confidently:

1. **"How do I know my submission is ready?"** — Run the 20-item technical audit and the 15-item communication audit. Clear all CRITICAL items first, then HIGH, then MEDIUM. Test every link in an incognito browser. Walk through the artifact manifest row by row.

2. **"How do I give useful peer feedback?"** — Be specific and actionable. Use the rubric to anchor scores. Identify the specific issue, suggest a concrete fix, and explain the expected improvement. Critique the work, not the person.

3. **"What does a good postmortem include?"** — Honest reflection on what worked, what did not, what you learned, and what you would change. Evidence-based, not vague. The postmortem is for *you* — it cements learning and prepares you for the next project.

4. **"Why do we do peer review?"** — You learn as much from evaluating others' work as from your own. Reviewing forces you to think critically about methodology, communication, and rigor. It also exposes you to different approaches and ideas.

5. **"What have I accomplished in this course?"** — A complete end-to-end predictive analytics project: problem framing, data preparation, model development, evaluation, fairness auditing, deployment preparation, executive communication, and peer review. These are the core skills of a professional data scientist.

---

## 7. Common Student Questions (Anticipate These)

**Q: "What if my notebook has errors when I run all?"**
A: Fix them before submitting. The most common causes are: (1) cells executed out of order during development — restart and run all to catch this; (2) missing imports — make sure all imports are in the setup cell; (3) variables defined in a cell that was later deleted — search for undefined variable errors. If you genuinely cannot fix an error, document it in a markdown cell explaining what you tried.

**Q: "How harsh should my peer feedback be?"**
A: Be honest and constructive, not harsh. The goal is to help your peer improve, not to demonstrate your superiority. Use "I notice" language instead of "You did wrong." Balance critical feedback with genuine positive observations. A good ratio is 2-3 strengths for every improvement suggestion.

**Q: "What if my peer's notebook does not run?"**
A: Score Reproducibility as 1 (Insufficient) and document the specific errors you encountered. This is the most important dimension to evaluate honestly — if the notebook cannot run, the analysis cannot be verified. Suggest specific fixes in your feedback.

**Q: "What if I run out of time?"**
A: Prioritize CRITICAL items in the technical audit first. A notebook that runs cleanly with visible outputs is worth more than a notebook with perfect formatting that crashes. Submit what you have — a partial submission with honest documentation of gaps is better than no submission.

**Q: "Do I need to fill out the postmortem?"**
A: Yes. The postmortem is part of the notebook submission and will be reviewed. It does not need to be long — 3-5 items per section is sufficient — but it needs to be *specific*. "I learned a lot" is not useful. "I learned that cross-validation gives more reliable estimates than a single hold-out split, which I discovered when my hold-out accuracy was 0.92 but my CV average was 0.85" is useful.

**Q: "What happens after this course?"**
A: The wrap-up section suggests three paths: continue learning (deep learning, NLP, causal inference, MLOps), build your portfolio (end-to-end projects, open source, writing, presenting), and stay connected (ML communities, industry blogs, Kaggle, networking). The most important next step is to apply these skills to a real problem — course projects are valuable, but nothing replaces hands-on experience with messy, stakeholder-facing data.

---

## 8. Connection to the Broader Course Arc

| Week | Notebooks | How NB20 Wraps It Up |
|---|---|---|
| **Week 1** | 01-05 | The technical audit checks for proper splitting (NB01), pipeline construction (NB02), baseline comparison (NB03), feature engineering (NB04), and regularization (NB05). Every Week 1 skill appears as a checklist item. |
| **Week 2** | 06-10 | The methodology dimension of the peer review rubric evaluates model selection, validation strategy, and metric appropriateness — all Week 2 topics. |
| **Week 3** | 11-15 | The results dimension evaluates whether findings are clear, well-supported, and include appropriate comparisons — skills from interpretation (NB13), calibration (NB15), and feature selection (NB12). |
| **Week 4** | 16-20 | NB16-17 provided model selection and fairness auditing. NB18 provided artifacts and monitoring. NB19 provided narrative and visuals. NB20 audits all of it, submits it, reviews a peer's version, and reflects on the journey. |

**NB20 is not just the last notebook — it is the notebook that proves the course worked.** A student who can self-audit rigorously, submit a complete package, evaluate a peer fairly, and reflect honestly has demonstrated mastery of the full predictive analytics lifecycle: from data to deployment to communication to continuous improvement.

---

## 9. Suggested Video Structure

Below are two recording options. Each segment includes the **speaking prompt** (what to say), **timestamps**, and the **notebook cells** to show on screen.

All cell references use the format `Cell N` where N is the zero-indexed cell number in `20_final_submission_peer_review.ipynb`.

---

### Option A: Single Video (~12 minutes)

#### Segment 1 — Opening & Motivation `[0:00–1:30]`

> **Show:** Cell 0 (header) and Cell 1 (learning objectives) on screen.

**Say:**

"Welcome to the final notebook of the course. Twenty notebooks. Four weeks. You started with a train-test split in notebook 01, and today you deliver a complete predictive analytics project: a reproducible notebook, an executive slide deck, a conference video, saved model artifacts, and a monitoring plan. But this notebook is not just a submission portal. It serves three purposes. First, you will self-audit your work — systematically checking every technical and communicative element before you submit. Second, you will review a peer's work using a structured rubric — learning as much from evaluating their project as you learned from building your own. Third, you will write a postmortem — reflecting on what worked, what did not, and what you would do differently. Let us get started."

> **Action:** Scroll through the 5 learning objectives in Cell 1, briefly reading each one aloud.

---

#### Segment 2 — Setup & Technical Self-Audit `[1:30–4:30]`

> **Show:** Cell 1 (setup explanation), then **run** Cell 2 (imports).

**Say:**

"Notice the imports: pandas, numpy, json, datetime. No scikit-learn. No matplotlib. This is not a modeling notebook — the modeling is done. This is an audit, review, and reflection notebook. The absence of modeling libraries is intentional: it signals that the technical work is complete and today is about quality assurance."

> **Run** Cell 2. Point to the `Setup complete!` output and the submission date timestamp.

> **Show:** Cell 3 (self-audit introduction), then **run** Cell 4 (technical reproducibility audit — 20 items).

**Say:**

"Here is the technical audit: twenty items across six categories. Let me walk you through the priority levels. Five items are CRITICAL — these are non-negotiable. Run-all must complete without errors. All outputs must be visible. There must be no test set leakage. Final test results must be reported. If any CRITICAL item fails, your submission is fundamentally broken, regardless of how sophisticated your model is. Ten items are HIGH priority — data source documented, baseline included, multiple models compared, appropriate metrics, model saved. These affect the credibility of your analysis. Five items are MEDIUM — no debug code, imports organized, functions documented. These affect professionalism.

Think of it like a building inspection. CRITICAL items are structural — the foundation. HIGH items are functional — the plumbing and electricity. MEDIUM items are cosmetic — the paint. You would never skip the structural inspection to focus on the paint."

> Point to the priority summary at the bottom of the output.

---

#### Segment 3 — Communication Audit `[4:30–6:00]`

> **Show:** Cell 5 (communication audit explanation), then **run** Cell 6 (communication quality audit — 15 items).

**Say:**

"The communication audit has fifteen items across five categories: Structure, Documentation, Visualizations, Tables, and Professionalism. Most are HIGH or MEDIUM priority. Pay particular attention to the Visualizations section: unlabeled axes and missing titles are among the most common deductions on presentation-heavy assignments. And check the Professionalism section: no typos in your title or conclusion, consistent formatting throughout, bibliography included at the end.

Here is a useful exercise: after running your technical audit, hand your notebook to a friend who is not in this course. Ask them to read it for five minutes and tell you what the project is about. If they can summarize it correctly, your communication is effective. If they look confused, your documentation needs work."

> **Mention:** "Pause the video here and complete Exercise 1 — run your project notebook through the restart-and-run-all test and document any issues you find." Point to Cell 7 (PAUSE-AND-DO 1).

---

#### Segment 4 — Artifact Manifest `[6:00–7:30]`

> **Run** Cell 10 (artifact manifest).

**Say:**

"Here is your submission manifest — a JSON file that lists every deliverable, its required format, and its submission status. Six items: Jupyter Notebook — the complete analysis with run-all outputs, shared as a Colab link. Slide Deck — 10 to 12 slides, PDF or PowerPoint. Conference Video — 2 to 3 minutes, MP4 format. Model Artifacts — your joblib pipeline and config JSON from notebook 18. Monitoring Plan — your signal table, included in the notebook or as a separate CSV. Model Card — a recommended but not required markdown section documenting your model's use case, data, performance, and limitations.

Before you click submit, walk through this manifest row by row. For every link, open it in an incognito browser window — not your logged-in browser — to verify that the sharing permissions are correct. The most common submission failure is a Colab link that only works for the person who created it. Test it as if you were a stranger clicking the link for the first time."

---

#### Segment 5 — Peer Review `[7:30–10:00]`

> **Show:** Cell 12 (peer review introduction and principles).

**Say:**

"Now let us talk about peer review. You learn as much from reviewing others' work as from your own. Good peer review requires three qualities. Be useful, not just nice — point out both strengths and weaknesses with specific, actionable feedback. Be professional — critique the work, not the person. Use 'I notice' instead of 'you did wrong.' Be thorough — check technical correctness, evaluate communication clarity, and consider business relevance."

> **Run** Cell 13 (peer review rubric — 5 dimensions).

**Say:**

"The rubric has five dimensions. Methodology carries the highest weight at 30 percent because sound analytical practice is the foundation. Reproducibility and Results each carry 20 percent. Communication carries 20 percent. Responsible AI carries 10 percent — it is the newest skill, introduced in notebook 17, and students are still developing their instincts here.

Each dimension has four scoring levels with concrete descriptors. Excellent means no issues. Good means minor issues. Needs Work means significant gaps. Insufficient means fundamental problems. Use the descriptors to anchor your scores — not gut feeling, not friendship, not how much effort you think they put in. Anchor on the evidence."

> **Run** Cell 15 (peer review form template). Highlight the structure.

**Say:**

"The review form asks for three things. First, scores for each dimension. Second, detailed feedback: three specific strengths and three specific areas for improvement. Third, a reviewer reflection: what did I learn from reviewing this project? Let me emphasize the feedback format: for each area of improvement, state the specific issue, suggest a concrete fix, and explain the expected improvement. 'Visuals could be better' is useless. 'The ROC curve on slide 6 is missing axis labels and an AUC annotation — adding those would strengthen credibility with the audience' is useful. The difference is specificity and actionability."

> **Mention:** "Pause the video here and complete Exercise 2 — complete one peer review using the rubric and form." Point to Cell 16 (PAUSE-AND-DO 2).

---

#### Segment 6 — Postmortem, Course Wrap-Up & Closing `[10:00–12:00]`

> **Run** Cell 19 (postmortem template). Highlight the nine sections.

**Say:**

"The postmortem is the final learning activity of the course. Nine sections, but three matter most. First: 'What I would do differently next time.' This is where you connect experience to future practice. A lesson that comes from your own struggle — 'I should have started with simpler models' — sticks far longer than a lesson from a textbook. Second: 'Most surprising finding.' This surfaces the unexpected insights that make data science interesting and reveals whether you engaged deeply with your data. Third: 'Advice for future students.' This creates a legacy of peer wisdom that is more persuasive than any instructor guidance."

> **Show:** Cell 22 (course wrap-up — four weeks summarized).

**Say:**

"Let me take a moment to acknowledge what you have accomplished. In four weeks, you built a complete end-to-end predictive analytics project. Week 1: you learned to split data, preprocess safely, build pipelines, and establish baselines. Week 2: you trained linear models, tree-based models, and ensembles, and learned to tune and evaluate them properly. Week 3: you tackled feature engineering, model interpretation, imbalanced data, and calibration. Week 4: you prepared your model for deployment, built an executive narrative, and now you are submitting, reviewing peers, and reflecting on the journey.

That is a significant accomplishment. You have developed technical skills — pipeline development, model selection, production-ready code. Analytical skills — problem framing, metric selection, risk assessment. And communication skills — executive storytelling, visual design, peer feedback. These three skill sets together — technical, analytical, communicative — are what define a professional data scientist."

> **Show:** Cell 23 (final submission instructions). Walk through the checklist.

**Say:**

"Final submission checklist. Five required deliverables: notebook, slide deck, video, model artifacts, monitoring plan. Peer review: completed with scores and three actionable suggestions. Postmortem: what worked, what did not, key learnings, what you would do differently. Before you click submit: all files uploaded, all links tested, all permissions verified, deadline confirmed, backup copies saved. Congratulations on completing the course. I hope the skills you built here serve you well in your careers. Keep learning, keep building, keep communicating. Thank you."

> **Show:** the final Thank You cell.

---

### Option B: Three Shorter Videos (~4 minutes each)

---

#### Video 1: "Self-Audit: Ensuring Your Submission is Bulletproof" (~4 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–0:45` | Show header + objectives | Cell 0, Cell 1 | "Welcome to the final notebook. Today: self-audit, submit, peer review, and postmortem. Four activities that wrap up four weeks of learning. Here are our five objectives." (Read briefly.) |
| `0:45–1:15` | Run setup | Cell 1 (explanation), Cell 2 | "Notice the imports: pandas, json, datetime. No scikit-learn, no matplotlib. The modeling is done. Today is about quality assurance, submission, and reflection." |
| `1:15–2:30` | Technical audit | Cell 3, Cell 4 | "Twenty items, three priority levels. Five CRITICAL: run-all works, outputs visible, no leakage, test results reported. Ten HIGH: data documented, baseline included, models compared, metrics appropriate, artifacts saved. Five MEDIUM: no debug code, imports organized, functions documented. Clear all CRITICAL items first — a notebook that crashes is an automatic failure regardless of model quality." |
| `2:30–3:15` | Communication audit | Cell 5, Cell 6 | "Fifteen items: structure, documentation, visuals, tables, professionalism. Key checks: all plots have titles and labeled axes, key findings are highlighted, bibliography included, no placeholder text. Hand your notebook to someone outside the course — if they can summarize it in two sentences, your communication works." |
| `3:15–4:00` | Exercise 1 prompt | Cell 7 | "Pause and complete Exercise 1 — restart your project notebook, run all cells, and document any issues. Fix them before continuing. This is the single most important quality check you will do today." |

---

#### Video 2: "Artifact Manifest and Peer Review" (~5 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–0:30` | Recap | — | "Welcome back. Your self-audit is complete. Now: submit your deliverables and review a peer's work." |
| `0:30–1:30` | Artifact manifest | Cell 9, Cell 10 | "Six deliverables: notebook (Colab link), slide deck (PDF/PPTX), video (MP4), model artifacts (joblib + JSON), monitoring plan (CSV), model card (markdown in notebook). Test every link in an incognito browser. The most common failure is a Colab link with incorrect sharing permissions." |
| `1:30–2:30` | Peer review principles | Cell 12 | "Three rules for good feedback. Be useful, not just nice — point out both strengths and weaknesses. Be professional — critique the work, not the person. Be thorough — check technical correctness, communication clarity, and business relevance." |
| `2:30–3:45` | Review rubric | Cell 13 | "Five dimensions. Methodology is highest at 30 percent — sound analysis matters most. Reproducibility and Results each 20 percent. Communication 20 percent. Responsible AI 10 percent. Four scoring levels with concrete descriptors. Use the descriptors to anchor your scores, not gut feeling." |
| `3:45–4:30` | Review form | Cell 14, Cell 15 | "The form asks for scores, three strengths, three improvements, and a self-reflection. For each improvement: state the specific issue, suggest a concrete fix, explain the expected benefit. 'Visuals could be better' is useless. 'The ROC curve needs axis labels and an AUC annotation' is useful." |
| `4:30–5:00` | Exercise 2 prompt | Cell 16 | "Pause and complete Exercise 2 — review the assigned project using the rubric and form. Score each dimension, identify three actionable improvements, and reflect on what you learned from the review." |

---

#### Video 3: "Postmortem, Course Wrap-Up & Farewell" (~4 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–1:00` | Postmortem template | Cell 18, Cell 19 | "The postmortem cements your learning. Nine sections, but three matter most. 'What I would do differently' — connects experience to future practice. 'Most surprising finding' — surfaces the insights that make data science engaging. 'Advice for future students' — creates peer wisdom that is more persuasive than any lecture. Be specific: 'I learned a lot' is not useful. 'I learned that cross-validation gives more reliable estimates than a single hold-out split' is useful." |
| `1:00–2:30` | Course wrap-up | Cell 22 | "Let me acknowledge what you accomplished. Week 1: splitting, preprocessing, pipelines, baselines. Week 2: linear models, trees, ensembles, tuning, metrics. Week 3: feature engineering, interpretation, imbalanced data, calibration. Week 4: reproducibility, deployment thinking, executive narrative, peer review. Three skill sets developed: technical — pipeline development, model selection, production code. Analytical — problem framing, metric selection, risk assessment. Communicative — storytelling, visual design, peer feedback. Together, these define a professional data scientist." |
| `2:30–3:15` | Final submission checklist | Cell 23 | "Final checklist: five deliverables submitted, peer review completed with scores and three suggestions, postmortem written. Before clicking submit: all files uploaded, all links tested in incognito, permissions verified, deadline confirmed, backup copies saved." |
| `3:15–4:00` | Farewell | Cell 22 (What's Next), final Thank You cell | "What is next: keep learning — deep learning, NLP, causal inference, MLOps. Build your portfolio — end-to-end projects, open source, writing, presenting. Stay connected — ML communities, Kaggle, industry blogs. The most important next step is to apply these skills to a real problem. Congratulations on completing the course. I hope the skills you built here serve you well. Keep learning, keep building, keep communicating. Thank you." |

---

*This guide was created to support video lecture recording for MGMT 47400 — Predictive Analytics, Purdue University.*
