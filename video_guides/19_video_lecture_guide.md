# Video Lecture Guide: Notebook 19 — Executive Narrative: Slide-Style Story + Conference Video Plan

## At a Glance

This guide covers:

1. **Why NB19 exists** — It teaches students to translate their technical work into a compelling, non-technical executive narrative — the skill that determines whether a model gets deployed or ignored

2. **Why it must follow NB18** — NB18 delivers reproducible artifacts, a monitoring plan, and a production checklist. NB19 transforms those artifacts into *evidence* within a persuasive story. Without NB18, the narrative would lack operational credibility

3. **Why it must precede NB20** — NB20 is final submission and peer review. Students need the slide outline, script, and visuals developed in NB19 to produce the deliverable package that NB20 requires. NB19 is preparation; NB20 is execution

4. **Why each library/tool is used:**
   - `pandas` — builds storyboard and checklist DataFrames for structured planning
   - `matplotlib` / `seaborn` — not for modeling but for creating executive-ready example visuals (model comparison charts)
   - `json` — stores Gemini prompts and script templates as reusable text files
   - No scikit-learn — this is deliberately a communication notebook, not a modeling notebook

5. **Common student questions** with answers (e.g., how long should my video be?, can I use technical terms?, what if my results are not impressive?)

6. **Connection to the full course arc** — how NB19 is the culmination of 18 notebooks of technical work, translating pipeline outputs into business recommendations

7. **Suggested video structure** — both single-video (~12 min) and three-video (~4 min each) formats with speaking prompts, timestamps, and notebook cell references

---

## Purpose of This Document

This guide is your **lecture companion** for recording the video(s) that accompany `19_project_narrative_video_studio.ipynb`. It explains the *why* behind every design choice in the notebook — why it exists, why it sits between notebooks 18 and 20, why it focuses on communication rather than code, and why each framework (Five-Act structure, storyboard, script template) is introduced at this point. Use it as speaking notes, not as a script: the goal is to internalize the reasoning so you can explain it naturally on camera.

---

## 1. Why This Notebook Exists

Eighteen notebooks of technical work — splitting, preprocessing, modeling, tuning, interpreting, auditing for fairness, serializing for deployment — have brought students to a critical juncture. They have a model that works, artifacts that prove it works, and a monitoring plan for when it stops working. But none of that matters if they cannot explain it to someone who controls the budget.

Notebook 19 addresses the single biggest gap in data science education: **communication**. Specifically, it teaches students to:

1. **Structure a narrative.** The Five-Act Framework (Problem, Approach, Results, Recommendation, Risks) gives students a repeatable formula for turning any analytical project into a persuasive story. One idea per slide. Full-sentence headlines. Evidence before assertion.

2. **Build credible visuals.** A model comparison table with a "Business Impact" column. A feature importance chart with plain-English labels. A decision policy flowchart. These are the visuals that earn trust from executives who do not know what ROC-AUC means.

3. **Script a tight video.** A 3-minute conference-style presentation forces students to distill 18 notebooks of work into 450 words. This constraint is the point: if you cannot explain your model in 3 minutes, you do not understand it well enough to deploy it.

4. **Use AI assistance effectively.** Five pre-written Gemini prompts help students tighten scripts, convert technical findings to executive bullets, strengthen headlines, simplify methodology descriptions, and quantify business impact. This is responsible AI-assisted communication — not asking Gemini to write the narrative, but asking it to sharpen what the student has already drafted.

This notebook is unusual in the course because it is **more about communication than code**. The code cells produce planning artifacts (storyboard tables, checklist CSVs, script templates) rather than model outputs. That is intentional. The skills taught here — structuring an argument, designing clear visuals, speaking concisely — are often the difference between a model that gets deployed and one that dies in a notebook.

---

## 2. Why It Must Come After Notebook 18

Notebook 18 establishes the **operational credibility** that NB19's narrative requires:

| NB18 Concept | How NB19 Uses It |
|---|---|
| **Reproducible pipeline (joblib artifact)** | NB19's "Approach" slides can claim: "This model is reproducible — we can load the saved pipeline and get identical predictions." Without NB18, this would be an unsupported claim. |
| **Configuration externalization (CONFIG + JSON)** | NB19's "Approach" slides can reference specific hyperparameters and settings from the config file, demonstrating transparency and rigor. |
| **Monitoring plan (8 signals, thresholds, owners)** | NB19's "Risks" slides can present a concrete monitoring strategy: "We will track PSI weekly and retrain if it exceeds 0.25." Without NB18's monitoring plan, the risk section would be vague. |
| **Production readiness checklist** | NB19's narrative can honestly disclose what is complete and what remains — a sign of professional maturity that builds audience trust. |
| **Three-file artifact package** | NB19's deliverable package section maps directly to NB18's saved artifacts. The narrative is grounded in concrete, verifiable deliverables. |

**In short:** NB18 provides the *evidence*. NB19 provides the *story*. Evidence without a story is ignored. A story without evidence is mistrusted. The two notebooks work as a pair.

If you tried to teach narrative *before* NB18, students would have no artifacts to reference — their slides would say "we built a model" with no proof of reproducibility or monitoring. If you skipped NB19, students would submit a technically correct but uncommunicative project in NB20 — all code, no story.

---

## 3. Why It Must Come Before Notebook 20

Notebook 20 is **final submission and peer review**. Its deliverable requirements include:

- A 10-12 slide executive deck
- A 2-3 minute conference video
- A complete artifact manifest with verified links

NB19 is where students *build* these deliverables:

```
NB18: Build reproducible artifacts + monitoring plan
            |
NB19: Design slides, write script, plan video (preparation)
            |
NB20: Submit everything, review peers, write postmortem (execution)
```

Without NB19, students would arrive at NB20's submission deadline without a slide outline, without a script, and without a visual strategy. NB19 provides the scaffolding — storyboard template, script template, visuals checklist, video production checklist — that makes the final submission manageable rather than overwhelming.

The deliberate separation of NB19 (preparation) from NB20 (execution) mirrors professional practice: you never walk into a board presentation without a rehearsal. NB19 is the rehearsal.

---

## 4. Why These Specific Libraries and Tools

### 4.1 `pandas` (for storyboards and checklists)

**What it does:** Creates DataFrames from dictionaries, displays them as formatted tables, and exports them as CSV files.

**Why we use it here:**

In previous notebooks, pandas manipulated data for modeling. Here, it serves a different purpose: **structured planning**. The 11-row storyboard DataFrame is a presentation plan. The 8-row visuals checklist is a design plan. The 13-row video production checklist is a logistics plan. Using DataFrames for planning (rather than free-form markdown) enforces completeness — every row is a task, every column is an attribute, and nothing is forgotten.

**What to emphasize on camera:** Students already know pandas as a data tool. Showing it as a *planning* tool reinforces the idea that structured thinking applies to communication, not just analysis.

### 4.2 `matplotlib` / `seaborn` (for example visuals)

**What it does:** Creates the model comparison table visualization with highlighted best performers.

**Why we use it here:**

The notebook does not generate model results — those come from the student's own project. Instead, it creates an *example* model comparison table with illustrative numbers and styling (green highlights for best values, a "Business Impact" column). This example teaches two lessons: (1) always include a baseline in your comparison, and (2) always translate metrics into business language.

**What to emphasize on camera:** The "Business Impact" column is the most important column in the table. Executives do not care about ROC-AUC; they care about "plus 26 percent accuracy, recommended." Every metrics table should have a column that answers the question "so what?"

### 4.3 `json` (for Gemini prompts)

**What it does:** Saves a dictionary of AI prompts as a JSON file for reuse.

**Why we use it here:**

The five Gemini prompts are not throwaway suggestions — they are structured, reusable tools for script refinement. Saving them as JSON means students can copy-paste them into the Gemini sidebar without retyping. Each prompt includes specific constraints (e.g., "reduce to 450-540 words," "keep each bullet to one sentence") that produce better AI output than vague requests.

**What to emphasize on camera:** The prompts model *good AI usage*: they are specific, constrained, and iterative. "Make it better" produces noise. "Reduce to 3 minutes, remove jargon, add timing markers" produces signal.

### 4.4 Why No scikit-learn

**This is deliberate.** NB19 is a communication notebook, not a modeling notebook. The absence of scikit-learn signals to students that technical skill alone is insufficient — you must also be able to *explain* what you built. If every notebook in the course imported scikit-learn, students would implicitly learn that modeling is all that matters. NB19 breaks that pattern on purpose.

---

## 5. Key Concepts to Explain on Camera

### 5.1 The Five-Act Framework

This is the structural backbone of the notebook. Every executive presentation follows the same arc:

1. **Problem (1-2 slides):** What business problem are you solving? Quantify the impact. Show the status quo.
2. **Approach (2-3 slides):** What data did you use? What methods did you try? How did you validate?
3. **Results (2-3 slides):** What did you find? How does it compare to baseline? What are the key drivers?
4. **Recommendation (1-2 slides):** What specific action should the audience take? What is the expected ROI? What is the timeline?
5. **Risks (1-2 slides):** What could go wrong? What limitations exist? What monitoring is in place?

**Analogy for students:** Think of it as a courtroom argument. Act 1 establishes the problem (the crime). Act 2 presents the evidence (the investigation). Act 3 delivers the verdict (the findings). Act 4 proposes the sentence (the recommendation). Act 5 acknowledges the appeal (the risks). Every element must be present for the argument to be credible.

### 5.2 The One-Idea-Per-Slide Rule

Every slide should have:
- **One clear headline** — a complete sentence, not a topic label
- **One key visual** — a chart, table, or diagram that supports the headline
- **2-3 supporting bullets** — optional, not required

The headline is the slide's argument. If someone reads only the headlines of all 11 slides, they should understand the entire narrative. This is the "glance test" — a busy executive should be able to flip through the deck in 30 seconds and get the key message.

**Bad headline:** "Model Performance" (a topic, not a finding)
**Good headline:** "Our model predicts customer churn with 85% accuracy" (a finding, quantified)

### 5.3 The Business Impact Column

The model comparison table includes a column called "Business Impact" that translates metrics into plain language:

| Model | ROC AUC | Business Impact |
|---|---|---|
| Baseline | 0.500 | Current state |
| Logistic Regression | 0.910 | +22% accuracy |
| Random Forest | 0.945 | +26% accuracy (recommended) |

This column is the bridge between the data scientist and the decision-maker. Without it, the table is a wall of numbers. With it, the recommendation is obvious.

### 5.4 The 3-Minute Script Constraint

The script template allocates time precisely:
- Opening: 15 seconds
- Problem: 30 seconds
- Approach: 30 seconds
- Results: 60 seconds
- Recommendation: 30 seconds
- Risks: 15 seconds

Total: 3 minutes = approximately 450-540 words at 150-180 words per minute.

**Why the constraint matters:** Most students will draft a 7-minute script on their first attempt. The constraint forces ruthless prioritization: which findings matter most? Which details can be cut? This editing process is where the real learning happens. If you cannot explain your model in 3 minutes, you do not understand it well enough.

### 5.5 Gemini as an Editing Partner, Not a Writer

The five Gemini prompts are designed for *refinement*, not *creation*:
- "Here is my script — reduce it to 3 minutes."
- "Here are my technical findings — convert to executive bullets."
- "Here is my headline — make it a complete sentence with numbers."

The student drafts first. Gemini sharpens second. This workflow produces better results than asking Gemini to generate a presentation from scratch, because the student's domain knowledge and analytical judgment drive the content, while Gemini handles concision and clarity.

---

## 6. What Students Should Take Away

After watching your video and completing this notebook, students should be able to answer these questions confidently:

1. **"How do I structure a data science presentation?"** — Use the Five-Act Framework: Problem, Approach, Results, Recommendation, Risks. One idea per slide. Full-sentence headlines. Evidence before assertion.

2. **"What makes a good slide headline?"** — It is a complete sentence with a subject and verb. It is specific and quantified. It focuses on the business outcome, not the methodology. It is under 15 words.

3. **"How do I explain my model to a non-technical audience?"** — Focus on *what you found*, not *how you found it*. Translate metrics into business impact. Use analogies. Eliminate jargon or define it immediately.

4. **"How long should my video be?"** — 2-3 minutes. At 150-180 words per minute, that is 450-540 words. If your first draft is longer, cut details from the Approach section (your audience cares about results, not methodology).

5. **"What visuals do I need?"** — At minimum: a model comparison table (with baseline and business impact column), a feature importance chart (with plain-English labels), and a monitoring plan summary. Each visual should support exactly one slide headline.

---

## 7. Common Student Questions (Anticipate These)

**Q: "What if my results are not impressive?"**
A: That is fine — and it is common in real projects. Focus on what you *learned*, not just what you *achieved*. A model that beats baseline by 5% is still valuable if you can explain why, what limits further improvement, and what data or methods might close the gap. Honest analysis of mediocre results is more professional than overstating strong results.

**Q: "Can I use technical terms in my video?"**
A: Only if you define them immediately. "Our model achieves an F1 score of 0.89 — that means it correctly identifies 89 out of every 100 cases while keeping false alarms low." The definition should use business language, not statistical language.

**Q: "Do I need to show code in my slides?"**
A: Almost never. Executives do not read code. If you need to show methodology, use a simple workflow diagram: "Data in, clean, model, predict, evaluate." The exception is a one-line code snippet that demonstrates something memorable, like `pipeline.fit(X_train, y_train)`.

**Q: "What tool should I use for slides?"**
A: Google Slides (free, collaborative, exports to PDF), PowerPoint, or Keynote — any tool you are comfortable with. The storyboard CSV from this notebook can be opened in Google Sheets to plan your slides before you open the slide editor.

**Q: "How do I handle Q&A?"**
A: Anticipate the three most likely questions (usually about data quality, model limitations, and implementation cost) and prepare one-sentence answers. If you do not know the answer, say "That is a great question — I would need to investigate X to give you a confident answer." Never make up numbers.

**Q: "What if I am not a good speaker?"**
A: Practice reduces anxiety. The script template is designed to be read almost verbatim — you do not need to improvise. Record yourself 2-3 times; the third take is almost always the best. Minor stumbles are fine; monotone reading is worse than natural mistakes.

---

## 8. Connection to the Broader Course Arc

| Week | Notebooks | How NB19 Connects |
|---|---|---|
| **Week 1** | 01-05 | Students learned to build a pipeline and evaluate a baseline. NB19 teaches them to *present* that baseline comparison in a model comparison table that executives can read. |
| **Week 2** | 06-10 | Students trained classification models and computed metrics. NB19 teaches them to translate those metrics into business impact: "We can reduce false alarms by 40%." |
| **Week 3** | 11-15 | Students tuned models, interpreted features, and handled imbalanced data. NB19 teaches them to *communicate* feature importance and model tradeoffs to a non-technical audience. |
| **Week 4** | 16-20 | NB16-17 finalized model selection and fairness auditing. NB18 created reproducible artifacts. NB19 wraps everything into a narrative. NB20 submits and reviews. |

**NB19 is where the entire course converges.** Every technical skill from Weeks 1-3 and every deployment skill from Week 4 feeds into the executive narrative. The student who cannot present their work effectively has not completed the learning journey — they have only completed the technical half.

---

## 9. Suggested Video Structure

Below are two recording options. Each segment includes the **speaking prompt** (what to say), **timestamps**, and the **notebook cells** to show on screen.

All cell references use the format `Cell N` where N is the zero-indexed cell number in `19_project_narrative_video_studio.ipynb`.

---

### Option A: Single Video (~12 minutes)

#### Segment 1 — Opening & Motivation `[0:00–1:30]`

> **Show:** Cell 0 (header) and Cell 1 (learning objectives) on screen.

**Say:**

"Welcome back. Over the past eighteen notebooks, you have built a complete predictive analytics pipeline — from data splitting and preprocessing through model training, tuning, fairness auditing, and deployment preparation. You have saved your model, verified its reproducibility, and designed a monitoring plan. But here is the uncomfortable truth: none of that matters if you cannot explain it to the person who controls the budget. Today's notebook is different from everything you have done so far. There is almost no scikit-learn, almost no modeling code. Instead, we focus on the skill that determines whether your model gets deployed or ignored: communication. By the end of this notebook, you will have a slide storyboard, a video script, and a visual design plan — the complete narrative package for your final project."

> **Action:** Scroll through the 5 learning objectives in Cell 1, briefly reading each one aloud.

---

#### Segment 2 — The Five-Act Framework `[1:30–3:30]`

> **Show:** Cell 3 (Five-Act Framework explanation).

**Say:**

"Every effective data science presentation follows the same five-act structure. Act 1: the Problem. What business problem are you solving, and why does it matter? Quantify the impact — how much is this problem costing? Act 2: the Approach. What data did you use, what methods did you try, and how did you validate? Keep this section short — your audience cares about results, not methodology. Act 3: the Results. What did you find? How does it compare to baseline? What are the key drivers? This is the longest section because it contains the evidence. Act 4: the Recommendation. What specific action should the audience take? What is the expected ROI? What is the timeline? Act 5: the Risks. What could go wrong? What limitations exist? What monitoring is in place?

Notice the balance. Two acts set up the problem. One act proposes the solution. Two acts present findings and recommendations. One act demonstrates intellectual honesty. This is not a formula I invented — it is how McKinsey, BCG, and every major consulting firm structures their deliverables. It works because it mirrors how decision-makers think: what is the problem, what is the evidence, what should I do."

---

#### Segment 3 — Storyboard Builder `[3:30–6:00]`

> **Show:** Cell 4 (storyboard explanation), then **run** Cell 5 (storyboard template).

**Say:**

"Before you open PowerPoint, you plan. The storyboard is an 11-row table where each row is one slide. For each slide, you specify the section it belongs to, a full-sentence headline, the key visual, supporting bullets, and speaker notes. Planning all 11 slides in a table forces you to think about the narrative arc before you start designing individual slides. The most common presentation mistake is building slides one at a time and ending up with no coherent story. The storyboard prevents that.

Let me draw your attention to the headline column. Every headline is a complete sentence — 'Our model predicts customer churn with 85 percent accuracy' — not a topic label like 'Model Performance.' If someone reads only the headlines of all 11 slides, they should understand the entire narrative. That is the glance test. A busy executive should be able to flip through your deck in thirty seconds and get the key message."

> **Run** Cell 5. Walk through 3-4 rows of the storyboard, pointing to the headline and visual columns.

> **Show:** Cell 6 (headline writing guidelines). Read the good/bad examples.

**Say:**

"Here are the headline rules. Complete sentences with subject and verb. Specific and quantified — include numbers. Action-oriented and jargon-free. Under fifteen words. Look at the examples. 'Model Performance' is bad — it tells me nothing. 'Our model predicts customer churn with 85 percent accuracy' is good — it tells me the finding, the metric, and the number in one sentence."

> **Mention:** "Pause the video here and complete Exercise 1 — create a 10-slide outline with full-sentence headlines and 2 bullets each." Point to Cell 7 (PAUSE-AND-DO 1).

---

#### Segment 4 — Visuals and Model Comparison `[6:00–8:00]`

> **Show:** Cell 9 (visuals checklist explanation), then **run** Cell 10 (visuals checklist).

**Say:**

"Here is your visual design checklist — eight required visuals for your presentation. Model comparison table, performance metrics chart, feature importance, confusion matrix, ROC or PR curve, business impact chart, decision policy, and monitoring dashboard. You do not need all eight on separate slides — some may be combined — but your final package should include each one. The checklist ensures nothing is forgotten."

> **Show:** Cell 12 (model comparison explanation), then **run** Cell 13 (model comparison table with styling).

**Say:**

"Now look at the model comparison table. Four models, six metrics, and — this is the key — a Business Impact column. The baseline gets 'Current state.' Logistic regression gets 'plus 22 percent accuracy.' Random forest gets 'plus 26 percent accuracy, recommended.' This column is the bridge between you and the decision-maker. Without it, the table is a wall of numbers that only a data scientist can interpret. With it, the recommendation is obvious to anyone in the room."

---

#### Segment 5 — Script Template and Video Planning `[8:00–10:30]`

> **Show:** Cell 14 (script structure and time allocation).

**Say:**

"Now let us plan the video. Your conference video is 2 to 3 minutes — about 450 to 540 words at speaking pace. The script template allocates time precisely: 15 seconds for the opening hook, 30 seconds for the problem, 30 seconds for the approach, 60 seconds for the results, 30 seconds for the recommendation, and 15 seconds for risks. Notice that results get the most time — that is where your evidence lives. Approach gets the least — your audience does not care how the sausage is made."

> **Run** Cell 15 (script template). Walk through the fill-in-the-blank sections.

**Say:**

"Each section has placeholder text you fill in. The opening starts with a hook — your key finding or a compelling question. The problem section establishes urgency: this costs us X dollars, or this error rate affects Y customers. The results section has room for three key findings, each with a supporting detail. The recommendation ends with a specific action, a quantified impact, and a timeline. Two important rules: no jargon, and every number should be rounded and meaningful."

> **Show:** Cell 16 (Gemini prompts introduction), then **run** Cell 17 (five Gemini prompts).

**Say:**

"Here are five Gemini prompts for refining your script. Tighten Script reduces your draft to exactly 3 minutes. Convert Findings to Executive Bullets removes jargon and adds business framing. Strengthen Headline turns topic labels into full-sentence findings. Simplify Methodology explains your methods without statistical terms. Quantify Business Impact translates accuracy into dollars. Notice that each prompt has specific constraints — 'reduce to 450 to 540 words,' 'keep each bullet to one sentence.' Constrained prompts produce far better AI output than vague requests like 'make it better.'"

> **Mention:** "Pause the video here and complete Exercise 2 — write your 2-3 minute script following the template." Point to Cell 18 (PAUSE-AND-DO 2).

---

#### Segment 6 — Video Production, Deliverables & Closing `[10:30–12:00]`

> **Run** Cell 21 (video production checklist). Highlight 2-3 key items.

**Say:**

"Before you record, run through this 13-item checklist. Pre-recording: is your script practiced and timed? Are your slides finalized? Is your recording environment quiet with good lighting? Recording: test your audio first — bad audio is the number one killer of otherwise good videos. Use an external microphone if you have one. Post-recording: trim dead air at the start and end, check audio sync, export as MP4."

> **Run** Cell 24 (final deliverable package).

**Say:**

"Your final project is not one file — it is a coordinated package: notebook, slide deck, conference video, model artifacts, monitoring plan, and model card. This table lists each deliverable, its required format, and how to submit it. Walk through it row by row before the deadline and verify every link in an incognito browser window."

> **Show:** Cell 26 (Wrap-Up).

**Say:**

"Let me close with the most important lesson of this notebook: your technical work is only as valuable as your ability to communicate it. A model with an F1 of 0.95 that nobody understands will not get deployed. A model with an F1 of 0.80 that is clearly explained, honestly caveated, and accompanied by a concrete recommendation and monitoring plan — that model gets deployed. Invest time in your narrative. In the next and final notebook, you will submit everything, review a peer's work, and write a postmortem reflection on the entire course. See you there."

> **Show:** Cell 27 (Submission Instructions) briefly, then the final Thank You cell.

---

### Option B: Three Shorter Videos (~4 minutes each)

---

#### Video 1: "The Five-Act Framework and Storyboard" (~4 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–0:45` | Show header + objectives | Cell 0, Cell 1 | "Welcome back. Today's notebook is different — almost no modeling code. Instead, we focus on communication: how to turn 18 notebooks of technical work into a 3-minute story that gets your model deployed. Here are our five learning objectives." (Read them briefly.) |
| `0:45–1:00` | Run setup | Cell 1 (explanation), Cell 2 | "Standard setup. Notice no scikit-learn imports — this is a communication notebook, not a modeling notebook. That is intentional." |
| `1:00–2:00` | Five-Act Framework | Cell 3 | "Every data science presentation follows five acts: Problem, Approach, Results, Recommendation, Risks. The Problem establishes urgency. The Approach establishes credibility. The Results present evidence. The Recommendation proposes action. The Risks demonstrate honesty. This is how consulting firms structure deliverables — and it works because it mirrors how decision-makers think." |
| `2:00–3:15` | Storyboard | Cell 4, Cell 5 | "Before you open PowerPoint, you plan. The storyboard has 11 rows — one per slide. Each row specifies a section, a full-sentence headline, a key visual, bullets, and speaker notes. The headline column is the most important: if someone reads only the headlines, they should understand your entire narrative. This is the glance test." |
| `3:15–3:45` | Headline guidelines | Cell 6 | "Headline rules: complete sentences, specific numbers, no jargon, under 15 words. Bad: 'Model Performance.' Good: 'Our model predicts customer churn with 85 percent accuracy.' The headline *is* the slide's argument." |
| `3:45–4:00` | Exercise 1 prompt | Cell 7 | "Pause and complete Exercise 1 — create your 10-slide outline with full-sentence headlines and 2 bullets each. Come back when you are done." |

---

#### Video 2: "Credible Visuals and Script Writing" (~5 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–0:30` | Recap | — | "Welcome back. You have a storyboard. Now we design the visuals and write the script." |
| `0:30–1:15` | Visuals checklist | Cell 9, Cell 10 | "Eight required visuals: model comparison table, performance chart, feature importance, confusion matrix, ROC curve, business impact chart, decision policy, monitoring dashboard. Not all need separate slides, but your package should include each one." |
| `1:15–2:15` | Model comparison table | Cell 12, Cell 13 | "The model comparison table is often the most persuasive visual. Four models, six metrics, and a Business Impact column. The Business Impact column translates numbers into language: 'plus 26 percent accuracy, recommended.' This column is the bridge between you and the executive." |
| `2:15–3:00` | Visual design principles | Cell 11 | "Design rules: large fonts, 3-4 colors maximum, clear titles and labels, no chart junk. No pie charts — bar charts are almost always better. Use direct labeling instead of legends. And always use colorblind-friendly palettes." |
| `3:00–4:00` | Script template | Cell 14, Cell 15 | "Your video is 2-3 minutes — about 450-540 words. The script allocates time: 15 seconds opening, 30 problem, 30 approach, 60 results, 30 recommendation, 15 risks. Results get the most time because that is where your evidence lives. Fill in each section with your project-specific content." |
| `4:00–5:00` | Gemini prompts | Cell 16, Cell 17 | "Five prompts for refining your script: tighten, convert to bullets, strengthen headlines, simplify methodology, quantify impact. Each has specific constraints — that is what makes them effective. Draft first, then refine with Gemini. AI is your editing partner, not your ghostwriter. Pause and complete Exercise 2 — write your full script." |

---

#### Video 3: "Video Production, Deliverables & Wrap-Up" (~3 min)

| Timestamp | Action | Notebook Cell | Speaking Prompt |
|---|---|---|---|
| `0:00–0:45` | Video checklist | Cell 20, Cell 21 | "Before you hit record, run through this 13-item checklist. Pre-recording: script practiced 3-5 times, slides tested, quiet environment. Recording: test audio first — bad audio kills good content. Post-recording: trim dead air, check sync, export as MP4." |
| `0:45–1:15` | Common mistakes | Cell 22 | "Seven mistakes to avoid. Number one: reading slides verbatim — slides are headlines, your voice is the explanation. Number two: too much technical detail — explain what you found, not how you found it. Number three: poor audio. Number four: going over time. Number five: no clear recommendation. Number six: ignoring limitations. Number seven: starting with methodology instead of the business problem." |
| `1:15–2:00` | Deliverable package | Cell 23, Cell 24 | "Your final project is six items: notebook, slide deck, video, model artifacts, monitoring plan, and model card. This table specifies the format and submission method for each. Walk through it row by row before the deadline. Test every link in an incognito browser." |
| `2:00–3:00` | Wrap-Up + closing | Cell 26, Cell 27 | "Key takeaway: your technical work is only as valuable as your ability to communicate it. A clearly explained, honestly caveated model with a concrete recommendation gets deployed. An uncommunicated masterpiece stays in the notebook. Invest time in your narrative. Next notebook is the grand finale: final submission, peer review, and postmortem. Submit this notebook to Brightspace, complete the quiz, and see you there." |

---

*This guide was created to support video lecture recording for MGMT 47400 — Predictive Analytics, Purdue University.*
