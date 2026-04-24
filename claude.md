# 2026 Summer Predictive Analytics Course - AI Assistant Guide

## Project Mission

This repository contains **MGMT 47400 - Predictive Analytics**, a 4-week intensive summer course (20 business days) for Purdue University's Daniels School of Business. The course runs **May 18 - June 12, 2026**, with **112.5 minutes of daily engagement** through micro-videos (≤12 min) and Google Colab notebooks.

### Key Context
- **Instructor:** Professor Davi Moreira
- **Institution:** Purdue University, Daniels School of Business
- **Format:** Fully online, 4-week intensive (20 business days, Mon-Fri only)
- **Pedagogy:** Concept → Demo → Practice (PAUSE-AND-DO) → Solution → Repeat
- **Technology:** Google Colab + Google Gemini AI assistance
- **Deployment:** GitHub Pages (Quarto-based static site)
- **Repository:** https://github.com/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474
- **Website:** https://davi-moreira.github.io/2026Summer_predictive_analytics_purdue_MGMT474/

---

## 🚨 READ THIS FIRST - Critical Guidelines

### Before Making ANY Changes:
1. **Read CONVERSATION_LOG.md** - Understand what's been done and why
2. **Check current git status** - `git status` to see uncommitted changes
3. **Never break established patterns** - Check existing notebooks/files for structure
4. **Test before committing** - Render Quarto site, test notebooks in Colab
5. **Update documentation** - CONVERSATION_LOG.md after major changes

### Core Principles:
- **Consistency is king:** All 20 notebooks follow identical structure
- **Documentation always:** Update CONVERSATION_LOG.md after major changes
- **Atomic commits:** One logical change per commit, clear messages
- **Student-first:** Every change should improve student learning experience
- **Reproducibility:** All code must run in fresh Colab environment

### 🚨 CRITICAL RULE - Voice and Audience in Student-Facing Content

**The student notebook is read BY students, not BY instructors who then teach it.** This means every sentence in a student notebook — including Gemini prompts and "After running, verify" checklists — must be written **TO the student**, never ABOUT the student and never TO the instructor.

**Hard rules:**

1. **Never write "students" as a third-party noun inside a student-facing cell.** If the text says "so students see X", "help students understand Y", "when students run this", or "as students work through", it is wrong. Rewrite in second person (`you`), neutral imperative (`print X to see Y`), or first person (`I want to see X`).
2. **Gemini prompts are scripts the student copies into Gemini.** They must sound like something a student would actually type. *Wrong:* `"... print classification_report so students see the per-class breakdown."` *Right:* `"... print classification_report to show the per-class breakdown."`
3. **No instructor-voice, video-guide, or camera language in student cells.** Forbidden phrases in student notebooks include `"on camera"`, `"the instructor should"`, `"speaking prompt"`, `"you (the instructor)"`. Those belong only in `video_guides/NN_video_lecture_guide.md`, which is gitignored and instructor-facing.
4. **The video guide can reference students in the third person.** The video guide is written FOR the instructor recording the video, so phrases like "Students should now understand…" are fine there. The student notebook is the opposite — write it as if the student is reading it alone at 11 PM, because usually they are.

**Before shipping any edit to a student notebook, grep for the failure modes:**

```bash
# Should return zero hits in any notebooks/*_student.ipynb
grep -iE '\bstudents?\b|\bthe instructor\b|on camera|speaking prompt' notebooks/NN_*_student.ipynb
```

If any hit shows up in a student file, rewrite before committing. (Hits in `video_guides/` are fine — those are instructor-facing.)

**When writing Gemini prompts specifically:**
- Use imperative verbs directed at Gemini (`"Load X, compute Y, print Z"`), not meta-commentary about what students will learn.
- The trailing `"so students see"` / `"so students understand"` pattern is the most common regression. If you feel the need to explain *why* Gemini should print something, say `"... to show the per-class breakdown"` or `"... so the comparison is explicit"` — the justification is part of the prompt, not a side-note about the audience.

**The voice rule applies differently to video guides (`video_guides/`)**, which are instructor-facing. Two zones inside every guide:

1. **Wrapper prose** (instructor-facing, read silently): `"Students often ask…"` is FINE — the instructor is being informed ABOUT the audience.
2. **Blockquote read-aloud scripts** (`> *"..."*`) — the instructor SPEAKS these on camera, so the viewer IS a student. Inside blockquotes, the rule from student notebooks applies: no third-party "students", no "the instructor", no "on camera". Address the viewer as *you*; when introducing a Q&A, use `"A question that often comes up here"` instead of `"Students often ask"`.

**Before shipping any video-guide edit, run the blockquote-only voice check** at `/tmp/nb09build/voice_check_guides.py` (or recreate it — the pattern is: flag any `students?` / `the instructor` / `on camera` / `speaking prompt` hit whose surrounding line starts with `>` AND is not `Student's t`). Fixing these in place is a one-line `sed` — the wrapper-prose hits are legitimate and must be left alone.

---

### 🚨 CRITICAL RULE - Narrative Polish Pattern (nb08 Style)

**Every student notebook markdown cell should follow the nb08 narrative style.** When polishing nb01–nb20, or writing any new notebook, use these patterns — they are the course's voice, and they have been consistently applied across all 21 notebooks.

**Five structural elements every student notebook has:**

1. **Business-case "Why This Matters" cell** with a named stakeholder (HomeValue CFO, MedScreen chief medical officer, TechCorp People Analytics lead). The stakeholder's concern is phrased as a direct quote. This cell opens the analytical work and motivates every section below.
2. **Narrative prose over bullet lists** — "Reading the output" cells are paragraphs, not terse enumerations. A bullet list is a fallback when the structure is genuinely list-like (a rubric, a checklist); flowing prose is the default for explanation.
3. **Inline Q&A blocks** with the exact phrase **"A question that often comes up here"** (or "A question that often comes up"). Placement: after each dense explanation, anticipate one specific student confusion and answer it in one paragraph. This phrase is grep-findable — tooling uses it to count and audit Q&A coverage.
4. **Section bridges** that explicitly name the transition: *"Section 2 landed the regression estimate with a tight CI. Now apply the identical four steps to the classification problem."* Never jump between sections without a one-sentence bridge.
5. **Warm wrap-ups with next-notebook bridges** — the "Wrap-Up: Key Takeaways" cell ends with a paragraph naming the next notebook and what it builds on today's work. The wrap-up also typically carries one closing Q&A.

**When polishing is warranted:**
- Any new markdown cell longer than \~150 words in a student notebook.
- Any "Reading the output" cell that is currently a bullet list.
- Any section transition that feels abrupt.
- Any "Why This Matters" cell that lacks a named stakeholder.

**When the polish script pattern has worked well:**

```python
# Pattern used across every NB polish batch in this course
def append_qa_if_missing(nb, signature_prefix, qa_block):
    for c in nb['cells']:
        if c['cell_type'] != 'markdown':
            continue
        src = ''.join(c['source'])
        if not src.lstrip().startswith(signature_prefix):
            continue
        if 'A question that often comes up' in src:
            return False  # already has Q&A — idempotent
        stripped = src.rstrip()
        if stripped.endswith('---'):
            stripped = stripped[:-3].rstrip()
        c['source'] = [stripped + '\n\n' + qa_block + '\n\n---\n']
        return True
    return False
```

The idempotent check (`if 'A question that often comes up' in src`) is critical — it prevents duplicating Q&As on re-runs.

**Batching rule of thumb** — polish in groups of 2–3 notebooks per commit, not one notebook at a time. Polish + voice-check + render + commit per batch. This keeps commit messages meaningful and docs rendering in sync.

---

### 🚨 CRITICAL RULE - CV-First Evaluation + Test-Set Lock

**From nb09 onward, all model-performance claims come from cross-validation.** Before nb14, the test set (`X_test`, `y_test`) is *locked* — no model evaluation touches it. nb14's "Opening the Locked Test Set" ceremony is the one and only authorized test-set opening in the entire course.

**The rule, stated crisply:**

| Where | What to use |
|---|---|
| nb01–nb07 | Single train/val/test split is introduced; `X_val` for mid-course evaluation |
| nb08 | k-fold CV + Student's *t* 95% CI becomes the course's evaluation spine |
| nb09–nb13, nb15, nb16, nb17 | `cross_val_score`, `cross_val_predict`, `GridSearchCV`, `RandomizedSearchCV` on `X_train`; held-out evaluation uses `X_val`, never `X_test` |
| **nb14 cell 33 ONLY** | `X_test` / `y_test` opened for the one-shot ceremony (INSIDE/ABOVE/BELOW verdict) |
| nb18 | `X_test` may be used in the Kaggle-submission demo to simulate predicting on a held-out CSV — this is legitimate because it is a production-pipeline pattern, not model evaluation |
| nb20 | No model evaluation — peer review + postmortem |

**The CV-first principle is not a style preference; it is the course's pedagogical spine.** nb14's ceremony loses its meaning if the test set has been touched 30 times before students get there. The value of the lock is the consistency.

**Before shipping any evaluation code in nb09–nb20, run the audit:**

```python
# Tight audit — finds MODEL-EVAL uses of X_test/y_test (not just train_test_split)
import json, re
from pathlib import Path

MODEL_EVAL_TEST_PATTERNS = [
    r'\.score\(X_test', r'\.predict\(X_test', r'\.predict_proba\(X_test',
    r'roc_auc_score\(y_test', r'accuracy_score\(y_test', r'f1_score\(y_test',
    r'precision_score\(y_test', r'recall_score\(y_test',
    r'classification_report\(y_test', r'brier_score_loss\(y_test',
    r'permutation_importance\([^,)]+,\s*X_test',
]

for path in sorted(Path('notebooks').glob('*_student.ipynb')):
    nb = json.loads(path.read_text())
    hits = []
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] != 'code':
            continue
        src = ''.join(c['source'])
        for pat in MODEL_EVAL_TEST_PATTERNS:
            for m in re.finditer(pat, src):
                line_start = src.rfind('\n', 0, m.start()) + 1
                line = src[line_start:src.find('\n', m.start())]
                if line.lstrip().startswith('#'):
                    continue
                hits.append((i, m.group(0)))
    if hits:
        print(f'{path.name}: {len(hits)} hits')
        for cell, pat in hits:
            print(f'  cell {cell}: {pat}')
```

**The only acceptable audit output** is 6 hits in `notebooks/nb14_model_selection_protocol_student.ipynb` cell 33, plus any submission-demo hits in `notebooks/nb18_reproducibility_monitoring_student.ipynb`. Anything else is a regression and must be fixed before committing.

**Common CV-first patterns to reach for:**

- Classifier comparison: `cross_val_score(model, X_train, y_train, cv=StratifiedKFold(5, ...), scoring='roc_auc')`, then report `mean ± (t_crit * sd / sqrt(k))` as a 95% CI.
- `classification_report` on held-out predictions: `y_pred = cross_val_predict(model, X_train, y_train, cv=cv_strat)` — every row's prediction comes from a fold that never saw it during fitting.
- Permutation importance that would otherwise touch `X_test`: split `X_train` further (e.g., 75/25 inside the cell), fit on the 75% slice, measure permutation importance on the 25% slice. Test set stays locked.
- Calibration that needs a held-out sample: use `CalibratedClassifierCV(base, cv=5)` fit on `X_train` (internal CV handles the calibrator fit), evaluate Brier on `X_val`.

**Before every commit in nb09–nb20, run both audits** — the voice-check grep *and* the CV-first audit above. Both must be clean (with the nb14 cell 33 exception on the test-set side).

---

### 🚨 CRITICAL WORKFLOW - Instructor-First Notebook Editing

**ALWAYS edit `notebooks/NN_*_instructor.ipynb` FIRST, then generate the student file.**

- The **instructor notebook** is the source of truth — it is NEVER modified by this procedure
- The **student notebook** (`NN_*_student.ipynb`) is generated from the instructor notebook by deleting solution cells
- Both files coexist in the `notebooks/` folder; only the student file is tracked in git

**Naming convention:**
- Instructor: `NN_topic_instructor.ipynb` (gitignored, local only)
- Student: `NN_topic_student.ipynb` (committed, rendered, published)

**Generating the student notebook:**
1. Copy the instructor file: `cp notebooks/NN_*_instructor.ipynb notebooks/NN_*_student.ipynb`
2. Delete all cells containing `INSTRUCTOR SOLUTION` in the student copy (markdown or code)
3. Update the Colab badge URL to match the student filename
4. Update the video guide (`video_guides/NN_video_lecture_guide.md`)
5. Commit only the student notebook (instructor notebooks are gitignored)

**Instructor notebook conventions (for this workflow to work):**
- Every cell that should be excluded from the student version MUST contain the string `INSTRUCTOR SOLUTION` somewhere in its source:
  - Markdown cells: `### INSTRUCTOR SOLUTION — Exercise N` (as the heading)
  - Code cells: `# INSTRUCTOR SOLUTION` (as the first comment line)
  - Hidden markdown: `<!-- INSTRUCTOR SOLUTION -->` (as the first line)
- Student placeholder cells (e.g., `### YOUR FINDINGS HERE:`) live in the instructor notebook and survive the deletion
- Placeholder cells must NOT contain `INSTRUCTOR SOLUTION`
- **Code exercise block structure (instructor notebook):**
  1. `## 📝 PAUSE-AND-DO Exercise X` (exercise prompt markdown)
  2. `> 💡 Gemini Prompt:` (Gemini suggestion with "After running, verify:" checklist)
  3. Student code cell: `# YOUR SOLUTION CODE HERE` (must NOT contain INSTRUCTOR SOLUTION)
  4. `### INSTRUCTOR SOLUTION — Exercise X` (solution heading markdown)
  5. `# INSTRUCTOR SOLUTION` code cell (solution implementation — removed from student)
  6. `<!-- INSTRUCTOR SOLUTION -->` "Reading the output" markdown (removed from student)
- The student code cell survives into the student notebook (students write their code here)
- The instructor solution code and reading-the-output are removed during student generation

**Remember:** The instructor file stays untouched in the folder. The student file is the one that gets committed, rendered by Quarto, and published to the course website.

### 🚨 CRITICAL WORKFLOW - Keep Video Lecture Guides in Sync

**EVERY TIME a notebook (`notebooks/NN_*_student.ipynb`) is updated, you MUST also update the corresponding video lecture guide (`video_guides/NN_video_lecture_guide.md`).**

- Video guides are local-only (gitignored) — no commit/push needed for them
- Update affected sections: cell references, speaking prompts, section content, timestamps
- Template reference: `video_guides/02_video_lecture_guide.md`
- Guide structure: At a Glance, Purpose, 9 sections (Why exists, Why after N-1, Why before N+1, Libraries/Tools, Key Concepts, Student Takeaways, Common Questions, Course Arc, Suggested Video Structure with Options A & B)

### 🚨 CRITICAL WORKFLOW - Keep Planning Documents in Sync

**When notebooks change significantly (new sections, reordered content, new tools/libraries), you MUST also update the sequencing rationale in:**

- `MGMT47400_Online4Week_Plan_2026Summer.md` — the "Notebook Sequence Rationale" table and dependency diagram
- `claude_course_plan.md` — the "Notebook Sequence and Content Justification" table

**What triggers an update:** Adding/removing notebook sections, changing the tools/libraries used, reordering content, or changing dependencies between notebooks. Minor fixes (typos, wording) do not require updates.

### 🚨 CRITICAL WORKFLOW - Commit AND Update Webpage

**EVERY TIME you make changes to course content, you MUST:**

1. **Commit your changes** to git
2. **Render the Quarto site** (`quarto render`)
3. **Commit the rendered docs/** folder
4. **Push to GitHub**

**Why this matters:**
- GitHub Pages serves the `docs/` folder
- Changes to `.qmd` files or notebooks won't appear on the website until docs/ is rendered and committed
- The course website is the student-facing interface - it MUST stay up-to-date

**Standard workflow after ANY content change:**
```bash
# 1. Commit your content changes
git add notebooks/XX_topic_student.ipynb  # or schedule.qmd, syllabus.qmd, etc.
git commit -m "feat: Update notebook XX

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"

# 2. Render Quarto site
/Applications/RStudio.app/Contents/Resources/app/quarto/bin/quarto render

# 3. Commit rendered docs/
git add docs/
git commit -m "build: Render Quarto site for Day X updates

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"

# 4. Push everything
git push origin main
```

**Remember:** If you don't render and commit docs/, students won't see your changes on the website!

---

## 📁 Repository Structure

```
├── notebooks/                  # 21 Jupyter notebooks (Day 0 + Days 1-20)
│   ├── 00_launchpad_course_setup_student.ipynb
│   ├── 00_launchpad_course_setup_instructor.ipynb  # gitignored
│   ├── 01_eda_splits_student.ipynb
│   ├── 01_eda_splits_instructor.ipynb  # gitignored
│   ├── 02_preprocessing_pipelines_student.ipynb
│   ├── ...
│   └── 20_final_submission_peer_review_student.ipynb
├── docs/                       # GitHub Pages output (compiled by Quarto)
│   ├── index.html
│   ├── schedule.html
│   ├── syllabus.html
│   └── notebooks/              # HTML versions of notebooks
├── lecture_slides/             # Legacy slides (maintained for reference)
├── images/                     # Course logo and assets
├── video_guides/              # EXCLUDED from git (local-only instructor video lecture guides)
├── _adm_stuff/                # EXCLUDED from git (admin materials)
├── index.qmd                   # Homepage source
├── schedule.qmd                # Schedule source
├── syllabus.qmd                # Syllabus source
├── _quarto.yml                 # Quarto configuration
├── README.md                   # Public documentation
├── CONVERSATION_LOG.md         # Development history
├── MGMT47400_Online4Week_Plan_2026Summer.md  # Master course plan
└── claude.md                   # This file
```

### Key Files (Always Check These)
- **CONVERSATION_LOG.md** - Development history and decisions
- **MGMT47400_Online4Week_Plan_2026Summer.md** - Master course plan (THE SOURCE OF TRUTH)
- **README.md** - Public-facing documentation
- **claude_course_plan.md** - Implementation plan
- **_quarto.yml** - Quarto configuration

### Where to Find Information
- Course dates/schedule → `MGMT47400_Online4Week_Plan_2026Summer.md`
- Notebook structure → Any notebook in `notebooks/`
- Git workflow → `CONVERSATION_LOG.md`
- Deployment steps → `GITHUB_SETUP_INSTRUCTIONS.md`

---

## 📐 Established Conventions & Patterns

### Notebook Structure (MUST FOLLOW)

> **Canonical reference:** `notebooks/nb01_eda_splits_student.ipynb` is the reference template for all notebook structure and formatting. When creating or updating notebooks, match its header format, section organization, and conventions exactly.

Every notebook MUST include these sections in order:

#### 1. Header Cell (Markdown)
```markdown
# [Topic Title]

<hr>

<center>
<div>
<img src="https://raw.githubusercontent.com/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/main/notebooks/figures/mgmt_474_ai_logo_02-modified.png" width="200"/>
</div>
</center>

# <center><a class="tocSkip"></center>
# <center>MGMT47400 Predictive Analytics</center>
# <center>Professor: Davi Moreira </center>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/XX_topic_student.ipynb)

---
```

**Important:** No "Day X:" prefix in titles. No date lines. Notebooks are self-paced and should not reference specific days or dates.

#### 2. Learning Objectives (Markdown)
```markdown
## Learning Objectives

By the end of this notebook, you will be able to:

1. [Objective 1]
2. [Objective 2]
3. [Objective 3]
4. [Objective 4]
5. [Objective 5]

---
```

#### 3. Setup Section (Code)
```python
# Standard imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings

# Display settings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.precision', 3)
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# Set random seed for reproducibility
RANDOM_SEED = 474
np.random.seed(RANDOM_SEED)

print("✓ Setup complete!")
print(f"Random seed: {RANDOM_SEED}")
```

#### 4. Content Sections (Numbered 1, 2, 3...)
- Clear section headers (## 1. Title, ## 2. Title, etc.)
- Markdown explanation before each code cell
- Visualizations with clear labels
- Subsections as needed (### 1.1, ### 1.2, etc.)

#### 5. PAUSE-AND-DO Exercises (2 per notebook, 10 min each)

**Text-only exercise (interpretation/analysis):**
```markdown
## 📝 PAUSE-AND-DO Exercise X (10 minutes)

**Task:** [Clear, specific task]

---

### YOUR ANSWER HERE:

**[Question 1]:**
[Student response]

---
```

**Code exercise (students write code):**
```markdown
## 📝 PAUSE-AND-DO Exercise X (10 minutes)

**Task:** [Clear, specific task]

---
```
```markdown
> 💡 **Gemini Prompt:** "[Step-by-step instructions for Gemini]"
>
> **After running, verify:**
> - [Expected output 1]
> - [Expected output 2]
```
```python
# YOUR SOLUTION CODE HERE
# Hint: Use the Gemini prompt above for step-by-step guidance
```
```markdown
### YOUR FINDINGS HERE:

**[Question 1]:**
[Student response]

---
```

#### 6. Wrap-Up Section (Markdown)
```markdown
## 6. Wrap-Up: Key Takeaways

### What We Learned Today:

1. [Key point 1]
2. [Key point 2]
3. [Key point 3]
4. [Key point 4]

### Remember:

> **"[Critical rule in blockquote]"**

---
```

#### 7. Bibliography (Markdown)
```markdown
## 7. Bibliography

- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2023). *An Introduction to Statistical Learning with Python* (ISLP). Springer.
- [Other relevant citations]
- scikit-learn User Guide: [Relevant section](URL)

---
```

#### 8. Thank You Cell (Markdown, final cell)
```markdown
<center>

Thank you!

</center>
```

### Naming Conventions

- **Notebooks (student):** `NN_topic_student.ipynb` (e.g., `nb01_eda_splits_student.ipynb`) — committed to git
- **Notebooks (instructor):** `NN_topic_instructor.ipynb` (e.g., `nb01_eda_splits_instructor.ipynb`) — gitignored
- **Git commits:** `<type>: <subject>`
  - Types: `feat`, `docs`, `chore`, `build`, `fix`
  - Always include: `Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>`
- **Variables:** `lowercase_with_underscores` (Python convention)
- **Constants:** `UPPERCASE` (e.g., `RANDOM_SEED = 474`)

### Style Guidelines

- **Random seed:** Always `RANDOM_SEED = 474` (NOT 42)
- **Train/Val/Test split:** Always 60/20/20
- **Figure size:** `plt.rcParams['figure.figsize'] = (10, 6)`
- **Display precision:** `pd.set_option('display.precision', 3)`
- **Emoji usage:**
  - ✓ for success
  - ⚠️ for warnings
  - 📝 for exercises
  - 💡 for insights
- **Dollar signs in markdown cells:** Always use `\$` (escaped) when referring to money in markdown cells (e.g., `\$50,000`, `\$100k`). An unescaped `$` triggers LaTeX math mode in Google Colab and breaks text rendering. This applies to markdown cells only — `$` in Python code strings is fine.
- **No unnecessary code:** Don't add features not explicitly requested

---

## 🔧 Common Tasks & Workflows

### Task 1: Add a New Notebook

1. **Choose the right notebook number** (01-20)
2. **Create the instructor notebook first** (`NN_topic_instructor.ipynb`)
   - Copy structure from canonical reference (`nb01_eda_splits_student.ipynb`)
   - Write all content including `INSTRUCTOR SOLUTION` cells
   - Include student placeholder cells (e.g., `### YOUR FINDINGS HERE:`)
3. **Generate the student notebook** using the copy-delete workflow:
   - Copy instructor → `NN_topic_student.ipynb`
   - Delete all cells containing `INSTRUCTOR SOLUTION`
   - Update the Colab badge URL to point to `_student.ipynb`
4. **Verify:** Student file has no `INSTRUCTOR SOLUTION` cells, correct Colab badge
5. **Test in Colab:** Click "Open in Colab" → "Runtime → Run all"
6. **Commit:**
    ```bash
    git add notebooks/XX_topic_student.ipynb
    git commit -m "feat: Add Day XX notebook

    Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
    ```

### Task 2: Update the Schedule

1. **Read the course plan** (`MGMT47400_Online4Week_Plan_2026Summer.md`)
2. **Edit `schedule.qmd`**
3. **Add/update row in table** with: Day | Date | Topic | Videos | Notebook | Assessment | Materials
4. **Use correct date** (business days only, May 18 - June 12, 2026)
5. **Link to notebook:**
   ```
   [XX_topic_student.ipynb](https://github.com/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/XX_topic_student.ipynb)
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](...)
   ```
6. **Render site:** `/Applications/RStudio.app/Contents/Resources/app/quarto/bin/quarto render`
7. **Commit:**
   ```bash
   git add schedule.qmd docs/
   git commit -m "docs: Update schedule for Day XX

   Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
   ```

### Task 3: Render & Deploy Website

```bash
# Render Quarto site
/Applications/RStudio.app/Contents/Resources/app/quarto/bin/quarto render

# Check rendered output
ls -la docs/

# Commit changes
git add docs/
git commit -m "build: Render Quarto site

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"

# Push to GitHub
git push origin main

# GitHub Pages will auto-deploy (wait 1-2 minutes)
# Visit: https://davi-moreira.github.io/2026Summer_predictive_analytics_purdue_MGMT474/
```

### Task 4: Update CONVERSATION_LOG.md

**When:** After completing any major work or at end of session

**Template:**
```markdown
## Session X: [Date]

### Objective
[What was the goal of this session?]

### Work Completed
- [List of accomplishments]
  - Files created: [list]
  - Files updated: [list]

### Decisions Made
- [Key choices with rationale]

### Problems Encountered
- [Issues and solutions]

### Next Steps
- [ ] [Remaining tasks]

---
```

### Task 5: Standard Git Workflow

```bash
# Check status
git status

# Stage specific files (preferred over git add .)
git add file1.ext file2.ext

# Commit with semantic message
git commit -m "feat: Add feature X

Detailed description if needed.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"

# Push to remote
git push origin main
```

---

## 🛠️ Technology Stack

### Primary Technologies

**Quarto (v1.4+):** Static site generator
- Location: `/Applications/RStudio.app/Contents/Resources/app/quarto/bin/quarto`
- Command: `quarto render` (renders .qmd → HTML in docs/)
- Documentation: https://quarto.org/docs/guide/

**Git:** Version control
- Remote: `https://github.com/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474.git`
- Branch: `main`

**GitHub Pages:** Hosting
- Source: `docs/` directory on main branch
- URL: `https://davi-moreira.github.io/2026Summer_predictive_analytics_purdue_MGMT474/`

### Python Stack (for notebooks)

**Platform:** Google Colab (Jupyter notebooks in cloud)

**Core libraries:**
- `pandas`, `numpy` (data manipulation)
- `matplotlib`, `seaborn` (visualization)
- `scikit-learn` (machine learning)
- `joblib` (model persistence)

**AI Assistant:** Google Gemini (integrated in Colab)

### Deployment Workflow
```
.qmd files → quarto render → docs/ → git push → GitHub Pages
```

---

## 🧠 Key Decisions & Rationale

### Decision 1: Flat Notebook Structure
**Decision:** All 20 notebooks in `/notebooks/` (flat, not nested by week)

**Rationale:**
- Easier to link/reference (simple URLs)
- Clear sequential numbering (01-20)
- Students navigate linearly through days
- GitHub displays flat lists better

### Decision 2: 60/20/20 Split for All Examples
**Decision:** Always use 60% train, 20% validation, 20% test

**Rationale:**
- Consistency across all 20 notebooks
- Students learn ONE splitting pattern
- Sufficient validation data for tuning
- Realistic test set size

### Decision 3: RANDOM_SEED = 474 Everywhere
**Decision:** All random operations use seed 474 (MGMT 474 course number)

**Rationale:**
- Complete reproducibility
- Students get identical outputs
- Easier to debug (same results every time)
- Course-specific seed (MGMT 474)

### Decision 4: Google Colab + Gemini (Not Local Jupyter)
**Decision:** Primary platform is Google Colab, not local installations

**Rationale:**
- Zero setup for students (no installation issues)
- Consistent environment (same Python/library versions)
- Built-in GPU access (for deep learning day)
- Gemini AI assistance integrated
- Accessible from any device

### Decision 5: Exclude Admin Materials from Git
**Decision:** `_adm_stuff/` in .gitignore

**Rationale:**
- Student privacy (contact info, accommodations)
- Sensitive data (grades, evaluations)
- Large files (homework solutions, zip archives)
- Public repo - can't include private materials

### Decision 6: Micro-Videos (≤12 min each)
**Decision:** All videos capped at 12 minutes maximum

**Rationale:**
- Attention span research (10-15 min optimal)
- Mobile-friendly (can watch on phone)
- Easy to re-watch specific topics
- Forces concise, focused content
- 6 videos per day = ~1 hour total

### Decision 7: PAUSE-AND-DO (Not "Exercise" or "Assignment")
**Decision:** Use "PAUSE-AND-DO" terminology

**Rationale:**
- Clear action signal (pause video, do this now)
- Distinguishes from graded assignments
- Emphasizes active learning
- 10-minute scope (not homework)
- Builds engagement habit

---

## 🚫 What NOT to Do (Anti-Patterns)

### ❌ DON'T: Commit Large Files
- No .zip, .mp4, .mp3, .mov files
- No datasets >10MB (link to external storage instead)
- **Why:** GitHub has 100MB file limit, slows down clones

### ❌ DON'T: Break Notebook Structure
- Don't skip Colab badge, learning objectives, or exercises
- Don't add features not in course plan
- **Why:** Consistency is critical for student experience

### ❌ DON'T: Use `git add .` Indiscriminately
- Always stage specific files
- Review `git status` first
- **Why:** Avoid committing temp files, secrets, or broken code

### ❌ DON'T: Change Random Seeds
- Always use `RANDOM_SEED = 474`
- **Why:** Breaks reproducibility, students get different results

### ❌ DON'T: Skip Testing in Colab
- Always click "Open in Colab" and "Run All" before committing
- **Why:** Notebooks MUST work in fresh Colab environment

### ❌ DON'T: Update a Notebook Without Updating Its Video Guide
- Every notebook change MUST be accompanied by updating `video_guides/NN_video_lecture_guide.md`
- **Why:** Guides contain cell references, speaking prompts, and content descriptions that become stale if not synced

### ❌ DON'T: Make Significant Notebook Changes Without Updating Planning Documents
- When notebooks gain new sections, change tools/libraries, or shift dependencies, update the sequencing rationale in `MGMT47400_Online4Week_Plan_2026Summer.md` and `claude_course_plan.md`
- **Why:** These documents contain dependency tables and arc descriptions that become inaccurate if not synced with actual notebook content

### ❌ DON'T: Leave Solution Cells Unmarked in Instructor Notebooks
- Every cell that should be excluded from the student version MUST contain `INSTRUCTOR SOLUTION` in its source
- This includes code cells (use `# INSTRUCTOR SOLUTION` as the first comment) and follow-up markdown cells (use `<!-- INSTRUCTOR SOLUTION -->` as the first line)
- **Why:** The copy-delete workflow relies on this marker to strip solutions. Unmarked cells will leak into the student notebook

### ❌ DON'T: Mix Student Placeholder and Instructor Solution in the Same Code Cell
- Student code cells must contain ONLY `# YOUR SOLUTION CODE HERE` (plus optional hints)
- Instructor solution code must be in a SEPARATE cell marked with `# INSTRUCTOR SOLUTION`
- **Why:** Mixed cells leak the full solution into the student notebook since the cell doesn't get removed

### ❌ DON'T: Forget Co-Authorship
- Every commit MUST include: `Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>`
- **Why:** Attribution and transparency

### ❌ DON'T: Overwrite CONVERSATION_LOG.md
- Always APPEND to the log, never replace
- **Why:** Lose project history and context

### ❌ DON'T: Push Content Changes Without Rendering Quarto
- ALWAYS run `quarto render` and commit docs/ after ANY content change
- This includes: .qmd files, notebooks, images, syllabus, schedule
- **Why:** GitHub Pages serves docs/, not the source files. If you don't render and commit docs/, the website won't update even though you pushed your changes!
- **Common mistake:** Updating a notebook, committing it, pushing, but forgetting to render → website shows old version

### ❌ DON'T: Use Unescaped `$` for Money in Markdown Cells
- In notebook markdown cells, always write `\$50,000` not `$50,000`
- An unescaped `$` triggers LaTeX math mode in Google Colab, breaking text rendering
- This applies to markdown cells only — `$` in Python code strings is fine
- **Why:** Colab renders markdown with MathJax; `$50,000` becomes a broken math expression instead of displaying as a dollar amount

### ❌ DON'T: Add Complexity Without Request
- No extra features, refactoring, or "improvements" unless asked
- Keep code simple and focused
- **Why:** Over-engineering confuses students and adds maintenance burden

---

## ⚡ Quick Reference Commands

### Quarto Operations
```bash
# Render entire site
/Applications/RStudio.app/Contents/Resources/app/quarto/bin/quarto render

# Preview site locally
/Applications/RStudio.app/Contents/Resources/app/quarto/bin/quarto preview

# Render specific file
/Applications/RStudio.app/Contents/Resources/app/quarto/bin/quarto render index.qmd

# Check Quarto version
/Applications/RStudio.app/Contents/Resources/app/quarto/bin/quarto --version
```

### Git Operations
```bash
# Status check
git status
git log --oneline -10

# Common workflow
git add [specific files]
git commit -m "type: message

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
git push origin main

# View remote
git remote -v

# Check branch
git branch -a
```

### File Operations
```bash
# List notebooks
ls -la notebooks/

# Count notebooks
ls notebooks/*.ipynb | wc -l

# Search in notebooks
grep -r "RANDOM_SEED" notebooks/

# Check docs directory
ls -la docs/ | head -20
```

### Repository URLs
- **Repository:** https://github.com/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474
- **Website:** https://davi-moreira.github.io/2026Summer_predictive_analytics_purdue_MGMT474/
- **Old course (reference):** https://davi-moreira.github.io/2025F_predictive_analytics_purdue_MGMT474/

---

## 🔧 Troubleshooting

### Issue: Quarto Render Fails
**Symptoms:** Error when running `quarto render`

**Solutions:**
1. Check _quarto.yml syntax (YAML is whitespace-sensitive)
2. Verify all .qmd files have valid frontmatter
3. Check for broken links in .qmd files
4. Try rendering individual files first: `quarto render index.qmd`

### Issue: GitHub Pages Not Updating
**Symptoms:** Website shows old content after push

**Solutions:**
1. Wait 2-5 minutes (first deployment can take longer)
2. Check GitHub Actions: Repository → Actions tab
3. Verify docs/ directory exists and contains index.html
4. Hard refresh browser (Cmd+Shift+R on Mac, Ctrl+Shift+R on Windows)
5. Check GitHub Pages settings: Settings → Pages → main branch, /docs folder

### Issue: Notebook Won't Run in Colab
**Symptoms:** Errors when clicking "Open in Colab"

**Solutions:**
1. Check Colab badge URL (must match notebook filename)
2. Verify all imports are standard Python libraries
3. Test in fresh Colab runtime (Runtime → Disconnect and delete runtime)
4. Check for hardcoded file paths (use URLs instead)

### Issue: Git Push Rejected
**Symptoms:** `! [rejected] main -> main (fetch first)`

**Solutions:**
1. Pull first: `git pull origin main`
2. Resolve any conflicts
3. Push again: `git push origin main`

### Issue: Missing Files After Clone
**Symptoms:** Expected files not present after git clone

**Solutions:**
1. Check .gitignore - files may be excluded intentionally
2. _adm_stuff/, large files are excluded by design
3. docs/ should be present (if not, run `quarto render`)

---

## ✅ Session Start Checklist

At the beginning of EVERY session:
- [ ] Read this file (claude.md)
- [ ] Read CONVERSATION_LOG.md (understand current state)
- [ ] Run `git status` (check for uncommitted changes)
- [ ] Run `git log --oneline -5` (see recent work)
- [ ] Ask user what they want to accomplish
- [ ] Check if related to existing task in CONVERSATION_LOG.md

---

## ✅ Session End Checklist

At the end of EVERY session:
- [ ] All changes committed with clear messages
- [ ] **CRITICAL — Voice-check was run** on any modified student notebook (grep returns zero non-"Student's t" hits). If video guides were modified, run the blockquote-only voice-check too — wrapper-prose "students" refs are fine, blockquote-script "students" refs are not.
- [ ] **CRITICAL — CV-first audit was run** if any nb09–nb20 evaluation code changed. The tight audit (see "CV-First Evaluation + Test-Set Lock" rule above) must return zero model-eval hits on `X_test`/`y_test` outside of nb14 cell 33 and nb18's Kaggle-submission demo.
- [ ] **CRITICAL — Narrative polish applied** if any new or rewritten student markdown cells landed. Check for: named business stakeholder in Why-This-Matters, narrative prose over bullet lists in Reading-the-output, at least one `"A question that often comes up here"` Q&A block, warm wrap-up with an explicit bridge to the next notebook.
- [ ] **CRITICAL:** If ANY content changed (.qmd files, notebooks, images):
  - [ ] Run `quarto render`
  - [ ] Commit docs/ folder
- [ ] CONVERSATION_LOG.md updated with session summary
- [ ] If notebooks changed: Tested in Colab
- [ ] Git pushed to origin main (includes BOTH content AND docs/)
- [ ] Provide clear summary to user of what was accomplished
- [ ] List any remaining work for next session

**Note:** The most common mistake is forgetting to render Quarto and commit docs/. This causes the website to be out of sync with the repository content. The second most common mistake is committing a student-notebook polish without running the voice-check grep — hits in student files are rejected on review, so catching them pre-commit is cheaper than a revert.

---

## 📚 Resources & References

### Project Documentation
- **CONVERSATION_LOG.md** - Session history and decisions
- **README.md** - Public documentation
- **MGMT47400_Online4Week_Plan_2026Summer.md** - Master course plan (THE SOURCE OF TRUTH)
- **claude_course_plan.md** - Implementation plan
- **GITHUB_SETUP_INSTRUCTIONS.md** - Deployment guide

### External Resources
- **Quarto Documentation:** https://quarto.org/docs/guide/
- **GitHub Pages Docs:** https://docs.github.com/en/pages
- **Jupyter Notebook Format:** https://nbformat.readthedocs.io/
- **ISLP Book (course textbook):** https://www.statlearning.com/
- **scikit-learn User Guide:** https://scikit-learn.org/stable/user_guide.html

### Course-Specific
- **Purdue Daniels School:** https://www.purdue.edu/daniels/
- **Instructor Website:** https://davi-moreira.github.io/
- **LMS:** Brightspace (purdue.brightspace.com)

### Git Conventions
- **Conventional Commits:** https://www.conventionalcommits.org/
- **Types:** feat, fix, docs, style, refactor, test, chore, build

---

## 🎯 Purpose of This File

This file serves as:
1. **Onboarding doc** - New AI assistant can start immediately
2. **Reference manual** - Quick lookup for conventions and commands
3. **Decision log** - Understanding WHY things are done this way
4. **Quality control** - Checklist to ensure consistency
5. **Efficiency tool** - Copy-paste commands and workflows

**Key Principle:** An AI assistant should be able to read this file and be 80% operational within 5 minutes.

---

**Last Updated:** April 24, 2026
**Version:** 1.1 — adds Narrative Polish Pattern + CV-First Evaluation / Test-Set Lock rules; extends Voice Rule to cover video-guide blockquotes; adds polish + audit gates to Session End Checklist
**Maintained by:** Professor Davi Moreira + AI Assistants
