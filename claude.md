# 2026 Summer Predictive Analytics Course - AI Assistant Guide

## Project Mission

This repository contains **MGMT 47400 - Predictive Analytics**, a 4-week intensive summer course (20 business days) for Purdue University's Daniels School of Business. The course runs **May 18 - June 14, 2027**, with **112.5 minutes of daily engagement** through micro-videos (≤12 min) and Google Colab notebooks.

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

### 🚨 CRITICAL WORKFLOW - Keep Video Lecture Guides in Sync

**EVERY TIME a notebook (`notebooks/NN_*.ipynb`) is updated, you MUST also update the corresponding video lecture guide (`video_guides/NN_video_lecture_guide.md`).**

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
git add notebooks/XX_topic.ipynb  # or schedule.qmd, syllabus.qmd, etc.
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
├── notebooks/                  # 20 Jupyter notebooks (Days 1-20)
│   ├── 01_launchpad_eda_splits.ipynb
│   ├── 02_preprocessing_pipelines.ipynb
│   ├── ...
│   └── 20_final_submission_peer_review.ipynb
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

> **Canonical reference:** `notebooks/01_launchpad_eda_splits.ipynb` is the reference template for all notebook structure and formatting. When creating or updating notebooks, match its header format, section organization, and conventions exactly.

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

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/XX_topic.ipynb)

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
```markdown
## 📝 PAUSE-AND-DO Exercise X (10 minutes)

**Task:** [Clear, specific task]

**Instructions:**
1. [Step 1]
2. [Step 2]
3. [Step 3]

**What to look for:**
[Guidance on interpretation]

---

### YOUR ANSWER HERE:

**[Question 1]:**
[Student response]

**[Question 2]:**
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

- **Notebooks:** `NN_topic_description.ipynb` (e.g., `01_launchpad_eda_splits.ipynb`)
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
- **No unnecessary code:** Don't add features not explicitly requested

---

## 🔧 Common Tasks & Workflows

### Task 1: Add a New Notebook

1. **Choose the right notebook number** (01-20)
2. **Copy structure from canonical notebook** (`01_launchpad_eda_splits.ipynb`)
3. **Update header** with correct topic title (no "Day X:" prefix, no date)
4. **Update Colab badge URL** to match filename
5. **Write 4-5 learning objectives**
6. **Follow the established section structure**
7. **Include 2 PAUSE-AND-DO exercises** (10 min each)
8. **Add bibliography section**
9. **Test in Colab:** Click "Open in Colab" → "Runtime → Run all"
10. **Commit:**
    ```bash
    git add notebooks/XX_topic.ipynb
    git commit -m "feat: Add Day XX notebook

    Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
    ```

### Task 2: Update the Schedule

1. **Read the course plan** (`MGMT47400_Online4Week_Plan_2026Summer.md`)
2. **Edit `schedule.qmd`**
3. **Add/update row in table** with: Day | Date | Topic | Videos | Notebook | Assessment | Materials
4. **Use correct date** (business days only, May 18 - June 14, 2027)
5. **Link to notebook:**
   ```
   [XX_topic.ipynb](https://github.com/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/XX_topic.ipynb)
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
- [ ] **CRITICAL:** If ANY content changed (.qmd files, notebooks, images):
  - [ ] Run `quarto render`
  - [ ] Commit docs/ folder
- [ ] CONVERSATION_LOG.md updated with session summary
- [ ] If notebooks changed: Tested in Colab
- [ ] Git pushed to origin main (includes BOTH content AND docs/)
- [ ] Provide clear summary to user of what was accomplished
- [ ] List any remaining work for next session

**Note:** The most common mistake is forgetting to render Quarto and commit docs/. This causes the website to be out of sync with the repository content.

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

**Last Updated:** January 27, 2026
**Version:** 1.0
**Maintained by:** Professor Davi Moreira + AI Assistants
