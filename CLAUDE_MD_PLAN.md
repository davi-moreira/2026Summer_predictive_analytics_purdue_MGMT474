# Plan for claude.md File

## Purpose
Create a comprehensive `claude.md` file that serves as the "operating manual" for AI assistants working on this project. This file should be read at the start of every session to ensure consistency, efficiency, and adherence to established patterns.

---

## Proposed Structure for claude.md

### 1. **Project Identity & Mission** (Top Priority)
**Purpose:** Immediately orient the AI to what this project is about

**Content:**
```markdown
# 2026 Summer Predictive Analytics Course - AI Assistant Guide

## Project Mission
This repository contains MGMT 47400 - Predictive Analytics, a 4-week intensive summer course (20 business days) for Purdue University's Daniels School of Business. The course runs May 18 - June 14, 2027, with 112.5 minutes of daily engagement through micro-videos (≤12 min) and Google Colab notebooks.

## Key Context
- **Instructor:** Professor Davi Moreira
- **Institution:** Purdue University, Daniels School of Business
- **Format:** Fully online, 4-week intensive (20 business days, Mon-Fri only)
- **Pedagogy:** Concept → Demo → Practice (PAUSE-AND-DO) → Solution → Repeat
- **Technology:** Google Colab + Google Gemini AI assistance
- **Deployment:** GitHub Pages (Quarto-based static site)
```

---

### 2. **Critical "Read This First" Section**
**Purpose:** Provide immediate actionable guidance

**Content:**
```markdown
## 🚨 READ THIS FIRST - Critical Guidelines

### Before Making ANY Changes:
1. **Read CONVERSATION_LOG.md** - Understand what's been done and why
2. **Check current git status** - `git status` to see uncommitted changes
3. **Never commit directly to main** - Always review changes first
4. **Follow established patterns** - Check existing notebooks/files for structure
5. **Test before committing** - Render Quarto site, test notebooks in Colab

### Core Principles:
- **Consistency is king:** All 20 notebooks follow identical structure
- **Documentation always:** Update CONVERSATION_LOG.md after major changes
- **Atomic commits:** One logical change per commit, clear messages
- **Student-first:** Every change should improve student learning experience
- **Reproducibility:** All code must run in fresh Colab environment
```

---

### 3. **File Structure & Navigation**
**Purpose:** Quick reference for finding files

**Content:**
```markdown
## 📁 Repository Structure

### Primary Directories
```
├── notebooks/               # 20 Jupyter notebooks (Days 1-20)
│   ├── 01_launchpad_eda_splits.ipynb
│   ├── ...
│   └── 20_final_submission_peer_review.ipynb
├── docs/                    # GitHub Pages output (compiled by Quarto)
│   ├── index.html
│   ├── schedule.html
│   ├── syllabus.html
│   └── notebooks/          # HTML versions of notebooks
├── lecture_slides/          # Legacy slides (maintained for reference)
├── images/                  # Course logo and assets
├── _adm_stuff/             # EXCLUDED from git (admin materials)
└── [Quarto source files]   # .qmd files in root
```

### Key Files (Always Check These)
- **CONVERSATION_LOG.md** - Development history and decisions
- **MGMT47400_Online4Week_Plan_2026Summer.md** - Master course plan (956 lines)
- **README.md** - Public-facing documentation
- **claude_course_plan.md** - Implementation plan
- **_quarto.yml** - Quarto configuration
- **index.qmd, schedule.qmd, syllabus.qmd** - Main website pages

### Where to Find Information
- Course dates/schedule → `MGMT47400_Online4Week_Plan_2026Summer.md`
- Notebook structure → Any notebook in `notebooks/`
- Git workflow → `CONVERSATION_LOG.md`
- Deployment steps → `GITHUB_SETUP_INSTRUCTIONS.md`
```

---

### 4. **Established Conventions & Patterns**
**Purpose:** Ensure all new content follows existing patterns

**Content:**
```markdown
## 📐 Established Conventions

### Notebook Structure (MUST FOLLOW)
Every notebook MUST include (in order):
1. **Header cell** (markdown)
   - Title: "Day X: [Topic]"
   - Course info: "MGMT 47400 - Predictive Analytics"
   - Date: "[Day of week] [Month Day], 2027"
   - Colab badge: `[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](...)`
   - Separator: "---"

2. **Learning Objectives** (markdown)
   - "By the end of this notebook, you will be able to:"
   - 4-5 numbered objectives (ability-focused)

3. **Setup Section** (code)
   - Standard imports: pandas, numpy, matplotlib, seaborn, sklearn
   - Settings: warnings, display options, style, random seed
   - Print: "✓ Setup complete!"

4. **Content Sections** (numbered 1, 2, 3...)
   - Clear section headers
   - Code cells with explanation markdown before each
   - Visualizations with clear labels

5. **PAUSE-AND-DO Exercises** (2 per notebook, 10 min each)
   - Header: "## 📝 PAUSE-AND-DO Exercise X (10 minutes)"
   - Task description
   - Instructions (3-4 bullets)
   - "What to look for" guidance
   - "### YOUR ANSWER HERE:" placeholder

6. **Wrap-Up Section** (markdown)
   - "What We Learned Today" (4-5 bullets)
   - Critical rules in blockquotes (>)
   - "Next Steps" preview

7. **Bibliography** (markdown)
   - Academic citations (ISLP, ESL, papers)
   - scikit-learn docs
   - Relevant online resources

8. **Footer** (markdown)
   - "**End of Day X Notebook** [emoji]"

### Naming Conventions
- **Notebooks:** `NN_topic_description.ipynb` (e.g., `01_launchpad_eda_splits.ipynb`)
- **Git commits:** `<type>: <subject>` (types: feat, docs, chore, build, fix)
  - Always include: "Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
- **Variables:** lowercase_with_underscores (Python convention)
- **Constants:** UPPERCASE (e.g., `RANDOM_SEED = 42`)

### Style Guidelines
- **Random seed:** Always `RANDOM_SEED = 42`
- **Train/Val/Test split:** Always 60/20/20
- **Figure size:** `plt.rcParams['figure.figsize'] = (10, 6)`
- **Display precision:** `pd.set_option('display.precision', 3)`
- **Emoji usage:** ✓ for success, ⚠️ for warnings, 📝 for exercises, 💡 for insights
- **No unnecessary code:** Don't add features not explicitly requested
```

---

### 5. **Common Tasks & Workflows**
**Purpose:** Step-by-step guides for frequent operations

**Content:**
```markdown
## 🔧 Common Tasks & Workflows

### Task 1: Add a New Notebook
1. **Choose the right day number** (01-20)
2. **Copy structure from existing notebook** (e.g., `01_launchpad_eda_splits.ipynb`)
3. **Update header** with correct day, date, topic
4. **Update Colab badge URL** to match filename
5. **Write 4-5 learning objectives**
6. **Follow the established section structure**
7. **Include 2 PAUSE-AND-DO exercises** (10 min each)
8. **Add bibliography section**
9. **Test in Colab:** Click "Open in Colab" → "Runtime → Run all"
10. **Commit:** `git add notebooks/XX_topic.ipynb && git commit -m "feat: Add Day XX notebook"`

### Task 2: Update the Schedule
1. **Read the course plan** (`MGMT47400_Online4Week_Plan_2026Summer.md`)
2. **Edit `schedule.qmd`**
3. **Add/update row in table** with: Day | Date | Topic | Videos | Notebook | Assessment | Materials
4. **Use correct date** (business days only, May 18 - June 14, 2027)
5. **Link to notebook:** `https://github.com/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/XX_topic.ipynb`
6. **Add Colab badge:** `[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](...)`
7. **Render site:** `/Applications/RStudio.app/Contents/Resources/app/quarto/bin/quarto render`
8. **Commit:** `git add schedule.qmd docs/ && git commit -m "docs: Update schedule for Day XX"`

### Task 3: Render & Deploy Website
```bash
# Render Quarto site
/Applications/RStudio.app/Contents/Resources/app/quarto/bin/quarto render

# Check rendered output
ls -la docs/

# Commit changes
git add docs/
git commit -m "build: Render Quarto site"

# Push to GitHub
git push origin main

# GitHub Pages will auto-deploy (wait 1-2 minutes)
# Visit: https://davi-moreira.github.io/2026Summer_predictive_analytics_purdue_MGMT474/
```

### Task 4: Update CONVERSATION_LOG.md
**When:** After completing any major work or at end of session

**Include:**
1. **Session date and objective**
2. **Work completed** (files created/updated)
3. **Key decisions made** (with rationale)
4. **Problems encountered** (and solutions)
5. **Next steps** (what's still needed)

**Template:**
```markdown
## Session X: [Date]

### Objective
[What was the goal of this session?]

### Work Completed
- [List of accomplishments]

### Decisions Made
- [Key choices with rationale]

### Next Steps
- [ ] [Remaining tasks]
```

### Task 5: Git Workflow (Standard)
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
```

---

### 6. **Technology Stack & Tools**
**Purpose:** Reference for all tools used in project

**Content:**
```markdown
## 🛠️ Technology Stack

### Primary Technologies
- **Quarto (v1.4+):** Static site generator
  - Location: `/Applications/RStudio.app/Contents/Resources/app/quarto/bin/quarto`
  - Command: `quarto render` (renders .qmd → HTML in docs/)
- **Git:** Version control
  - Remote: `https://github.com/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474.git`
  - Branch: `main`
- **GitHub Pages:** Hosting
  - Source: `docs/` directory on main branch
  - URL: `https://davi-moreira.github.io/2026Summer_predictive_analytics_purdue_MGMT474/`

### Python Stack (for notebooks)
- **Python 3.8+**
- **Core libraries:**
  - pandas, numpy (data manipulation)
  - matplotlib, seaborn (visualization)
  - scikit-learn (machine learning)
  - joblib (model persistence)
- **Platform:** Google Colab (Jupyter notebooks in cloud)
- **AI Assistant:** Google Gemini (integrated in Colab)

### Deployment Workflow
```
.qmd files → quarto render → docs/ → git push → GitHub Pages
```
```

---

### 7. **Decision History & Rationale**
**Purpose:** Understand WHY things are the way they are

**Content:**
```markdown
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

### Decision 3: RANDOM_SEED = 42 Everywhere
**Decision:** All random operations use seed 42
**Rationale:**
- Complete reproducibility
- Students get identical outputs
- Easier to debug (same results every time)
- Industry-standard "joke" seed

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
```

---

### 8. **What NOT to Do (Anti-Patterns)**
**Purpose:** Prevent common mistakes

**Content:**
```markdown
## 🚫 What NOT to Do

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
- Always use `RANDOM_SEED = 42`
- **Why:** Breaks reproducibility, students get different results

### ❌ DON'T: Skip Testing in Colab
- Always click "Open in Colab" and "Run All" before committing
- **Why:** Notebooks MUST work in fresh Colab environment

### ❌ DON'T: Forget Co-Authorship
- Every commit MUST include: "Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
- **Why:** Attribution and transparency

### ❌ DON'T: Overwrite CONVERSATION_LOG.md
- Always APPEND to the log, never replace
- **Why:** Lose project history and context

### ❌ DON'T: Push Without Rendering Quarto
- Always run `quarto render` before pushing website changes
- **Why:** GitHub Pages serves docs/, must be up-to-date

### ❌ DON'T: Add Complexity Without Request
- No extra features, refactoring, or "improvements" unless asked
- Keep code simple and focused
- **Why:** Over-engineering confuses students and adds maintenance burden
```

---

### 9. **Quick Reference Commands**
**Purpose:** Copy-paste common commands

**Content:**
```markdown
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
git commit -m "type: message"
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

# Search in notebooks (requires jq)
grep -r "RANDOM_SEED" notebooks/

# Check docs directory
ls -la docs/ | head -20
```

### Repository URLs
- **Repository:** https://github.com/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474
- **Website:** https://davi-moreira.github.io/2026Summer_predictive_analytics_purdue_MGMT474/
- **Old course (reference):** https://davi-moreira.github.io/2025F_predictive_analytics_purdue_MGMT474/
```

---

### 10. **Troubleshooting Guide**
**Purpose:** Quick fixes for common issues

**Content:**
```markdown
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
2. Check GitHub Actions: Settings → Pages
3. Verify docs/ directory exists and contains index.html
4. Hard refresh browser (Cmd+Shift+R on Mac, Ctrl+Shift+R on Windows)
5. Check GitHub Pages settings: main branch, /docs folder

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
```

---

### 11. **Session Checklist**
**Purpose:** Standard checklist for every session

**Content:**
```markdown
## ✅ Session Start Checklist

At the beginning of EVERY session:
- [ ] Read this file (claude.md)
- [ ] Read CONVERSATION_LOG.md (understand current state)
- [ ] Run `git status` (check for uncommitted changes)
- [ ] Run `git log --oneline -5` (see recent work)
- [ ] Ask user what they want to accomplish
- [ ] Check if related to existing task in CONVERSATION_LOG

## ✅ Session End Checklist

At the end of EVERY session:
- [ ] All changes committed with clear messages
- [ ] CONVERSATION_LOG.md updated with session summary
- [ ] If website changed: Quarto rendered and docs/ committed
- [ ] If notebooks changed: Tested in Colab
- [ ] Git pushed to origin main
- [ ] Provide clear summary to user of what was accomplished
- [ ] List any remaining work for next session
```

---

### 12. **Resources & References**
**Purpose:** Where to find more information

**Content:**
```markdown
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
```

---

## 🎯 Summary: The Purpose of claude.md

This file should serve as:
1. **Onboarding doc** - New AI assistant can start immediately
2. **Reference manual** - Quick lookup for conventions and commands
3. **Decision log** - Understanding WHY things are done this way
4. **Quality control** - Checklist to ensure consistency
5. **Efficiency tool** - Copy-paste commands and workflows

**Key Principle:** An AI assistant should be able to read this file and be 80% operational within 5 minutes.
```

---

## Implementation Notes

### File Location
- **Filename:** `claude.md` (lowercase, in repository root)
- **Why lowercase:** Convention for dotfiles and config files

### File Format
- **Markdown** with clear section headers
- **Emoji** for visual scanning (🚨, 📁, 🔧, etc.)
- **Code blocks** with syntax highlighting
- **Tables** for structured data
- **Blockquotes** for emphasis

### Maintenance
- **Update after major changes** to structure/conventions
- **Keep concise** - remove outdated info
- **Version in git** - track changes over time
- **Reference from CONVERSATION_LOG** when decisions change

### Reading Priority (for AI)
1. "READ THIS FIRST" section
2. Established Conventions
3. Common Tasks relevant to current work
4. Decision History (as needed)
5. Everything else (reference)

---

## Next Steps

1. **Review this plan** with user
2. **Create claude.md** based on this structure
3. **Test it** - Have AI read it at start of next session
4. **Iterate** - Refine based on actual usage
5. **Commit** with message: "docs: Add claude.md AI assistant guide"

---

This plan ensures future AI assistants can work effectively on this project with minimal onboarding, maintaining consistency and quality across all sessions.
