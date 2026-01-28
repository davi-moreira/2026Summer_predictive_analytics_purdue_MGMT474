# Implementation Plan: 2026 Summer Predictive Analytics Course Migration

## Overview
Transform the 2025F semester-based course into a 2026 Summer 4-week intensive format (20 business days) while maintaining GitHub Pages deployment and git best practices.

## Context Summary

### Current State
- **Repository:** `/Users/dcordeir/Dropbox/academic/cursos/cursos-davi/predictive_analytics/2026Summer_predictive_analytics_purdue_MGMT474`
- **Old format:** 16-week semester with Quarto slides in `/lecture_slides/`
- **New format:** 4-week intensive with Jupyter notebooks in `/notebooks/`
- **Status:** 17/20 notebooks created (Days 1-17 complete, Days 18-20 missing)
- **Course plan:** `MGMT47400_Online4Week_Plan_2026Summer.md` (complete, 956 lines)
- **GitHub Pages:** Uses Quarto → `docs/` → GitHub Pages
- **Old course URL:** https://davi-moreira.github.io/2025F_predictive_analytics_purdue_MGMT474/

### What Needs to Be Done
1. Complete missing notebooks (Days 18-20)
2. Update Quarto website configuration for new 20-day structure
3. Create new schedule page mapping to 20 business days
4. Update syllabus for 4-week format
5. Initialize local git repository
6. Connect to GitHub remote
7. Configure for GitHub Pages deployment
8. Create conversation log for future resumption

---

## Phase 1: Complete Notebook Creation (Days 18-20)

### Files to Create

#### 1.1 Day 18: `notebooks/18_reproducibility_monitoring.ipynb`
**Source:** MGMT47400_Online4Week_Plan_2026Summer.md lines 812-856
**Structure:**
- Header with Colab badge
- Learning objectives (5 items: packaging, saving/loading, monitoring, checklists, reproducibility)
- Section 1: Setup (imports: joblib, Pipeline, StandardScaler)
- Section 2: Refactor notebook into functions
  - `train_model(config)` → returns pipeline + metrics dict
  - `predict(model, X)` → returns predictions
  - `evaluate(model, X, y)` → returns metrics
- Section 3: Save/load model artifacts
  - Use joblib to save pipeline
  - Load and verify reproducibility
- Section 4: Monitoring plan template
  - Data drift signals (feature distribution changes)
  - Performance drift (metric degradation)
  - Calibration drift
  - Table with Signal | Threshold | Owner | Action
- Section 5: "Ready-to-share" notebook hygiene checklist
- Section 6: PAUSE-AND-DO exercises (2 exercises)
  - Exercise 1: Implement `train_model(config)` returning pipeline + metrics (10 min)
  - Exercise 2: Draft monitoring plan with 5-8 signals and owners (10 min)
- Section 7: Wrap-up (key takeaways, critical rules)
- Bibliography (Chip Huyen, scikit-learn User Guide, Dataset Shift papers)

#### 1.2 Day 19: `notebooks/19_project_narrative_video_studio.ipynb`
**Source:** MGMT47400_Online4Week_Plan_2026Summer.md lines 859-899
**Structure:**
- Header with Colab badge
- Learning objectives (5 items: executive narrative, slide structure, visuals, video script, deliverable finalization)
- Section 1: Setup (imports: none needed - mostly markdown)
- Section 2: Executive narrative structure
  - Problem → Approach → Results → Recommendation → Risks
  - 1-slide-per-idea principle
- Section 3: Storyboard builder (10-slide target)
  - Template with markdown cells for each slide
  - Slide 1: Problem statement
  - Slide 2: Data overview
  - Slide 3: Approach summary
  - Slides 4-6: Results (comparison table, key plots)
  - Slide 7: Recommendation
  - Slide 8: Risks & limitations
  - Slide 9: Next steps
  - Slide 10: Q&A preparation
- Section 4: Required visuals checklist
  - Model comparison table
  - Best model performance plot
  - Feature importance or PDP
  - Error analysis segment
  - Decision policy (if classification)
- Section 5: Script template (time-coded for 2-3 minute video)
- Section 6: PAUSE-AND-DO exercises (2 exercises)
  - Exercise 1: Create 10-slide outline (titles + 2 bullets each) (10 min)
  - Exercise 2: Write 2-3 minute script aligned to outline (10 min)
- Section 7: Wrap-up
- Bibliography (Cole Nussbaumer Knaflic, Barbara Minto, Provost & Fawcett)

#### 1.3 Day 20: `notebooks/20_final_submission_peer_review.ipynb`
**Source:** MGMT47400_Online4Week_Plan_2026Summer.md lines 902-947
**Structure:**
- Header with Colab badge
- Learning objectives (5 items: complete package, executive deck, video, peer review, postmortem)
- Section 1: Setup (imports: none needed)
- Section 2: Final self-audit checklist
  - Run-all check (all cells execute without errors)
  - Outputs present for all code cells
  - Links working (Colab, GitHub, data sources)
  - Test set lockbox maintained
  - Model card complete
  - Monitoring plan included
- Section 3: Submission links + artifact manifest
  - Final notebook link
  - Slide deck link (Google Slides or PDF)
  - Video link (Loom, YouTube, Google Drive)
  - GitHub repo link (if applicable)
- Section 4: Peer review rubric
  - Reproducibility (run-all works?) - 20 points
  - Methodology (split/CV/metrics sound?) - 25 points
  - Results (comparison fair? interpretation honest?) - 25 points
  - Communication (narrative clear? visuals effective?) - 20 points
  - Responsible AI (limitations? fairness? risks?) - 10 points
- Section 5: Peer review form template
  - Rubric scores (checkboxes)
  - 3 actionable edits (text boxes)
  - 1 strength to highlight
- Section 6: PAUSE-AND-DO exercises (2 exercises)
  - Exercise 1: Run-all audit and fix one reproducibility issue (10 min)
  - Exercise 2: Complete one peer review with rubric + 3 actionable edits (10 min)
- Section 7: Postmortem prompts (8-10 lines)
  - What worked well?
  - What didn't work?
  - What would you do differently?
  - What's the next iteration?
- Section 8: Wrap-up (course completion, next steps for learning)
- Bibliography (Mitchell et al., Chip Huyen, Storytelling with Data)

**Pattern Consistency:**
All three notebooks follow the established template:
- Colab badge in header
- 112.5 minute time budget (videos + notebook work + quiz/submission)
- 2 PAUSE-AND-DO exercises (10 min each)
- Setup → Content sections → Exercises → Wrap-up → Bibliography
- Docstrings for any functions
- Blockquotes for critical rules
- ✓ Checkmarks for confirmations

---

## Phase 2: Update Quarto Website Structure

### 2.1 Update `_quarto.yml`
**File:** `_quarto.yml`
**Changes:**
- Keep existing structure (website, docs output)
- Update title to: "MGMT 47400: Predictive Analytics (Summer 2026 - 4-Week Intensive)"
- Update GitHub link to new repo name (TBD)
- Keep sidebar structure (Home, Syllabus, Schedule and Material)

### 2.2 Update `index.qmd` (Homepage)
**File:** `index.qmd`
**Changes:**
- Update course dates: "May 18 - June 14, 2027 (20 business days)"
- Update format description: "4-week fully online intensive"
- Update course structure:
  - 20 business days (Mon-Fri)
  - 112.5 minutes per day engagement
  - Micro-videos (≤12 min each) + Google Colab notebooks
  - Single capstone project with 4 weekly milestones
- Update instructor info if needed
- Link to new GitHub repo

### 2.3 Create New `schedule.qmd`
**File:** `schedule.qmd`
**Strategy:** Replace 16-week semester schedule with 20-day intensive schedule
**Structure:**
- Introduction: 20 business days, May 18 - June 14, 2027
- Weekly breakdown with daily rows
- Columns:
  - **Day** (1-20)
  - **Date** (May 18 - June 14, business days only)
  - **Topic** (from course plan)
  - **Videos** (micro-video count + total time)
  - **Notebook** (link to Colab-ready notebook on GitHub)
  - **Quiz/Assessment** (auto-graded quiz or milestone)
  - **Materials** (bibliography references)

**Table structure:**
```markdown
| Day | Date | Topic | Videos | Notebook | Assessment | Materials |
|-----|------|-------|--------|----------|------------|-----------|
| 1 | Tue May 18 | Launchpad: EDA, Splits, Leakage | 6 videos (54 min) | [01_launchpad](link) | Concept Quiz | ISLP Ch2, sklearn |
| 2 | Wed May 19 | Preprocessing Pipelines | 6 videos (54 min) | [02_preprocessing](link) | Concept Quiz | sklearn Pipelines |
...
```

**Project milestones highlighted:**
- Day 5: Project Proposal due
- Day 10: Baseline Model + Midterm due
- Day 15: Improved Model due
- Day 20: Final Deliverable due (notebook + deck + video)

### 2.4 Update `syllabus.qmd`
**File:** `syllabus.qmd`
**Changes:**
- Update course title: "MGMT 47400 - Predictive Analytics (Summer 2026, 4-Week Online Intensive)"
- Update dates: May 18 - June 14, 2027
- Update course description:
  - 20 business days
  - 112.5 minutes daily engagement
  - Micro-videos + Colab notebooks
  - Single capstone project
- Update grading breakdown:
  - Daily quizzes: 20% (20 quizzes × 1%)
  - Midterm (Day 10): 15%
  - Project milestones:
    - Proposal (Day 5): 5%
    - Baseline (Day 10): 10%
    - Improved model (Day 15): 15%
    - Final deliverable (Day 20): 25%
  - Peer review (Day 20): 10%
- Update calendar/schedule section to reference 20-day structure
- Update textbooks (same: ISLP, ESL, Provost & Fawcett)
- Update technology requirements:
  - Google Colab (primary platform)
  - Google Gemini (AI assistance)
  - Brightspace LMS
- Add daily engagement expectations section

---

## Phase 3: Git Repository Setup

### 3.1 Initialize Local Git Repository
**Location:** Current directory
**Steps:**
1. Check if `.git/` already exists (it does based on git status output)
2. If exists, clean up current state
3. Create comprehensive `.gitignore`

### 3.2 Create/Update `.gitignore`
**File:** `.gitignore`
**Content:**
```gitignore
# R/RStudio
.Rproj.user
.Rhistory
.RData
.Ruserdata
*.Rproj

# Quarto
/.quarto/
_freeze/

# Python
__pycache__/
*.py[cod]
*$py.class
.ipynb_checkpoints/
*.pyc
.venv/
venv/
env/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Admin (keep out of public repo)
_adm_stuff/
_grades/
_students_contact/
_tas/
_accomodation/
_course_eval/
_reflection_*/

# Large files (exclude from repo, link instead)
*.zip
*.mp4
*.mp3
*.m4a
*.mov

# Keep docs for GitHub Pages
!docs/

# Temporary
*.tmp
*.bak
.scratch/
```

### 3.3 Initial Commit Structure
**Strategy:** Atomic commits for different components

**Commit sequence:**
1. `docs: Add course plan for 2026 Summer intensive`
   - Add `MGMT47400_Online4Week_Plan_2026Summer.md`

2. `feat: Add Days 1-17 notebooks for 4-week intensive`
   - Add all 17 existing notebooks in `notebooks/`

3. `feat: Add Days 18-20 notebooks (reproducibility, narrative, final)`
   - Add three newly created notebooks

4. `docs: Update Quarto website for 2026 Summer format`
   - Update `_quarto.yml`, `index.qmd`, `schedule.qmd`, `syllabus.qmd`

5. `chore: Add .gitignore for project`
   - Add `.gitignore`

6. `build: Render Quarto site for GitHub Pages`
   - Render site with `quarto render`
   - Commit updated `docs/` directory

**Commit message format:**
```
<type>: <subject>

<body>

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

---

## Phase 4: GitHub Remote Connection

### 4.1 Create GitHub Repository
**Method:** Use `gh` CLI (GitHub CLI)
**Steps:**
1. Check if `gh` is authenticated: `gh auth status`
2. Create new repo: `gh repo create 2026Summer_predictive_analytics_purdue_MGMT474 --public --description "MGMT 47400 Predictive Analytics - 4-Week Summer Intensive (2026)" --source=. --remote=origin`
3. Verify remote: `git remote -v`

**Repository settings:**
- **Visibility:** Public
- **Description:** "MGMT 47400 - Predictive Analytics | 4-Week Online Intensive | Purdue Daniels School of Business | Python/scikit-learn | Colab notebooks"
- **Topics:** `predictive-analytics`, `machine-learning`, `python`, `jupyter-notebook`, `scikit-learn`, `data-science`, `mba-course`, `colab`
- **Homepage:** (Will be set after GitHub Pages is configured)

### 4.2 Push to Remote
**Steps:**
1. `git branch -M main` (ensure main branch)
2. `git push -u origin main` (push with upstream tracking)

---

## Phase 5: GitHub Pages Configuration

### 5.1 Repository Settings
**Method:** Use `gh` CLI or web interface
**Configuration:**
- **Source:** Deploy from `docs/` directory on `main` branch
- **Custom domain:** None (use default `*.github.io`)
- **HTTPS:** Enforce HTTPS

**CLI command:**
```bash
gh repo edit --enable-pages --pages-branch main --pages-path docs
```

### 5.2 Verify Deployment
**Expected URL:** `https://davi-moreira.github.io/2026Summer_predictive_analytics_purdue_MGMT474/`

**Verification steps:**
1. Check GitHub Pages build status: `gh run list --workflow pages-build-deployment`
2. Wait for deployment (usually 1-2 minutes)
3. Visit URL and verify:
   - Homepage loads
   - Sidebar navigation works
   - Schedule page shows 20-day structure
   - Notebook links work (should point to GitHub raw notebooks)

### 5.3 Update Repository Homepage
**Command:**
```bash
gh repo edit --homepage "https://davi-moreira.github.io/2026Summer_predictive_analytics_purdue_MGMT474/"
```

---

## Phase 6: Documentation and Conversation Log

### 6.1 Create `CONVERSATION_LOG.md`
**File:** `CONVERSATION_LOG.md` (in repository root)
**Purpose:** Track all work done and decisions made for future session resumption

**Structure:**
```markdown
# Conversation Log: 2026 Summer Course Development

## Session 1: January 27, 2026

### Objective
Transform 2025F semester-based course into 2026 Summer 4-week intensive format (20 business days)

### Context
- **User:** Professor Davi Moreira
- **Course:** MGMT 47400 - Predictive Analytics
- **Institution:** Purdue Daniels School of Business
- **Old format:** 16-week semester (2025 Fall)
- **New format:** 4-week intensive (May 18 - June 14, 2027, 20 business days)

### Work Completed

#### 1. Notebooks Created
- **Days 1-17:** Already existed (created earlier in session)
- **Days 18-20:** Created new notebooks:
  - `18_reproducibility_monitoring.ipynb` - Deployment thinking, packaging, monitoring
  - `19_project_narrative_video_studio.ipynb` - Executive communication, slide narrative, video
  - `20_final_submission_peer_review.ipynb` - Final deliverable, peer review, postmortem

#### 2. Quarto Website Updated
- `_quarto.yml` - Updated title and configuration
- `index.qmd` - Updated for 4-week intensive format
- `schedule.qmd` - Complete rewrite for 20-day structure
- `syllabus.qmd` - Updated grading, calendar, format

#### 3. Git Repository
- Initialized local repository
- Created comprehensive `.gitignore`
- Made atomic commits for each component

#### 4. GitHub Connection
- Created remote repository
- Pushed all commits to main branch
- Configured GitHub Pages

### Key Decisions

1. **Notebook organization:** Keep flat structure in `/notebooks/` (01-20), don't nest by week
2. **Website deployment:** Continue using Quarto → docs/ → GitHub Pages (same as 2025F)
3. **Git strategy:** Atomic commits by component, maintain clean history
4. **Admin materials:** Keep `_adm_stuff/` but exclude from git
5. **Large files:** Link to external storage (Google Drive), don't commit to git

### File Structure
```
Repository Root
├── notebooks/                         # 20 Jupyter notebooks (Days 1-20)
├── docs/                              # GitHub Pages output (compiled by Quarto)
├── images/                            # Course logo and assets
├── _quarto.yml                        # Quarto configuration
├── index.qmd                          # Homepage
├── schedule.qmd                       # 20-day schedule table
├── syllabus.qmd                       # Course syllabus
├── styles.css                         # Custom styling
├── MGMT47400_Online4Week_Plan_2026Summer.md  # Master course plan
├── CONVERSATION_LOG.md                # This file
├── .gitignore                         # Git ignore rules
└── 2026Summer_predictive_analytics_purdue_MGMT474.Rproj  # RStudio project file
```
```

### 6.2 Create `README.md`
**File:** `README.md` (in repository root)
**Purpose:** Public-facing repository documentation

---

## Critical Files Summary

### Files to Create (New)
1. `notebooks/18_reproducibility_monitoring.ipynb`
2. `notebooks/19_project_narrative_video_studio.ipynb`
3. `notebooks/20_final_submission_peer_review.ipynb`
4. `CONVERSATION_LOG.md`
5. `README.md`
6. `.gitignore` (update existing or create)

### Files to Update (Existing)
1. `_quarto.yml` (minimal - update title and GitHub link)
2. `index.qmd` (update course dates, format, structure)
3. `schedule.qmd` (complete rewrite for 20-day structure)
4. `syllabus.qmd` (update grading, calendar, format)

### Files to Keep (No Changes)
1. All 17 existing notebooks (Days 1-17)
2. `MGMT47400_Online4Week_Plan_2026Summer.md` (master plan)
3. `styles.css` (custom CSS)
4. `images/` directory (logos, assets)
5. `2026Summer_predictive_analytics_purdue_MGMT474.Rproj`

---

## Success Criteria

### Must Have (Blocking)
- [ ] All 20 notebooks exist and have consistent structure
- [ ] Git repository initialized with clean history
- [ ] GitHub remote connected and pushed
- [ ] GitHub Pages deployed and accessible
- [ ] Schedule page shows all 20 days with correct dates
- [ ] Syllabus updated for 4-week format
- [ ] All Colab badges work in notebooks

### Should Have (High Priority)
- [ ] Conversation log created for future resumption
- [ ] README.md with comprehensive documentation
- [ ] All notebook links tested in Colab
- [ ] Website responsive on mobile

### Nice to Have (Future Work)
- [ ] Auto-graded quizzes created in Brightspace
- [ ] Micro-videos recorded and linked
- [ ] Sample project deliverable created
- [ ] All notebooks tested end-to-end in Colab

---

## Implementation Order

1. **Complete Notebooks** (Phase 1) - Days 18-20
2. **Update Quarto Website** (Phase 2) - _quarto.yml, index, schedule, syllabus
3. **Git Setup** (Phase 3) - .gitignore, commits
4. **GitHub Connection** (Phase 4) - create repo, push
5. **GitHub Pages** (Phase 5) - configure, verify
6. **Documentation** (Phase 6) - conversation log, README
7. **Verification** - Test everything works

---

## End of Plan

This plan provides a complete roadmap for migrating the 2025F semester course to the 2026 Summer 4-week intensive format while maintaining git best practices and GitHub Pages deployment.
