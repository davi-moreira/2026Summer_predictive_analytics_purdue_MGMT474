# GitHub Setup Instructions

## What's Been Completed ✓

All local work is complete! Here's what has been done:

### 1. Course Materials ✓
- **20 Jupyter notebooks** created (Days 1-20)
- **Quarto website** updated for 2026 Summer format
- **Course plan** documented (MGMT47400_Online4Week_Plan_2026Summer.md)
- **Implementation plan** saved (claude_course_plan.md)

### 2. Git Repository ✓
- Local git repository initialized
- Comprehensive `.gitignore` configured
- **6 atomic commits** made with proper messages:
  1. Course plan documents
  2. Complete 20-day notebook sequence
  3. Quarto website updates
  4. Project configuration
  5. Rendered Quarto site
  6. README and conversation log
- All commits include co-authorship attribution

### 3. Documentation ✓
- **README.md** - Public-facing repository documentation
- **CONVERSATION_LOG.md** - Development tracking for future sessions
- **GITHUB_SETUP_INSTRUCTIONS.md** - This file

---

## What You Need to Do

To complete the GitHub setup and deployment, follow these steps:

### Step 1: Authenticate GitHub CLI (if not already done)

```bash
gh auth login
```

Follow the prompts to authenticate with your GitHub account.

### Step 2: Create New GitHub Repository

Option A: Using GitHub CLI (Recommended)
```bash
cd "/Users/dcordeir/Dropbox/academic/cursos/cursos-davi/predictive_analytics/2026Summer_predictive_analytics_purdue_MGMT474"

gh repo create davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474 \
  --public \
  --description "MGMT 47400 - Predictive Analytics | 4-Week Summer Intensive | Purdue Daniels School of Business" \
  --source=. \
  --remote=origin \
  --push
```

Option B: Using GitHub Web Interface
1. Go to https://github.com/new
2. **Repository name:** `2026Summer_predictive_analytics_purdue_MGMT474`
3. **Description:** "MGMT 47400 - Predictive Analytics | 4-Week Summer Intensive | Purdue Daniels School of Business"
4. **Visibility:** Public
5. **Do NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"
7. Then run:
   ```bash
   git remote set-url origin https://github.com/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474.git
   git push -u origin main
   ```

### Step 3: Configure GitHub Pages

Option A: Using GitHub CLI
```bash
gh repo edit davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474 \
  --enable-pages \
  --pages-branch main \
  --pages-path docs
```

Option B: Using GitHub Web Interface
1. Go to repository Settings → Pages (left sidebar)
2. Under "Build and deployment":
   - **Source:** Deploy from a branch
   - **Branch:** `main`
   - **Folder:** `/docs`
3. Click "Save"
4. Wait 1-2 minutes for deployment

### Step 4: Verify Deployment

1. Check deployment status:
   ```bash
   gh run list --workflow pages-build-deployment
   ```

2. Visit your course website:
   ```
   https://davi-moreira.github.io/2026Summer_predictive_analytics_purdue_MGMT474/
   ```

3. Verify:
   - Homepage loads correctly
   - Sidebar navigation works
   - Schedule page shows 20-day structure
   - Notebook links work

### Step 5: Update Repository Homepage (Optional)

```bash
gh repo edit davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474 \
  --homepage "https://davi-moreira.github.io/2026Summer_predictive_analytics_purdue_MGMT474/"
```

Or via web interface: Repository Settings → General → Website → Enter URL

### Step 6: Add Repository Topics (Optional but Recommended)

```bash
gh repo edit davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474 \
  --add-topic predictive-analytics \
  --add-topic machine-learning \
  --add-topic python \
  --add-topic jupyter-notebook \
  --add-topic scikit-learn \
  --add-topic data-science \
  --add-topic mba-course \
  --add-topic colab
```

Or via web interface: Repository page → About (gear icon) → Topics

---

## Current Git Status

```
Branch: main
Commits: 6 new commits ready to push
Remote: Currently set to old repository (will be updated in Step 2)
```

### Recent Commits:
```
d3a993a docs: Add README and conversation log
6a6ddb9 build: Render Quarto site for GitHub Pages
af0761d chore: Update project configuration
d029be7 docs: Update Quarto website for 2026 Summer format
75d2130 feat: Add complete 20-day notebook sequence
c08c2e6 docs: Add course plan for 2026 Summer intensive
```

---

## Troubleshooting

### If GitHub Pages doesn't deploy:
1. Check that `docs/` directory exists and contains `index.html`
2. Verify GitHub Pages settings (Settings → Pages)
3. Check GitHub Actions logs: `gh run list --workflow pages-build-deployment`
4. Wait a few minutes - first deployment can take 5-10 minutes

### If Quarto needs re-rendering:
```bash
cd "/Users/dcordeir/Dropbox/academic/cursos/cursos-davi/predictive_analytics/2026Summer_predictive_analytics_purdue_MGMT474"
/Applications/RStudio.app/Contents/Resources/app/quarto/bin/quarto render
git add docs/
git commit -m "build: Re-render Quarto site"
git push
```

### If remote URL is incorrect:
```bash
git remote -v  # Check current remote
git remote set-url origin https://github.com/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474.git
git remote -v  # Verify updated remote
```

---

## Next Steps After Deployment

Once the site is live, consider:

1. **Test notebooks in Colab**
   - Click "Open in Colab" badges in each notebook
   - Verify they load correctly
   - Test "Run All" functionality

2. **Update Brightspace LMS**
   - Add course website link
   - Upload notebooks or link to GitHub
   - Set up quiz/assignment submission

3. **Create supplementary materials**
   - Record micro-videos (6 per day × 20 days = 120 videos)
   - Create auto-graded quizzes (20 quizzes)
   - Develop project rubrics (4 milestones)

4. **Share with colleagues**
   - The repository is public and ready to share
   - Course website provides all necessary context

---

## Files Created This Session

### Notebooks (20 files)
- notebooks/01_launchpad_eda_splits.ipynb through notebooks/20_final_submission_peer_review.ipynb
- All include Colab badges, PAUSE-AND-DO exercises, and consistent structure

### Documentation (4 files)
- README.md - Public repository documentation
- CONVERSATION_LOG.md - Development tracking
- claude_course_plan.md - Implementation plan
- MGMT47400_Online4Week_Plan_2026Summer.md - Master course plan
- GITHUB_SETUP_INSTRUCTIONS.md - This file

### Configuration (3 files)
- _quarto.yml - Updated for 2026 Summer
- .gitignore - Comprehensive exclusions
- 2026Summer_predictive_analytics_purdue_MGMT474.Rproj - RStudio project

### Website (3 files)
- index.qmd - Homepage (updated)
- schedule.qmd - 20-day schedule (complete rewrite)
- syllabus.qmd - Course syllabus (updated)

### Generated (docs/ directory)
- docs/ - Complete rendered website (115 files updated)
  - docs/index.html
  - docs/schedule.html
  - docs/syllabus.html
  - docs/notebooks/ (20 HTML versions)
  - docs/lecture_slides/ (existing lecture slides maintained)

---

## Contact

If you have questions or need assistance with any step:
- Check CONVERSATION_LOG.md for detailed context
- Review README.md for course structure
- Consult claude_course_plan.md for implementation details

Ready to complete the GitHub setup whenever you are!
