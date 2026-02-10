# Tutorial: Editing Jupyter Notebooks and Pushing to GitHub with VS Code

**Welcome!** I'll guide you through everything step-by-step. By the end, you'll know how to edit your notebook and push changes to GitHub using VS Code.

---

## 📚 Part 1: Setting Up VS Code (One-Time Setup)

### Step 1.1: Install Required VS Code Extensions

**Why?** VS Code needs extensions to work with Jupyter notebooks and Git effectively.

1. **Open VS Code**
2. **Click the Extensions icon** in the left sidebar (or press `Cmd+Shift+X` on Mac)
3. **Install these extensions** (search for each and click "Install"):
   - **Python** (by Microsoft) - for Python support
   - **Jupyter** (by Microsoft) - for editing .ipynb files
   - **GitLens** (optional but helpful) - for enhanced Git features

**How to verify:** You should see checkmarks next to installed extensions.

---

### Step 1.2: Configure Git (One-Time Setup)

**Why?** Git needs to know who you are for commit attribution.

1. **Open VS Code Terminal:**
   - Menu: `Terminal` → `New Terminal` (or press `` Ctrl+` ``)
   - You'll see a terminal panel at the bottom

2. **Check if Git is configured:**
   ```bash
   git config --global user.name
   git config --global user.email
   ```

3. **If empty, configure Git:**
   ```bash
   git config --global user.name "Davi Moreira"
   git config --global user.email "your-email@purdue.edu"
   ```

**How to verify:** Run the check commands again - you should see your name and email.

---

### Step 1.3: Authenticate with GitHub

**Why?** You need permission to push changes to your repository.

1. **In the terminal, check authentication:**
   ```bash
   gh auth status
   ```

2. **If not authenticated:**
   ```bash
   gh auth login
   ```
   - Follow the prompts:
     - Choose: **GitHub.com**
     - Choose: **HTTPS**
     - Choose: **Login with a web browser**
     - Copy the code shown, press Enter, paste in browser, authorize

**How to verify:** Run `gh auth status` again - should show "Logged in to github.com".

---

## 📝 Part 2: Opening and Editing the Notebook

### Step 2.1: Open Your Project Folder

1. **Menu: `File` → `Open Folder...`**
2. **Navigate to:**
   ```
   /Users/dcordeir/Dropbox/academic/cursos/cursos-davi/predictive_analytics/2026Summer_predictive_analytics_purdue_MGMT474
   ```
3. **Click "Open"**

**What you'll see:** The left sidebar (Explorer) shows all your project files.

---

### Step 2.2: Open the Notebook

1. **In the Explorer (left sidebar), navigate to:**
   ```
   notebooks/01_launchpad_eda_splits.ipynb
   ```

2. **Click on the file** - it opens in the main editor area

**What you'll see:**
- The notebook renders as cells (markdown and code)
- Each cell has a small toolbar when you hover over it
- You should see "Select Kernel" at the top right

---

### Step 2.3: Select a Python Kernel (First Time Only)

**Why?** The kernel runs Python code in code cells.

1. **Click "Select Kernel"** (top right of notebook)
2. **Choose:** `Python Environments...`
3. **Select:** Any Python 3.x environment (system Python or conda)

**How to verify:** Top right should show "Python 3.x.x" instead of "Select Kernel".

---

### Step 2.4: Edit a Cell

Let's practice editing a markdown cell.

**Example: Update the instructor section**

1. **Find Cell 1** (the "Welcome and Introductions" section with instructor photo)

2. **Click inside the cell** - you'll see it's in "preview mode" (rendered)

3. **Double-click the cell** - it switches to "edit mode" (raw markdown)

4. **Make a small change** - for example, add a line:
   ```markdown
   **Office Hours:** Tuesdays 2-4 PM via Zoom
   ```

5. **Exit edit mode:**
   - **Click the checkmark** ✓ in the cell toolbar, OR
   - **Press `Shift+Enter`** (runs the cell and moves to next)

**What you'll see:** The cell re-renders with your change.

---

### Step 2.5: Edit a Code Cell

Let's practice editing a code cell.

**Example: Change a comment**

1. **Find Cell 10** (the first code cell with imports)

2. **Click inside the code cell** - cursor appears

3. **Edit the first comment line:**
   ```python
   # Day 1 Setup Test Cell - Updated by Professor Moreira
   ```

4. **Run the cell to verify it works:**
   - **Click the play button** ▶ to the left of the cell, OR
   - **Press `Shift+Enter`**

**What you'll see:** The code runs, output appears below the cell.

---

### Step 2.6: Save Your Changes

**Important:** Always save before committing!

1. **Menu: `File` → `Save`** OR **Press `Cmd+S`**

2. **Look for the indicator:**
   - Unsaved files have a **white dot** next to the filename in the tab
   - Saved files have **no dot**

**How to verify:** The dot disappears from the tab.

---

## 🔄 Part 3: Committing and Pushing to GitHub

### Step 3.1: Open the Source Control Panel

1. **Click the Source Control icon** in the left sidebar (looks like a branching diagram) OR **Press `Ctrl+Shift+G`**

**What you'll see:**
- "Changes" section showing modified files
- You should see `01_launchpad_eda_splits.ipynb` listed

---

### Step 3.2: Review Your Changes

**Why?** Always review before committing to catch mistakes.

1. **Click on `01_launchpad_eda_splits.ipynb`** in the Changes list

**What you'll see:**
- A **diff view** (side-by-side comparison)
- **Left side:** Old version (red = removed)
- **Right side:** New version (green = added)
- Scroll through to see what changed

**Best practice:** Make sure the changes are what you intended!

---

### Step 3.3: Stage the File

**Why?** Git has a two-step process: stage (prepare) → commit (save).

1. **Hover over `01_launchpad_eda_splits.ipynb`** in the Changes list

2. **Click the "+" button** that appears next to the filename

**What you'll see:**
- The file moves from "Changes" to "Staged Changes"
- This means "ready to commit"

**Alternative:** You can click the "+" button next to "Changes" to stage ALL files at once.

---

### Step 3.4: Write a Commit Message

**Why?** Commit messages document what you changed and why.

1. **Find the message box** at the top of the Source Control panel (says "Message")

2. **Type a descriptive commit message:**
   ```
   docs: Update Day 1 notebook with office hours

   - Added office hours to instructor section
   - Updated setup cell comment

   Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
   ```

**Commit message format:**
- **Line 1:** Short summary (< 72 characters)
  - Use prefixes: `docs:` (documentation), `fix:` (bug fix), `feat:` (new feature)
- **Line 2:** Blank line
- **Lines 3+:** Detailed description (bullet points work well)
- **Last line:** Co-author credit (if applicable)

---

### Step 3.5: Commit Your Changes

1. **Click the "✓ Commit" button** above the message box

2. **If prompted** "Would you like to stage all changes and commit them directly?"
   - **Choose "No"** (since we already staged what we want)

**What you'll see:**
- The file disappears from Staged Changes
- The commit is saved locally (not yet on GitHub)

---

### Step 3.6: Push to GitHub

**Why?** Commits are saved locally; pushing sends them to GitHub (the remote server).

1. **Click the "..." menu** (three dots) in the Source Control panel

2. **Select: `Push`** OR **Click the cloud upload icon** in the bottom status bar

**What you'll see:**
- Status bar shows upload progress
- When done, it shows "Successfully pushed"

**How to verify:** Visit your GitHub repository in a browser and check the latest commit.

---

## 🎯 Part 4: Updating the Course Website (Bonus)

**Why?** Your course website uses Quarto, which needs to be re-rendered after notebook changes.

### Step 4.1: Render the Quarto Site

1. **In VS Code Terminal:**
   ```bash
   /Applications/RStudio.app/Contents/Resources/app/quarto/bin/quarto render
   ```

2. **Wait for rendering** (takes 1-2 minutes)

**What you'll see:**
- Progress messages as each page renders
- "Output created: docs/index.html" at the end

---

### Step 4.2: Commit the Rendered Site

1. **Go back to Source Control panel**

2. **You'll see `docs/` folder changed**

3. **Stage the docs/ folder:**
   - Hover over the changed files in `docs/`
   - Click "+" to stage them

4. **Commit message:**
   ```
   build: Render Quarto site with updated Day 1 notebook

   Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
   ```

5. **Click "✓ Commit"**

6. **Push to GitHub** (same as Step 3.6)

---

## 📋 Quick Reference: The Complete Workflow

Once you're comfortable, here's the streamlined process:

### **Editing:**
1. `File` → `Open Folder` → Select project
2. Open `notebooks/01_launchpad_eda_splits.ipynb`
3. Double-click cell to edit
4. `Shift+Enter` to run/save cell
5. `Cmd+S` to save file

### **Committing:**
1. `Ctrl+Shift+G` → Source Control
2. Review changes (click on file)
3. Click "+" to stage
4. Write commit message
5. Click "✓ Commit"

### **Pushing:**
1. Click "..." menu → `Push`
2. Or click sync icon in status bar

### **Rendering Website:**
1. Terminal: `/Applications/RStudio.app/Contents/Resources/app/quarto/bin/quarto render`
2. Stage `docs/` changes
3. Commit with "build:" prefix
4. Push to GitHub

---

## ⚠️ Common Issues and Solutions

### **Issue 1: "No kernel selected"**
- **Solution:** Click "Select Kernel" → Choose a Python environment

### **Issue 2: "Changes not showing in Source Control"**
- **Solution:** Make sure you saved the file (`Cmd+S`)

### **Issue 3: "Push rejected / Permission denied"**
- **Solution:** Run `gh auth login` in terminal to re-authenticate

### **Issue 4: "Merge conflict"**
- **Solution:** Someone else modified the same file. Click "Accept Current Change" or "Accept Incoming Change" in the diff view

### **Issue 5: "Quarto command not found"**
- **Solution:** Use the full path: `/Applications/RStudio.app/Contents/Resources/app/quarto/bin/quarto render`

---

## 🎓 Learning Tips

1. **Practice makes perfect:** Try making small edits and committing them frequently
2. **Read commit messages:** Look at your repo's history on GitHub to see good examples
3. **Use descriptive messages:** Future you will thank present you!
4. **Commit often:** Small, focused commits are easier to review and revert if needed
5. **Always review diffs:** Catch mistakes before they go to GitHub

---

## 📚 Next Steps

Now that you know the basics, try:
- [ ] Edit a different cell in the notebook
- [ ] Add a new markdown cell (click "+" in notebook toolbar)
- [ ] Make a commit with your own message
- [ ] View your commit history on GitHub

**Questions?** The VS Code interface is intuitive - hover over buttons to see what they do!

---

**Congratulations!** 🎉 You now know how to edit notebooks and push to GitHub using VS Code. This is a fundamental skill for managing your course materials.
