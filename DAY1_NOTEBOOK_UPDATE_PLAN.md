# Day 1 Notebook Update Plan: 01_launchpad_eda_splits.ipynb

**Date:** February 5, 2026
**Purpose:** Transform Day 1 notebook into comprehensive course launchpad covering welcome, predictive analytics fundamentals, and hands-on EDA/splitting workflow

---

## Executive Summary

This plan outlines the complete restructuring of `01_launchpad_eda_splits.ipynb` to serve as the **course foundation**. The updated notebook will:

1. Welcome students and introduce the instructor
2. Provide placeholder for course syllabus discussion
3. Establish Google Colab workflow and notebook conventions
4. Deliver comprehensive introduction to predictive analytics concepts
5. Demonstrate end-to-end predictive modeling workflow
6. Guide students through structured EDA and proper data splitting
7. Emphasize data leakage prevention throughout

**Time Budget:** 112.5 minutes
- Micro-videos: 6 videos × 9 min = 54 minutes
- Notebook work: 40 minutes
- PAUSE-AND-DO exercises: 2 × 10 min = 20 minutes (included in notebook time)
- Concept quiz: 8.5 minutes

---

## Source Materials Summary

### From `01_introduction.qmd` (Lecture Slides)

**Images to incorporate:**
- `images/davi_moreira_photo.JPG` - Instructor photo
- `lecture_slides/01_introduction/figs/2_1-1.png` - Sales vs TV, Radio, Newspaper (regression example)
- `lecture_slides/01_introduction/figs/2_1_2.png` - Ideal f(X) visualization
- `lecture_slides/01_introduction/figs/2_1_3.png` - Neighborhood averaging (KNN)
- `lecture_slides/01_introduction/figs/2_1_4_1.png` - Curse of dimensionality (1D)
- `lecture_slides/01_introduction/figs/2_1_4_2.png` - Curse of dimensionality (2D)
- `lecture_slides/01_introduction/figs/2_1_5.png` - Linear model fit
- `lecture_slides/01_introduction/figs/2_1_6.png` - Quadratic model fit
- `lecture_slides/01_introduction/figs/2_3-1.png` - Simulated income data (3D)
- `lecture_slides/01_introduction/figs/2_4-1.png` - Linear regression fit (3D)
- `lecture_slides/01_introduction/figs/2_5-1.png` - Flexible spline fit
- `lecture_slides/01_introduction/figs/2_6-1.png` - Overfitting example
- `lecture_slides/01_introduction/figs/2_7-1.png` - Flexibility vs Interpretability spectrum
- `lecture_slides/01_introduction/figs/2_9-1-1.png` and `2_9-1-2.png` - Bias-variance trade-off (top and bottom panels)
- `lecture_slides/01_introduction/figs/2_10-1.png` - Smooth truth example
- `lecture_slides/01_introduction/figs/2_11-1.png` - Wiggly truth example
- `lecture_slides/01_introduction/figs/2_12-1.png` - Bias-variance decomposition

**Key formulas:**
```latex
Y = f(X) + \epsilon
f(x) = E(Y|X=x)
E[(Y - \hat{f}(X))^2 | X = x] = [f(x) - \hat{f}(x)]^2 + \text{Var}(\varepsilon)
\text{MSE}_{Tr} = \text{Ave}_{i \in Tr}[(y_i - \hat{f}(x_i))^2]
\text{MSE}_{Te} = \text{Ave}_{i \in Te}[(y_i - \hat{f}(x_i))^2]
\mathbb{E}[(Y-\hat{f}(X))^2] = (\text{Bias}[\hat{f}(X)])^2 + \text{Var}[\hat{f}(X)] + \sigma^2
```

**Conceptual content:**
- Motivation examples: spam detection, zip code recognition, Netflix Prize
- Statistical learning framework: $Y = f(X) + \epsilon$
- Prediction vs. inference goals
- Supervised vs. unsupervised learning
- Parametric vs. non-parametric methods
- Curse of dimensionality
- Flexibility vs. interpretability trade-off
- Bias-variance trade-off decomposition
- Train vs. test error
- Overfitting and underfitting

### From `churn_exit_prediction_workflow_student.ipynb`

**End-to-end workflow structure:**
1. Setup (imports, seed, display settings)
2. Data loading and sanity checks
3. Exploratory Data Analysis (EDA)
4. Data preparation and feature engineering
5. Basic model (baseline)
6. More complex models
7. Model comparison with visualization
8. Model selection and rationale
9. Final training on full data
10. Predict on test set and export
11. Reporting and reproducibility

**Key workflow principles:**
- Set reproducible seed (SEED = 474 for competition and course material)
- Separate train/validation/test properly
- Use cross-validation within training set
- Don't touch test set until final evaluation
- Document every step with clear rationale
- Export requirements.txt for reproducibility

**Business case framing:**
- Customer churn prediction (bank exit)
- Clear target variable: `Exited` (binary: 0 = stayed, 1 = left)
- Business impact: retention campaigns, revenue protection
- Feature types: demographics, account info, behavior

### From Kaggle Data Leakage Tutorial
**URL:** https://www.kaggle.com/code/alexisbcook/data-leakage

**Key concepts to integrate:**
1. **Target leakage:** When predictors include data that won't be available at prediction time
2. **Train-test contamination:** When validation data influences training
3. **Examples:**
   - Pneumonia prediction using post-treatment data
   - Credit card fraud using transaction confirmation
   - House prices using post-sale renovation data

---

## Updated Notebook Structure

### Header (unchanged except title update)

```markdown
# Day 1: Launchpad - Welcome, Predictive Analytics Fundamentals, and Data Workflow

**MGMT 47400 - Predictive Analytics**
**4-Week Online Course**
**Day 1**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/01_launchpad_eda_splits.ipynb)
```

### Learning Objectives (updated)

```markdown
By the end of this notebook, you will be able to:

1. Understand the course structure, expectations, and Google Colab workflow
2. Explain what predictive analytics is and distinguish supervised from unsupervised learning
3. Describe an end-to-end predictive modeling workflow
4. Identify common data leakage patterns and why they invalidate models
5. Understand the bias-variance trade-off and curse of dimensionality
6. Perform structured EDA on a business dataset
7. Create proper train/validation/test splits with reproducible seeds
8. Frame a business problem with clear target variable identification
```

---

## Section-by-Section Plan

### Section 1: Welcome and Introductions

**Duration:** ~8 minutes (micro-video)
**Format:** Markdown cells with images

#### 1.1 Instructor Introduction

**Content:**
```markdown
## 1. Welcome and Introductions

### 1.1 Your Instructor: Professor Davi Moreira

<img src="https://raw.githubusercontent.com/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/main/images/davi_moreira_photo.JPG" width="200" align="left" style="margin-right: 20px; margin-bottom: 10px;">

**Professor Davi Moreira** is a Clinical Assistant Professor at Purdue University's Daniels School of Business. His research focuses on [brief description]. He teaches predictive analytics, machine learning, and data science courses at both undergraduate and graduate levels.

**Why this course matters:** In today's data-driven business environment, the ability to build, evaluate, and deploy predictive models is essential. Whether you're working in marketing, finance, operations, or strategy, predictive analytics helps you make better decisions, optimize processes, and create competitive advantages.

**What makes this course different:**
- **Hands-on from Day 1:** Every concept is immediately applied in Google Colab notebooks
- **Real business cases:** We use industry-relevant datasets and scenarios
- **AI-assisted learning:** You'll learn to use Google Gemini responsibly as a coding partner
- **End-to-end workflow:** From raw data to deployed model to business recommendation

<br clear="all">

---
```

**Why:** Establishing personal connection and setting expectations. Students need to know who is teaching them and why the course is designed this way.

#### 1.2 Student Engagement Activity

**Content:**
```markdown
### 1.2 Your Turn: Student Introductions

**PAUSE-AND-THINK (3 minutes):**

In the markdown cell below, introduce yourself by answering these questions:

1. **Name and background:** Where are you from? What program are you in?
2. **Data experience:** Have you worked with Python before? Any machine learning exposure?
3. **Goals:** What do you hope to learn or build by the end of this course?
4. **Predictive analytics in your field:** Can you think of one way predictive analytics could be applied in your area of interest? (e.g., predicting customer churn in retail, forecasting stock prices in finance, optimizing supply chains in operations)

**Why this matters:** Understanding your classmates' backgrounds helps build a collaborative learning environment. You'll find study partners, get different perspectives on business problems, and see how predictive analytics applies across industries.

---

### YOUR INTRODUCTION (edit this cell):

**Name and background:**
[Your answer here]

**Data experience:**
[Your answer here]

**Goals:**
[Your answer here]

**Predictive analytics application:**
[Your answer here]

---
```

**Why:** Active engagement from the start. Students articulate their goals and see diverse applications of predictive analytics.

---

### Section 2: Course Syllabus and Logistics

**Duration:** ~6 minutes (micro-video - instructor covers this verbally, students refer to website)
**Format:** Markdown placeholder

**Content:**
```markdown
## 2. Course Syllabus and Logistics

**Note:** Detailed syllabus, grading policies, and schedule are available on the course website:

📘 **Course Website:** [https://davi-moreira.github.io/2026Summer_predictive_analytics_purdue_MGMT474/](https://davi-moreira.github.io/2026Summer_predictive_analytics_purdue_MGMT474/)

**Key points covered in the welcome video:**
- **Format:** 4-week intensive, 20 business days, 112.5 min/day commitment
- **Structure:** Micro-videos (≤12 min) + Google Colab notebooks + exercises + daily quiz
- **Grading:** Daily quizzes (20%), Midterm (15%), Project milestones (55%), Peer review (10%)
- **Project:** Single capstone with 4 milestones (Proposal Day 5, Baseline Day 10, Improved Day 15, Final Day 20)
- **Technology:** Google Colab (primary), Google Gemini (AI assistant), Brightspace (LMS)
- **Textbooks:** ISLP (Introduction to Statistical Learning with Python) + supplementary readings

**Daily workflow:**
1. Watch micro-videos (~54 min total)
2. Work through notebook with PAUSE-AND-DO exercises (~40 min)
3. Complete daily auto-graded quiz (~8.5 min)
4. Post questions in discussion forum

---
```

**Why:** Sets clear expectations without overwhelming Day 1 with logistics. Detailed policies live on the website; this section just orients students.

---

### Section 3: Colab Setup + Course Notebook Conventions

**Duration:** ~9 minutes (micro-video with live demo)
**Format:** Markdown explanation + code cells for practice

#### 3.1 Google Colab Basics

**Content:**
```markdown
## 3. Google Colab Setup and Notebook Conventions

### 3.1 Why Google Colab?

**Google Colab** (Colaboratory) is a free, cloud-based Jupyter notebook environment that requires no local setup. It's perfect for this course because:

- ✓ **Zero installation:** No Python setup, no package management, no environment conflicts
- ✓ **Free GPU access:** Useful for deep learning later in the course
- ✓ **Cloud storage:** Work from anywhere, automatic saves to Google Drive
- ✓ **Built-in AI:** Google Gemini integration for coding assistance
- ✓ **Reproducible:** Share notebooks via links, ensure everyone has the same environment

**How to open this notebook in Colab:**
1. Click the "Open in Colab" badge at the top of this notebook
2. Or visit: [https://colab.research.google.com/](https://colab.research.google.com/)
3. Select "GitHub" tab and paste the course repo URL

---

### 3.2 Colab Navigation and Workflow

**Notebook structure:**
- **Markdown cells:** Text explanations (like this one). Double-click to edit, Shift+Enter to render.
- **Code cells:** Python code that you can run. Click the play button or Shift+Enter to execute.

**Essential keyboard shortcuts:**
- **Shift + Enter:** Run cell and move to next
- **Ctrl + Enter (Cmd + Enter on Mac):** Run cell and stay
- **Ctrl + M B (Cmd + M B):** Insert cell below
- **Ctrl + M A (Cmd + M A):** Insert cell above
- **Ctrl + M D (Cmd + M D):** Delete cell

**Menu options:**
- **Runtime → Run all:** Execute all cells in order (use this to test reproducibility)
- **Runtime → Restart runtime:** Clear all variables and outputs, start fresh
- **File → Save a copy in Drive:** Create your own editable version

**⚠️ Important:** Colab notebooks are **temporary** unless saved to your Drive. Always use "File → Save a copy in Drive" to preserve your work.

---

### 3.3 Course Notebook Conventions

Throughout this course, notebooks follow consistent patterns:

**1. Setup Cell (always first code cell):**
Every notebook starts with imports, random seed, and display settings.

**2. RANDOM_SEED = 42:**
We use seed 42 for all random operations to ensure reproducibility. This means every time you run the notebook, you'll get the same results.

**3. PAUSE-AND-DO Exercises:**
These are 10-minute guided practice sections where you'll apply concepts immediately. Look for this format:

> ## 📝 PAUSE-AND-DO Exercise N (10 minutes)
> **Task:** [What you need to do]
> **Instructions:** [Step-by-step guidance]

**4. Blockquotes for Critical Rules:**
Important warnings and best practices appear like this:

> ⚠️ **Critical Rule:** Always split data BEFORE preprocessing to avoid leakage.

**5. ✓ Checkmarks for Confirmations:**
When code completes successfully, you'll see: `✓ Setup complete!`

**6. Comments in Code:**
All code includes explanatory comments starting with `#` to help you understand each step.

**7. Google Gemini Integration:**
We'll use Gemini following the **"Ask → Verify → Document"** pattern (explained in Section 3.4).

---
```

#### 3.4 Google Gemini Workflow Rules

**Content:**
```markdown
### 3.4 Using Google Gemini Responsibly: The "Ask → Verify → Document" Pattern

**Google Gemini in Colab** is an AI coding assistant integrated directly into your notebooks. It can help you write code, debug errors, and understand concepts. However, **you are responsible for all code you submit**.

**The Three-Step Pattern:**

**1. ASK Gemini to draft code or explain concepts**

Example prompts:
- "Create a histogram of the Age variable colored by Exited status"
- "Explain what this error message means: KeyError: 'CustomerID'"
- "Write code to calculate the correlation between numeric features"

**2. VERIFY the code works and you understand it**
- Run the generated code in a cell
- Check the output matches what you expected
- Read through the code line by line
- Ask yourself: "Could I explain what each line does?"

**3. DOCUMENT with your own comments**
- Add `#` comments explaining the logic in your own words
- Annotate any parts that were unclear
- Note any modifications you made

**✓ What's allowed:**
- Using Gemini to generate boilerplate code (imports, data loading, standard plots)
- Asking Gemini to explain error messages or debugging
- Getting suggestions for visualizations or analyses
- Learning new pandas/numpy/sklearn functions

**⚠️ What's required:**
- You must understand **every line** of code you submit
- You must verify that generated code works correctly on your data
- You must add your own comments and documentation
- You must be able to explain your code and methodology

**❌ Accountability:**
- You are responsible for all code you submit, even if Gemini generated it
- If you can't explain your code in the project presentation, you don't understand it well enough
- Gemini can make mistakes - you need to catch them

**Example of proper use:**

```python
# PROMPT TO GEMINI: "Calculate the mean Age for customers who Exited vs stayed"

# Original Gemini output (verified and commented by me):
# Group by Exited status and calculate mean Age
mean_age_by_exit = df.groupby('Exited')['Age'].mean()

# Add my own descriptive print statement
print("Average age by exit status:")
print(f"  Stayed (0): {mean_age_by_exit[0]:.1f} years")
print(f"  Exited (1): {mean_age_by_exit[1]:.1f} years")

# My interpretation: Customers who exited are on average older,
# which might indicate different banking needs or service preferences.
```

**Why this matters:** AI tools like Gemini are powerful accelerators, but they don't replace understanding. In business, you need to explain your analyses to stakeholders, defend your methodology, and catch errors. Blindly trusting AI outputs is risky.

---
```

#### 3.5 Test Your Setup

**Content:**
```markdown
### 3.5 Let's Test Your Setup

Run the cell below to verify your Colab environment is ready. You should see:
- Python version
- Library versions
- A simple plot
- ✓ Setup confirmation

If you see any errors, use Gemini to help debug!
```

**Code cell:**
```python
# Day 1 Setup Test Cell
# This cell verifies your Colab environment is properly configured

# Import essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import sys

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.precision', 3)
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# Version check
print("=== ENVIRONMENT CHECK ===")
print(f"Python version: {sys.version.split()[0]}")
print(f"pandas: {pd.__version__}")
print(f"numpy: {np.__version__}")
print(f"matplotlib: {plt.matplotlib.__version__}")
print(f"seaborn: {sns.__version__}")
print(f"scikit-learn: {import sklearn; sklearn.__version__}")

# Create a simple test plot
x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.1, 100)

plt.figure(figsize=(8, 4))
plt.plot(x, y, 'o', alpha=0.5, label='Noisy sine wave')
plt.plot(x, np.sin(x), 'r-', linewidth=2, label='True function')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Test Plot: If you see this, your environment works!')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\n✓ Setup complete! Your Colab environment is ready.")
print(f"✓ Random seed set to {RANDOM_SEED} for reproducibility")
```

**Why:** Confirms technical setup works before diving into concepts. Reinforces RANDOM_SEED convention.

---

### Section 4: Introduction to Predictive Analytics

**Duration:** ~12 minutes (micro-video)
**Format:** Markdown with images and formulas

#### 4.1 Motivation: Real-World Examples

**Content:**
```markdown
## 4. Introduction to Predictive Analytics

### 4.1 What is Predictive Analytics? Motivation Examples

**Predictive analytics** is the practice of extracting patterns from historical data to make predictions about future or unknown outcomes. It's used everywhere in modern business and society:

**Example 1: Email Spam Detection**
- **Business problem:** Users receive hundreds of unwanted emails daily, wasting time and risking security
- **Predictive task:** Classify incoming emails as "spam" or "ham" (legitimate)
- **How it works:** Train a model on historical emails labeled as spam/ham; model learns patterns (keywords, sender characteristics, formatting) and predicts labels for new emails
- **Business impact:** Gmail blocks 99.9% of spam, saving billions of user hours annually

**Example 2: Handwritten Digit Recognition (Zip Codes)**
- **Business problem:** US Postal Service processes millions of handwritten envelopes daily; manual sorting is slow and expensive
- **Predictive task:** Read handwritten zip codes from envelope images (0-9 classification)
- **How it works:** Convolutional neural networks trained on thousands of handwritten digit examples
- **Business impact:** Automated mail sorting reduced delivery time and costs by 40%

**Example 3: Netflix Prize (Movie Recommendations)**
- **Business problem:** Netflix wants to recommend movies users will enjoy to increase engagement and retention
- **Predictive task:** Predict how a user would rate a movie they haven't watched yet
- **How it works:** Collaborative filtering using millions of past user ratings
- **Business impact:** $1M prize competition (2006-2009) improved recommendation accuracy by 10%, resulting in billions in retained subscriptions

**Common thread:** All three examples follow the same workflow:
1. Collect historical data with known outcomes
2. Extract relevant features
3. Train a predictive model
4. Deploy the model to make predictions on new data
5. Measure business impact

**This course:** You'll learn the entire workflow, from raw data to deployed model, with emphasis on doing it **correctly** (avoiding leakage, overfitting) and **responsibly** (fairness, transparency).

---
```

**Why:** Concrete examples make abstract concepts real. Students see the business value before diving into mathematics.

#### 4.2 Statistical Learning Framework

**Content:**
```markdown
### 4.2 The Statistical Learning Framework

At the core of predictive analytics is **statistical learning:** the set of tools for understanding and modeling relationships between variables.

**The fundamental equation:**

$$Y = f(X) + \epsilon$$

Where:
- **Y:** The outcome variable (target, response, dependent variable) we want to predict
  - Examples: spam/ham label, digit 0-9, movie rating, customer churn (Exited)
- **X:** The input variables (features, predictors, independent variables) we use to make predictions
  - Examples: email text, pixel intensities, user demographics and viewing history, account balance and tenure
- **f:** The unknown function relating X to Y (this is what we're trying to learn from data)
- **ε (epsilon):** Random error/noise that cannot be predicted (irreducible error)

**Visual intuition:**

<img src="https://raw.githubusercontent.com/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/main/lecture_slides/01_introduction/figs/2_1-1.png" width="700">

*Figure: Advertising data showing Sales (Y) vs. TV, Radio, Newspaper spending (X). Our goal is to estimate f(X) that best captures the relationship.*

---

### Ideal Function f(X)

The **regression function** is defined as:

$$f(x) = E(Y | X = x)$$

This is the **expected value** (average) of Y given X = x. It represents the systematic part of the relationship, removing the random noise ε.

<img src="https://raw.githubusercontent.com/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/main/lecture_slides/01_introduction/figs/2_1_2.png" width="500">

*Figure: The blue curve shows the true f(X), while red points show observed data Y = f(X) + ε with noise.*

**Why estimate f?**

**1. Prediction:** Make accurate forecasts for new observations
- Example: Predict which customers will churn next month so we can target them with retention offers

**2. Inference:** Understand which predictors affect the outcome and how
- Example: Does increasing credit limit reduce churn? By how much?

**This course focuses primarily on prediction,** though we'll discuss inference when relevant.

---

### How Do We Estimate f?

**Non-parametric example: K-Nearest Neighbors (KNN)**

One simple approach: for a new point x, find the K closest training points and average their Y values.

<img src="https://raw.githubusercontent.com/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/main/lecture_slides/01_introduction/figs/2_1_3.png" width="600">

*Figure: KNN prediction (red line) based on averaging neighboring points.*

**Pros:** Flexible, no assumptions about f's form
**Cons:** Requires lots of data, slow predictions, suffers from curse of dimensionality (next section)

---
```

**Why:** Establishes mathematical foundation without being overly technical. Students see the big picture before details.

#### 4.3 Supervised vs. Unsupervised Learning

**Content:**
```markdown
### 4.3 Supervised vs. Unsupervised Learning

**Machine learning** tasks fall into two broad categories:

**Supervised Learning** *(focus of this course)*

**Definition:** We have historical data where the outcome Y is **known** for all training observations. The goal is to learn f(X) so we can predict Y for new observations where it's unknown.

**Two types:**
1. **Regression:** Y is quantitative (continuous number)
   - Examples: predict house price, forecast sales, estimate customer lifetime value
   - Evaluation: mean squared error (MSE), R², mean absolute error (MAE)

2. **Classification:** Y is qualitative (discrete category)
   - Examples: spam/ham, churn/stay, approve/deny loan, diagnose disease
   - Evaluation: accuracy, precision, recall, ROC AUC

**Key point:** We train on **labeled data** (X and Y pairs) and deploy the model to predict Y for unlabeled data (just X).

---

**Unsupervised Learning** *(not covered in this course)*

**Definition:** We only have X (features), no outcome Y. The goal is to discover structure, patterns, or groupings in the data.

**Common tasks:**
- **Clustering:** Group similar observations together
  - Example: segment customers into personas based on demographics and behavior
- **Dimensionality reduction:** Compress high-dimensional data while preserving information
  - Example: visualize high-dimensional product reviews in 2D
- **Anomaly detection:** Find unusual observations
  - Example: detect fraudulent transactions

**Why unsupervised is harder:** No "correct answer" to check against, so evaluation is subjective.

**This course:** We focus exclusively on **supervised learning** (regression and classification) because these are the most common business applications of predictive analytics.

---
```

**Why:** Clarifies scope of the course. Students understand what we will and won't cover.

#### 4.4 End-to-End Predictive Modeling Workflow

**Content:**
```markdown
### 4.4 End-to-End Predictive Modeling Workflow

**Real-world predictive analytics projects** follow a structured workflow with 9 core steps. Let's preview the journey using a **customer churn prediction** example.

**Business Problem:** A bank notices customers are closing accounts. They want to predict which customers will churn next quarter so they can proactively offer retention incentives (fee waivers, better rates, personalized service).

**Target Variable (Y):** `Exited` (binary: 0 = stayed, 1 = closed account)
**Features (X):** Customer demographics (age, gender, geography), account info (balance, # products, credit score), and behavior (tenure, active member status)

---

**Step 0: Setup and Configuration**
- Import libraries (pandas, numpy, matplotlib, scikit-learn)
- Set random seed for reproducibility (`RANDOM_SEED = 42`)
- Configure display settings

**Step 1: Data Loading and Sanity Checks**
- Load train and test CSVs
- Validate data types and shapes
- Identify and exclude non-feature columns (IDs, names)
- Check target variable is binary and present in train, absent in test

**Step 2: Exploratory Data Analysis (EDA)**
- Examine target distribution (class balance)
- Analyze univariate distributions (histograms, boxplots)
- Explore bivariate relationships (features vs. target)
- Compute correlations and mutual information
- **Business insight:** Older customers, inactive members, and those with 3-4 products have higher churn rates

**Step 3: Data Preparation and Feature Engineering**
- Handle missing values (imputation)
- Encode categorical variables (one-hot encoding)
- Scale numeric features (standardization)
- Create new features if needed (interactions, ratios)
- Build preprocessing pipeline to ensure reproducibility

**Step 4: Baseline Model**
- Train simple logistic regression (interpretable, fast)
- Evaluate with cross-validation (ROC AUC, accuracy)
- Establish performance benchmark
- **Baseline AUC:** ~0.75

**Step 5: Advanced Models**
- Try more complex algorithms (LASSO, tree-based methods, ensembles)
- Tune hyperparameters
- Compare performance vs. baseline
- **Best model AUC:** ~0.82 (LASSO with feature selection)

**Step 6: Model Comparison and Visualization**
- Plot performance metrics with confidence intervals
- Create comparison table (AUC, error rates, sparsity)

**Step 7: Model Selection and Rationale**
- Choose best model based on performance, stability, and parsimony
- Document decision criteria transparently

**Step 8: Final Training**
- Retrain selected model on **full training data**
- Save model artifact (`.joblib` file)
- Record metadata (timestamp, features, hyperparameters)

**Step 9: Test Set Predictions and Submission**
- Load test data
- Generate predictions (probabilities)
- Export submission CSV with IDs and predictions
- Validate format and alignment

**Step 99: Reporting and Reproducibility**
- Write `requirements.txt` for package versions
- Document assumptions and decisions
- Create executive summary of findings

---

**Key principles throughout:**
- ✓ **Split first, preprocess second:** Avoid leakage by separating data before any transformations
- ✓ **Use cross-validation:** Don't trust a single train/test split; average performance across multiple folds
- ✓ **Lock the test set:** Don't peek at test data until final evaluation (one-time use only)
- ✓ **Document everything:** Future you (and your stakeholders) need to understand your choices
- ✓ **Reproducibility matters:** Set seeds, version packages, save pipelines

**In this notebook:** We'll practice Steps 0-2 (setup, EDA, splitting). Future notebooks will cover Steps 3-9.

---
```

**Why:** Shows the big picture before diving into details. Students see where EDA and splitting fit in the broader workflow, informed by the churn notebook structure.

#### 4.5 Data Leakage: The Silent Killer of Predictive Models

**Content:**
```markdown
### 4.5 Data Leakage: The #1 Mistake in Predictive Modeling

**Data leakage** occurs when information from outside the training dataset unintentionally "leaks" into the model, causing it to perform well during development but fail catastrophically in production.

**Why it's dangerous:**
- Your model will show excellent performance on validation data (high AUC, low error)
- You'll deploy the model thinking it works
- In production, performance will collapse because the leaked information isn't available for new predictions
- **Business impact:** Wasted resources, lost revenue, damaged credibility

**Two types of leakage:**

---

**Type 1: Target Leakage**

**Definition:** A predictor includes information that will not be available at the time you need to make a prediction.

**Example 1 - Pneumonia Prediction (from Kaggle tutorial):**
- **Task:** Predict which patients admitted to the hospital have pneumonia
- **Leaked feature:** `got_antibiotic` (whether patient received antibiotics)
- **Problem:** Antibiotics are prescribed **after** diagnosis. The feature encodes the target directly.
- **Result:** Model shows 99% accuracy in validation, but in production you don't know if a newly admitted patient will get antibiotics (that's what you're trying to predict!).

**Example 2 - Customer Churn:**
- **Task:** Predict which customers will close accounts next month
- **Leaked feature:** `num_service_calls_next_month` (how many times customer called support next month)
- **Problem:** This data doesn't exist yet when you make the prediction.
- **Result:** Model learns that customers with many calls next month churn, but you can't know future call volume.

**How to detect target leakage:**
- Ask: "Will this feature be available at prediction time?"
- Check for suspiciously high correlations between a feature and the target (r > 0.9)
- Think about temporal order: does the feature come before or after the outcome?

---

**Type 2: Train-Test Contamination**

**Definition:** Information from the validation or test set influences the training process, causing overly optimistic performance estimates.

**Example 1 - Preprocessing Leakage:**
```python
# WRONG: Standardize before splitting
X_scaled = StandardScaler().fit_transform(X)  # Uses ALL data including test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

# Problem: Mean and std from test set influenced training data scaling
```

**Correct approach:**
```python
# RIGHT: Split first, then fit scaler only on training data
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler().fit(X_train)  # Learn from train only
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Apply learned parameters
```

**Example 2 - Feature Selection Leakage:**
- **Wrong:** Compute feature correlations on full dataset, then split
- **Right:** Split first, then compute correlations only on training data

**Example 3 - Hyperparameter Tuning Leakage:**
- **Wrong:** Use test set to choose between models or tune hyperparameters
- **Right:** Use cross-validation on training set; lock test set until final evaluation

---

**The Golden Rule to Prevent Leakage:**

> **⚠️ Split first, preprocess second, model third**

**Workflow that prevents contamination:**
1. **Split** data into train/validation/test **before any analysis**
2. **EDA and preprocessing** should be fitted on training data only
3. **Modeling and evaluation** use cross-validation within training data
4. **Test set** is opened exactly once at the end for final performance estimate

**Additional leakage prevention checklist:**
- [ ] Remove ID columns and names (can encode target information)
- [ ] Check feature descriptions for future information
- [ ] Verify temporal alignment (features collected before outcome)
- [ ] Use pipelines to ensure preprocessing is part of cross-validation
- [ ] Never look at test set until final evaluation

**Real-world impact:** Kaggle competitions have been won and lost based on proper leakage prevention. In business, a leaky model can cost millions in bad decisions.

**Further reading:** [Kaggle Data Leakage Tutorial](https://www.kaggle.com/code/alexisbcook/data-leakage)

---
```

**Why:** Leakage is the most common and costly mistake. Students need to understand it deeply before modeling. Integrates Kaggle tutorial content.

#### 4.6 Assessing Model Accuracy

**Content:**
```markdown
### 4.6 Assessing Model Accuracy: Train vs. Test Error

**Question:** How do we know if our model f̂(X) is any good?

**Naive approach:** Compute prediction error on the training data:

$$\text{MSE}_{\text{train}} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{f}(x_i))^2$$

**Problem:** Training error is **optimistically biased**. As model complexity increases, training error always decreases (even if the model is overfitting).

<img src="https://raw.githubusercontent.com/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/main/lecture_slides/01_introduction/figs/2_6-1.png" width="500">

*Figure: A highly flexible spline fits every training point perfectly (MSE = 0), but this is **overfitting** - the model won't generalize to new data.*

---

**Better approach:** Use fresh **test data** that the model has never seen:

$$\text{MSE}_{\text{test}} = \frac{1}{m} \sum_{i=1}^{m} (y_i^{\text{test}} - \hat{f}(x_i^{\text{test}}))^2$$

**Key insight:** Test error measures **generalization** - how well the model performs on new data from the same population.

**What we want:** A model that minimizes **test error**, not training error.

**The fundamental trade-off:**
- Simple models (e.g., linear regression) may have high training error but generalize well (low test error)
- Complex models (e.g., deep neural networks) can have zero training error but poor test error (overfitting)

**Goal of this course:** Learn to build models that generalize well by:
1. Using proper train/validation/test splits
2. Employing cross-validation for robust evaluation
3. Understanding the bias-variance trade-off (next section)
4. Preventing overfitting through regularization

---
```

**Why:** Clarifies why we split data and use test sets. Sets up bias-variance discussion.

#### 4.7 Curse of Dimensionality

**Content:**
```markdown
### 4.7 The Curse of Dimensionality

**Question:** Why can't we just use more features to make better predictions?

**Intuition:** In high dimensions, data becomes sparse, and "similar" observations become rare. This makes pattern recognition exponentially harder.

**Visual demonstration:**

<img src="https://raw.githubusercontent.com/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/main/lecture_slides/01_introduction/figs/2_1_4_1.png" width="350">
<img src="https://raw.githubusercontent.com/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/main/lecture_slides/01_introduction/figs/2_1_4_2.png" width="350">

*Figure: To capture 10% of data points in 1D, we need a small interval. In 2D, we need a much larger circle. As dimensions increase, the "neighborhood" explodes.*

**The mathematical problem:**

Suppose we want to predict Y using K-Nearest Neighbors by averaging the 10% closest training points:

- **1 dimension (p=1):** If we have 100 points, we need to extend ±5 units to capture 10
- **2 dimensions (p=2):** Need a circle with radius ~18 to capture 10 points (area grows as r²)
- **10 dimensions (p=10):** Need a hypersphere with radius ~80 (volume grows as r¹⁰)

**Bottom panel in figures above:** As dimensionality increases, neighborhoods must expand dramatically to capture the same fraction of points. In 10 dimensions, even capturing 1% of data requires looking across 80% of the feature space!

**Practical implications:**

1. **Data sparsity:** In high dimensions, all points are far apart. "Nearest neighbors" aren't actually nearby.
2. **Overfitting:** Models have too much flexibility relative to available data, memorizing noise instead of learning patterns.
3. **Computational cost:** Algorithms that search neighborhoods (KNN, kernel methods) become prohibitively slow.

**Solutions:**

- **Feature selection:** Keep only the most predictive features
- **Dimensionality reduction:** PCA, autoencoders compress high-dimensional data
- **Regularization:** Penalize model complexity (LASSO, Ridge)
- **Parametric models:** Linear models assume structure, reducing effective dimensionality

**Rule of thumb:** You need exponentially more data as the number of features grows. With 10 features and 100 samples, you're in good shape. With 1000 features and 100 samples, you're in trouble.

**This course:** We'll learn feature selection techniques (forward stepwise, LASSO) to combat the curse of dimensionality.

---
```

**Why:** Explains why feature engineering and selection matter. Students understand why "more data" and "more features" aren't always better.

#### 4.8 Flexibility vs. Interpretability

**Content:**
```markdown
### 4.8 Flexibility vs. Interpretability Trade-off

**Question:** Should we use simple or complex models?

**Answer:** It depends on your goals and constraints.

**The spectrum:**

<img src="https://raw.githubusercontent.com/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/main/lecture_slides/01_introduction/figs/2_7-1.png" width="600">

*Figure: Different models trade off flexibility (ability to fit complex patterns) with interpretability (ability to explain predictions).*

---

**Left side: High Interpretability, Low Flexibility**

**Linear Models:**
$$\hat{Y} = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \ldots + \beta_p X_p$$

**Pros:**
- Coefficients β directly quantify feature importance and direction of effect
- Easy to explain to non-technical stakeholders
- Fast to train and predict
- Less prone to overfitting

**Cons:**
- Can't capture non-linear relationships or interactions
- May underfit if true relationship is complex

**When to use:** Regulatory environments (lending, healthcare), need to explain decisions, small datasets

---

**Middle: Moderate Flexibility and Interpretability**

**Tree-based Models (Decision Trees, Random Forests):**
- Pros: Handle non-linearities and interactions, feature importance scores, visual decision rules
- Cons: Can overfit (single trees), hard to interpret individual predictions (forests)

**Generalized Additive Models (GAMs):**
- Pros: Capture non-linearities while maintaining some interpretability
- Cons: Don't automatically capture interactions

**When to use:** Exploratory analysis, moderate-sized datasets, need balance of accuracy and explainability

---

**Right side: High Flexibility, Low Interpretability**

**Deep Neural Networks, Gradient Boosting (XGBoost, LightGBM):**

**Pros:**
- Can learn extremely complex patterns
- Often achieve best predictive accuracy
- Handle high-dimensional data

**Cons:**
- "Black box" - hard or impossible to explain individual predictions
- Require large datasets and careful tuning
- Slow to train
- Easy to overfit

**When to use:** Image/text/speech data, large datasets, prediction accuracy is paramount, don't need to explain decisions

---

**Practical guidance:**

**Start simple:** Always begin with logistic regression or a simple tree. This establishes a baseline and helps you understand the data.

**Add complexity only if needed:** If the simple model performs well enough for the business problem, stop there. Explainability is valuable.

**Use interpretability tools:** Even for complex models, SHAP values and partial dependence plots can provide some explanation.

**Consider the stakes:** High-risk decisions (medical diagnosis, loan approval) require more interpretability than low-risk ones (movie recommendations).

**This course:** We'll cover the full spectrum, but emphasize interpretable models (linear, regularized regression, trees) since these are most common in business analytics.

---
```

**Why:** Helps students choose appropriate models. Not all problems require deep learning.

#### 4.9 Bias-Variance Trade-off

**Content:**
```markdown
### 4.9 The Bias-Variance Trade-off

**The most important concept in predictive modeling:** Understanding why models fail and how to fix them.

**Test error decomposition:**

$$\mathbb{E}[(Y - \hat{f}(X))^2] = \underbrace{(\text{Bias}[\hat{f}(X)])^2}_{\text{underfitting}} + \underbrace{\text{Var}[\hat{f}(X)]}_{\text{overfitting}} + \underbrace{\sigma^2}_{\text{irreducible}}$$

Three components:

---

**1. Bias²: Error from Wrong Assumptions (Underfitting)**

**Definition:** The error introduced by approximating a complex real-world problem with a simplified model.

**High bias means:**
- Model is too rigid
- Misses important patterns (systematic error)
- **Underfits** the training data

**Example:** Using a straight line to fit a curved relationship

<img src="https://raw.githubusercontent.com/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/main/lecture_slides/01_introduction/figs/2_1_5.png" width="400">

*Figure: Linear model (high bias) misses the curvature in the data.*

**Typical causes:**
- Overly simple model (linear when should be non-linear)
- Too few features
- Heavy regularization

**How to detect:** Both training and test error are high

**How to fix:** Add features, increase model complexity, reduce regularization

---

**2. Variance: Error from Sensitivity to Training Data (Overfitting)**

**Definition:** The amount by which the model's predictions would change if we trained on a different sample from the same population.

**High variance means:**
- Model is too flexible
- Overly sensitive to noise in training data
- **Overfits** by memorizing training examples

**Example:** High-degree polynomial fits training data perfectly but wiggles wildly

<img src="https://raw.githubusercontent.com/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/main/lecture_slides/01_introduction/figs/2_6-1.png" width="400">

*Figure: Very flexible spline (high variance) fits every training point but doesn't generalize.*

**Typical causes:**
- Overly complex model
- Too many features relative to sample size
- No regularization

**How to detect:** Training error is low, but test error is high (large gap)

**How to fix:** Simplify model, remove features, add regularization, get more training data

---

**3. Irreducible Error σ²: Random Noise**

**Definition:** Variability in Y that cannot be explained by X, no matter how good our model is.

**Sources:**
- Measurement error
- Omitted variables (factors affecting Y that aren't in X)
- True randomness in the process

**Key point:** This sets a lower bound on test error. No amount of modeling can reduce it.

---

**Visualizing the Trade-off:**

<img src="https://raw.githubusercontent.com/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/main/lecture_slides/01_introduction/figs/2_9-1-1.png" width="400">
<img src="https://raw.githubusercontent.com/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/main/lecture_slides/01_introduction/figs/2_9-1-2.png" width="400">

**Top panel:** Three fitted models with different flexibility
- **Orange (low flexibility):** High bias (misses pattern), low variance (stable)
- **Blue (medium flexibility):** Balanced bias and variance (best)
- **Green (high flexibility):** Low bias (captures pattern), high variance (overfits noise)

**Bottom panel:** Error curves vs. model flexibility
- **Gray (training error):** Monotonically decreases as flexibility increases
- **Red (test error):** U-shaped - decreases initially, then increases due to overfitting
- **Sweet spot:** Minimum test error (blue vertical line) balances bias and variance

---

**The Trade-off:**

As model flexibility increases:
- **Bias decreases** (better fit to training data)
- **Variance increases** (more sensitive to training sample)

**Goal:** Choose complexity that **minimizes expected test error** by balancing bias and variance.

---

**Different scenarios:**

<img src="https://raw.githubusercontent.com/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/main/lecture_slides/01_introduction/figs/2_10-1.png" width="400">
<img src="https://raw.githubusercontent.com/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/main/lecture_slides/01_introduction/figs/2_11-1.png" width="400">

**Left:** True relationship is smooth → simpler models perform well
**Right:** True relationship is wiggly → more flexible models needed

**Lesson:** Optimal model complexity depends on the problem. There's no universal "best" model.

---

**Decomposition across complexities:**

<img src="https://raw.githubusercontent.com/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/main/lecture_slides/01_introduction/figs/2_12-1.png" width="700">

*Figure: Test MSE (red) = Squared Bias (blue) + Variance (orange) + Irreducible Error (dotted). The sweet spot (vertical line) minimizes test error.*

---

**Practical implications:**

**Tools to manage the trade-off:**
1. **Cross-validation:** Estimate test error without touching test set
2. **Regularization:** Constrain model complexity (LASSO, Ridge, Elastic Net)
3. **Ensemble methods:** Reduce variance by averaging many models (Random Forest, Bagging)
4. **Early stopping:** Stop training before overfitting (neural networks)
5. **Feature selection:** Remove irrelevant features that add variance without reducing bias

**This course:** Every modeling decision we make (train/val/test splits, cross-validation, hyperparameter tuning, regularization) is about managing the bias-variance trade-off.

**Key takeaway:** Understanding this trade-off is the difference between building models that work in notebooks vs. models that work in production.

---
```

**Why:** This is the theoretical foundation for all model selection and evaluation. Students need to deeply understand this before Day 2 (preprocessing) and Day 3 (model evaluation).

---

### Section 5: Hands-On Practice - Loading and Exploring Data

**Duration:** ~9 minutes (micro-video demo + student practice)
**Format:** Code cells with markdown guidance

#### 5.1 Load Dataset

**Content:**
```markdown
## 5. Hands-On Practice: Customer Churn EDA and Splitting

Now that we understand the theory, let's apply it to a **real business case**.

### 5.1 Business Case: Bank Customer Churn Prediction

**Business Context:**

You are a data analyst at a European bank. The bank has noticed an increasing number of customers closing their accounts and moving to competitors. Customer acquisition costs are high (€500-€1000 per customer), so retaining existing customers is far more profitable than acquiring new ones.

The marketing team wants to launch a **proactive retention campaign**, offering targeted incentives (fee waivers, higher interest rates, personalized service) to at-risk customers **before** they churn.

**Your task:** Build a predictive model that identifies customers likely to exit in the next quarter.

**Target Variable:** `Exited` (binary)
- **0 = Stayed:** Customer retained account
- **1 = Exited:** Customer closed account

**Available Features:**
- **Demographics:** Age, Gender, Geography (France/Spain/Germany)
- **Account Information:** Credit Score, Balance, Number of Products, Has Credit Card, Is Active Member
- **Tenure:** How many years the customer has been with the bank
- **Financial:** Estimated Salary

**Success Metrics:**
- **Primary:** ROC AUC (ability to rank customers by churn risk)
- **Secondary:** Precision at top 20% (of those we target, how many actually churn?)

**Business Constraint:** We can only afford to target the top 20% highest-risk customers with retention offers. Model must prioritize **precision** among high-risk predictions.

---

### Load the Data

We'll use a simulated bank churn dataset (based on real patterns but anonymized).

**Two options:**

**Option 1:** Load from URL (recommended for Colab)
```python
url = 'https://raw.githubusercontent.com/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/main/data/bank_churn.csv'
df = pd.read_csv(url)
```

**Option 2:** Upload CSV file
```python
from google.colab import files
uploaded = files.upload()
df = pd.read_csv('bank_churn.csv')
```

Let's load it:
```

**Code cell:**
```python
# Load bank churn dataset
# This dataset contains customer information and whether they exited the bank

import pandas as pd
import numpy as np

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Load data from URL
url = 'https://raw.githubusercontent.com/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/main/data/bank_churn.csv'
df = pd.read_csv(url)

# Display basic information
print("=== DATASET OVERVIEW ===")
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nColumn names:\n{df.columns.tolist()}")
print(f"\nFirst 5 rows:")
df.head()
```

**Expected output structure:**
```
Shape: 10000 rows × 14 columns
Columns: ['RowNumber', 'CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited']
```

**Why:** Introduces real business context. Students see how theory connects to practice. Establishes target variable clearly.

---

#### 5.2 Identify Non-Feature Columns

**Content:**
```markdown
### 5.2 Identify and Remove Non-Feature Columns

**Critical step:** Before EDA, we must identify columns that should **not** be used as features.

**Why?** Some columns can cause **target leakage** or are simply identifiers with no predictive value.

**Columns to remove:**
1. **`RowNumber`:** Index column, no business meaning
2. **`CustomerId`:** Unique identifier, no predictive power
3. **`Surname`:** Customer's last name (privacy issue, potential data leakage if surnames correlate with geography/ethnicity)

**Checking for leakage:**
- `RowNumber`, `CustomerId`: Pure identifiers, safe to drop
- `Surname`: Could encode protected attributes (ethnicity), also not available at prediction time for new customers

Let's remove these and keep only modeling features:
```

**Code cell:**
```python
# Remove non-feature columns
# These are identifiers or variables that should not be used for prediction

# Define columns to drop
cols_to_drop = ['RowNumber', 'CustomerId', 'Surname']

# Create clean feature DataFrame
df_clean = df.drop(columns=cols_to_drop)

print("=== CLEANED DATASET ===")
print(f"Shape after removing non-features: {df_clean.shape}")
print(f"\nRemaining columns:\n{df_clean.columns.tolist()}")

# Verify target variable
print(f"\n=== TARGET VARIABLE CHECK ===")
print(f"Target: 'Exited'")
print(f"Values: {sorted(df_clean['Exited'].unique())}")
print(f"Type: {'Binary (Classification)' if df_clean['Exited'].nunique() == 2 else 'Multi-class'}")

# Check class balance
print(f"\n=== CLASS DISTRIBUTION ===")
print(df_clean['Exited'].value_counts())
print(f"\nProportions:")
print(df_clean['Exited'].value_counts(normalize=True))

# Calculate churn rate
churn_rate = df_clean['Exited'].mean()
print(f"\n**Churn Rate:** {churn_rate:.1%}")
print("Interpretation: About 1 in 5 customers closed their account.")
```

**Expected output:**
```
Shape after removing non-features: (10000, 11)
Remaining columns: ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited']

Target: 'Exited'
Values: [0, 1]
Type: Binary (Classification)

CLASS DISTRIBUTION:
0    7963
1    2037

Proportions:
0    0.7963
1    0.2037

Churn Rate: 20.4%
```

**Why:** Establishes clean dataset and highlights class imbalance (important for model evaluation later).

---

### Sections 6-9: Complete EDA and Splitting Workflow

These sections will follow the existing notebook structure but with the churn dataset and business framing:

**Section 6: EDA Checklist (Data Types, Missingness, Target Distribution)**
- Audit data types (numeric vs. categorical)
- Check for missing values
- Visualize target distribution
- Compute descriptive statistics

**Section 7: Feature Distributions and Relationships**
- Univariate analysis (histograms for numeric, bar charts for categorical)
- Bivariate analysis (features vs. `Exited`)
- Correlation matrix
- Feature importance via mutual information

**Section 8: Train/Validation/Test Splits**
- 60/20/20 split with `stratify=y` to maintain class balance
- Verify split sizes
- Check target distribution across splits
- Sanity checks (no data overlap)

**Section 9: Leakage Sniff Test**
- Review feature names for suspicious variables
- Check for perfect correlations
- Verify no future information in features
- Document leakage prevention checklist

**Section 10: PAUSE-AND-DO Exercise 1 (EDA Findings)**
- Student completes EDA checklist
- Identifies 3 key patterns in the data
- Writes interpretations in business terms

**Section 11: PAUSE-AND-DO Exercise 2 (Leakage Analysis)**
- Student reviews features for potential leakage
- Writes 3 specific risks for this dataset
- Proposes mitigation strategies

**Section 12: Wrap-Up and Next Steps**
- Recap of Day 1 concepts
- Preview of Day 2 (preprocessing pipelines)
- Critical rules to remember

---

## Implementation Notes

### Images

**Storage location:** All images from `lecture_slides/01_introduction/figs/` should be pushed to GitHub so they can be referenced via raw.githubusercontent.com URLs:

```markdown
![](https://raw.githubusercontent.com/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/main/lecture_slides/01_introduction/figs/2_1-1.png)
```

Alternatively, use relative paths if images are in the same repo:
```markdown
![](../lecture_slides/01_introduction/figs/2_1-1.png)
```

### Formulas

Use LaTeX in markdown cells:
```markdown
$$Y = f(X) + \epsilon$$
```

Inline math: `$\hat{f}(x)$`

### Code Style

**Conventions from existing notebooks:**
- Extensive comments explaining "why" and "what"
- Print statements with clear headers (`=== SECTION NAME ===`)
- Checkmarks for confirmations (`✓ Setup complete!`)
- Docstrings for any functions
- Follow PEP 8 style

### Data File

**Required:** Create or find a suitable bank churn CSV with these columns:
- `RowNumber`, `CustomerId`, `Surname` (to be dropped)
- `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary` (features)
- `Exited` (target)

**Sample size:** ~10,000 rows (manageable for notebooks, large enough to show patterns)

**Source options:**
1. Use existing Kaggle Churn dataset
2. Generate synthetic data with realistic correlations
3. Use churn_exit_prediction_workflow_student.ipynb data (if compatible)

---

## Completion Checklist

- [ ] Section 1: Welcome (instructor + student intros) - 2 markdown cells
- [ ] Section 2: Syllabus placeholder - 1 markdown cell
- [ ] Section 3: Colab setup (3.1-3.5) - 5 markdown + 1 code cell
- [ ] Section 4: Predictive Analytics Intro (4.1-4.9) - 9 markdown cells with images/formulas
- [ ] Section 5: Load data + business case - 2 markdown + 2 code cells
- [ ] Section 6: EDA checklist - adapt existing cells with churn data
- [ ] Section 7: Feature exploration - adapt existing cells
- [ ] Section 8: Splitting - adapt existing cells
- [ ] Section 9: Leakage sniff test - enhance existing cells
- [ ] Section 10: PAUSE-AND-DO 1 (EDA findings) - existing
- [ ] Section 11: PAUSE-AND-DO 2 (leakage analysis) - existing
- [ ] Section 12: Wrap-up - update with new content references
- [ ] All images uploaded to GitHub and links working
- [ ] All code cells tested and producing correct output
- [ ] Notebook runs end-to-end without errors (Run All)
- [ ] Learning objectives aligned with content
- [ ] Time budget validated (~112.5 min total)
- [ ] Colab badge link working
- [ ] Bibliography updated with all references

---

## Bibliography Updates

Add to existing references:

**New citations:**
- Kaggle. (2023). *Data Leakage Tutorial*. https://www.kaggle.com/code/alexisbcook/data-leakage
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning with Python* (ISLP), Chapter 2. Springer.
- Google Colab Documentation: https://colab.research.google.com/
- scikit-learn User Guide: https://scikit-learn.org/stable/user_guide.html

**Retain existing:**
- James et al. ISLP
- sklearn documentation
- Kaggle Learn: Data Leakage
- Previous course materials

---

## Risk Mitigation

**Risk 1: Notebook becomes too long**
- **Mitigation:** Split theoretical content (Section 4) into collapsible markdown sections
- Use "accordion" style with `<details>` tags if needed
- Keep code cells focused and modular

**Risk 2: Images don't render in Colab**
- **Mitigation:** Test all raw.githubusercontent.com URLs in incognito browser
- Have local copies as backup
- Use `<img>` tags with explicit width for control

**Risk 3: Churn dataset not available**
- **Mitigation:** Generate synthetic dataset with numpy
- Use sklearn make_classification with specified correlations
- Document data generation process in separate notebook

**Risk 4: Time budget exceeds 112.5 minutes**
- **Mitigation:** Mark optional content (curse of dimensionality details, extra bias-variance examples)
- Provide "fast path" vs. "deep dive" options
- Test with actual students if possible

---

## Next Steps (After Plan Approval)

1. **Prepare data file:** Locate or generate bank_churn.csv
2. **Upload images:** Push all figs/ images to GitHub main branch
3. **Implement notebook:** Create updated 01_launchpad_eda_splits.ipynb following this plan
4. **Test thoroughly:**
   - Run all cells in fresh Colab session
   - Verify images render
   - Check formulas display correctly
   - Time each section
5. **Review outputs:** Ensure all print statements are clear and outputs match expectations
6. **Commit to git:** Use atomic commit with clear message
7. **Update schedule.qmd:** Verify Day 1 description matches new content
8. **Create micro-video script:** Draft speaking notes for instructor videos

---

**End of Plan**
