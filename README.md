# QM47400: Predictive Analytics

[![Course Website](https://img.shields.io/badge/Website-Course%20Page-blue)](https://davi-moreira.github.io/2026Summer_predictive_analytics_purdue_MGMT474/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/)

## Course Information

**Instructor:** [Professor Davi Moreira](https://davi-moreira.github.io)
**Email:** dmoreira@purdue.edu
**Office:** Young Hall 1007
**Institution:** Mitch Daniels School of Business, Purdue University

**Course Format:** 4-Week Fully Online Intensive
**Course Dates:** May 18 - June 12, 2026 (20 business days, Mon-Fri)
**Daily Time Commitment:** 112.5 minutes per business day
**Credits:** 3 credit hours

## Course Description

This 4-week fully online intensive course enables students to navigate the entire predictive analytics pipeline skillfully—from data preparation and exploration to modeling, assessment, and interpretation. Running over **20 business days** with **112.5 minutes of daily engagement**, the course combines short micro-videos (≤12 minutes each), hands-on Google Colab notebooks, exercises, quizzes, and progressive project work. **Every Friday (Days 5, 10, 15, 20)**, the class meets for a **synchronous 112.5-minute lecture** (recorded) covering weekly topics, project milestones, the Kaggle case competition, and student Q&A.

Students engage with real-world examples through interactive Jupyter notebooks, all designed to run in **Google Colab with Google Gemini AI assistance**. The course emphasizes essential programming and analytical skills through a "vibe coding" approach: **draft → verify → document**. Topics include exploratory data analysis, train/validation/test splits, linear and logistic regression, classification metrics, resampling methods, regularization techniques, tree-based approaches, gradient boosting, model interpretation, fairness considerations, and deployment thinking.

The course centers on **one comprehensive capstone project** completed in **groups of four randomly assigned members**, progressing through four weekly milestones and culminating in a **final research poster**, plus a **Kaggle case competition** (individual or pairs) predicting bank customer churn. Project milestones: initial proposal (Day 5), simple model + performance evaluation (Day 10), more complex model + draft abstract (Day 15), and final research poster + intra-group peer evaluation (Day 20). Optional presentation at the Fall 2026 Purdue Undergraduate Research Conference is strongly encouraged but not required. Single source of truth: [`_final_project/2026Summer/final_project_milestone_reference.md`](_final_project/2026Summer/final_project_milestone_reference.md).

## Course Structure

### Week 1 (Days 1-5): Foundations, EDA, Splits, Linear Regression, Regularization
- **Project Milestone:** Proposal + dataset selection (Day 5)
- Topics: Colab workflow, EDA, preprocessing pipelines, regression metrics, linear regression, Ridge/Lasso regularization

### Week 2 (Days 6-10): Classification, Metrics, Resampling, Comparison + Midterm
- **Project Milestone:** Baseline model + evaluation plan (Day 10)
- **Midterm:** Business-case strategy practicum (Day 10)
- Topics: Logistic regression, classification metrics, ROC/PR curves, cross-validation, model comparison, hyperparameter tuning

### Week 3 (Days 11-15): Trees, Ensembles, Tuning, Interpretation
- **Project Milestone:** Improved model + interpretation (Day 15)
- Topics: Decision trees, random forests, gradient boosting, model selection protocols, feature importance, partial dependence plots, error analysis

### Week 4 (Days 16-20): Error Analysis, Fairness/Ethics, Deployment, Executive Narrative, Final Project
- **Project Milestone:** Final deliverable (notebook + deck + video) (Day 20)
- Topics: Decision thresholds, calibration, fairness and ethics, model cards, reproducibility, monitoring, drift detection, executive communication, peer review

## Learning Approach

The course follows a consistent pedagogical pattern for each topic:

1. **Concept + demo in notebook** (micro-video)
2. **Guided practice** with a 10-minute "pause-and-do" exercise
3. **Solution video** covering common mistakes + extensions
4. **Next concept + demo** ... and repeat

This loop ensures active learning, immediate practice, and iterative improvement throughout the intensive 4-week format.

## Technology Stack

- **Platform:** Google Colab (cloud-based Jupyter notebooks)
- **AI Assistance:** Google Gemini in Colab for guided "vibe coding"
- **Primary Libraries:**
  - `scikit-learn` - Machine learning models and pipelines
  - `pandas` - Data manipulation and analysis
  - `numpy` - Numerical computing
  - `matplotlib` / `seaborn` - Data visualization
  - `statsmodels` - Statistical modeling
- **Additional Tools:** Microsoft Copilot (optional productivity enhancement)

All course materials are designed to run directly in Google Colab with no local installation required.

## Repository Structure

```
2026Summer_predictive_analytics_purdue_MGMT474/
│
├── notebooks/                          # All 20 daily Colab-ready notebooks
│   ├── 01_launchpad_eda_splits.ipynb
│   ├── 02_preprocessing_pipelines.ipynb
│   ├── 03_regression_metrics_baselines.ipynb
│   ├── 04_linear_features_diagnostics.ipynb
│   ├── 05_regularization_project_proposal.ipynb
│   ├── 06_logistic_pipelines.ipynb
│   ├── 07_classification_metrics_thresholding.ipynb
│   ├── 08_cross_validation_model_comparison.ipynb
│   ├── 09_tuning_feature_engineering_project_baseline.ipynb
│   ├── 10_midterm_casebook.ipynb
│   ├── 11_decision_trees.ipynb
│   ├── 12_random_forests_importance.ipynb
│   ├── 13_gradient_boosting.ipynb
│   ├── 14_model_selection_protocol.ipynb
│   ├── 15_interpretation_error_analysis_project.ipynb
│   ├── 16_decision_thresholds_calibration.ipynb
│   ├── 17_fairness_slicing_model_cards.ipynb
│   ├── 18_reproducibility_monitoring.ipynb
│   ├── 19_data_communication_poster.ipynb
│   └── 20_final_submission_peer_review.ipynb
│
├── docs/                               # Quarto-generated course website (GitHub Pages)
├── images/                             # Course images and logos
│
├── _quarto.yml                         # Quarto website configuration
├── index.qmd                           # Course home page (Quarto)
├── syllabus.qmd                        # Course syllabus (Quarto)
├── schedule.qmd                        # Course schedule (Quarto)
├── styles.css                          # Website styling
│
├── _project_docs/                      # AI-assistant + planning reference docs
│   ├── MGMT47400_Online4Week_Plan_2026Summer.md  # Detailed course planning document
│   ├── claude_course_plan.md           # Implementation plan / content justification
│   ├── DECISIONS.md                    # Load-bearing conventions and rationale
│   ├── NOTEBOOK_TEMPLATE.md            # Canonical 8-section notebook structure
│   └── TROUBLESHOOTING.md              # Failure-mode playbook
├── README.md                           # This file
├── CONVERSATION_LOG.md                 # Development tracking for future sessions
├── LICENSE                             # Repository license
└── .gitignore                          # Git ignore rules
```

## Course Notebooks

All notebooks are designed to run in Google Colab with one-click access. Each notebook follows a standardized format: course logo and instructor header, learning objectives, setup, content sections, PAUSE-AND-DO exercises, wrap-up, bibliography, and a closing cell. The canonical reference for notebook structure is `nb01_launchpad_eda_splits.ipynb`. All notebooks use `RANDOM_SEED = 474` for reproducibility.

| Day | Date | Topic | Notebook | Colab Link |
|-----|------|-------|----------|------------|
| 1 | Mon May 18 | Launchpad: Colab workflow, Gemini vibe-coding, EDA, and splitting | [nb01_launchpad_eda_splits.ipynb](notebooks/nb01_launchpad_eda_splits.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/nb01_launchpad_eda_splits.ipynb) |
| 2 | Tue May 19 | Data setup and preprocessing pipelines | [nb02_preprocessing_pipelines.ipynb](notebooks/nb02_preprocessing_pipelines.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/nb02_preprocessing_pipelines.ipynb) |
| 3 | Wed May 20 | Train/validation/test rigor + regression metrics + baseline modeling | [nb03_regression_metrics_baselines.ipynb](notebooks/nb03_regression_metrics_baselines.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/nb03_regression_metrics_baselines.ipynb) |
| 4 | Thu May 21 | Linear regression: features, interactions, diagnostics | [nb04_linear_features_diagnostics.ipynb](notebooks/nb04_linear_features_diagnostics.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/nb04_linear_features_diagnostics.ipynb) |
| 5 | Fri May 22 | Regularization (Ridge/Lasso) + Project proposal | [nb05_regularization_project_proposal.ipynb](notebooks/nb05_regularization_project_proposal.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/nb05_regularization_project_proposal.ipynb) |
| 6 | Mon May 25 | Logistic regression: probabilities, decision boundaries, pipelines | [nb06_logistic_pipelines.ipynb](notebooks/nb06_logistic_pipelines.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/nb06_logistic_pipelines.ipynb) |
| 7 | Tue May 26 | Classification metrics: confusion matrix, ROC/PR, calibration | [nb07_classification_metrics_thresholding.ipynb](notebooks/nb07_classification_metrics_thresholding.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/nb07_classification_metrics_thresholding.ipynb) |
| 8 | Wed May 27 | Resampling and CV: model comparison discipline | [nb08_cross_validation_model_comparison.ipynb](notebooks/nb08_cross_validation_model_comparison.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/nb08_cross_validation_model_comparison.ipynb) |
| 9 | Thu May 28 | Feature engineering + model selection workflow | [nb09_tuning_feature_engineering_project_baseline.ipynb](notebooks/nb09_tuning_feature_engineering_project_baseline.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/nb09_tuning_feature_engineering_project_baseline.ipynb) |
| 10 | Fri May 29 | Midterm: Business-case strategy practicum | [nb10_midterm_casebook.ipynb](notebooks/nb10_midterm_casebook.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/nb10_midterm_casebook.ipynb) |
| 11 | Mon Jun 1 | Decision trees: interpretable models with sharp edges | [nb11_decision_trees.ipynb](notebooks/nb11_decision_trees.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/nb11_decision_trees.ipynb) |
| 12 | Tue Jun 2 | Random forests: bagging, OOB, feature importance | [nb12_random_forests_importance.ipynb](notebooks/nb12_random_forests_importance.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/nb12_random_forests_importance.ipynb) |
| 13 | Wed Jun 3 | Gradient boosting: performance with discipline | [nb13_gradient_boosting.ipynb](notebooks/nb13_gradient_boosting.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/nb13_gradient_boosting.ipynb) |
| 14 | Thu Jun 4 | Model selection and comparison protocols | [nb14_model_selection_protocol.ipynb](notebooks/nb14_model_selection_protocol.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/nb14_model_selection_protocol.ipynb) |
| 15 | Fri Jun 5 | Interpretation: feature importance + partial dependence | [nb15_interpretation_error_analysis_project.ipynb](notebooks/nb15_interpretation_error_analysis_project.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/nb15_interpretation_error_analysis_project.ipynb) |
| 16 | Mon Jun 8 | Error analysis to decisions: thresholds, calibration, KPI alignment | [nb16_decision_thresholds_calibration.ipynb](notebooks/nb16_decision_thresholds_calibration.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/nb16_decision_thresholds_calibration.ipynb) |
| 17 | Tue Jun 9 | Fairness and ethics: responsible predictive analytics | [nb17_fairness_slicing_model_cards.ipynb](notebooks/nb17_fairness_slicing_model_cards.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/nb17_fairness_slicing_model_cards.ipynb) |
| 18 | Wed Jun 10 | Deployment thinking: reproducibility, monitoring, drift | [nb18_reproducibility_monitoring.ipynb](notebooks/nb18_reproducibility_monitoring.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/nb18_reproducibility_monitoring.ipynb) |
| 19 | Thu Jun 11 | Elements of data communication and poster design: six principles applied to the eleven-section research-poster architecture | [nb19_data_communication_poster.ipynb](notebooks/nb19_data_communication_poster.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/nb19_data_communication_poster.ipynb) |
| 20 | Fri Jun 12 | Final delivery: project package + peer review + closeout | [nb20_final_submission_peer_review.ipynb](notebooks/nb20_final_submission_peer_review.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/nb20_final_submission_peer_review.ipynb) |

## Textbooks and References

### Primary Textbook
- **James, G., Witten, D., Hastie, T., & Tibshirani, R.** (2023). *An Introduction to Statistical Learning with Applications in Python (ISLP)*. Springer. [Download Here](https://www.statlearning.com/)

### Supporting References
- **Hastie, T., Tibshirani, R., & Friedman, J.** *The Elements of Statistical Learning (ESL)*
- **Provost, F., & Fawcett, T.** *Data Science for Business*
- **Pedregosa et al.** "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research*
- **Chip Huyen.** *Designing Machine Learning Systems*
- **scikit-learn User Guide** - Comprehensive documentation on pipelines, preprocessing, model selection, metrics, and inspection tools

### Additional References (Topic-Specific)
- **Barocas, S., Hardt, M., & Narayanan, A.** *Fairness and Machine Learning*
- **Mitchell, M., et al.** "Model Cards for Model Reporting"
- **Breiman, L.** "Random Forests"
- **Friedman, J.H.** "Greedy Function Approximation: A Gradient Boosting Machine"
- **Fawcett, T.** "An introduction to ROC analysis"
- **Cole Nussbaumer Knaflic.** *Storytelling with Data*
- **Barbara Minto.** *The Pyramid Principle*

## Assessment and Grading

The course assessment is based on multiple components designed to ensure continuous engagement and progressive skill development:

| Assessment Component | Weight |
|---------------------|--------|
| Participation | 10% |
| Daily Concept Quizzes | 15% |
| Midterm (Business Case Practicum) | 20% |
| Kaggle Case Competition | 20% |
| Final Project + Milestones | 35% |

### Project Milestones (groups of four randomly assigned members)
1. **Week 1 (Day 5):** Initial Project Proposal — prediction goal + motivation + data overview + preliminary methods + expected contributions
2. **Week 2 (Day 10):** Simple Model + Performance Evaluation — dataset exploration + feature engineering + missing-value handling + baseline pipeline with k-fold CV
3. **Week 3 (Day 15):** More Complex Model + Hyperparameter Tuning + Draft Abstract (~250 words)
4. **Week 4 (Day 20):** Final Research Poster (single PDF named `<group-number>.pdf`) + intra-group Peer Evaluation form. Optional Fall 2026 Purdue Undergraduate Research Conference presentation. Detail files: [`_final_project/2026Summer/`](_final_project/2026Summer/).

### Kaggle Case Competition (individual or pairs)
- **Competition:** Summer 2026 QM47400 Case Competition: Bank Churn
- **Task:** Predict bank customer churn probability (AUC-ROC)
- **Deadline:** Fri June 12, 2026 at 11:59 PM (Kaggle + Brightspace code submission)

## Getting Started

### For Students

1. **Access the Course Website:** Visit [https://davi-moreira.github.io/2026Summer_predictive_analytics_purdue_MGMT474/](https://davi-moreira.github.io/2026Summer_predictive_analytics_purdue_MGMT474/)
2. **Set Up Google Colab:** Ensure you have a Google account and can access [Google Colab](https://colab.research.google.com/)
3. **Start with Day 1:** Click the "Open in Colab" button for Day 1 notebook and follow the setup instructions
4. **Enable Gemini in Colab:** Follow in-notebook instructions to activate Google Gemini AI assistance
5. **Daily Workflow:**
   - Watch micro-videos (posted on Brightspace)
   - Open the day's notebook in Colab
   - Complete "pause-and-do" exercises
   - Submit completed notebook and concept quiz on Brightspace

### For Instructors

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474.git
   ```

2. **Quarto Website (optional local preview):**
   ```bash
   quarto preview
   ```

3. **Customize Materials:** All notebooks and Quarto files can be edited to adapt to your specific context

4. **Video Recording:** Each day requires 6 micro-videos (8-10 minutes each) following the concept-demo-practice-solution pattern detailed in `_project_docs/MGMT47400_Online4Week_Plan_2026Summer.md`

## Contact Information

**Professor Davi Moreira**
Email: dmoreira@purdue.edu
Office: Young Hall 1007
Course Website: [https://davi-moreira.github.io/2026Summer_predictive_analytics_purdue_MGMT474/](https://davi-moreira.github.io/2026Summer_predictive_analytics_purdue_MGMT474/)
Personal Website: [https://davi-moreira.github.io](https://davi-moreira.github.io)

**Brightspace:** Course announcements, video content, quizzes, and assignment submissions are managed through Purdue Brightspace at [https://purdue.brightspace.com/](https://purdue.brightspace.com/)

## License

This course material is licensed under the terms specified in the [LICENSE](LICENSE) file.

---

**Built with [Quarto](https://quarto.org/) | Hosted on [GitHub Pages](https://pages.github.com/)**
