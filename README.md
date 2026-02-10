# MGMT 47400: Predictive Analytics

[![Course Website](https://img.shields.io/badge/Website-Course%20Page-blue)](https://davi-moreira.github.io/2026Summer_predictive_analytics_purdue_MGMT474/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/)

## Course Information

**Instructor:** [Professor Davi Moreira](https://davi-moreira.github.io)
**Email:** dmoreira@purdue.edu
**Office:** Young Hall 1007
**Institution:** Mitch Daniels School of Business, Purdue University

**Course Format:** 4-Week Fully Online Intensive
**Course Dates:** May 18 - June 14, 2027 (20 business days, Mon-Fri)
**Daily Time Commitment:** 112.5 minutes per business day
**Credits:** 3 credit hours

## Course Description

This 4-week fully online intensive course enables students to navigate the entire predictive analytics pipeline skillfully—from data preparation and exploration to modeling, assessment, and interpretation. Running over **20 business days** with **112.5 minutes of daily engagement**, the course combines short micro-videos (12 minutes each), hands-on Google Colab notebooks, exercises, quizzes, and progressive project work.

Students engage with real-world examples through interactive Jupyter notebooks, all designed to run in **Google Colab with Google Gemini AI assistance**. The course emphasizes essential programming and analytical skills through a "vibe coding" approach: **draft → verify → document**. Topics include exploratory data analysis, train/validation/test splits, linear and logistic regression, classification metrics, resampling methods, regularization techniques, tree-based approaches, gradient boosting, model interpretation, fairness considerations, and deployment thinking.

The course centers on **one comprehensive capstone project** that progresses through four weekly milestones: proposal (Day 5), baseline model (Day 10), improved model with interpretation (Day 15), and final executive-ready deliverable including slide narrative and conference-style video (Day 20).

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
│   ├── 19_project_narrative_video_studio.ipynb
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
├── MGMT47400_Online4Week_Plan_2026Summer.md  # Detailed course planning document
├── README.md                           # This file
├── CONVERSATION_LOG.md                 # Development tracking for future sessions
├── LICENSE                             # Repository license
└── .gitignore                          # Git ignore rules
```

## Course Notebooks

All notebooks are designed to run in Google Colab with one-click access. Each notebook follows a standardized format: course logo and instructor header, learning objectives, setup, content sections, PAUSE-AND-DO exercises, wrap-up, bibliography, and a closing cell. The canonical reference for notebook structure is `01_launchpad_eda_splits.ipynb`. All notebooks use `RANDOM_SEED = 474` for reproducibility.

| Day | Date | Topic | Notebook | Colab Link |
|-----|------|-------|----------|------------|
| 1 | Tue May 18 | Launchpad: Colab workflow, Gemini vibe-coding, EDA, and splitting | [01_launchpad_eda_splits.ipynb](notebooks/01_launchpad_eda_splits.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/01_launchpad_eda_splits.ipynb) |
| 2 | Wed May 19 | Data setup and preprocessing pipelines | [02_preprocessing_pipelines.ipynb](notebooks/02_preprocessing_pipelines.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/02_preprocessing_pipelines.ipynb) |
| 3 | Thu May 20 | Train/validation/test rigor + regression metrics + baseline modeling | [03_regression_metrics_baselines.ipynb](notebooks/03_regression_metrics_baselines.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/03_regression_metrics_baselines.ipynb) |
| 4 | Fri May 21 | Linear regression: features, interactions, diagnostics | [04_linear_features_diagnostics.ipynb](notebooks/04_linear_features_diagnostics.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/04_linear_features_diagnostics.ipynb) |
| 5 | Mon May 24 | Regularization (Ridge/Lasso) + Project proposal | [05_regularization_project_proposal.ipynb](notebooks/05_regularization_project_proposal.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/05_regularization_project_proposal.ipynb) |
| 6 | Tue May 25 | Logistic regression: probabilities, decision boundaries, pipelines | [06_logistic_pipelines.ipynb](notebooks/06_logistic_pipelines.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/06_logistic_pipelines.ipynb) |
| 7 | Wed May 26 | Classification metrics: confusion matrix, ROC/PR, calibration | [07_classification_metrics_thresholding.ipynb](notebooks/07_classification_metrics_thresholding.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/07_classification_metrics_thresholding.ipynb) |
| 8 | Thu May 27 | Resampling and CV: model comparison discipline | [08_cross_validation_model_comparison.ipynb](notebooks/08_cross_validation_model_comparison.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/08_cross_validation_model_comparison.ipynb) |
| 9 | Fri May 28 | Feature engineering + model selection workflow | [09_tuning_feature_engineering_project_baseline.ipynb](notebooks/09_tuning_feature_engineering_project_baseline.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/09_tuning_feature_engineering_project_baseline.ipynb) |
| 10 | Mon May 31 | Midterm: Business-case strategy practicum | [10_midterm_casebook.ipynb](notebooks/10_midterm_casebook.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/10_midterm_casebook.ipynb) |
| 11 | Tue June 1 | Decision trees: interpretable models with sharp edges | [11_decision_trees.ipynb](notebooks/11_decision_trees.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/11_decision_trees.ipynb) |
| 12 | Wed June 2 | Random forests: bagging, OOB, feature importance | [12_random_forests_importance.ipynb](notebooks/12_random_forests_importance.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/12_random_forests_importance.ipynb) |
| 13 | Thu June 3 | Gradient boosting: performance with discipline | [13_gradient_boosting.ipynb](notebooks/13_gradient_boosting.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/13_gradient_boosting.ipynb) |
| 14 | Fri June 4 | Model selection and comparison protocols | [14_model_selection_protocol.ipynb](notebooks/14_model_selection_protocol.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/14_model_selection_protocol.ipynb) |
| 15 | Mon June 7 | Interpretation: feature importance + partial dependence | [15_interpretation_error_analysis_project.ipynb](notebooks/15_interpretation_error_analysis_project.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/15_interpretation_error_analysis_project.ipynb) |
| 16 | Tue June 8 | Error analysis to decisions: thresholds, calibration, KPI alignment | [16_decision_thresholds_calibration.ipynb](notebooks/16_decision_thresholds_calibration.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/16_decision_thresholds_calibration.ipynb) |
| 17 | Wed June 9 | Fairness and ethics: responsible predictive analytics | [17_fairness_slicing_model_cards.ipynb](notebooks/17_fairness_slicing_model_cards.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/17_fairness_slicing_model_cards.ipynb) |
| 18 | Thu June 10 | Deployment thinking: reproducibility, monitoring, drift | [18_reproducibility_monitoring.ipynb](notebooks/18_reproducibility_monitoring.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/18_reproducibility_monitoring.ipynb) |
| 19 | Fri June 11 | Executive narrative: slide-style story + conference video plan | [19_project_narrative_video_studio.ipynb](notebooks/19_project_narrative_video_studio.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/19_project_narrative_video_studio.ipynb) |
| 20 | Mon June 14 | Final delivery: project package + peer review + closeout | [20_final_submission_peer_review.ipynb](notebooks/20_final_submission_peer_review.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davi-moreira/2026Summer_predictive_analytics_purdue_MGMT474/blob/main/notebooks/20_final_submission_peer_review.ipynb) |

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
| Daily Concept Quizzes | 20% |
| Notebook Checkpoints | 15% |
| Midterm (Business Case Practicum) | 20% |
| Project Milestones (4 deliverables) | 35% |
| Peer Review | 5% |
| Final Concept Quiz | 5% |

### Project Milestones
1. **Week 1 (Day 5):** Proposal + dataset selection
2. **Week 2 (Day 10):** Baseline model + evaluation plan
3. **Week 3 (Day 15):** Improved model + interpretation
4. **Week 4 (Day 20):** Final deliverable (notebook + slide narrative + conference-style video)

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
   - Submit notebook checkpoint and concept quiz on Brightspace

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

4. **Video Recording:** Each day requires 6 micro-videos (8-10 minutes each) following the concept-demo-practice-solution pattern detailed in `MGMT47400_Online4Week_Plan_2026Summer.md`

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
