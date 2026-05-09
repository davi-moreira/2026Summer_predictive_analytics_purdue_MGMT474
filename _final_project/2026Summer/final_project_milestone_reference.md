# Final Project — Milestone Reference (Single Source of Truth)

**Course:** QM47400 — Predictive Analytics
**Term:** Summer 2026 (4-week intensive, May 18 – June 12, 2026)
**Instructor:** Professor Davi Moreira

> **This file is the canonical reference for everything related to the Final Project.** Every other document — `syllabus.qmd`, `schedule.qmd`, planning docs, the four milestone instruction files in this folder, and the project-related notebooks (nb05, nb15, nb20) — must be kept consistent with what is written here. Update this file first whenever the project structure changes.

---

## 1. What the Final Project Is

In **groups of four randomly assigned members**, students complete a practical predictive analytics project culminating in a **final research poster**. Each group selects a real-world prediction problem, builds a defensible end-to-end modeling pipeline, and communicates the result in a poster that an academic-and-industry audience can read in five minutes.

Although presenting the poster at the **Fall 2026 Purdue Undergraduate Research Conference** (also referred to as the *Fall Undergraduate Research Expo*) is **not required**, it is **strongly encouraged**. All groups must still prepare and submit the final poster as part of the course requirements. Professor Moreira is happy to serve as faculty mentor for any group choosing to present at the conference.

> **Note on conference year:** the original course-guidelines document references the *2025 Fall Undergraduate Research Expo*; for this Summer 2026 cohort the relevant conference is **Fall 2026**.

---

## 2. Group Logistics

| Item | Detail |
|---|---|
| **Group size** | 4 members |
| **Group formation** | Randomly assigned by the instructor at the start of the course (no later than Day 2) |
| **Group changes** | Not permitted after Day 3 except for documented attendance issues, at instructor discretion |
| **Communication channel** | Each group selects its own (Brightspace group thread, Slack, etc.) and shares the link with the instructor |
| **Working repository** | Recommended: a shared Google Drive folder + a Colab notebook stored in the group folder |

---

## 3. Assessment Structure

The Final Project is worth **35% of the overall course grade**. Within the project, the three components below combine to 100%.

| Component | Weight (of project) | Weight (of course) | What it covers |
|---|---:|---:|---|
| **Milestone Deliverables** | 40% | 14.0% | Incremental project components (M1 – M4) submitted on the Friday lecture days. Graded for clarity, completeness, and timely submission. |
| **Peer Evaluation** | 20% | 7.0% | Each group member confidentially evaluates teammates' contributions. Encourages accountability and balanced participation. |
| **Instructor / TA Evaluation** | 40% | 14.0% | Final research poster, graded against the rubric (poster template + rubric provided separately on Brightspace). |
| **Total** | **100%** | **35.0%** | |

### 3.1 What "Milestone Deliverables (40%)" includes

The 40% milestone score is the average of the four milestone scores (M1, M2, M3, M4-non-poster components). The poster artifact itself is **not** counted here — it is graded under the Instructor/TA Evaluation 40%.

### 3.2 What "Peer Evaluation (20%)" measures

A confidential intra-group peer evaluation collected at the end of the course (with the M4 submission). Each member rates the other three on a small set of dimensions (commitment, technical contribution, communication, dependability, fairness of work distribution). The instructor uses the aggregated peer scores to compute the 20% component for each member individually — so two members of the same group can receive different peer-evaluation scores.

### 3.3 What "Instructor/TA Evaluation (40%)" covers

The final poster, submitted as a single PDF on the deadline indicated in Brightspace. Graded by the instructor (with TA support) against a rubric covering: prediction problem framing, methodology, results and interpretation, visual design and clarity, and reproducibility (link to the supporting notebook/code).

---

## 4. Milestone Schedule

The four course milestones are aligned with the course's Friday synchronous lecture days (Days 5, 10, 15, 20). All milestones are submitted via Brightspace by **11:59 PM** on the date listed.

| # | Course day | Date (2026) | Deliverable | Detail file |
|---|---|---|---|---|
| **M1** | Day 5 | Fri May 22 | Initial Project Proposal | [`milestone_01_proposal.md`](milestone_01_proposal.md) |
| **M2** | Day 10 | Fri May 29 | Simple Model + Performance Evaluation | [`milestone_02_baseline_model.md`](milestone_02_baseline_model.md) |
| **M3** | Day 15 | Fri Jun 5 | More Complex Model + Tuning + Draft Abstract | [`milestone_03_complex_model_and_abstract.md`](milestone_03_complex_model_and_abstract.md) |
| **M4** | Day 20 | Fri Jun 12 | Final Research Poster + Peer Evaluation | [`milestone_04_final_poster.md`](milestone_04_final_poster.md) |

Each milestone .md file in this folder contains the full instruction set: purpose, required components, submission format, and the grading rubric.

### 4.1 Notebook anchors

Each milestone is supported by a course notebook that walks through the relevant techniques on a worked example before students apply them to their own dataset:

| Milestone | Supporting notebook(s) |
|---|---|
| M1 | `nb05_regularization_project_proposal_student.ipynb` (proposal sprint) |
| M2 | `nb09_tuning_feature_engineering_project_baseline_student.ipynb` (baseline + CV scaffold), `nb10_midterm_casebook_student.ipynb` (baseline submission day) |
| M3 | `nb15_interpretation_error_analysis_project_student.ipynb` (champion model + interpretation + abstract scaffold) |
| M4 | `nb19_data_communication_poster_student.ipynb` (six principles of data communication + poster outline + abstract drafting), `nb20_final_submission_peer_review_student.ipynb` (final-day peer review + submission audit) |

---

## 5. Final Poster — Format and Submission

### 5.1 Format

- **Single PDF file**, sized to the standard Purdue Undergraduate Research Conference poster dimensions (the template provided on Brightspace specifies the exact size).
- **Filename convention:** `<group-number>.pdf` — e.g., a group assigned the number "01" submits `01.pdf`; group "17" submits `17.pdf`. Do **not** include the section number in the filename. Following this convention is what allows the instructor to print the posters for free.
- **Poster template:** provided on Brightspace; based on the standard Purdue Undergraduate Research Symposium poster format.

### 5.2 Required content

A standout poster reads in roughly five minutes and includes:

1. **Project title** (concise; if the project uses a synthetic generated dataset, the title must say so)
2. **Group members + section**
3. **Prediction problem** (framed as a clear question with a question mark — e.g., *"Can we predict customer churn within six months from transaction history?"*)
4. **Motivation and significance**
5. **Data overview** (source, size, key variables)
6. **Methodology** (preprocessing, feature engineering, baseline + complex models, evaluation protocol, hyperparameter tuning)
7. **Results** (key tables, figures, performance metrics with cross-validated 95% CIs where appropriate)
8. **Interpretation and insights**
9. **Limitations + next steps**
10. **References** (including the supporting code repository / Colab link)

### 5.3 Examples to draw from

Award-winning student posters from prior cohorts: <https://davi-moreira.github.io/applied_projects.html>

### 5.4 Optional: presenting at the Fall 2026 conference

Students who plan to present at the **Fall 2026 Purdue Undergraduate Research Conference** are encouraged to email the instructor early (ideally before the M3 deadline) so mentorship, feedback, and guidance can be arranged. Conference details: <https://www.purdue.edu/undergrad-research/conferences/index.php>

Presenting is **optional** and has no impact on the course grade.

---

## 6. Peer Evaluation — Logistics

The intra-group peer evaluation is administered at the end of the course (collected with the M4 submission window). Mechanics:

- Each member evaluates the other **three teammates only** (no self-evaluation).
- Confidential — only the instructor (and TA) sees individual evaluations; aggregated feedback may be returned to the team.
- Submitted via a Brightspace form (link provided in Module 4 of the course Brightspace site). Form structure:
  - Numerical rating (1–5) on each of: commitment, technical contribution, communication, dependability, fairness of work distribution
  - Free-text field for one specific strength and one specific area for improvement per teammate
  - One overall comment about how the group functioned
- Failure to submit a peer evaluation reduces the submitter's own peer-evaluation score (not the teammates').

The 20% peer-evaluation component is calculated **per member** from the average of the three ratings each member receives, with light moderation by the instructor when ratings appear strategically inflated or deflated.

---

## 7. Mentor Support

Professor Moreira is available as faculty mentor for any group that opts to present at the **Fall 2026 Purdue Undergraduate Research Conference**. Reach out by email early (before the M3 deadline) to arrange mentorship and conference-specific guidance.

For ordinary milestone questions during the course, use the regular office-hours channels documented in the syllabus.

---

## 8. Resources and Cross-References

- **This folder:** `_final_project/2026Summer/`
  - `final_project_milestone_reference.md` (this file — canonical reference)
  - `milestone_01_proposal.md`
  - `milestone_02_baseline_model.md`
  - `milestone_03_complex_model_and_abstract.md`
  - `milestone_04_final_poster.md`
  - `final_project_milestone_reference.docx` (legacy long-form reference; superseded by this .md but retained for archive)
- **Award-winning posters:** <https://davi-moreira.github.io/applied_projects.html>
- **Purdue Undergraduate Research Conferences:** <https://www.purdue.edu/undergrad-research/conferences/index.php>
- **Course syllabus + schedule:** `syllabus.qmd`, `schedule.qmd` (rendered to the course website's `docs/`)

---

## 9. Change Log

Maintain this section when the structure changes.

| Date | Change | Authored by |
|---|---|---|
| 2026-05-04 | Initial canonical version: groups of 4, 40/20/40 assessment, four milestones (M1–M4), poster as final deliverable. Compressed the long-form .docx (six milestones) into the 4-week intensive schedule. | Davi Moreira (with Claude assistance) |

---

**End of canonical reference.**
