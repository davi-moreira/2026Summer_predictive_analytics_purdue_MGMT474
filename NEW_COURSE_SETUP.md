# New-Course Seed — Hand-off Guide

This directory was **seeded from** `2026Summer_predictive_analytics_purdue_MGMT474`
using `scripts/make_seed.sh`. It carries the *portable core* of the course
(notebooks, AI operating manual, planning docs, templates, scripts) but **not**
the heavy or term-specific material (video, slides, rendered site, grades).

The new course is the **same course, full-semester format** — so the *content
spine is identical* and the *scheduling shell is different*. This guide tells the
next Claude session exactly what is reusable as-is versus what to rewrite.

---

## What came across in the seed

| Reuse as-is (the "core") | Rewrite for the new term | Regenerate (do NOT carry over) |
|---|---|---|
| `notebooks/*_student.ipynb` (the 21-notebook arc) | `schedule.qmd` (20 business days → full semester) | `docs/` (Quarto render output) |
| `claude.md` → **rename to `CLAUDE.md`** | `syllabus.qmd` (dates, term, policies) | `_notebook_lm/` (NotebookLM transcripts) |
| `_project_docs/DECISIONS.md`, `NOTEBOOK_TEMPLATE.md`, `TROUBLESHOOTING.md` | `_project_docs/MGMT47400_Online4Week_Plan_2026Summer.md` → new full-semester plan | `videos/`, `lecture_slides/` (record fresh) |
| `scripts/` (audit, voice-check, sync, this seed script) | `_project_docs/claude_course_plan.md` (cadence/sequence) | `_adm_stuff/`, `_announcements/` (per-term, private) |
| `video_guides/` structure + template | `index.qmd` (title/landing) | `notebooks/data/` (re-download datasets) |
| `_production_kit/` (poster/video/competition templates) | `brightspace/` daily pages (re-pace to weeks) | `.git/` (start a fresh repo) |
| `_final_project/`, `_course_case_competition/` structure, rubrics | `_quarto.yml` site `title:` + GitHub URL | `.venv/`, `.quarto/`, `_freeze/` |
| `_homework/`, `_handouts/`, `_quizzes/`, `_midterm_exam/`, `_lecture_notes/` | `README.md` (repo name, dates) | |

The single most important framing for the next session: **the 4-week intensive
cadence (~112.5 min/day across 20 business days) is the part that does NOT
transfer.** A full semester re-paces the same 21 notebooks across ~15 weeks with
different activity spacing. Everything that mentions "4-Week", "20 business days",
"May 18 – June 12, 2026", or "112.5 minutes/day" is a rewrite target.

---

## Setup steps (run these in order)

### 1. Create the seed (from the *source* repo, not here)
```bash
cd /Users/dcordeir/Dropbox/academic/cursos/cursos-davi/predictive_analytics/2026Summer_predictive_analytics_purdue_MGMT474
# dry-run first to see what copies:
bash scripts/make_seed.sh "../2026F_predictive_analytics__QM474"
# then for real:
bash scripts/make_seed.sh "../2026F_predictive_analytics__QM474" --go
```

### 2. Rename the operating manual + initialize git
```bash
cd ../2026F_predictive_analytics__QM474
git mv claude.md CLAUDE.md 2>/dev/null || mv claude.md CLAUDE.md   # canonical, case-sensitive-safe
git init
git add -A
git commit -m "chore: Seed 2026F QM474 from 2026Summer course core"
```

### 3. Create the remote repo and push
```bash
# GitHub CLI (recommended):
gh repo create 2026F_predictive_analytics__QM474 --public --source=. --remote=origin --push

# …or manually, if you create the repo in the GitHub web UI first:
git remote add origin https://github.com/davi-moreira/2026F_predictive_analytics__QM474.git
git branch -M main
git push -u origin main
```

### 4. Enable GitHub Pages (Quarto site)
- Repo → **Settings → Pages → Source: Deploy from a branch → `main` / `docs`**.
- Then render locally and push:
```bash
/Applications/RStudio.app/Contents/Resources/app/quarto/bin/quarto render
git add docs/ && git commit -m "build: Initial render" && git push
```

### 5. Bootstrap the new Claude session
Open Claude Code in `2026F_predictive_analytics__QM474` and paste:

> Read `CLAUDE.md`, `_project_docs/MGMT47400_Online4Week_Plan_2026Summer.md`, and
> this `NEW_COURSE_SETUP.md`. This repo was seeded from a 4-week intensive
> version of the same course. Your first job is to convert it to a **full-semester
> QM474** offering. Do NOT touch notebook content yet. Instead: (1) produce a new
> full-semester course plan that re-paces the same 21 notebooks across ~15 weeks;
> (2) list every file containing "4-Week", "20 business days", "112.5 minutes", or
> the old dates so we can rewrite them; (3) update `CLAUDE.md`'s Project Mission,
> `_quarto.yml` title, `syllabus.qmd`, `schedule.qmd`, and `README.md` to the new
> format and the new repo URL. Confirm the plan with me before editing notebooks.

---

## Notes
- `make_seed.sh` is **dry-run by default** — it only copies when you add `--go`.
- Editing `scripts/make_seed.sh`'s `EXCLUDES` array changes what carries over;
  e.g. drop `_homework/` from excludes if you want to seed homework too (already
  included by default), or add it to excludes to start homework from scratch.
- Once the new course is established and adapted, delete this file.
