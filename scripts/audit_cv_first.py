#!/usr/bin/env python3
"""
CV-First / Test-Set-Lock Audit
==============================

Scans notebooks/*_student.ipynb for model-evaluation uses of X_test/y_test that
violate the CV-first rule (see claude.md → "CRITICAL RULE - CV-First Evaluation
+ Test-Set Lock").

The only acceptable hits are:
  * notebooks/nb14_model_selection_protocol_student.ipynb cells 30 (clf
    ceremony) and 34 (reg ceremony) — the two authorized test-set openings,
    one per spine. Each test set still opens exactly ONCE per case; the
    singleness rule is preserved per project, not diluted across spines.
  * notebooks/nb18_reproducibility_monitoring_student.ipynb (Kaggle-submission
    demo — production pipeline, not model evaluation)

Anything else is a regression and must be fixed before commit.

Usage
-----
    python scripts/audit_cv_first.py

Run from the repository root.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

MODEL_EVAL_TEST_PATTERNS = [
    r"\.score\(X_test",
    r"\.predict\(X_test",
    r"\.predict_proba\(X_test",
    r"roc_auc_score\(y_test",
    r"accuracy_score\(y_test",
    r"f1_score\(y_test",
    r"precision_score\(y_test",
    r"recall_score\(y_test",
    r"classification_report\(y_test",
    r"brier_score_loss\(y_test",
    r"permutation_importance\([^,)]+,\s*X_test",
]


def audit_notebook(path: Path) -> list[tuple[int, str]]:
    """Return a list of (cell_index, matched_pattern) hits for one notebook."""
    nb = json.loads(path.read_text())
    hits: list[tuple[int, str]] = []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        for pat in MODEL_EVAL_TEST_PATTERNS:
            for m in re.finditer(pat, src):
                line_start = src.rfind("\n", 0, m.start()) + 1
                next_newline = src.find("\n", m.start())
                line = src[line_start : next_newline if next_newline != -1 else len(src)]
                if line.lstrip().startswith("#"):
                    continue
                hits.append((i, m.group(0)))
    return hits


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    notebooks_dir = repo_root / "notebooks"
    if not notebooks_dir.is_dir():
        print(f"ERROR: {notebooks_dir} not found.", file=sys.stderr)
        return 2

    any_hits = False
    for path in sorted(notebooks_dir.glob("*_student.ipynb")):
        hits = audit_notebook(path)
        if not hits:
            continue
        any_hits = True
        print(f"{path.name}: {len(hits)} hits")
        for cell, pat in hits:
            print(f"  cell {cell}: {pat}")

    if not any_hits:
        print("Clean — no model-evaluation uses of X_test/y_test found.")
        return 0

    print()
    print("Acceptable: nb14 cells 30 + 34 (clf + reg ceremonies, one per spine);")
    print("            nb18 (Kaggle-submission demo).")
    print("Anything else is a regression — fix before commit.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
