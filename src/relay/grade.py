"""Human-in-the-loop grading of eval answers.

``propose`` writes ``eval/grades.csv`` with a rule-based proposed grade per
answer; a person reviews each row and fills ``final`` (and ``note``).
``apply`` copies ``final`` grades into the ``answer_grade`` column of
``results/results.csv``. Human edits are never overwritten by ``propose``.

Proposal rules:

* answerable + answer + every ``must_contain`` string present + cites the
  paper named in ``location`` -> ``correct``; otherwise ``wrong``.
* answerable + refusal or extractive -> ``refused_wrongly``.
* unanswerable + refusal -> ``refused_correctly``; + answer -> ``wrong``.

Only answers produced from the real service are graded: stand-in content is
synthetic.

Typical usage example:

    uv run python -m relay.grade propose
    # edit eval/grades.csv: fill the `final` column
    uv run python -m relay.grade apply
"""

from __future__ import annotations

import argparse
import csv
import json

import yaml

from relay.config import ROOT
from relay.results import ANSWERS_JSONL, COLUMNS, RESULTS_CSV

GRADES_CSV = ROOT / "eval/grades.csv"
FIELDS = [
    "request_id",
    "question_id",
    "setting",
    "answer_kind",
    "proposed",
    "final",
    "note",
    "question",
    "gold",
    "answer",
]
VALID = {"correct", "wrong", "refused_correctly", "refused_wrongly"}


def propose_grade(q: dict, ans: dict) -> str:
    """Proposes a grade for one answer.

    Args:
        q: Eval question record.
        ans: Answer record from the sidecar.

    Returns:
        One of the four grade values.
    """
    kind = ans["answer_kind"]
    if not q["answerable"]:
        return "refused_correctly" if kind in ("refusal", "extractive") else "wrong"
    if kind != "answer":
        return "refused_wrongly"
    text = ans["text"].lower()
    has_all = all(s.lower() in text for s in q.get("must_contain") or [])
    any_of = q.get("must_contain_any") or []
    has_all = has_all and (not any_of or any(s.lower() in text for s in any_of))
    paper = (q.get("location") or "").split()[0] if q.get("location") else ""
    cites_paper = not paper or any(c.startswith(paper) for c in ans.get("citations") or [])
    return "correct" if has_all and cites_paper else "wrong"


def propose(scenario_prefix: str) -> None:
    """Writes or extends ``eval/grades.csv`` with proposals for ungraded answers.

    Args:
        scenario_prefix: Only answers from scenarios starting with this are graded.
    """
    with open(ROOT / "eval/questions.yaml") as f:
        questions = {q["id"]: q for q in yaml.safe_load(f)}
    existing: dict[str, dict] = {}
    if GRADES_CSV.exists():
        with open(GRADES_CSV) as f:
            existing = {r["request_id"]: r for r in csv.DictReader(f)}
    settings = {}
    with open(RESULTS_CSV) as f:
        for r in csv.DictReader(f):
            settings[r["request_id"]] = r["retrieval_setting"]
    rows = list(existing.values())
    added = 0
    with open(ANSWERS_JSONL) as f:
        for line in f:
            a = json.loads(line)
            if (
                a["request_id"] in existing
                or a["source"] != "real"
                or not a["scenario"].startswith(scenario_prefix)
                or a["question_id"] not in questions
            ):
                continue
            q = questions[a["question_id"]]
            rows.append(
                {
                    "request_id": a["request_id"],
                    "question_id": a["question_id"],
                    "setting": settings.get(a["request_id"], ""),
                    "answer_kind": a["answer_kind"],
                    "proposed": propose_grade(q, a),
                    "final": "",
                    "note": "",
                    "question": q["question"],
                    "gold": q["gold"],
                    "answer": a["text"],
                }
            )
            added += 1
    with open(GRADES_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(
        f"{added} new proposals; {sum(1 for r in rows if not r['final'])} rows await review "
        f"in {GRADES_CSV.relative_to(ROOT)}"
    )


def apply() -> None:
    """Copies reviewed ``final`` grades into ``results.csv``.

    Raises:
        SystemExit: If a ``final`` value is not a valid grade.
    """
    with open(GRADES_CSV) as f:
        grades = {
            r["request_id"]: r["final"].strip() for r in csv.DictReader(f) if r["final"].strip()
        }
    bad = {g for g in grades.values() if g not in VALID}
    if bad:
        raise SystemExit(f"invalid grades in {GRADES_CSV.name}: {bad}")
    with open(RESULTS_CSV) as f:
        rows = list(csv.DictReader(f))
    n = 0
    for r in rows:
        if r["request_id"] in grades:
            r["answer_grade"] = grades[r["request_id"]]
            n += 1
    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"applied {n} grades to {RESULTS_CSV.relative_to(ROOT)}")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Grade eval answers.")
    ap.add_argument("action", choices=["propose", "apply"])
    ap.add_argument("--scenario-prefix", default="eval")
    a = ap.parse_args()
    propose(a.scenario_prefix) if a.action == "propose" else apply()


if __name__ == "__main__":
    main()
