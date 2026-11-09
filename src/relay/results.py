"""The raw results file (one row per request), plus a sidecar with answer text.

``results/results.csv`` has these columns, one row per request:

    request_id, scenario, source, start_time, first_token_ms, total_ms, status,
    retries, tokens_in, tokens_out, retrieval_setting, answer_grade

Conventions:

* ``start_time`` is the arrival time as unix epoch seconds with millisecond
  precision, so in-system concurrency can be recomputed from
  ``[start_time, start_time + total_ms]``.
* ``first_token_ms`` is empty when nothing was shown to the user.
* ``answer_grade`` is empty until graded (``relay.grade``); rows whose
  content came from the stand-in's synthetic replies are never graded.

``results/answers.jsonl`` holds the answer text, kind and question id per
``request_id`` for grading.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from relay.config import RESULTS_DIR
from relay.pipeline import Outcome

COLUMNS = [
    "request_id",
    "scenario",
    "source",
    "start_time",
    "first_token_ms",
    "total_ms",
    "status",
    "retries",
    "tokens_in",
    "tokens_out",
    "retrieval_setting",
    "answer_grade",
]
RESULTS_CSV = RESULTS_DIR / "results.csv"
ANSWERS_JSONL = RESULTS_DIR / "answers.jsonl"


class ResultsWriter:
    """Appends request rows to ``results.csv`` and answer records to the sidecar.

    Attributes:
        csv_path: CSV destination.
        answers_path: Sidecar destination.
    """

    def __init__(self, csv_path: Path = RESULTS_CSV, answers_path: Path = ANSWERS_JSONL):
        """Initialises the writer, creating the CSV header if needed.

        Args:
            csv_path: CSV destination.
            answers_path: Sidecar destination.
        """
        self.csv_path = csv_path
        self.answers_path = answers_path
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not csv_path.exists() or csv_path.stat().st_size == 0:
            with open(csv_path, "w", newline="") as f:
                csv.writer(f).writerow(COLUMNS)

    def write(
        self,
        o: Outcome,
        scenario: str,
        source: str,
        start_time: float,
        question_id: str = "",
        extra: dict | None = None,
    ) -> None:
        """Appends one request.

        Args:
            o: The request outcome.
            scenario: Scenario name.
            source: ``"real"`` or ``"stand-in"``.
            start_time: Arrival time, unix epoch seconds.
            question_id: Eval question id, if the question came from the eval set.
            extra: Additional sidecar fields (cache hit, queue wait, ...).
        """
        row = [
            o.request_id,
            scenario,
            source,
            f"{start_time:.3f}",
            "" if o.first_token_ms is None else f"{o.first_token_ms:.1f}",
            f"{o.total_ms:.1f}",
            o.status,
            o.retries,
            o.tokens_in,
            o.tokens_out,
            o.retrieval_setting,
            "",
        ]
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        rec = {
            "request_id": o.request_id,
            "scenario": scenario,
            "source": source,
            "question_id": question_id,
            "question": o.question,
            "answer_kind": o.answer_kind,
            "text": o.text,
            "citations": o.citations,
            "draft_model": o.draft_model,
            "degraded": o.degraded,
            "refusal_reason": o.refusal_reason,
            **(extra or {}),
        }
        with open(self.answers_path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
