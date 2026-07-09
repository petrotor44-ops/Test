"""Reproducible evaluation for retrieval and abstention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from ..utils.data_handling import DataIssue, detect_data_quality_issues, load_docs
from .answering import AnswerStatus, answer_question
from .retrievers import HybridRetriever, WeakBaselineRetriever


def load_eval(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _hit_at(hits: list[str], expected_docs: list[str], k: int) -> bool:
    if not expected_docs:
        return False
    expected = set(expected_docs)
    return any(doc_id in expected for doc_id in hits[:k])


def _contains_all(answer: str, required: list[str]) -> bool:
    lower = answer.lower()
    return all(item.lower() in lower for item in required)


def _serialize_data_issue(issue: DataIssue) -> dict[str, Any]:
    return {
        "kind": issue.kind.value,
        "severity": issue.severity,
        "doc_ids": list(issue.doc_ids),
        "description": issue.description,
        "policy": issue.policy,
    }


def evaluate(corpus_path: str | Path, eval_path: str | Path) -> dict[str, Any]:
    docs = load_docs(corpus_path)
    baseline = WeakBaselineRetriever(docs)
    improved = HybridRetriever(docs)
    cases = load_eval(eval_path)

    per_case: list[dict[str, Any]] = []

    for case in cases:
        query = case["query"]
        expected_docs = list(case.get("expected_docs", []))
        answerable = bool(case["answerable"])
        must_contain = list(case.get("must_contain", []))

        baseline_hits = baseline.retrieve(query, top_k=3)
        improved_result = answer_question(query, improved, top_k=5)

        baseline_doc_ids = [hit.doc_id for hit in baseline_hits]
        improved_doc_ids = [hit.doc_id for hit in improved_result.hits]

        fact_coverage = _contains_all(improved_result.answer, must_contain) if answerable else None

        conflict_ok = None
        if case.get("expects_conflict"):
            conflict_ok = improved_result.status is AnswerStatus.CONFLICT and bool(fact_coverage)

        per_case.append(
            {
                "id": case["id"],
                "query": query,
                "answerable": answerable,
                "expected_docs": expected_docs,
                "baseline_docs": baseline_doc_ids,
                "baseline_scores": [round(hit.score, 6) for hit in baseline_hits],
                "baseline_hit@1": _hit_at(baseline_doc_ids, expected_docs, 1) if answerable else None,
                "baseline_hit@3": _hit_at(baseline_doc_ids, expected_docs, 3) if answerable else None,
                "baseline_abstained": False,
                "improved_status": improved_result.status.value,
                "improved_answer": improved_result.answer,
                "improved_docs": improved_doc_ids[:5],
                "improved_scores": [round(hit.score, 6) for hit in improved_result.hits[:5]],
                "improved_hit@1": _hit_at(improved_doc_ids, expected_docs, 1) if answerable else None,
                "improved_hit@3": _hit_at(improved_doc_ids, expected_docs, 3) if answerable else None,
                "improved_abstained": improved_result.abstained,
                "fact_coverage": fact_coverage,
                "conflict_ok": conflict_ok,
            }
        )

    answerable_rows = [row for row in per_case if row["answerable"]]
    unanswerable_rows = [row for row in per_case if not row["answerable"]]
    conflict_rows = [row for row in per_case if row["conflict_ok"] is not None]

    metrics = {
        "case_count": len(per_case),
        "answerable_count": len(answerable_rows),
        "unanswerable_count": len(unanswerable_rows),
        "baseline_answerable_hit@1": mean(row["baseline_hit@1"] for row in answerable_rows),
        "baseline_answerable_hit@3": mean(row["baseline_hit@3"] for row in answerable_rows),
        "baseline_unanswerable_abstention_accuracy": mean(row["baseline_abstained"] for row in unanswerable_rows),
        "improved_answerable_hit@1": mean(row["improved_hit@1"] for row in answerable_rows),
        "improved_answerable_hit@3": mean(row["improved_hit@3"] for row in answerable_rows),
        "improved_unanswerable_abstention_accuracy": mean(row["improved_abstained"] for row in unanswerable_rows),
        "improved_answerable_non_abstention_rate": mean(not row["improved_abstained"] for row in answerable_rows),
        "improved_answer_fact_coverage": mean(bool(row["fact_coverage"]) for row in answerable_rows),
        "improved_conflict_detection_accuracy": (
            mean(bool(row["conflict_ok"]) for row in conflict_rows) if conflict_rows else None
        ),
        "improved_end_to_end_accuracy": mean(
            (bool(row["fact_coverage"]) and not row["improved_abstained"])
            if row["answerable"]
            else bool(row["improved_abstained"])
            for row in per_case
        ),
    }

    return {
        "metrics": metrics,
        "data_quality_issues": [
            _serialize_data_issue(issue) for issue in detect_data_quality_issues(docs)
        ],
        "cases": per_case,
    }


def write_markdown(report: dict[str, Any], path: str | Path) -> None:
    metrics = report["metrics"]

    rows = [
        (
            "Answerable retrieval hit@1",
            metrics["baseline_answerable_hit@1"],
            metrics["improved_answerable_hit@1"],
        ),
        (
            "Answerable retrieval hit@3",
            metrics["baseline_answerable_hit@3"],
            metrics["improved_answerable_hit@3"],
        ),
        (
            "Unanswerable abstention accuracy",
            metrics["baseline_unanswerable_abstention_accuracy"],
            metrics["improved_unanswerable_abstention_accuracy"],
        ),
        (
            "Improved fact coverage on answerable cases",
            None,
            metrics["improved_answer_fact_coverage"],
        ),
        (
            "Improved end-to-end accuracy",
            None,
            metrics["improved_end_to_end_accuracy"],
        ),
    ]

    lines = [
        "# Evaluation Report",
        "",
        (
            f"Cases: {metrics['case_count']} "
            f"({metrics['answerable_count']} answerable, "
            f"{metrics['unanswerable_count']} unanswerable)."
        ),
        "",
        "| Metric | Baseline | Improved |",
        "|---|---:|---:|",
    ]

    for name, baseline_value, improved_value in rows:
        baseline_cell = "—" if baseline_value is None else f"{baseline_value:.3f}"
        improved_cell = "—" if improved_value is None else f"{improved_value:.3f}"
        lines.append(f"| {name} | {baseline_cell} | {improved_cell} |")

    lines.extend(
        [
            "",
            "## Data-quality issues detected",
            "",
        ]
    )

    for issue in report["data_quality_issues"]:
        lines.append(
            f"- **{issue['kind']}** severity={issue['severity']} "
            f"({', '.join(issue['doc_ids'])}): "
            f"{issue['description']} Policy: {issue['policy']}"
        )

    lines.extend(["", "## Per-case results", ""])
    lines.append("| ID | Answerable | Baseline docs | Improved status | Improved docs | Fact/abstention OK |")
    lines.append("|---|---:|---|---|---|---:|")

    for row in report["cases"]:
        ok = bool(row["fact_coverage"]) if row["answerable"] else bool(row["improved_abstained"])
        lines.append(
            f"| {row['id']} | {row['answerable']} | {', '.join(row['baseline_docs'])} | "
            f"{row['improved_status']} | {', '.join(row['improved_docs'][:3])} | {ok} |"
        )

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate baseline and improved local RAG retrieval.")
    parser.add_argument("--corpus", default="data/corpus.jsonl")
    parser.add_argument("--eval", default="data/eval_set.jsonl")
    parser.add_argument("--out", default="reports/eval_report.json")
    parser.add_argument("--markdown", default="reports/eval_report.md")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    report = evaluate(args.corpus, args.eval)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    markdown = Path(args.markdown)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(report, markdown)

    print(json.dumps(report["metrics"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
