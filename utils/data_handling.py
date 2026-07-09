from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .text_handling import attribute_key, canonical_for_similarity, contains_vague_value_claim, extract_entities, extract_measurements, jaccard_similarity, split_sentences, tokenize


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str


class IssueKind(Enum):
    CONFLICT = "conflict"
    NEAR_DUPLICATE = "near_duplicate"
    UNDERSPECIFIED_VALUE = "underspecified_value"

    @property
    def policy(self) -> str:
        """The remediation/response policy for this issue type."""
        return {
            IssueKind.CONFLICT: (
                "Do not silently choose one value or claim when relevant sources disagree. "
                "Surface the disagreement, cite the source IDs for each distinct value or claim, "
                "and mark the answer as conflicting unless an explicit, domain-approved precedence rule resolves it."
            ),
            IssueKind.NEAR_DUPLICATE: (
                "Treat substantially similar records as redundant or corroborating evidence rather than fully "
                "independent facts. Deduplicate them during scoring or aggregation, preserve provenance, "
                "and cite multiple copies only when it improves auditability or user trust."
            ),
            IssueKind.UNDERSPECIFIED_VALUE: (
                "Do not infer missing exact values, units, thresholds, dates, intervals, or parameters from "
                "underspecified text. Answer only the portion directly supported by the documents, and abstain "
                "or ask for clarification when the requested detail is not explicitly provided."
            ),
        }[self]

    @property
    def severity(self) -> int:
        """Higher number = more severe."""
        return {
            IssueKind.CONFLICT: 3,
            IssueKind.UNDERSPECIFIED_VALUE: 2,
            IssueKind.NEAR_DUPLICATE: 1,
        }[self]


@dataclass(frozen=True)
class DataIssue:
    kind: IssueKind
    doc_ids: tuple[str, ...]
    description: str

    @property
    def policy(self) -> str:
        """Derived remediation/response policy for this issue."""
        return self.kind.policy

    @property
    def severity(self) -> int:
        """Derived severity for this issue."""
        return self.kind.severity


def load_docs(path: str | Path) -> list[Document]:
    docs: list[Document] = []
    with Path(path).open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = {"id", "title", "text"} - set(row)
            if missing:
                raise ValueError(f"Line {line_number} is missing required keys: {sorted(missing)}")
            docs.append(Document(id=str(row["id"]), title=str(row["title"]), text=str(row["text"])))
    return docs


def detect_data_quality_issues(docs: list[Document]) -> list[DataIssue]:
    """Detect data-quality issues that can affect grounded QA.

    This detector avoids corpus-specific document IDs and instead uses general,
    reproducible heuristics: pairwise token similarity for near duplicates,
    normalized measurement extraction for conflicts, and vague-value pattern
    detection for underspecified values.

    The implementation is intentionally deterministic for offline reproducibility.
    On larger or less structured corpora, these heuristics should be treated as
    candidate generation and followed by stricter schema validation, offline
    contradiction detection, an offline LLM/NLI judge, or human review for
    high-impact cases.
    """
    issues: list[DataIssue] = []

    # 1) Conflicts: same normalized entity + attribute + unit, different values.
    measurements: dict[tuple[str, str, str], dict[str, set[str]]] = {}

    for doc in docs:
        doc_text = f"{doc.title}. {doc.text}"
        doc_entities = extract_entities(doc_text)

        for sentence in split_sentences(doc_text):
            sentence_entities = extract_entities(sentence) or doc_entities
            if not sentence_entities:
                continue

            for value, unit, _raw_text, start_offset in extract_measurements(sentence):
                attribute = attribute_key(sentence, start_offset)
                if not attribute:
                    continue

                for entity in sentence_entities:
                    key = (entity, attribute, unit)
                    measurements.setdefault(key, {}).setdefault(value, set()).add(doc.id)

    for (entity, attribute, unit), values_by_doc in sorted(measurements.items()):
        if len(values_by_doc) <= 1:
            continue

        doc_ids = tuple(sorted({doc_id for ids in values_by_doc.values() for doc_id in ids}))
        value_summary = ", ".join(
            f"{value} {unit} in {', '.join(sorted(doc_ids_for_value))}"
            for value, doc_ids_for_value in sorted(values_by_doc.items())
        )

        issues.append(
            DataIssue(
                kind=IssueKind.CONFLICT,
                doc_ids=doc_ids,
                description=(
                    f"{entity} has conflicting documented values for '{attribute}': "
                    f"{value_summary}."
                ),
            )
        )

    # 2) Near duplicates: high token overlap after normalizing entities and numbers.
    seen_duplicate_pairs: set[tuple[str, str]] = set()

    for left_index, left_doc in enumerate(docs):
        left_text = f"{left_doc.title} {left_doc.text}"
        left_tokens = set(tokenize(canonical_for_similarity(left_text), expand=False))
        left_entities = set(extract_entities(left_text))

        if not left_tokens:
            continue

        for right_doc in docs[left_index + 1 :]:
            right_text = f"{right_doc.title} {right_doc.text}"
            right_tokens = set(tokenize(canonical_for_similarity(right_text), expand=False))
            right_entities = set(extract_entities(right_text))

            if not right_tokens:
                continue

            if left_entities and right_entities and left_entities.isdisjoint(right_entities):
                continue

            overlap = jaccard_similarity(left_tokens, right_tokens)
            if overlap < 0.30:
                continue

            pair = tuple(sorted((left_doc.id, right_doc.id)))
            if pair in seen_duplicate_pairs:
                continue

            seen_duplicate_pairs.add(pair)
            issues.append(
                DataIssue(
                    kind=IssueKind.NEAR_DUPLICATE,
                    doc_ids=pair,
                    description=(
                        f"Documents {pair[0]} and {pair[1]} have high normalized token overlap "
                        f"({overlap:.2f}) and appear to describe the same or substantially similar fact set."
                    ),
                )
            )

    # 3) Underspecified values: value-bearing vague claims without concrete numeric detail.
    for doc in docs:
        for sentence in split_sentences(doc.text):
            if contains_vague_value_claim(sentence):
                issues.append(
                    DataIssue(
                        kind=IssueKind.UNDERSPECIFIED_VALUE,
                        doc_ids=(doc.id,),
                        description=(
                            "The document contains a value-bearing but underspecified statement: "
                            f'"{sentence}"'
                        ),
                    )
                )
                break

    return sorted(
        issues,
        key=lambda issue: (-issue.severity, issue.kind.value, issue.doc_ids, issue.description),
    )
