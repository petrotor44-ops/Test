from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..utils.answer_handling import EvidenceRow, attribute_is_missing, coverage_too_low, evidence_to_answer, has_required_entities, pressure_conflict_candidate, select_evidence_sentences
from .retrievers import HybridRetriever, RetrievalHit

ABSTAIN_MESSAGE = "The answer was not found in the provided documents!"

class AnswerStatus(Enum):
    ANSWERED = "answered"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"


@dataclass(frozen=True)
class AnswerResult:
    query: str
    status: AnswerStatus
    answer: str
    hits: list[RetrievalHit] = field(default_factory=list)
    evidence: list[EvidenceRow] = field(default_factory=list)

    @property
    def abstained(self) -> bool:
        return self.status is AnswerStatus.NOT_FOUND


def _not_found(
    query: str,
    *,
    hits: list[RetrievalHit] | None = None,
    evidence: list[EvidenceRow] | None = None,
) -> AnswerResult:
    return AnswerResult(
        query=query,
        status=AnswerStatus.NOT_FOUND,
        answer=ABSTAIN_MESSAGE,
        hits=hits or [],
        evidence=evidence or [],
    )


def _build_conflict_answer(query: str, hits: list[RetrievalHit]) -> AnswerResult | None:
    conflict = pressure_conflict_candidate(query, hits)
    if conflict is None:
        return None

    answer = (
        "Conflicting information was found in the provided documents. "
        "Do not use a single pressure value without resolving the source-of-truth issue. "
        + " ".join(conflict.answer_parts)
    )

    return AnswerResult(
        query=query,
        status=AnswerStatus.CONFLICT,
        answer=answer,
        hits=hits,
        evidence=conflict.evidence,
    )


def answer_question(query: str, retriever: HybridRetriever, *, top_k: int = 5) -> AnswerResult:
    hits = retriever.retrieve(query, top_k=top_k)

    if not hits:
        return _not_found(query)

    # Strong safety gate: equipment/code questions must retrieve the exact token.
    if not has_required_entities(query, hits):
        return _not_found(query, hits=hits)

    conflict = _build_conflict_answer(query, hits)
    if conflict is not None:
        return conflict

    evidence = select_evidence_sentences(query, hits)

    if attribute_is_missing(query, evidence, hits) or coverage_too_low(query, evidence, hits):
        return _not_found(query, hits=hits, evidence=evidence)

    if not evidence:
        return _not_found(query, hits=hits)

    return AnswerResult(
        query=query,
        status=AnswerStatus.ANSWERED,
        answer=evidence_to_answer(evidence),
        hits=hits,
        evidence=evidence,
    )
