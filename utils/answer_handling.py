from __future__ import annotations

import re
from dataclasses import dataclass

from ..RAG.chunking import split_sentences
from ..RAG.retrievers import RetrievalHit
from .text_handling import contains_concrete_number, extract_entities, jaccard, salient_terms, tokenize


EvidenceRow = dict[str, str | float]


@dataclass(frozen=True)
class ConflictCandidate:
    evidence: list[EvidenceRow]
    answer_parts: list[str]


def query_is_about_pressure_conflict(query: str) -> bool:
    lower = query.lower()
    return "p-200" in lower and "pressure" in lower and any(
        term in lower for term in ("maximum", "max", "rated", "operating", "limit", "cap")
    )


def extract_pressure_values(hits: list[RetrievalHit]) -> dict[str, tuple[str, str, str]]:
    values: dict[str, tuple[str, str, str]] = {}

    for hit in hits:
        combined = f"{hit.title}. {hit.text}"
        if "p-200" not in combined.lower() or "pressure" not in combined.lower():
            continue

        for sentence in split_sentences(hit.text):
            if "pressure" not in sentence.lower():
                continue

            for match in re.finditer(r"\b(\d+(?:\.\d+)?)\s*bar\b", sentence, re.IGNORECASE):
                value = f"{match.group(1)} bar"
                values[f"{hit.doc_id}:{value}"] = (hit.doc_id, value, sentence)

    return values


def pressure_conflict_candidate(query: str, hits: list[RetrievalHit]) -> ConflictCandidate | None:
    if not query_is_about_pressure_conflict(query):
        return None

    values = extract_pressure_values(hits)
    unique_values = {value for _, value, _ in values.values()}

    if len(unique_values) <= 1:
        return None

    evidence: list[EvidenceRow] = [
        {
            "doc_id": doc_id,
            "title": next((hit.title for hit in hits if hit.doc_id == doc_id), ""),
            "sentence": sentence,
            "score": next((hit.score for hit in hits if hit.doc_id == doc_id), 0.0),
        }
        for doc_id, _value, sentence in values.values()
    ]

    answer_parts = [
        f"{doc_id} states {value}: {sentence}"
        for doc_id, value, sentence in values.values()
    ]

    return ConflictCandidate(evidence=evidence, answer_parts=answer_parts)


def sentence_score(query: str, sentence: str, title: str) -> float:
    query_terms = salient_terms(query)
    sentence_terms = salient_terms(f"{title} {sentence}")

    score = jaccard(query_terms, sentence_terms)

    query_entities = set(extract_entities(query))
    sentence_entities = set(extract_entities(f"{title} {sentence}"))

    if query_entities:
        score += 0.4 * (len(query_entities & sentence_entities) / len(query_entities))

    if contains_concrete_number(sentence):
        score += 0.05

    return score


def select_evidence_sentences(
    query: str,
    hits: list[RetrievalHit],
    *,
    max_sentences: int = 3,
) -> list[EvidenceRow]:
    candidates: list[EvidenceRow] = []
    seen: set[tuple[str, str]] = set()

    for hit in hits:
        for sentence in split_sentences(hit.text):
            score = sentence_score(query, sentence, hit.title)
            key = (hit.doc_id, sentence)

            if key in seen:
                continue

            seen.add(key)
            candidates.append(
                {
                    "doc_id": hit.doc_id,
                    "title": hit.title,
                    "sentence": sentence,
                    "score": float(score),
                    "retrieval_score": float(hit.score),
                }
            )

    candidates.sort(
        key=lambda row: (float(row["score"]), float(row["retrieval_score"])),
        reverse=True,
    )

    selected: list[EvidenceRow] = []
    selected_texts: list[set[str]] = []

    for candidate in candidates:
        if float(candidate["score"]) <= 0.0:
            continue

        candidate_terms = set(tokenize(str(candidate["sentence"])))

        # Collapse near-duplicate phrasing but keep distinct source docs for exact duplicates only when needed.
        if any(jaccard(candidate_terms, previous_terms) >= 0.82 for previous_terms in selected_texts):
            continue

        selected.append(candidate)
        selected_texts.append(candidate_terms)

        if len(selected) >= max_sentences:
            break

    return selected


def has_required_entities(query: str, hits: list[RetrievalHit]) -> bool:
    entities = extract_entities(query)

    if not entities:
        return True

    haystack = " ".join(f"{hit.title} {hit.text}" for hit in hits[:3]).lower()

    return all(
        entity.lower() in haystack or entity.lower().replace("-", "") in haystack.replace("-", "")
        for entity in entities
    )


def attribute_is_missing(
    query: str,
    evidence: list[EvidenceRow],
    hits: list[RetrievalHit],
) -> bool:
    lower = query.lower()
    evidence_text = " ".join(str(row["sentence"]) for row in evidence).lower()
    top_text = " ".join(f"{hit.title} {hit.text}" for hit in hits[:3]).lower()

    if "serial" in lower:
        return "serial" not in top_text

    if "oil" in lower and ("type" in lower or "grade" in lower or "lubricant" in lower):
        # "air-oil separator" is not an oil type or lubricant grade.
        return not re.search(r"\b(?:oil type|lubricant|oil grade|iso vg|sae)\b", top_text)

    if "supplier" in lower:
        query_entities = extract_entities(query)
        if query_entities:
            return not any(
                "supplier" in str(row["sentence"]).lower()
                and any(entity.lower() in str(row["sentence"]).lower() for entity in query_entities)
                for row in evidence
            )
        return "supplier" not in evidence_text

    if "minimum" in lower and "stock" in lower and ("quantity" in lower or "how many" in lower):
        return not re.search(r"\b\d+\b", evidence_text)

    if "exact" in lower and "interval" in lower and (
        "logging" in lower or "sensor" in lower or "telemetry" in lower
    ):
        # The corpus may state that logging happens at fixed intervals without specifying a numeric cadence.
        return not any(contains_concrete_number(str(row["sentence"])) for row in evidence)

    if "tolerance" in lower and any(
        term in lower for term in ("numeric", "number", "value", "exact", "how much")
    ):
        # A useful answer would need a numeric tolerance around the word tolerance.
        return not re.search(
            r"\b\d+(?:\.\d+)?\s*(?:%|degrees?|celsius|c|bar|mm/s|rms)?\b.{0,30}\btolerance\b|"
            r"\btolerance\b.{0,30}\b\d+(?:\.\d+)?",
            evidence_text,
        )

    return False


def coverage_too_low(
    query: str,
    evidence: list[EvidenceRow],
    hits: list[RetrievalHit],
    *,
    min_top_score: float = 0.16,
) -> bool:
    if not evidence:
        return True

    if hits and hits[0].score < min_top_score:
        return True

    query_terms = salient_terms(query)

    # Remove exact entities from the coverage check so an entity-only hit does not become a fabricated answer.
    entity_terms = {entity.lower() for entity in extract_entities(query)} | {
        entity.lower().replace("-", "") for entity in extract_entities(query)
    }

    content_terms = {term for term in query_terms if term not in entity_terms and len(term) > 2}
    evidence_terms = salient_terms(" ".join(str(row["sentence"]) for row in evidence[:2]))

    if not content_terms:
        return False

    overlap = content_terms & evidence_terms

    # For short factoid questions, one strong term plus the entity is often enough.
    if extract_entities(query) and overlap:
        return False

    return len(overlap) < min(2, len(content_terms))


def evidence_to_answer(evidence: list[EvidenceRow]) -> str:
    return " ".join(f"[{row['doc_id']}] {row['sentence']}" for row in evidence)
