from __future__ import annotations

import re
from collections.abc import Iterable

ENTITY_RE = re.compile(
    r"\b(?:[A-Z]{1,5}-\d{1,5}|BRG-\d{3,6}|DN\d{1,4}|P-\d{1,5}|C-\d{1,5}|M-\d{1,5}|F-\d{1,5}|E-\d{1,5})\b",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-/\.][a-z0-9]+)*", re.IGNORECASE)
NUMBER_UNIT_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:bar|m3/h|m3/min|kw|mm/s|rms|hours?|months?|years?|days?|minutes?|seconds?|celsius)\b",
    re.IGNORECASE,
)

STOPWORDS = {"a", "an", "and", "any", "are", "as", "at", "be", "before", "between", "by", "can", "could", "do", "does", "for", "from", "has", "have", "how", "i", "in", "is", "it", "its", "me", "must", "of", "on", "or", "our",
             "should", "that", "the", "their", "there", "to", "unit", "units", "was", "we", "what", "when", "where", "which", "who", "why", "with", "without",}

# Sparse, auditable, industrial-domain query expansion
EXPANSIONS: dict[str, tuple[str, ...]] = {
    "thermal": ("temperature",),
    "probe": ("sensor", "thermometer"),
    "probes": ("sensors", "thermometer"),
    "cadence": ("interval", "schedule", "frequency"),
    "frequency": ("interval", "schedule"),
    "service": ("maintenance", "serviced"),
    "servicing": ("maintenance", "serviced"),
    "buy": ("order", "ordering", "purchase"),
    "purchase": ("order", "ordering"),
    "replacement": ("replace", "removed", "spare"),
    "parts": ("spares", "part"),
    "spare": ("part", "parts"),
    "telemetry": ("sensor", "readings", "data", "logged"),
    "message": ("data", "reading"),
    "messages": ("data", "readings"),
    "restarted": ("retried", "restart"),
    "re-sent": ("retried", "duplicates"),
    "duplicate": ("duplicates", "counted"),
    "duplicates": ("duplicate", "counted"),
    "over-pressure": ("overpressure", "pressure"),
    "overpressure": ("over-pressure", "pressure"),
    "trip": ("fault", "code"),
    "fault": ("code", "fault"),
    "threshold": ("limit",),
    "cap": ("limit", "maximum"),
    "limit": ("threshold", "maximum"),
    "proof": ("verify", "confirm", "test"),
    "prove": ("verify", "confirm", "test"),
    "cannot": ("zero", "isolate", "test"),
    "startup": ("starting", "start"),
    "start-up": ("starting", "start", "startup"),
    "energizing": ("starting", "start", "energy"),
    "output": ("rated", "capacity"),
    "capacity": ("rated", "output"),
    "rating": ("rated", "nameplate"),
    "shutdown": ("emergency", "stop"),
    "intervention": ("maintenance", "service"),
    "logged": ("record", "readings"),
}

def normalize(text: str) -> str:
    """Normalize text for matching while preserving technical tokens."""
    text = text.replace("—", "-").replace("–", "-").replace("/", " / ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def tokenize(text: str, *, expand: bool = False, keep_stopwords: bool = False) -> list[str]:
    """Tokenize text into lowercase terms, with optional domain expansion."""
    normalized = normalize(text).lower()
    tokens = TOKEN_RE.findall(normalized)
    output: list[str] = []
    for token in tokens:
        if keep_stopwords or token not in STOPWORDS:
            output.append(token)
        if "-" in token:
            compact = token.replace("-", "")
            if compact != token:
                output.append(compact)
        if "/" in token:
            output.extend(part for part in token.split("/") if part)
        if expand:
            for expansion in EXPANSIONS.get(token, ()):  # visible and deterministic
                if keep_stopwords or expansion not in STOPWORDS:
                    output.append(expansion)
    return output

def extract_entities(text: str) -> list[str]:
    """Extract exact equipment/code/entity tokens such as C-100, E-207, DN65."""
    seen: set[str] = set()
    entities: list[str] = []
    for match in ENTITY_RE.findall(text):
        entity = match.upper()
        if entity not in seen:
            seen.add(entity)
            entities.append(entity)
    return entities

def salient_terms(text: str) -> set[str]:
    """Return non-stopword query terms, including compact forms for hyphenated tokens."""
    return set(tokenize(text, expand=True, keep_stopwords=False))

def contains_concrete_number(text: str) -> bool:
    """True if text contains a number with a technical/time unit or a concrete cadence word."""
    if NUMBER_UNIT_RE.search(text):
        return True
    return bool(re.search(r"\b(?:daily|monthly|yearly|annually|weekly)\b", text, re.IGNORECASE))
