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

MEASUREMENT_RE = re.compile(
    r"\b(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>"
    r"bar|m3/h|m3/min|kw|mm/s(?:\s+rms)?|rms|"
    r"operating\s+hours|hours?|months?|years?|days?|minutes?|seconds?|"
    r"degrees\s+celsius|celsius"
    r")\b",
    re.IGNORECASE,
)

VAGUE_VALUE_RE = re.compile(
    r"\b("
    r"fixed intervals?|regular intervals?|periodic(?:ally)?|routine(?:ly)?|"
    r"as needed|within tolerance|within range|within rating|"
    r"below (?:the )?limit|above (?:the )?limit|minimum stock|baseline readings|"
    r"per (?:the )?equipment sheet|according to (?:the )?datasheet"
    r")\b",
    re.IGNORECASE,
)

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


def split_sentences(text: str) -> list[str]:
    """Split normalized technical prose into simple sentence-like units."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", normalize(text)) if s.strip()]


def normalize_unit(unit: str) -> str:
    """Normalize common technical/time units for comparison."""
    unit = normalize(unit).lower().strip()
    return {
        "kw": "kW",
        "bar": "bar",
        "m3/h": "m3/h",
        "m3/min": "m3/min",
        "mm/s": "mm/s",
        "mm/s rms": "mm/s RMS",
        "rms": "RMS",
        "hour": "hour",
        "hours": "hour",
        "operating hour": "operating hour",
        "operating hours": "operating hour",
        "month": "month",
        "months": "month",
        "year": "year",
        "years": "year",
        "day": "day",
        "days": "day",
        "minute": "minute",
        "minutes": "minute",
        "second": "second",
        "seconds": "second",
        "degree celsius": "degree Celsius",
        "degrees celsius": "degree Celsius",
        "celsius": "degree Celsius",
    }.get(unit, unit)


def normalize_number(value: str) -> str:
    """Normalize numeric strings so 12 and 12.0 compare equal."""
    number = float(value)
    return str(int(number)) if number.is_integer() else str(number)


def canonical_for_similarity(text: str) -> str:
    """Normalize text for duplicate detection while hiding IDs and exact numbers."""
    text = normalize(text).lower()
    text = re.sub(r"\b\d+(?:\.\d+)?\b", "<num>", text)
    text = ENTITY_RE.sub("<entity>", text)
    return text


def attribute_key(sentence: str, value_start: int, *, window: int = 4) -> str:
    """Build a compact attribute key from the terms immediately before a value."""
    prefix = sentence[:value_start]
    prefix = ENTITY_RE.sub(" ", prefix)
    prefix = re.sub(r"\b\d+(?:\.\d+)?\b", " ", prefix)
    tokens = tokenize(prefix, expand=True, keep_stopwords=False)
    return " ".join(tokens[-window:])


def extract_measurements(sentence: str) -> list[tuple[str, str, str, int]]:
    """Extract normalized measurements as value, unit, raw matched text, start offset."""
    measurements: list[tuple[str, str, str, int]] = []
    for match in MEASUREMENT_RE.finditer(sentence):
        value = normalize_number(match.group("value"))
        unit = normalize_unit(match.group("unit"))
        measurements.append((value, unit, match.group(0), match.start()))
    return measurements


def contains_vague_value_claim(text: str) -> bool:
    """True when text implies a value/threshold/cadence but does not state one concretely."""
    return bool(VAGUE_VALUE_RE.search(text)) and not contains_concrete_number(text)


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    """Return Jaccard similarity for two token sets."""
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
