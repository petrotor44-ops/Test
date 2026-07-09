from __future__ import annotations

import re
from dataclasses import dataclass

from ..utils.data_handling import Document
from ..utils.text_handling import normalize

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    index_text: str


def _preview(text: str, *, max_chars: int = 160) -> str:
    """Return a single-line preview suitable for debug logs."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def split_sentences(text: str) -> list[str]:
    text = normalize(text)
    if not text:
        return []
    return [part.strip() for part in _SENTENCE_BOUNDARY_RE.split(text) if part.strip()]


def sentence_chunks(
    docs: list[Document],
    *,
    max_chars: int = 900,
    overlap_sentences: int = 1,
    debug: bool = False,
    debug_preview_chars: int = 160,
) -> list[Chunk]:
    """Pack whole sentences into chunks while preserving titles.

    The supplied documents are short, so this usually returns one chunk per
    document. Unlike the baseline, it avoids cutting units, codes, or procedures
    in the middle of a sentence and makes the title available to retrieval.
    """
    chunks: list[Chunk] = []

    if debug:
        print(
            "[chunking] sentence_chunks started "
            f"docs={len(docs)} max_chars={max_chars} "
            f"overlap_sentences={overlap_sentences} "
            f"debug_preview_chars={debug_preview_chars}"
        )

    for doc in docs:
        sentences = split_sentences(doc.text)

        if debug:
            print(f"[chunking] doc={doc.id} title={doc.title!r} sentences={len(sentences)}")

        if not sentences:
            if debug:
                print(f"[chunking] doc={doc.id} skipped: no sentences")
            continue

        current: list[str] = []
        chunk_index = 0

        for sent in sentences:
            candidate = " ".join(current + [sent]).strip()

            if current and len(candidate) > max_chars:
                text = " ".join(current).strip()
                chunk = Chunk(
                    chunk_id=f"{doc.id}:{chunk_index}",
                    doc_id=doc.id,
                    title=doc.title,
                    text=text,
                    index_text=f"{doc.title}. {text}",
                )
                chunks.append(chunk)

                if debug:
                    print(
                        f"[chunking] emitted chunk={chunk.chunk_id} "
                        f"chars={len(chunk.text)} sentences={len(current)} "
                        f"preview={_preview(chunk.text, max_chars=debug_preview_chars)!r}"
                    )

                chunk_index += 1
                current = current[-overlap_sentences:] if overlap_sentences else []

                if debug and current:
                    print(
                        f"[chunking] doc={doc.id} "
                        f"carried_overlap_sentences={len(current)} "
                        f"overlap_preview={_preview(' '.join(current), max_chars=debug_preview_chars)!r}"
                    )

            current.append(sent)

        if current:
            text = " ".join(current).strip()
            chunk = Chunk(
                chunk_id=f"{doc.id}:{chunk_index}",
                doc_id=doc.id,
                title=doc.title,
                text=text,
                index_text=f"{doc.title}. {text}",
            )
            chunks.append(chunk)

            if debug:
                print(
                    f"[chunking] emitted chunk={chunk.chunk_id} "
                    f"chars={len(chunk.text)} sentences={len(current)} "
                    f"preview={_preview(chunk.text, max_chars=debug_preview_chars)!r}"
                )

    if debug:
        print(f"[chunking] sentence_chunks finished total_chunks={len(chunks)}")

    return chunks


def fixed_char_chunks(
    docs: list[Document],
    *,
    size: int = 400,
    debug: bool = False,
    debug_preview_chars: int = 160,
) -> list[Chunk]:
    """Replicate the baseline's fixed-size character windows for comparison."""
    chunks: list[Chunk] = []

    if debug:
        print(
            "[chunking] fixed_char_chunks started "
            f"docs={len(docs)} size={size} debug_preview_chars={debug_preview_chars}"
        )

    for doc in docs:
        doc_chunk_count = 0

        for idx, start in enumerate(range(0, len(doc.text), size)):
            text = doc.text[start : start + size]
            chunk = Chunk(
                chunk_id=f"{doc.id}:{idx}",
                doc_id=doc.id,
                title=doc.title,
                text=text,
                index_text=text,  # baseline does not index title metadata
            )
            chunks.append(chunk)
            doc_chunk_count += 1

            if debug:
                print(
                    f"[chunking] emitted chunk={chunk.chunk_id} "
                    f"start={start} chars={len(chunk.text)} "
                    f"preview={_preview(chunk.text, max_chars=debug_preview_chars)!r}"
                )

        if debug:
            print(f"[chunking] doc={doc.id} fixed_char_chunks={doc_chunk_count}")

    if debug:
        print(f"[chunking] fixed_char_chunks finished total_chunks={len(chunks)}")

    return chunks
