from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunking import Chunk, fixed_char_chunks, sentence_chunks
from ..utils.data_handling import Document
from ..utils.text_handling import expand_query, extract_entities, salient_terms, tokenize, jaccard_similarity

EQUIPMENT_TERM_GROUPS = (
    {"compressor", "compressors"},
    {"pump", "pumps"},
    {"motor", "motors"},
    {"fan", "fans"},
    {"sensor", "sensors", "probe", "probes"},
    {"bearing", "bearings"},
    {"spare", "spares", "part", "parts"},
)

CHUNK_SIZE = 400
EMBED_MODEL = "all-MiniLM-L6-v2"


@dataclass(frozen=True)
class RetrievalHit:
    chunk: Chunk
    score: float
    component_scores: dict[str, float]

    @property
    def doc_id(self) -> str:
        return self.chunk.doc_id

    @property
    def title(self) -> str:
        return self.chunk.title

    @property
    def text(self) -> str:
        return self.chunk.text
  

  class BaselineRetriever:
    def __init__(
        self,
        docs: list[Document],
        *,
        chunk_size: int = CHUNK_SIZE,
        embed_model: str = EMBED_MODEL,
    ) -> None:
        self.chunks = fixed_char_chunks(docs, size=chunk_size)
        self.model = SentenceTransformer(embed_model)

        vectors = self.model.encode([chunk.text for chunk in self.chunks])
        vectors = np.asarray(vectors, dtype="float32")
        self.vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def retrieve(self, query: str, *, top_k: int = 1) -> list[RetrievalHit]:
        q = self.model.encode([query])[0].astype("float32")
        q = q / np.linalg.norm(q)

        scores = self.vectors @ q
        order = np.argsort(-scores)[:top_k]

        return [
            RetrievalHit(
                chunk=self.chunks[int(i)],
                score=float(scores[int(i)]),
                component_scores={"embedding_cosine": float(scores[int(i)])},
            )
            for i in order
        ]

class BM25:
    """Thin wrapper around rank-bm25's BM25Okapi.

    Using a standard package keeps the retrieval component recognizable while
    preserving our project-level API and deterministic tokenization.
    """

    def __init__(self, corpus_tokens: list[list[str]]) -> None:
        self.corpus_tokens = corpus_tokens
        self.model = BM25Okapi(corpus_tokens)

    def score(self, query_tokens: list[str]) -> np.ndarray:
        if not query_tokens or not self.corpus_tokens:
            return np.zeros(len(self.corpus_tokens), dtype="float64")
        return np.asarray(self.model.get_scores(query_tokens), dtype="float64")

class HybridRetriever:
    """Offline hybrid retriever with lexical, character, BM25, and exact-token signals."""

    def __init__(
        self,
        docs: list[Document],
        *,
        max_chunk_chars: int = 900,
        bm25_weight: float = 0.45,
        word_weight: float = 0.35,
        char_weight: float = 0.20,
    ) -> None:
        self.chunks = sentence_chunks(docs, max_chars=max_chunk_chars)
        self.bm25_weight = bm25_weight
        self.word_weight = word_weight
        self.char_weight = char_weight

        self.word_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b[a-zA-Z0-9][a-zA-Z0-9\-/\.]*\b",
            sublinear_tf=True,
        )
        self.char_vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
        )
        texts = [chunk.index_text for chunk in self.chunks]
        self.word_matrix = self.word_vectorizer.fit_transform(texts)
        self.char_matrix = self.char_vectorizer.fit_transform(texts)
        self.bm25 = BM25([tokenize(text, expand=False) for text in texts])

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievalHit]:
        expanded_query = expand_query(query)
        word_q = self.word_vectorizer.transform([expanded_query])
        char_q = self.char_vectorizer.transform([expanded_query])
        word_scores = cosine_similarity(word_q, self.word_matrix).ravel()
        char_scores = cosine_similarity(char_q, self.char_matrix).ravel()
        bm25_scores = self.bm25.score(tokenize(expanded_query, expand=False))
        bm25_scores = _safe_minmax(bm25_scores)

        base_scores = (
            self.bm25_weight * bm25_scores
            + self.word_weight * word_scores
            + self.char_weight * char_scores
        )
        final_scores = base_scores.copy()

        query_entities = extract_entities(query)
        query_terms = salient_terms(query)
        for i, chunk in enumerate(self.chunks):
            text = f"{chunk.title} {chunk.text}"
            text_lower = text.lower()
            chunk_entities = set(extract_entities(text))
            chunk_terms = salient_terms(text)

            if query_entities:
                matches = sum(1 for entity in query_entities if entity in chunk_entities or entity.lower() in text_lower)
                if matches:
                    final_scores[i] += 0.10 * matches / len(query_entities)
                else:
                    # Exact equipment/code mismatch is costly in industrial QA.
                    final_scores[i] -= 0.18

            title_terms = salient_terms(chunk.title)
            if query_terms and title_terms:
                final_scores[i] += 0.05 * jaccard_similarity(query_terms, title_terms)

            # Equipment-type agreement matters. A compressor maintenance query should not
            # rank a pump-maintenance document above a compressor-interval document.
            for group in EQUIPMENT_TERM_GROUPS:
                if query_terms & group:
                    if chunk_terms & group:
                        final_scores[i] += 0.06
                    else:
                        final_scores[i] -= 0.12

        order = np.argsort(-final_scores)[:top_k]
        hits: list[RetrievalHit] = []
        for idx in order:
            i = int(idx)
            hits.append(
                RetrievalHit(
                    chunk=self.chunks[i],
                    score=float(final_scores[i]),
                    component_scores={
                        "bm25": float(bm25_scores[i]),
                        "word_tfidf": float(word_scores[i]),
                        "char_tfidf": float(char_scores[i]),
                        "base": float(base_scores[i]),
                    },
                )
            )
        return hits
