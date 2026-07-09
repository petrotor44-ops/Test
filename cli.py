"""Command-line interface for querying the offline RAG pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils.data_handling import load_docs
from RAG.retrievers import HybridRetriever
from RAG.answering import answer_question


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask a question over the local corpus.")
    parser.add_argument("question", help="Question to answer from the corpus")
    parser.add_argument("--corpus", default="data/corpus.jsonl", help="Path to corpus JSONL")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved chunks to inspect")
    parser.add_argument("--show-hits", action="store_true", help="Print retrieval hits and scores")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    docs = load_docs(Path(args.corpus))
    retriever = HybridRetriever(docs)
    result = answer_question(args.question, retriever, top_k=args.top_k)
    print(result.answer)
    print(f"status: {result.status}")
    if args.show_hits:
        print("\nretrieval hits:")
        for hit in result.hits:
            print(f"- {hit.doc_id} | score={hit.score:.4f} | {hit.title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
