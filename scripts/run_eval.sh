#!/usr/bin/env bash
set -euo pipefail
python -m RAG.evaluate --corpus data/corpus.jsonl --eval data/eval_set.jsonl --out reports/eval_report.json --markdown reports/eval_report.md
