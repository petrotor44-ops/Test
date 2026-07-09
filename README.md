## Baseline diagnosis

The supplied `baseline_rag.py` is useful as a starter, but its retrieval and answer trustworthiness are weak for industrial QA.

1. **Fixed character windows**: `chunk_text()` splits text into 400-character windows without respecting sentences, units, procedures, or document boundaries. The current corpus is short enough that this is not catastrophic, but the design will cut longer industrial procedures mid-step.
2. **Single-hit retrieval**: `retrieve()` returns only the highest-scoring chunk. It cannot compare corroborating documents, expose alternatives, or recover when the top result is a near miss.
3. **No re-ranking or exact-token handling**: equipment IDs and fault codes such as `C-100`, `M-50`, `BRG-4410`, and `E-207` should be treated as high-precision constraints. The baseline treats them as ordinary embedding text.
4. **No abstention**: `answer()` always returns a chunk. For questions such as “What oil type should be used for the C-100 compressor?”, the baseline will still return the C-100 specification chunk even though the oil type is absent.
5. **No conflict or duplicate policy**: the corpus contains conflicting P-200 pressure values and near-duplicate F-30 vibration-limit documents. A single top chunk can silently hide this.
6. **Title metadata is not indexed**: the baseline embeds only chunk text. Important equipment context in titles, such as `Compressor C-100 — Specifications` or `Spare Parts — Ordering Process`, receives no direct retrieval signal.
