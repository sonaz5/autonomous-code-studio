"""
Phase 3 -- Hybrid RAG: Dense + BM25 Retrieval with Reranking

Goal: search a codebase for chunks relevant to a query, combining two very
different retrieval signals:
  - BM25 (sparse/lexical): great when the query shares exact words/identifiers
    with the code ("ADMIN_LIST", "cache").
  - Dense embeddings (semantic): great when the query describes a *concept*
    that doesn't share vocabulary with the code ("function that leaks state
    between calls" should still find a mutable-default-argument bug even if
    the code never uses the word "leaks").
Neither alone is reliable. BM25 misses paraphrases; dense embeddings miss
exact identifier matches buried in noise. We fuse both rankings with
Reciprocal Rank Fusion (RRF), then use a cross-encoder to rerank the fused
candidates with a more expensive, more accurate pairwise (query, chunk) score.

Deliberate choice: embeddings and reranking run on small, well-established,
LOCAL open-source models (sentence-transformers) instead of a hosted API.
No API key, no rate limit, and -- after three rounds of hosted model names
breaking this project -- no risk of a retrieval model disappearing under you
mid-project. These specific model names have been stable community defaults
for years.

First run downloads ~200MB of model weights (cached afterward, offline from
then on).

Usage:
    python phase3_hybrid_rag.py <folder-of-py-files> "<natural language query>"

Example (search your own project's earlier phases):
    python phase3_hybrid_rag.py . "function that keeps state between calls by accident"
"""

import ast
import glob
import re
import sys

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---- Chunking ---------------------------------------------------------------


def chunk_python_file(filepath: str) -> list[dict]:
    """Split a Python file into function/class-level chunks using the AST,
    instead of a fixed-size character window that could cut a function in
    half. Nested functions/methods each become their own chunk too, so a
    class chunk and its methods' chunks will overlap -- fine for a demo,
    a production system would pick one granularity and dedupe."""
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError as exc:
        print(f"  skipping {filepath}: {exc}", file=sys.stderr)
        return []

    lines = source.splitlines()
    chunks = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno - 1
            end = getattr(node, "end_lineno", start + 1)
            text = "\n".join(lines[start:end])
            chunks.append(
                {
                    "id": f"{filepath}:{node.name}:{node.lineno}",
                    "file": filepath,
                    "name": node.name,
                    "type": type(node).__name__,
                    "start_line": node.lineno,
                    "text": text,
                }
            )
    return chunks


def simple_tokenize(text: str) -> list[str]:
    """Crude but workable: pull out identifier-like words, lowercase them.
    Real code-search tokenizers also split camelCase/snake_case into parts
    (getUserPermissions -> get, user, permissions) for better lexical
    matching -- worth adding if BM25 recall feels weak on your codebase."""
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text.lower())


# ---- Index --------------------------------------------------------------


class HybridIndex:
    def __init__(self, embedding_model_name: str = EMBEDDING_MODEL):
        self.embedder = SentenceTransformer(embedding_model_name)
        self.chunks: list[dict] = []
        self.bm25: BM25Okapi | None = None
        self.embeddings: np.ndarray | None = None

    def build(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        tokenized = [simple_tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        texts = [c["text"] for c in chunks]
        self.embeddings = self.embedder.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )

    def search_bm25(self, query: str, top_k: int) -> list[int]:
        scores = self.bm25.get_scores(simple_tokenize(query))
        return list(np.argsort(scores)[::-1][:top_k])

    def search_dense(self, query: str, top_k: int) -> list[int]:
        q_emb = self.embedder.encode([query], normalize_embeddings=True)[0]
        sims = self.embeddings @ q_emb
        return list(np.argsort(sims)[::-1][:top_k])


# ---- Fusion + Reranking ---------------------------------------------------


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Combine multiple ranked lists (each a list of chunk indices, best
    first) into one fused ranking.

    RRF score(d) = sum over rankers of 1 / (k + rank_of_d_in_that_ranker)

    A chunk missing from a given ranker just contributes nothing from that
    ranker -- no need to normalize BM25 scores and cosine similarities onto
    the same scale first, which is RRF's main advantage over averaging raw
    scores directly.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


class Reranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[tuple[dict, float]]:
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)
        order = np.argsort(scores)[::-1][:top_k]
        return [(candidates[i], float(scores[i])) for i in order]


def hybrid_search(
    index: HybridIndex,
    reranker: Reranker,
    query: str,
    retrieve_k: int = 15,
    final_k: int = 5,
) -> list[tuple[dict, float]]:
    bm25_ranking = index.search_bm25(query, top_k=retrieve_k)
    dense_ranking = index.search_dense(query, top_k=retrieve_k)

    fused = reciprocal_rank_fusion([bm25_ranking, dense_ranking])
    candidate_indices = [i for i, _ in fused[:retrieve_k]]
    candidates = [index.chunks[i] for i in candidate_indices]

    return reranker.rerank(query, candidates, top_k=final_k)


# ---- CLI --------------------------------------------------------------------

if __name__ == "__main__":
    debug = "--debug" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--debug"]

    if len(args) < 2:
        print('Usage: python phase3_hybrid_rag.py <folder-of-py-files> "<query>" [--debug]')
        sys.exit(1)

    folder = args[0]
    query = args[1]

    py_files = glob.glob(f"{folder}/**/*.py", recursive=True)
    if not py_files:
        print(f"No .py files found under {folder}")
        sys.exit(1)

    print(f"Chunking {len(py_files)} file(s)...")
    all_chunks: list[dict] = []
    for path in py_files:
        all_chunks.extend(chunk_python_file(path))
    print(f"Got {len(all_chunks)} function/class chunks.\n")

    if not all_chunks:
        print("No chunks extracted -- nothing to search.")
        sys.exit(1)

    print("Loading embedding model (first run downloads it, then it's cached)...")
    index = HybridIndex()
    index.build(all_chunks)

    print("Loading reranker...")
    reranker = Reranker()

    def describe(i: int) -> str:
        c = index.chunks[i]
        return f"{c['type']} {c['name']} ({c['file']}:{c['start_line']})"

    if debug:
        print(f"\n{'=' * 60}\nDEBUG: per-stage rankings for: {query}\n{'=' * 60}")

        bm25_ranking = index.search_bm25(query, top_k=10)
        print("\n[BM25 -- lexical only] top 10:")
        if all(index.bm25.get_scores(simple_tokenize(query))[i] == 0 for i in bm25_ranking):
            print("  (all scores are 0 -- no shared vocabulary at all with the query)")
        for rank, i in enumerate(bm25_ranking, 1):
            print(f"  {rank}. {describe(i)}")

        dense_ranking = index.search_dense(query, top_k=10)
        print("\n[Dense -- semantic only] top 10:")
        for rank, i in enumerate(dense_ranking, 1):
            print(f"  {rank}. {describe(i)}")

        fused = reciprocal_rank_fusion([bm25_ranking, dense_ranking])
        print("\n[Fused -- RRF combination of the two above] top 10:")
        for rank, (i, score) in enumerate(fused[:10], 1):
            print(f"  {rank}. score={score:.4f}  {describe(i)}")

        print(f"\n[Reranked -- cross-encoder re-scores the fused candidates] final:")

    print(f"\nQuery: {query}\n{'-' * 60}")
    results = hybrid_search(index, reranker, query)

    for rank, (chunk, score) in enumerate(results, 1):
        print(f"#{rank}  score={score:.3f}  {chunk['type']} {chunk['name']}  ({chunk['file']}:{chunk['start_line']})")
        first_line = chunk["text"].strip().splitlines()[0]
        print(f"      {first_line}")
    print()