from dataclasses import replace
from typing import Protocol

from rag.models import RetrievedChunk


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[RetrievedChunk]: ...


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]], k: int = 60, top_n: int = 5
) -> list[RetrievedChunk]:
    scores: dict[str, float] = {}
    first_seen: dict[str, RetrievedChunk] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1 / (k + rank)
            first_seen.setdefault(chunk.chunk_id, chunk)
    order = sorted(scores, key=lambda cid: (-scores[cid], cid))
    return [replace(first_seen[cid], score=scores[cid]) for cid in order[:top_n]]


class HybridRetriever:
    def __init__(self, dense: Retriever, bm25: Retriever) -> None:
        self._dense = dense
        self._bm25 = bm25

    def retrieve(
        self, query: str, k_each: int = 20, top_n: int = 5
    ) -> list[RetrievedChunk]:
        dense_results = self._dense.search(query, k_each)
        ranked_lists = [dense_results, self._bm25.search(query, k_each)]
        dense_scores = {c.chunk_id: c.score for c in dense_results}
        fused = reciprocal_rank_fusion(ranked_lists, top_n=top_n)
        return [
            replace(c, dense_score=dense_scores.get(c.chunk_id)) for c in fused
        ]
