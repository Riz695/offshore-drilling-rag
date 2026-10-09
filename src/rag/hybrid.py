from dataclasses import replace

from rag.models import RetrievedChunk


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
