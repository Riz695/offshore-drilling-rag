import pytest

from rag.hybrid import HybridRetriever, reciprocal_rank_fusion
from rag.models import RetrievedChunk


def rc(chunk_id: str, page: int = 1, score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        text=f"text of {chunk_id}",
        source_file="manual.pdf",
        page_number=page,
        chunk_id=chunk_id,
        score=score,
    )


def test_rrf_rank_one_in_both_lists_scores_2_over_61() -> None:
    fused = reciprocal_rank_fusion([[rc("a")], [rc("a")]], k=60, top_n=5)
    assert fused[0].chunk_id == "a"
    assert fused[0].score == pytest.approx(2 / 61)


def test_rrf_hand_computed_scores() -> None:
    fused = reciprocal_rank_fusion(
        [[rc("a"), rc("b")], [rc("b"), rc("c")]], k=60, top_n=5
    )
    scores = {c.chunk_id: c.score for c in fused}
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["c"] == pytest.approx(1 / 62)


def test_rrf_orders_by_fused_score() -> None:
    fused = reciprocal_rank_fusion(
        [[rc("a"), rc("b")], [rc("b"), rc("c")]], k=60, top_n=5
    )
    assert [c.chunk_id for c in fused] == ["b", "a", "c"]


def test_rrf_doc_in_one_list_still_appears() -> None:
    fused = reciprocal_rank_fusion([[rc("a")], [rc("z")]], k=60, top_n=5)
    assert {c.chunk_id for c in fused} == {"a", "z"}


def test_rrf_ties_break_by_chunk_id() -> None:
    first = reciprocal_rank_fusion([[rc("b")], [rc("a")]], k=60, top_n=5)
    second = reciprocal_rank_fusion([[rc("a")], [rc("b")]], k=60, top_n=5)
    assert [c.chunk_id for c in first] == ["a", "b"]
    assert [c.chunk_id for c in second] == ["a", "b"]


def test_rrf_truncates_to_top_n() -> None:
    fused = reciprocal_rank_fusion([[rc("a"), rc("b"), rc("c")]], k=60, top_n=2)
    assert [c.chunk_id for c in fused] == ["a", "b"]


def test_rrf_keeps_metadata() -> None:
    fused = reciprocal_rank_fusion([[rc("a", page=9)]], k=60, top_n=5)
    assert fused[0].page_number == 9
    assert fused[0].source_file == "manual.pdf"
    assert fused[0].text == "text of a"


def test_rrf_empty_input_returns_empty() -> None:
    assert reciprocal_rank_fusion([[], []], k=60, top_n=5) == []


class StubRetriever:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self._results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        self.calls.append((query, k))
        return self._results[:k]


def test_hybrid_fuses_and_deduplicates() -> None:
    dense = StubRetriever([rc("a"), rc("b")])
    bm25 = StubRetriever([rc("b"), rc("c")])
    fused = HybridRetriever(dense, bm25).retrieve("q", k_each=20, top_n=5)
    ids = [c.chunk_id for c in fused]
    assert ids == ["b", "a", "c"]
    assert len(ids) == len(set(ids))


def test_hybrid_keeps_metadata() -> None:
    dense = StubRetriever([rc("a", page=4)])
    bm25 = StubRetriever([])
    fused = HybridRetriever(dense, bm25).retrieve("q")
    assert fused[0].page_number == 4
    assert fused[0].source_file == "manual.pdf"


def test_hybrid_passes_query_and_k_each_to_both() -> None:
    dense, bm25 = StubRetriever([]), StubRetriever([])
    HybridRetriever(dense, bm25).retrieve("BOP test", k_each=7, top_n=3)
    assert dense.calls == [("BOP test", 7)]
    assert bm25.calls == [("BOP test", 7)]


def test_hybrid_respects_top_n() -> None:
    dense = StubRetriever([rc("a"), rc("b"), rc("c")])
    bm25 = StubRetriever([rc("d")])
    assert len(HybridRetriever(dense, bm25).retrieve("q", top_n=2)) == 2
