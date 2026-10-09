from pathlib import Path

import pytest

from rag.bm25_index import BM25Index, tokenize
from rag.models import Chunk


def make_chunk(text: str, page: int, index: int = 0, source: str = "manual.pdf") -> Chunk:
    return Chunk(
        text=text,
        source_file=source,
        page_number=page,
        chunk_id=f"{source}::p{page}::c{index}",
    )


@pytest.fixture
def chunks() -> list[Chunk]:
    return [
        make_chunk("The BOP stack must be pressure tested weekly", page=3),
        make_chunk("Riser tension limits depend on water depth", page=7),
        make_chunk("Crane lifts require a banksman and a lift plan", page=12),
        make_chunk("Mud weight is checked every shift by the mud engineer", page=15),
    ]


@pytest.fixture
def index(chunks: list[Chunk]) -> BM25Index:
    return BM25Index.build(chunks)


def test_tokenize_lowercases() -> None:
    assert tokenize("BOP Stack") == ["bop", "stack"]


def test_tokenize_keeps_acronyms_with_digits() -> None:
    assert "h2s" in tokenize("Detect H2S gas.")


def test_tokenize_keeps_hyphenated_codes() -> None:
    assert "rp-54" in tokenize("See API RP-54 for details")


def test_tokenize_drops_punctuation() -> None:
    assert tokenize("pressure, tested; (weekly)") == ["pressure", "tested", "weekly"]


def test_search_ranks_keyword_match_first(index: BM25Index) -> None:
    results = index.search("BOP", k=3)
    assert results[0].page_number == 3


def test_search_returns_full_metadata(index: BM25Index) -> None:
    top = index.search("riser", k=1)[0]
    assert top.source_file == "manual.pdf"
    assert top.page_number == 7
    assert top.chunk_id == "manual.pdf::p7::c0"
    assert "Riser tension" in top.text
    assert top.score > 0


def test_search_respects_k(index: BM25Index) -> None:
    assert len(index.search("the", k=1)) <= 1


def test_search_skips_chunks_with_no_match(index: BM25Index) -> None:
    assert index.search("helicopter", k=5) == []


def test_search_on_empty_index_returns_empty() -> None:
    assert BM25Index.build([]).search("bop", k=5) == []


def test_save_load_roundtrip_keeps_results(index: BM25Index, tmp_path: Path) -> None:
    path = tmp_path / "bm25.pkl"
    index.save(path)
    loaded = BM25Index.load(path)
    assert loaded.search("mud engineer", k=3) == index.search("mud engineer", k=3)
