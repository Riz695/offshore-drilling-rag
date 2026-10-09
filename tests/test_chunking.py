import pytest
import tiktoken

from rag.chunking import chunk_pages, chunk_text
from rag.models import Chunk

ENCODING = tiktoken.get_encoding("cl100k_base")


def token_count(text: str) -> int:
    return len(ENCODING.encode(text))


def make_text(n_words: int) -> str:
    return " ".join(f"drill{i}" for i in range(n_words))


def test_chunk_text_returns_nothing_for_empty_or_whitespace() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_chunk_text_keeps_short_text_as_one_chunk() -> None:
    assert chunk_text("A short sentence about blowout preventers.") == [
        "A short sentence about blowout preventers."
    ]


def test_chunk_text_never_exceeds_size() -> None:
    chunks = chunk_text(make_text(2000), size=500, overlap=50)
    assert len(chunks) > 1
    assert all(token_count(c) <= 500 for c in chunks)


def test_chunk_text_overlaps_consecutive_chunks_by_exact_token_count() -> None:
    chunks = chunk_text(make_text(2000), size=500, overlap=50)
    for previous, current in zip(chunks, chunks[1:]):
        assert ENCODING.encode(previous)[-50:] == ENCODING.encode(current)[:50]


def test_chunk_text_does_not_drop_the_tail() -> None:
    text = make_text(2000)
    chunks = chunk_text(text, size=500, overlap=50)
    assert chunks[0].startswith("drill0")
    assert chunks[-1].endswith("drill1999")


def test_chunk_text_rejects_overlap_not_smaller_than_size() -> None:
    with pytest.raises(ValueError):
        chunk_text("anything", size=50, overlap=50)


def test_chunk_pages_attaches_metadata_and_deterministic_ids() -> None:
    pages = [(1, make_text(30)), (2, make_text(1200))]
    chunks = chunk_pages(pages, source_file="Drilling_Manual.pdf")

    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.source_file == "Drilling_Manual.pdf" for c in chunks)
    assert chunks[0].page_number == 1
    assert chunks[0].chunk_id == "Drilling_Manual.pdf::p1::c0"
    page_two = [c for c in chunks if c.page_number == 2]
    assert len(page_two) > 1
    assert [c.chunk_id for c in page_two] == [
        f"Drilling_Manual.pdf::p2::c{i}" for i in range(len(page_two))
    ]


def test_chunk_pages_never_mixes_text_from_two_pages() -> None:
    pages = [(1, "alpha " * 40), (2, "omega " * 40)]
    for chunk in chunk_pages(pages, source_file="x.pdf"):
        assert ("alpha" in chunk.text) != ("omega" in chunk.text)


def test_chunk_pages_skips_empty_pages() -> None:
    chunks = chunk_pages([(1, ""), (2, "   "), (3, "real content")], source_file="x.pdf")
    assert [c.page_number for c in chunks] == [3]


def test_chunk_ids_are_unique_across_a_document() -> None:
    chunks = chunk_pages([(i, make_text(900)) for i in range(1, 6)], source_file="x.pdf")
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
