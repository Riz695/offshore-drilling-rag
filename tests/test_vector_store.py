from pathlib import Path

import pytest

from rag.models import Chunk, RetrievedChunk
from rag.vector_store import VectorStore, make_hf_embed_fn

VOCAB: list[str] = ["bop", "riser", "crane", "mud", "safety"]


def fake_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic bag-of-words vectors; the 0.01 keeps every vector non-zero."""
    vectors: list[list[float]] = []
    for text in texts:
        words = text.lower().split()
        vectors.append([words.count(term) + 0.01 for term in VOCAB])
    return vectors


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
        make_chunk("bop bop test pressure", page=3),
        make_chunk("riser riser tension limits", page=7),
        make_chunk("crane crane lift safety", page=12),
    ]


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    return VectorStore(persist_dir=tmp_path / "chroma", embed_fn=fake_embed)


def test_search_returns_retrieved_chunks_with_metadata_intact(
    store: VectorStore, chunks: list[Chunk]
) -> None:
    store.add(chunks)
    top = store.search("bop", k=1)[0]

    assert isinstance(top, RetrievedChunk)
    assert top.text == "bop bop test pressure"
    assert top.source_file == "manual.pdf"
    assert top.page_number == 3
    assert top.chunk_id == "manual.pdf::p3::c0"


def test_search_ranks_most_similar_first_with_descending_scores(
    store: VectorStore, chunks: list[Chunk]
) -> None:
    store.add(chunks)
    results = store.search("riser", k=3)

    assert results[0].chunk_id == "manual.pdf::p7::c0"
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_respects_k(store: VectorStore, chunks: list[Chunk]) -> None:
    store.add(chunks)
    assert len(store.search("crane", k=2)) == 2


def test_search_never_returns_more_than_stored(store: VectorStore, chunks: list[Chunk]) -> None:
    store.add(chunks)
    assert len(store.search("crane", k=50)) == 3


def test_adding_the_same_chunks_twice_does_not_duplicate(
    store: VectorStore, chunks: list[Chunk]
) -> None:
    store.add(chunks)
    store.add(chunks)
    assert store.count() == 3


def test_adding_a_changed_chunk_overwrites_its_text(store: VectorStore) -> None:
    store.add([make_chunk("mud mud pumps", page=1)])
    store.add([make_chunk("mud mud pumps updated", page=1)])

    assert store.count() == 1
    assert store.search("mud", k=1)[0].text == "mud mud pumps updated"


def test_search_on_empty_store_returns_empty_list(store: VectorStore) -> None:
    assert store.search("bop", k=5) == []


def test_add_empty_list_is_a_noop(store: VectorStore) -> None:
    store.add([])
    assert store.count() == 0


def test_add_handles_more_chunks_than_one_batch(store: VectorStore) -> None:
    many = [make_chunk(f"mud chunk {i}", page=i) for i in range(250)]
    store.add(many)
    assert store.count() == 250


def test_data_persists_on_disk_between_instances(tmp_path: Path, chunks: list[Chunk]) -> None:
    path = tmp_path / "chroma"
    VectorStore(persist_dir=path, embed_fn=fake_embed).add(chunks)

    reopened = VectorStore(persist_dir=path, embed_fn=fake_embed)
    assert reopened.count() == 3
    assert reopened.search("crane", k=1)[0].page_number == 12


@pytest.mark.integration
def test_real_embeddings_rank_a_semantically_related_chunk_first(tmp_path: Path) -> None:
    store = VectorStore(
        persist_dir=tmp_path / "chroma",
        embed_fn=make_hf_embed_fn("BAAI/bge-small-en-v1.5"),
    )
    store.add(
        [
            make_chunk("The blowout preventer stack seals the wellbore during a kick.", page=1),
            make_chunk("Crew members must wear life jackets during helicopter transfers.", page=2),
            make_chunk("Drilling mud density is adjusted to balance formation pressure.", page=3),
        ]
    )

    top = store.search("well control equipment", k=1)[0]
    assert top.page_number == 1
