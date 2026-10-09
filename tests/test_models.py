from rag.models import RetrievedChunk


def test_dense_score_defaults_to_none() -> None:
    chunk = RetrievedChunk(
        text="t", source_file="m.pdf", page_number=1, chunk_id="a", score=0.5
    )
    assert chunk.dense_score is None
