from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    text: str
    source_file: str
    page_number: int
    chunk_id: str


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source_file: str
    page_number: int
    chunk_id: str
    score: float
    dense_score: float | None = None
