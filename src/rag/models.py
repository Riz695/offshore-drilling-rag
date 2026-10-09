from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    text: str
    source_file: str
    page_number: int
    chunk_id: str
