import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from rag.models import Chunk, RetrievedChunk

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class BM25Index:
    def __init__(self, chunks: list[Chunk], bm25: BM25Okapi | None) -> None:
        self._chunks = chunks
        self._bm25 = bm25

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "BM25Index":
        if not chunks:
            return cls([], None)
        return cls(chunks, BM25Okapi([tokenize(c.text) for c in chunks]))

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [
            RetrievedChunk(
                text=self._chunks[i].text,
                source_file=self._chunks[i].source_file,
                page_number=self._chunks[i].page_number,
                chunk_id=self._chunks[i].chunk_id,
                score=float(scores[i]),
            )
            for i in ranked[:k]
            if scores[i] > 0
        ]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump((self._chunks, self._bm25), f)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        with path.open("rb") as f:
            chunks, bm25 = pickle.load(f)
        return cls(chunks, bm25)
