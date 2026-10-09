from collections.abc import Callable
from pathlib import Path

import chromadb

from rag.models import Chunk, RetrievedChunk

EmbedFn = Callable[[list[str]], list[list[float]]]

COLLECTION_NAME = "manuals"
BATCH_SIZE = 100


def make_hf_embed_fn(model_name: str) -> EmbedFn:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def embed(texts: list[str]) -> list[list[float]]:
        return model.encode(texts, normalize_embeddings=True).tolist()

    return embed


class VectorStore:
    def __init__(self, persist_dir: Path, embed_fn: EmbedFn) -> None:
        self._embed_fn = embed_fn
        client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    def count(self) -> int:
        return self._collection.count()

    def add(self, chunks: list[Chunk]) -> None:
        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start : start + BATCH_SIZE]
            texts = [c.text for c in batch]
            self._collection.upsert(
                ids=[c.chunk_id for c in batch],
                documents=texts,
                embeddings=self._embed_fn(texts),
                metadatas=[
                    {"source_file": c.source_file, "page_number": c.page_number}
                    for c in batch
                ],
            )

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        total = self.count()
        if total == 0:
            return []
        result = self._collection.query(
            query_embeddings=self._embed_fn([query]),
            n_results=min(k, total),
        )
        return [
            RetrievedChunk(
                text=text,
                source_file=str(meta["source_file"]),
                page_number=int(meta["page_number"]),
                chunk_id=chunk_id,
                score=1.0 - distance,
            )
            for chunk_id, text, meta, distance in zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
            )
        ]
