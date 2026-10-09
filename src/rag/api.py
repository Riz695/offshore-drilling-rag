from functools import lru_cache
from pathlib import Path
from typing import Protocol

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag.bm25_index import BM25Index
from rag.hybrid import HybridRetriever
from rag.llm import LLMError, OllamaLLM
from rag.models import RetrievedChunk
from rag.prompt import REFUSAL_MESSAGE, SYSTEM_PROMPT, build_prompt
from rag.vector_store import VectorStore, make_hf_embed_fn

ROOT = Path(__file__).resolve().parents[2]
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EXCERPT_CHARS = 300

app = FastAPI(title="Offshore Drilling RAG")


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_n: int = Field(default=5, ge=1, le=20)


class Source(BaseModel):
    source_file: str
    page_number: int
    chunk_id: str
    excerpt: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


class RetrieverLike(Protocol):
    def retrieve(self, query: str, top_n: int = 5) -> list[RetrievedChunk]: ...


class LLMLike(Protocol):
    def generate(self, system: str, user: str) -> str: ...


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    store = VectorStore(ROOT / "data" / "chroma", make_hf_embed_fn(EMBEDDING_MODEL))
    bm25 = BM25Index.load(ROOT / "data" / "bm25.pkl")
    return HybridRetriever(store, bm25)


@lru_cache(maxsize=1)
def get_llm() -> OllamaLLM:
    return OllamaLLM()


def to_source(chunk: RetrievedChunk) -> Source:
    return Source(
        source_file=chunk.source_file,
        page_number=chunk.page_number,
        chunk_id=chunk.chunk_id,
        excerpt=chunk.text[:EXCERPT_CHARS],
        score=chunk.score,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(
    req: QueryRequest,
    retriever: RetrieverLike = Depends(get_retriever),
    llm: LLMLike = Depends(get_llm),
) -> QueryResponse:
    chunks = retriever.retrieve(req.question, top_n=req.top_n)
    if not chunks:
        return QueryResponse(answer=REFUSAL_MESSAGE, sources=[])
    try:
        answer = llm.generate(SYSTEM_PROMPT, build_prompt(req.question, chunks))
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return QueryResponse(answer=answer, sources=[to_source(c) for c in chunks])
