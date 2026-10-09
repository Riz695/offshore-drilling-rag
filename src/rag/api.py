import re
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
# Observed cosine: ~0.49 off-topic, 0.73-0.86 on-topic. Tune on real queries.
MIN_DENSE_SCORE = 0.6

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
        excerpt=chunk.text,
        score=chunk.score,
    )


def is_relevant(chunks: list[RetrievedChunk]) -> bool:
    return any(
        c.dense_score is not None and c.dense_score >= MIN_DENSE_SCORE for c in chunks
    )


CITATION_RE = re.compile(r"\[([^\[\]]+?) p\.(\d+)\]")


def citations_valid(answer: str, chunks: list[RetrievedChunk]) -> bool:
    """True if the answer cites at least once and every cite is a retrieved chunk."""
    cited = {(f, int(p)) for f, p in CITATION_RE.findall(answer)}
    allowed = {(c.source_file, c.page_number) for c in chunks}
    return bool(cited) and cited <= allowed


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
    if not is_relevant(chunks):
        return QueryResponse(answer=REFUSAL_MESSAGE, sources=[])
    try:
        answer = llm.generate(SYSTEM_PROMPT, build_prompt(req.question, chunks))
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if REFUSAL_MESSAGE in answer or not citations_valid(answer, chunks):
        return QueryResponse(answer=REFUSAL_MESSAGE, sources=[])
    return QueryResponse(answer=answer, sources=[to_source(c) for c in chunks])
