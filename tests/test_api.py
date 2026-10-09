from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from rag.api import app, get_llm, get_retriever
from rag.llm import LLMError
from rag.models import RetrievedChunk
from rag.prompt import REFUSAL_MESSAGE


def rc(chunk_id: str, page: int = 3) -> RetrievedChunk:
    return RetrievedChunk(
        text=f"text of {chunk_id}",
        source_file="manual.pdf",
        page_number=page,
        chunk_id=chunk_id,
        score=0.8,
    )


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.last_top_n: int | None = None

    def retrieve(self, query: str, top_n: int = 5) -> list[RetrievedChunk]:
        self.last_top_n = top_n
        return self.chunks[:top_n]


class FakeLLM:
    def __init__(self, answer: str = "A BOP seals the well [manual.pdf p.3]") -> None:
        self.answer = answer
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.answer


class FailingLLM:
    def generate(self, system: str, user: str) -> str:
        raise LLMError("ollama is not running")


@pytest.fixture
def client() -> Iterator[TestClient]:
    yield TestClient(app)
    app.dependency_overrides.clear()


def use(retriever: object, llm: object) -> None:
    app.dependency_overrides[get_retriever] = lambda: retriever
    app.dependency_overrides[get_llm] = lambda: llm


def test_health(client: TestClient) -> None:
    assert client.get("/health").status_code == 200


def test_query_returns_answer_and_sources(client: TestClient) -> None:
    use(FakeRetriever([rc("a"), rc("b", page=4)]), FakeLLM())
    resp = client.post("/query", json={"question": "What is a BOP?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "A BOP seals the well [manual.pdf p.3]"
    assert len(body["sources"]) == 2


def test_sources_have_required_fields(client: TestClient) -> None:
    use(FakeRetriever([rc("a")]), FakeLLM())
    source = client.post("/query", json={"question": "q"}).json()["sources"][0]
    assert source["source_file"] == "manual.pdf"
    assert source["page_number"] == 3
    assert source["chunk_id"] == "a"
    assert "text of a" in source["excerpt"]
    assert source["score"] == pytest.approx(0.8)


def test_sources_come_from_retrieval_not_llm(client: TestClient) -> None:
    use(FakeRetriever([rc("a")]), FakeLLM(answer="see [fake.pdf p.99]"))
    sources = client.post("/query", json={"question": "q"}).json()["sources"]
    assert [s["chunk_id"] for s in sources] == ["a"]


def test_prompt_sent_to_llm_contains_chunks_and_question(client: TestClient) -> None:
    llm = FakeLLM()
    use(FakeRetriever([rc("a")]), llm)
    client.post("/query", json={"question": "What is a BOP?"})
    system, user = llm.calls[0]
    assert REFUSAL_MESSAGE in system
    assert "text of a" in user
    assert "What is a BOP?" in user


def test_top_n_is_passed_to_retriever(client: TestClient) -> None:
    retriever = FakeRetriever([rc("a"), rc("b"), rc("c")])
    use(retriever, FakeLLM())
    resp = client.post("/query", json={"question": "q", "top_n": 2})
    assert retriever.last_top_n == 2
    assert len(resp.json()["sources"]) == 2


def test_empty_question_is_422(client: TestClient) -> None:
    use(FakeRetriever([rc("a")]), FakeLLM())
    assert client.post("/query", json={"question": ""}).status_code == 422


def test_missing_question_is_422(client: TestClient) -> None:
    use(FakeRetriever([rc("a")]), FakeLLM())
    assert client.post("/query", json={}).status_code == 422


def test_invalid_top_n_is_422(client: TestClient) -> None:
    use(FakeRetriever([rc("a")]), FakeLLM())
    resp = client.post("/query", json={"question": "q", "top_n": 0})
    assert resp.status_code == 422


def test_no_chunks_returns_refusal_without_calling_llm(client: TestClient) -> None:
    llm = FakeLLM()
    use(FakeRetriever([]), llm)
    resp = client.post("/query", json={"question": "capital of France?"})
    assert resp.status_code == 200
    assert resp.json() == {"answer": REFUSAL_MESSAGE, "sources": []}
    assert llm.calls == []


def test_llm_failure_is_503_with_message(client: TestClient) -> None:
    use(FakeRetriever([rc("a")]), FailingLLM())
    resp = client.post("/query", json={"question": "q"})
    assert resp.status_code == 503
    assert "ollama is not running" in resp.json()["detail"]
