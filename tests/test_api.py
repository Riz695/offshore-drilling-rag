from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from rag.api import MIN_DENSE_SCORE, app, get_llm, get_retriever
from rag.llm import LLMError
from rag.models import RetrievedChunk
from rag.prompt import REFUSAL_MESSAGE


def rc(
    chunk_id: str, page: int = 3, dense_score: float | None = 0.8
) -> RetrievedChunk:
    return RetrievedChunk(
        text=f"text of {chunk_id}",
        source_file="manual.pdf",
        page_number=page,
        chunk_id=chunk_id,
        score=0.8,
        dense_score=dense_score,
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


def test_low_dense_score_refuses_without_calling_llm(client: TestClient) -> None:
    llm = FakeLLM()
    weak = rc("a", dense_score=MIN_DENSE_SCORE - 0.01)
    use(FakeRetriever([weak]), llm)
    resp = client.post("/query", json={"question": "capital of France?"})
    assert resp.status_code == 200
    assert resp.json() == {"answer": REFUSAL_MESSAGE, "sources": []}
    assert llm.calls == []


def test_dense_score_at_threshold_is_answered(client: TestClient) -> None:
    llm = FakeLLM()
    use(FakeRetriever([rc("a", dense_score=MIN_DENSE_SCORE)]), llm)
    resp = client.post("/query", json={"question": "q"})
    assert len(llm.calls) == 1
    assert resp.json()["answer"] == llm.answer


def test_one_strong_chunk_is_enough(client: TestClient) -> None:
    llm = FakeLLM()
    chunks = [rc("a", dense_score=0.1), rc("b", dense_score=0.9)]
    use(FakeRetriever(chunks), llm)
    resp = client.post("/query", json={"question": "q"})
    assert len(llm.calls) == 1
    assert len(resp.json()["sources"]) == 2


def test_no_dense_scores_at_all_refuses(client: TestClient) -> None:
    llm = FakeLLM()
    use(FakeRetriever([rc("a", dense_score=None)]), llm)
    resp = client.post("/query", json={"question": "q"})
    assert resp.json()["answer"] == REFUSAL_MESSAGE
    assert llm.calls == []


# --- code-side citation check -------------------------------------------------


def ask(client: TestClient, answer: str, chunks: list[RetrievedChunk]) -> dict:
    use(FakeRetriever(chunks), FakeLLM(answer))
    return client.post("/query", json={"question": "q"}).json()


def test_valid_citation_passes_through(client: TestClient) -> None:
    body = ask(client, "Seals the well [manual.pdf p.3].", [rc("a", page=3)])
    assert body["answer"] == "Seals the well [manual.pdf p.3]."
    assert len(body["sources"]) == 1


def test_stray_bracket_before_answer_still_passes(client: TestClient) -> None:
    answer = "[BOP stands for Blowout Preventer [manual.pdf p.3]."
    assert ask(client, answer, [rc("a", page=3)])["answer"] == answer


def test_citing_a_page_that_was_not_retrieved_is_refused(client: TestClient) -> None:
    body = ask(client, "Seals the well [manual.pdf p.99].", [rc("a", page=3)])
    assert body == {"answer": REFUSAL_MESSAGE, "sources": []}


def test_citing_a_file_that_was_not_retrieved_is_refused(client: TestClient) -> None:
    body = ask(client, "Seals the well [other.pdf p.3].", [rc("a", page=3)])
    assert body == {"answer": REFUSAL_MESSAGE, "sources": []}


def test_one_bad_citation_among_good_ones_is_refused(client: TestClient) -> None:
    answer = "A [manual.pdf p.3]. B [manual.pdf p.99]."
    body = ask(client, answer, [rc("a", page=3)])
    assert body["answer"] == REFUSAL_MESSAGE


def test_answer_without_any_citation_is_refused(client: TestClient) -> None:
    body = ask(client, "A BOP seals the well.", [rc("a", page=3)])
    assert body == {"answer": REFUSAL_MESSAGE, "sources": []}


def test_model_refusal_with_trailing_period_returns_clean_refusal(
    client: TestClient,
) -> None:
    body = ask(client, REFUSAL_MESSAGE + ".", [rc("a", page=3)])
    assert body == {"answer": REFUSAL_MESSAGE, "sources": []}


def test_numeric_citation_is_not_accepted(client: TestClient) -> None:
    body = ask(client, "Seals the well [1].", [rc("a", page=3)])
    assert body["answer"] == REFUSAL_MESSAGE
