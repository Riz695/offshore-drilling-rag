from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import requests
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")

SOURCE = {
    "source_file": "bop_manual.pdf",
    "page_number": 12,
    "chunk_id": "bop_manual.pdf-p12-c0",
    "excerpt": "Test the BOP to 5000 psi every 14 days.",
    "score": 0.03,
}
OK_BODY = {
    "answer": "Test every 14 days [bop_manual.pdf p.12].",
    "sources": [SOURCE],
}


def fake_response(status: int, body: dict[str, Any]) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = body
    resp.ok = status < 400
    return resp


def new_app() -> AppTest:
    return AppTest.from_file(APP_PATH, default_timeout=10).run()


def ask(at: AppTest, question: str) -> AppTest:
    at.text_input[0].set_value(question)
    at.button[0].click()
    return at.run()


def test_initial_page_makes_no_backend_call() -> None:
    with patch("requests.post") as post:
        at = new_app()
    post.assert_not_called()
    assert not at.exception
    assert len(at.text_input) == 1
    assert len(at.button) == 1
    assert len(at.expander) == 0


def test_ask_posts_question_to_query_endpoint() -> None:
    with patch("requests.post", return_value=fake_response(200, OK_BODY)) as post:
        ask(new_app(), "How often is the BOP tested?")
    post.assert_called_once()
    assert post.call_args.args[0].endswith("/query")
    assert post.call_args.kwargs["json"]["question"] == "How often is the BOP tested?"


def test_answer_is_displayed() -> None:
    with patch("requests.post", return_value=fake_response(200, OK_BODY)):
        at = ask(new_app(), "How often is the BOP tested?")
    assert not at.exception
    assert any(OK_BODY["answer"] in m.value for m in at.markdown)


def test_each_source_is_an_expander_with_file_page_and_excerpt() -> None:
    with patch("requests.post", return_value=fake_response(200, OK_BODY)):
        at = ask(new_app(), "How often is the BOP tested?")
    assert len(at.expander) == 1
    assert at.expander[0].label == "bop_manual.pdf, p.12"
    assert any(SOURCE["excerpt"] in m.value for m in at.expander[0].markdown)


def test_empty_sources_shows_answer_and_no_expanders() -> None:
    body = {"answer": "Not found in the manuals.", "sources": []}
    with patch("requests.post", return_value=fake_response(200, body)):
        at = ask(new_app(), "What is the capital of France?")
    assert any("Not found in the manuals." in m.value for m in at.markdown)
    assert len(at.expander) == 0


def test_blank_question_does_not_call_backend() -> None:
    with patch("requests.post") as post:
        ask(new_app(), "   ")
    post.assert_not_called()


def test_503_shows_error_with_detail() -> None:
    resp = fake_response(503, {"detail": "Ollama is not running"})
    with patch("requests.post", return_value=resp):
        at = ask(new_app(), "How often is the BOP tested?")
    assert len(at.error) == 1
    assert "Ollama is not running" in at.error[0].value
    assert len(at.expander) == 0


def test_connection_error_shows_error() -> None:
    with patch("requests.post", side_effect=requests.ConnectionError("refused")):
        at = ask(new_app(), "How often is the BOP tested?")
    assert not at.exception
    assert len(at.error) == 1


def test_result_survives_rerun() -> None:
    with patch("requests.post", return_value=fake_response(200, OK_BODY)) as post:
        at = ask(new_app(), "How often is the BOP tested?")
        at.run()
    assert post.call_count == 1
    assert any(OK_BODY["answer"] in m.value for m in at.markdown)
    assert len(at.expander) == 1
