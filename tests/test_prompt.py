from rag.models import RetrievedChunk
from rag.prompt import REFUSAL_MESSAGE, SYSTEM_PROMPT, build_prompt


def rc(chunk_id: str, source: str, page: int, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        text=text, source_file=source, page_number=page, chunk_id=chunk_id, score=0.5
    )


CHUNKS = [
    rc("a", "bop_manual.pdf", 12, "The BOP stack seals the wellbore."),
    rc("b", "safety.pdf", 61, "H2S alarms trigger at 10 ppm."),
]


def test_prompt_contains_every_chunk_text() -> None:
    prompt = build_prompt("What is a BOP?", CHUNKS)
    assert "The BOP stack seals the wellbore." in prompt
    assert "H2S alarms trigger at 10 ppm." in prompt


def test_prompt_labels_each_chunk_with_the_exact_citation_string() -> None:
    prompt = build_prompt("What is a BOP?", CHUNKS)
    assert "[bop_manual.pdf p.12]\nThe BOP stack" in prompt
    assert "[safety.pdf p.61]\nH2S alarms" in prompt


def test_prompt_has_no_numbered_labels_the_model_could_copy() -> None:
    prompt = build_prompt("What is a BOP?", CHUNKS)
    assert "[1]" not in prompt
    assert "(bop_manual.pdf, p.12)" not in prompt


def test_prompt_contains_the_question() -> None:
    assert "What is a BOP?" in build_prompt("What is a BOP?", CHUNKS)


def test_prompt_keeps_chunk_order() -> None:
    prompt = build_prompt("q", CHUNKS)
    assert prompt.index("bop_manual.pdf") < prompt.index("safety.pdf")


def test_system_prompt_demands_grounding_and_citations() -> None:
    lowered = SYSTEM_PROMPT.lower()
    assert "only" in lowered
    assert "cite" in lowered
    assert "[file p.n]" in lowered


def test_system_prompt_contains_refusal_phrase() -> None:
    assert REFUSAL_MESSAGE in SYSTEM_PROMPT


def test_refusal_message_wording() -> None:
    assert REFUSAL_MESSAGE == "The provided manuals do not contain this information"


def test_system_prompt_shows_a_worked_citation_example_and_forbids_numbers() -> None:
    assert "[example.pdf p.7]" in SYSTEM_PROMPT
    assert "[1]" in SYSTEM_PROMPT  # named as a forbidden form
    assert "never" in SYSTEM_PROMPT.lower()
