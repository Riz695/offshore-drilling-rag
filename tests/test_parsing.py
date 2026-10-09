from pathlib import Path

import pytest

from conftest import RAW_MANUALS_DIR
from rag.parsing import parse_pdf, table_to_markdown


def test_table_to_markdown_builds_header_separator_and_rows() -> None:
    rows = [["Zone", "Class"], ["1", "A"], ["2", "B"]]
    assert table_to_markdown(rows) == (
        "| Zone | Class |\n"
        "| --- | --- |\n"
        "| 1 | A |\n"
        "| 2 | B |"
    )


def test_table_to_markdown_turns_none_cells_into_empty_strings() -> None:
    rows = [["Zone", "Class"], ["1", None]]
    assert table_to_markdown(rows).splitlines()[2] == "| 1 |  |"


def test_table_to_markdown_escapes_pipes_inside_cells() -> None:
    rows = [["Item"], ["a|b"]]
    assert table_to_markdown(rows).splitlines()[2] == r"| a\|b |"


def test_table_to_markdown_flattens_newlines_inside_cells() -> None:
    rows = [["Item"], ["line one\nline two"]]
    assert table_to_markdown(rows).splitlines()[2] == "| line one line two |"


def test_table_to_markdown_returns_empty_string_for_no_rows() -> None:
    assert table_to_markdown([]) == ""


@pytest.mark.integration
def test_parse_pdf_returns_numbered_pages_with_a_markdown_table() -> None:
    pdf_path: Path = RAW_MANUALS_DIR / "cds-guide-feb21.pdf"
    pages = parse_pdf(pdf_path)

    assert len(pages) > 0
    assert pages[0][0] == 1  # page numbers are 1-based
    assert all(isinstance(text, str) for _, text in pages)
    assert any(text.strip() for _, text in pages)
    assert any("| --- |" in text for _, text in pages)
