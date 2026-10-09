from pathlib import Path

import pytest

from conftest import RAW_MANUALS_DIR
from rag.parsing import is_real_table, parse_pdf, strip_repeated_lines, table_to_markdown


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


def test_is_real_table_accepts_multi_column_table_with_content() -> None:
    assert is_real_table([["Zone", "Class"], ["1", "A"]])


def test_is_real_table_rejects_single_column_title_box() -> None:
    assert not is_real_table([[""], ["Section 1.0 Prerequisite Material 4/68"], [""]])


def test_is_real_table_rejects_table_with_only_empty_cells() -> None:
    assert not is_real_table([["", None], [None, ""]])


def test_is_real_table_rejects_no_rows() -> None:
    assert not is_real_table([])


def _pages(lines_per_page: list[list[str]]) -> list[tuple[int, str]]:
    return [(i, "\n".join(lines)) for i, lines in enumerate(lines_per_page, start=1)]


def test_strip_repeated_lines_removes_header_found_on_most_pages() -> None:
    header = "Weatherford confidential header"
    pages = _pages([[header, f"body {i}"] for i in range(10)])
    result = strip_repeated_lines(pages)
    assert all(header not in text for _, text in result)
    assert [text for _, text in result] == [f"body {i}" for i in range(10)]


def test_strip_repeated_lines_keeps_page_numbers() -> None:
    pages = _pages([["header", f"body {i}"] for i in range(10)])
    assert [n for n, _ in strip_repeated_lines(pages)] == list(range(1, 11))


def test_strip_repeated_lines_keeps_lines_that_are_not_widespread() -> None:
    pages = _pages([["header", f"body {i}"] for i in range(10)])
    pages[3] = (4, "header\nunique warning text")
    result = dict(strip_repeated_lines(pages))
    assert result[4] == "unique warning text"
    assert "unique" not in result[1]


def test_strip_repeated_lines_leaves_short_documents_alone() -> None:
    pages = _pages([["header", "a"], ["header", "b"]])
    assert strip_repeated_lines(pages) == pages


def test_strip_repeated_lines_ignores_blank_lines() -> None:
    pages = _pages([["", f"body {i}", ""] for i in range(10)])
    assert [text for _, text in strip_repeated_lines(pages)][0].count("body") == 1


@pytest.mark.integration
def test_parse_pdf_returns_numbered_pages_with_a_markdown_table() -> None:
    pdf_path: Path = RAW_MANUALS_DIR / "cds-guide-feb21.pdf"
    pages = parse_pdf(pdf_path)

    assert len(pages) > 0
    assert pages[0][0] == 1  # page numbers are 1-based
    assert all(isinstance(text, str) for _, text in pages)
    assert any(text.strip() for _, text in pages)
    assert any("| --- |" in text for _, text in pages)
