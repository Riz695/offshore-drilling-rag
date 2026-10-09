from pathlib import Path

import pdfplumber
from pdfplumber.page import Page
from pypdf import PdfReader
from pypdf import PageObject


def _clean_cell(cell: str | None) -> str:
    if cell is None:
        return ""
    return cell.replace("\n", " ").replace("|", r"\|").strip()


def table_to_markdown(rows: list[list[str | None]]) -> str:
    if not rows:
        return ""
    cleaned = [[_clean_cell(c) for c in row] for row in rows]
    header, *body = cleaned
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def extract_page(plumber_page: Page, pypdf_page: PageObject) -> str:
    text = (pypdf_page.extract_text() or "").strip()
    tables = [table_to_markdown(t) for t in plumber_page.extract_tables()]
    return "\n\n".join([text, *tables]).strip()


def parse_pdf(path: Path, max_pages: int | None = None) -> list[tuple[int, str]]:
    reader = PdfReader(path)
    pages: list[tuple[int, str]] = []
    with pdfplumber.open(path) as pdf:
        for number, (plumber_page, pypdf_page) in enumerate(
            zip(pdf.pages, reader.pages), start=1
        ):
            if max_pages is not None and number > max_pages:
                break
            pages.append((number, extract_page(plumber_page, pypdf_page)))
    return pages
