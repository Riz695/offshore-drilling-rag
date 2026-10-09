from pathlib import Path


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


def parse_pdf(path: Path) -> list[tuple[int, str]]:
    raise NotImplementedError("Next step: parse_pdf")
