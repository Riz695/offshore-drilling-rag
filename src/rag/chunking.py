import tiktoken

from rag.models import Chunk

_ENCODING = tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    tokens = _ENCODING.encode(text.strip())
    if not tokens:
        return []
    step = size - overlap
    chunks: list[str] = []
    for start in range(0, len(tokens), step):
        chunks.append(_ENCODING.decode(tokens[start : start + size]))
        if start + size >= len(tokens):
            break
    return chunks


def chunk_pages(
    pages: list[tuple[int, str]],
    source_file: str,
    size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page_number, text in pages:
        for index, piece in enumerate(chunk_text(text, size, overlap)):
            chunks.append(
                Chunk(
                    text=piece,
                    source_file=source_file,
                    page_number=page_number,
                    chunk_id=f"{source_file}::p{page_number}::c{index}",
                )
            )
    return chunks
