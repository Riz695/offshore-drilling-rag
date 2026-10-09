"""Ingest every PDF in a folder into the ChromaDB vector store."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag.bm25_index import BM25Index  # noqa: E402
from rag.chunking import chunk_pages  # noqa: E402
from rag.models import Chunk  # noqa: E402
from rag.parsing import parse_pdf  # noqa: E402
from rag.vector_store import VectorStore, make_hf_embed_fn  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def ingest_directory(
    input_dir: Path, store: VectorStore, bm25_path: Path | None = None
) -> dict[str, int]:
    """Parse, chunk and store every PDF; return chunks added per filename.

    If bm25_path is given, also build the BM25 index from this run's chunks.
    """
    counts: dict[str, int] = {}
    all_chunks: list[Chunk] = []
    for pdf_path in sorted(input_dir.glob("*.pdf")):
        started = time.time()
        pages = parse_pdf(pdf_path)
        chunks = chunk_pages(pages, source_file=pdf_path.name)
        store.add(chunks)
        all_chunks.extend(chunks)
        counts[pdf_path.name] = len(chunks)
        print(
            f"{pdf_path.name}: {len(pages)} pages, {len(chunks)} chunks "
            f"({time.time() - started:.1f}s)",
            flush=True,
        )
        if not chunks:
            print(
                f"WARNING: {pdf_path.name} produced 0 chunks; it may be a scanned "
                "PDF with no text layer (needs OCR) and is NOT searchable.",
                flush=True,
            )
    if bm25_path is not None:
        BM25Index.build(all_chunks).save(bm25_path)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "raw_manuals")
    parser.add_argument("--chroma-dir", type=Path, default=ROOT / "data" / "chroma")
    parser.add_argument("--bm25-path", type=Path, default=ROOT / "data" / "bm25.pkl")
    args = parser.parse_args()

    store = VectorStore(args.chroma_dir, make_hf_embed_fn(EMBEDDING_MODEL))
    counts = ingest_directory(args.input, store, args.bm25_path)
    print(f"Done: {sum(counts.values())} chunks added, {store.count()} in store.")


if __name__ == "__main__":
    main()
