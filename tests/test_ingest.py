import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from rag.vector_store import VectorStore
from tests.test_vector_store import fake_embed

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "ingest.py"


def load_ingest() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ingest_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def ingest(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module = load_ingest()

    def fake_parse(path: Path, max_pages: int | None = None) -> list[tuple[int, str]]:
        return [(1, f"bop riser {path.stem}"), (2, "crane mud safety")]

    monkeypatch.setattr(module, "parse_pdf", fake_parse)
    return module


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    return VectorStore(persist_dir=tmp_path / "chroma", embed_fn=fake_embed)


@pytest.fixture
def pdf_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "pdfs"
    folder.mkdir()
    (folder / "a.pdf").write_bytes(b"")
    (folder / "b.pdf").write_bytes(b"")
    (folder / "notes.txt").write_text("not a pdf")
    return folder


def test_ingests_every_pdf_and_reports_counts(
    ingest: ModuleType, store: VectorStore, pdf_dir: Path
) -> None:
    counts = ingest.ingest_directory(pdf_dir, store)
    assert counts == {"a.pdf": 2, "b.pdf": 2}
    assert store.count() == 4


def test_ignores_non_pdf_files(
    ingest: ModuleType, store: VectorStore, pdf_dir: Path
) -> None:
    counts = ingest.ingest_directory(pdf_dir, store)
    assert "notes.txt" not in counts


def test_citations_use_real_filename(
    ingest: ModuleType, store: VectorStore, pdf_dir: Path
) -> None:
    ingest.ingest_directory(pdf_dir, store)
    hit = store.search("bop riser a", k=1)[0]
    assert hit.source_file in {"a.pdf", "b.pdf"}
    assert hit.page_number in {1, 2}


def test_reingesting_does_not_duplicate(
    ingest: ModuleType, store: VectorStore, pdf_dir: Path
) -> None:
    ingest.ingest_directory(pdf_dir, store)
    ingest.ingest_directory(pdf_dir, store)
    assert store.count() == 4


def test_empty_directory_returns_nothing(
    ingest: ModuleType, store: VectorStore, tmp_path: Path
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert ingest.ingest_directory(empty, store) == {}
    assert store.count() == 0
