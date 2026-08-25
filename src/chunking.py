from __future__ import annotations

import json
import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src.config import settings
from src.embedding import get_embeddings


# ── Metadata ───────────────────────────────────────────────────────────────────

_FIELD_PATTERNS = {
    "document_type":  [r"^DOCUMENT TYPE:\s*(.+)$"],
    "product_family": [r"^PRODUCT FAMILY:\s*(.+)$"],
    "product":        [r"^PRODUCT:\s*(.+)$"],
    "component":      [r"^COMPONENT:\s*(.+)$"],
    "component_id":   [r"^COMPONENT ID:\s*(.+)$"],
    "material":       [r"^PROPOSED MATERIAL:\s*(.+)$", r"^MATERIAL:\s*(.+)$", r"^CURRENT MATERIAL:\s*(.+)$"],
    "supplier":       [r"^PROPOSED SUPPLIER:\s*(.+)$", r"^SUPPLIER:\s*(.+)$"],
    "revision":       [r"^REVISION:\s*(.+)$"],
    "status":         [r"^STATUS:\s*(.+)$"],
    "effective_date": [r"^EFFECTIVE DATE:\s*(.+)$", r"^EFFECTIVE:\s*(.+)$"],
}


def extract_metadata(text: str, source_file: str) -> dict:
    meta = {
        "source_file": source_file,
        "document_id": Path(source_file).name,
        "document_type": "unknown",
        "product_family": None,
        "product": None, "component": None, "component_id": None,
        "material": None, "supplier": None, "revision": None,
        "status": None, "effective_date": None,
    }
    for field, patterns in _FIELD_PATTERNS.items():
        for pat in patterns:
            m = re.search(pat, text, flags=re.MULTILINE | re.IGNORECASE)
            if m:
                meta[field] = m.group(1).strip()
                break
    if meta["product_family"] is None:
        low = text.lower()
        if any(x in low for x in ["coolant pump", "battery thermal", "cp-420", "cp-500", "cp-600"]):
            meta["product_family"] = "EV Battery Thermal Management"
    return meta


# ── Parsing ────────────────────────────────────────────────────────────────────

def parse_document(path: Path) -> list[dict]:
    if path.suffix.lower() in {".txt", ".md"}:
        return [{"page_number": 1, "text": path.read_text(encoding="utf-8", errors="ignore")}]
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return [{"page_number": i + 1, "text": p.extract_text() or ""} for i, p in enumerate(reader.pages)]
    return []


# ── Chunking ───────────────────────────────────────────────────────────────────

_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1200, chunk_overlap=150, separators=["\n\n", "\n", ". ", " "]
)


def chunk_documents(documents: list[Document]) -> list[Document]:
    chunks = _SPLITTER.split_documents(documents)
    counter: dict[str, int] = {}
    for chunk in chunks:
        doc_id = str(chunk.metadata.get("document_id", "unknown"))
        page   = int(chunk.metadata.get("page_number", 0) or 0)
        idx    = counter.get(doc_id, 0)
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["chunk_id"]    = f"{doc_id}_p{page}_c{idx}"
        counter[doc_id] = idx + 1
    return chunks


def build_documents(path: Path) -> list[Document]:
    pages = parse_document(path)
    if not pages:
        return []
    combined  = "\n".join(p["text"] for p in pages)
    base_meta = extract_metadata(combined, path.name)
    return [
        Document(page_content=p["text"], metadata={**base_meta, "page_number": p["page_number"]})
        for p in pages if p["text"].strip()
    ]


# ── Ingest ─────────────────────────────────────────────────────────────────────

def ingest_folder(data_dir: str | None = None) -> dict:
    folder = Path(data_dir or settings.data_dir)
    folder.mkdir(parents=True, exist_ok=True)

    files = [
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".txt", ".md", ".pdf"}
        and not p.name.lower().startswith("readme")
    ]

    all_chunks: list[Document] = []
    for path in sorted(files):
        all_chunks.extend(chunk_documents(build_documents(path)))

    if not all_chunks:
        return {"documents": 0, "chunks": 0}

    import shutil
    shutil.rmtree(settings.chroma_dir, ignore_errors=True)

    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name=settings.chroma_collection,
        persist_directory=settings.chroma_dir,
        embedding_function=embeddings,
    )
    vector_store.add_documents(all_chunks, ids=[str(c.metadata["chunk_id"]) for c in all_chunks])

    bm25_path = Path(settings.chroma_dir) / "bm25_corpus.json"
    bm25_path.parent.mkdir(parents=True, exist_ok=True)
    bm25_path.write_text(
        json.dumps([{"text": c.page_content, "metadata": c.metadata} for c in all_chunks], indent=2),
        encoding="utf-8",
    )
    return {"documents": len(files), "chunks": len(all_chunks)}
