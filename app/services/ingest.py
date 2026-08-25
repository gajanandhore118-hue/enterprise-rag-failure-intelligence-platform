from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

from app.config import settings
from app.services.metadata import extract_metadata


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode([text], normalize_embeddings=True)[0]
        return vector.tolist()


def get_embeddings() -> Embeddings:
    return SentenceTransformerEmbeddings(settings.embedding_model)


def parse_txt(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [{"page_number": 1, "text": text}]


def parse_pdf(path: Path) -> list[dict]:
    # POC parser. Replace with Azure AI Document Intelligence / Textract
    # for scanned, form-heavy, or table-heavy production documents.
    reader = PdfReader(str(path))
    pages = []
    for idx, page in enumerate(reader.pages, start=1):
        pages.append({
            "page_number": idx,
            "text": page.extract_text() or "",
        })
    return pages


def parse_document(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return parse_txt(path)
    if suffix == ".pdf":
        return parse_pdf(path)
    return []


def build_langchain_documents(path: Path) -> list[Document]:
    pages = parse_document(path)
    if not pages:
        return []

    combined = "\n".join(p["text"] for p in pages)
    base_metadata = extract_metadata(combined, path.name)

    docs = []
    for p in pages:
        if not p["text"].strip():
            continue
        docs.append(
            Document(
                page_content=p["text"],
                metadata={
                    **base_metadata,
                    "page_number": p["page_number"],
                },
            )
        )
    return docs


def chunk_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(documents)

    per_doc_counter: dict[str, int] = {}
    for chunk in chunks:
        doc_id = str(chunk.metadata.get("document_id", "unknown"))
        page = int(chunk.metadata.get("page_number", 0) or 0)
        idx = per_doc_counter.get(doc_id, 0)
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["chunk_id"] = f"{doc_id}_p{page}_c{idx}"
        per_doc_counter[doc_id] = idx + 1
    return chunks


def ingest_folder(data_dir: str | None = None) -> dict:
    folder = Path(data_dir or settings.data_dir)
    folder.mkdir(parents=True, exist_ok=True)

    source_files = [
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".txt", ".md", ".pdf"}
        and not p.name.lower().startswith("readme")
    ]

    all_chunks: list[Document] = []
    for path in sorted(source_files):
        docs = build_langchain_documents(path)
        all_chunks.extend(chunk_documents(docs))

    if not all_chunks:
        return {"documents": 0, "chunks": 0}

    embeddings = get_embeddings()

    # Rebuild collection for repeatable demo ingestion.
    try:
        old = Chroma(
            collection_name=settings.chroma_collection,
            persist_directory=settings.chroma_dir,
            embedding_function=embeddings,
        )
        old.delete_collection()
    except Exception:
        pass

    vector_store = Chroma(
        collection_name=settings.chroma_collection,
        persist_directory=settings.chroma_dir,
        embedding_function=embeddings,
    )

    ids = [str(c.metadata["chunk_id"]) for c in all_chunks]
    vector_store.add_documents(all_chunks, ids=ids)

    # Save corpus for BM25.
    bm25_path = Path(settings.chroma_dir) / "bm25_corpus.json"
    bm25_path.parent.mkdir(parents=True, exist_ok=True)
    bm25_path.write_text(
        json.dumps(
            [
                {
                    "text": c.page_content,
                    "metadata": c.metadata,
                }
                for c in all_chunks
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "documents": len(source_files),
        "chunks": len(all_chunks),
    }
