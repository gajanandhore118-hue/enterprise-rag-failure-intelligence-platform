from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.config import settings
from src.embedding import get_embeddings


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_.-]+", text.lower())


def _rrf_merge(vector_docs: list[Document], bm25_docs: list[Document], k: int = 60) -> list[tuple[Document, float]]:
    scores: dict[str, float] = {}
    docs:   dict[str, Document] = {}
    for rank, doc in enumerate(vector_docs, start=1):
        cid = str(doc.metadata.get("chunk_id"))
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        docs[cid] = doc
    for rank, doc in enumerate(bm25_docs, start=1):
        cid = str(doc.metadata.get("chunk_id"))
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        docs[cid] = doc
    return [(docs[cid], score) for cid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


class HybridRetriever:
    def __init__(self):
        embeddings = get_embeddings()
        self.vector_store = Chroma(
            collection_name=settings.chroma_collection,
            persist_directory=settings.chroma_dir,
            embedding_function=embeddings,
        )
        corpus_path = Path(settings.chroma_dir) / "bm25_corpus.json"
        if not corpus_path.exists():
            raise RuntimeError("BM25 corpus not found. Run: python scripts/ingest.py")

        raw = json.loads(corpus_path.read_text(encoding="utf-8"))
        self.corpus_docs = [Document(page_content=item["text"], metadata=item["metadata"]) for item in raw]
        self.bm25        = BM25Okapi([_tokenize(d.page_content) for d in self.corpus_docs])
        self.reranker    = CrossEncoder(settings.reranker_model) if settings.enable_reranker else None

    def _allowed(self, doc: Document, product_family: str, search_scope: str) -> bool:
        if search_scope == "all":
            return True
        doc_pf = (doc.metadata.get("product_family") or "").strip().lower()
        wanted = (product_family or "").strip().lower()
        return not wanted or doc_pf == wanted

    def retrieve(self, query: str, product_family: str, search_scope: str = "selected") -> list[tuple[Document, float]]:
        vector_docs = [
            doc for doc, _ in self.vector_store.similarity_search_with_relevance_scores(query, k=settings.top_k_vector)
            if self._allowed(doc, product_family, search_scope)
        ]
        bm25_scores = self.bm25.get_scores(_tokenize(query))
        bm25_docs = [
            self.corpus_docs[int(i)]
            for i in np.argsort(bm25_scores)[::-1][: settings.top_k_bm25]
            if self._allowed(self.corpus_docs[int(i)], product_family, search_scope)
        ]
        merged = _rrf_merge(vector_docs, bm25_docs)

        if self.reranker and merged:
            ce_scores = self.reranker.predict([(query, doc.page_content) for doc, _ in merged])
            merged = sorted(zip([d for d, _ in merged], ce_scores.tolist()), key=lambda x: x[1], reverse=True)

        return merged[: settings.top_k_final]
