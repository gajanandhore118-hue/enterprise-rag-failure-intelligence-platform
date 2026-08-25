from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app.config import settings
from app.services.ingest import get_embeddings


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_.-]+", text.lower())


def rrf_merge(vector_docs: list[Document], bm25_docs: list[Document], k: int = 60) -> list[tuple[Document, float]]:
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}

    for rank, doc in enumerate(vector_docs, start=1):
        cid = str(doc.metadata.get("chunk_id"))
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        docs[cid] = doc

    for rank, doc in enumerate(bm25_docs, start=1):
        cid = str(doc.metadata.get("chunk_id"))
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        docs[cid] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(docs[cid], score) for cid, score in ranked]


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
        self.corpus_docs = [
            Document(page_content=item["text"], metadata=item["metadata"])
            for item in raw
        ]
        self.bm25 = BM25Okapi([tokenize(d.page_content) for d in self.corpus_docs])

        self.reranker = None
        if settings.enable_reranker:
            self.reranker = CrossEncoder(settings.reranker_model)

    def _metadata_allowed(self, doc: Document, product_family: str, search_scope: str) -> bool:
        if search_scope == "all":
            return True

        doc_pf = (doc.metadata.get("product_family") or "").strip().lower()
        wanted = (product_family or "").strip().lower()
        if not wanted:
            return True

        return doc_pf == wanted

    def retrieve(self, query: str, product_family: str, search_scope: str = "selected") -> list[tuple[Document, float]]:
        # Vector results
        vector_pairs = self.vector_store.similarity_search_with_relevance_scores(
            query, k=settings.top_k_vector
        )
        vector_docs = [
            doc for doc, _score in vector_pairs
            if self._metadata_allowed(doc, product_family, search_scope)
        ]

        # BM25 results
        bm25_scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(bm25_scores)[::-1][: settings.top_k_bm25]
        bm25_docs = [
            self.corpus_docs[int(i)]
            for i in order
            if self._metadata_allowed(self.corpus_docs[int(i)], product_family, search_scope)
        ]

        merged = rrf_merge(vector_docs, bm25_docs)

        # Cross-encoder reranking
        if self.reranker and merged:
            pairs = [(query, doc.page_content) for doc, _ in merged]
            scores = self.reranker.predict(pairs)
            reranked = sorted(
                [(doc, float(score)) for (doc, _), score in zip(merged, scores)],
                key=lambda x: x[1],
                reverse=True,
            )
            return reranked[: settings.top_k_final]

        return merged[: settings.top_k_final]
