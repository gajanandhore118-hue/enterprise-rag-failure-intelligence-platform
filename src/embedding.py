from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from src.config import settings


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()


def get_embeddings() -> Embeddings:
    return SentenceTransformerEmbeddings(settings.embedding_model)
