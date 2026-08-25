from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = "openai"

    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1-mini"

    azure_openai_api_key: str | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_api_version: str = "2024-12-01-preview"
    azure_openai_chat_deployment: str | None = None

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    top_k_vector: int = 12
    top_k_bm25: int = 12
    top_k_final: int = 6

    enable_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    chroma_dir: str = "./chroma_db"
    chroma_collection: str = "engineering_knowledge"
    data_dir: str = "./data"


settings = Settings()
