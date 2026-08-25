from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    product_family: str = Field(default="EV Battery Thermal Management")
    search_scope: str = Field(default="selected", description="selected | all")
    question: str
    conversation: list[Message] = []


class SourceItem(BaseModel):
    document_id: str | None = None
    document_type: str | None = None
    source_file: str | None = None
    page_number: int | None = None
    revision: str | None = None
    section: str | None = None
    score: float | None = None
    excerpt: str | None = None


class ChatResponse(BaseModel):
    intent: str
    contextualized_query: str
    answer: str
    sources: list[SourceItem]
