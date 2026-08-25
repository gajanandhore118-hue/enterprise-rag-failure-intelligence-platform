from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.schemas import ChatRequest, ChatResponse, SourceItem
from app.services.llm import classify_intent, contextualize_query, answer_question
from app.services.retrieval import HybridRetriever


app = FastAPI(
    title="Enterprise Engineering Intelligence & Failure Analysis Platform",
    version="1.0.0",
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_retriever = None


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/product-families")
def product_families():
    return {
        "items": [
            "EV Battery Thermal Management",
            "Powertrain Thermal Management",
            "Climate-Controlled Seating",
            "Automotive Electronic Systems",
            "Pumps & Fluid-Control Systems",
        ]
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        conversation = [m.model_dump() for m in req.conversation]

        intent = classify_intent(req.question, conversation)
        standalone = contextualize_query(req.question, conversation)

        retriever = get_retriever()
        retrieved = retriever.retrieve(
            standalone,
            product_family=req.product_family,
            search_scope=req.search_scope,
        )

        if not retrieved:
            return ChatResponse(
                intent=intent,
                contextualized_query=standalone,
                answer="I could not find sufficient engineering evidence for this question.",
                sources=[],
            )

        answer = answer_question(
            question=req.question,
            contextualized_query=standalone,
            intent=intent,
            retrieved_docs=retrieved,
        )

        sources = []
        for doc, score in retrieved:
            md = doc.metadata
            sources.append(
                SourceItem(
                    document_id=md.get("document_id"),
                    document_type=md.get("document_type"),
                    source_file=md.get("source_file"),
                    page_number=md.get("page_number"),
                    revision=md.get("revision"),
                    section=md.get("section"),
                    score=float(score),
                    excerpt=doc.page_content[:300],
                )
            )

        return ChatResponse(
            intent=intent,
            contextualized_query=standalone,
            answer=answer,
            sources=sources,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
