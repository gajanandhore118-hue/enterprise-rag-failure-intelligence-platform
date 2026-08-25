from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from src.config import settings


def get_chat_model():
    if settings.llm_provider.lower() == "azure":
        if not all([settings.azure_openai_key, settings.azure_openai_endpoint, settings.azure_openai_gpt_deployment]):
            raise RuntimeError("Azure OpenAI settings are incomplete in .env")
        return AzureChatOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_key,
            api_version=settings.azure_openai_api_version,
            azure_deployment=settings.azure_openai_gpt_deployment,
            temperature=0,
        )
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is missing in .env")
    return ChatOpenAI(api_key=settings.openai_api_key, model=settings.openai_model, temperature=0)


SYSTEM_PROMPT = """\
You are an enterprise Engineering Intelligence assistant.

You support:
1. Engineering Change Impact Analysis.
2. Product Failure Investigation.

Rules:
- Answer only from the provided engineering evidence.
- Do not invent engineering facts.
- Distinguish confirmed evidence from possible investigation areas.
- Never independently approve an engineering change.
- Never claim a final root cause unless the supplied evidence explicitly does so.
- If evidence is insufficient, say so.
- Use source markers such as [S1], [S2] after important claims.\
"""

_CONTEXTUALIZE_TMPL = """\
Rewrite the latest engineering question as a standalone retrieval query.

Rules:
- Preserve product IDs, component IDs, materials, suppliers, dates and failure terms.
- Do not answer the question.
- Do not add facts that were not stated.
- Return only the rewritten query.

Conversation:
{history}

Latest question:
{question}\
"""

_ANSWER_TMPL = """\
Intent: {intent}

Original question:
{question}

Standalone retrieval query:
{contextualized_query}

Engineering evidence:
{context}

Provide a concise but useful answer with supporting source markers.\
"""


def classify_intent(question: str, conversation: list[dict]) -> str:
    q = question.lower()
    change_terms  = ["change", "replace", "replacement", "material a", "material b",
                     "before approving", "before approval", "risk", "proposed"]
    failure_terms = ["failure", "failed", "crack", "leak", "root cause", "corrective action",
                     "ncr", "8d", "field", "warranty"]
    cs = sum(t in q for t in change_terms)
    fs = sum(t in q for t in failure_terms)
    if cs > fs:
        return "engineering_change_analysis"
    if fs > cs:
        return "failure_investigation"
    history = " ".join(m.get("content", "").lower() for m in conversation[-4:])
    if any(t in history for t in change_terms):
        return "engineering_change_analysis"
    if any(t in history for t in failure_terms):
        return "failure_investigation"
    return "general_engineering_search"


def contextualize_query(question: str, conversation: list[dict]) -> str:
    if not conversation:
        return question
    history  = "\n".join(f"{m.get('role','user')}: {m.get('content','')}" for m in conversation[-6:])
    response = get_chat_model().invoke(
        [HumanMessage(content=_CONTEXTUALIZE_TMPL.format(history=history, question=question))]
    )
    return response.content.strip()


def answer_question(question: str, contextualized_query: str, intent: str, retrieved_docs: list[tuple]) -> str:
    context_parts = []
    for idx, (doc, _) in enumerate(retrieved_docs, start=1):
        md = doc.metadata
        header = (f"[S{idx}] document_id={md.get('document_id')} | type={md.get('document_type')} | "
                  f"revision={md.get('revision')} | page={md.get('page_number')} | file={md.get('source_file')}")
        context_parts.append(f"{header}\n{doc.page_content}")

    user_msg = _ANSWER_TMPL.format(
        intent=intent,
        question=question,
        contextualized_query=contextualized_query,
        context="\n\n---\n\n".join(context_parts),
    )
    response = get_chat_model().invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_msg)])
    return response.content.strip()
