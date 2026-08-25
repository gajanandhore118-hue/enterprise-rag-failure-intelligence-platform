from __future__ import annotations

import json
import re

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings


def get_chat_model():
    if settings.llm_provider.lower() == "azure":
        if not all([
            settings.azure_openai_api_key,
            settings.azure_openai_endpoint,
            settings.azure_openai_chat_deployment,
        ]):
            raise RuntimeError("Azure OpenAI settings are incomplete in .env")

        return AzureChatOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_deployment=settings.azure_openai_chat_deployment,
            temperature=0,
        )

    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is missing in .env")

    return ChatOpenAI(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        temperature=0,
    )


def classify_intent(question: str, conversation: list[dict]) -> str:
    q = question.lower()

    change_terms = [
        "change", "replace", "replacement", "material a", "material b",
        "before approving", "before approval", "risk", "proposed"
    ]
    failure_terms = [
        "failure", "failed", "crack", "leak", "root cause", "corrective action",
        "ncr", "8d", "field", "warranty"
    ]

    change_score = sum(term in q for term in change_terms)
    failure_score = sum(term in q for term in failure_terms)

    if change_score > failure_score:
        return "engineering_change_analysis"
    if failure_score > change_score:
        return "failure_investigation"

    # Follow-up: preserve last inferred conversational direction when possible.
    history = " ".join(m.get("content", "").lower() for m in conversation[-4:])
    if any(term in history for term in change_terms):
        return "engineering_change_analysis"
    if any(term in history for term in failure_terms):
        return "failure_investigation"

    return "general_engineering_search"


def contextualize_query(question: str, conversation: list[dict]) -> str:
    # For a standalone question, avoid an extra LLM call.
    if not conversation:
        return question

    model = get_chat_model()

    history = "\n".join(
        f"{m.get('role', 'user')}: {m.get('content', '')}"
        for m in conversation[-6:]
    )

    prompt = f'''
Rewrite the latest engineering question as a standalone retrieval query.

Rules:
- Preserve product IDs, component IDs, materials, suppliers, dates and failure terms.
- Do not answer the question.
- Do not add facts that were not stated.
- Return only the rewritten query.

Conversation:
{history}

Latest question:
{question}
'''

    response = model.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def answer_question(
    question: str,
    contextualized_query: str,
    intent: str,
    retrieved_docs: list[tuple],
) -> str:
    model = get_chat_model()

    context_parts = []
    for idx, (doc, score) in enumerate(retrieved_docs, start=1):
        md = doc.metadata
        source = (
            f"[S{idx}] document_id={md.get('document_id')} | "
            f"type={md.get('document_type')} | "
            f"revision={md.get('revision')} | "
            f"page={md.get('page_number')} | "
            f"file={md.get('source_file')}"
        )
        context_parts.append(f"{source}\n{doc.page_content}")

    context = "\n\n---\n\n".join(context_parts)

    system = '''
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
- Use source markers such as [S1], [S2] after important claims.
- Keep the answer clear for both engineers and non-domain readers.
'''

    user = f'''
Intent: {intent}

Original question:
{question}

Standalone retrieval query:
{contextualized_query}

Engineering evidence:
{context}

Provide a concise but useful answer with supporting source markers.
'''

    response = model.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user),
    ])

    return response.content.strip()
