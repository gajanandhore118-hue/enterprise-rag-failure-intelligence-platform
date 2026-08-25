# Enterprise Engineering Intelligence & Failure Analysis Platform

End-to-end Advanced RAG proof-of-concept for automotive engineering knowledge.

## Main use cases

1. Engineering Change Impact Analysis
2. Product Failure Investigation

The same chatbot supports both use cases. The user selects a broad product family, asks a natural-language question, and can continue with follow-up questions.

## Architecture

Documents → Parse → Chunk → Metadata → Embeddings → ChromaDB

User Query → Contextualize → Intent + Entity Extraction → Hybrid Retrieval → Metadata Filter → Cross-Encoder Re-ranking → LLM → Grounded Answer + Citations

## Project structure

```text
engineering_intelligence_rag/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── schemas.py
│   ├── services/
│   │   ├── ingest.py
│   │   ├── retrieval.py
│   │   ├── llm.py
│   │   └── metadata.py
│   └── static/
│       ├── index.html
│       ├── app.js
│       └── styles.css
├── scripts/
│   └── ingest.py
├── data/
├── chroma_db/
├── requirements.txt
├── .env.example
└── README.md
```

## 1. Create environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Configure LLM

Copy:

```bash
cp .env.example .env
```

### OpenAI

```text
LLM_PROVIDER=openai
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4.1-mini
```

### Azure OpenAI

```text
LLM_PROVIDER=azure
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
AZURE_OPENAI_CHAT_DEPLOYMENT=<deployment-name>
AZURE_OPENAI_API_VERSION=2024-12-01-preview
```

## 4. Add documents

The project already includes synthetic engineering TXT documents in `data/`.

You can also copy PDFs into `data/`.

For a production system, replace the basic PDF extraction in `app/services/ingest.py` with Azure AI Document Intelligence or Amazon Textract for scanned/table-heavy documents.

## 5. Ingest documents

```bash
python scripts/ingest.py
```

This creates local ChromaDB embeddings and also writes a BM25 corpus file for hybrid retrieval.

## 6. Run the application

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

## API endpoints

### Health

```http
GET /api/health
```

### Product families

```http
GET /api/product-families
```

### Chat

```http
POST /api/chat
```

Example body:

```json
{
  "product_family": "EV Battery Thermal Management",
  "search_scope": "selected",
  "question": "What risks should we consider before changing P104 from Material A to Material B?",
  "conversation": []
}
```

## Good demo questions

- What risks should we consider before changing P104 from Material A to Material B?
- Has Material B caused a previous coolant-pump housing problem?
- What did TR-1845 conclude?
- What was Supplier-Z's historical issue and was it corrected?
- Which validation procedure is currently applicable to ECR-2026-1048?
- What changed between VP-PUMP-04 Rev 4 and Rev 6?
- Why did CP-600 housings crack in the field?
- Was Polymer Grade B responsible for the CP-600 failures?
- What historical lessons influenced the CP-500 validation plan?
- What additional tests are required before approving ECR-2026-1048?

Follow-up example:

1. Show historical failures for P104.
2. Only temperature-related failures.
3. What were the root causes?
4. Which suppliers were involved?
5. What corrective actions were taken?

## Production upgrades

For a real enterprise deployment, replace or extend:

- PyPDF → Azure AI Document Intelligence / Amazon Textract
- Local ChromaDB → Azure AI Search / Pinecone / Qdrant / OpenSearch
- Local embedding model → approved enterprise embedding model
- In-memory conversation history → Redis / database
- Local HTML → enterprise React/Angular frontend
- Static product-family list → PLM/master-data service
- Basic authentication placeholder → Microsoft Entra ID / SSO
- Local logs → Application Insights / OpenTelemetry
- Simple metadata extraction → enterprise metadata + document parser + normalization layer
