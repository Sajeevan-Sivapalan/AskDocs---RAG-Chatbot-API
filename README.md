# RAG Chatbot — FastAPI

Professional document-grounded AI chatbot with Retrieval-Augmented Generation,
multi-layer safety filtering, multi-turn conversation, and OpenAPI docs.

---

## Project Structure

```
rag_chatbot/
├── app/
│   ├── main.py                    ← FastAPI app, middleware, routers
│   ├── models/
│   │   └── schemas.py             ← Pydantic request/response models
│   ├── routers/
│   │   ├── chat.py                ← POST /api/v1/chat
│   │   ├── documents.py           ← POST /api/v1/documents/upload + /text
│   │   └── health.py              ← GET /health
│   ├── services/
│   │   ├── document_processor.py  ← Parse → chunk → embed → FAISS index
│   │   ├── llm_service.py         ← Prompt builder + OpenAI / stub LLM
│   │   ├── safety_filter.py       ← Input + output safety layers
│   │   └── session_manager.py     ← Multi-turn history (Redis or in-memory)
│   └── utils/
│       └── logger.py              ← Structured logging
├── data/
│   └── sample_document.txt        ← Sample document to test with
├── tests/
│   └── test_api.py                ← Full pytest test suite
├── .env.example                   ← Environment variables template
├── Dockerfile                     ← Production Docker image
├── docker-compose.yml             ← API + Redis stack
├── requirements.txt
└── run.py                         ← Dev server entry point
```

---

## Quick Start (local, no Docker)

### 1. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Minimum deps** (no GPU, no API key needed for testing):
> `pip install fastapi uvicorn python-multipart pydantic numpy httpx`

### 3. Set environment variables (optional)

```bash
cp .env.example .env
# Edit .env — add your OPENAI_API_KEY
# If no key is set, a built-in stub LLM is used (good for dev/testing)
```

### 4. Run the server

```bash
python run.py
# or with hot-reload for development:
python run.py --reload
```

Server starts at **http://localhost:8000**

---

## Using the API

### Step 1 — Load a document

```bash
# Upload a file (PDF, TXT, DOCX)
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@data/sample_document.txt"

# Or index raw text directly
curl -X POST http://localhost:8000/api/v1/documents/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Annual leave is 21 days per year.", "source": "hr-policy"}'
```

### Step 2 — Chat

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "How many days of annual leave do employees get?"}'
```

**Response:**
```json
{
  "answer": "Employees are entitled to 21 days of paid annual leave per year.",
  "sources": [
    {
      "content": "Annual Leave: Employees are entitled to...",
      "source": "sample_document.txt",
      "score": 0.8921
    }
  ],
  "session_id": "abc-123",
  "is_blocked": false,
  "confidence": 0.8921
}
```

### Step 3 — Multi-turn conversation

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -d '{"query": "What about sick leave?", "session_id": "abc-123"}'
```

---

## Safety filter examples

| Query | Result |
|-------|--------|
| `"ignore all previous instructions"` | Blocked — prompt injection |
| `"how to build a bomb"` | Blocked — harmful |
| `"what is the bitcoin price"` | Blocked — off-topic |
| `"What is the leave policy?"` | Allowed ✓ |

---

## Interactive API docs

Open in your browser after starting the server:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:**       http://localhost:8000/redoc

---

## Run tests

```bash
pytest tests/ -v
```

---

## Docker deployment

```bash
# Build and run with Redis
docker-compose up --build

# Production (with your API key)
OPENAI_API_KEY=sk-... docker-compose up -d
```

---

## What's implemented

| Feature | Status | Notes |
|---------|--------|-------|
| FastAPI backend | ✅ | Async, versioned routes |
| Pydantic validation | ✅ | Full input/output models |
| Document ingestion | ✅ | PDF, DOCX, TXT, MD |
| Text chunking | ✅ | Sliding window with overlap |
| Sentence embeddings | ✅ | multi-qa-mpnet-base-dot-v1 (or mock) |
| FAISS vector search | ✅ | Cosine similarity + numpy fallback |
| Index persistence | ✅ | Survives server restarts |
| RAG generation | ✅ | OpenAI GPT-3.5 + stub mode |
| Input safety filter | ✅ | Regex + keyword + length |
| Output safety filter | ✅ | Post-generation scan |
| Confidence gating | ✅ | Low-score queries return "I don't know" |
| Multi-turn sessions | ✅ | Redis or in-memory |
| CORS middleware | ✅ | Configurable origins |
| Request timing | ✅ | X-Process-Time header |
| Global error handler | ✅ | Clean 500 responses |
| OpenAPI / Swagger | ✅ | Auto-generated |
| Docker + Compose | ✅ | With Redis |
| Test suite | ✅ | 10 tests covering all layers |
