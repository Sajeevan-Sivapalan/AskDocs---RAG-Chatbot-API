"""
Chat Router — POST /api/v1/chat
Full RAG pipeline: safety check → retrieve → generate → safety check output
"""

import uuid
import logging
from fastapi import APIRouter, HTTPException

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.models.schemas import ChatRequest, ChatResponse, SourceChunk
from app.services.safety_filter    import safety_filter
from app.services.document_processor import document_processor
from app.services.llm_service      import llm_service
from app.services.session_manager  import session_manager

logger = logging.getLogger(__name__)
router = APIRouter()

CONFIDENCE_THRESHOLD = 0.30   # below this → "not enough info"

# Rate limiting: 10 requests per minute per IP for chat
limiter = Limiter(key_func=get_remote_address)


@router.post("/chat", response_model=ChatResponse, summary="Send a chat message")
@limiter.limit("10/minute")
async def chat(request: ChatRequest):
    """
    Full RAG pipeline:
    1. Safety filter (input)
    2. Retrieve relevant chunks from vector store
    3. Generate grounded answer via LLM
    4. Safety filter (output)
    5. Return answer + sources + confidence
    """

    # ── 1. Input safety check ─────────────────────────────────────────────────
    is_safe, block_reason = safety_filter.check_input(request.query)
    if not is_safe:
        logger.info(f"Query blocked: {request.query[:80]}")
        return ChatResponse(
            answer       = block_reason,
            is_blocked   = True,
            block_reason = block_reason,
            confidence   = 0.0,
        )

    # ── 2. Session handling ───────────────────────────────────────────────────
    session_id = request.session_id or str(uuid.uuid4())
    history    = session_manager.get_history(session_id)
    # Merge any history passed in the request body
    if request.history:
        for msg in request.history:
            history.append({"role": msg.role.value, "content": msg.content})

    # ── 3. Retrieve ───────────────────────────────────────────────────────────
    if not document_processor.is_ready:
        return ChatResponse(
            answer     = (
                "No documents have been loaded yet. "
                "Please upload a document first via POST /api/v1/documents/upload"
            ),
            session_id = session_id,
            confidence = 0.0,
        )

    raw_chunks = document_processor.retrieve(request.query, top_k=request.top_k)

    # ── 4. Confidence gate ────────────────────────────────────────────────────
    if not raw_chunks or raw_chunks[0][1] < CONFIDENCE_THRESHOLD:
        logger.info(f"Low confidence ({raw_chunks[0][1] if raw_chunks else 0:.2f}) for: {request.query[:60]}")
        return ChatResponse(
            answer     = "I don't have enough information in the loaded documents to answer that question.",
            sources    = [],
            session_id = session_id,
            confidence = raw_chunks[0][1] if raw_chunks else 0.0,
        )

    # ── 5. Generate ───────────────────────────────────────────────────────────
    answer = await llm_service.generate(
        query   = request.query,
        chunks  = raw_chunks,
        history = history,
    )

    # ── 6. Output safety check ────────────────────────────────────────────────
    out_safe, out_reason = safety_filter.check_output(answer)
    if not out_safe:
        return ChatResponse(
            answer       = out_reason,
            is_blocked   = True,
            block_reason = "Output safety filter triggered.",
            session_id   = session_id,
            confidence   = 0.0,
        )

    # ── 7. Persist turn & respond ─────────────────────────────────────────────
    session_manager.append_turn(session_id, request.query, answer)

    sources = [
        SourceChunk(
            content = chunk.content[:300],
            source  = chunk.source,
            page    = chunk.page,
            score   = round(score, 4),
        )
        for chunk, score in raw_chunks
    ]

    avg_confidence = sum(s for _, s in raw_chunks) / len(raw_chunks)

    return ChatResponse(
        answer     = answer,
        sources    = sources,
        session_id = session_id,
        confidence = round(avg_confidence, 4),
    )
