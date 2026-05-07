"""Health check router"""

from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app.services.document_processor import document_processor

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health():
    return HealthResponse(
        status             = "ok",
        version            = "1.0.0",
        vector_store_ready = document_processor.is_ready,
    )


@router.get("/", include_in_schema=False)
async def root():
    return {"message": "RAG Chatbot API — visit /docs for the interactive API reference"}
