"""
RAG Chatbot - Main FastAPI Application
Professional document-grounded chatbot with safety filtering
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import logging
import os
from typing import List

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded

from app.routers import chat, health, documents
from app.config import limiter
from app.services.metrics import MetricsMiddleware, get_metrics_response
from app.utils.logger import setup_logger

# ── Setup ─────────────────────────────────────────────────────────────────────
logger = setup_logger(__name__)

app = FastAPI(
    title="AskDocs - RAG Chatbot API",
    description="Professional AI chatbot powered by Retrieval-Augmented Generation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add metrics middleware (must be first)
app.add_middleware(MetricsMiddleware)

# Add rate limiting middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# ── CORS Configuration ────────────────────────────────────────────────────────
def get_cors_origins() -> List[str]:
    """Get allowed CORS origins from environment or use secure defaults."""
    origins_str = os.getenv("CORS_ORIGINS", "")
    if origins_str:
        return [origin.strip() for origin in origins_str.split(",") if origin.strip()]
    # Secure default: no origins allowed in production
    env = os.getenv("ENVIRONMENT", "production").lower()
    if env == "development":
        return ["http://localhost:3000", "http://localhost:8000", "http://127.0.0.1:3000"]
    return []  # Production: no CORS by default

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    max_age=86400,  # 24 hours
)

# ── Request Timing Middleware ─────────────────────────────────────────────────
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = str(round(time.time() - start, 4))
    return response

# ── Global Exception Handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later."},
    )

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health.router, tags=["Health"])
app.include_router(chat.router,   prefix="/api/v1", tags=["Chat"])
app.include_router(documents.router, prefix="/api/v1", tags=["Documents"])

# ── Metrics Endpoint ──────────────────────────────────────────────────────────
@app.get("/metrics", summary="Prometheus metrics", include_in_schema=False)
async def metrics():
    """Expose Prometheus metrics for monitoring."""
    return get_metrics_response()

# ── Startup / Shutdown ────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("RAG Chatbot API starting up...")
    # Initialize document processor
    from app.services.document_processor import document_processor
    logger.info("Document processor initialized")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("RAG Chatbot API shutting down...")
    # Clean up resources
    try:
        from app.services.document_processor import document_processor
        # Save any pending index changes
        document_processor._save_index()
        logger.info("Document processor cleaned up")
    except Exception as e:
        logger.error(f"Error during shutdown cleanup: {e}")
