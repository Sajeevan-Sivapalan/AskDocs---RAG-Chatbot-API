"""
Documents Router
POST /api/v1/documents/upload  — upload & index a file
POST /api/v1/documents/text    — index raw text directly
GET  /api/v1/documents/status  — index statistics
DELETE /api/v1/documents/reset — wipe the index
"""

import os
import shutil
import logging
import mimetypes
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.services.document_processor import document_processor

logger = logging.getLogger(__name__)
router = APIRouter()

UPLOAD_DIR     = Path("data/uploads")
ALLOWED_EXTS   = {".pdf", ".txt", ".docx", ".md"}
ALLOWED_MIMES  = {
    "application/pdf",
    "text/plain",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
    "text/markdown",
}
MAX_SIZE_BYTES = 20 * 1024 * 1024   # 20 MB

# Rate limiting: stricter limits for resource-intensive operations
from app.config import limiter


@router.post("/documents/upload", summary="Upload and index a document")
@limiter.limit("5/minute")  # Stricter limit for uploads
async def upload_document(request: Request, file: UploadFile = File(...)):
    """
    Accepts PDF, TXT, DOCX, or MD files.
    Parses → chunks → embeds → indexes them for retrieval.
    """
    # ── Security Validation ───────────────────────────────────────────────────
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Check file extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTS)}",
        )

    # Check MIME type
    content_type = file.content_type or mimetypes.guess_type(file.filename)[0]
    if content_type not in ALLOWED_MIMES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type '{content_type}'. File may be corrupted or malicious.",
        )

    # Sanitize filename to prevent path traversal
    safe_filename = Path(file.filename).name  # Remove any path components
    if safe_filename != file.filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / safe_filename

    # Check if file already exists
    if dest.exists():
        raise HTTPException(
            status_code=409,
            detail=f"File '{safe_filename}' already exists. Use a different name or delete the existing file.",
        )

    # Stream to disk with size validation
    size = 0
    try:
        with open(dest, "wb") as out:
            while chunk := await file.read(8192):
                size += len(chunk)
                if size > MAX_SIZE_BYTES:
                    out.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds maximum size of {MAX_SIZE_BYTES // (1024*1024)} MB"
                    )
                out.write(chunk)
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

    # Validate file is not empty
    if size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # Process the document
    try:
        n_chunks = document_processor.ingest_file(str(dest))
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        dest.unlink(missing_ok=True)  # Clean up failed upload
        raise HTTPException(status_code=422, detail=f"Failed to process document: {e}")

    return {
        "message":       f"Successfully indexed '{safe_filename}'",
        "chunks_indexed": n_chunks,
        "filename":      safe_filename,
        "total_chunks":  document_processor.chunk_count,
    }


@router.post("/documents/text", summary="Index raw text directly")
@limiter.limit("10/minute")  # Moderate limit for text uploads
async def index_text(request: Request, payload: dict):
    """
    Body: { "text": "...", "source": "my-doc" }
    Useful for programmatic ingestion without a file upload.
    """
    text   = payload.get("text", "").strip()
    source = payload.get("source", "inline")

    if not text:
        raise HTTPException(status_code=400, detail="'text' field is required and cannot be empty")

    try:
        n_chunks = document_processor.ingest_text(text, source=source)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {
        "message":       f"Indexed {n_chunks} chunks from source '{source}'",
        "chunks_indexed": n_chunks,
        "total_chunks":  document_processor.chunk_count,
    }


@router.get("/documents/status", summary="Index statistics")
async def index_status():
    return {
        "ready":        document_processor.is_ready,
        "total_chunks": document_processor.chunk_count,
        "upload_dir":   str(UPLOAD_DIR.resolve()),
    }


@router.delete("/documents/reset", summary="Wipe the entire index")
@limiter.limit("2/minute")  # Very strict limit for destructive operations
async def reset_index(request: Request):
    """Removes all indexed documents. Use with caution."""
    import pickle, numpy as np, os

    document_processor._chunks     = []
    document_processor._embeddings = None
    document_processor._index      = None

    for path in ["data/faiss.index", "data/faiss.index.npy", "data/chunks.pkl"]:
        if os.path.exists(path):
            os.remove(path)

    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR)

    return {"message": "Index wiped successfully"}
