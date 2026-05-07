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
import hashlib
from pathlib import Path
from typing import List, Dict, Any

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

# Suspicious content patterns (basic malware detection)
SUSPICIOUS_PATTERNS = [
    b'<script', b'javascript:', b'vbscript:', b'onload=', b'onerror=',
    b'eval(', b'exec(', b'system(', b'<?php', b'<%', b'powershell',
    b'cmd.exe', b'/bin/sh', b'/bin/bash', b'rundll32.exe'
]

# Rate limiting: stricter limits for resource-intensive operations
from app.config import limiter


def validate_file_content(file_path: str) -> None:
    """Perform content-based validation to detect potential malware."""
    try:
        with open(file_path, 'rb') as f:
            # Read first 1MB for validation
            content = f.read(min(1024 * 1024, os.path.getsize(file_path)))

        # Check for suspicious patterns
        content_lower = content.lower()
        for pattern in SUSPICIOUS_PATTERNS:
            if pattern in content_lower:
                raise HTTPException(
                    status_code=400,
                    detail=f"Suspicious content detected in file. Pattern: {pattern.decode('utf-8', errors='ignore')}"
                )

        # Check for excessive null bytes (potential binary obfuscation)
        null_ratio = content.count(b'\x00') / len(content) if content else 0
        if null_ratio > 0.1:  # More than 10% null bytes
            raise HTTPException(
                status_code=400,
                detail="File contains suspicious binary content (excessive null bytes)"
            )

        # Check for file type consistency
        ext = Path(file_path).suffix.lower()
        detected_mime = mimetypes.guess_type(file_path)[0]

        if ext == '.pdf' and not content.startswith(b'%PDF'):
            raise HTTPException(status_code=400, detail="File claims to be PDF but doesn't have PDF header")

        if ext in ['.txt', '.md'] and b'\x00' in content[:1024]:
            raise HTTPException(status_code=400, detail="Text file contains null bytes")

        # Calculate file hash for integrity checking
        file_hash = hashlib.sha256(content).hexdigest()

        # Additional validation for text files
        if ext in ['.txt', '.md']:
            try:
                # Try to decode as UTF-8
                content.decode('utf-8')
            except UnicodeDecodeError:
                raise HTTPException(status_code=400, detail="File encoding is not valid UTF-8")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Content validation failed for {file_path}: {e}")
        raise HTTPException(status_code=400, detail=f"File validation failed: {str(e)}")


def get_document_info(filename: str) -> Dict[str, Any]:
    """Get information about a stored document."""
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        return None

    stat = file_path.stat()
    return {
        "filename": filename,
        "size": stat.st_size,
        "modified": stat.st_mtime,
        "path": str(file_path),
        "exists": True
    }


def list_documents() -> List[Dict[str, Any]]:
    """List all uploaded documents."""
    if not UPLOAD_DIR.exists():
        return []

    documents = []
    for file_path in UPLOAD_DIR.iterdir():
        if file_path.is_file():
            stat = file_path.stat()
            documents.append({
                "filename": file_path.name,
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "extension": file_path.suffix.lower()
            })

    return sorted(documents, key=lambda x: x["modified"], reverse=True)


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

    # Enhanced content validation
    try:
        validate_file_content(str(dest))
    except Exception:
        dest.unlink(missing_ok=True)
        raise

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
        "file_hash":     hashlib.sha256(open(dest, 'rb').read()).hexdigest(),
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


@router.get("/documents", summary="List all uploaded documents")
async def list_uploaded_documents():
    """List all documents currently stored in the system."""
    # Get documents from file system
    file_documents = list_documents()

    # Get documents from index
    index_documents = document_processor.list_documents()

    # Merge information
    merged_docs = []
    for file_doc in file_documents:
        filename = file_doc["filename"]
        index_info = next((d for d in index_documents if d["document_id"] == filename), None)

        merged_doc = {
            **file_doc,
            "indexed": index_info is not None,
            "chunks": index_info["chunk_count"] if index_info else 0,
            "total_chars": index_info["total_chars"] if index_info else 0
        }
        merged_docs.append(merged_doc)

    return {
        "documents": merged_docs,
        "total_count": len(merged_docs),
        "total_size": sum(doc["size"] for doc in merged_docs),
        "indexed_documents": len([d for d in merged_docs if d["indexed"]])
    }


@router.get("/documents/{filename}", summary="Get document information")
async def get_document(filename: str):
    """Get detailed information about a specific document."""
    doc_info = get_document_info(filename)
    if doc_info is None:
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found")

    # Get index information
    index_info = document_processor.get_document_info(filename)

    if index_info:
        doc_info.update({
            "indexed": True,
            "chunks": index_info["chunk_count"],
            "total_chars": index_info["total_chars"]
        })
    else:
        doc_info.update({
            "indexed": False,
            "chunks": 0,
            "total_chars": 0
        })

    return doc_info


@router.put("/documents/{filename}", summary="Update/replace a document")
@limiter.limit("3/minute")  # Moderate limit for updates
async def update_document(request: Request, filename: str, file: UploadFile = File(...)):
    """
    Update or replace an existing document.
    This will re-index the document with the new content.
    """
    # Validate filename
    safe_filename = Path(filename).name
    if safe_filename != filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Check if document exists
    file_path = UPLOAD_DIR / safe_filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Document '{safe_filename}' not found")

    # Validate new file
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTS)}",
        )

    content_type = file.content_type or mimetypes.guess_type(file.filename)[0]
    if content_type not in ALLOWED_MIMES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type '{content_type}'. File may be corrupted or malicious.",
        )

    # Create backup of old file
    backup_path = file_path.with_suffix(f"{file_path.suffix}.backup")
    try:
        shutil.copy2(file_path, backup_path)
    except Exception as e:
        logger.warning(f"Failed to create backup: {e}")

    # Upload new file
    size = 0
    temp_path = file_path.with_suffix(f"{file_path.suffix}.temp")

    try:
        with open(temp_path, "wb") as out:
            while chunk := await file.read(8192):
                size += len(chunk)
                if size > MAX_SIZE_BYTES:
                    out.close()
                    temp_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds maximum size of {MAX_SIZE_BYTES // (1024*1024)} MB"
                    )
                out.write(chunk)

        # Validate content
        validate_file_content(str(temp_path))

        # Replace old file
        temp_path.replace(file_path)

        # Re-index the document using document_id
        try:
            n_chunks = document_processor.ingest_file(str(file_path), document_id=safe_filename)
        except Exception as e:
            # Restore backup if re-indexing fails
            if backup_path.exists():
                try:
                    backup_path.replace(file_path)
                except Exception:
                    pass
            raise HTTPException(status_code=422, detail=f"Failed to re-index document: {e}")

        # Calculate new hash
        with open(file_path, 'rb') as f:
            new_hash = hashlib.sha256(f.read()).hexdigest()

        # Clean up backup
        backup_path.unlink(missing_ok=True)

        return {
            "message": f"Document '{safe_filename}' updated successfully",
            "filename": safe_filename,
            "new_size": size,
            "chunks_indexed": n_chunks,
            "file_hash": new_hash,
            "total_chunks": document_processor.chunk_count
        }

    except Exception as e:
        # Restore backup if it exists
        if backup_path.exists():
            try:
                backup_path.replace(file_path)
            except Exception:
                pass
        temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")


@router.delete("/documents/{filename}", summary="Delete a document")
@limiter.limit("5/minute")  # Moderate limit for deletions
async def delete_document(request: Request, filename: str):
    """Delete a document from the system."""
    safe_filename = Path(filename).name
    if safe_filename != filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = UPLOAD_DIR / safe_filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Document '{safe_filename}' not found")

    try:
        # Get file info before deletion
        stat = file_path.stat()

        # Remove from document processor index first
        chunks_removed = document_processor.delete_document(safe_filename)

        # Delete the file
        file_path.unlink()

        return {
            "message": f"Document '{safe_filename}' deleted successfully",
            "filename": safe_filename,
            "size_deleted": stat.st_size,
            "chunks_removed": chunks_removed,
            "total_chunks_remaining": document_processor.chunk_count
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")


@router.post("/documents/reindex", summary="Re-index all documents")
@limiter.limit("1/minute")  # Very strict limit for re-indexing
async def reindex_documents(request: Request):
    """Re-index all documents in the system. Use when document updates don't reflect in search."""
    try:
        # Reset the index
        document_processor._chunks = []
        document_processor._embeddings = None
        document_processor._index = None

        # Clear persisted data
        for path in ["data/faiss.index", "data/chunks.pkl"]:
            if os.path.exists(path):
                os.remove(path)

        # Re-ingest all documents
        total_chunks = 0
        processed_files = []

        if UPLOAD_DIR.exists():
            for file_path in UPLOAD_DIR.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in ALLOWED_EXTS:
                    try:
                        chunks = document_processor.ingest_file(str(file_path))
                        total_chunks += chunks
                        processed_files.append({
                            "filename": file_path.name,
                            "chunks": chunks
                        })
                        logger.info(f"Re-indexed {file_path.name}: {chunks} chunks")
                    except Exception as e:
                        logger.error(f"Failed to re-index {file_path.name}: {e}")

        # Save the new index
        document_processor._save_index()

        return {
            "message": "Re-indexing completed",
            "total_chunks": total_chunks,
            "files_processed": len(processed_files),
            "processed_files": processed_files
        }

    except Exception as e:
        logger.error(f"Re-indexing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Re-indexing failed: {str(e)}")
