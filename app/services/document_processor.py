"""
Document Processing Service
- Ingests PDF, DOCX, TXT files
- Chunks text with overlap
- Generates embeddings (sentence-transformers)
- Stores & retrieves via FAISS
"""

import os
import logging
import pickle
import threading
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Try importing optional heavy deps ─────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False
    logger.warning("sentence-transformers not installed — using mock embeddings")

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    logger.warning("faiss-cpu not installed — using numpy fallback search")

try:
    import PyPDF2
    _PDF_AVAILABLE = True
except ImportError:
    _PDF_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    _DOCX_AVAILABLE = True
except ImportError:
    _DOCX_AVAILABLE = False


# ── Data classes (no dataclasses dep needed) ──────────────────────────────────
class Chunk:
    def __init__(self, content: str, source: str, page: Optional[int] = None, document_id: Optional[str] = None):
        self.content = content
        self.source  = source
        self.page    = page
        self.document_id = document_id or source  # Use source as document_id if not specified


class DocumentProcessor:
    """
    Full RAG document pipeline:
    1. Parse  → extract raw text per file type
    2. Chunk  → sliding window with overlap
    3. Embed  → dense vector per chunk
    4. Index  → FAISS (or numpy fallback) for fast ANN search
    """

    CHUNK_SIZE    = 500    # characters
    CHUNK_OVERLAP = 100
    MODEL_NAME    = "multi-qa-mpnet-base-dot-v1"  # 384-dim, fast & accurate
    INDEX_PATH    = "data/faiss.index"
    CHUNKS_PATH   = "data/chunks.pkl"
    CACHE_SIZE    = 1000  # Max cached query embeddings

    def __init__(self):
        self._chunks:     List[Chunk]       = []
        self._embeddings: Optional[np.ndarray] = None
        self._index       = None            # FAISS index or None
        self._model       = None
        self._lock        = threading.RLock()
        self._query_cache: Dict[str, np.ndarray] = {}  # Cache for query embeddings

        self._load_model()
        self._load_persisted_index()

    # ── Model ─────────────────────────────────────────────────────────────────
    def _load_model(self):
        if _ST_AVAILABLE:
            logger.info(f"Loading embedding model: {self.MODEL_NAME}")
            self._model = SentenceTransformer(self.MODEL_NAME)
        else:
            logger.warning("Using mock embeddings (install sentence-transformers)")

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Embed texts with caching for single queries."""
        if len(texts) == 1:
            # Check cache for single query
            query = texts[0]
            cache_key = hashlib.md5(query.encode()).hexdigest()
            if cache_key in self._query_cache:
                return self._query_cache[cache_key].reshape(1, -1)

        if self._model:
            embeddings = self._model.encode(texts, normalize_embeddings=True,
                                          show_progress_bar=False)
        else:
            # Deterministic mock: hash-based 384-dim unit vectors
            embeddings = []
            for t in texts:
                rng = np.random.default_rng(abs(hash(t)) % (2**32))
                v = rng.standard_normal(384).astype("float32")
                embeddings.append(v / (np.linalg.norm(v) + 1e-10))
            embeddings = np.array(embeddings, dtype="float32")

        # Cache single query embeddings
        if len(texts) == 1:
            if len(self._query_cache) >= self.CACHE_SIZE:
                # Remove oldest entry (simple LRU approximation)
                oldest_key = next(iter(self._query_cache))
                del self._query_cache[oldest_key]
            self._query_cache[cache_key] = embeddings[0]

        return embeddings

    # ── Parsing ───────────────────────────────────────────────────────────────
    def _parse_txt(self, path: str) -> str:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read()

    def _parse_pdf(self, path: str) -> str:
        if not _PDF_AVAILABLE:
            raise RuntimeError("PyPDF2 not installed. Run: pip install PyPDF2")
        text = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text.append(page.extract_text() or "")
        return "\n".join(text)

    def _parse_docx(self, path: str) -> str:
        if not _DOCX_AVAILABLE:
            raise RuntimeError("python-docx not installed. Run: pip install python-docx")
        doc = DocxDocument(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    def _parse_file(self, path: str) -> str:
        ext = Path(path).suffix.lower()
        if ext == ".pdf":
            return self._parse_pdf(path)
        elif ext in (".docx", ".doc"):
            return self._parse_docx(path)
        else:
            return self._parse_txt(path)

    # ── Chunking ──────────────────────────────────────────────────────────────
    def _chunk_text(self, text: str, source: str, document_id: Optional[str] = None) -> List[Chunk]:
        chunks = []
        start  = 0
        while start < len(text):
            end     = min(start + self.CHUNK_SIZE, len(text))
            content = text[start:end].strip()
            if len(content) > 30:              # skip tiny fragments
                chunks.append(Chunk(content=content, source=source, document_id=document_id))
            start += self.CHUNK_SIZE - self.CHUNK_OVERLAP
        return chunks

    def _remove_chunks_by_document(self, document_id: str) -> int:
        """Remove all chunks belonging to a specific document. Returns count removed."""
        with self._lock:
            original_count = len(self._chunks)
            self._chunks = [chunk for chunk in self._chunks if chunk.document_id != document_id]
            removed_count = original_count - len(self._chunks)

            if removed_count > 0:
                # Rebuild index and embeddings
                if self._chunks:
                    self._embeddings = self._embed([c.content for c in self._chunks])
                    self._build_index(self._embeddings)
                else:
                    self._embeddings = None
                    self._index = None

                self._save_index()

            return removed_count

    # ── Indexing ──────────────────────────────────────────────────────────────
    def _build_index(self, embeddings: np.ndarray):
        with self._lock:
            dim = embeddings.shape[1]
            n_vectors = embeddings.shape[0]

            if _FAISS_AVAILABLE:
                if n_vectors < 1000:
                    # Use exact search for small datasets
                    index = faiss.IndexFlatIP(dim)
                else:
                    # Use IVF for scalability (approximate but much faster)
                    nlist = min(100, max(4, n_vectors // 39))  # Rule of thumb: sqrt(n)/4
                    quantizer = faiss.IndexFlatIP(dim)
                    index = faiss.IndexIVFFlat(quantizer, dim, nlist)
                    # Train the index
                    index.train(embeddings)
                    logger.info(f"Trained IVF index with {nlist} cells for {n_vectors} vectors")

                index.add(embeddings)
                self._index = index
            else:
                self._index = None               # fallback: brute-force numpy

    def _save_index(self):
        with self._lock:
            os.makedirs("data", exist_ok=True)
            if _FAISS_AVAILABLE and self._index is not None:
                faiss.write_index(self._index, self.INDEX_PATH)
            np.save(self.INDEX_PATH + ".npy", self._embeddings)
            with open(self.CHUNKS_PATH, "wb") as f:
                pickle.dump(self._chunks, f)
            logger.info(f"Index saved ({len(self._chunks)} chunks)")

    def _load_persisted_index(self):
        with self._lock:
            if not os.path.exists(self.CHUNKS_PATH):
                return
            try:
                with open(self.CHUNKS_PATH, "rb") as f:
                    self._chunks = pickle.load(f)
                self._embeddings = np.load(self.INDEX_PATH + ".npy")
                if _FAISS_AVAILABLE and os.path.exists(self.INDEX_PATH):
                    self._index = faiss.read_index(self.INDEX_PATH)
                logger.info(f"Loaded persisted index: {len(self._chunks)} chunks")
            except Exception as e:
                logger.warning(f"Could not load persisted index: {e}")

    # ── Public API ────────────────────────────────────────────────────────────
    def ingest_file(self, path: str, document_id: Optional[str] = None) -> int:
        """Parse → chunk → embed → index. Returns chunk count."""
        import time
        start_time = time.time()

        try:
            logger.info(f"Ingesting: {path}")

            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")

            raw_text = self._parse_file(path)

            if not raw_text.strip():
                raise ValueError(f"No text content extracted from file: {path}")

            # Use filename as document_id if not provided
            if document_id is None:
                document_id = Path(path).name

            new_chunks = self._chunk_text(raw_text, source=Path(path).name, document_id=document_id)

            if not new_chunks:
                raise ValueError("No chunks created from document content.")

            new_embeddings = self._embed([c.content for c in new_chunks])

            with self._lock:
                # Remove existing chunks for this document if any
                self._remove_chunks_by_document(document_id)

                # Append to existing
                self._chunks.extend(new_chunks)
                if self._embeddings is None:
                    self._embeddings = new_embeddings
                else:
                    self._embeddings = np.vstack([self._embeddings, new_embeddings])

                self._build_index(self._embeddings)
                self._save_index()

            chunk_count = len(new_chunks)
            processing_time = time.time() - start_time

            # Track metrics
            from app.services.metrics import track_document_processing, track_chunks_created
            file_type = Path(path).suffix.lower()
            track_document_processing(file_type, success=True)
            track_chunks_created(chunk_count)

            logger.info(f"Successfully indexed {chunk_count} chunks from {path} in {processing_time:.2f}s")
            return chunk_count

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Failed to ingest file {path} after {processing_time:.2f}s: {e}")

            # Track failed processing
            from app.services.metrics import track_document_processing
            file_type = Path(path).suffix.lower()
            track_document_processing(file_type, success=False)

            raise

    def ingest_text(self, text: str, source: str = "inline", document_id: Optional[str] = None) -> int:
        """Ingest raw text directly (no file needed)."""
        if document_id is None:
            document_id = source

        chunks = self._chunk_text(text, source=source, document_id=document_id)
        if not chunks:
            return 0
        embeddings = self._embed([c.content for c in chunks])

        with self._lock:
            # Remove existing chunks for this document if any
            self._remove_chunks_by_document(document_id)

            self._chunks.extend(chunks)
            if self._embeddings is None:
                self._embeddings = embeddings
            else:
                self._embeddings = np.vstack([self._embeddings, embeddings])

            self._build_index(self._embeddings)
            self._save_index()

        return len(chunks)

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """
        Return top_k (chunk, score) pairs sorted by cosine similarity.
        Returns [] if no documents are indexed.
        """
        import time
        start_time = time.time()

        try:
            with self._lock:
                if not self._chunks or self._embeddings is None:
                    return []

                q_vec = self._embed([query])   # shape (1, dim)

            if _FAISS_AVAILABLE and self._index is not None:
                # Configure search parameters for IVF index
                if hasattr(self._index, 'nprobe'):
                    self._index.nprobe = min(10, self._index.nlist)  # Search more cells for better accuracy

                scores, indices = self._index.search(q_vec, min(top_k, len(self._chunks)))
                results = [
                    (self._chunks[i], float(scores[0][j]))
                    for j, i in enumerate(indices[0])
                    if i >= 0
                ]
            else:
                # Numpy brute-force cosine (embeddings already normalised)
                sims = (self._embeddings @ q_vec.T).flatten()
                top  = np.argsort(sims)[::-1][:top_k]
                results = [(self._chunks[i], float(sims[i])) for i in top]

            # Track search metrics
            search_time = time.time() - start_time
            from app.services.metrics import track_search
            track_search(search_time)

            return results

        except Exception as e:
            search_time = time.time() - start_time
            logger.error(f"Search failed for query '{query}' after {search_time:.2f}s: {e}")
            # Return empty results on error rather than crashing
            return []

    @property
    def is_ready(self) -> bool:
        """Check if the document processor is fully operational."""
        try:
            # Check if model is loaded
            if self._model is None:
                return False
            # Check if we can perform basic operations
            test_embedding = self._embed(["test"])
            return len(test_embedding) > 0 and test_embedding.shape[1] == self.MODEL_DIM
        except Exception:
            return False

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents with their metadata."""
        with self._lock:
            doc_info = {}
            for chunk in self._chunks:
                doc_id = chunk.document_id or chunk.source
                if doc_id not in doc_info:
                    doc_info[doc_id] = {
                        "document_id": doc_id,
                        "source": chunk.source,
                        "chunk_count": 0,
                        "total_chars": 0
                    }
                doc_info[doc_id]["chunk_count"] += 1
                doc_info[doc_id]["total_chars"] += len(chunk.content)

            return list(doc_info.values())

    def delete_document(self, document_id: str) -> bool:
        """Delete all chunks belonging to a specific document. Returns True if any chunks were removed."""
        removed_count = self._remove_chunks_by_document(document_id)
        return removed_count > 0

    def get_document_info(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific document."""
        with self._lock:
            for chunk in self._chunks:
                if chunk.document_id == document_id or chunk.source == document_id:
                    # Count all chunks for this document
                    chunk_count = sum(1 for c in self._chunks if (c.document_id == document_id or c.source == document_id))
                    total_chars = sum(len(c.content) for c in self._chunks if (c.document_id == document_id or c.source == document_id))

                    return {
                        "document_id": document_id,
                        "source": chunk.source,
                        "chunk_count": chunk_count,
                        "total_chars": total_chars
                    }
            return None


# Singleton
document_processor = DocumentProcessor()
