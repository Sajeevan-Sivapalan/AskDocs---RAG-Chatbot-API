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
from pathlib import Path
from typing import List, Tuple, Optional

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
    def __init__(self, content: str, source: str, page: Optional[int] = None):
        self.content = content
        self.source  = source
        self.page    = page


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

    def __init__(self):
        self._chunks:     List[Chunk]       = []
        self._embeddings: Optional[np.ndarray] = None
        self._index       = None            # FAISS index or None
        self._model       = None

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
        if self._model:
            return self._model.encode(texts, normalize_embeddings=True,
                                      show_progress_bar=False)
        # Deterministic mock: hash-based 384-dim unit vectors
        vecs = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.standard_normal(384).astype("float32")
            vecs.append(v / (np.linalg.norm(v) + 1e-10))
        return np.array(vecs, dtype="float32")

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
    def _chunk_text(self, text: str, source: str) -> List[Chunk]:
        chunks = []
        start  = 0
        while start < len(text):
            end     = min(start + self.CHUNK_SIZE, len(text))
            content = text[start:end].strip()
            if len(content) > 30:              # skip tiny fragments
                chunks.append(Chunk(content=content, source=source))
            start += self.CHUNK_SIZE - self.CHUNK_OVERLAP
        return chunks

    # ── Indexing ──────────────────────────────────────────────────────────────
    def _build_index(self, embeddings: np.ndarray):
        dim = embeddings.shape[1]
        if _FAISS_AVAILABLE:
            index = faiss.IndexFlatIP(dim)   # Inner-product on unit vecs = cosine
            index.add(embeddings)
            self._index = index
        else:
            self._index = None               # fallback: brute-force numpy

    def _save_index(self):
        os.makedirs("data", exist_ok=True)
        if _FAISS_AVAILABLE and self._index is not None:
            faiss.write_index(self._index, self.INDEX_PATH)
        np.save(self.INDEX_PATH + ".npy", self._embeddings)
        with open(self.CHUNKS_PATH, "wb") as f:
            pickle.dump(self._chunks, f)
        logger.info(f"Index saved ({len(self._chunks)} chunks)")

    def _load_persisted_index(self):
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
    def ingest_file(self, path: str) -> int:
        """Parse → chunk → embed → index. Returns chunk count."""
        logger.info(f"Ingesting: {path}")
        raw_text = self._parse_file(path)
        new_chunks = self._chunk_text(raw_text, source=Path(path).name)

        if not new_chunks:
            raise ValueError("No content extracted from document.")

        new_embeddings = self._embed([c.content for c in new_chunks])

        # Append to existing
        self._chunks.extend(new_chunks)
        if self._embeddings is None:
            self._embeddings = new_embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, new_embeddings])

        self._build_index(self._embeddings)
        self._save_index()
        logger.info(f"Indexed {len(new_chunks)} chunks from {path}")
        return len(new_chunks)

    def ingest_text(self, text: str, source: str = "inline") -> int:
        """Ingest raw text directly (no file needed)."""
        chunks = self._chunk_text(text, source=source)
        if not chunks:
            return 0
        embeddings = self._embed([c.content for c in chunks])
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
        if not self._chunks or self._embeddings is None:
            return []

        q_vec = self._embed([query])   # shape (1, dim)

        if _FAISS_AVAILABLE and self._index is not None:
            scores, indices = self._index.search(q_vec, min(top_k, len(self._chunks)))
            return [
                (self._chunks[i], float(scores[0][j]))
                for j, i in enumerate(indices[0])
                if i >= 0
            ]
        else:
            # Numpy brute-force cosine (embeddings already normalised)
            sims = (self._embeddings @ q_vec.T).flatten()
            top  = np.argsort(sims)[::-1][:top_k]
            return [(self._chunks[i], float(sims[i])) for i in top]

    @property
    def is_ready(self) -> bool:
        return bool(self._chunks)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)


# Singleton
document_processor = DocumentProcessor()
