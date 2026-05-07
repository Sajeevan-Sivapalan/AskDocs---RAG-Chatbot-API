"""
Metrics and Observability Module
Provides Prometheus metrics for monitoring application health and performance
"""

import time
from typing import Optional
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
import logging

logger = logging.getLogger(__name__)

# Request metrics
REQUEST_COUNT = Counter(
    'rag_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'rag_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

# Document processing metrics
DOCUMENTS_PROCESSED = Counter(
    'rag_documents_processed_total',
    'Total number of documents processed',
    ['file_type', 'status']
)

CHUNKS_CREATED = Counter(
    'rag_chunks_created_total',
    'Total number of text chunks created'
)

# Vector search metrics
SEARCH_REQUESTS = Counter(
    'rag_search_requests_total',
    'Total number of search requests'
)

SEARCH_LATENCY = Histogram(
    'rag_search_duration_seconds',
    'Search duration in seconds'
)

# Chat metrics
CHAT_REQUESTS = Counter(
    'rag_chat_requests_total',
    'Total number of chat requests'
)

CHAT_LATENCY = Histogram(
    'rag_chat_duration_seconds',
    'Chat request duration in seconds'
)

# System metrics
ACTIVE_CONNECTIONS = Gauge(
    'rag_active_connections',
    'Number of active connections'
)

MEMORY_USAGE = Gauge(
    'rag_memory_usage_bytes',
    'Memory usage in bytes'
)

# Error metrics
ERROR_COUNT = Counter(
    'rag_errors_total',
    'Total number of errors',
    ['type', 'endpoint']
)


class MetricsMiddleware:
    """Middleware to collect request metrics."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start_time = time.time()
        method = scope["method"]
        path = scope["path"]

        # Track active connections
        ACTIVE_CONNECTIONS.inc()

        try:
            await self.app(scope, receive, send)
            status = "success"
        except Exception as e:
            ERROR_COUNT.labels(type=type(e).__name__, endpoint=path).inc()
            status = "error"
            raise
        finally:
            # Record metrics
            duration = time.time() - start_time
            REQUEST_COUNT.labels(method=method, endpoint=path, status=status).inc()
            REQUEST_LATENCY.labels(method=method, endpoint=path).observe(duration)
            ACTIVE_CONNECTIONS.dec()


def get_metrics_response() -> Response:
    """Return Prometheus metrics as HTTP response."""
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


# Utility functions for tracking metrics
def track_document_processing(file_type: str, success: bool = True):
    """Track document processing metrics."""
    status = "success" if success else "error"
    DOCUMENTS_PROCESSED.labels(file_type=file_type, status=status).inc()


def track_chunks_created(count: int):
    """Track chunk creation metrics."""
    CHUNKS_CREATED.inc(count)


def track_search(duration: float):
    """Track search performance."""
    SEARCH_REQUESTS.inc()
    SEARCH_LATENCY.observe(duration)


def track_chat(duration: float):
    """Track chat performance."""
    CHAT_REQUESTS.inc()
    CHAT_LATENCY.observe(duration)