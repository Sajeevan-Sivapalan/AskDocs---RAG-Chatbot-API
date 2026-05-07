"""
Session Manager
- In-memory multi-turn conversation history
- Optional Redis backend (set REDIS_URL env var)
- Auto-expiry after 30 minutes of inactivity
"""

import os
import time
import uuid
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Try Redis
try:
    import redis
    import json as _json
    _REDIS_URL = os.getenv("REDIS_URL", "")
    if _REDIS_URL:
        # Use connection pool for better performance and connection management
        _redis_pool = redis.ConnectionPool.from_url(_REDIS_URL, max_connections=10, decode_responses=True)
        _redis_client = redis.Redis(connection_pool=_redis_pool)
        _redis_client.ping()
        logger.info("Redis session store connected with connection pool")
    else:
        _redis_client = None
        logger.info("No REDIS_URL — using in-memory session store")
except Exception as e:
    _redis_client = None
    logger.info(f"Redis unavailable ({e}) — using in-memory session store")


TTL_SECONDS = 1800   # 30 minutes


class SessionManager:
    """
    Manages per-user conversation history.
    Falls back to in-memory dict when Redis is unavailable.
    """

    def __init__(self):
        self._store: Dict[str, dict] = {}   # in-memory fallback

    # ── Session helpers ───────────────────────────────────────────────────────
    def new_session(self) -> str:
        sid = str(uuid.uuid4())
        self._set(sid, {"history": [], "created": time.time()})
        return sid

    def get_history(self, session_id: str) -> List[dict]:
        data = self._get(session_id)
        if data is None:
            return []
        return data.get("history", [])

    def append_turn(self, session_id: str, user_msg: str, assistant_msg: str):
        data = self._get(session_id) or {"history": [], "created": time.time()}
        data["history"].append({"role": "user",      "content": user_msg})
        data["history"].append({"role": "assistant", "content": assistant_msg})
        # Keep last 20 messages (10 turns)
        data["history"] = data["history"][-20:]
        data["last_active"] = time.time()
        self._set(session_id, data)

    # ── Backend ───────────────────────────────────────────────────────────────
    def _get(self, key: str) -> Optional[dict]:
        if _redis_client:
            try:
                raw = _redis_client.get(f"session:{key}")
                return _json.loads(raw) if raw else None
            except Exception:
                pass
        # In-memory
        entry = self._store.get(key)
        if entry and time.time() - entry.get("last_active", entry["created"]) < TTL_SECONDS:
            return entry
        return None

    def _set(self, key: str, value: dict):
        if _redis_client:
            try:
                _redis_client.setex(f"session:{key}", TTL_SECONDS, _json.dumps(value))
                return
            except Exception:
                pass
        value.setdefault("last_active", time.time())
        self._store[key] = value
        self._evict_old()

    def _evict_old(self):
        """Purge stale in-memory sessions."""
        cutoff = time.time() - TTL_SECONDS
        stale  = [k for k, v in self._store.items()
                  if v.get("last_active", v.get("created", 0)) < cutoff]
        for k in stale:
            del self._store[k]


# Singleton
session_manager = SessionManager()
