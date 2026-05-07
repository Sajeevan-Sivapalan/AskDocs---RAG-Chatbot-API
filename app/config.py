"""
Rate Limiting Configuration
Centralized rate limiter configuration to avoid circular imports
"""

import os
from slowapi import Limiter
from slowapi.util import get_remote_address


def get_limiter() -> Limiter:
    """Configure rate limiter with Redis backend if available."""
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        # Use Redis for distributed rate limiting
        from slowapi import RedisLimiter
        return RedisLimiter(redis_url=redis_url, key_func=get_remote_address)
    else:
        # In-memory limiter for development
        return Limiter(key_func=get_remote_address)


# Global limiter instance
limiter = get_limiter()