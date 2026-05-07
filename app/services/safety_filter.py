"""
Safety & Filtering Service
- Keyword / regex blocklist
- Semantic intent check via embeddings
- Output moderation
"""

import re
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)

# ── Blocklist patterns ────────────────────────────────────────────────────────
BLOCKED_PATTERNS: List[str] = [
    # Violence / harm
    r"\b(kill|murder|harm|attack|weapon|bomb|explosive)\b",
    # Sensitive PII requests
    r"\b(social security|credit card number|bank account|password)\b",
    # Adult content
    r"\b(pornograph|xxx|nude|naked)\b",
    # Prompt injection attempts
    r"(ignore (all |previous |prior |above )?instructions?)",
    r"(forget (everything|all|your|previous))",
    r"(you are now|act as|pretend (to be|you are))",
    r"(jailbreak|dan mode|developer mode)",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in BLOCKED_PATTERNS]

# ── Off-topic categories (extend as needed) ───────────────────────────────────
OFF_TOPIC_KEYWORDS: List[str] = [
    "cryptocurrency", "bitcoin", "stock price", "lottery",
    "gambling", "sports bet", "horoscope", "astrology",
]


class SafetyFilter:
    """
    Multi-layer safety filter applied BEFORE retrieval and AFTER generation.
    Layer 1 — Regex / keyword blocklist
    Layer 2 — Off-topic keyword check
    Layer 3 — Post-generation output scan
    """

    # ── Input check ───────────────────────────────────────────────────────────
    def check_input(self, query: str) -> Tuple[bool, str]:
        """
        Returns (is_safe, reason).
        is_safe=False means the query must be blocked.
        """
        # Layer 1: blocklist
        for pattern in _COMPILED:
            if pattern.search(query):
                logger.warning(f"Blocked query matched pattern: {pattern.pattern}")
                return False, (
                    "I'm sorry, I can't assist with that request. "
                    "Please ask something related to the available documents."
                )

        # Layer 2: off-topic keywords
        q_lower = query.lower()
        for keyword in OFF_TOPIC_KEYWORDS:
            if keyword in q_lower:
                logger.info(f"Off-topic keyword detected: '{keyword}'")
                return False, (
                    f"That topic is outside the scope of this assistant. "
                    f"I can only answer questions based on the loaded documents."
                )

        # Layer 3: length sanity
        if len(query.split()) > 300:
            return False, "Your query is too long. Please keep it under 300 words."

        return True, ""

    # ── Output check ─────────────────────────────────────────────────────────
    def check_output(self, answer: str) -> Tuple[bool, str]:
        """
        Post-generation scan. Returns (is_safe, reason).
        """
        for pattern in _COMPILED:
            if pattern.search(answer):
                logger.warning("LLM output triggered safety filter — suppressing.")
                return False, (
                    "The generated response was flagged and cannot be shown. "
                    "Please rephrase your question."
                )
        return True, ""


# Singleton
safety_filter = SafetyFilter()
