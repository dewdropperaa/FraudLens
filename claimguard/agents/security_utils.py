"""Shared input sanitization, injection detection, and structured output validation."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("claimguard.agents.security")

_MAX_FIELD_LEN = 2000

_INJECTION_SUBSTRINGS = (
    "ignore previous instructions",
    "act as",
    "system override",
    "you are now",
)

# Broader detection (pre-sanitization) for audit / risk bump
_INJECTION_DETECT_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE | re.DOTALL)
    for p in (
        r"ignore\s+(all\s+)?(previous|prior)\s+instructions?",
        r"\bact\s+as\b",
        r"system\s+override",
        r"\byou\s+are\s+now\b",
        r"disregard\s+(the\s+)?(above|prior)",
        r"<\s*/?\s*system\s*>",
        r"\[?\s*INST\s*\]?",
    )
)

_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)


def sanitize_input(text: str) -> str:
    """Strip risky patterns, code fences, truncate, and normalize whitespace."""
    if not isinstance(text, str):
        text = str(text)
    t = text
    for sub in _INJECTION_SUBSTRINGS:
        t = re.sub(re.escape(sub), " ", t, flags=re.IGNORECASE)
    t = _CODE_BLOCK_RE.sub(" ", t)
    # Inline backtick blocks / stray fences
    t = re.sub(r"`{3,}", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > _MAX_FIELD_LEN:
        t = t[:_MAX_FIELD_LEN]
    return t


def detect_prompt_injection(text: str) -> bool:
    """Return True if untrusted text shows common prompt-injection or jailbreak patterns."""
    if not text or not isinstance(text, str):
        return False
    low = text.lower()
    for sub in _INJECTION_SUBSTRINGS:
        if sub in low:
            return True
    for rx in _INJECTION_DETECT_PATTERNS:
        if rx.search(text):
            return True
    return False


def hash_text(text: str) -> str:
    """SHA-256 hex digest for logging (never log raw PHI/PII)."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


class RiskOutputSchema(BaseModel):
    """Enforced post-processing shape for agent outputs."""

    risk_score: float = Field(..., ge=0.0, le=1.0)
    flags: list[str] = Field(default_factory=list)
    explanation: str = Field(default="")

    @field_validator("flags", mode="before")
    @classmethod
    def _coerce_flags(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v]
        return [str(v)]


_FALLBACK_OUTPUT = RiskOutputSchema(
    risk_score=0.5,
    flags=["validation_error"],
    explanation="Output failed schema validation; defensive default applied.",
)


def validate_risk_output(data: dict[str, Any]) -> RiskOutputSchema | None:
    try:
        return RiskOutputSchema.model_validate(data)
    except Exception:
        return None


def coerce_risk_output(
    primary: dict[str, Any],
    rebuild: Any | None = None,
) -> RiskOutputSchema:
    """
    Validate structured output; retry once via rebuild callable, else safe default.
    `rebuild` takes no args and returns a dict with risk_score, flags, explanation.
    """
    v = validate_risk_output(primary)
    if v is not None:
        return v
    if rebuild is not None:
        try:
            second = rebuild()
            v2 = validate_risk_output(second) if isinstance(second, dict) else None
            if v2 is not None:
                return v2
        except Exception:
            pass
    return _FALLBACK_OUTPUT.model_copy(deep=True)


def score_to_risk_score(agent_score_0_100: float) -> float:
    """Map legacy 0–100 score (higher = safer) to 0–1 risk (higher = riskier)."""
    try:
        s = float(agent_score_0_100)
    except (TypeError, ValueError):
        return 0.5
    s = max(0.0, min(100.0, s))
    return round(1.0 - (s / 100.0), 4)


def bump_risk(risk: float, delta: float) -> float:
    return max(0.0, min(1.0, round(risk + delta, 4)))


def log_security_event(
    *,
    agent_name: str,
    payload_fingerprint: str,
    flags: list[str],
    risk_score: float,
) -> None:
    logger.info(
        "agent_security agent=%s sanitized_input_sha256=%s flags=%s risk_score=%s",
        agent_name,
        payload_fingerprint,
        json.dumps(sorted(set(flags)), sort_keys=True),
        risk_score,
    )
