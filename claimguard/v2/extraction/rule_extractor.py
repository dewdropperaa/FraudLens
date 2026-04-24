from __future__ import annotations

import re
from typing import Any, Dict

_PATTERNS: dict[str, list[str]] = {
    "name": [r"Nom complet\s*[:\-]\s*([^\n\r]+)"],
    "cin": [r"\bCIN\s*[:\-]?\s*([A-Z0-9]{6,12})\b"],
    "dob": [
        r"Date de naissance\s*[:\-]\s*([0-3]?\d[/-][0-1]?\d[/-](?:\d{4}|\d{2}))",
        r"Date de naissance\s*[:\-]\s*((?:\d{4})[/-][0-1]?\d[/-][0-3]?\d)",
    ],
    "insurance": [r"Mutuelle\s*[:\-]\s*([^\n\r]+)"],
    "patient_id": [r"N[°ºo]?\s*IPP\s*[:\-]?\s*([A-Z0-9\-]{3,20})"],
}


class RuleExtractor:
    """Deterministic regex-only extractor for medical identity fields."""

    @staticmethod
    def _match_first(text: str, patterns: list[str]) -> str | None:
        for pattern in patterns:
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if m:
                value = m.group(1).strip()
                if value:
                    return value
        return None

    @classmethod
    def extract(cls, text: str) -> Dict[str, Any]:
        source = str(text or "")
        fields: Dict[str, Any] = {
            "name": None,
            "cin": None,
            "dob": None,
            "insurance": None,
            "patient_id": None,
        }
        for field, patterns in _PATTERNS.items():
            fields[field] = cls._match_first(source, patterns)

        detected_count = sum(1 for value in fields.values() if value is not None)
        if detected_count == 0:
            confidence = 0.0
        elif detected_count == 1:
            confidence = 60.0
        else:
            confidence = min(100.0, 80.0 + ((detected_count - 2) * 10.0))

        return {
            "status": "OK",
            "engine": "rule",
            "fields": fields,
            "detected_fields": detected_count,
            "confidence": confidence,
            "should_return_immediately": detected_count >= 2,
        }
