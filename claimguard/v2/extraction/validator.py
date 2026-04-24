from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict

_CIN_PATTERN = re.compile(r"^[A-Z0-9]{6,12}$")


def _is_valid_date(raw: str) -> bool:
    if not raw:
        return False
    value = str(raw).strip()
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d", "%Y-%m-%d", "%d/%m/%y", "%d-%m-%y"):
        try:
            datetime.strptime(value, fmt)
            return True
        except ValueError:
            continue
    return False


def validate_extraction(payload: Dict[str, Any]) -> Dict[str, Any]:
    fields = payload.get("fields", {}) if isinstance(payload.get("fields"), dict) else {}
    confidence = payload.get("confidence")
    name = fields.get("name")
    cin = fields.get("cin")
    dob = fields.get("dob")

    if name is not None and not isinstance(name, str):
        return {"status": "ERROR", "reason": "Invalid name type", "stage": "validation"}

    if cin is not None and not _CIN_PATTERN.fullmatch(str(cin)):
        return {"status": "ERROR", "reason": "Invalid CIN format", "stage": "validation"}

    if dob is not None and not _is_valid_date(str(dob)):
        return {"status": "ERROR", "reason": "Invalid DOB date", "stage": "validation"}

    try:
        conf_val = float(confidence)
    except (TypeError, ValueError):
        return {"status": "ERROR", "reason": "Confidence must be numeric", "stage": "validation"}
    if conf_val < 0 or conf_val > 100:
        return {"status": "ERROR", "reason": "Confidence out of range", "stage": "validation"}

    return {"status": "OK"}
