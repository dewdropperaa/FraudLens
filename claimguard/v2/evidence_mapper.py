from __future__ import annotations

import re
from difflib import SequenceMatcher, get_close_matches
from typing import Any, Dict, List, Tuple

try:
    import fitz  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    fitz = None


_SUSPICIOUS_KEYWORDS = (
    "invoice",
    "facture",
    "prescription",
    "ordonnance",
    "tamper",
    "duplicate",
    "anomaly",
    "forged",
    "counterfeit",
)
_NUMERIC_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)?(?:\s*(?:MAD|DH|EUR|USD))?\b", re.IGNORECASE)


def _normalize(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _severity_for_reason(reason: str) -> str:
    lowered = _normalize(reason)
    if any(token in lowered for token in ("mismatch", "not found", "anomaly", "forged", "critical")):
        return "HIGH"
    if any(token in lowered for token in ("suspicious", "inconsistent", "unusual")):
        return "MEDIUM"
    return "LOW"


def extract_suspicious_spans(blackboard: Dict[str, Any], agent_outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Returns suspicious evidence spans that can be mapped on a PDF.
    """
    spans: List[Dict[str, Any]] = []
    extracted_text = str(blackboard.get("extracted_text", ""))
    identity = blackboard.get("identity", {}) if isinstance(blackboard.get("identity"), dict) else {}
    structured = blackboard.get("structured_data", {}) if isinstance(blackboard.get("structured_data"), dict) else {}
    verified = blackboard.get("verified_structured_data", {}) if isinstance(blackboard.get("verified_structured_data"), dict) else {}

    cin = str(identity.get("cin") or "").strip()
    ipp = str(identity.get("ipp") or "").strip()
    if cin and not bool(identity.get("cin_found")):
        spans.append(
            {
                "text": cin,
                "reason": "CIN not found in OCR text",
                "severity": "HIGH",
                "agent": "IdentityAgent",
            }
        )
    if ipp and not bool(identity.get("ipp_found")):
        spans.append(
            {
                "text": ipp,
                "reason": "IPP not found in OCR text",
                "severity": "HIGH",
                "agent": "IdentityAgent",
            }
        )

    field_rows = blackboard.get("field_verification", [])
    if isinstance(field_rows, list):
        for row in field_rows:
            if not isinstance(row, dict):
                continue
            field = _normalize(row.get("field"))
            verified_flag = bool(row.get("verified"))
            value = str(row.get("value") or "").strip()
            if "amount" in field and not verified_flag:
                spans.append(
                    {
                        "text": value or str(structured.get("amount", "")),
                        "reason": "Amount mismatch",
                        "severity": "HIGH",
                        "agent": "DocumentAgent",
                    }
                )

    structured_amount = str(structured.get("amount") or "").strip()
    verified_amount = str(verified.get("amount") or "").strip()
    if structured_amount and verified_amount and _normalize(structured_amount) != _normalize(verified_amount):
        spans.append(
            {
                "text": structured_amount,
                "reason": "Amount mismatch between extracted and verified fields",
                "severity": "HIGH",
                "agent": "DocumentAgent",
            }
        )

    lines = [line.strip() for line in extracted_text.splitlines() if line.strip()]
    for line in lines:
        lowered = _normalize(line)
        if any(keyword in lowered for keyword in _SUSPICIOUS_KEYWORDS):
            spans.append(
                {
                    "text": line[:120],
                    "reason": "Suspicious keyword context",
                    "severity": "MEDIUM",
                    "agent": "AnomalyAgent",
                }
            )
        for numeric in _NUMERIC_PATTERN.findall(line):
            if any(token in lowered for token in ("mismatch", "anomaly", "suspicious")):
                spans.append(
                    {
                        "text": numeric,
                        "reason": "Numeric anomaly marker in text",
                        "severity": "MEDIUM",
                        "agent": "DocumentAgent",
                    }
                )

    for output in agent_outputs or []:
        if not isinstance(output, dict):
            continue
        agent_name = str(output.get("agent") or "UnknownAgent")
        reason = str(output.get("explanation") or "").strip()
        claims = output.get("claims", [])
        if isinstance(claims, list):
            for claim in claims:
                if not isinstance(claim, dict):
                    continue
                statement = str(claim.get("statement") or "").strip()
                evidence = str(claim.get("evidence") or "").strip()
                verified_flag = bool(claim.get("verified"))
                if (not verified_flag) and (statement or evidence):
                    span_text = evidence or statement
                    span_reason = "Unverified claim evidence"
                    spans.append(
                        {
                            "text": span_text[:160],
                            "reason": span_reason,
                            "severity": "HIGH",
                            "agent": agent_name,
                        }
                    )
        lowered_reason = _normalize(reason)
        if any(token in lowered_reason for token in ("mismatch", "anomaly", "suspicious", "not found")):
            candidate = ""
            words = [w for w in re.split(r"\s+", reason) if len(w) > 3]
            if words:
                candidate = " ".join(words[:6])
            spans.append(
                {
                    "text": candidate or reason[:100],
                    "reason": reason[:200] or "Suspicious explanation",
                    "severity": _severity_for_reason(reason),
                    "agent": agent_name,
                }
            )

    seen: set[Tuple[str, str, str]] = set()
    deduped: List[Dict[str, Any]] = []
    for span in spans:
        text = str(span.get("text") or "").strip()
        reason = str(span.get("reason") or "").strip()
        severity = str(span.get("severity") or _severity_for_reason(reason)).upper()
        if not text or not reason:
            continue
        key = (_normalize(text), _normalize(reason), severity)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(
            {
                "text": text,
                "reason": reason,
                "severity": severity if severity in {"HIGH", "MEDIUM", "LOW"} else "LOW",
                "agent": str(span.get("agent") or "UnknownAgent"),
            }
        )
    return deduped[:120]


def _extract_pdf_text_boxes(pdf_path: str) -> List[Dict[str, Any]]:
    if fitz is None:
        return []
    boxes: List[Dict[str, Any]] = []
    with fitz.open(pdf_path) as doc:  # type: ignore[union-attr]
        for index, page in enumerate(doc, start=1):
            width = float(page.rect.width)
            height = float(page.rect.height)
            for word in page.get_text("words"):
                if len(word) < 5:
                    continue
                x0, y0, x1, y1, text = word[:5]
                boxes.append(
                    {
                        "text": str(text or ""),
                        "bbox": [float(x0), float(y0), float(x1), float(y1)],
                        "page": index,
                        "page_width": width,
                        "page_height": height,
                    }
                )
    return boxes


def _match_span_to_box(span_text: str, boxes: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    needle = _normalize(span_text)
    if not needle:
        return None
    best_score = 0.0
    best: Dict[str, Any] | None = None
    candidates = get_close_matches(needle, [_normalize(box.get("text")) for box in boxes], n=5, cutoff=0.7)
    for box in boxes:
        hay = _normalize(box.get("text"))
        if not hay:
            continue
        score = 0.0
        if needle == hay:
            score = 1.0
        elif needle in hay or hay in needle:
            score = 0.93
        elif hay in candidates:
            score = 0.85
        else:
            score = SequenceMatcher(a=needle, b=hay).ratio()
        if score > best_score:
            best_score = score
            best = box
    return best if best_score >= 0.72 else None


def build_fraud_heatmap(
    *,
    blackboard: Dict[str, Any],
    agent_outputs: List[Dict[str, Any]],
    pdf_path: str | None,
) -> Dict[str, Any]:
    suspicious_spans = extract_suspicious_spans(blackboard, agent_outputs)
    if not pdf_path:
        return {"heatmap": [], "fallback": suspicious_spans, "status": "missing_pdf"}
    boxes = _extract_pdf_text_boxes(str(pdf_path))
    if not boxes:
        return {"heatmap": [], "fallback": suspicious_spans, "status": "missing_coordinates"}

    highlights: List[Dict[str, Any]] = []
    unmatched: List[Dict[str, Any]] = []
    per_page_count: Dict[int, int] = {}
    for span in suspicious_spans:
        matched_box = _match_span_to_box(str(span.get("text") or ""), boxes)
        if not matched_box:
            unmatched.append(span)
            continue
        page = int(matched_box.get("page") or 1)
        per_page_count[page] = per_page_count.get(page, 0) + 1
        if per_page_count[page] > 20:
            continue
        highlights.append(
            {
                "page": page,
                "bbox": matched_box.get("bbox", [0, 0, 0, 0]),
                "page_width": matched_box.get("page_width"),
                "page_height": matched_box.get("page_height"),
                "text": span.get("text"),
                "reason": span.get("reason"),
                "severity": span.get("severity"),
                "agent": span.get("agent"),
            }
        )

    return {
        "heatmap": highlights,
        "fallback": unmatched,
        "status": "ok" if highlights else "fallback_only",
    }
