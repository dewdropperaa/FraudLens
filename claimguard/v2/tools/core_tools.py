from __future__ import annotations

import re
from typing import Any, Dict, List

from claimguard.v2.tools.registry import register_tool

_REQUIRED = ("medical_report", "invoice", "prescription")

_KEYWORD_GROUPS: Dict[str, tuple[str, ...]] = {
    "medical_report": (
        "medical_report",
        "medical report",
        "clinical report",
        "compte rendu",
        "diagnostic",
        "hospitalisation",
        "rapport medical",
        "rapport médical",
        "certificat medical",
        "certificat médical",
        "consultation",
        "traitement",
    ),
    "invoice": (
        "invoice",
        "facture",
        "total ttc",
        "montant ttc",
        "tva",
        "reçu",
        "recu",
        "note d'honoraires",
        "remboursement",
        "montant",
    ),
    "prescription": (
        "prescription",
        "ordonnance",
        "medicament",
        "médicament",
        "posologie",
        "traitement prescrit",
        "dose",
    ),
}

_INSURANCE_DOC_KEYWORDS: tuple[str, ...] = (
    "assurance",
    "police d'assurance",
    "attestation",
    "carte d'assuré",
    "numéro d'assuré",
    "cnss",
    "cnops",
    "mutuelle",
    "bénéficiaire",
    "adhérent",
)

_FRAUD_TEXT_SIGNALS: tuple[str, ...] = (
    "fake invoice",
    "forged",
    "tampered",
    "altered",
    "counterfeit",
    "fabricated",
    "falsified",
    "duplicate billing",
    "stolen identity",
)

_SUSPICIOUS_FILENAME_SIGNALS: tuple[str, ...] = ("duplicate", "copy", "scan")


def _tool_result(tool: str, status: str, output: Dict[str, Any], confidence: float, reason: str = "") -> Dict[str, Any]:
    result = {
        "tool": tool,
        "status": status,
        "output": output,
        "confidence": max(0.0, min(1.0, float(confidence))),
    }
    if reason:
        result["reason"] = reason
    return result


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _build_corpus(documents: List[str], extractions: List[Dict[str, Any]]) -> str:
    parts: List[str] = [d.lower() for d in documents]
    for ex in extractions:
        parts.append(_safe_text(ex.get("file_name")).lower())
        parts.append(_safe_text(ex.get("extracted_text")).lower())
    return " ".join(parts)


def ocr_extractor_tool(input_data: Dict[str, Any]) -> Dict[str, Any]:
    extractions_raw = input_data.get("document_extractions")
    if extractions_raw is None:
        extractions_raw = []
    if not isinstance(extractions_raw, list):
        return _tool_result("ocr_extractor", "ERROR", {}, 0.0, "document_extractions must be a list")

    structured: List[Dict[str, str]] = []
    empty_count = 0
    for ex in extractions_raw:
        if not isinstance(ex, dict):
            continue
        file_name = _safe_text(ex.get("file_name"))
        extracted_text = _safe_text(ex.get("extracted_text"))
        if not extracted_text:
            empty_count += 1
        structured.append({"file_name": file_name, "extracted_text": extracted_text})

    total_text_len = sum(len(x["extracted_text"]) for x in structured)
    confidence = 0.0 if not structured else min(1.0, total_text_len / 2000.0)
    return _tool_result(
        "ocr_extractor",
        "DONE",
        {
            "extractions": structured,
            "extraction_count": len(structured),
            "empty_extractions": empty_count,
            "total_text_len": total_text_len,
        },
        confidence,
    )


def regex_identity_extractor_tool(input_data: Dict[str, Any]) -> Dict[str, Any]:
    text = _safe_text(input_data.get("text"))
    if not text:
        return _tool_result("identity_extractor", "ERROR", {}, 0.0, "Input text is empty")

    def extract_field(pattern: str, source: str) -> str | None:
        match = re.search(pattern, source, re.IGNORECASE)
        return match.group(1).strip() if match else None

    name = extract_field(r"Nom complet\s*:\s*([A-Za-zÀ-ÖØ-öø-ÿ\s'-]+)", text)
    cin = extract_field(r"CIN\s*:\s*([A-Z0-9]+)", text)
    ipp = extract_field(r"(IPP[-\w]+)", text)
    dob = extract_field(r"Date de naissance\s*:\s*([0-9/]+)", text)
    provider = extract_field(r"(Clinique\s+[A-Za-zÀ-ÖØ-öø-ÿ\s'-]+)", text)

    output = {
        "name": name,
        "cin": cin.upper() if cin else None,
        "ipp": ipp,
        "dob": dob,
        "provider": provider,
    }
    detected = sum(1 for v in output.values() if v)
    confidence = detected / 5.0
    return _tool_result("identity_extractor", "DONE", output, confidence)


def document_classifier_tool(input_data: Dict[str, Any]) -> Dict[str, Any]:
    documents = [_safe_text(d) for d in (input_data.get("documents") or []) if _safe_text(d)]
    extractions = input_data.get("document_extractions") or []
    if not isinstance(extractions, list):
        return _tool_result("document_classifier", "ERROR", {}, 0.0, "document_extractions must be a list")

    corpus = _build_corpus(documents, [e for e in extractions if isinstance(e, dict)])
    missing_docs: List[str] = []
    found_docs: List[str] = []
    for req in _REQUIRED:
        req_hit = req in " ".join(documents).lower()
        keyword_hit = any(kw in corpus for kw in _KEYWORD_GROUPS.get(req, ()))
        if req_hit or keyword_hit:
            found_docs.append(req)
        else:
            missing_docs.append(req)

    insurance_only = any(kw in corpus for kw in _INSURANCE_DOC_KEYWORDS) and not found_docs
    doc_type = "incomplete_claim_bundle" if missing_docs else "medical_claim_bundle"
    confidence = 1.0 - (len(missing_docs) / max(1, len(_REQUIRED)))
    return _tool_result(
        "document_classifier",
        "DONE",
        {
            "document_type": doc_type,
            "missing_docs": missing_docs,
            "found_docs": found_docs,
            "insurance_doc_only": insurance_only,
        },
        confidence,
    )


def fraud_pattern_detector_tool(input_data: Dict[str, Any]) -> Dict[str, Any]:
    documents = [_safe_text(d) for d in (input_data.get("documents") or []) if _safe_text(d)]
    extractions = [e for e in (input_data.get("document_extractions") or []) if isinstance(e, dict)]
    amount_raw = input_data.get("amount")
    try:
        amount = float(amount_raw or 0.0)
    except (TypeError, ValueError):
        amount = 0.0

    corpus = _build_corpus(documents, extractions)
    fraud_hits = [kw for kw in _FRAUD_TEXT_SIGNALS if kw in corpus]
    suspicious_doc_names = []
    for doc in documents:
        dl = doc.lower()
        if any(sig in dl for sig in _SUSPICIOUS_FILENAME_SIGNALS):
            suspicious_doc_names.append(doc)

    high_amount_low_docs = bool(amount > 20000 and max(len(documents), len(extractions)) < 3)
    risk_indicators = len(fraud_hits) + len(suspicious_doc_names) + (1 if high_amount_low_docs else 0)
    confidence = min(1.0, 0.4 + (0.2 * risk_indicators)) if risk_indicators else 0.85
    return _tool_result(
        "fraud_detector",
        "DONE",
        {
            "fraud_text_signals": fraud_hits,
            "suspicious_doc_names": suspicious_doc_names,
            "high_amount_doc_requirement": high_amount_low_docs,
            "risk_indicators": risk_indicators,
        },
        confidence,
    )


def register_core_tools() -> None:
    register_tool("ocr_extractor", ocr_extractor_tool)
    register_tool("identity_extractor", regex_identity_extractor_tool)
    register_tool("document_classifier", document_classifier_tool)
    register_tool("fraud_detector", fraud_pattern_detector_tool)
