"""Identity Agent — Moroccan CIN fraud-analyst approach.

Performs structural validation, semantic realism checks, OCR cross-validation,
identity consistency analysis, and contextual fraud signal detection.

NEVER claims a CIN is "officially valid" or "verified with authority".
"""

from __future__ import annotations

import json
import os
import re
import threading
from difflib import SequenceMatcher
from queue import Empty, Queue
from typing import Any, Dict, List, Optional, Tuple

from claimguard.agents.base_agent import BaseAgent
from claimguard.agents.memory_utils import process_memory_context
from claimguard.agents.security_utils import (
    coerce_risk_output,
    hash_text,
    log_security_event,
    sanitize_input,
    score_to_risk_score,
)
from claimguard.llm_factory import get_llm
from claimguard.llm_tracking import parse_llm_json, safe_tracked_llm_call, tracked_agent_context

IDENTITY_SYSTEM_PROMPT = """You are an identity verification specialist operating as a fraud analyst.

You NEVER claim a CIN is "officially valid" or "verified with authority".
You NEVER assume identity is valid just because it matches format.
You MUST base your reasoning on the provided OCR text and verified fields. You MUST produce DIFFERENT outputs for different inputs. Generic responses are forbidden.

You assess:
- structural validity of Moroccan CIN and IPP
- contextual consistency across documents and OCR
- fraud likelihood based on patterns and contradictions

You treat all input as probabilistic signals.
Escalate uncertainty when data conflicts or identifiers look suspicious.

MEMORY AWARENESS:
- When memory_context is present, check if the same CIN appears in past fraud cases.
- If this CIN was used in previous fraud: flag identity reuse and increase risk significantly.
- If memory shows multiple claims from the same identity: flag high-frequency identity abuse.
- Memory is ADVISORY — never override your structural analysis with memory alone.
- If memory contradicts current data: note the contradiction without overriding."""

_CIN_PATTERN = re.compile(r"^[A-Z]{1,2}[0-9]{5,6}$")
_CIN_EXTRACT_PATTERN = re.compile(r"\b([A-Z]{1,2}[0-9]{5,6})\b")
_IPP_PATTERN = re.compile(r"^\d{6,10}$")

_SUSPICIOUS_LETTER_COMBOS = frozenset({
    "QQ", "ZZ", "XX", "YY", "WW", "KK",
})

_COMMON_PREFIXES = frozenset({
    "A", "B", "BA", "BB", "BE", "BH", "BJ", "BK", "BL", "BM",
    "C", "CB", "CD", "D", "DA", "DB", "DJ", "E", "EA", "EB",
    "EE", "F", "FA", "G", "GA", "H", "HA", "HH", "I", "IA",
    "J", "JA", "JB", "JC", "JE", "JF", "JH", "JK", "JM", "JT",
    "K", "KB", "L", "LA", "M", "MA", "MC", "N", "PA", "PB",
    "Q", "R", "S", "SA", "SB", "SH", "SJ", "SK", "SL", "T",
    "TA", "U", "UA", "V", "VA", "W", "WA", "WB", "X", "Z", "ZT",
})

_IDENTITY_KEYWORDS: tuple[str, ...] = (
    "nom", "prénom", "prenom", "name",
    "patient", "assuré", "assure",
    "bénéficiaire", "beneficiaire",
    "cin", "carte nationale", "carte d'identité", "carte d identite",
    "numéro d'assuré", "numero d assure",
    "immatriculation",
    "date de naissance", "né le", "ne le", "née le", "nee le",
)

_DOB_PATTERNS = (
    re.compile(r"\b(\d{2})[/\-.](\d{2})[/\-.](\d{4})\b"),
    re.compile(r"\b(\d{4})[/\-.](\d{2})[/\-.](\d{2})\b"),
)

IDENTITY_PROMPT = """
You are an information extraction engine.

Your task is to extract identity fields from a medical document.

You MUST return VALID JSON.

DO NOT explain.
DO NOT add text.
DO NOT return anything outside JSON.

If you cannot find a field, return null.

If you fail to return valid JSON, the system will consider your answer invalid.

Return EXACTLY this schema:

{{
  "name": string | null,
  "cin": string | null,
  "dob": string | null,
  "is_valid": boolean,
  "confidence": number (0-100),
  "issues": array
}}

DOCUMENT:
{document_text}
"""

_IDENTITY_REGEX_NAME = re.compile(r"Nom complet\s*:\s*(.+)", re.IGNORECASE)
_IDENTITY_REGEX_CIN = re.compile(r"CIN\s*:\s*(\w+)", re.IGNORECASE)
_IDENTITY_REGEX_DOB = re.compile(r"Date de naissance\s*:\s*([\d/]+)", re.IGNORECASE)


def _extract_identity_llm(document_text: str, agent_name: str) -> Dict[str, Any]:
    safe_text = sanitize_input(document_text or "")
    if len(safe_text.strip()) < 100:
        raise ValueError("Document text too short for identity extraction (<100 chars)")
    print("[LLM INPUT LENGTH]", len(safe_text))
    prompt = IDENTITY_PROMPT.format(document_text=safe_text)
    timeout_s = float(os.getenv("CLAIMGUARD_IDENTITY_LLM_TIMEOUT_S", "6"))
    llm = get_llm("simple", tracked=False)
    result_queue: Queue[Dict[str, Any]] = Queue(maxsize=1)
    log_state = {"received": False, "parsed": False, "used": False}

    def _invoke_and_parse(prompt_text: str) -> Dict[str, Any]:
        with tracked_agent_context(agent_name):
            result = safe_tracked_llm_call(agent_name, prompt_text, llm.invoke)
            if not isinstance(result, dict):
                raise RuntimeError("LLM_RESPONSE_LOST")
            raw_response = str(result.get("response") or "")
            if not raw_response.strip():
                raise RuntimeError("LLM_RESPONSE_LOST")
            print(f"[AGENT RAW RESPONSE] {raw_response[:500]}")
            print("RAW LLM RESPONSE:", raw_response)
            print("[LLM RESPONSE RECEIVED]")
            log_state["received"] = True
            parsed_local = result.get("parsed")
            if parsed_local is None:
                raise RuntimeError("LLM_RESPONSE_LOST")
            print("[LLM RESPONSE PARSED]")
        log_state["parsed"] = True
        if not isinstance(parsed_local, dict):
            parsed_local = parse_llm_json(raw_response)
        if not isinstance(parsed_local, dict):
            return {"error": "INVALID_JSON", "raw": raw_response}
        print("[LLM OUTPUT USED]")
        log_state["used"] = True
        return parsed_local

    def _worker() -> None:
        parsed = _invoke_and_parse(prompt)
        if parsed.get("error") == "INVALID_JSON":
            retry_prompt = f"{prompt}\n\nReturn ONLY valid JSON. No explanation."
            parsed = _invoke_and_parse(retry_prompt)
        result_queue.put(parsed)

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()
    worker.join(timeout=timeout_s)
    if worker.is_alive():
        return {"error": "IDENTITY_LLM_TIMEOUT", "raw": ""}
    try:
        parsed = result_queue.get_nowait()
    except Empty:
        return {"error": "IDENTITY_LLM_NO_RESULT", "raw": ""}
    if not log_state["received"]:
        raise RuntimeError("IdentityAgent missing [LLM RESPONSE RECEIVED] log.")
    if not log_state["parsed"] or not isinstance(parsed, dict):
        raise RuntimeError("IdentityAgent missing [LLM RESPONSE PARSED] payload.")
    if not log_state["used"]:
        raise RuntimeError("IdentityAgent missing [LLM OUTPUT USED] trace.")
    if parsed.get("error"):
        print("[AGENT VALIDATION RESULT] failed: parse_error")
        return parsed
    required_fields = ("name", "cin", "dob", "is_valid", "confidence")
    missing_fields = [field for field in required_fields if field not in parsed]
    if missing_fields:
        print(f"[AGENT VALIDATION RESULT] failed: missing_fields={missing_fields}")
        return {"error": "MISSING_FIELDS", "missing_fields": missing_fields, "raw": json.dumps(parsed, default=str)}
    issues = parsed.get("issues", [])
    if not isinstance(issues, list):
        issues = [str(issues)]
    print("[AGENT VALIDATION RESULT] passed")
    return {
        "name": str(parsed.get("name") or "").strip(),
        "cin": str(parsed.get("cin") or "").strip().upper(),
        "dob": str(parsed.get("dob") or "").strip(),
        "is_valid": bool(parsed.get("is_valid", False)),
        "confidence": max(0, min(100, int(parsed.get("confidence", 0) or 0))),
        "issues": [str(i) for i in issues if str(i).strip()],
    }


def _regex_identity_fallback(document_text: str) -> Dict[str, Any]:
    text = document_text or ""
    extracted: Dict[str, Any] = {}

    name_match = _IDENTITY_REGEX_NAME.search(text)
    cin_match = _IDENTITY_REGEX_CIN.search(text)
    dob_match = _IDENTITY_REGEX_DOB.search(text)

    if name_match:
        extracted["name"] = name_match.group(1).strip()
    if cin_match:
        extracted["cin"] = cin_match.group(1).strip().upper()
    if dob_match:
        extracted["dob"] = dob_match.group(1).strip()

    return extracted


def _clamp(value: int, lo: int = 0, hi: int = 100) -> int:
    return max(lo, min(hi, value))


def _normalize_text(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[àâä]", "a", t)
    t = re.sub(r"[éèêë]", "e", t)
    t = re.sub(r"[ïî]", "i", t)
    t = re.sub(r"[ôö]", "o", t)
    t = re.sub(r"[ùûü]", "u", t)
    t = re.sub(r"[ç]", "c", t)
    t = re.sub(r"\s+", " ", t)
    return t


def _name_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    na, nb = _normalize_text(a), _normalize_text(b)
    if na == nb:
        return 1.0
    return SequenceMatcher(None, na, nb).ratio()


# ── CIN structure validation ───────────────────────────────────────────

class CINStructureResult:
    __slots__ = ("valid", "reasons")

    def __init__(self, valid: bool, reasons: List[str]):
        self.valid = valid
        self.reasons = reasons


def _validate_cin_structure(cin: str) -> CINStructureResult:
    """Strict structural check per Moroccan CIN rules."""
    reasons: List[str] = []

    if not cin:
        return CINStructureResult(False, ["CIN is empty"])

    if cin != cin.strip():
        reasons.append("CIN contains leading/trailing whitespace")
    cin_clean = cin.strip()

    if " " in cin_clean:
        return CINStructureResult(False, ["CIN contains spaces"])
    if re.search(r"[^A-Za-z0-9]", cin_clean):
        return CINStructureResult(False, ["CIN contains symbols or special characters"])
    if cin_clean != cin_clean.upper():
        reasons.append("CIN contains lowercase letters (normalized to uppercase)")
        cin_clean = cin_clean.upper()

    if not _CIN_PATTERN.match(cin_clean):
        letter_part = re.match(r"^([A-Z]*)", cin_clean)
        digit_part = re.search(r"([0-9]*)$", cin_clean)
        n_letters = len(letter_part.group(1)) if letter_part else 0
        n_digits = len(digit_part.group(1)) if digit_part else 0

        if n_letters == 0:
            reasons.append("CIN missing letter prefix")
        elif n_letters > 2:
            reasons.append(f"CIN has {n_letters} letters (max 2 allowed)")
        if n_digits < 5:
            reasons.append(f"CIN has {n_digits} digits (min 5 required)")
        elif n_digits > 6:
            reasons.append(f"CIN has {n_digits} digits (max 6 allowed)")
        if not reasons:
            reasons.append("CIN does not match expected pattern [A-Z]{1,2}[0-9]{5,6}")
        return CINStructureResult(False, reasons)

    return CINStructureResult(True, reasons)


# ── Semantic realism checks ────────────────────────────────────────────

def _check_suspicious_patterns(cin: str) -> List[str]:
    """Flag suspicious-but-not-invalid numeric/letter patterns."""
    flags: List[str] = []
    cin_upper = cin.strip().upper()

    letter_part = re.match(r"^([A-Z]{1,2})", cin_upper)
    digit_part = re.search(r"([0-9]{5,6})$", cin_upper)
    if not letter_part or not digit_part:
        return flags

    prefix = letter_part.group(1)
    digits = digit_part.group(1)

    if len(prefix) == 2 and prefix in _SUSPICIOUS_LETTER_COMBOS:
        flags.append(f"Uncommon letter prefix '{prefix}'")

    if prefix not in _COMMON_PREFIXES:
        flags.append(f"Prefix '{prefix}' is not among commonly seen Moroccan CIN prefixes")

    if len(set(digits)) == 1:
        flags.append(f"All digits are identical ({digits})")

    seq_asc = "".join(str(i % 10) for i in range(int(digits[0]), int(digits[0]) + len(digits)))
    seq_desc = "".join(str(i % 10) for i in range(int(digits[0]), int(digits[0]) - len(digits), -1))
    if digits == seq_asc:
        flags.append(f"Digits form an ascending sequence ({digits})")
    elif digits == seq_desc:
        flags.append(f"Digits form a descending sequence ({digits})")

    if int(digits) < 100:
        flags.append(f"Extremely low numeric range ({digits})")

    return flags


# ── OCR cross-validation ───────────────────────────────────────────────

def _build_ocr_corpus(claim_data: Dict[str, Any]) -> str:
    parts: List[str] = []
    for doc in (claim_data.get("documents") or []):
        parts.append(str(doc))
    for ex in (claim_data.get("document_extractions") or []):
        if isinstance(ex, dict):
            parts.append(ex.get("file_name") or "")
            parts.append(ex.get("extracted_text") or "")
    return " ".join(parts)


def _extract_cins_from_text(text: str) -> List[str]:
    return _CIN_EXTRACT_PATTERN.findall(text.upper())


def _ocr_cross_validate(
    input_cin: str, corpus: str
) -> Tuple[bool, bool, List[str]]:
    """Returns (cin_found_in_ocr, cin_matches, observations)."""
    observations: List[str] = []
    if not corpus.strip():
        return False, False, ["No OCR/document text available for cross-validation"]

    ocr_cins = _extract_cins_from_text(corpus)
    cin_upper = input_cin.strip().upper()

    if not ocr_cins:
        return False, False, ["No CIN found in OCR/document text"]

    observations.append(f"Found {len(ocr_cins)} CIN-like pattern(s) in documents: {ocr_cins}")

    if cin_upper in ocr_cins:
        return True, True, observations + [
            "Input CIN matches a CIN found in document text"
        ]

    observations.append(
        f"Input CIN '{cin_upper}' does NOT match any CIN in documents {ocr_cins}"
    )
    return True, False, observations


# ── Identity consistency ───────────────────────────────────────────────

def _extract_name_from_claim(claim_data: Dict[str, Any]) -> Optional[str]:
    for key in ("patient_name", "full_name", "name", "nom_complet"):
        val = claim_data.get(key)
        if val and str(val).strip():
            return str(val).strip()
    identity = claim_data.get("identity")
    if isinstance(identity, dict):
        for key in ("full_name", "name", "nom_complet", "nom"):
            val = identity.get(key)
            if val and str(val).strip():
                return str(val).strip()
    return None


def _extract_name_from_ocr(corpus: str) -> Optional[str]:
    """Heuristic: look for 'nom.*:' patterns in OCR text."""
    patterns = [
        re.compile(r"nom\s*(?:et\s*pr[ée]nom)?\s*[:]\s*([A-Za-zÀ-ÿ\s\-]+)", re.IGNORECASE),
        re.compile(r"name\s*[:]\s*([A-Za-zÀ-ÿ\s\-]+)", re.IGNORECASE),
    ]
    for p in patterns:
        m = p.search(corpus)
        if m:
            name = m.group(1).strip()
            name = re.sub(r"\s+", " ", name)
            if len(name) >= 3:
                return name
    return None


def _extract_dob_from_claim(claim_data: Dict[str, Any]) -> Optional[str]:
    for key in ("date_of_birth", "dob", "date_naissance"):
        val = claim_data.get(key)
        if val and str(val).strip():
            return str(val).strip()
    identity = claim_data.get("identity")
    if isinstance(identity, dict):
        for key in ("date_of_birth", "dob", "date_naissance"):
            val = identity.get(key)
            if val and str(val).strip():
                return str(val).strip()
    return None


def _extract_dob_from_ocr(corpus: str) -> Optional[str]:
    for p in _DOB_PATTERNS:
        m = p.search(corpus)
        if m:
            return m.group(0)
    return None


# ── Main agent ─────────────────────────────────────────────────────────

class IdentityAgent(BaseAgent):
    system_prompt: str = IDENTITY_SYSTEM_PROMPT

    def __init__(self):
        super().__init__(
            name="Identity Agent",
            role="Identity Verification Specialist",
            goal=(
                "Assess Moroccan CIN structural validity, contextual consistency, "
                "and fraud likelihood — never claim official verification"
            ),
        )

    @staticmethod
    def _resolve_cin(claim_data: Dict[str, Any]) -> str:
        """Pull CIN from the most likely field."""
        identity = claim_data.get("identity")
        if isinstance(identity, dict):
            for key in ("cin", "CIN", "carte_nationale", "national_id"):
                val = identity.get(key)
                if val and str(val).strip():
                    return str(val).strip()
        for key in ("patient_id", "cin", "CIN"):
            val = claim_data.get(key)
            if val and str(val).strip():
                return str(val).strip()
        return ""

    @staticmethod
    def _resolve_ipp(claim_data: Dict[str, Any]) -> str:
        identity = claim_data.get("identity")
        if isinstance(identity, dict):
            for key in ("ipp", "IPP"):
                val = identity.get(key)
                if val and str(val).strip():
                    return str(val).strip()
        return ""

    def analyze(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        cin_raw = self._resolve_cin(claim_data).upper().strip()
        ipp_raw = self._resolve_ipp(claim_data).strip()
        corpus = sanitize_input(_build_ocr_corpus(claim_data))
        tool_results = self.run_tool_pipeline(
            claim_data,
            {
                "identity_extractor": {"text": corpus},
                "fraud_detector": {
                    "documents": claim_data.get("documents") or [],
                    "document_extractions": claim_data.get("document_extractions") or [],
                    "amount": claim_data.get("amount", 0),
                },
            },
        )
        extracted = dict(tool_results["identity_extractor"].get("output") or {})
        if not cin_raw and extracted.get("cin"):
            cin_raw = str(extracted.get("cin") or "").upper().strip()

        # SCORE-FIX: deterministic scoring contract for IdentityAgent.
        score = 100.0
        reasons: List[str] = []
        flags: List[str] = []
        ocr_upper = corpus.upper()
        cin_found_in_ocr = bool(cin_raw) and cin_raw in ocr_upper
        ipp_found_in_ocr = bool(ipp_raw) and ipp_raw in ocr_upper
        if not cin_found_in_ocr:
            score -= 40
            flags.append("CIN_NOT_FOUND")
            reasons.append("CIN introuvable dans le texte OCR")
        if not ipp_found_in_ocr:
            score -= 20
            flags.append("IPP_NOT_FOUND")
            reasons.append("IPP introuvable dans le texte OCR")
        claim_name = _extract_name_from_claim(claim_data) or ""
        ocr_name = _extract_name_from_ocr(corpus) or ""
        if claim_name and ocr_name:
            similarity = _name_similarity(claim_name, ocr_name)
            if similarity < 0.80:
                score -= 15
                flags.append("NAME_FUZZY_MISMATCH")
                reasons.append("Similarite nom faible entre declaration et OCR")
        structure = _validate_cin_structure(cin_raw) if cin_raw else CINStructureResult(False, ["CIN missing"])
        if not _CIN_PATTERN.match(cin_raw or ""):
            score -= 20
            flags.append("CIN_FORMAT_INVALID")
            reasons.append("Format CIN invalide")
        fraud_indicators = int(tool_results["fraud_detector"].get("output", {}).get("risk_indicators", 0) or 0)
        if fraud_indicators > 0:
            score -= min(30, fraud_indicators * 10)
            flags.append("IDENTITY_FRAUD_INDICATORS")
            reasons.append("Indicateurs de risque identitaire detectes")
        score = max(0.0, min(100.0, score))
        decision = score > 60.0

        llm_fallback = self.should_use_llm_fallback(tool_results)
        reasoning = "; ".join(reasons) if reasons else "Identity signals look consistent"
        if llm_fallback:
            print("[LLM FALLBACK USED] True")
            llm_reasoning, _ = run_agent_consistency_check(
                agent_name=self.name,
                claim_data=claim_data,
                draft_reasoning=reasoning,
            )
            reasoning = llm_reasoning
        else:
            print("[LLM FALLBACK USED] False")

        payload = {
            "agent_name": self.name,
            "status": "PASS" if score >= 70 else ("REVIEW" if score >= 40 else "FAIL"),
            "decision": decision,
            "score": round(score, 2),
            "confidence": round(min(100.0, score + 10.0), 2),
            "reasoning": reasoning,
            "explanation": reasoning,
            "signals": list(flags),
            "data_used": extracted,
            "details": {
                "cin": cin_raw or None,
                "ipp": ipp_raw or None,
                "cin_structure_valid": structure.valid,
                "contradictions": [],
                "identity_extraction": extracted,
                "tool_results": tool_results,
                "cin_found_in_ocr": cin_found_in_ocr,
                "ipp_found_in_ocr": ipp_found_in_ocr,
            },
        }
        assert payload["score"] is not None
        assert str(payload["explanation"]).strip() != ""
        self.enforce_tool_trace(tool_results, llm_fallback)
        return self._build_result(  # SCORE-FIX
            status="DONE",
            score=float(payload["score"]),
            reason=reasoning,
            output=payload,
            flags=flags,
        )
