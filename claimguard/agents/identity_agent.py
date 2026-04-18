"""Identity Agent — Moroccan CIN fraud-analyst approach.

Performs structural validation, semantic realism checks, OCR cross-validation,
identity consistency analysis, and contextual fraud signal detection.

NEVER claims a CIN is "officially valid" or "verified with authority".
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from .base_agent import BaseAgent
from .memory_utils import process_memory_context
from .security_utils import (
    bump_risk,
    coerce_risk_output,
    detect_prompt_injection,
    hash_text,
    log_security_event,
    sanitize_input,
    score_to_risk_score,
)

IDENTITY_SYSTEM_PROMPT = """You are an identity verification specialist operating as a fraud analyst.

You NEVER claim a CIN is "officially valid" or "verified with authority".
You NEVER assume identity is valid just because it matches format.

You assess:
- structural validity of Moroccan CIN
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

    def analyze(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        cin_raw = self._resolve_cin(claim_data)
        history = claim_data.get("history", [])
        if not isinstance(history, list):
            history = []

        raw_corpus = _build_ocr_corpus(claim_data)
        sanitized_corpus = sanitize_input(raw_corpus)

        injection_detected = detect_prompt_injection(raw_corpus)

        score = 100
        confidence = 90
        observations: List[str] = []
        missing_data: List[str] = []
        contradictions: List[str] = []
        fraud_signals: List[str] = []
        reasoning_steps: List[str] = []
        details: Dict[str, Any] = {"system_prompt_version": "identity_v3_cin_analyst"}

        # ── 1. CIN structure validation ────────────────────────────────
        reasoning_steps.append("Step 1: Validate CIN structural format")

        if not cin_raw:
            format_valid = False
            score -= 70
            confidence = min(confidence, 40)
            observations.append("No CIN provided — cannot perform identity analysis")
            missing_data.append("cin")
            reasoning_steps.append("CIN is missing; applied -70 penalty")
        else:
            structure = _validate_cin_structure(cin_raw)
            format_valid = structure.valid

            if not structure.valid:
                score -= 70
                confidence = min(confidence, 50)
                observations.extend(structure.reasons)
                reasoning_steps.append(
                    f"CIN '{cin_raw}' failed structure validation: "
                    + "; ".join(structure.reasons)
                    + " → applied -70 penalty"
                )
            else:
                if structure.reasons:
                    observations.extend(structure.reasons)
                reasoning_steps.append(
                    f"CIN '{cin_raw}' is structurally valid "
                    "(matches [A-Z]{{1,2}}[0-9]{{5,6}})"
                )

        details["cin_input"] = cin_raw or None
        details["format_valid"] = format_valid

        # ── 2. Semantic realism check ──────────────────────────────────
        reasoning_steps.append("Step 2: Check for suspicious numeric/letter patterns")

        suspicious_patterns: List[str] = []
        if cin_raw and format_valid:
            suspicious_patterns = _check_suspicious_patterns(cin_raw)
            if suspicious_patterns:
                score -= 15
                confidence = min(confidence, 75)
                observations.extend(suspicious_patterns)
                reasoning_steps.append(
                    f"Suspicious pattern(s) detected: {suspicious_patterns} → applied -15 penalty"
                )
            else:
                reasoning_steps.append("No suspicious patterns detected in CIN digits/prefix")
        elif cin_raw and not format_valid:
            reasoning_steps.append("Skipped realism check (CIN format already invalid)")

        details["suspicious_patterns"] = suspicious_patterns

        # ── 3. OCR cross-validation ────────────────────────────────────
        reasoning_steps.append("Step 3: Cross-validate CIN against OCR/document text")

        doc_count = max(
            len(claim_data.get("documents") or []),
            len(claim_data.get("document_extractions") or []),
        )
        ocr_checked = False
        ocr_match = False

        if doc_count == 0:
            missing_data.append("documents")
            observations.append("No documents provided — OCR cross-validation impossible")
            score -= 30
            confidence = min(confidence, 50)
            reasoning_steps.append(
                "No documents available; cannot cross-validate CIN → applied -30 penalty"
            )
        elif not cin_raw:
            reasoning_steps.append("No CIN to cross-validate against documents")
        else:
            cin_found_in_ocr, cin_matches, ocr_obs = _ocr_cross_validate(
                cin_raw, sanitized_corpus
            )
            ocr_checked = True
            observations.extend(ocr_obs)

            if not cin_found_in_ocr:
                score -= 30
                confidence = min(confidence, 60)
                reasoning_steps.append(
                    "CIN not found anywhere in document text → applied -30 penalty"
                )
            elif not cin_matches:
                ocr_match = False
                score -= 40
                confidence = min(confidence, 50)
                contradictions.append(
                    f"Input CIN '{cin_raw}' contradicts CIN(s) found in documents"
                )
                reasoning_steps.append(
                    "CIN mismatch between input and documents → applied -40 penalty"
                )
            else:
                ocr_match = True
                reasoning_steps.append("CIN confirmed present in document text (OCR match)")

        details["ocr_checked"] = ocr_checked
        details["ocr_match"] = ocr_match

        # ── 4. Identity consistency ────────────────────────────────────
        reasoning_steps.append("Step 4: Check identity field consistency (name, DOB)")

        input_name = _extract_name_from_claim(claim_data)
        ocr_name = _extract_name_from_ocr(sanitized_corpus) if sanitized_corpus else None

        if not input_name:
            missing_data.append("patient_name")
        if input_name and ocr_name:
            sim = _name_similarity(input_name, ocr_name)
            details["name_similarity"] = round(sim, 3)
            if sim < 0.5:
                score -= 25
                confidence = min(confidence, 55)
                contradictions.append(
                    f"Name mismatch: input='{input_name}' vs OCR='{ocr_name}' "
                    f"(similarity {sim:.0%})"
                )
                reasoning_steps.append(
                    f"Major name mismatch (similarity={sim:.0%}) → applied -25 penalty"
                )
            elif sim < 0.8:
                score -= 10
                observations.append(
                    f"Minor name discrepancy: input='{input_name}' vs OCR='{ocr_name}' "
                    f"(similarity {sim:.0%})"
                )
                reasoning_steps.append(
                    f"Minor name discrepancy (similarity={sim:.0%}) → applied -10 penalty"
                )
            else:
                reasoning_steps.append(
                    f"Name consistent across sources (similarity={sim:.0%})"
                )
        elif input_name and not ocr_name:
            observations.append("Could not extract name from documents for comparison")
            reasoning_steps.append("Name present in input but not extractable from OCR")
        elif not input_name and ocr_name:
            observations.append(f"Name found in OCR ('{ocr_name}') but not in input fields")
            reasoning_steps.append("Name found in OCR but missing from claim input")

        input_dob = _extract_dob_from_claim(claim_data)
        ocr_dob = _extract_dob_from_ocr(sanitized_corpus) if sanitized_corpus else None

        if not input_dob:
            missing_data.append("date_of_birth")
            score -= 20
            confidence = min(confidence, 65)
            reasoning_steps.append("Date of birth missing → applied -20 penalty")
        elif ocr_dob:
            dob_norm_input = re.sub(r"[/\-.]", "", input_dob)
            dob_norm_ocr = re.sub(r"[/\-.]", "", ocr_dob)
            if dob_norm_input == dob_norm_ocr:
                reasoning_steps.append("Date of birth consistent across input and OCR")
            else:
                score -= 15
                contradictions.append(
                    f"DOB mismatch: input='{input_dob}' vs OCR='{ocr_dob}'"
                )
                reasoning_steps.append(
                    f"DOB mismatch (input='{input_dob}' vs OCR='{ocr_dob}') → applied -15 penalty"
                )

        # ── 5. Contextual fraud signals ────────────────────────────────
        reasoning_steps.append("Step 5: Evaluate contextual fraud signals")

        if len(history) > 0 and cin_raw:
            seen_cins = set()
            for h in history:
                h_pid = str(h.get("patient_id", ""))
                if h_pid:
                    seen_cins.add(h_pid.strip().upper())

            cin_upper = cin_raw.strip().upper()
            patient_names_in_history = set(
                str(h.get("patient_name", "")).strip()
                for h in history if h.get("patient_name")
            )

            if len(patient_names_in_history) > 1:
                fraud_signals.append(
                    f"Multiple distinct names associated with this identity in history: "
                    f"{patient_names_in_history}"
                )
                score -= 20
                confidence = min(confidence, 55)
                reasoning_steps.append(
                    f"Multiple names in history ({len(patient_names_in_history)}) → "
                    "applied -20 penalty"
                )

            if seen_cins and cin_upper not in seen_cins:
                fraud_signals.append(
                    f"Input CIN '{cin_raw}' not seen in claim history "
                    f"(history IDs: {seen_cins})"
                )
                score -= 10
                reasoning_steps.append(
                    "CIN not present in claim history → applied -10 penalty"
                )
        elif len(history) == 0:
            observations.append("No claim history to cross-reference identity against")
            reasoning_steps.append("No history available for contextual checks")

        if len(history) > 5:
            fraud_signals.append(
                f"High claim frequency ({len(history)} claims) may indicate identity misuse"
            )
            score -= 10
            reasoning_steps.append(
                f"High claim frequency ({len(history)}) → applied -10 penalty"
            )

        if cin_raw and doc_count > 0 and not any(
            cin_raw.upper() in str(d).upper()
            for d in (claim_data.get("documents") or [])
        ) and not any(
            cin_raw.upper() in (ex.get("extracted_text") or "").upper()
            for ex in (claim_data.get("document_extractions") or [])
            if isinstance(ex, dict)
        ):
            fraud_signals.append(
                "CIN exists in input but is not referenced in any submitted document"
            )

        if doc_count > 0:
            ext_list = claim_data.get("document_extractions") or []
            empty_extractions = sum(
                1 for ex in ext_list
                if isinstance(ex, dict) and not (ex.get("extracted_text") or "").strip()
            )
            if ext_list and empty_extractions == len(ext_list):
                fraud_signals.append("All documents have empty/unreadable text — possible poor quality or tampering")
                score -= 10
                reasoning_steps.append("All document extractions empty → applied -10 penalty")

        if injection_detected:
            fraud_signals.append("Prompt injection patterns detected in document text")
            score -= 15
            confidence = min(confidence, 50)
            reasoning_steps.append(
                "Prompt injection detected in document corpus → applied -15 penalty"
            )

        details["fraud_signals_count"] = len(fraud_signals)

        # ── 6. Scoring — clamp ─────────────────────────────────────────
        score = _clamp(score)

        # ── 7. Confidence rules ────────────────────────────────────────
        reasoning_steps.append("Step 6: Apply confidence rules")

        if missing_data or contradictions:
            confidence = min(confidence, 69)
            reasoning_steps.append(
                "Missing data or contradictions present → confidence capped below 70"
            )

        if format_valid and ocr_match and not contradictions and not suspicious_patterns:
            confidence = max(confidence, 85)
        else:
            confidence = min(confidence, 84)

        confidence = _clamp(confidence)

        # ── 8. Build explanation ───────────────────────────────────────
        reasoning_steps.append("Step 7: Build final assessment")

        explanation_parts: List[str] = []
        if not cin_raw:
            explanation_parts.append(
                "No CIN was provided; identity assessment is severely limited."
            )
        elif format_valid:
            explanation_parts.append(
                f"CIN '{cin_raw}' is structurally valid (matches Moroccan CIN format)."
            )
        else:
            explanation_parts.append(
                f"CIN '{cin_raw}' does NOT match the expected Moroccan CIN format."
            )

        if ocr_checked:
            if ocr_match:
                explanation_parts.append(
                    "CIN is consistent with provided documents."
                )
            else:
                explanation_parts.append(
                    "CIN could NOT be confirmed in the submitted documents, "
                    "indicating a potential inconsistency."
                )

        if contradictions:
            explanation_parts.append(
                f"Detected {len(contradictions)} contradiction(s) across identity sources."
            )
        if fraud_signals:
            explanation_parts.append(
                f"Identified {len(fraud_signals)} contextual fraud signal(s)."
            )
        if missing_data:
            explanation_parts.append(
                f"Missing data points: {', '.join(missing_data)}."
            )

        explanation_parts.append(
            f"Final score: {score}/100 | Confidence: {confidence}/100."
        )
        explanation = " ".join(explanation_parts)

        # ── 9. Fail conditions ─────────────────────────────────────────
        if score == 100 and not ocr_checked:
            raise RuntimeError(
                "IdentityAgent: score=100 but OCR was not checked. "
                "This violates the mandatory OCR cross-validation requirement."
            )
        if not reasoning_steps:
            raise RuntimeError(
                "IdentityAgent: no reasoning_steps produced. "
                "Analysis must always document its reasoning."
            )
        if len(explanation) < 20:
            raise RuntimeError(
                "IdentityAgent: explanation is too generic/short. "
                "A meaningful assessment is required."
            )

        # ── Structured output ──────────────────────────────────────────
        cin_analysis = {
            "format_valid": format_valid,
            "suspicious_patterns": suspicious_patterns,
            "ocr_match": ocr_match,
        }

        details["cin_analysis"] = cin_analysis
        details["observations"] = observations
        details["missing_data"] = missing_data
        details["contradictions"] = contradictions
        details["fraud_signals"] = fraud_signals
        details["reasoning_steps"] = reasoning_steps
        details["confidence"] = confidence

        # ── Security layer (mirrors DocumentAgent) ─────────────────────
        all_flags: List[str] = []
        if injection_detected:
            all_flags.append("prompt_injection_detected")
        if not format_valid and cin_raw:
            all_flags.append("invalid_cin_format")
        if contradictions:
            all_flags.append("identity_contradictions")
        if missing_data:
            all_flags.append("missing_identity_data")

        risk_base = score_to_risk_score(float(score))
        if injection_detected:
            risk_base = max(0.5, bump_risk(risk_base, 0.1))

        structured = {
            "risk_score": risk_base,
            "flags": list(dict.fromkeys(all_flags)),
            "explanation": explanation,
        }

        def rebuild() -> Dict[str, Any]:
            return {
                "risk_score": 0.5,
                "flags": list(dict.fromkeys([*all_flags, "validation_error"])),
                "explanation": explanation,
            }

        validated = coerce_risk_output(structured, rebuild=rebuild)

        fp = hash_text(json.dumps(
            {"cin": cin_raw, "doc_count": doc_count, "history_len": len(history)},
            sort_keys=True,
            default=str,
        ))
        final_risk = float(validated.risk_score)
        log_security_event(
            agent_name=self.name,
            payload_fingerprint=fp,
            flags=validated.flags,
            risk_score=final_risk,
        )

        decision = score > 60

        # Memory context integration
        memory_adjusted_score, memory_insights = process_memory_context(
            agent_name=self.name,
            claim_data=claim_data,
            current_score=float(score),
            current_cin=cin_raw,
        )
        if memory_adjusted_score != float(score):
            score = _clamp(int(memory_adjusted_score))
            decision = score > 60
        details["memory_insights"] = memory_insights

        return {
            "agent_name": self.name,
            "decision": decision,
            "score": round(score, 2),
            "reasoning": "; ".join(reasoning_steps),
            "details": details,
            "cin_analysis": cin_analysis,
            "observations": observations,
            "missing_data": missing_data,
            "contradictions": contradictions,
            "fraud_signals": fraud_signals,
            "reasoning_steps": reasoning_steps,
            "confidence": confidence,
            "explanation": explanation,
            "risk_score": final_risk,
            "flags": validated.flags,
            "memory_insights": memory_insights,
        }
