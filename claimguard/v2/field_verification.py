from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple

_CURRENCY_TOKENS = ("mad", "dh", "dirham", "dirhams")
_MONTH_MAP = {
    "january": 1,
    "janvier": 1,
    "february": 2,
    "fevrier": 2,
    "febrier": 2,
    "march": 3,
    "mars": 3,
    "april": 4,
    "avril": 4,
    "may": 5,
    "mai": 5,
    "june": 6,
    "juin": 6,
    "july": 7,
    "juillet": 7,
    "august": 8,
    "aout": 8,
    "september": 9,
    "septembre": 9,
    "october": 10,
    "octobre": 10,
    "november": 11,
    "novembre": 11,
    "december": 12,
    "decembre": 12,
}
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_DATE_RE = re.compile(r"\b(\d{1,4})[\/\-. ](\d{1,2})[\/\-. ](\d{1,4})\b")
_TEXTUAL_DATE_RE = re.compile(r"\b(\d{1,2})\s+([a-z]+)\s+(\d{4})\b")
_OCR_CONFUSABLE_TRANSLATIONS = str.maketrans(
    {
        "O": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "|": "1",
        "S": "5",
        "$": "5",
        "B": "8",
        "Z": "2",
        "G": "6",
    }
)


@dataclass(frozen=True)
class _Window:
    text: str
    normalized: str
    token_set: set[str]
    start: int
    end: int


def _strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_text(value: Any) -> str:
    text = _strip_accents(str(value or "").lower())
    text = re.sub(r"[^\w\s./-]", " ", text)
    text = re.sub(r"[-_/]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(\d)\s+[,.]\s*(\d)", r"\1.\2", text)
    text = re.sub(r"(?<=\d)[, ](?=\d{3}\b)", "", text)
    return text


def _normalize_identity_token(value: Any) -> str:
    text = str(value or "").lower()
    return re.sub(r"[^a-z0-9]", "", text)


def _normalize_identity_for_fuzzy(value: Any) -> str:
    token = re.sub(r"[^A-Z0-9|$]", "", str(value or "").upper())
    return token.translate(_OCR_CONFUSABLE_TRANSLATIONS)


def _tokenize(value: str) -> List[str]:
    return _TOKEN_RE.findall(value)


def _levenshtein_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    distance = prev[n]
    return 1.0 - (distance / max(m, n))


def _extract_number(value: Any) -> float | None:
    text = normalize_text(value)
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _parse_date(value: Any) -> datetime | None:
    text = normalize_text(value)
    if not text:
        return None
    m = _DATE_RE.search(text)
    if m:
        a, b, c = [int(x) for x in m.groups()]
        candidates: list[Tuple[int, int, int]] = []
        if a > 31:
            candidates.append((a, b, c))
        elif c > 31:
            candidates.append((c, b, a))
            candidates.append((c, a, b))
        for y, mo, d in candidates:
            try:
                return datetime(y, mo, d)
            except ValueError:
                continue
    tm = _TEXTUAL_DATE_RE.search(text)
    if tm:
        day = int(tm.group(1))
        month = _MONTH_MAP.get(tm.group(2))
        year = int(tm.group(3))
        if month:
            try:
                return datetime(year, month, day)
            except ValueError:
                return None
    return None


def _extract_windows(raw_text: str, max_ngram: int = 5) -> List[_Window]:
    matches = list(_TOKEN_RE.finditer(raw_text))
    if not matches:
        return []
    windows: List[_Window] = []
    upper = min(max_ngram, 5)
    for idx in range(len(matches)):
        for size in range(1, upper + 1):
            if idx + size > len(matches):
                break
            start = matches[idx].start()
            end = matches[idx + size - 1].end()
            snippet = raw_text[start:end]
            normalized = normalize_text(snippet)
            windows.append(
                _Window(
                    text=snippet,
                    normalized=normalized,
                    token_set=set(_tokenize(normalized)),
                    start=start,
                    end=end,
                )
            )
    return windows


def _status_from_score(score: int) -> str:
    if score >= 85:
        return "VERIFIED"
    if score >= 60:
        return "WEAK_MATCH"
    return "NOT_FOUND"


def _score_window(field: str, normalized_value: str, value_tokens: List[str], window: _Window) -> int:
    if not normalized_value:
        return 0
    if normalized_value == window.normalized:
        return 100
    if normalized_value and normalized_value in window.normalized:
        return 97
    levenshtein = _levenshtein_similarity(normalized_value, window.normalized) * 100.0
    overlap = (len(set(value_tokens).intersection(window.token_set)) / max(1, len(set(value_tokens)))) * 100.0
    score = (0.6 * levenshtein) + (0.4 * overlap)
    if field == "provider":
        score = (0.7 * levenshtein) + (0.3 * overlap)
    return int(max(0.0, min(100.0, score)))


def _score_cin(value: Any, windows: Iterable[_Window]) -> Dict[str, Any]:
    cin = normalize_text(value).replace(" ", "")
    if not re.fullmatch(r"[a-z]{1,4}\d{3,12}", cin):
        return {"best_match": "", "confidence": 0, "found": False, "location": None}
    best: Dict[str, Any] = {"best_match": "", "confidence": 0, "found": False, "location": None}
    for window in windows:
        candidate = window.normalized.replace(" ", "")
        if not re.fullmatch(r"[a-z0-9]{4,16}", candidate):
            continue
        score = int(_levenshtein_similarity(cin, candidate) * 100)
        if cin == candidate:
            score = 100
        if score > best["confidence"]:
            best = {
                "best_match": window.text,
                "confidence": score,
                "found": score >= 90,
                "location": {"start": window.start, "end": window.end},
            }
    return best


def _score_identity_fuzzy(
    *,
    value: Any,
    raw_text: str,
    candidate_pattern: str,
    verified_threshold: int = 90,
) -> Dict[str, Any]:
    expected_raw = str(value or "").strip()
    expected_norm = _normalize_identity_for_fuzzy(expected_raw)
    if not expected_norm:
        return {"best_match": "", "confidence": 0, "found": False, "location": None}

    best: Dict[str, Any] = {"best_match": "", "confidence": 0, "found": False, "location": None}
    for match in re.finditer(candidate_pattern, str(raw_text or "").upper()):
        candidate = match.group(0)
        candidate_norm = _normalize_identity_for_fuzzy(candidate)
        if not candidate_norm:
            continue
        score = int(round(_levenshtein_similarity(expected_norm, candidate_norm) * 100))
        if expected_norm == candidate_norm:
            score = 100
        if score > int(best["confidence"]):
            best = {
                "best_match": candidate,
                "confidence": score,
                "found": score >= verified_threshold,
                "location": {"start": match.start(), "end": match.end()},
            }
    return best


def _score_amount(value: Any, windows: Iterable[_Window], normalized_text: str) -> Dict[str, Any]:
    expected = _extract_number(value)
    if expected is None or expected <= 0:
        return {"best_match": "", "confidence": 0, "found": False, "location": None}
    best = {"best_match": "", "confidence": 0, "found": False, "location": None}
    currency_nearby = any(token in normalized_text for token in _CURRENCY_TOKENS)
    for window in windows:
        observed = _extract_number(window.text)
        if observed is None:
            continue
        delta_ratio = abs(observed - expected) / max(expected, 1.0)
        if delta_ratio <= 0.01:
            score = 100
        elif delta_ratio <= 0.02:
            score = 90
        elif delta_ratio <= 0.05:
            score = 65
        else:
            score = max(0, 55 - int(delta_ratio * 100))
        if currency_nearby and any(t in window.normalized for t in _CURRENCY_TOKENS):
            score = min(100, score + 5)
        if score > best["confidence"]:
            best = {
                "best_match": window.text,
                "confidence": score,
                "found": score >= 85,
                "location": {"start": window.start, "end": window.end},
            }
    return best


def _score_date(value: Any, windows: Iterable[_Window], normalized_text: str) -> Dict[str, Any]:
    expected = _parse_date(value)
    if expected is None:
        return {"best_match": "", "confidence": 0, "found": False, "location": None}
    best = {"best_match": "", "confidence": 0, "found": False, "location": None}
    for window in windows:
        observed = _parse_date(window.text)
        if observed is None:
            continue
        if observed.date() == expected.date():
            score = 100
        else:
            score = 0
        if score > best["confidence"]:
            best = {
                "best_match": window.text,
                "confidence": score,
                "found": score >= 85,
                "location": {"start": window.start, "end": window.end},
            }
    if not best["best_match"]:
        # fallback on whole normalized OCR text for format differences
        observed_whole = _parse_date(normalized_text)
        if observed_whole and observed_whole.date() == expected.date():
            idx = normalized_text.find(normalize_text(value))
            location = {"start": idx, "end": idx + len(normalize_text(value))} if idx >= 0 else None
            best = {"best_match": normalize_text(value), "confidence": 95, "found": True, "location": location}
    return best


def _score_provider(value: Any, windows: Iterable[_Window]) -> Dict[str, Any]:
    target = normalize_text(value)
    tokens = _tokenize(target)
    if not target:
        return {"best_match": "", "confidence": 0, "found": False, "location": None}
    best = {"best_match": "", "confidence": 0, "found": False, "location": None}
    for window in windows:
        score = _score_window("provider", target, tokens, window)
        if score > best["confidence"]:
            best = {
                "best_match": window.text,
                "confidence": score,
                "found": score >= 80,
                "location": {"start": window.start, "end": window.end},
            }
    return best


def _score_generic(value: Any, windows: Iterable[_Window]) -> Dict[str, Any]:
    target = normalize_text(value)
    tokens = _tokenize(target)
    if not target:
        return {"best_match": "", "confidence": 0, "found": False, "location": None}
    best = {"best_match": "", "confidence": 0, "found": False, "location": None}
    for window in windows:
        score = _score_window("generic", target, tokens, window)
        if score > best["confidence"]:
            best = {
                "best_match": window.text,
                "confidence": score,
                "found": score >= 85,
                "location": {"start": window.start, "end": window.end},
            }
    return best


def verify_structured_fields(structured_fields: Dict[str, Any], extracted_text: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw_text = str(extracted_text or "")
    normalized_text = normalize_text(raw_text)
    windows = _extract_windows(raw_text, max_ngram=5)

    critical_fields = {"cin", "ipp", "amount", "date", "provider"}
    verification_rows: List[Dict[str, Any]] = []
    verified_fields: Dict[str, Any] = {}
    unverified_critical = 0
    weak_critical = 0
    duplicate_flags: List[str] = []
    identity_failures: List[str] = []

    cin_value = str(structured_fields.get("cin") or "").strip()
    ipp_value = str(structured_fields.get("ipp") or "").strip()
    cin_present = bool(cin_value)
    ipp_present = bool(ipp_value)
    normalized_ocr_identity = _normalize_identity_token(raw_text)

    cin_row: Dict[str, Any] | None = None
    ipp_row: Dict[str, Any] | None = None

    for field, value in structured_fields.items():
        field_key = str(field).strip().lower()
        input_present = bool(str(value or "").strip())
        if field_key == "cin":
            cin_candidate = str(value or "").strip().upper()
            cin_token = _normalize_identity_token(cin_candidate)
            exact_found = bool(cin_token and cin_token in normalized_ocr_identity)
            if exact_found:
                best = {
                    "best_match": cin_candidate,
                    "confidence": 100,
                    "found": True,
                    "location": None,
                }
            else:
                best = _score_identity_fuzzy(
                    value=cin_candidate,
                    raw_text=raw_text,
                    candidate_pattern=r"\b[A-Z0-9]{5,12}\b",
                    verified_threshold=88,
                )
        elif field_key == "amount":
            best = _score_amount(value, windows, normalized_text)
        elif field_key == "date":
            best = _score_date(value, windows, normalized_text)
        elif field_key == "provider":
            best = _score_provider(value, windows)
        elif field_key == "ipp":
            ipp_candidate = str(value or "").strip()
            ipp_token = _normalize_identity_token(ipp_candidate)
            exact_found = bool(ipp_token and ipp_token in normalized_ocr_identity)
            if exact_found:
                best = {
                    "best_match": ipp_candidate,
                    "confidence": 100,
                    "found": True,
                    "location": None,
                }
            else:
                best = _score_identity_fuzzy(
                    value=ipp_candidate,
                    raw_text=raw_text,
                    candidate_pattern=r"\b[0-9OILSBZG]{6,12}\b",
                    verified_threshold=85,
                )
        else:
            best = _score_generic(value, windows)

        confidence = int(max(0, min(100, best.get("confidence", 0))))
        status = _status_from_score(confidence) if input_present else "NOT_PROVIDED"
        verified = input_present and status == "VERIFIED"
        if verified:
            verified_fields[field] = value
        if field_key in critical_fields and input_present and not verified:
            unverified_critical += 1
            if status == "WEAK_MATCH":
                weak_critical += 1

        location = best.get("location")
        snippet = ""
        if isinstance(location, dict):
            start = int(location.get("start", 0))
            end = int(location.get("end", start))
            pad_start = max(0, start - 20)
            pad_end = min(len(raw_text), end + 20)
            snippet = raw_text[pad_start:pad_end].strip()

        if len(snippet) > 0 and normalized_text.count(normalize_text(best.get("best_match", ""))) > 1:
            duplicate_flags.append(field_key)

        verification_rows.append(
            {
                "field": field,
                "value": value,
                "best_match": best.get("best_match", ""),
                "confidence": confidence,
                "match_confidence": confidence,
                "status": status,
                "found": status in {"VERIFIED", "WEAK_MATCH"},
                "found_in_text": status in {"VERIFIED", "WEAK_MATCH"},
                "verified": verified,
                "input_present": input_present,
                "critical": field_key in critical_fields,
                "location": location,
                "snippet": snippet,
            }
        )
        if field_key == "cin":
            cin_row = verification_rows[-1]
        if field_key == "ipp":
            ipp_row = verification_rows[-1]

    if not cin_present and not ipp_present:
        identity_failures.append("NO_IDENTITY")
    if cin_present and cin_row and str(cin_row.get("status", "")).upper() == "NOT_FOUND":
        identity_failures.append("CIN_NOT_FOUND")
    if ipp_present and ipp_row and str(ipp_row.get("status", "")).upper() == "NOT_FOUND":
        identity_failures.append("IPP_NOT_FOUND")

    if not cin_present and not ipp_present:
        identity_status = "MISSING"
    elif not identity_failures:
        identity_status = "VERIFIED"
    else:
        identity_status = "NOT_FOUND"

    cin_found = bool(cin_row and cin_row.get("verified"))
    ipp_found = bool(ipp_row and ipp_row.get("verified"))
    amount_row = next((row for row in verification_rows if str(row.get("field", "")).strip().lower() == "amount"), None)
    amount_found = bool(amount_row and amount_row.get("verified"))
    identity_present = bool(cin_present or ipp_present)
    identity_found = bool(cin_found or ipp_found)
    critical_stop_reasons: List[str] = []
    if not identity_present:
        critical_stop_reasons.append("MISSING_IDENTITY")
    elif not identity_found:
        critical_stop_reasons.append("CIN_OR_IPP_NOT_FOUND")
    if not amount_found:
        critical_stop_reasons.append("AMOUNT_NOT_FOUND")

    should_degrade = bool(critical_stop_reasons)
    summary = {
        "critical_fields": sorted(critical_fields),
        "unverified_critical_fields": unverified_critical,
        "weak_match_critical_fields": weak_critical,
        "should_stop_pipeline": False,
        "should_degrade_to_human_review": should_degrade,
        "has_unverified_fields": any(not row["verified"] for row in verification_rows),
        "duplicate_field_matches": sorted(set(duplicate_flags)),
        "identity_failures": sorted(set(identity_failures)),
        "critical_stop_reasons": critical_stop_reasons,
        "cin_found": cin_found,
        "ipp_found": ipp_found,
        "amount_found": amount_found,
    }
    return verification_rows, {
        "verified_fields": verified_fields,
        "summary": summary,
        "identity": {
            "cin": cin_value,
            "ipp": ipp_value,
            "cin_found": cin_found,
            "ipp_found": ipp_found,
            "status": identity_status,
        },
    }
