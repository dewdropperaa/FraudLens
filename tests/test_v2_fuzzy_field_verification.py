from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from claimguard.v2.field_verification import verify_structured_fields


def _row(rows: list[dict], name: str) -> dict:
    return next(item for item in rows if item["field"] == name)


def test_amount_normalization_and_currency_match() -> None:
    fields = {"amount": "1200 MAD"}
    text = "facture: montant total 1,200 mad"
    rows, meta = verify_structured_fields(fields, text)
    amount = _row(rows, "amount")
    assert amount["status"] == "VERIFIED"
    assert amount["verified"] is True
    assert meta["summary"]["unverified_critical_fields"] == 0


def test_cin_near_exact_threshold_blocks_ocr_error() -> None:
    fields = {"cin": "AB123456"}
    text = "cin AB12345G"
    rows, meta = verify_structured_fields(fields, text)
    cin = _row(rows, "cin")
    assert cin["verified"] is False
    assert cin["status"] != "VERIFIED"
    assert meta["summary"]["unverified_critical_fields"] == 1


def test_provider_fuzzy_tolerates_minor_ocr_noise() -> None:
    fields = {"provider": "clinic xyz"}
    text = "stamp from clinlc xyz and payment details"
    rows, _ = verify_structured_fields(fields, text)
    provider = _row(rows, "provider")
    assert provider["status"] in {"VERIFIED", "WEAK_MATCH"}
    assert provider["found"] is True


def test_date_format_normalization_matches_same_calendar_date() -> None:
    fields = {"date": "2026-01-10"}
    text = "date de service: 10/01/2026"
    rows, _ = verify_structured_fields(fields, text)
    date_row = _row(rows, "date")
    assert date_row["status"] == "VERIFIED"
    assert date_row["verified"] is True


def test_random_text_returns_not_found_and_low_confidence() -> None:
    fields = {"cin": "AB123456", "amount": "1200 MAD", "date": "2026-01-10", "provider": "clinic xyz"}
    text = "random unrelated promo text for electronics and groceries"
    rows, meta = verify_structured_fields(fields, text)
    assert all(row["status"] == "NOT_FOUND" for row in rows)
    assert meta["summary"]["unverified_critical_fields"] >= 2
