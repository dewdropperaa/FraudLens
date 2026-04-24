from __future__ import annotations

from claimguard.v2.extraction.hybrid_extractor import HybridExtractor


def test_normal_medical_document_full_extraction_success() -> None:
    extractor = HybridExtractor(llm_enabled=False)
    text = (
        "Nom complet: Jean Dupont\n"
        "CIN: AB123456\n"
        "Date de naissance: 05/07/1988\n"
        "Mutuelle: CNSS\n"
        "N° IPP: 998877"
    )
    result = extractor.extract(text)
    assert result["status"] == "OK"
    assert result["fields"]["name"] == "Jean Dupont"
    assert result["fields"]["cin"] == "AB123456"
    assert result["fields"]["dob"] == "05/07/1988"
    assert result["fields"]["insurance"] == "CNSS"
    assert result["fields"]["patient_id"] == "998877"


def test_missing_cin_returns_null_without_crash() -> None:
    extractor = HybridExtractor(llm_enabled=False)
    text = "Nom complet: Amal Karim\nDate de naissance: 12/03/1992\nMutuelle: CNOPS\nN° IPP: 123456"
    result = extractor.extract(text)
    assert result["status"] == "OK"
    assert result["fields"]["cin"] is None
    assert result["fields"]["name"] == "Amal Karim"


def test_corrupted_ocr_forces_llm_fallback() -> None:
    extractor = HybridExtractor(llm_enabled=True)

    def fake_llm_extract(_: str) -> dict:
        return {
            "status": "OK",
            "engine": "llm",
            "fields": {
                "name": "Recovered Name",
                "cin": "ZXCVB123",
                "dob": None,
                "insurance": None,
                "patient_id": None,
            },
        }

    extractor._llm_extract = fake_llm_extract  # type: ignore[method-assign]
    result = extractor.extract("### ??? OCR noise ###")
    assert result["status"] == "OK"
    assert result["engine"] == "hybrid"
    assert result["fields"]["cin"] == "ZXCVB123"


def test_empty_input_fails_explicitly() -> None:
    extractor = HybridExtractor(llm_enabled=False)
    result = extractor.extract("")
    assert result == {"status": "ERROR", "reason": "Input text is empty", "stage": "rule"}


def test_mixed_language_extracts_name_and_cin() -> None:
    extractor = HybridExtractor(llm_enabled=False)
    text = "Patient information\nNom complet: Sara Ben Ali\nIdentifiant CIN: AZ987654\nDate de naissance: 2001-10-22"
    result = extractor.extract(text)
    assert result["status"] == "OK"
    assert result["fields"]["name"] == "Sara Ben Ali"
    assert result["fields"]["cin"] == "AZ987654"
