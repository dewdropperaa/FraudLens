import importlib
import sys
import types

from claimguard.services import document_extraction as de


def test_configure_tesseract_uses_env_path(monkeypatch):
    fake_mod = types.SimpleNamespace(
        pytesseract=types.SimpleNamespace(tesseract_cmd="")
    )
    monkeypatch.setenv("TESSERACT_CMD", "/tmp/tesseract")
    monkeypatch.setattr(de.os.path, "exists", lambda p: p == "/tmp/tesseract")

    de._configure_tesseract_if_needed(fake_mod)
    assert fake_mod.pytesseract.tesseract_cmd == "/tmp/tesseract"


def test_pdf_falls_back_to_ocr_when_no_native_text(monkeypatch):
    class _FakePage:
        def extract_text(self):
            return ""

    class _FakePdfReader:
        def __init__(self, _):
            self.pages = [_FakePage()]

    fake_pypdf = types.SimpleNamespace(PdfReader=_FakePdfReader)
    monkeypatch.setitem(sys.modules, "pypdf", fake_pypdf)
    monkeypatch.setattr(
        de,
        "_ocr_pdf_bytes",
        lambda _data: {
            "extracted_text": "medical report invoice prescription",
            "extraction_method": "pdf_ocr",
            "error": None,
        },
    )

    out = de.extract_text_from_bytes(b"%PDF-scan", "scan.pdf")
    assert out["extraction_method"] == "pdf_ocr"
    assert "medical report" in out["extracted_text"]


def test_pdf_keeps_pypdf_when_text_exists(monkeypatch):
    class _FakePage:
        def extract_text(self):
            return "invoice"

    class _FakePdfReader:
        def __init__(self, _):
            self.pages = [_FakePage()]

    fake_pypdf = types.SimpleNamespace(PdfReader=_FakePdfReader)
    monkeypatch.setitem(sys.modules, "pypdf", fake_pypdf)
    called = {"ocr": False}

    def _fake_ocr(_data):
        called["ocr"] = True
        return {"extracted_text": "ignored", "extraction_method": "pdf_ocr", "error": None}

    monkeypatch.setattr(de, "_ocr_pdf_bytes", _fake_ocr)

    out = de.extract_text_from_bytes(b"%PDF-text", "text.pdf")
    assert out["extraction_method"] == "pypdf"
    assert out["extracted_text"].strip() == "invoice"
    assert called["ocr"] is False

