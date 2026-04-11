"""
Extract plain text from uploaded bytes: PDF (pypdf), UTF-8 text, images (Tesseract OCR).

OCR requires Tesseract installed on the host and on PATH; if unavailable, extraction
returns empty text with method ``ocr_unavailable`` / ``ocr_failed``.
"""
from __future__ import annotations

import io
import os
import re
from typing import Any, Dict, List

from fastapi import UploadFile

# Optional heavy imports — failures fall back gracefully
_MAX_TEXT_PER_FILE = 500_000  # cap to protect memory / agents


def _normalize_ext(name: str) -> str:
    base = name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if "." not in base:
        return ""
    return base.rsplit(".", 1)[-1].lower()


def _truncate(s: str) -> str:
    s = s.strip()
    if len(s) > _MAX_TEXT_PER_FILE:
        return s[:_MAX_TEXT_PER_FILE] + "\n...[truncated]"
    return s


def _configure_tesseract_if_needed(pytesseract_module: Any) -> None:
    """
    Best-effort Tesseract executable discovery for hosts where PATH is not set.
    """
    configured = str(getattr(pytesseract_module.pytesseract, "tesseract_cmd", "") or "").strip()
    if configured and os.path.exists(configured):
        return

    env_cmd = os.getenv("TESSERACT_CMD", "").strip()
    if env_cmd and os.path.exists(env_cmd):
        pytesseract_module.pytesseract.tesseract_cmd = env_cmd
        return

    candidates: List[str] = []
    if os.name == "nt":
        candidates.extend(
            [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            ]
        )
    else:
        candidates.extend(["/usr/bin/tesseract", "/usr/local/bin/tesseract"])

    for c in candidates:
        if os.path.exists(c):
            pytesseract_module.pytesseract.tesseract_cmd = c
            return


def _ocr_image_bytes(data: bytes) -> Dict[str, Any]:
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return {
            "extracted_text": "",
            "extraction_method": "ocr_unavailable",
            "error": "Pillow/pytesseract not installed",
        }
    try:
        _configure_tesseract_if_needed(pytesseract)
        img = Image.open(io.BytesIO(data))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        text = pytesseract.image_to_string(img)
        return {
            "extracted_text": _truncate(text),
            "extraction_method": "tesseract_ocr",
            "error": None,
        }
    except Exception as e:
        return {
            "extracted_text": "",
            "extraction_method": "ocr_failed",
            "error": str(e),
        }


def _ocr_pdf_bytes(data: bytes) -> Dict[str, Any]:
    """
    OCR fallback for scanned PDFs by rasterizing pages then using Tesseract.
    """
    try:
        import pytesseract
        import pypdfium2 as pdfium
    except ImportError:
        return {
            "extracted_text": "",
            "extraction_method": "pdf_ocr_unavailable",
            "error": "pypdfium2/pytesseract not installed",
        }
    try:
        _configure_tesseract_if_needed(pytesseract)
        pdf = pdfium.PdfDocument(io.BytesIO(data))
        chunks: List[str] = []
        for i in range(len(pdf)):
            page = pdf[i]
            bitmap = page.render(scale=2.0).to_pil()
            text = pytesseract.image_to_string(bitmap) or ""
            chunks.append(text)
        return {
            "extracted_text": _truncate("\n".join(chunks)),
            "extraction_method": "pdf_ocr",
            "error": None,
        }
    except Exception as e:
        return {
            "extracted_text": "",
            "extraction_method": "pdf_ocr_failed",
            "error": str(e),
        }


def extract_text_from_bytes(data: bytes, filename: str) -> Dict[str, Any]:
    """
    Return ``{ "extracted_text", "extraction_method", "error" }``.
    """
    ext = _normalize_ext(filename)
    err: str | None = None

    if not data:
        return {
            "extracted_text": "",
            "extraction_method": "empty",
            "error": "empty file",
        }

    if ext in {"txt", "csv", "md", "json", "xml", "html", "htm"}:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("utf-8", errors="replace")
        return {
            "extracted_text": _truncate(text),
            "extraction_method": "utf8_text",
            "error": None,
        }

    if ext == "pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            return {
                "extracted_text": "",
                "extraction_method": "unsupported",
                "error": "pypdf not installed",
            }
        try:
            reader = PdfReader(io.BytesIO(data))
            chunks: List[str] = []
            for page in reader.pages:
                try:
                    t = page.extract_text() or ""
                except Exception as e:  # pragma: no cover - defensive
                    err = str(e)
                    t = ""
                chunks.append(t)
            text = _truncate("\n".join(chunks))
            if text.strip():
                return {
                    "extracted_text": text,
                    "extraction_method": "pypdf",
                    "error": err,
                }
            ocr_res = _ocr_pdf_bytes(data)
            # If OCR unavailable/failed, preserve pypdf status for observability.
            if (ocr_res.get("extracted_text") or "").strip():
                return ocr_res
            return {
                "extracted_text": text,
                "extraction_method": "pypdf",
                "error": err,
            }
        except Exception as e:
            return {
                "extracted_text": "",
                "extraction_method": "pypdf_failed",
                "error": str(e),
            }

    if ext in {"png", "jpg", "jpeg", "webp", "tif", "tiff", "bmp", "gif"}:
        return _ocr_image_bytes(data)

    return {
        "extracted_text": "",
        "extraction_method": "unsupported",
        "error": f"unsupported extension: .{ext or '?'}",
    }


def _one_extraction(file_name: str, data: bytes) -> Dict[str, Any]:
    raw = extract_text_from_bytes(data, file_name)
    text = raw.get("extracted_text") or ""
    return {
        "file_name": file_name,
        "extracted_text": text,
        "extraction_method": raw.get("extraction_method", "unknown"),
        "char_count": len(text),
        "error": raw.get("error"),
    }


async def build_extractions_from_upload_files(files: List[UploadFile]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for uf in files:
        name = uf.filename or "unnamed"
        data = await uf.read()
        out.append(_one_extraction(name, data))
    return out


def build_extractions_from_base64_parts(parts: List[Any]) -> List[Dict[str, Any]]:
    """``parts`` items must have ``name`` and ``content_base64`` (see models)."""
    import base64

    out: List[Dict[str, Any]] = []
    for p in parts:
        if isinstance(p, dict):
            name = str(p.get("name") or "unnamed")
            b64 = str(p.get("content_base64") or "")
        else:
            name = getattr(p, "name", None) or "unnamed"
            b64 = getattr(p, "content_base64", "") or ""
        # Allow data-URL style prefixes
        if "base64," in b64:
            b64 = b64.split("base64,", 1)[-1]
        b64 = re.sub(r"\s+", "", b64)
        try:
            data = base64.b64decode(b64, validate=True)
        except Exception as e:
            out.append(
                {
                    "file_name": str(name),
                    "extracted_text": "",
                    "extraction_method": "base64_decode_failed",
                    "char_count": 0,
                    "error": str(e),
                }
            )
            continue
        out.append(_one_extraction(str(name), data))
    return out
