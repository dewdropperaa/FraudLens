from __future__ import annotations

import json
import logging
import os
import threading
from queue import Empty, Queue
from typing import Any, Dict

from claimguard.llm_factory import get_llm
from claimguard.llm_tracking import safe_tracked_llm_call
from claimguard.v2.extraction.rule_extractor import RuleExtractor
from claimguard.v2.extraction.validator import validate_extraction

LOGGER = logging.getLogger("claimguard.v2.extraction")


class HybridExtractor:
    def __init__(self, *, llm_enabled: bool = True) -> None:
        self._llm_enabled = llm_enabled

    @staticmethod
    def _empty_fields() -> Dict[str, Any]:
        return {
            "name": None,
            "cin": None,
            "dob": None,
            "insurance": None,
            "patient_id": None,
        }

    def _llm_extract(self, text: str) -> Dict[str, Any]:
        if not self._llm_enabled:
            return {"status": "ERROR", "reason": "LLM fallback disabled", "stage": "llm"}
        prompt = (
            "You are an extraction engine. Return STRICT JSON only.\n"
            "Extract these fields from the provided medical text: name, cin, dob, insurance, patient_id.\n"
            "If a field is not present, set it to null.\n"
            "Output JSON schema: "
            '{"name": null|string, "cin": null|string, "dob": null|string, "insurance": null|string, "patient_id": null|string}\n'
            "No markdown. No explanation.\n\n"
            f"TEXT:\n{text}"
        )
        llm = get_llm("deepseek-r1", tracked=False)
        timeout_s = float(os.getenv("CLAIMGUARD_HYBRID_LLM_TIMEOUT_S", "12"))
        result_queue: Queue[Dict[str, Any]] = Queue(maxsize=1)

        def _worker() -> None:
            try:
                result = safe_tracked_llm_call("HybridExtractor", prompt, llm.invoke)
                result_queue.put(result)
            except Exception as exc:
                result_queue.put({"error": "LLM_CALL_FAILED", "reason": str(exc)})

        worker = threading.Thread(target=_worker, daemon=True)
        worker.start()
        worker.join(timeout=timeout_s)
        if worker.is_alive():
            LOGGER.warning("[LLM TIMEOUT] HybridExtractor timed out after %ss", timeout_s)
            return {
                "status": "ERROR",
                "reason": f"Hybrid extraction LLM timeout after {int(timeout_s)}s",
                "stage": "llm",
            }
        try:
            result = result_queue.get_nowait()
        except Empty:
            return {"status": "ERROR", "reason": "LLM returned no result", "stage": "llm"}
        if result.get("error"):
            return {
                "status": "ERROR",
                "reason": str(result.get("reason") or result.get("error")),
                "stage": "llm",
            }
        parsed = result.get("parsed")
        if not isinstance(parsed, dict):
            return {"status": "ERROR", "reason": "LLM output is not JSON object", "stage": "llm"}
        normalized = self._empty_fields()
        for key in normalized:
            value = parsed.get(key)
            normalized[key] = None if value is None else str(value).strip() or None
        return {"status": "OK", "engine": "llm", "fields": normalized}

    def extract(self, text: str) -> Dict[str, Any]:
        source = str(text or "").strip()
        LOGGER.info("[EXTRACTION START]")
        if not source:
            return {"status": "ERROR", "reason": "Input text is empty", "stage": "rule"}

        rule_result = RuleExtractor.extract(source)
        LOGGER.info("[RULE ENGINE RESULT] %s", json.dumps(rule_result, ensure_ascii=False))

        if float(rule_result.get("confidence", 0.0)) >= 80.0 and bool(rule_result.get("should_return_immediately")):
            merged = {
                "status": "OK",
                "engine": "rule",
                "fields": rule_result["fields"],
                "confidence": float(rule_result["confidence"]),
            }
        else:
            LOGGER.info("[LLM FALLBACK USED]")
            llm_result = self._llm_extract(source)
            if llm_result.get("status") != "OK":
                return llm_result
            merged_fields = dict(llm_result.get("fields", {}))
            for key, value in rule_result["fields"].items():
                if value is not None:
                    merged_fields[key] = value
            detected = sum(1 for value in merged_fields.values() if value is not None)
            merged = {
                "status": "OK",
                "engine": "hybrid",
                "fields": merged_fields,
                "confidence": round((detected / 5.0) * 100, 2),
            }

        validation = validate_extraction(merged)
        LOGGER.info("[VALIDATION RESULT] %s", json.dumps(validation, ensure_ascii=False))
        if validation.get("status") != "OK":
            return validation

        if not any(value is not None for value in merged.get("fields", {}).values()):
            return {"status": "ERROR", "reason": "No extracted fields", "stage": "validation"}

        LOGGER.info("[FINAL MERGED OUTPUT] %s", json.dumps(merged, ensure_ascii=False))
        return merged

    def self_test(self) -> bool:
        sample = (
            "Nom complet: Test Patient\n"
            "CIN: AB123456\n"
            "Date de naissance: 01/01/1990\n"
            "Mutuelle: CNOPS\n"
            "N° IPP: 774411"
        )
        result = self.extract(sample)
        return result.get("status") == "OK" and result.get("fields", {}).get("cin") == "AB123456"
