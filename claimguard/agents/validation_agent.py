"""ClaimValidationAgent — Strict claim validation layer that runs BEFORE all other agents.

This agent ensures that ONLY valid medical claims are processed by the system.
Any non-claim or incomplete submission is rejected BEFORE fraud analysis.

VALIDATION CRITERIA:
- Document must be a REAL medical claim/invoice
- Required fields: patient identity, medical service, provider, date, cost
- Missing >= 2 required elements → INVALID
- Document type != medical claim → INVALID

HARD GATE (NON-BYPASSABLE):
- If validation_status == INVALID → STOP ENTIRE PIPELINE
- Do NOT run other agents
- Do NOT compute fraud score
- Return immediate REJECTED decision
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from claimguard.agents.base_agent import BaseAgent
from claimguard.agents.security_utils import (
    detect_prompt_injection,
    sanitize_input,
)

LOGGER = logging.getLogger("claimguard.agents.validation")
DEBUG_EXPLANATION_MODE = True

# ── Document type classification ───────────────────────────────────────
DocumentType = Literal[
    "medical_invoice",
    "medical_prescription",
    "lab_report",
    "medical_certificate",
    "insurance_attestation",
    "pharmacy_invoice",
    "hospital_bill",
    "irrelevant_document",
    "unknown"
]

# Keywords for document classification
_MEDICAL_INVOICE_KEYWORDS: tuple[str, ...] = (
    "facture", "invoice", "note d'honoraires", "note d honoraires",
    "reçu", "recu", "bordereau", "décompte", "decompte",
    "total ttc", "montant ttc", "tva", "montant à payer",
    "montant a payer", "net à payer", "net a payer",
)

_PRESCRIPTION_KEYWORDS: tuple[str, ...] = (
    "ordonnance", "prescription", "medicament", "médicament",
    "posologie", "traitement prescrit", "dose", "comprimé", "comprime",
)

_LAB_REPORT_KEYWORDS: tuple[str, ...] = (
    "résultat d'analyse", "resultat d analyse", "analyses médicales",
    "analyses medicales", "laboratoire", "bilan sanguin", "hémogramme",
    "hemogramme", "glycémie", "glycemie",
)

_INSURANCE_ATTESTATION_KEYWORDS: tuple[str, ...] = (
    "attestation", "carte d'assuré", "carte d assure",
    "police d'assurance", "police d assurance",
    "numéro d'assuré", "numero d assure", "cnss", "cnops",
    "mutuelle", "couverture médicale", "couverture medicale",
)

_HOSPITAL_BILL_KEYWORDS: tuple[str, ...] = (
    "hospitalisation", "séjour hospitalier", "sejour hospitalier",
    "facture hospitalière", "facture hospitaliere",
    "acte médical", "acte medical", "frais d'hospitalisation",
    "frais d hospitalisation",
)

# Required claim elements
_PATIENT_IDENTITY_KEYWORDS: tuple[str, ...] = (
    "nom", "prénom", "prenom", "patient", "cin", "carte nationale",
    "carte d'identité", "carte d identite", "assuré", "assure",
    "bénéficiaire", "beneficiaire", "numéro d'assuré", "numero d assure",
)

_MEDICAL_SERVICE_KEYWORDS: tuple[str, ...] = (
    "consultation", "examen", "acte", "intervention", "opération", "operation",
    "diagnostic", "traitement", "soin", "prestation", "service médical",
    "service medical", "chirurgie", "radiologie", "échographie", "echographie",
)

_PROVIDER_KEYWORDS: tuple[str, ...] = (
    "médecin", "medecin", "docteur", "dr.", "dr ", "hôpital", "hopital",
    "clinique", "centre médical", "centre medical", "cabinet médical",
    "cabinet medical", "établissement", "etablissement",
)

_DATE_SERVICE_KEYWORDS: tuple[str, ...] = (
    "date", "le ", "effectué le", "effectue le", "consultation du",
    "intervention du", "séjour du", "sejour du",
)

_COST_KEYWORDS: tuple[str, ...] = (
    "montant", "prix", "coût", "cout", "tarif", "total", "ttc",
    "à payer", "a payer", "net à payer", "net a payer",
    "honoraires", "frais",
)

# Date patterns for extraction
_DATE_PATTERNS = [
    re.compile(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})\b"),  # DD/MM/YYYY
    re.compile(r"\b(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})\b"),  # YYYY-MM-DD
    re.compile(r"\b(\d{1,2})\s+(janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre)\s+(\d{4})\b", re.IGNORECASE),
]

# Amount patterns
_AMOUNT_PATTERNS = [
    re.compile(r"\b(\d{1,3}(?:\s?\d{3})*(?:[.,]\d{1,2})?)\s*(?:dh|mad|dirham|€|euro)\b", re.IGNORECASE),
    re.compile(r"(?:montant|total|prix|coût|cout|tarif)\s*:?\s*(\d{1,3}(?:\s?\d{3})*(?:[.,]\d{1,2})?)", re.IGNORECASE),
]

VALIDATION_SYSTEM_PROMPT = """You are a strict claim validation specialist.

Your ONLY job is to determine if a document is a VALID medical claim/invoice.

A VALID medical claim MUST contain:
1. Patient identity (name, CIN, or patient ID)
2. Medical service description
3. Provider/hospital name
4. Date of service
5. Cost/billing amount

You do NOT assess fraud risk — only validity.

CLASSIFICATION RULES:
- medical_invoice: Invoice for medical services with cost breakdown
- pharmacy_invoice: Pharmacy receipt for medications
- hospital_bill: Hospital billing statement
- medical_prescription: Doctor's prescription (NOT a claim itself)
- lab_report: Medical test results (NOT a claim itself)
- medical_certificate: Sick leave certificate (NOT a claim itself)
- insurance_attestation: Insurance card/policy (NOT a claim itself)
- irrelevant_document: Non-medical document
- unknown: Cannot determine type

VALIDATION RULES:
- Missing >= 2 required elements → INVALID
- Document type NOT in (medical_invoice, pharmacy_invoice, hospital_bill) → INVALID
- Irrelevant document → INVALID

Return JSON:
{
  "document_type": "...",
  "validation_status": "VALID" or "INVALID",
  "validation_score": 0-100,
  "missing_fields": [...],
  "found_fields": [...],
  "reason": "..."
}
"""


class ClaimValidationAgent(BaseAgent):
    """Validation agent that ensures only valid medical claims are processed."""
    
    system_prompt: str = VALIDATION_SYSTEM_PROMPT

    def __init__(self):
        super().__init__(
            name="ClaimValidationAgent",
            role="Claim Validation Specialist",
            goal="Ensure only valid medical claims are processed by the system",
        )

    @staticmethod
    def _build_document_corpus(claim_data: Dict[str, Any]) -> str:
        """Build a text corpus from all document sources."""
        parts: List[str] = []
        
        # Add document names
        documents = claim_data.get("documents", [])
        if isinstance(documents, list):
            for doc in documents:
                parts.append(sanitize_input(str(doc)).lower())
        
        # Add document extractions
        extractions = claim_data.get("document_extractions", [])
        if isinstance(extractions, list):
            for ex in extractions:
                if isinstance(ex, dict):
                    file_name = ex.get("file_name", "")
                    extracted_text = ex.get("extracted_text", "")
                    parts.append(sanitize_input(str(file_name)).lower())
                    parts.append(sanitize_input(str(extracted_text)).lower())
        
        # Add metadata
        metadata = claim_data.get("metadata", {})
        if isinstance(metadata, dict):
            for value in metadata.values():
                if isinstance(value, str):
                    parts.append(sanitize_input(value).lower())
        
        # Add policy info
        policy = claim_data.get("policy", {})
        if isinstance(policy, dict):
            for value in policy.values():
                if isinstance(value, str):
                    parts.append(sanitize_input(value).lower())
        
        return " ".join(parts)

    @staticmethod
    def _keyword_match(corpus: str, keywords: tuple[str, ...]) -> bool:
        """Check if any keyword from the group is present in corpus."""
        return any(kw in corpus for kw in keywords)

    @staticmethod
    def _extract_dates(corpus: str) -> List[str]:
        """Extract date strings from corpus."""
        dates: List[str] = []
        for pattern in _DATE_PATTERNS:
            matches = pattern.findall(corpus)
            dates.extend([match if isinstance(match, str) else match[0] for match in matches])
        return dates[:5]  # Limit to first 5 dates found

    @staticmethod
    def _extract_amounts(corpus: str) -> List[str]:
        """Extract monetary amounts from corpus."""
        amounts: List[str] = []
        for pattern in _AMOUNT_PATTERNS:
            matches = pattern.findall(corpus)
            amounts.extend(matches)
        return amounts[:5]  # Limit to first 5 amounts found

    def _classify_document_type(self, corpus: str) -> DocumentType:
        """Classify document type based on keyword presence."""
        # Medical invoice
        if self._keyword_match(corpus, _MEDICAL_INVOICE_KEYWORDS):
            if self._keyword_match(corpus, _HOSPITAL_BILL_KEYWORDS):
                return "hospital_bill"
            return "medical_invoice"
        
        # Pharmacy invoice
        if self._keyword_match(corpus, _PRESCRIPTION_KEYWORDS) and \
           self._keyword_match(corpus, _COST_KEYWORDS):
            return "pharmacy_invoice"
        
        # Hospital bill
        if self._keyword_match(corpus, _HOSPITAL_BILL_KEYWORDS):
            return "hospital_bill"
        
        # Lab report
        if self._keyword_match(corpus, _LAB_REPORT_KEYWORDS):
            return "lab_report"
        
        # Prescription only
        if self._keyword_match(corpus, _PRESCRIPTION_KEYWORDS):
            return "medical_prescription"
        
        # Insurance attestation
        if self._keyword_match(corpus, _INSURANCE_ATTESTATION_KEYWORDS):
            return "insurance_attestation"
        
        # Medical certificate
        if "certificat" in corpus and "médical" in corpus:
            return "medical_certificate"
        
        # Check if medical-related at all
        medical_terms = ["médical", "medical", "patient", "docteur", "medecin"]
        if not any(term in corpus for term in medical_terms):
            return "irrelevant_document"
        
        return "unknown"

    def _validate_required_fields(
        self, 
        claim_data: Dict[str, Any], 
        corpus: str
    ) -> Dict[str, Any]:
        """Validate presence of all required claim elements."""
        
        missing_fields: List[str] = []
        found_fields: List[str] = []
        field_details: Dict[str, Any] = {}
        
        # 1. Patient Identity
        identity = claim_data.get("identity", {})
        cin = identity.get("cin") or identity.get("CIN") or claim_data.get("patient_id")
        patient_name = (
            identity.get("name") or 
            identity.get("full_name") or 
            claim_data.get("patient_name")
        )
        
        has_patient_identity = (
            bool(cin and str(cin).strip()) or
            bool(patient_name and str(patient_name).strip()) or
            self._keyword_match(corpus, _PATIENT_IDENTITY_KEYWORDS)
        )
        
        if has_patient_identity:
            found_fields.append("patient_identity")
            field_details["patient_identity"] = {
                "cin": str(cin) if cin else None,
                "name": str(patient_name) if patient_name else None,
            }
        else:
            missing_fields.append("patient_identity")
        
        # 2. Medical Service Description
        policy = claim_data.get("policy", {})
        diagnosis = policy.get("diagnosis") or policy.get("service_description")
        
        has_medical_service = (
            bool(diagnosis and str(diagnosis).strip()) or
            self._keyword_match(corpus, _MEDICAL_SERVICE_KEYWORDS)
        )
        
        if has_medical_service:
            found_fields.append("medical_service")
            field_details["medical_service"] = str(diagnosis) if diagnosis else "detected_in_documents"
        else:
            missing_fields.append("medical_service")
        
        # 3. Provider/Hospital Name
        hospital = (
            identity.get("hospital") or 
            policy.get("hospital") or 
            policy.get("provider") or
            claim_data.get("metadata", {}).get("hospital")
        )
        doctor = (
            identity.get("doctor") or 
            policy.get("doctor") or
            claim_data.get("metadata", {}).get("doctor")
        )
        
        has_provider = (
            bool(hospital and str(hospital).strip()) or
            bool(doctor and str(doctor).strip()) or
            self._keyword_match(corpus, _PROVIDER_KEYWORDS)
        )
        
        if has_provider:
            found_fields.append("provider")
            field_details["provider"] = {
                "hospital": str(hospital) if hospital else None,
                "doctor": str(doctor) if doctor else None,
            }
        else:
            missing_fields.append("provider")
        
        # 4. Date of Service
        metadata = claim_data.get("metadata", {})
        service_date = (
            metadata.get("service_date") or 
            policy.get("service_date") or
            claim_data.get("service_date")
        )
        
        extracted_dates = self._extract_dates(corpus)
        
        has_date = (
            bool(service_date and str(service_date).strip()) or
            len(extracted_dates) > 0 or
            self._keyword_match(corpus, _DATE_SERVICE_KEYWORDS)
        )
        
        if has_date:
            found_fields.append("date_of_service")
            field_details["date_of_service"] = {
                "structured": str(service_date) if service_date else None,
                "extracted": extracted_dates,
            }
        else:
            missing_fields.append("date_of_service")
        
        # 5. Cost/Billing Amount
        amount = (
            claim_data.get("amount") or
            metadata.get("amount") or
            policy.get("amount")
        )
        
        extracted_amounts = self._extract_amounts(corpus)
        
        # Cost is considered present only if:
        # - Structured amount > 0, OR
        # - Extracted amounts found in documents
        has_valid_amount = False
        try:
            if amount is not None:
                amt_float = float(amount)
                if amt_float > 0:
                    has_valid_amount = True
        except (TypeError, ValueError):
            pass
        
        has_cost = (
            has_valid_amount or
            len(extracted_amounts) > 0
        )
        
        if has_cost:
            found_fields.append("cost")
            field_details["cost"] = {
                "structured": float(amount) if amount else None,
                "extracted": extracted_amounts,
            }
        else:
            missing_fields.append("cost")
        
        return {
            "missing_fields": missing_fields,
            "found_fields": found_fields,
            "field_details": field_details,
            "completeness_ratio": len(found_fields) / 5.0,
        }

    @staticmethod
    def _ratio(numerator: int, denominator: int) -> float:
        if denominator <= 0:
            return 0.0
        return max(0.0, min(1.0, float(numerator) / float(denominator)))

    def _compute_coverage_metrics(
        self,
        *,
        claim_data: Dict[str, Any],
        corpus: str,
        classifier_out: Dict[str, Any],
        field_validation: Dict[str, Any],
    ) -> Dict[str, Any]:
        extractions = claim_data.get("document_extractions", [])
        ocr_text = " ".join(
            sanitize_input(str(ex.get("extracted_text", ""))).lower()
            for ex in extractions
            if isinstance(ex, dict)
        )
        ocr_len = len(ocr_text.strip())
        ocr_quality = 0.0 if ocr_len == 0 else (0.4 if ocr_len < 40 else 0.7 if ocr_len < 120 else 1.0)

        invoice_hits = sum(1 for kw in _MEDICAL_INVOICE_KEYWORDS if kw in corpus)
        prescription_hits = sum(1 for kw in _PRESCRIPTION_KEYWORDS if kw in corpus)
        medical_hits = sum(
            1
            for kw in (_MEDICAL_SERVICE_KEYWORDS + _LAB_REPORT_KEYWORDS + _HOSPITAL_BILL_KEYWORDS)
            if kw in corpus
        )
        structure_ratio = float(field_validation.get("completeness_ratio", 0.0))
        cost_present = "cost" in list(field_validation.get("found_fields", []))
        provider_present = "provider" in list(field_validation.get("found_fields", []))
        date_present = "date_of_service" in list(field_validation.get("found_fields", []))

        invoice_coverage = max(
            0.0,
            min(
                1.0,
                (0.45 * self._ratio(invoice_hits, len(_MEDICAL_INVOICE_KEYWORDS)))
                + (0.25 * (1.0 if cost_present else 0.0))
                + (0.20 * (1.0 if provider_present else 0.0))
                + (0.10 * ocr_quality),
            ),
        )
        prescription_coverage = max(
            0.0,
            min(
                1.0,
                (0.70 * self._ratio(prescription_hits, len(_PRESCRIPTION_KEYWORDS)))
                + (0.20 * (1.0 if date_present else 0.0))
                + (0.10 * ocr_quality),
            ),
        )
        medical_report_coverage = max(
            0.0,
            min(
                1.0,
                (0.45 * self._ratio(medical_hits, len(_MEDICAL_SERVICE_KEYWORDS + _LAB_REPORT_KEYWORDS + _HOSPITAL_BILL_KEYWORDS)))
                + (0.35 * structure_ratio)
                + (0.20 * ocr_quality),
            ),
        )

        classifier_label = str(classifier_out.get("document_type") or classifier_out.get("label") or "").lower()
        classifier_hint = 0.0
        if classifier_label in {"medical_claim_bundle", "hybrid_bundle"}:
            classifier_hint = 0.08
        elif classifier_label == "unknown_bundle":
            classifier_hint = -0.05

        coverage_score = max(
            0.0,
            min(
                1.0,
                (0.50 * invoice_coverage)
                + (0.30 * medical_report_coverage)
                + (0.20 * prescription_coverage)
                + classifier_hint,
            ),
        )
        return {
            "medical_report_coverage": round(medical_report_coverage, 4),
            "invoice_coverage": round(invoice_coverage, 4),
            "prescription_coverage": round(prescription_coverage, 4),
            "coverage_score": round(coverage_score, 4),
            "ocr_quality": round(ocr_quality, 4),
            "classifier_hint": round(classifier_hint, 4),
            "classifier_cluster_label": classifier_label or "unknown_bundle",
        }

    def analyze(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze claim validity and return validation result.
        
        Returns:
            {
                "agent_name": str,
                "validation_status": "VALID" | "INVALID",
                "validation_score": int (0-100),
                "document_type": str,
                "missing_fields": List[str],
                "found_fields": List[str],
                "reason": str,
                "should_stop_pipeline": bool,
                "details": Dict[str, Any],
            }
        """
        LOGGER.info("ClaimValidationAgent: Starting validation")
        tool_results = self.run_tool_pipeline(
            claim_data,
            {
                "ocr_extractor": {"document_extractions": claim_data.get("document_extractions") or []},
                "document_classifier": {
                    "documents": claim_data.get("documents") or [],
                    "document_extractions": claim_data.get("document_extractions") or [],
                },
            },
        )

        corpus = self._build_document_corpus(claim_data)
        injection_detected = detect_prompt_injection(corpus)
        classifier_out = tool_results["document_classifier"].get("output") or {}
        document_type = str(classifier_out.get("document_type") or self._classify_document_type(corpus))
        field_validation = self._validate_required_fields(claim_data, corpus)
        coverage_metrics = self._compute_coverage_metrics(
            claim_data=claim_data,
            corpus=corpus,
            classifier_out=classifier_out,
            field_validation=field_validation,
        )
        coverage_score = float(coverage_metrics["coverage_score"])
        validation_score = int(round(coverage_score * 100.0))
        
        missing_count = len(field_validation["missing_fields"])
        if injection_detected:
            validation_status = "INVALID"
            reason = "Prompt injection detected in document corpus"
        elif coverage_score < 0.4:
            validation_status = "INVALID"
            reason = "Coverage score below threshold; evidence quality insufficient"
        elif coverage_score < 0.6:
            validation_status = "VALID"
            reason = "Coverage score is low but acceptable; continue with degraded confidence"
        else:
            validation_status = "VALID"
            reason = "Coverage score supports claim processing"

        should_stop_pipeline = False
        
        LOGGER.info(
            f"ClaimValidationAgent: validation_status={validation_status}, "
            f"score={validation_score}, should_stop={should_stop_pipeline}"
        )
        
        payload = {
            "agent_name": self.name,
            "validation_status": validation_status,
            "validation_score": validation_score,
            "document_type": document_type,
            "missing_fields": field_validation["missing_fields"],
            "found_fields": field_validation["found_fields"],
            "reason": reason,
            "should_stop_pipeline": should_stop_pipeline,
            "details": {
                "corpus_length": len(corpus),
                "completeness_ratio": field_validation["completeness_ratio"],
                "field_details": field_validation["field_details"],
                "coverage_metrics": coverage_metrics,
                "injection_detected": injection_detected,
                "debug_reasoning_chain": {
                    "missing_required_fields_count": missing_count,
                    "decision_threshold": 0.4,
                    "coverage_score": coverage_score,
                    "document_type_informational": document_type,
                    "classifier_cluster_only": str(
                        coverage_metrics.get("classifier_cluster_label", "unknown_bundle")
                    ),
                } if DEBUG_EXPLANATION_MODE else {},
            },
            "explanation": {
                "summary": reason,
                "reasons": [
                    f"coverage_score={coverage_score:.3f}",
                    f"found_fields={len(field_validation['found_fields'])}/5",
                    f"missing_fields={missing_count}",
                    "Document classifier label treated as informational clustering only",
                ],
                "signals": {
                    "medical_report_coverage": coverage_metrics["medical_report_coverage"],
                    "invoice_coverage": coverage_metrics["invoice_coverage"],
                    "prescription_coverage": coverage_metrics["prescription_coverage"],
                    "completeness_ratio": field_validation["completeness_ratio"],
                    "injection_detected": injection_detected,
                },
                "tool_outputs": tool_results if DEBUG_EXPLANATION_MODE else {
                    "document_classifier": tool_results.get("document_classifier", {}),
                    "ocr_extractor": tool_results.get("ocr_extractor", {}),
                },
            },
        }
        self.enforce_tool_trace(tool_results, False)
        return self.build_agent_result(
            output=payload,
            score=float(payload["validation_score"]),
            reason=str(payload["reason"]),
            tools_used=list(tool_results.keys()),
            llm_fallback_used=False,
        )
