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

VALID_CLAIM_TYPES: frozenset[DocumentType] = frozenset({
    "medical_invoice",
    "pharmacy_invoice",
    "hospital_bill",
})

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

    def _compute_validation_score(
        self,
        document_type: DocumentType,
        field_validation: Dict[str, Any],
        corpus: str,
    ) -> int:
        """Compute validation score from 0-100."""
        score = 100
        
        # Document type validity
        if document_type not in VALID_CLAIM_TYPES:
            score -= 50
        elif document_type == "unknown":
            score -= 30
        
        # Missing required fields (20 points per field)
        missing_count = len(field_validation["missing_fields"])
        score -= missing_count * 20
        
        # Partial completeness bonus
        completeness_ratio = field_validation["completeness_ratio"]
        if completeness_ratio >= 0.8:
            score += 10
        elif completeness_ratio <= 0.4:
            score -= 10
        
        # Document quality signals
        if len(corpus.strip()) < 50:
            score -= 15
        elif len(corpus.strip()) < 20:
            score -= 25
        
        # Empty extractions penalty
        extractions = field_validation.get("field_details", {})
        if not extractions:
            score -= 10
        
        return max(0, min(100, score))

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
        
        # Build document corpus
        corpus = self._build_document_corpus(claim_data)
        
        # Check for prompt injection
        injection_detected = detect_prompt_injection(corpus)
        if injection_detected:
            LOGGER.warning("ClaimValidationAgent: Prompt injection detected")
            payload = {
                "agent_name": self.name,
                "validation_status": "INVALID",
                "validation_score": 0,
                "document_type": "irrelevant_document",
                "missing_fields": ["all"],
                "found_fields": [],
                "reason": "Prompt injection or adversarial content detected in documents",
                "should_stop_pipeline": True,
                "details": {
                    "injection_detected": True,
                    "corpus_length": len(corpus),
                },
            }
            return self._ensure_contract(
                {
                    "agent": self.name,
                    "status": "DONE",
                    "output": payload,
                    "score": float(payload["validation_score"]),
                    "reason": str(payload["reason"]),
                }
            )
        
        # Classify document type
        document_type = self._classify_document_type(corpus)
        LOGGER.info(f"ClaimValidationAgent: Document classified as '{document_type}'")
        
        # Validate required fields
        field_validation = self._validate_required_fields(claim_data, corpus)
        
        # Compute validation score
        validation_score = self._compute_validation_score(
            document_type, 
            field_validation, 
            corpus
        )
        
        # Determine validation status
        missing_count = len(field_validation["missing_fields"])
        is_valid_type = document_type in VALID_CLAIM_TYPES
        
        if missing_count >= 2:
            validation_status = "INVALID"
            reason = f"Missing {missing_count} required fields: {', '.join(field_validation['missing_fields'])}"
        elif not is_valid_type:
            validation_status = "INVALID"
            if document_type == "irrelevant_document":
                reason = "Document is not medical-related"
            elif document_type in ("medical_prescription", "lab_report", "medical_certificate", "insurance_attestation"):
                reason = f"Document is a {document_type.replace('_', ' ')}, not a billable claim"
            else:
                reason = f"Document type '{document_type}' is not a valid medical claim"
        elif validation_score < 40:
            validation_status = "INVALID"
            reason = "Insufficient claim data to validate"
        else:
            validation_status = "VALID"
            reason = f"Valid {document_type.replace('_', ' ')} with {len(field_validation['found_fields'])}/5 required fields"
        
        should_stop_pipeline = validation_status == "INVALID"
        
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
                "classification_confidence": "high" if is_valid_type else "low",
                "injection_detected": injection_detected,
            },
        }
        return self._ensure_contract(
            {
                "agent": self.name,
                "status": "DONE",
                "output": payload,
                "score": float(payload["validation_score"]),
                "reason": str(payload["reason"]),
            }
        )
