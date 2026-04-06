from typing import Any, Dict, List

from .base_agent import BaseAgent

_REQUIRED = ("medical_report", "invoice", "prescription")

# Filename / text hints (French + English) when literal IDs are not used
_KEYWORD_GROUPS: Dict[str, tuple[str, ...]] = {
    "medical_report": (
        "medical_report",
        "medical report",
        "clinical report",
        "compte rendu",
        "cr medical",
        "diagnostic",
        "hospitalisation",
    ),
    "invoice": ("invoice", "facture", "facture n", "total ttc", "montant ttc", "tva"),
    "prescription": ("prescription", "ordonnance", "medicament", "médicament", "posologie"),
}


class DocumentAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Document Agent",
            role="Document Authenticity Specialist",
            goal="Verify authenticity of submitted documents",
        )

    @staticmethod
    def _corpus(documents: List[str], extractions: List[Dict[str, Any]]) -> str:
        parts: List[str] = [str(d).lower() for d in documents]
        for ex in extractions:
            parts.append((ex.get("file_name") or "").lower())
            parts.append((ex.get("extracted_text") or "").lower())
        return " ".join(parts)

    @staticmethod
    def _legacy_list_has(required_id: str, documents: List[str]) -> bool:
        if required_id in documents:
            return True
        rid = required_id.lower().replace("_", " ")
        for d in documents:
            dl = str(d).lower()
            if required_id in dl or rid in dl:
                return True
        return False

    @staticmethod
    def _keywords_hit(required_id: str, corpus: str) -> bool:
        for kw in _KEYWORD_GROUPS.get(required_id, ()):
            if kw in corpus:
                return True
        return False

    def analyze(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        documents = claim_data.get("documents") or []
        extractions = claim_data.get("document_extractions") or []
        amount = claim_data.get("amount", 0)

        score = 100.0
        reasoning: List[str] = []
        details: Dict[str, Any] = {}

        corpus = self._corpus(documents, extractions)
        effective_count = max(len(documents), len(extractions))

        if effective_count == 0:
            score -= 50
            reasoning.append("No documents submitted")
            details["doc_count"] = 0
        elif effective_count < 2:
            score -= 20
            reasoning.append("Insufficient documentation")
            details["doc_count"] = effective_count
        else:
            details["doc_count"] = effective_count

        missing_docs: List[str] = []
        for req in _REQUIRED:
            if self._legacy_list_has(req, documents):
                continue
            if self._keywords_hit(req, corpus):
                continue
            missing_docs.append(req)

        if missing_docs:
            score -= 15 * len(missing_docs)
            reasoning.append(f"Missing required documents: {', '.join(missing_docs)}")
            details["missing_docs"] = missing_docs

        if amount > 20000 and effective_count < 3:
            score -= 25
            reasoning.append("High amount claim requires additional documentation")
            details["high_amount_doc_requirement"] = True

        suspicious_patterns = ["duplicate", "copy", "scan"]
        for doc in documents:
            if any(pattern in str(doc).lower() for pattern in suspicious_patterns):
                score -= 30
                reasoning.append(f"Suspicious document name detected: {doc}")
                details["suspicious_doc"] = doc
                break

        # Weak extraction signals (empty PDF / failed OCR)
        failed_ext = [ex for ex in extractions if (ex.get("extracted_text") or "").strip() == ""]
        if extractions and len(failed_ext) == len(extractions):
            score -= 15
            reasoning.append("No extractable text from uploaded files (try OCR or text-based PDFs)")
            details["extraction_all_empty"] = True
        elif failed_ext:
            details["extraction_partial_failures"] = len(failed_ext)

        score = max(0.0, score)
        decision = score >= 50

        return {
            "agent_name": self.name,
            "decision": decision,
            "score": round(score, 2),
            "reasoning": "; ".join(reasoning) if reasoning else "Documents verified successfully",
            "details": details,
        }
