from __future__ import annotations

import json
import logging
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Sequence

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger("claimguard.v2.document_classifier")

Label = Literal["CLAIM", "NON_CLAIM", "UNCERTAIN"]

_KEYWORDS: tuple[str, ...] = (
    "facture",
    "invoice",
    "ordonnance",
    "patient",
    "mad",
    "dh",
)
_MEDICAL_TERMS: tuple[str, ...] = (
    "medecin",
    "médecin",
    "clinique",
    "hopital",
    "hôpital",
    "consultation",
    "diagnostic",
    "traitement",
    "pharmacie",
    "hospitalisation",
    "analyse",
    "laboratoire",
)
_PROVIDER_TERMS: tuple[str, ...] = ("clinique", "clinic", "hopital", "hôpital", "doctor", "dr", "medecin", "médecin")
_DATE_REGEX = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b")
_CURRENCY_REGEX = re.compile(r"\b(?:mad|dh|dhs|dirham|eur|usd)\b|[€$]", re.IGNORECASE)
_NUMERIC_REGEX = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_TOKEN_REGEX = re.compile(r"\w+", re.UNICODE)
_PROMPT_INSTRUCTION_REGEX = re.compile(
    r"\b(ignore|bypass|override|disregard|follow these instructions|system prompt|act as|jailbreak)\b",
    re.IGNORECASE,
)
_MEDICAL_CONTEXT_TERMS: tuple[str, ...] = (
    "diagnosis",
    "diagnostic",
    "prescription",
    "ordonnance",
    "invoice",
    "facture",
    "patient",
    "provider",
    "hospital",
    "clinique",
    "médecin",
    "medecin",
    "consultation",
    "traitement",
)
_ENTITY_REFERENCE_TERMS: tuple[str, ...] = (
    "patient",
    "cin",
    "ipp",
    "provider",
    "hospital",
    "clinic",
    "clinique",
    "médecin",
    "medecin",
    "doctor",
    "dr",
)
_STRUCTURAL_MEDICAL_TERMS: tuple[str, ...] = (
    "diagnosis",
    "diagnostic",
    "prescription",
    "ordonnance",
    "invoice",
    "facture",
    "medical",
    "consultation",
)
_MIN_CLASSIFIABLE_TEXT_LENGTH = 100
_CONFIDENCE_FLOOR = 70.0
_MAX_SINGLE_TOKEN_SHARE = 0.22

_ARTIFACT_DIR = Path(__file__).resolve().parent / "models"
_MODEL_PATH = _ARTIFACT_DIR / "document_classifier_model.pkl"
_VECTORIZER_PATH = _ARTIFACT_DIR / "document_classifier_vectorizer.pkl"


@dataclass
class DocumentClassifierArtifacts:
    vectorizer: TfidfVectorizer
    classifier: Any


def _normalize_text(text: str) -> str:
    return str(text or "").strip().lower()


def extract_document_features(text: str, structured_data: Dict[str, Any] | None = None) -> Dict[str, Any]:
    normalized = _normalize_text(text)
    tokens = _TOKEN_REGEX.findall(normalized)
    numeric_tokens = _NUMERIC_REGEX.findall(normalized)
    dates = _DATE_REGEX.findall(normalized)
    keyword_hits = {keyword: int(keyword in normalized) for keyword in _KEYWORDS}
    medical_hits = sum(1 for term in _MEDICAL_TERMS if term in normalized)
    provider_hits = sum(1 for term in _PROVIDER_TERMS if term in normalized)
    structured_data = structured_data or {}
    structured_signal = int(bool(structured_data))
    return {
        "text_length": len(normalized),
        "token_count": len(tokens),
        "numeric_token_count": len(numeric_tokens),
        "date_count": len(dates),
        "currency_present": int(bool(_CURRENCY_REGEX.search(normalized))),
        "medical_vocabulary_score": medical_hits,
        "provider_term_hits": provider_hits,
        "keyword_hits": keyword_hits,
        "structured_data_present": structured_signal,
    }


def _structural_signal_count(normalized: str) -> tuple[int, Dict[str, bool]]:
    monetary_signal = bool(_CURRENCY_REGEX.search(normalized) or _NUMERIC_REGEX.search(normalized))
    medical_signal = any(term in normalized for term in _STRUCTURAL_MEDICAL_TERMS)
    entity_signal = any(term in normalized for term in _ENTITY_REFERENCE_TERMS)
    categories = {
        "monetary_amount_or_number": monetary_signal,
        "medical_terms": medical_signal,
        "entity_references": entity_signal,
    }
    return sum(1 for value in categories.values() if value), categories


def _detect_negative_signals(normalized: str) -> Dict[str, Any]:
    tokens = [tok for tok in _TOKEN_REGEX.findall(normalized) if len(tok) > 2]
    token_count = len(tokens)
    frequency: Dict[str, int] = {}
    for token in tokens:
        frequency[token] = frequency.get(token, 0) + 1
    dominant_token_share = (
        (max(frequency.values()) / max(1, token_count))
        if frequency
        else 0.0
    )
    repeated_keywords = bool(
        token_count >= 15
        and dominant_token_share > _MAX_SINGLE_TOKEN_SHARE
    )
    prompt_like_instructions = bool(_PROMPT_INSTRUCTION_REGEX.search(normalized))
    has_medical_context = any(term in normalized for term in _MEDICAL_CONTEXT_TERMS)
    random_unrelated_sentences = bool(token_count >= 12 and not has_medical_context)
    return {
        "random_unrelated_sentences": random_unrelated_sentences,
        "excessive_repeated_keywords": repeated_keywords,
        "prompt_like_instructions": prompt_like_instructions,
        "dominant_token_share": round(float(dominant_token_share), 4),
    }


def _pre_classification_hard_gate(text: str) -> Dict[str, Any] | None:
    normalized = _normalize_text(text)
    if len(normalized) < _MIN_CLASSIFIABLE_TEXT_LENGTH:
        return {
            "label": "NON_CLAIM",
            "confidence": 99,
            "source": "hard_gate",
            "reason": "ocr_text_too_short",
            "gate_flags": ["OCR_TEXT_TOO_SHORT", "NON_CLAIM_HARD_GATE"],
        }
    structural_count, structural_categories = _structural_signal_count(normalized)
    if structural_count < 2:
        return {
            "label": "NON_CLAIM",
            "confidence": 99,
            "source": "hard_gate",
            "reason": "insufficient_claim_structure",
            "gate_flags": ["INSUFFICIENT_CLAIM_STRUCTURE", "NON_CLAIM_HARD_GATE"],
            "structural_signal_count": structural_count,
            "structural_categories": structural_categories,
        }
    negative_signals = _detect_negative_signals(normalized)
    if any(
        bool(negative_signals.get(key, False))
        for key in ("random_unrelated_sentences", "excessive_repeated_keywords", "prompt_like_instructions")
    ):
        return {
            "label": "NON_CLAIM",
            "confidence": 98,
            "source": "hard_gate",
            "reason": "negative_signal_detected",
            "gate_flags": ["NEGATIVE_SIGNAL_DETECTED", "NON_CLAIM_HARD_GATE"],
            "negative_signals": negative_signals,
        }
    return {
        "label": "CLAIM",
        "confidence": 100,
        "source": "hard_gate_pass",
        "structural_signal_count": structural_count,
        "structural_categories": structural_categories,
        "negative_signals": negative_signals,
    }


def _manual_confidence(features: Dict[str, Any]) -> float:
    keyword_hits = sum(int(v) for v in features.get("keyword_hits", {}).values())
    medical_score = int(features.get("medical_vocabulary_score", 0))
    currency_present = int(features.get("currency_present", 0))
    date_count = int(features.get("date_count", 0))
    numeric_count = int(features.get("numeric_token_count", 0))
    token_count = int(features.get("token_count", 0))
    provider_hits = int(features.get("provider_term_hits", 0))
    raw_score = (
        min(40, keyword_hits * 10)
        + min(20, medical_score * 3)
        + (10 if currency_present else 0)
        + min(15, date_count * 5)
        + (10 if numeric_count >= 3 else 0)
        + (5 if token_count > 25 else 0)
    )
    if date_count > 0 and currency_present:
        raw_score += 10
    if medical_score >= 2 and token_count > 12:
        raw_score += 8
    if provider_hits > 0:
        raw_score += 6
    return max(0.0, min(100.0, float(raw_score)))


def _confidence_to_label(confidence: float) -> Label:
    if confidence >= 80.0:
        return "CLAIM"
    if confidence >= 50.0:
        return "UNCERTAIN"
    return "NON_CLAIM"


def _model_confidence_to_label(predicted_label: str, probability: float) -> Label:
    conf = max(0.0, min(100.0, probability * 100.0))
    if conf < 50.0:
        return "NON_CLAIM"
    if conf < 80.0:
        return "UNCERTAIN"
    if predicted_label == "CLAIM":
        return "CLAIM"
    return "NON_CLAIM"


def train_document_classifier(
    dataset: Sequence[Dict[str, Any]],
    *,
    model_type: Literal["logistic_regression", "random_forest"] = "logistic_regression",
    model_path: Path | None = None,
    vectorizer_path: Path | None = None,
) -> Dict[str, Any]:
    if len(dataset) < 6:
        raise ValueError("Training dataset must contain at least 6 rows.")
    texts = [str(row.get("extracted_text", "")) for row in dataset]
    labels = [str(row.get("label", "NON_CLAIM")).upper() for row in dataset]
    model_path = model_path or _MODEL_PATH
    vectorizer_path = vectorizer_path or _VECTORIZER_PATH
    model_path.parent.mkdir(parents=True, exist_ok=True)

    x_train, x_valid, y_train, y_valid = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=10000)
    x_train_vec = vectorizer.fit_transform(x_train)
    x_valid_vec = vectorizer.transform(x_valid)
    if model_type == "random_forest":
        classifier = RandomForestClassifier(n_estimators=150, random_state=42)
    else:
        classifier = LogisticRegression(max_iter=1200, class_weight="balanced", random_state=42)
    classifier.fit(x_train_vec, y_train)
    accuracy = float(classifier.score(x_valid_vec, y_valid))

    with model_path.open("wb") as fp:
        pickle.dump(classifier, fp)
    with vectorizer_path.open("wb") as fp:
        pickle.dump(vectorizer, fp)
    return {
        "accuracy": round(accuracy, 4),
        "train_size": len(x_train),
        "validation_size": len(x_valid),
        "model_path": str(model_path),
        "vectorizer_path": str(vectorizer_path),
    }


def _load_artifacts(
    model_path: Path | None = None,
    vectorizer_path: Path | None = None,
) -> DocumentClassifierArtifacts | None:
    model_path = model_path or _MODEL_PATH
    vectorizer_path = vectorizer_path or _VECTORIZER_PATH
    if not model_path.exists() or not vectorizer_path.exists():
        return None
    try:
        with model_path.open("rb") as fp:
            model = pickle.load(fp)
        with vectorizer_path.open("rb") as fp:
            vectorizer = pickle.load(fp)
        return DocumentClassifierArtifacts(vectorizer=vectorizer, classifier=model)
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("document_classifier_artifacts_load_failed error=%s", exc)
        return None


def classify_document(text: str, structured_data: Dict[str, Any] | None = None) -> Dict[str, Any]:
    features = extract_document_features(text, structured_data)
    gate = _pre_classification_hard_gate(text)
    if gate and gate.get("label") == "NON_CLAIM":
        result = {
            "label": "NON_CLAIM",
            "confidence": int(gate.get("confidence", 99)),
            "source": str(gate.get("source", "hard_gate")),
            "features": features,
            "raw_prediction": None,
            "reason": str(gate.get("reason", "")),
            "gate_flags": list(gate.get("gate_flags", [])),
            "structural_signal_count": int(gate.get("structural_signal_count", 0)),
            "structural_categories": dict(gate.get("structural_categories", {})),
            "negative_signals": dict(gate.get("negative_signals", {})),
        }
        LOGGER.info(
            "doc_classifier hard_gate prediction=%s confidence=%d reason=%s features=%s",
            result["label"],
            result["confidence"],
            result["reason"],
            json.dumps(features, ensure_ascii=False),
        )
        return result

    artifacts = _load_artifacts()
    normalized = _normalize_text(text)

    if artifacts is None:
        confidence = _manual_confidence(features)
        label = _confidence_to_label(confidence)
        result = {
            "label": label,
            "confidence": int(round(confidence)),
            "source": "heuristic_fallback",
            "features": features,
            "raw_prediction": None,
            "reason": "",
            "gate_flags": [],
            "structural_signal_count": int(gate.get("structural_signal_count", 0)) if gate else 0,
            "structural_categories": dict(gate.get("structural_categories", {})) if gate else {},
            "negative_signals": dict(gate.get("negative_signals", {})) if gate else {},
        }
        if confidence < _CONFIDENCE_FLOOR:
            result["label"] = "UNCERTAIN"
            result["reason"] = "confidence_below_floor"
            result["gate_flags"] = ["CONFIDENCE_BELOW_FLOOR", "FORCE_HUMAN_REVIEW"]
        LOGGER.info(
            "doc_classifier prediction=%s confidence=%d source=%s features=%s",
            result["label"],
            result["confidence"],
            result["source"],
            json.dumps(features, ensure_ascii=False),
        )
        return result

    try:
        vectorized = artifacts.vectorizer.transform([normalized])
        probabilities = artifacts.classifier.predict_proba(vectorized)[0]
        class_labels = list(artifacts.classifier.classes_)
        best_idx = max(range(len(probabilities)), key=lambda idx: probabilities[idx])
        predicted_label = str(class_labels[best_idx]).upper()
        best_proba = float(probabilities[best_idx])
        mapped_label = _model_confidence_to_label(predicted_label, best_proba)
        confidence = int(round(best_proba * 100.0))
        result = {
            "label": mapped_label,
            "confidence": confidence,
            "source": "ml_model",
            "features": features,
            "raw_prediction": predicted_label,
            "reason": "",
            "gate_flags": [],
            "structural_signal_count": int(gate.get("structural_signal_count", 0)) if gate else 0,
            "structural_categories": dict(gate.get("structural_categories", {})) if gate else {},
            "negative_signals": dict(gate.get("negative_signals", {})) if gate else {},
        }
        if best_proba * 100.0 < _CONFIDENCE_FLOOR:
            result["label"] = "UNCERTAIN"
            result["reason"] = "confidence_below_floor"
            result["gate_flags"] = ["CONFIDENCE_BELOW_FLOOR", "FORCE_HUMAN_REVIEW"]
        LOGGER.info(
            "doc_classifier prediction=%s confidence=%d source=%s raw=%s features=%s",
            result["label"],
            result["confidence"],
            result["source"],
            predicted_label,
            json.dumps(features, ensure_ascii=False),
        )
        return result
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("doc_classifier_runtime_failed error=%s", exc)
        fallback_conf = _manual_confidence(features)
        label = _confidence_to_label(fallback_conf)
        reason = ""
        flags: list[str] = []
        if fallback_conf < _CONFIDENCE_FLOOR:
            label = "UNCERTAIN"
            reason = "confidence_below_floor"
            flags = ["CONFIDENCE_BELOW_FLOOR", "FORCE_HUMAN_REVIEW"]
        return {
            "label": label,
            "confidence": int(round(fallback_conf)),
            "source": "heuristic_after_runtime_error",
            "features": features,
            "raw_prediction": None,
            "reason": reason,
            "gate_flags": flags,
            "structural_signal_count": int(gate.get("structural_signal_count", 0)) if gate else 0,
            "structural_categories": dict(gate.get("structural_categories", {})) if gate else {},
            "negative_signals": dict(gate.get("negative_signals", {})) if gate else {},
        }
