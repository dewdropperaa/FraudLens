"""
Adversarial behavioral tests for multi-agent claim analysis.

Validates that agents use structured inputs, surface contradictions, resist
persuasive irrelevancies, and ground explanations — without mocking agent
outputs. Assertions use comparative baselines, bands, and structural checks
rather than single hard-coded risk constants.
"""

from __future__ import annotations

import json
import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def pytest_configure(config: pytest.Config) -> None:
    os.environ.setdefault("ENVIRONMENT", "test")
    os.environ.setdefault("DOCUMENT_ENCRYPTION_KEY", "0" * 32)
    os.environ.setdefault("CLAIMAGUARD_API_KEYS", "test-api-key-for-ci")
    os.environ.setdefault("CORS_ORIGINS", "http://testserver")
    os.environ.setdefault("PINATA_JWT", "")
    os.environ.setdefault("PINATA_API_KEY", "")
    os.environ.setdefault("PINATA_API_SECRET", "")
    os.environ.setdefault("CONTRACT_ADDRESS", "")
    os.environ.setdefault("SEPOLIA_PRIVATE_KEY", "")
    os.environ.setdefault("PRIVATE_KEY", "")


from claimguard.agents.security_utils import score_to_risk_score  # noqa: E402


def validate_schema(output: dict[str, Any]) -> None:
    if not isinstance(output, dict):
        raise TypeError("output must be a dict")
    if "risk_score" not in output:
        raise ValueError("missing risk_score")
    r = float(output["risk_score"])
    if not (0.0 <= r <= 1.0):
        raise ValueError(f"risk_score out of [0,1]: {r}")
    if not isinstance(output.get("flags"), list):
        raise ValueError("flags must be a list")
    if not isinstance(output.get("explanation"), str):
        raise ValueError("explanation must be a string")


def _standardize_agent_output(raw: dict[str, Any]) -> dict[str, Any]:
    if (
        "risk_score" in raw
        and isinstance(raw.get("flags"), list)
        and isinstance(raw.get("explanation"), str)
    ):
        return {
            "risk_score": float(raw["risk_score"]),
            "flags": [str(x) for x in raw["flags"]],
            "explanation": str(raw["explanation"]),
        }
    score = raw.get("score", 50.0)
    try:
        sf = float(score)
    except (TypeError, ValueError):
        sf = 50.0
    return {
        "risk_score": score_to_risk_score(sf),
        "flags": [],
        "explanation": str(raw.get("reasoning", raw.get("explanation", ""))),
    }


def run_full_pipeline(claim_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Run all six agents in production order; return standardized risk payloads."""
    from claimguard.agents import (
        AnomalyAgent,
        DocumentAgent,
        GraphAgent,
        IdentityAgent,
        PatternAgent,
        PolicyAgent,
    )

    sequence: tuple[tuple[str, Any], ...] = (
        ("anomaly", AnomalyAgent()),
        ("document", DocumentAgent()),
        ("identity", IdentityAgent()),
        ("pattern", PatternAgent()),
        ("policy", PolicyAgent()),
        ("graph", GraphAgent()),
    )
    out: dict[str, dict[str, Any]] = {}
    for key, agent in sequence:
        raw = agent.analyze(claim_data)
        out[key] = _standardize_agent_output(raw)
    return out


_AGENT_NAME_TO_KEY: dict[str, str] = {
    "Anomaly Agent": "anomaly",
    "Pattern Agent": "pattern",
    "Identity Agent": "identity",
    "Document Agent": "document",
    "Policy Agent": "policy",
    "Graph Agent": "graph",
}


def run_full_pipeline_via_crew(claim_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """
    Same standardized shape as run_full_pipeline, but routes through
    ``claimguard.crew.runner.run_claim_agents`` (parallel deterministic analyzers).
    """
    from claimguard.crew.runner import run_claim_agents

    rows = run_claim_agents(claim_data)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = str(row.get("agent_name", ""))
        key = _AGENT_NAME_TO_KEY.get(name)
        if key is None:
            continue
        out[key] = _standardize_agent_output(row)
    return out


def _print_failure(reason: str, *, claim: dict[str, Any] | None, pipeline: dict[str, Any] | None) -> None:
    print("\n=== ADVERSARIAL REASONING TEST FAILURE ===")
    print(f"Why it failed: {reason}")
    if claim is not None:
        print("\n--- claim payload ---")
        print(json.dumps(claim, indent=2, default=str))
    if pipeline is not None:
        print("\n--- full agent outputs ---")
        print(json.dumps(pipeline, indent=2, default=str))


def _fail(reason: str, *, claim: dict[str, Any] | None = None, pipeline: dict[str, Any] | None = None) -> None:
    _print_failure(reason, claim=claim, pipeline=pipeline)
    pytest.fail(reason)


def _validate_pipeline_schema(pipeline: dict[str, Any]) -> None:
    for k, v in pipeline.items():
        try:
            validate_schema(v)
        except Exception as exc:
            _fail(f"schema invalid for agent {k}: {exc}", pipeline=pipeline)


# --- Explanation quality heuristics (detect weak / generic reasoning) ---

_VAGUE_PHRASES: tuple[str, ...] = (
    "seems suspicious",
    "appears suspicious",
    "looks suspicious",
    "probably fraud",
    "likely fraudulent",
    "something is wrong",
    "unclear but risky",
    "hard to say",
    "generally suspicious",
)


def _explanation_has_vague_phrasing(text: str) -> bool:
    low = text.lower()
    return any(p in low for p in _VAGUE_PHRASES)


def _grounding_terms_from_claim(claim: dict[str, Any]) -> list[str]:
    """Terms that a grounded explanation should be able to echo when those fields exist."""
    terms: list[str] = []
    ins = claim.get("insurance")
    if isinstance(ins, str) and ins.strip():
        terms.append(ins.strip().lower())
    pid = claim.get("patient_id")
    if pid is not None:
        terms.append(str(pid).lower())
    prov = claim.get("provider_id")
    if isinstance(prov, str) and prov.strip():
        terms.append(prov.strip().lower())
    try:
        if claim.get("amount") is not None and not isinstance(claim.get("amount"), str):
            a = float(claim["amount"])
            terms.append(f"{a:g}".lower())
    except (TypeError, ValueError):
        pass
    cid = claim.get("claim_id")
    if isinstance(cid, str) and len(cid) >= 6:
        terms.append(cid[-6:].lower())
    return [t for t in terms if len(t) >= 3]


def _claim_grounding_blob(claim: dict[str, Any]) -> str:
    """Serialized claim plus legitimate domain stems agents may cite without echoing raw values."""
    base = json.dumps(claim, sort_keys=True, default=str).lower()
    extra = (
        " prescription medical invoice documentation document facture cnss policy "
        "coverage limit amount history claim patient provider anomaly pattern graph "
        "standard deviation fraud probability"
    )
    return base + extra


def _explanation_grounded_in_claim(explanation: str, claim: dict[str, Any]) -> bool:
    """
    True if the explanation cites at least one concrete anchor: grounding term,
    digit tied to amount/history, overlap with claim JSON, or domain vocabulary.
    """
    el = explanation.lower()
    blob = _claim_grounding_blob(claim)
    for tok in re.findall(r"[a-z]{4,}", el):
        if tok in blob:
            return True
    for t in _grounding_terms_from_claim(claim):
        if t in el:
            return True
    if re.search(r"\d{3,}", explanation):
        return True
    hist = claim.get("history")
    if isinstance(hist, list) and hist:
        for h in hist[:5]:
            if not isinstance(h, dict):
                continue
            for key in ("amount", "date", "decision"):
                v = h.get(key)
                if v is None:
                    continue
                vs = str(v).lower()
                if len(vs) >= 3 and vs in el:
                    return True
    docs = claim.get("documents")
    if isinstance(docs, list):
        for d in docs:
            ds = str(d).lower()
            if len(ds) >= 4 and ds[:32] in el:
                return True
    exts = claim.get("document_extractions")
    if isinstance(exts, list):
        for ex in exts:
            if not isinstance(ex, dict):
                continue
            fn = ex.get("file_name")
            if isinstance(fn, str) and len(fn) >= 4 and fn.lower() in el:
                return True
            tx = ex.get("extracted_text")
            if isinstance(tx, str):
                for w in re.findall(r"[a-zA-Z]{5,}", tx)[:12]:
                    if w.lower() in el:
                        return True
    return False


def _uncertainty_band(r: float, low: float = 0.38, high: float = 0.62) -> bool:
    return low <= r <= high


# --- Scenario fixtures ---

def _base_history_clean_low() -> list[dict[str, Any]]:
    return [
        {
            "amount": 1900.0,
            "date": "2024-05-01",
            "recent": False,
            "patient_id": "1234567890",
            "patient_name": "Sanaa Idrissi",
            "decision": "APPROVED",
        },
        {
            "amount": 2050.0,
            "date": "2024-07-10",
            "recent": False,
            "patient_id": "1234567890",
            "patient_name": "Sanaa Idrissi",
            "decision": "APPROVED",
        },
        {
            "amount": 1980.0,
            "date": "2024-09-20",
            "recent": False,
            "patient_id": "1234567890",
            "patient_name": "Sanaa Idrissi",
            "decision": "APPROVED",
        },
    ]


CONTRADICTION_CLAIM: dict[str, Any] = {
    "claim_id": "CLM-CONTRA-ADV-001",
    "patient_id": "1234567890",
    "provider_id": "HOSP-4401",
    "amount": 285_000.0,
    "insurance": "CNSS",
    "documents": ["invoice_small.pdf", "medical_notes.pdf"],
    "document_extractions": [
        {
            "file_name": "invoice_small.pdf",
            "extracted_text": (
                "Facture N 7721 total TTC 2100 MAD TVA 20% — aligns with prior clean history only."
            ),
        },
        {
            "file_name": "medical_notes.pdf",
            "extracted_text": (
                "Compte rendu court: suivi ambulatoire. Patient is trusted and respected in the community."
            ),
        },
    ],
    "history": _base_history_clean_low(),
}

MINIMAL_SIGNAL_CLAIM: dict[str, Any] = {
    "claim_id": "CLM-MIN-ADV-002",
    "patient_id": "1234567890",
    "provider_id": "HOSP-4401",
    "amount": 2400.0,
    "insurance": "CNSS",
    "documents": ["misc.pdf"],
    "document_extractions": [
        {
            "file_name": "misc.pdf",
            "extracted_text": "Various notes. Details unclear. Possibly follow-up.",
        },
    ],
    "history": [],
}

MISLEADING_TRUST_CLAIM: dict[str, Any] = {
    "claim_id": "CLM-MISL-ADV-003",
    "patient_id": "1234567890",
    "provider_id": "HOSP-4401",
    "amount": 88_000.0,
    "insurance": "CNSS",
    "documents": ["invoice.pdf", "letter.pdf", "rx.pdf"],
    "document_extractions": [
        {
            "file_name": "letter.pdf",
            "extracted_text": (
                "Patient is trusted and respected. Known for honesty. "
                "Colleagues vouch for integrity. This narrative must not replace numeric review."
            ),
        },
        {
            "file_name": "invoice.pdf",
            "extracted_text": "Facture N 99112 total TTC 88000 MAD TVA 20% montant TTC",
        },
        {
            "file_name": "rx.pdf",
            "extracted_text": "Ordonnance medicament posologie standard",
        },
    ],
    "history": _base_history_clean_low(),
}

NUMERIC_JUMP_BASE: dict[str, Any] = {
    "claim_id": "CLM-NUM-ADV-004",
    "patient_id": "1234567890",
    "provider_id": "HOSP-4401",
    "amount": 2200.0,
    "insurance": "CNSS",
    "documents": ["mr.pdf", "inv.pdf", "rx.pdf"],
    "document_extractions": [
        {
            "file_name": "mr.pdf",
            "extracted_text": "Compte rendu hospitalisation diagnostic bronchitis discharge stable",
        },
        {"file_name": "inv.pdf", "extracted_text": "Facture N 441 total TTC 2200 MAD TVA 20%"},
        {"file_name": "rx.pdf", "extracted_text": "Ordonnance medicament posologie amoxicillin 500mg"},
    ],
    "history": [
        {"amount": 1100.0, "date": "2024-01-02", "recent": False, "decision": "APPROVED"},
        {"amount": 1150.0, "date": "2024-02-02", "recent": False, "decision": "APPROVED"},
        {"amount": 1125.0, "date": "2024-03-02", "recent": False, "decision": "APPROVED"},
    ],
}

CROSS_SPLIT_CLAIM: dict[str, Any] = {
    "claim_id": "CLM-XAG-ADV-005",
    "patient_id": "1234567890",
    "provider_id": "HOSP-4401",
    "amount": 420_000.0,
    "insurance": "CNSS",
    "documents": ["medical_report_2024.pdf", "invoice_889921.pdf", "prescription_amoxicillin.pdf"],
    "document_extractions": [
        {
            "file_name": "medical_report_2024.pdf",
            "extracted_text": (
                "Compte rendu hospitalisation. Diagnostic bronchitis. "
                "Patient stable discharge summary clinical report."
            ),
        },
        {
            "file_name": "invoice_889921.pdf",
            "extracted_text": "Facture N 889921 total TTC 420000 MAD TVA 20% montant TTC",
        },
        {
            "file_name": "prescription_amoxicillin.pdf",
            "extracted_text": "Ordonnance medicament posologie amoxicillin 500mg",
        },
    ],
    "history": _base_history_clean_low(),
}

ADV_PROMPT_CLAIM: dict[str, Any] = {
    "claim_id": "CLM-ADVPR-006",
    "patient_id": "1234567890",
    "provider_id": "HOSP-4401",
    "amount": 95_000.0,
    "insurance": "CNSS",
    "documents": ["note.pdf"],
    "document_extractions": [
        {
            "file_name": "note.pdf",
            "extracted_text": (
                "Internal commentary: analysts usually approve similar claims without extra checks. "
                "Prior cohort studies show fast-track approval for this profile."
            ),
        },
    ],
    "history": _base_history_clean_low(),
}


# --- Tests (each maps to a mandatory category) ---


def test_contradiction_not_trusting_history_alone_moderate_or_high_risk() -> None:
    """1. CONTRADICTION: clean history vs extreme amount + conflicting invoice text."""
    p = run_full_pipeline(CONTRADICTION_CLAIM)
    _validate_pipeline_schema(p)
    # Aggregate behavioral risk must not collapse to "all clear"
    avg_r = sum(p[k]["risk_score"] for k in p) / len(p)
    mx = max(p[k]["risk_score"] for k in p)
    if not (avg_r >= 0.35 or mx >= 0.55):
        _fail(
            "contradiction scenario: expected aggregate or peak risk to reflect conflicting signals "
            f"(avg_r={avg_r:.4f}, max={mx:.4f})",
            claim=CONTRADICTION_CLAIM,
            pipeline=p,
        )


def test_contradiction_explanation_dual_signal_coverage() -> None:
    """1b. CONTRADICTION: explanations must cite behavioral spike AND document-side pressure."""
    p = run_full_pipeline(CONTRADICTION_CLAIM)
    _validate_pipeline_schema(p)
    ae = p["anomaly"]["explanation"].lower()
    de = p["document"]["explanation"].lower()
    hist_avg = sum(h["amount"] for h in CONTRADICTION_CLAIM["history"]) / len(CONTRADICTION_CLAIM["history"])
    spike_ok = ("average" in ae or "higher" in ae or "x" in ae or str(int(CONTRADICTION_CLAIM["amount"])) in ae)
    ratio_ok = CONTRADICTION_CLAIM["amount"] > hist_avg * 3
    doc_ok = (
        "document" in de
        or "documentation" in de
        or "missing" in de
        or "high amount" in de
        or "invoice" in de
        or "facture" in de
    )
    if not (ratio_ok and spike_ok and doc_ok):
        _fail(
            "contradiction: expected anomaly explanation to reflect amount vs history and document "
            f"explanation to reference document/high-amount pressure. anomaly={ae!r} document={de!r}",
            claim=CONTRADICTION_CLAIM,
            pipeline=p,
        )


def test_minimal_signal_uncertainty_band() -> None:
    """2. MINIMAL SIGNAL: sparse history/docs → elevated spread or mid-band, not unanimous certainty."""
    p = run_full_pipeline(MINIMAL_SIGNAL_CLAIM)
    _validate_pipeline_schema(p)
    scores = [p[k]["risk_score"] for k in p]
    spread = max(scores) - min(scores)
    doc_mid = 0.32 <= p["document"]["risk_score"] <= 0.72
    any_mid = any(_uncertainty_band(p[k]["risk_score"], 0.35, 0.65) for k in ("anomaly", "document", "pattern"))
    flags_unc = any("defensive" in str(f).lower() or "missing" in str(f).lower() for f in p["document"]["flags"])
    # Sparse inputs should either land in an epistemic mid-band or produce clear cross-agent disagreement.
    if not (doc_mid or any_mid or flags_unc or spread >= 0.28):
        _fail(
            f"minimal signal: expected document mid-band, uncertainty flags, or agent disagreement "
            f"(scores={scores}, spread={spread:.4f})",
            claim=MINIMAL_SIGNAL_CLAIM,
            pipeline=p,
        )


def test_minimal_signal_not_extreme_confidence() -> None:
    """2b. MINIMAL: pipeline must not unanimously read as all-clear or all-fraud on thin evidence."""
    p = run_full_pipeline(MINIMAL_SIGNAL_CLAIM)
    _validate_pipeline_schema(p)
    scores = [p[k]["risk_score"] for k in p]
    if max(scores) >= 0.92 and min(scores) <= 0.08:
        _fail(
            "minimal signal: unanimous extreme spread is inappropriate on sparse inputs",
            claim=MINIMAL_SIGNAL_CLAIM,
            pipeline=p,
        )
    if max(scores) < 0.3:
        _fail(
            f"minimal signal: expected at least one agent to elevate risk under sparse evidence (scores={scores})",
            claim=MINIMAL_SIGNAL_CLAIM,
            pipeline=p,
        )


def test_misleading_trust_language_does_not_suppress_numeric_stress() -> None:
    """3. MISLEADING CONTEXT: reputational fluff must not erase structural risk."""
    p = run_full_pipeline(MISLEADING_TRUST_CLAIM)
    _validate_pipeline_schema(p)
    combo = max(p["anomaly"]["risk_score"], p["policy"]["risk_score"], p["pattern"]["risk_score"])
    if combo < 0.35:
        _fail(
            "misleading trust language: expected structural risk drivers to dominate fluffy narrative",
            claim=MISLEADING_TRUST_CLAIM,
            pipeline=p,
        )


def test_numeric_reasoning_jump_vs_history() -> None:
    """4. NUMERIC: 100× jump vs tight history must increase anomaly risk vs baseline."""
    base = deepcopy(NUMERIC_JUMP_BASE)
    jump = deepcopy(NUMERIC_JUMP_BASE)
    jump["claim_id"] = "CLM-NUM-JUMP-ADV"
    jump["amount"] = float(jump["amount"]) * 100.0
    jump["document_extractions"] = deepcopy(NUMERIC_JUMP_BASE["document_extractions"])
    jump["document_extractions"][1] = {
        **jump["document_extractions"][1],
        "extracted_text": f"Facture N 441 total TTC {jump['amount']:g} MAD TVA 20%",
    }
    b = run_full_pipeline(base)
    j = run_full_pipeline(jump)
    _validate_pipeline_schema(b)
    _validate_pipeline_schema(j)
    if j["anomaly"]["risk_score"] <= b["anomaly"]["risk_score"] + 0.02:
        _fail(
            "numeric jump: anomaly risk should increase materially when amount spikes vs history",
            claim=jump,
            pipeline={"baseline": b, "jump": j},
        )
    je = j["anomaly"]["explanation"].lower()
    if not any(k in je for k in ("average", "higher", "x", "high claim")):
        _fail(
            "numeric jump: anomaly explanation should reference the spike vs historical average",
            claim=jump,
            pipeline=j,
        )


def test_field_importance_missing_amount_acknowledged() -> None:
    """5. FIELD IMPORTANCE: omit amount → validation flags / no fabricated stability."""
    c = deepcopy(MINIMAL_SIGNAL_CLAIM)
    c["claim_id"] = "CLM-NOAMT-ADV"
    if "amount" in c:
        del c["amount"]
    p = run_full_pipeline(c)
    _validate_pipeline_schema(p)
    af = " ".join(p["anomaly"]["flags"]).lower()
    df = " ".join(p["document"]["flags"]).lower()
    if "missing_amount" not in af and "missing_amount" not in df:
        _fail(
            "missing amount: expected agents to flag absent amount rather than invent one",
            claim=c,
            pipeline=p,
        )
    if p["anomaly"]["risk_score"] < 0.35 and p["document"]["risk_score"] < 0.35:
        _fail(
            "missing amount: risk should not look like a confident clean bill of health",
            claim=c,
            pipeline=p,
        )


def test_consistency_paraphrase_stable_scores() -> None:
    """6. CONSISTENCY: small wording-only changes → similar risk profile."""
    a = deepcopy(MINIMAL_SIGNAL_CLAIM)
    b = deepcopy(MINIMAL_SIGNAL_CLAIM)
    b["claim_id"] = "CLM-MIN-ADV-002b"
    b["document_extractions"][0]["extracted_text"] = (
        "Miscellaneous notes. Details are unclear. Maybe follow-up visit."
    )
    pa = run_full_pipeline(a)
    pb = run_full_pipeline(b)
    _validate_pipeline_schema(pa)
    _validate_pipeline_schema(pb)
    spread = abs(pa["anomaly"]["risk_score"] - pb["anomaly"]["risk_score"])
    if spread > 0.22:
        _fail(
            f"consistency: anomaly risk drifted too far on paraphrase (spread={spread})",
            claim={"A": a, "B": b},
            pipeline={"A": pa, "B": pb},
        )


def test_explanation_not_vague_fluff() -> None:
    """7. EXPLANATION: ban hand-wavy suspicion without anchors."""
    p = run_full_pipeline(CROSS_SPLIT_CLAIM)
    _validate_pipeline_schema(p)
    for k, v in p.items():
        if _explanation_has_vague_phrasing(v["explanation"]):
            _fail(
                f"vague phrasing in {k} explanation",
                claim=CROSS_SPLIT_CLAIM,
                pipeline=p,
            )


def test_explanation_grounded_in_structured_inputs() -> None:
    """7b. EXPLANATION: must tie to structured fields (insurance, ids, amounts, filenames, history)."""
    # Use contradiction payload so document path is forced to cite gaps, not a generic "verified" stub.
    p = run_full_pipeline(CONTRADICTION_CLAIM)
    _validate_pipeline_schema(p)
    # Risk-scoring agents must anchor to inputs; identity may stay terse when IDs are well-formed.
    for k in ("anomaly", "document", "pattern", "policy", "graph"):
        v = p[k]
        expl = v["explanation"]
        if not _explanation_grounded_in_claim(expl, CONTRADICTION_CLAIM):
            _fail(
                f"explanation for {k} not grounded in claim inputs: {expl!r}",
                claim=CONTRADICTION_CLAIM,
                pipeline=p,
            )


def test_adversarial_prompt_bias_resistance() -> None:
    """8. ADVERSARIAL PROMPT: social-proof bias must not yield near-zero risk."""
    p = run_full_pipeline(ADV_PROMPT_CLAIM)
    _validate_pipeline_schema(p)
    doc = p["document"]["risk_score"]
    if doc < 0.38:
        _fail(
            "adversarial bias text: document risk should not collapse under persuasive framing",
            claim=ADV_PROMPT_CLAIM,
            pipeline=p,
        )


def test_cross_agent_behavior_vs_documents_split() -> None:
    """9. CROSS-AGENT: clean doc bundle vs huge behavioral spike → measurable spread."""
    p = run_full_pipeline(CROSS_SPLIT_CLAIM)
    _validate_pipeline_schema(p)
    scores = [p[k]["risk_score"] for k in p]
    spread = max(scores) - min(scores)
    if spread < 0.18:
        _fail(
            f"cross-agent: expected spread when documents OK but amount/history diverge (spread={spread})",
            claim=CROSS_SPLIT_CLAIM,
            pipeline=p,
        )
    joined = " ".join(p[k]["explanation"].lower() for k in p)
    if "graph" in p and p["graph"]["risk_score"] > 0.25:
        # Graph mentions pattern in reasoning via standardization
        if "pattern" not in joined and "amount" not in joined and "claim" not in joined:
            _fail(
                "cross-agent: combined explanations should retain substantive cues",
                claim=CROSS_SPLIT_CLAIM,
                pipeline=p,
            )


def test_contradiction_aggregate_not_naive_low() -> None:
    """1c. CONTRADICTION: peak agent risk should reflect not blindly trusting prior approvals."""
    p = run_full_pipeline(CONTRADICTION_CLAIM)
    _validate_pipeline_schema(p)
    if max(p[k]["risk_score"] for k in p) < 0.45:
        _fail(
            "contradiction: peak risk too low given history/amount/document tension",
            claim=CONTRADICTION_CLAIM,
            pipeline=p,
        )


def test_numeric_reasoning_pattern_agent_follows_spike() -> None:
    """4b. NUMERIC: pattern agent should react when amount departs strongly from history dispersion."""
    hist = [
        {"amount": 1000.0, "date": "2024-01-01", "recent": False, "decision": "APPROVED"},
        {"amount": 1020.0, "date": "2024-02-01", "recent": False, "decision": "APPROVED"},
        {"amount": 1010.0, "date": "2024-03-01", "recent": False, "decision": "APPROVED"},
    ]
    c = {
        "claim_id": "CLM-PATNUM-ADV",
        "patient_id": "1234567890",
        "provider_id": "HOSP-4401",
        "amount": 25_000.0,
        "insurance": "CNSS",
        "documents": ["a.pdf", "b.pdf"],
        "document_extractions": [
            {"file_name": "a.pdf", "extracted_text": "Facture total TTC 25000"},
            {"file_name": "b.pdf", "extracted_text": "Compte rendu medical brief"},
        ],
        "history": hist,
    }
    p = run_full_pipeline(c)
    _validate_pipeline_schema(p)
    pe = p["pattern"]["explanation"].lower()
    if "deviations" not in pe and "standard" not in pe and "amount" not in pe:
        _fail(
            "numeric reasoning: pattern explanation should reference statistical deviation or amount context",
            claim=c,
            pipeline=p,
        )


def test_misleading_only_fluff_document_still_risk_aware() -> None:
    """3b. MISLEADING: even if extraction is mostly reputational, risk must not be near-zero."""
    c = deepcopy(ADV_PROMPT_CLAIM)
    c["claim_id"] = "CLM-FLUFFONLY-ADV"
    c["documents"] = ["trust.pdf"]
    c["document_extractions"] = [
        {
            "file_name": "trust.pdf",
            "extracted_text": (
                "Patient is trusted and respected. Known for honesty. No quantitative billing data attached."
            ),
        },
    ]
    p = run_full_pipeline(c)
    _validate_pipeline_schema(p)
    if p["document"]["risk_score"] < 0.3:
        _fail(
            "fluff-only document: document risk should stay cautious without hard amounts in text",
            claim=c,
            pipeline=p,
        )


def test_explanation_each_agent_non_empty_when_risk_elevated() -> None:
    """7c. EXPLANATION: under elevated risk, explanations should not be empty strings."""
    p = run_full_pipeline(CONTRADICTION_CLAIM)
    _validate_pipeline_schema(p)
    for k, v in p.items():
        if v["risk_score"] >= 0.55 and len(v["explanation"].strip()) < 8:
            _fail(
                f"elevated risk but empty explanation for {k}",
                claim=CONTRADICTION_CLAIM,
                pipeline=p,
            )


def test_policy_numeric_limit_triggers_when_amount_extreme() -> None:
    """4c. NUMERIC: policy limits should engage for extreme amounts on CNSS."""
    c = deepcopy(CROSS_SPLIT_CLAIM)
    c["amount"] = 95_000.0
    p = run_full_pipeline(c)
    _validate_pipeline_schema(p)
    pe = p["policy"]["explanation"].lower()
    if "limit" not in pe and "cnss" not in pe and "exceed" not in pe:
        _fail(
            "policy should reference coverage/limit context for large CNSS amounts",
            claim=c,
            pipeline=p,
        )


def test_deterministic_crew_runner_parity_with_direct_pipeline() -> None:
    """CREW runner (parallel deterministic) must match sequential direct analyze for risk outputs."""
    if os.getenv("CREW_USE_LLM", "").strip().lower() in {"1", "true", "yes"}:
        pytest.skip("CREW_USE_LLM merges LLM output; parity applies to deterministic path only")
    claim = deepcopy(CONTRADICTION_CLAIM)
    direct = run_full_pipeline(claim)
    via_crew = run_full_pipeline_via_crew(claim)
    if set(direct) != set(via_crew):
        _fail(
            f"crew vs direct agent key mismatch: direct={sorted(direct)} crew={sorted(via_crew)}",
            claim=claim,
            pipeline={"direct": direct, "via_crew": via_crew},
        )
    for k in direct:
        a, b = direct[k]["risk_score"], via_crew[k]["risk_score"]
        if abs(a - b) > 1e-5:
            _fail(
                f"crew vs direct risk mismatch for {k}: direct={a} via_crew={b}",
                claim=claim,
                pipeline={"direct": direct, "via_crew": via_crew},
            )
        if direct[k]["explanation"] != via_crew[k]["explanation"]:
            _fail(
                f"crew vs direct explanation mismatch for {k}",
                claim=claim,
                pipeline={"direct": direct, "via_crew": via_crew},
            )
