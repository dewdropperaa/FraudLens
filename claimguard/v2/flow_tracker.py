class ClaimFlowTracker:
    def __init__(self, claim_id: str):
        self.claim_id = claim_id
        self.steps = {
            "OCR Extraction": {"status": "PENDING", "score": None, "confidence": None, "explanation": "", "is_fraud": False},
            "ClaimValidation": {"status": "PENDING", "score": None, "confidence": None, "explanation": "", "is_fraud": False},
            "IdentityAgent": {"status": "PENDING", "score": None, "confidence": None, "explanation": "", "is_fraud": False},
            "DocumentAgent": {"status": "PENDING", "score": None, "confidence": None, "explanation": "", "is_fraud": False},
            "PolicyAgent": {"status": "PENDING", "score": None, "confidence": None, "explanation": "", "is_fraud": False},
            "AnomalyAgent": {"status": "PENDING", "score": None, "confidence": None, "explanation": "", "is_fraud": False},
            "PatternAgent": {"status": "PENDING", "score": None, "confidence": None, "explanation": "", "is_fraud": False},
            "GraphRiskAgent": {"status": "PENDING", "score": None, "confidence": None, "explanation": "", "is_fraud": False},
            "Consensus": {"status": "PENDING", "score": None, "confidence": None, "explanation": "", "is_fraud": False},
            "HumanReview": {"status": "PENDING", "score": None, "confidence": None, "explanation": "", "is_fraud": False}
        }

    def update(self, step: str, status: str, score: float = None, confidence: float = None, explanation: str = "", is_fraud: bool = False):
        if step == "GraphAgent":
            step = "GraphRiskAgent"
        if step in self.steps:
            self.steps[step].update({
                "status": status,
                "score": score if score is not None else self.steps[step].get("score"),
                "confidence": confidence if confidence is not None else self.steps[step].get("confidence"),
                "explanation": explanation if explanation else self.steps[step].get("explanation"),
                "is_fraud": is_fraud if is_fraud is not None else self.steps[step].get("is_fraud")
            })

    def get_state(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "steps": self.steps
        }

_trackers = {}

def get_tracker(claim_id: str) -> ClaimFlowTracker:
    if claim_id not in _trackers:
        _trackers[claim_id] = ClaimFlowTracker(claim_id)
    return _trackers[claim_id]
