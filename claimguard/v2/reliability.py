from __future__ import annotations

import hashlib
import json
import os
import os
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def hash_payload(payload: Any) -> str:
    return hashlib.sha256(stable_dumps(payload).encode("utf-8")).hexdigest()


@dataclass
class ExternalValidationResult:
    ok: bool = True
    source: str = "none"
    reason: str = ""

    def to_flag(self) -> Optional[str]:
        if self.ok:
            return None
        reason = (self.reason or "mismatch").strip().replace(" ", "_").upper()
        return f"EXTERNAL_VALIDATION_FAILED:{self.source}:{reason}"


class ReliabilityStore:
    """
    Reliability state + persistence:
    - decision traces
    - replay packages
    - human feedback outcomes
    - rolling system metrics
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._trace_collection = os.getenv("RELIABILITY_TRACE_COLLECTION", "decision_traces")
        self._feedback_collection = os.getenv("RELIABILITY_FEEDBACK_COLLECTION", "human_feedback")
        self._investigator_reviews_collection = os.getenv(
            "RELIABILITY_INVESTIGATOR_REVIEWS_COLLECTION",
            "investigator_reviews",
        )
        self._metrics_window = int(os.getenv("RELIABILITY_METRICS_WINDOW", "200"))
        self._approval_rate_limit = float(os.getenv("RELIABILITY_APPROVAL_RATE_LIMIT", "0.85"))
        self._approvals_disable_until = ""
        self._recent_decisions: deque[str] = deque(maxlen=self._metrics_window)
        self._recent_ts: deque[float] = deque(maxlen=self._metrics_window)
        self._replay_registry: Dict[str, Dict[str, Any]] = {}
        self._feedback_index: Dict[str, Dict[str, Any]] = {}
        self._trace_cache: Dict[str, Dict[str, Any]] = {}
        self._investigator_reviews: List[Dict[str, Any]] = []
        self._feedback_events: List[Dict[str, Any]] = []
        self._reviewer_profiles: Dict[str, Dict[str, Any]] = {}

    def _try_firestore_client(self):
        try:
            import firebase_admin
            from firebase_admin import credentials, firestore
        except Exception:
            return None
        app_name = "claimguard-v2-reliability"
        try:
            app = firebase_admin.get_app(app_name)
        except ValueError:
            cred_json = (os.getenv("FIREBASE_CREDENTIALS_JSON") or "").strip()
            cred_path = (os.getenv("FIREBASE_CREDENTIALS_PATH") or "").strip()
            if cred_json:
                cred = credentials.Certificate(json.loads(cred_json))
            elif cred_path:
                cred = credentials.Certificate(cred_path)
            else:
                cred = credentials.ApplicationDefault()
            app = firebase_admin.initialize_app(cred, name=app_name)
        try:
            return firestore.client(app=app)
        except Exception:
            return None

    def persist_decision_trace(self, claim_id: str, trace: Dict[str, Any]) -> str:
        trace_hash = hash_payload(trace)
        payload = {
            "claim_id": claim_id,
            "trace_hash": trace_hash,
            "trace": trace,
            "timestamp": _utc_now_iso(),
        }
        with self._lock:
            self._trace_cache[claim_id] = payload
        client = self._try_firestore_client()
        if client is not None:
            timeout_s = float(os.getenv("FIRESTORE_WRITE_TIMEOUT_S", "8"))
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(
                client.collection(self._trace_collection).document(claim_id).set,
                payload,
                True,  # merge=True
            )
            try:
                future.result(timeout=timeout_s)
            except FuturesTimeoutError:
                executor.shutdown(wait=False, cancel_futures=True)
                raise RuntimeError(f"Firestore trace write timed out after {timeout_s}s")
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
        return trace_hash

    def get_trace(self, claim_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            cached = self._trace_cache.get(claim_id)
        if cached:
            return dict(cached)
        client = self._try_firestore_client()
        if client is None:
            return None
        snap = client.collection(self._trace_collection).document(claim_id).get()
        if not snap.exists:
            return None
        data = snap.to_dict() or {}
        with self._lock:
            self._trace_cache[claim_id] = data
        return data

    def register_replay_package(self, claim_id: str, package: Dict[str, Any]) -> None:
        with self._lock:
            self._replay_registry[claim_id] = dict(package)

    def get_replay_package(self, claim_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            item = self._replay_registry.get(claim_id)
        return dict(item) if item else None

    def push_decision_metrics(self, *, decision: str, ts: float) -> Dict[str, Any]:
        with self._lock:
            self._recent_decisions.append(decision)
            self._recent_ts.append(float(ts))
            n = len(self._recent_decisions)
            approvals = sum(1 for d in self._recent_decisions if d == "APPROVED")
            rejections = sum(1 for d in self._recent_decisions if d == "REJECTED")
            approval_rate = approvals / max(1, n)
            rejection_rate = rejections / max(1, n)
            avg_ts = (sum(self._recent_ts) / max(1, len(self._recent_ts))) if self._recent_ts else 0.0
            rate_limited = approval_rate > self._approval_rate_limit and n >= 10
            if rate_limited:
                self._approvals_disable_until = _utc_now_iso()
        return {
            "window": n,
            "approval_rate": round(approval_rate, 4),
            "rejection_rate": round(rejection_rate, 4),
            "average_ts": round(avg_ts, 4),
            "approved_disabled": rate_limited or bool(self._approvals_disable_until),
            "system_alert": rate_limited,
        }

    def is_auto_approve_disabled(self) -> bool:
        with self._lock:
            return bool(self._approvals_disable_until)

    @staticmethod
    def _parse_timestamp(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if isinstance(value, str) and value:
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except Exception:
                pass
        return datetime.now(timezone.utc)

    def _check_feedback_rate_limits(self, *, reviewer_id: str, now: datetime) -> tuple[bool, str]:
        cutoff_hour = now - timedelta(hours=1)
        cutoff_day = now - timedelta(days=1)
        hourly = 0
        daily = 0
        for event in self._feedback_events:
            if str(event.get("reviewer_id")) != reviewer_id:
                continue
            event_ts = self._parse_timestamp(event.get("timestamp"))
            if event_ts >= cutoff_hour:
                hourly += 1
            if event_ts >= cutoff_day:
                daily += 1
        if hourly >= 10:
            return False, "Hourly feedback limit reached"
        if daily >= 50:
            return False, "Daily feedback limit reached"
        return True, ""

    def _update_reviewer_trust_profile(self, *, reviewer_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        now_iso = str(payload.get("timestamp") or _utc_now_iso())
        with self._lock:
            profile = self._reviewer_profiles.get(
                reviewer_id,
                {
                    "reviewer_id": reviewer_id,
                    "reviewer_trust_score": 1.0,
                    "total_feedback": 0,
                    "reversal_count": 0,
                    "abnormal_pattern_count": 0,
                    "last_feedback_at": "",
                },
            )
            profile["total_feedback"] = int(profile.get("total_feedback", 0)) + 1
            profile["last_feedback_at"] = now_iso

            existing = self._feedback_index.get(f"{payload['cin']}::{payload['provider']}")
            if existing and existing.get("claim_id") == payload.get("claim_id"):
                existing_outcome = str(existing.get("outcome", "")).upper()
                if existing_outcome and existing_outcome != payload.get("outcome"):
                    profile["reversal_count"] = int(profile.get("reversal_count", 0)) + 1

            recent = [
                e
                for e in self._feedback_events
                if str(e.get("reviewer_id")) == reviewer_id
                and self._parse_timestamp(e.get("timestamp")) >= (self._parse_timestamp(now_iso) - timedelta(hours=1))
            ]
            reject_count = sum(1 for e in recent if str(e.get("outcome", "")).upper() == "REJECTED")
            if len(recent) >= 8 and reject_count / max(1, len(recent)) >= 0.95:
                profile["abnormal_pattern_count"] = int(profile.get("abnormal_pattern_count", 0)) + 1

            total_feedback = max(1, int(profile.get("total_feedback", 1)))
            reversal_penalty = min(0.4, int(profile.get("reversal_count", 0)) * 0.1)
            abnormal_penalty = min(0.4, int(profile.get("abnormal_pattern_count", 0)) * 0.08)
            volume_penalty = 0.0
            if total_feedback >= 20:
                volume_penalty = 0.05
            trust = max(0.0, min(1.0, 1.0 - reversal_penalty - abnormal_penalty - volume_penalty))
            profile["reviewer_trust_score"] = round(trust, 4)
            self._reviewer_profiles[reviewer_id] = profile
            return dict(profile)

    def add_human_feedback(
        self,
        *,
        claim_id: str,
        outcome: str,
        reviewer_id: str,
        cin: str,
        provider: str,
        ip_address: str = "",
    ) -> Dict[str, Any]:
        outcome_n = (outcome or "").strip().upper()
        now_iso = _utc_now_iso()
        payload = {
            "claim_id": claim_id,
            "outcome": outcome_n,
            "reviewer_id": reviewer_id,
            "cin": (cin or "").strip().upper(),
            "provider": (provider or "").strip().upper(),
            "decision": outcome_n,
            "timestamp": now_iso,
            "ip_address": str(ip_address or "").strip(),
        }
        allowed, limit_reason = self._check_feedback_rate_limits(
            reviewer_id=str(reviewer_id),
            now=self._parse_timestamp(now_iso),
        )
        if not allowed:
            raise ValueError(limit_reason)

        profile = self._update_reviewer_trust_profile(reviewer_id=str(reviewer_id), payload=payload)
        reviewer_trust_score = float(profile.get("reviewer_trust_score", 0.0))
        key = f"{payload['cin']}::{payload['provider']}"
        with self._lock:
            self._feedback_events.append(dict(payload))
            if reviewer_trust_score > 0.7:
                self._feedback_index[key] = payload
        client = self._try_firestore_client()
        if client is not None:
            doc_id = f"{claim_id}::{reviewer_id}::{int(datetime.now(timezone.utc).timestamp())}"
            client.collection(self._feedback_collection).document(doc_id).set(payload, merge=True)
            if reviewer_trust_score > 0.7:
                client.collection(self._feedback_collection).document(claim_id).set(payload, merge=True)
        return {
            **payload,
            "reviewer_trust_score": reviewer_trust_score,
            "memory_write_allowed": reviewer_trust_score > 0.7,
            "memory_write_reason": "accepted" if reviewer_trust_score > 0.7 else "trust_score_below_threshold",
        }

    def get_feedback_signal(self, *, cin: str, provider: str) -> Dict[str, Any]:
        key = f"{(cin or '').strip().upper()}::{(provider or '').strip().upper()}"
        with self._lock:
            payload = self._feedback_index.get(key)
        if payload:
            return dict(payload)
        return {}

    def add_investigator_review(
        self,
        *,
        claim_id: str,
        investigator_id: str,
        system_decision: str,
        system_ts: float,
        system_risk_level: str,
        investigator_decision: str,
        review_time_seconds: float,
        notes: str,
    ) -> Dict[str, Any]:
        payload = {
            "claim_id": str(claim_id).strip(),
            "investigator_id": str(investigator_id or "unknown").strip(),
            "system_decision": str(system_decision or "").strip().upper(),
            "system_Ts": round(float(system_ts or 0.0), 3),
            "system_risk_level": str(system_risk_level or "UNKNOWN").strip().upper(),
            "investigator_decision": str(investigator_decision or "").strip().upper(),
            "timestamp": _utc_now_iso(),
            "review_time_seconds": max(0.0, float(review_time_seconds or 0.0)),
            "notes": str(notes or "").strip(),
        }
        with self._lock:
            self._investigator_reviews.append(payload)
        client = self._try_firestore_client()
        if client is not None:
            doc_id = f"{payload['claim_id']}::{payload['timestamp']}"
            client.collection(self._investigator_reviews_collection).document(doc_id).set(payload, merge=True)
        return payload

    def _list_all_reviews(self) -> List[Dict[str, Any]]:
        with self._lock:
            local = [dict(item) for item in self._investigator_reviews]
        client = self._try_firestore_client()
        if client is None:
            return local
        try:
            docs = list(client.collection(self._investigator_reviews_collection).stream())
            remote = [d.to_dict() or {} for d in docs]
            return remote if remote else local
        except Exception:
            return local

    @staticmethod
    def _safe_ts(record: Dict[str, Any]) -> float:
        try:
            return float(record.get("system_Ts") or 0.0)
        except Exception:
            return 0.0

    @staticmethod
    def _parse_when(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if isinstance(value, str) and value:
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except Exception:
                pass
        return datetime.now(timezone.utc)

    @staticmethod
    def _variance(values: List[float]) -> float:
        if len(values) <= 1:
            return 0.0
        avg = sum(values) / len(values)
        return sum((v - avg) ** 2 for v in values) / len(values)

    @staticmethod
    def _decision_numeric(decision: str) -> float:
        dn = (decision or "").upper()
        if "REJECT" in dn:
            return 1.0
        if "APPROV" in dn:
            return 0.0
        return 0.5

    def get_investigator_analytics(self) -> Dict[str, Any]:
        records = self._list_all_reviews()
        by_investigator: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for rec in records:
            inv = str(rec.get("investigator_id") or "unknown").strip() or "unknown"
            by_investigator[inv].append(rec)

        leaderboard: List[Dict[str, Any]] = []
        all_alerts: List[Dict[str, Any]] = []

        for investigator_id, rows in by_investigator.items():
            total = len(rows)
            approvals = sum(1 for r in rows if r.get("investigator_decision") == "APPROVED")
            rejections = sum(1 for r in rows if r.get("investigator_decision") == "REJECTED")
            agreements = sum(
                1
                for r in rows
                if str(r.get("investigator_decision", "")).upper() == str(r.get("system_decision", "")).upper()
            )
            agreement_rate = agreements / total if total else 0.0

            high_risk_rows = [r for r in rows if str(r.get("system_risk_level", "")).upper() == "HIGH"]
            high_risk_rejected = sum(1 for r in high_risk_rows if str(r.get("investigator_decision", "")).upper() == "REJECTED")
            high_risk_detection_accuracy = (
                high_risk_rejected / len(high_risk_rows) if high_risk_rows else 0.0
            )

            note_scores: List[float] = []
            review_times: List[float] = []
            consistency_buckets: Dict[Tuple[str, str], List[float]] = defaultdict(list)
            borderline = 0
            escalated_borderline = 0
            false_approvals = 0
            weekly_agreement: Dict[str, List[float]] = defaultdict(list)
            decision_distribution = Counter()

            for rec in rows:
                decision = str(rec.get("investigator_decision", "")).upper()
                system_decision = str(rec.get("system_decision", "")).upper()
                risk = str(rec.get("system_risk_level", "")).upper()
                ts = self._safe_ts(rec)
                notes = str(rec.get("notes") or "").strip()
                rt = float(rec.get("review_time_seconds") or 0.0)
                review_times.append(rt)
                decision_distribution[decision or "UNKNOWN"] += 1

                justification = 0.0
                if notes:
                    justification += 0.5
                if len(notes) >= 40:
                    justification += 0.5
                note_scores.append(justification)

                key = (risk or "UNKNOWN", system_decision or "UNKNOWN")
                consistency_buckets[key].append(self._decision_numeric(decision))

                if (45.0 <= ts <= 70.0) or risk == "MEDIUM":
                    borderline += 1
                    if decision in {"REJECTED", "HUMAN_REVIEW"}:
                        escalated_borderline += 1

                if decision == "APPROVED" and risk == "HIGH":
                    false_approvals += 1

                when = self._parse_when(rec.get("timestamp"))
                week = f"{when.isocalendar().year}-W{when.isocalendar().week:02d}"
                weekly_agreement[week].append(1.0 if decision == system_decision else 0.0)

            bucket_variances = [self._variance(vals) for vals in consistency_buckets.values() if vals]
            consistency_variance = sum(bucket_variances) / len(bucket_variances) if bucket_variances else 0.0
            decision_consistency = max(0.0, min(1.0, 1.0 - consistency_variance))
            average_review_time = sum(review_times) / len(review_times) if review_times else 0.0
            override_justification_quality = sum(note_scores) / len(note_scores) if note_scores else 0.0
            risk_sensitivity_score = escalated_borderline / borderline if borderline else 0.0
            false_approval_rate = false_approvals / approvals if approvals else 0.0

            weekly_rates = [
                sum(vals) / len(vals) for _, vals in sorted(weekly_agreement.items(), key=lambda item: item[0])
            ]
            stability_score = max(0.0, min(1.0, 1.0 - self._variance(weekly_rates))) if weekly_rates else 1.0

            profile = {
                "investigator_id": investigator_id,
                "total_claims_reviewed": total,
                "approval_rejection_ratio": round(approvals / max(1, rejections), 4),
                "approval_count": approvals,
                "rejection_count": rejections,
                "agreement_with_system": round(agreement_rate, 4),
                "average_review_time": round(average_review_time, 2),
                "recent_decisions": sorted(
                    rows,
                    key=lambda r: str(r.get("timestamp", "")),
                    reverse=True,
                )[:10],
                "decision_distribution": dict(decision_distribution),
            }

            metrics = {
                "agreement_rate": round(agreement_rate, 4),
                "high_risk_detection_accuracy": round(high_risk_detection_accuracy, 4),
                "override_justification_quality": round(override_justification_quality, 4),
                "average_review_time": round(average_review_time, 2),
                "decision_consistency": round(decision_consistency, 4),
                "risk_sensitivity_score": round(risk_sensitivity_score, 4),
                "false_approval_rate": round(false_approval_rate, 4),
                "stability_score": round(stability_score, 4),
            }

            alerts: List[str] = []
            approval_rate = approvals / total if total else 0.0
            if total >= 10 and approval_rate > 0.85:
                alerts.append("Unusually high approval rate detected.")
            if total >= 10 and agreement_rate < 0.45:
                alerts.append("Very low agreement with system decisions.")
            if total >= 10 and average_review_time < 15:
                alerts.append("Extremely fast average review time detected.")

            leaderboard.append(
                {
                    "investigator_id": investigator_id,
                    "total_reviews": total,
                    "agreement_rate": round(agreement_rate, 4),
                    "high_risk_detection_accuracy": round(high_risk_detection_accuracy, 4),
                    "stability_score": round(stability_score, 4),
                    "risk_sensitivity_score": round(risk_sensitivity_score, 4),
                    "false_approval_rate": round(false_approval_rate, 4),
                    "alerts_count": len(alerts),
                    "profile": profile,
                    "metrics": metrics,
                    "alerts": alerts,
                }
            )
            for a in alerts:
                all_alerts.append({"investigator_id": investigator_id, "message": a})

        leaderboard.sort(key=lambda row: (row["agreement_rate"], row["stability_score"], row["total_reviews"]), reverse=True)
        return {
            "total_reviews": len(records),
            "leaderboard": leaderboard,
            "alerts": all_alerts,
            "fairness_rules": {
                "do_not_auto_penalize_disagreement": True,
                "purpose": "Pattern highlighting for coaching and quality improvement.",
            },
            "feedback_loop_signals": {
                "refine_system_confidence": True,
                "update_memory_signals": True,
            },
        }


_singleton: ReliabilityStore | None = None


def get_reliability_store() -> ReliabilityStore:
    global _singleton
    if _singleton is None:
        _singleton = ReliabilityStore()
    return _singleton

