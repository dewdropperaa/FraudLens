import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from claimguard.v2.reliability import ReliabilityStore


def test_investigator_analytics_metrics_and_alerts() -> None:
    store = ReliabilityStore()
    store._try_firestore_client = lambda: None  # type: ignore[method-assign]
    for i in range(12):
        store.add_investigator_review(
            claim_id=f"c-{i}",
            investigator_id="investigator-a",
            system_decision="REJECTED" if i % 3 == 0 else "APPROVED",
            system_ts=75.0 if i % 2 == 0 else 52.0,
            system_risk_level="HIGH" if i % 2 == 0 else "MEDIUM",
            investigator_decision="APPROVED",
            review_time_seconds=5.0,
            notes="ok",
        )

    analytics = store.get_investigator_analytics()
    assert analytics["total_reviews"] == 12
    assert len(analytics["leaderboard"]) == 1
    row = analytics["leaderboard"][0]
    assert row["investigator_id"] == "investigator-a"
    assert row["metrics"]["agreement_rate"] < 0.7
    assert row["metrics"]["average_review_time"] <= 5.0
    assert row["alerts_count"] >= 2
    assert analytics["fairness_rules"]["do_not_auto_penalize_disagreement"] is True
