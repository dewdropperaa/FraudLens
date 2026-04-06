import numpy as np
import pytest

from claimguard.agents.graph_agent import GraphAgent
from claimguard.services.graph_fraud import GraphFraudDetector


def test_graph_build_and_features_from_claims_csv():
    df = GraphFraudDetector.generate_synthetic_fraud_scenarios(
        normal_claims=80,
        collusion_ring_claims=40,
        burst_claims=40,
        random_state=3,
    )
    detector = GraphFraudDetector.from_dataframe(df)
    graph = detector.build_graph()
    features = detector.compute_graph_features()

    assert graph.number_of_nodes() > 0
    assert graph.number_of_edges() > 0
    assert {
        "degree_centrality",
        "clustering_coefficient",
        "shared_connections",
        "claim_frequency",
    }.issubset(set(features.columns))


def test_synthetic_fraud_detection_and_model_training():
    df = GraphFraudDetector.generate_synthetic_fraud_scenarios(
        normal_claims=120,
        collusion_ring_claims=70,
        burst_claims=70,
        random_state=12,
    )
    detector = GraphFraudDetector.from_dataframe(df)
    detector.build_graph()
    detector.compute_graph_features()
    patterns = detector.detect_fraud_patterns(
        provider_degree_quantile=0.95,
        provider_patient_min=10,
        burst_days=7,
        burst_claim_threshold=15,
        repeated_pair_min=3,
    )
    report = detector.train_model()

    assert len(patterns["high_degree_providers"]) > 0
    assert any(p["provider_id"] == "PR_BURST" for p in patterns["short_time_bursts"])
    assert report.accuracy >= 0.70


def test_graph_agent_output_contract():
    agent = GraphAgent()
    out = agent.analyze_graph_risk(
        {
            "patient_id": "99999999",
            "provider_id": "53",
            "claim_id": "integration_test_claim",
            "amount": 125000.0,
        }
    )
    assert isinstance(out["fraud_probability"], float)
    assert isinstance(out["pattern_detected"], str)
    assert isinstance(out["risk_nodes"], list)
    assert "claim::integration_test_claim" in out["risk_nodes"]
    assert "provider::53" in out["risk_nodes"]
    assert "patient::99999999" in out["risk_nodes"]


def test_random_forest_generalization_not_severely_overfit_or_underfit():
    """Train vs holdout gap flags extreme overfitting; holdout performance flags underfitting."""
    df = GraphFraudDetector.generate_synthetic_fraud_scenarios(
        normal_claims=200,
        collusion_ring_claims=90,
        burst_claims=90,
        random_state=21,
    )
    detector = GraphFraudDetector.from_dataframe(df)
    detector.build_graph()
    detector.compute_graph_features()
    report = detector.train_model(test_size=0.25, random_state=21)

    assert report.train_accuracy is not None and report.train_f1 is not None
    # Underfitting: model should beat random on held-out data (binary, balanced-ish labels).
    assert report.accuracy >= 0.65
    assert report.f1 >= 0.55
    # Severe overfitting: train memorization with poor generalization (large train/test gap).
    gap = report.train_accuracy - report.accuracy
    assert gap < 0.22, f"train-test accuracy gap too large (possible overfitting): {gap:.3f}"


def _rigorous_synthetic_detector(random_state: int = 44) -> GraphFraudDetector:
    """Enough rows per class for 5-fold stratified CV and learning curves."""
    df = GraphFraudDetector.generate_synthetic_fraud_scenarios(
        normal_claims=220,
        collusion_ring_claims=100,
        burst_claims=100,
        random_state=random_state,
    )
    detector = GraphFraudDetector.from_dataframe(df)
    detector.build_graph()
    detector.compute_graph_features()
    return detector


def test_stratified_kfold_cv_and_per_class_metrics():
    """k-fold CV: stable scores, per-class precision/recall, no severe underfitting."""
    detector = _rigorous_synthetic_detector(random_state=44)
    cv_report = detector.cross_validate_stratified(n_splits=5, random_state=99)

    assert cv_report.n_splits == 5
    assert cv_report.n_samples >= 400
    # Underfitting: macro metrics should clearly beat chance on this synthetic task.
    assert cv_report.mean_accuracy >= 0.58
    assert cv_report.mean_f1_macro >= 0.52
    assert cv_report.mean_precision_macro >= 0.55
    assert cv_report.mean_recall_macro >= 0.55
    # High variance across folds suggests instability (small data or bad split luck).
    assert cv_report.std_accuracy < 0.14
    assert cv_report.std_f1_macro < 0.14
    # Per-class monitoring (fraud vs non-fraud).
    for label in (0, 1):
        assert cv_report.precision_by_class[label] >= 0.48
        assert cv_report.recall_by_class[label] >= 0.48
        assert cv_report.f1_by_class[label] >= 0.48
    assert len(cv_report.fold_accuracy) == 5


def test_learning_curve_f1_macro_generalization():
    """Learning curve: validation score at full data and train–val gap (overfitting)."""
    detector = _rigorous_synthetic_detector(random_state=45)
    lc = detector.learning_curve_stratified(
        n_splits=5,
        random_state=101,
        train_sizes=np.linspace(0.25, 1.0, 5),
        scoring="f1_macro",
    )
    assert len(lc.train_sizes) >= 3
    assert len(lc.train_scores_mean) == len(lc.val_scores_mean)
    # With most training data, validation F1 should not collapse (underfitting).
    assert lc.val_scores_mean[-1] >= 0.52
    # Large gap at max training size → model fits noise (overfitting).
    assert lc.gap_at_max_training < 0.20, (
        f"train–val F1 gap at max training size too large: {lc.gap_at_max_training:.3f}"
    )


def test_score_claim_rejects_missing_provider_id():
    df = GraphFraudDetector.generate_synthetic_fraud_scenarios(
        normal_claims=40,
        collusion_ring_claims=20,
        burst_claims=20,
        random_state=7,
    )
    detector = GraphFraudDetector.from_dataframe(df)
    detector.build_graph()
    detector.compute_graph_features()
    detector.train_model()
    with pytest.raises(ValueError, match="provider_id"):
        detector.score_claim(
            {"patient_id": "P1", "claim_id": "C1", "amount": 500.0}
        )
