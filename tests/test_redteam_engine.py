from __future__ import annotations

from claimguard.v2.redteam import StrictModeConfig, run_red_teaming


def test_redteam_engine_report_shape_and_sections() -> None:
    report = run_red_teaming(
        claim_count=10,
        random_seed=7,
        generate_visualizations=False,
        strict_mode=StrictModeConfig(
            critical_failures_threshold=999,
            hallucination_rate_max=1.0,
            fraud_detection_target=0.0,
        ),
    )
    assert set(report.keys()) >= {
        "summary",
        "accuracy_metrics",
        "reasoning_metrics",
        "system_weaknesses",
        "agent_performance",
        "memory_impact",
        "graph_detection",
        "alerts",
        "performance_metrics",
        "recommendations",
        "strict_mode",
    }
    assert report["summary"]["total_tests"] == 10
    assert report["run_status"] in {"PASSED", "FAILED"}
    assert isinstance(report["sample_traces"], list) and report["sample_traces"]


def test_redteam_strict_mode_can_fail_run() -> None:
    report = run_red_teaming(
        claim_count=10,
        random_seed=11,
        generate_visualizations=False,
        strict_mode=StrictModeConfig(
            critical_failures_threshold=0,
            hallucination_rate_max=0.0,
            fraud_detection_target=1.0,
        ),
    )
    assert report["strict_mode"]["failed"] is True
    assert report["run_status"] == "FAILED"
