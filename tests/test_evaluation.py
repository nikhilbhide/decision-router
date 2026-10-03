import json
from dataclasses import asdict
from pathlib import Path

import pytest

from decision_router.core import LABELS, Prediction
from decision_router.evaluation import (
    calibration_metrics,
    load_dataset,
    policy_metrics,
    run_benchmark,
)
from decision_router.providers import Simulation


def record(label, predicted, probability):
    probs = dict.fromkeys(LABELS, (1 - probability) / 4)
    probs[predicted] = probability
    return {"label": label, "prediction": asdict(Prediction(predicted, "test", 10, probs))}


def test_coverage_counts_failures_and_out_of_scope_predictions():
    rows = [
        record("billing", "billing", 0.9),
        record("account", "billing", 0.9),
        record("other", "other", 1),
        {"label": "sales", "prediction": None},
    ]
    summary = policy_metrics(rows, 0.85)
    assert summary["automation_coverage"] == 0.5
    assert summary["selective_accuracy"] == 0.5
    assert summary["classification_accuracy"] == 0.5
    assert summary["auto_routing_errors"] == 1
    assert summary["failed_requests"] == 1


def test_zero_accepted_cases_does_not_report_perfect_accuracy():
    summary = policy_metrics([record("billing", "billing", 0.7)], 0.9)
    assert summary["selective_accuracy"] is None
    assert summary["review_rate"] == 1


def test_perfect_predictions_have_zero_brier_and_ece():
    summary = calibration_metrics(
        [record("billing", "billing", 1), record("account", "account", 1)]
    )
    assert summary["brier_score"] == 0
    assert summary["ece"] == 0


def test_bad_calibration_is_measured_not_assumed():
    summary = calibration_metrics([record("account", "billing", 1)])
    assert summary["brier_score"] == 2
    assert summary["ece"] == 1


def test_dataset_group_leakage_is_rejected(tmp_path):
    path = tmp_path / "tickets.jsonl"
    rows = [
        {"id": "1", "group": "same", "split": "dev", "label": "billing", "text": "one"},
        {"id": "2", "group": "same", "split": "test", "label": "billing", "text": "two"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="groups"):
        load_dataset(path, "dev")


def test_bundled_benchmark_is_reproducible_and_marks_simulation():
    path = Path(__file__).parents[1] / "data/tickets.jsonl"
    report = run_benchmark(Simulation(), path, "dev")
    assert report["simulation"] is True
    assert report["split"] == "dev"
    assert len(report["records"]) == 20
    assert len(report["dataset_sha256"]) == 64
    assert len(report["threshold_sweep"]) == 21
    json.dumps(report, allow_nan=False)
