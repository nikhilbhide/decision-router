"""Frozen predictions let us sweep policy thresholds without rerunning models."""

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from .core import LABELS


def load_dataset(path: Path, split: str):
    all_rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [row["id"] for row in all_rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Dataset ticket IDs must be unique")
    groups = {}
    for row in all_rows:
        if (
            row["label"] not in LABELS
            or not isinstance(row["text"], str)
            or not row["text"].strip()
        ):
            raise ValueError("Every ticket needs text and a supported label")
        if row["split"] not in {"dev", "test"}:
            raise ValueError("Dataset split must be dev or test")
        group = row.get("group", row["id"])
        if group in groups and groups[group] != row["split"]:
            raise ValueError("Related ticket groups cannot cross development and test splits")
        groups[group] = row["split"]
    rows = [row for row in all_rows if row["split"] == split]
    if not rows:
        raise ValueError("Dataset has no rows in the selected split")
    return rows


def percentile(values, fraction):
    ordered = sorted(values)
    pos = (len(ordered) - 1) * fraction
    left = math.floor(pos)
    right = math.ceil(pos)
    return ordered[left] + (ordered[right] - ordered[left]) * (pos - left)


def policy_metrics(records, threshold):
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and between zero and one")
    if not records:
        raise ValueError("No predictions to evaluate")
    successful = [row for row in records if row.get("prediction") is not None]
    has_probabilities = any(row["prediction"]["probabilities"] is not None for row in successful)
    accepted = [
        row
        for row in successful
        if (
            row["prediction"]["probabilities"] is not None
            and row["prediction"]["label"] != "other"
            and row["prediction"]["probabilities"][row["prediction"]["label"]] >= threshold
        )
    ]
    errors = sum(row["label"] != row["prediction"]["label"] for row in accepted)
    correct = sum(row["label"] == row["prediction"]["label"] for row in successful)
    latency = [row["prediction"]["latency_ms"] for row in successful]
    return {
        "threshold": threshold,
        "tickets": len(records),
        "failed_requests": len(records) - len(successful),
        "classification_accuracy": correct / len(records),
        "auto_routed": len(accepted) if has_probabilities else None,
        "auto_routing_errors": errors if has_probabilities else None,
        "automation_coverage": len(accepted) / len(records) if has_probabilities else None,
        "review_rate": 1 - len(accepted) / len(records) if has_probabilities else None,
        "selective_accuracy": 1 - errors / len(accepted) if accepted else None,
        "p50_latency_ms": percentile(latency, 0.5) if latency else None,
        "p95_latency_ms": percentile(latency, 0.95) if latency else None,
    }


def calibration_metrics(records):
    rows = [row for row in records if row.get("prediction") and row["prediction"]["probabilities"]]
    if not rows:
        return {"brier_score": None, "ece": None, "reliability_bins": []}
    brier = sum(
        sum(
            (p - int(label == row["label"])) ** 2
            for label, p in row["prediction"]["probabilities"].items()
        )
        for row in rows
    ) / len(rows)
    bins = []
    for index in range(10):
        selected = [
            row
            for row in rows
            if min(9, int(max(row["prediction"]["probabilities"].values()) * 10)) == index
        ]
        if not selected:
            continue
        confidence = sum(
            max(row["prediction"]["probabilities"].values()) for row in selected
        ) / len(selected)
        accuracy = sum(row["label"] == row["prediction"]["label"] for row in selected) / len(
            selected
        )
        bins.append(
            {
                "bin": index,
                "count": len(selected),
                "mean_top_probability": confidence,
                "accuracy": accuracy,
            }
        )
    ece = sum(b["count"] / len(rows) * abs(b["mean_top_probability"] - b["accuracy"]) for b in bins)
    return {"brier_score": brier, "ece": ece, "reliability_bins": bins}


def run_benchmark(provider, path, split, threshold=0.85):
    rows = load_dataset(path, split)
    warmup_start = perf_counter()
    provider.predict(rows[0]["text"])
    warmup_ms = (perf_counter() - warmup_start) * 1000
    records = []
    for row in rows:
        record = dict(row)
        try:
            record["prediction"] = asdict(provider.predict(row["text"]))
            record["error"] = None
        except Exception as exc:
            record["prediction"] = None
            record["error"] = type(exc).__name__
        records.append(record)
    predictions = [row["prediction"] for row in records if row["prediction"]]
    is_simulation = any(p["provider"] == "simulation" for p in predictions)
    return {
        "schema_version": 1,
        "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "dataset_name": path.name,
        "split": split,
        "simulation": is_simulation,
        "note": "Synthetic smoke dataset; not a production benchmark. Simulation scores are invented."
        if is_simulation
        else "Synthetic smoke dataset; not a production benchmark.",
        "warmup_ms": warmup_ms,
        "latency_scope": "Successful warm inference calls, excluding model loading and warmup",
        "token_usage": {
            "input_tokens": sum(p.get("input_tokens") or 0 for p in predictions),
            "output_tokens": sum(p.get("output_tokens") or 0 for p in predictions),
            "note": "Excludes warmup. Check provider billing separately; local compute is not free.",
        },
        "summary": policy_metrics(records, threshold),
        "calibration": calibration_metrics(records),
        "threshold_sweep": [policy_metrics(records, i / 20) for i in range(21)],
        "records": records,
    }
