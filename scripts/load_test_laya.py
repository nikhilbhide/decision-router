"""Closed-loop load test of a FIFO, single-worker local Laya service.

Measures the application's real adapter, including preparation and validation.
No HTTP, hosted models, batching, or paid API requests are involved.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import queue
import resource
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from decision_router.core import QUESTIONS
from decision_router.evaluation import load_dataset, percentile, policy_metrics
from decision_router.providers import DEFAULT_REVISION, Laya


class Worker:
    def __init__(self, provider):
        self.provider = provider
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def run(self):
        while True:
            task = self.queue.get()
            if task is None:
                return
            text, submitted, future = task
            started = perf_counter()
            try:
                prediction = asdict(self.provider.predict(text))
                error = None
            except Exception as exc:
                prediction, error = None, type(exc).__name__
            ended = perf_counter()
            future.set_result(
                {
                    "response_ms": (ended - submitted) * 1000,
                    "queue_ms": (started - submitted) * 1000,
                    "service_ms": (ended - started) * 1000,
                    "prediction": prediction,
                    "error": error,
                }
            )

    def request(self, text):
        future = Future()
        self.queue.put((text, perf_counter(), future))
        return future.result()

    def close(self):
        self.queue.put(None)
        self.thread.join()


def full_token_count(provider, text):
    from laya.common import build_sequence

    question = provider.agent._to_internal(QUESTIONS["department"])
    ids, _ = build_sequence(
        provider.agent.tok, text, question, 1_000_000, provider.agent.cfg["head_max_len"]
    )
    return len(ids)


def long_tickets(provider, rows, target=650):
    texts = []
    filler = " Historical record: this note supplies background context for the support team."
    for row in rows:
        text = row["text"]
        while full_token_count(provider, text) < target:
            text += filler
        if full_token_count(provider, text) > provider.agent.cfg["max_len"]:
            raise ValueError("Long workload exceeds checkpoint budget")
        texts.append(text)
    return texts


def summary(records, elapsed):
    good = [r for r in records if r["prediction"] is not None]

    def values(field):
        return [r[field] for r in records]

    return {
        "requests": len(records),
        "successful": len(good),
        "errors": len(records) - len(good),
        "elapsed_seconds": elapsed,
        "successful_rps": len(good) / elapsed,
        "response_p50_ms": percentile(values("response_ms"), 0.5),
        "response_p95_ms": percentile(values("response_ms"), 0.95),
        "response_p99_ms": percentile(values("response_ms"), 0.99),
        "queue_p95_ms": percentile(values("queue_ms"), 0.95),
        "service_p50_ms": percentile(values("service_ms"), 0.5),
        "service_p95_ms": percentile(values("service_ms"), 0.95),
    }


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/tickets.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/laya-load-test.json"))
    parser.add_argument("--short-requests", type=int, default=200)
    parser.add_argument("--long-requests", type=int, default=80)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if min(args.short_requests, args.long_requests, args.repeats) < 1:
        parser.error("Request counts and repeats must be positive")
    torch.set_num_threads(4)
    rows = load_dataset(args.dataset, "dev")
    report = {
        "schema_version": 1,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": {"platform": platform.platform(), "machine": platform.machine()},
        "packages": {
            n: importlib.metadata.version(n)
            for n in ("torch", "laya", "transformers", "huggingface_hub")
        },
        "revision": DEFAULT_REVISION,
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "method": {
            "arrival_model": "Closed loop: each client waits for its response before next request",
            "service": "FIFO single inference worker, one resident model per device",
            "latency": "Response includes queue and adapter; service includes prepare/infer/validate",
            "excluded": "Model loading, warmup, HTTP, database, hosted fallback, dynamic batching",
            "threads": 4,
            "quality_scope": "20 unique synthetic development tickets; repeats add no quality evidence",
            "long_workload": "Neutral filler extends input to about 650 Laya tokens; speed only",
        },
        "runs": [],
    }
    for device in ("mps", "cpu"):
        loaded = perf_counter()
        provider = Laya(device=device)
        if str(provider.agent.device) != device:
            raise RuntimeError("Laya silently fell back to another device")
        print(f"Loaded {device} in {perf_counter() - loaded:.2f}s", flush=True)
        workloads = {"short": [r["text"] for r in rows]}
        if device == "mps":
            workloads["long"] = long_tickets(provider, rows)
        worker = Worker(provider)
        try:
            if device == "mps":
                quality = []
                for row in rows:
                    result = worker.request(row["text"])
                    quality.append(
                        {**row, "prediction": result["prediction"], "error": result["error"]}
                    )
                report["quality_records"] = quality
                report["policy_sweep"] = [
                    policy_metrics(quality, threshold)
                    for threshold in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9)
                ]
            for workload, texts in workloads.items():
                token_counts = [full_token_count(provider, text) for text in texts]
                for text in texts[:5]:
                    if worker.request(text)["error"]:
                        raise RuntimeError("Warmup failed")
                clients_list = (1, 4, 8, 16) if device == "mps" else (1, 8)
                count = args.short_requests if workload == "short" else args.long_requests
                for repetition in range(args.repeats):
                    # Reverse order in repeat two to reduce order/thermal bias.
                    order = clients_list if repetition % 2 == 0 else clients_list[::-1]
                    for clients in order:
                        with ThreadPoolExecutor(max_workers=clients) as executor:
                            start = perf_counter()
                            records = list(
                                executor.map(
                                    worker.request, [texts[i % len(texts)] for i in range(count)]
                                )
                            )
                            elapsed = perf_counter() - start
                        run = {
                            "device": device,
                            "workload": workload,
                            "clients": clients,
                            "repeat": repetition + 1,
                            "input_tokens_min": min(token_counts),
                            "input_tokens_mean": sum(token_counts) / len(token_counts),
                            "input_tokens_max": max(token_counts),
                            **summary(records, elapsed),
                            "records": records,
                        }
                        report["runs"].append(run)
                        print(
                            json.dumps({k: v for k, v in run.items() if k != "records"}), flush=True
                        )
                        args.output.parent.mkdir(parents=True, exist_ok=True)
                        args.output.write_text(json.dumps(report, indent=2) + "\n")
        finally:
            worker.close()
        report.setdefault("memory", {})[device] = {
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "note": "Process cumulative peak on macOS, not per-request or isolated per-device",
        }
        if device == "mps":
            report["memory"][device]["mps_allocated_bytes"] = torch.mps.current_allocated_memory()
        del worker, provider
        if device == "mps":
            torch.mps.empty_cache()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
