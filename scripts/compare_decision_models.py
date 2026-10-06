"""Run one local decision model against the shared development dataset.

Use separate environments for Laya (Transformers 4) and OpenDecider (Transformers 5).
The model is selected before reading any expected labels; prompts are shared unchanged.
"""

import argparse
import importlib.metadata
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from decision_router.core import QUESTIONS, Prediction
from decision_router.evaluation import run_benchmark
from decision_router.providers import DEFAULT_REVISION, Laya

NANO_REVISION = "10c4b349f63d28186a46eacb6478cfce29c488e8"


class OpenDeciderNano:
    def __init__(self, device, revision=NANO_REVISION):
        from opendecider import __version__, load

        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Use an immutable Hugging Face revision SHA")
        self.agent = load(
            "manjunathshiva/opendecider-nano",
            device=device,
            revision=revision,
            dtype="float32",
        )
        self.revision = revision
        self.sdk_version = __version__

    def predict(self, text):
        start = perf_counter()
        result = self.agent.system_one(text, QUESTIONS)
        elapsed = (perf_counter() - start) * 1000
        answer = result["answers"]["department"]
        if answer.get("truncated") or result.get("warnings"):
            raise ValueError("OpenDecider truncated the input or returned a warning")
        if answer.get("type") != "choice":
            raise ValueError("OpenDecider returned an unexpected answer type")
        usage = result.get("usage", {})
        return Prediction(
            label=answer["choice"],
            provider="opendecider-nano",
            latency_ms=elapsed,
            probabilities=answer["probabilities"],
            model=f"manjunathshiva/opendecider-nano / opendecider {self.sdk_version}",
            revision=self.revision,
            sdk_confidence=answer.get("confidence"),
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["laya", "opendecider-nano"])
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    parser.add_argument("--dataset", type=Path, default=Path("data/tickets.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(f"Loading {args.model} on {args.device} (float32)", flush=True)
    start = perf_counter()
    provider = (
        Laya(checkpoint="typed-decisions", device=args.device, revision=DEFAULT_REVISION)
        if args.model == "laya"
        else OpenDeciderNano(device=args.device)
    )
    load_ms = (perf_counter() - start) * 1000
    actual_device = (
        str(provider.agent.device) if args.model == "laya" else provider.agent.impl.device
    )
    if actual_device != args.device:
        raise RuntimeError("Model fell back to a different device; comparison would be misleading")
    print("Running one warmup, then 20 development predictions", flush=True)
    report = run_benchmark(provider, args.dataset, "dev", threshold=0.85)
    report["runtime"] = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": args.device,
        "dtype": "float32",
        "model_load_ms": load_ms,
        "model_load_scope": "Includes download if not already cached",
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "huggingface_hub")
        },
        "question": QUESTIONS,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
