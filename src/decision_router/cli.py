import argparse
import json
from pathlib import Path

from .core import Policy, route
from .evaluation import run_benchmark
from .providers import HostedLLM, make_provider


def main():
    parser = argparse.ArgumentParser(
        description="Hosted and local decision routing with reproducible evaluation"
    )
    parser.add_argument("command", choices=["route", "benchmark"])
    parser.add_argument(
        "--provider",
        choices=["jev", "laya", "simulation", "llm"],
        default="laya",
        help="Decision provider (default: local Laya)",
    )
    parser.add_argument(
        "--checkpoint",
        choices=["english", "typed-decisions", "multilingual"],
        default="typed-decisions",
    )
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--revision", help="Immutable Hugging Face checkpoint SHA")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--agreement-threshold", type=float, default=0.60)
    parser.add_argument("--llm-fallback", action="store_true")
    parser.add_argument("--text")
    parser.add_argument("--dataset", type=Path, default=Path("data/tickets.jsonl"))
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    policy = Policy(args.threshold, args.agreement_threshold)
    if args.command == "route" and args.text is None:
        parser.error("route requires --text")
    if args.command == "benchmark" and args.llm_fallback:
        parser.error("Benchmark measures individual providers; --llm-fallback is for route only")
    try:
        provider = make_provider(args.provider, args.checkpoint, args.device, args.revision)
        if args.command == "route":
            fallback = HostedLLM() if args.llm_fallback else None
            result = route(args.text, provider, policy, fallback).to_dict()
        else:
            result = run_benchmark(provider, args.dataset, args.split, args.threshold)
    except Exception as exc:
        parser.exit(
            1, f"Could not run {args.command}: {type(exc).__name__}. Check setup in README.md.\n"
        )
    rendered = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
        print(f"Saved {args.output}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
