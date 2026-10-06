"""Produce a free, reproducible LinkedIn figure and extrapolation evidence."""

import json
from pathlib import Path

from laya_economics import PRICING, SECONDS_PER_MONTH, api_cost_per_request, scenario

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/laya-economics"


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads((ROOT / "results/laya-load-test.json").read_text())
    runs = [{k: v for k, v in run.items() if k != "records"} for run in report["runs"]]
    measured = []
    for device, workload in (("mps", "short"), ("mps", "long"), ("cpu", "short")):
        for clients in sorted(
            {r["clients"] for r in runs if r["device"] == device and r["workload"] == workload}
        ):
            group = [
                r
                for r in report["runs"]
                if (r["device"], r["workload"], r["clients"]) == (device, workload, clients)
            ]
            records = [record for run in group for record in run["records"]]
            elapsed = sum(r["elapsed_seconds"] for r in group)
            measured.append(
                {
                    "device": device,
                    "workload": workload,
                    "clients": clients,
                    "requests": len(records),
                    "errors": sum(r["errors"] for r in group),
                    "pooled_rps": sum(r["successful"] for r in group) / elapsed,
                    "repeat_p95_ms": [r["response_p95_ms"] for r in group],
                }
            )
    assumptions = {
        "billable_input_tokens": 500,
        "billable_output_tokens_including_reasoning": 50,
        "cache_hit_fraction": 0,
        "processing": "standard",
        "illustrative_node_monthly_usd": 50,
        "illustrative_node_rps": 25,
        "utilization": 0.6,
        "peak_factor": 1,
        "month_days": 30,
        "hosting_quote": False,
        "cloud_throughput_verified": False,
        "api_calls_made": 0,
        "cloud_resources_provisioned": 0,
    }
    scenarios = [
        {"fallback_fraction": f, **scenario(v, 50, 25, f)}
        for f in (1, 0.55, 0.1)
        for v in (1000, 10000, 100000, 1000000, 10000000, 100000000)
    ]
    summary = {
        **{k: v for k, v in report.items() if k not in ("runs", "quality_records")},
        "hardware_detail": "Apple M4 Pro, 24 GiB unified memory",
        "total_load_requests": sum(r["requests"] for r in runs),
        "total_runtime_errors": sum(r["errors"] for r in runs),
        "runs": runs,
        "pooled_measurements": measured,
        "quality_records": report["quality_records"],
        "pricing_verified_on": "2026-10-06",
        "pricing": PRICING,
        "assumptions": assumptions,
        "scenarios": scenarios,
        "requests_per_second_projections": [
            {"sustained_rps": rps, **scenario(rps * SECONDS_PER_MONTH, 50, 25, 0)}
            for rps in (0.1, 1, 10, 100, 1000)
        ],
        "first_node_break_even_at_10_percent_fallback": {
            model: 50 / (0.9 * api_cost_per_request(model)) for model in PRICING
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "evidence.json").write_text(json.dumps(summary, indent=2) + "\n")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
        }
    )
    fig = plt.figure(figsize=(12, 12), facecolor="#FAFAF7")
    fig.text(0.07, 0.945, "LLM vs decision model: cost at scale", fontsize=25, weight="bold")
    fig.text(
        0.07,
        0.91,
        "Every request makes one decision. Traffic runs continuously for 30 days.",
        fontsize=13,
    )
    fig.text(
        0.07, 0.86, "Laya measured: ~25–26 requests/s short  |  ~6 requests/s long", fontsize=14
    )
    fig.text(0.07, 0.83, "M4 Pro, 24 GiB  ·  3,040 requests  ·  0 runtime errors", fontsize=13)
    ax = fig.add_axes([0.13, 0.41, 0.80, 0.34])
    traffic = [0.1, 1, 10, 100, 1000]
    projections = [scenario(r * SECONDS_PER_MONTH, 50, 25, 0) for r in traffic]
    ax.set_facecolor("#FAFAF7")
    for key, name, color, dash in (
        ("sonnet", "Claude Sonnet 5.5 only", "#838A91", "-"),
        ("luna", "GPT-6 Luna only", "#343C44", "--"),
        (None, "Laya only: illustrative hosting", "#226D68", "-"),
    ):
        costs = [s[key]["llm_only"] if key else s["laya_compute"] for s in projections]
        ax.plot(traffic, costs, label=name, color=color, linestyle=dash, marker="o", linewidth=2.4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(traffic, ["0.1", "1", "10", "100", "1,000"])
    ax.set_xlabel("Sustained requests per second (log scale)", labelpad=10)
    ax.set_ylabel("Monthly inference cost, USD (log scale)", labelpad=8)
    ax.grid(True, axis="y", which="major", alpha=0.15)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    fig.text(
        0.07, 0.34, "At 10 requests/s: 25.92 million decisions/month", fontsize=16, weight="bold"
    )
    fig.text(
        0.07,
        0.30,
        "Sonnet: $38,880   ·   Luna: $1,944   ·   Laya: $50*",
        fontsize=18,
        weight="bold",
    )
    fig.text(
        0.07, 0.25, "*Assumed $50/node/month, 25 requests/s/node at 60% utilization.", fontsize=11
    )
    fig.text(
        0.07,
        0.228,
        "Cloud cost, throughput and linear replica scaling have not been validated.",
        fontsize=11,
    )
    fig.text(
        0.07,
        0.18,
        "This compares inference spend, not equivalent decision quality.",
        fontsize=12,
        weight="bold",
    )
    fig.text(
        0.07,
        0.157,
        "Laya: 16/20 top-label accuracy on synthetic dev tickets; 0/20 auto-routed at 0.85.",
        fontsize=10,
    )
    fig.text(
        0.07,
        0.135,
        "Laya-only assumes every label is accepted. LLM accuracy/latency were not tested.",
        fontsize=10,
    )
    fig.text(
        0.07,
        0.09,
        "API assumptions: 500 input + 50 total billed output tokens; standard, uncached.",
        fontsize=10,
    )
    fig.text(
        0.07,
        0.068,
        "No paid API calls or cloud resources. Excludes labor, review, energy/capex, tools and retries.",
        fontsize=10,
    )
    fig.text(
        0.07,
        0.046,
        "Sources: official Claude/OpenAI pricing checked 6 Oct 2026; local benchmark. Links in post.",
        fontsize=10,
    )
    fig.savefig(OUT / "linkedin-economics.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {OUT / 'evidence.json'} and LinkedIn figure")


if __name__ == "__main__":
    main()
