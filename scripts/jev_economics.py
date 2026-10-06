"""Offline projections using published prices; never calls a model API."""

import json
import math
from pathlib import Path

SECONDS_PER_MONTH = 30 * 24 * 3600
PRICES = {
    "jev": {
        "name": "TypeSafe Jev 1.13",
        "input": 0.042,
        "output": 0,
        "source": "https://docs.typesafe.ai/models",
    },
    "luna": {
        "name": "GPT-6 Luna",
        "input": 0.1,
        "output": 0.5,
        "source": "https://developers.openai.com/api/docs/models/gpt-6-luna",
    },
    "sonnet": {
        "name": "Claude Sonnet 5.5",
        "input": 2,
        "output": 10,
        "source": "https://platform.claude.com/docs/en/models/sonnet-5-5/overview",
    },
}


def project(rps, jev_input=500, llm_input=500, llm_output=50, hours_per_day=24):
    args = (rps, jev_input, llm_input, llm_output, hours_per_day)
    if any(not math.isfinite(v) or v < 0 for v in args) or hours_per_day > 24:
        raise ValueError("Values must be finite/nonnegative and hours/day at most 24")
    requests = rps * 30 * hours_per_day * 3600
    costs = {
        model: requests
        * ((jev_input if model == "jev" else llm_input) * p["input"] + llm_output * p["output"])
        / 1e6
        for model, p in PRICES.items()
    }
    return {
        "active_requests_per_second": rps,
        "requests_per_30_day_month": requests,
        "monthly_usd": costs,
        "jev_tokens_per_second": rps * jev_input,
        "within_published_jev_limits": rps <= 80 and rps * jev_input <= 100000,
    }


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(__file__).resolve().parents[1] / "docs/jev-economics"
    out.mkdir(parents=True, exist_ok=True)
    traffic = [0.1, 1, 10, 50, 80, 100, 1000]
    projections = [project(r) for r in traffic]
    report = {
        "pricing_checked_on": "2026-10-06",
        "method": "Price-only extrapolation, no paid API calls or hosted load test",
        "prices_usd_per_million_tokens": PRICES,
        "assumptions": {
            "jev_billable_input_tokens": 500,
            "llm_billable_input_tokens": 500,
            "llm_total_billable_output_tokens_including_reasoning": 50,
            "decisions_per_request": 1,
            "hours_per_day": 24,
            "days_per_month": 30,
            "api_mode": "standard_uncached",
            "fallback_fraction": 0,
            "hosted_requests_made": 0,
        },
        "jev_published_limits": {
            "requests_per_second": 80,
            "tokens_per_second": 100000,
            "source": PRICES["jev"]["source"],
            "note": "Dynamically changing vendor limits; not measured capacity or latency",
        },
        "projections": projections,
    }
    (out / "evidence.json").write_text(json.dumps(report, indent=2) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "text.parse_math": False})
    fig = plt.figure(figsize=(12, 12), facecolor="#FAFAF7")
    fig.text(0.07, 0.945, "Decision APIs: cost versus requests/second", fontsize=23, weight="bold")
    fig.text(0.07, 0.91, "TypeSafe Jev vs GPT-6 Luna vs Claude Sonnet 5.5", fontsize=15)
    fig.text(
        0.07, 0.865, "Published prices + explicit token budgets. No paid API calls.", fontsize=13
    )
    ax = fig.add_axes([0.13, 0.44, 0.80, 0.34])
    ax.set_facecolor("#FAFAF7")
    for model, color, dash in (
        ("sonnet", "#838A91", "-"),
        ("luna", "#343C44", "--"),
        ("jev", "#226D68", "-"),
    ):
        ax.plot(
            traffic,
            [p["monthly_usd"][model] for p in projections],
            label=PRICES[model]["name"],
            color=color,
            linestyle=dash,
            marker="o",
            linewidth=2.4,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([0.1, 1, 10, 100, 1000], ["0.1", "1", "10", "100", "1,000"])
    ax.set_xlabel("Sustained requests per second (log scale)", labelpad=10)
    ax.set_ylabel("Projected monthly API cost, USD (log scale)", labelpad=8)
    ax.grid(True, axis="y", which="major", alpha=0.15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(80, color="#838A91", linestyle=":", linewidth=1)
    ax.text(80, 4e6, "80 RPS published Jev limit", ha="right", fontsize=9)
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    fig.text(
        0.07,
        0.37,
        "At 10 requests/s: 25.92 million calls per 30-day month",
        fontsize=15,
        weight="bold",
    )
    fig.text(
        0.07, 0.325, "Jev $544.32  ·  Luna $1,944  ·  Sonnet $38,880", fontsize=18, weight="bold"
    )
    fig.text(
        0.07, 0.28, "Per million decisions: Jev $21  ·  Luna $75  ·  Sonnet $1,500", fontsize=13
    )
    fig.text(
        0.07,
        0.235,
        "Assumptions: 500 billable input tokens/model; LLM output 50 total tokens.",
        fontsize=11,
    )
    fig.text(
        0.07,
        0.213,
        "One decision/call, 24/7, standard uncached rates; Jev output is free.",
        fontsize=11,
    )
    fig.text(
        0.07,
        0.169,
        "Price projection, not a measured throughput or quality benchmark.",
        fontsize=12,
        weight="bold",
    )
    fig.text(
        0.07,
        0.145,
        "Above 80 RPS requires higher Jev limits; enterprise rates may differ.",
        fontsize=11,
    )
    fig.text(
        0.07,
        0.123,
        "100+ RPS points are arithmetic only. Provider tokenizers/prompt sizes can differ.",
        fontsize=10,
    )
    fig.text(
        0.07,
        0.079,
        "Prices: Jev input $0.042/M, output $0; Luna $0.10/$0.50; Sonnet $2/$10.",
        fontsize=10,
    )
    fig.text(
        0.07,
        0.057,
        "Sources: docs.typesafe.ai/models; official OpenAI & Claude model docs. Checked 6 Oct 2026.",
        fontsize=9,
    )
    fig.text(
        0.07,
        0.035,
        "Excludes fallback, retries, tools, caching, taxes and review. Output budget includes billed reasoning.",
        fontsize=9,
    )
    fig.savefig(out / "linkedin-economics.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote Jev projections and figure to {out}")


if __name__ == "__main__":
    main()
