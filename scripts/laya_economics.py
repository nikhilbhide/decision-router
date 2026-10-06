"""Transparent inference-cost scenarios; model quality and human review are separate."""

import math

PRICING = {
    "sonnet": {
        "model": "claude-sonnet-5-5",
        "input_per_million": 2.0,
        "cached_input_per_million": 0.2,
        "output_per_million": 10.0,
        "source": "https://platform.claude.com/docs/en/models/sonnet-5-5/overview",
    },
    "luna": {
        "model": "gpt-6-luna",
        "input_per_million": 0.1,
        "cached_input_per_million": 0.01,
        "output_per_million": 0.5,
        "source": "https://developers.openai.com/api/docs/models/gpt-6-luna",
    },
}
SECONDS_PER_MONTH = 30 * 24 * 60 * 60


def api_cost_per_request(model, input_tokens=500, output_tokens=50, cached_fraction=0, batch=False):
    values = (input_tokens, output_tokens, cached_fraction)
    if any(not math.isfinite(v) for v in values):
        raise ValueError("Inputs must be finite")
    if min(input_tokens, output_tokens) < 0 or not 0 <= cached_fraction <= 1:
        raise ValueError("Token counts must be nonnegative and cache fraction in [0, 1]")
    price = PRICING[model]
    input_rate = (1 - cached_fraction) * price["input_per_million"]
    input_rate += cached_fraction * price["cached_input_per_million"]
    return (
        (input_tokens * input_rate + output_tokens * price["output_per_million"])
        / 1e6
        * (0.5 if batch else 1)
    )


def scenario(
    volume, node_cost, node_rps, fallback_fraction, utilization=0.6, peak_factor=1, overhead=0
):
    values = (volume, node_cost, node_rps, fallback_fraction, utilization, peak_factor, overhead)
    if any(not math.isfinite(v) for v in values):
        raise ValueError("Scenario values must be finite")
    if min(volume, node_cost, overhead) < 0 or node_rps <= 0 or peak_factor < 1:
        raise ValueError("Invalid volume, cost, throughput, or peak factor")
    if not 0 <= fallback_fraction <= 1 or not 0 < utilization <= 1:
        raise ValueError("Invalid fallback fraction or utilization")
    capacity = node_rps * utilization * SECONDS_PER_MONTH / peak_factor
    nodes = max(1, math.ceil(volume / capacity))
    hosting = nodes * node_cost + overhead
    result = {
        "volume": volume,
        "nodes": nodes,
        "capacity_per_node": capacity,
        "laya_compute": hosting,
    }
    for model in PRICING:
        api = volume * api_cost_per_request(model)
        result[model] = {
            "llm_only": api,
            "cascade": hosting + fallback_fraction * api,
            "cascade_savings": api - hosting - fallback_fraction * api,
        }
    return result
