# Laya load test and inference-cost projections

This study makes zero paid model API calls and provisions zero cloud resources. It measures local Laya performance, then applies public token prices to assumed API workloads. It does not benchmark LLM latency, throughput or decision quality.

## Reproduce

Run from the decision-router repository with the existing Laya environment and cached checkpoint:

```sh
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 PYTHONPATH=src .venv/bin/python scripts/load_test_laya.py
models/.venv-security/bin/python scripts/report_laya_economics.py
.venv/bin/python -m pytest tests/test_load_economics.py
```

The report script needs matplotlib, installed separately in the ignored analysis/security environment; it makes no network requests. Raw load data is in ignored `results/laya-load-test.json`. The portable `evidence.json` includes run summaries, quality predictions, package versions, checkpoint/dataset hashes, assumptions and projections. The generated PNG is suitable for attaching to a LinkedIn post. Nothing is posted automatically.

## Local measurement

Hardware: Apple M4 Pro, 24 GiB unified memory, macOS 15.7.7. Laya 0.3.4, PyTorch 2.14.1, Transformers 4.57.6. Typed-decisions checkpoint revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`.

One resident model with one FIFO inference worker. Clients issue their next request only after receiving a response (closed loop). Response time includes queueing and the adapter. Service time includes request preparation, model inference and prediction validation. Four CPU threads. No HTTP, database, remote fallback or dynamic batching. Model loading and warmup are excluded. Two repeats with reversed client order in the second repeat.

MPS: 1/4/8/16 concurrent clients, 200 short or 80 long requests per repeat. CPU: 1/8 clients, 200 short requests per repeat. Total: 3,040 timed requests, zero runtime errors. Another 20 unique development predictions and warmups are excluded from that request total.

Short inputs contain 121–132 complete Laya input tokens (mean 125.85). Neutral filler extends long inputs to 652–662 tokens (mean 658.2); this workload tests speed only. These counts include Laya's prepared input and do not represent a provider's billable API token count.

Pooled throughput is total successful requests divided by total elapsed time across repeats. MPS short: 25.60/25.54/25.90/20.84 requests/s at 1/4/8/16 clients. MPS long: 6.04/6.03/6.07/5.74 requests/s. CPU short: 7.71/9.13 requests/s at 1/8 clients.

Short MPS response p95 at one client: 42.78–44.93 ms; at 16 clients: 1,216.66–1,637.24 ms. Long MPS one-client p95: 171.18–177.60 ms; at 16 clients: 2,614.54–3,730.49 ms. Ranges are between repeat p95s, not confidence intervals. Timing spikes are retained.

This is a brief microbenchmark. It does not establish a sustained production SLO, open-loop capacity, cloud performance, horizontal scaling efficiency or API provider rate-limit availability.

## Quality and fallback

On 20 unique synthetic development tickets, Laya's top label matched 16/20. At normalized top-label threshold 0.85, 0/20 were auto-routed and 20/20 required review. At threshold 0.5, 9/20 were auto-routed with no mistakes among those nine. This small development set cannot establish production quality or a safe automation rate. Repeated requests contribute no additional independent accuracy evidence.

The current policy's review rate becomes a 100% API fallback rate only if every review is configured to call an LLM. A 10% fallback rate is an explicitly hypothetical target. A 55% rate illustrates the threshold-0.5 development observation, not a validated production policy. At 100% API fallback, a Laya cascade costs the direct API amount plus hosting. Laya-only assumes accepting every classification, bypassing review; its compute cost is not an equally accurate replacement estimate. Human review expense is unknown and excluded.

## Cost assumptions and formulas

Standard, uncached, short-context API rates checked 6 October 2026:

- [Claude Sonnet 5.5](https://platform.claude.com/docs/en/models/sonnet-5-5/overview): $2/million input tokens, $10/million output tokens.
- [GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna): $0.10/million input tokens, $0.50/million output tokens.

Default request budgets: **500 billable input tokens and 50 total billable output tokens, including any reasoning**. These are assumptions, not measured API usage. Provider tokenizers differ; adjust each budget to a real prompt. Thinking defaults can exceed a small output assumption. No rate-limit sufficiency at projected traffic is established. Batch processing reduces token prices by 50% for both models but is asynchronous and is not equivalent to a live decision service. Cache-read/write behavior, tools, regional premiums and long-context premiums are not included in the default model.

`monthly_requests = sustained_requests_per_second × 30 × 24 × 3600`

`api_cost_per_request = (input_tokens × input_price + output_tokens × output_price) / 1,000,000`

`effective_node_rps = measured_or_assumed_node_rps × utilization / peak_to_average_ratio`

`nodes = max(1, ceil(sustained_requests_per_second / effective_node_rps))`

`laya_only_monthly_compute = nodes × node_monthly_cost + overhead`

`cascade_monthly_cost = laya_only_monthly_compute + fallback_fraction × monthly_api_cost`

Default hosting assumption: **$50 per node per month**, **25 requests/s per node**, **60% utilization**, peak/average traffic factor 1, zero other overhead. Effective capacity is 15 requests/s per node. The cost is illustrative, not a hosting quote; no equivalently performing cloud node was tested. Linear replica scaling is an assumption. Changing to the long-input measurement (~6 requests/s) changes effective capacity to ~3.6 requests/s and increases workers at 10 requests/s from one to three. Bursty traffic, redundancy, hosting type and longer inputs can increase costs.

At 1/10/100/1,000 sustained requests/s, Laya requires 1/1/7/67 assumed workers and costs $50/$50/$350/$3,350 per month. Sonnet costs $3,888/$38,880/$388,800/$3,888,000. Luna costs $194.40/$1,944/$19,440/$194,400. This is continuous traffic, not business-hours-only demand. A duty cycle reduces volume; always-on hosting may stay fixed.

Without fallback, the illustrative first worker breaks even with Sonnet at ~33,333 decisions/month (~0.0129 requests/s), or Luna at ~666,667/month (~0.2572 requests/s). With 10% fallback, those crossings become ~37,037 (~0.0143 requests/s) and ~740,741 (~0.2858 requests/s). These are cost crossings within the first worker's assumed capacity, not endorsements of decision quality. Replica steps change later crossings.

Figures compare inference spend. They exclude engineering, maintenance, human review, retries, taxes, tools, hardware purchase/amortization, energy and financial consequences of incorrect decisions. Add realistic overhead and fallback before making a deployment decision.
