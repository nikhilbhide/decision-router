# Jev versus LLMs: price-only projections

This is the current public-post analysis. It replaces the earlier illustrative local Laya hosting scenario with TypeSafe Jev's published token price. No paid inference calls, hosted load tests or cloud provisioning are performed. Existing local Laya measurements are separate and do not establish Jev performance.

Prices checked 6 October 2026:

- [TypeSafe Jev 1.13](https://docs.typesafe.ai/models): $0.042 per million **input** tokens; output free. Direct provider list price used, rather than a rounded gateway display.
- [GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna): $0.10/million input, $0.50/million output.
- [Claude Sonnet 5.5](https://platform.claude.com/docs/en/models/sonnet-5-5/overview): $2/million input, $10/million output.

Default assumptions: one decision per request; 500 total billable input tokens per model; 50 total billable LLM output tokens including reasoning; standard uncached pricing; continuous traffic for 30 days. These are adjustable scenarios, not measured token usage. Jev input includes the state and question definitions, instructions, options and criteria. Provider tokenizers and prompt formats differ. Jev can accept multiple questions per request, but this analysis assumes one; do not equate request throughput with decision throughput when batching questions.

Monthly requests = requests per second × active hours per day × 3,600 × 30.

Jev cost = requests × Jev input tokens × 0.042 / 1,000,000.

LLM cost = requests × (LLM input tokens × input price + total billed output tokens × output price) / 1,000,000.

Default per-million-decision costs: Jev $21, Luna $75, Sonnet $1,500. Jev/LLM cost ratios are ~0.28 against Luna and ~0.014 against Sonnet: projected spend is 72% and 98.6% lower under these budgets. There is no fixed hosting break-even threshold in this pure variable token-price model.

At 1/10/100/1,000 sustained requests/s, monthly requests are 2.592M/25.92M/259.2M/2.592B. Jev costs $54.432/$544.32/$5,443.20/$54,432; Luna $194.40/$1,944/$19,440/$194,400; Sonnet $3,888/$38,880/$388,800/$3,888,000. Display rounding is applied only at presentation, not during calculations.

TypeSafe's current published limits are 80 requests/s and 100,000 tokens/s, subject to dynamic changes. For a 500-token request, the request limit binds before the token limit. Points at 100/1,000 requests/s are hypothetical list-price arithmetic requiring higher limits; enterprise prices and service availability are not established. Being within documented limits is not evidence of sustained throughput, latency or a service guarantee. LLM availability and rate limits at the projected scale were not evaluated.

No quality comparison was performed for Jev, Luna or Sonnet. This cost model concerns a fixed-choice decision task; it does not establish capability equivalence, reliability, safety or performance on complex reasoning. No Laya accuracy or latency results are attributed to Jev.

The default analysis excludes caching and cache writes, batch discounts, retries, fallback calls, tools, regional and long-context premiums, surrounding infrastructure, labor, review and taxes. The token price includes hosted model inference; no illustrative local-worker hosting cost is added. A 50-token output assumption can underestimate billed reasoning under default model settings; use the full billed budget when changing scenarios.

Reproduce without model access or credentials:

```sh
models/.venv-security/bin/python scripts/jev_economics.py
.venv/bin/python -m pytest tests/test_jev_economics.py -q
```

The report generator requires matplotlib for the image. Arithmetic in `project()` uses only Python's standard library. Outputs: `evidence.json` with prices, sources, assumptions, rate-limit flags and projections; `linkedin-economics.png`; and the separately authored `linkedin-post.md`. Nothing is posted automatically.
