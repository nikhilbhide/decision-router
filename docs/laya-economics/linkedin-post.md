# LinkedIn draft

What does it cost to make a decision at 1, 10 or 100 requests per second?

I tested Laya locally, then projected the inference cost of sending every decision to an LLM versus a dedicated decision model. No paid API calls or cloud machines were used.

The local benchmark: Apple M4 Pro, 24 GiB RAM, 3,040 requests, zero runtime errors.

Laya handled about 25–26 requests/second for short inputs and about 6 requests/second for longer inputs. Adding concurrent clients increased queue latency; it did not multiply the single worker's throughput.

For cost, I assumed 500 billable input tokens + 50 total billable output tokens per LLM decision, including any reasoning. Standard API rates, no caching, traffic running continuously for 30 days.

At 1 request/second: 2.592 million decisions/month.
• Claude Sonnet 5.5: $3,888/month
• GPT-6 Luna: $194.40/month
• Laya only: $50/month under the illustrative hosting assumption below

At 10 requests/second: 25.92 million decisions/month.
• Claude Sonnet 5.5: $38,880/month
• GPT-6 Luna: $1,944/month
• Laya only: $50/month under the same assumption

At 100 requests/second: 259.2 million decisions/month.
• Claude Sonnet 5.5: $388,800/month
• GPT-6 Luna: $19,440/month
• Laya only: $350/month across seven assumed workers

The hosting assumption is $50 per worker per month, 25 requests/second per worker and 60% utilization: 15 requests/second of usable capacity. This is a sensitivity scenario, not a cloud quote. I have not validated that a $50 server matches my laptop, or that replicas scale linearly. The formula rounds worker counts up.

The quality caveat matters. Laya got 16/20 top labels correct on a tiny synthetic development set. At the current 0.85 threshold, none qualified for automatic routing. If every uncertain decision still calls an LLM, adding Laya adds cost. Accepting every label gives the Laya-only cost above, but does not establish equivalent decision quality.

My takeaway: a specialized model can make inference spend much smaller at sustained traffic. Whether it is the right system depends on acceptable error rates, real hosting capacity and how often it needs help from a larger model.

These are inference-cost projections. They exclude human review, engineering, hardware capex/energy, retries, tools and taxes. LLM latency and accuracy were not tested. Prices checked 6 October 2026.

Sources:
https://platform.claude.com/docs/en/models/sonnet-5-5/overview
https://developers.openai.com/api/docs/models/gpt-6-luna

#AI #MachineLearning #Inference #LLM
