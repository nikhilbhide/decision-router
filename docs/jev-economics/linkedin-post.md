# LinkedIn draft

What happens to your decision-making API bill at 1, 10 or 100 requests per second?

I compared published pricing for TypeSafe Jev, GPT-6 Luna and Claude Sonnet 5.5. No paid calls were made: this is an extrapolation, not a performance benchmark.

The scenario: one decision per request, 500 billable input tokens for each model, 50 total billable output tokens for the LLMs, including any reasoning. Standard uncached prices, traffic running 24/7 for 30 days.

Jev charges $0.042 per million input tokens, with free output. Under those assumptions, one million decisions costs:
• Jev: $21
• GPT-6 Luna: $75
• Claude Sonnet 5.5: $1,500

At 1 request/second → 2.592 million decisions/month:
• Jev: $54.43
• Luna: $194.40
• Sonnet: $3,888

At 10 requests/second → 25.92 million decisions/month:
• Jev: $544.32
• Luna: $1,944
• Sonnet: $38,880

At 100 requests/second → 259.2 million decisions/month:
• Jev: $5,443.20
• Luna: $19,440
• Sonnet: $388,800

That 100 RPS row is arithmetic only: Jev currently publishes limits of 80 requests/second and 100,000 input tokens/second. Higher limits require a custom arrangement; rates could differ.

In this scenario, Jev's API cost is 72% lower than Luna's and 98.6% lower than Sonnet's. Those percentages depend on the token budget, not traffic volume. Longer prompts, different tokenizers, reasoning, caching, retries and fallback all change the comparison.

The question I find useful: for a fixed-choice routing or classification task, how much are we paying for text generation when the output we need is a decision?

This does not establish equal accuracy, faster hosted throughput or interchangeability for complex reasoning. Measure task quality before replacing a model. Costs exclude surrounding infrastructure, review, tools, taxes and fallback.

Prices checked 6 October 2026. Primary sources:
https://docs.typesafe.ai/models
https://developers.openai.com/api/docs/models/gpt-6-luna
https://platform.claude.com/docs/en/models/sonnet-5-5/overview

#AI #LLM #Inference #DecisionMaking
