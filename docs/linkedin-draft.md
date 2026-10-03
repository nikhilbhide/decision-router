# Follow-up draft — complete only after running the experiment

## From decision models to a working decision workflow

In my last post, I explored why decision models could become a useful part of
enterprise AI. This time, I built a small experiment around an open-weight
alternative: Laya.

The task is simple: given a support ticket, choose the team that should handle it.
The engineering question is harder: when should the software act on that choice?

The router handles cases above a probability threshold automatically, can ask an
LLM for a second opinion, and sends unresolved cases to a human-review queue.

I measured [INSERT CHECKPOINT, DEVICE AND DATASET] on [N] independently labeled
test tickets after choosing the threshold on a separate development split:

- Classification accuracy: [INSERT]
- Automated cases: [ACCEPTED / TOTAL]
- Errors among automated cases: [ERRORS / ACCEPTED]
- Warm inference latency: [p50 / p95]

[INSERT ONE ACTUAL FAILURE AND WHAT YOU LEARNED FROM IT.]

The bundled synthetic tickets only verify the pipeline. They are too small to
establish production performance, and the current benchmark does not measure
the combined Laya + LLM cascade.

The project is on GitHub: [INSERT VERIFIED REPOSITORY LINK]. It includes the
local demo, the evaluation command, the routing policy and reproducibility details.

A structured answer is useful. Knowing which answers to automate takes evaluation.

#AIEngineering #DecisionIntelligence #OpenSource #EnterpriseAI
