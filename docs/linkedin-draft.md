# LinkedIn draft

I built a small support-ticket router to explore a practical question:
when should an AI decision become an action, and when should it ask for help?

Laya runs locally and returns a department plus probabilities. Python applies
an explicit routing threshold. If the case needs a second opinion, an optional
hosted LLM can classify it too. Agreement above a separate threshold can route
the ticket; unresolved cases go to a human-review queue.

The interface also lets me compare Laya and the LLM independently on the same
message. Comparison and fallback answer different questions: how do the models
behave, and how should the workflow use their outputs?

One observation from the saved development run: Laya classified 16 of 20 synthetic
tickets correctly, but none reached the initial 85% routing threshold. Previewing
60% accepted 7 of 20, with no errors among those seven in this small sample.
Choosing a department and deciding whether to act are separate decisions.

These are hand-authored smoke tests, not evidence of production quality. The
hosted LLM and combined fallback workflow still need evaluation; I am not
claiming a measured speed, cost, or accuracy advantage over an LLM.

Next: a larger labeled dataset, a frozen policy, and an untouched test split.

Built with the independent open-weight Laya model:
https://github.com/NandhaKishorM/laya

#AIEngineering #DecisionIntelligence #HumanInTheLoop #OpenSource

## Before publishing

- Confirm these development figures against `results/laya-dev.json`.
- Link the repository only after it has a verified shareable URL.
- Add a real LLM comparison and test-set results when available; describe their limits.
- Credit Laya as an independent project. This application does not use Jev in the UI.
