# Decision Router architecture

## Interactive routing

```mermaid
flowchart TD
    User["User enters a ticket"] --> UI["Streamlit UI"]
    UI --> Provider{"Selected engine"}
    Provider -->|"Live local inference"| Laya["Laya pinned checkpoint"]
    Provider -->|"Illustrative mode"| Sim["Labeled keyword simulation"]
    HF["Hugging Face weights cache"] -.->|"Initial download; cached thereafter"| Laya
    Laya --> Valid["Validate typed choice and probabilities"]
    Sim --> Valid
    Valid --> Gate{"In scope and probability above threshold?"}
    Gate -->|"Yes"| Auto["Automatically select department"]
    Gate -->|"No"| Enabled{"LLM fallback enabled?"}
    Enabled -->|"No"| Human["Session-local human review queue"]
    Enabled -->|"Yes"| LLM["Configured LLM: structured classification"]
    LLM --> Agree{"Same department and probability above agreement threshold?"}
    Agree -->|"Yes; in scope"| FallbackAuto["Route after model agreement"]
    Agree -->|"No"| Human
    Laya -.->|"Overflow or inference failure"| Human
    Valid -.->|"Invalid output"| Human
    LLM -.->|"Failure or invalid output"| Human
    Auto --> Trace["Decision trace and JSON export"]
    FallbackAuto --> Trace
    Human --> Trace
    Trace --> UI
```

The model estimates a distribution; ordinary Python owns the action policy.
`other` is always out of scope for automatic routing. The fallback agreement rule
is an experimental policy, not a guarantee. The queue is a demonstration, not an
external support integration. Live Laya has no per-request inference API; the
optional hosted LLM receives the ticket when enabled.

## Evaluation and live comparison

```mermaid
flowchart LR
    Dataset["Labeled tickets and isolated groups"] --> Dev["Development split"]
    Dataset --> Test["Untouched test split"]
    Dev --> Predict["One provider per benchmark run"]
    Laya["Local Laya"] --> Predict
    LLM["Structured LLM baseline"] --> Predict
    Predict --> Frozen["Frozen predictions; checkpoint SHA; dataset hash"]
    Frozen --> Sweep["Threshold sweep: coverage versus routing errors"]
    Sweep --> Policy["Choose and freeze a policy on development data"]
    Policy --> Final["Test evaluation"]
    Test --> Final
    Final --> Report["Accuracy; warm latency; review rate; calibration diagnostics"]
    Report --> Export["Inspectable JSON report"]
    Ticket["One ticket in the live comparison UI"] --> Both["Run actual Laya and configured LLM"]
    Both --> Compare["Compare decisions and measured latency"]
```

Benchmark commands evaluate providers individually. The interactive comparison
measures one ticket; neither establishes combined cascade performance. Warm latency
excludes checkpoint loading and warmup, while reports preserve the warmup duration.
Laya's rounded probabilities are normalized only within their four-decimal rounding
tolerance; no domain calibration is fitted by the application.

