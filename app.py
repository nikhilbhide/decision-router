"""Run with: streamlit run app.py --server.address 127.0.0.1"""

import json
import os
import platform
from pathlib import Path

import pandas as pd
import streamlit as st

from decision_router.core import LABELS, Policy, Prediction, apply_policy, route
from decision_router.evaluation import policy_metrics, run_benchmark
from decision_router.providers import HF_BASE_URL, HF_DEFAULT_MODEL, HostedLLM, make_provider

st.set_page_config(page_title="Decision Router Lab", page_icon="↗", layout="wide")

st.markdown("<small>LOCAL DECISION MODELS / EXPERIMENT 01</small>", unsafe_allow_html=True)
st.title("Decide. Escalate. Measure.")
st.write("A support-ticket router that makes uncertainty part of the workflow.")

ENGINE_NAMES = {
    "laya": "Laya · live local model",
    "simulation": "Mock rules · no AI model",
    "llm": "LLM · configured endpoint",
}


with st.sidebar:
    st.header("Routing policy")
    provider_name = st.selectbox(
        "Decision engine", ["laya", "simulation"], format_func=ENGINE_NAMES.get
    )
    checkpoint = st.selectbox("Laya checkpoint", ["typed-decisions", "english", "multilingual"])
    device = st.selectbox(
        "Device",
        ["mps", "cpu", "cuda"]
        if platform.system() == "Darwin" and platform.machine() == "arm64"
        else ["cpu", "cuda"],
    )
    threshold = st.slider("Automatic routing: top probability", 0.40, 1.0, 0.85, 0.01)
    use_fallback = st.checkbox("Ask an LLM when uncertain")
    agreement_threshold = st.slider(
        "Minimum probability for LLM agreement", 0.0, threshold, min(0.60, threshold), 0.01
    )
    st.caption(
        "An agreeing LLM can route a case above the agreement threshold. Other cases enter review. Both thresholds need validation on your data."
    )
    with st.expander("Hosted LLM connection", expanded=True):
        connection = st.selectbox("LLM service", ["Hugging Face", "Custom compatible API"])
        if connection == "Hugging Face":
            model_choice = st.selectbox(
                "Hosted model", [HF_DEFAULT_MODEL, "Qwen/Qwen3-32B:deepinfra", "Custom model ID"]
            )
            llm_model = (
                st.text_input("Hugging Face model ID")
                if model_choice == "Custom model ID"
                else model_choice
            )
            llm_base = HF_BASE_URL
            token_input = st.text_input(
                "Hugging Face token",
                type="password",
                key="hf_token_input",
                help="Use a token with Inference Providers permission. It stays in this browser session; it is not written to disk.",
            )
            llm_key = token_input or os.environ.get("HF_TOKEN", "")
            st.caption(
                "Hosted inference: no LLM weights are downloaded. Requests use your Hugging Face inference credits."
            )
        else:
            llm_base = st.text_input(
                "API base URL", os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
            )
            llm_model = st.text_input("Model ID", os.environ.get("LLM_MODEL", ""))
            token_input = st.text_input("API key", type="password", key="llm_api_input")
            llm_key = token_input or os.environ.get("LLM_API_KEY", "")
        llm_configured = bool(llm_key and llm_model)
        st.caption(
            "Ready to call the hosted model"
            if llm_configured
            else "Add your token to enable the LLM baseline and fallback"
        )
    if provider_name == "simulation":
        st.warning("Simulation uses keyword rules and invented scores. It is not Laya inference.")
    else:
        st.info(
            "The first run downloads one checkpoint from Hugging Face. Subsequent inference runs locally."
        )


@st.cache_resource
def get_provider(name, selected_checkpoint, selected_device):
    return make_provider(name, selected_checkpoint, selected_device)


def configured_llm():
    # Credentials are session-specific, never placed in the shared resource cache.
    return HostedLLM(api_key=llm_key, model=llm_model, base_url=llm_base)


route_tab, compare_tab, eval_tab, review_tab = st.tabs(
    ["Route a ticket", "Live comparison", "Evaluate the model", "Review queue"]
)

with route_tab:
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.subheader("What needs a decision?")
        sample = st.selectbox(
            "Example ticket",
            [
                "We were charged twice this month. Please refund the duplicate payment.",
                "The integration returns an error every time we call the API.",
                "I can't log in, and I also have a question about my invoice.",
                "Something is wrong. Can someone help?",
            ],
        )
        text = st.text_area("Customer message", value=sample, height=180, max_chars=16000)
        clicked = st.button("Route ticket", type="primary", disabled=not text.strip())
        if clicked:
            try:
                with st.spinner("Loading the selected engine and evaluating the ticket…"):
                    primary = get_provider(provider_name, checkpoint, device)
                    fallback = configured_llm() if use_fallback else None
                    result = route(text, primary, Policy(threshold, agreement_threshold), fallback)
                st.session_state["last_result"] = {
                    "text": text,
                    "submitted_threshold": threshold,
                    **result.to_dict(),
                }
                st.session_state.setdefault("history", []).append(st.session_state["last_result"])
            except Exception as exc:
                st.error(
                    f"Engine setup failed ({type(exc).__name__}). Check dependencies, device and optional LLM configuration in README.md."
                )
                st.session_state.pop("last_result", None)
    with right:
        st.subheader("Decision trace")
        result = st.session_state.get("last_result")
        if not result:
            st.write("Submit a ticket to see its decision, probabilities and review path.")
        else:
            if result["primary"] and not result["reason"].startswith("Fallback model failed"):
                current = apply_policy(
                    Prediction(**result["primary"]),
                    Policy(threshold, agreement_threshold),
                    Prediction(**result["fallback"])
                    if result["fallback"] and use_fallback
                    else None,
                )
                result = {**result, **current.to_dict()}
            st.caption(result["text"])
            st.caption(
                "Move the threshold to preview this decision without rerunning the model. The queue keeps the policy used when you submitted it."
            )
            if result["status"] == "human_review":
                st.warning("Human review required")
            else:
                st.success("Route to " + result["department"].title())
            st.write(result["reason"])
            first = result["primary"]
            if first:
                st.caption(f"{first['provider']} · {first['model']}")
                cols = st.columns(2)
                cols[0].metric("Inference latency", f"{first['latency_ms']:.1f} ms")
                if first["probabilities"]:
                    cols[1].metric("Top probability", f"{max(first['probabilities'].values()):.1%}")
                    st.bar_chart(
                        pd.Series(first["probabilities"], name="Probability"), horizontal=True
                    )
                if result["fallback"]:
                    second = result["fallback"]
                    st.write(
                        f"LLM recommendation: **{second['label']}** ({second['latency_ms']:.1f} ms)"
                    )
            with st.expander("Inspect the trace"):
                st.json(result)

with compare_tab:
    st.subheader("Same ticket. Two ways to make the decision.")
    st.write(
        "Run the local decision model and your configured LLM on the same input. Compare the actual decisions and measured inference time."
    )
    compare_text = st.text_area(
        "Comparison ticket",
        "Please refund the extra payment. I was billed twice for one subscription.",
        height=140,
    )
    if not llm_configured:
        st.info(
            "Laya is ready locally. Add your Hugging Face token under Hosted LLM connection in the sidebar to enable the live comparison."
        )
    compare_llm = st.checkbox(
        "Include the configured LLM", value=llm_configured, disabled=not llm_configured
    )
    if st.button("Run live comparison", type="primary", disabled=not compare_text.strip()):
        outputs = {"text": compare_text, "laya": None, "llm": None, "errors": {}}
        with st.spinner("Running actual local Laya inference…"):
            try:
                local = get_provider("laya", checkpoint, device)
                local.predict(compare_text)  # warmup, excluded from the measurement
                from dataclasses import asdict

                outputs["laya"] = asdict(local.predict(compare_text))
            except Exception as exc:
                outputs["errors"]["laya"] = type(exc).__name__
        if compare_llm:
            with st.spinner("Calling the configured LLM…"):
                try:
                    outputs["llm"] = asdict(configured_llm().predict(compare_text))
                except Exception as exc:
                    outputs["errors"]["llm"] = type(exc).__name__
        st.session_state["comparison"] = outputs
    comparison = st.session_state.get("comparison")
    if comparison:
        st.caption(comparison["text"])
        cols = st.columns(2, gap="large")
        for column, key, title in zip(
            cols, ["laya", "llm"], ["Local decision model", "Structured LLM"]
        ):
            with column:
                st.markdown("### " + title)
                prediction = comparison[key]
                if prediction:
                    st.metric("Measured inference time", f"{prediction['latency_ms']:.1f} ms")
                    st.write("Selected department: **" + prediction["label"] + "**")
                    st.caption(prediction["model"])
                    if prediction["probabilities"]:
                        st.bar_chart(
                            pd.Series(prediction["probabilities"], name="Probability"),
                            horizontal=True,
                        )
                        decision = apply_policy(
                            Prediction(**prediction), Policy(threshold, agreement_threshold)
                        )
                        st.write(
                            "Current policy: "
                            + (
                                "route automatically"
                                if decision.status == "auto_routed"
                                else "request review"
                            )
                        )
                    else:
                        st.caption(
                            "No calibrated probability distribution is available from this adapter."
                        )
                elif key in comparison["errors"]:
                    st.error(f"Request failed ({comparison['errors'][key]}).")
                else:
                    st.write("Configure and enable the LLM to measure this side.")
        if comparison["laya"] and comparison["llm"]:
            factor = comparison["llm"]["latency_ms"] / max(comparison["laya"]["latency_ms"], 0.001)
            st.metric("LLM / Laya latency ratio for this ticket", f"{factor:.1f}×")
            st.write(
                "Models "
                + (
                    "agreed"
                    if comparison["laya"]["label"] == comparison["llm"]["label"]
                    else "disagreed"
                )
                + " on the selected department."
            )
        st.caption(
            "One warm local inference call versus one hosted request, including network time. This is a live demonstration, not a broad speed or quality benchmark."
        )
        st.download_button(
            "Export comparison",
            json.dumps(comparison, indent=2),
            "comparison.json",
            "application/json",
        )

with eval_tab:
    st.subheader("How much can the model handle automatically?")
    st.caption(
        "Bundled tickets are hand-authored synthetic examples for smoke testing. They do not establish production accuracy."
    )
    c1, c2 = st.columns(2)
    eval_engine = c1.selectbox(
        "Evaluate", ["laya", "simulation", "llm"], format_func=ENGINE_NAMES.get
    )
    split = c2.selectbox("Dataset split", ["dev", "test"])
    st.write(
        "Explore thresholds on development tickets. Freeze the policy before evaluating test tickets."
    )
    if st.button("Run evaluation", type="primary"):
        try:
            with st.spinner("Running warmup and evaluating each ticket…"):
                provider = (
                    configured_llm()
                    if eval_engine == "llm"
                    else get_provider(eval_engine, checkpoint, device)
                )
                report = run_benchmark(
                    provider, Path(__file__).parent / "data/tickets.jsonl", split, threshold
                )
            st.session_state["report"] = report
        except Exception as exc:
            st.error(
                f"Evaluation failed ({type(exc).__name__}). Check the setup instructions in README.md."
            )
    report = st.session_state.get("report")
    if report:
        st.caption(
            f"Showing saved run: {report['split']} split · {report['records'][0].get('prediction', {}).get('provider', 'unknown') if report['records'][0].get('prediction') else 'failed first request'}"
        )
        if report["simulation"]:
            st.warning(
                "Illustrative simulation results. Do not publish these as model measurements."
            )
        summary = policy_metrics(report["records"], threshold)
        cols = st.columns(4)
        cols[0].metric("Classification accuracy", f"{summary['classification_accuracy']:.1%}")
        cols[1].metric(
            "Automation coverage",
            "Unavailable"
            if summary["automation_coverage"] is None
            else f"{summary['automation_coverage']:.1%}",
        )
        cols[2].metric(
            "Accuracy when automated",
            "No accepted cases"
            if summary["selective_accuracy"] is None
            else f"{summary['selective_accuracy']:.1%}",
        )
        cols[3].metric(
            "Routing errors / accepted",
            "Unavailable"
            if summary["auto_routed"] is None
            else f"{summary['auto_routing_errors']} / {summary['auto_routed']}",
        )
        curve = pd.DataFrame(report["threshold_sweep"]).set_index("threshold")
        if summary["automation_coverage"] is not None:
            st.line_chart(curve[["automation_coverage", "selective_accuracy"]])
            daily_volume = st.slider("Explore daily ticket volume", 100, 100000, 10000, 100)
            projected = round(daily_volume * summary["automation_coverage"])
            st.metric(
                "Projected automatic routes at this threshold", f"{projected:,} / {daily_volume:,}"
            )
            st.caption(
                "Illustrative projection using this run's coverage. Real ticket mix and error rates may differ. Moving the threshold does not call the model again."
            )
        st.caption(
            "Inference latency excludes checkpoint loading and warmup. LLM runs have no probability estimates, so automatic coverage and calibration are unavailable for them."
        )
        st.json(
            {
                "latency_ms": {"p50": summary["p50_latency_ms"], "p95": summary["p95_latency_ms"]},
                "calibration": report["calibration"],
                "failed_requests": summary["failed_requests"],
            },
            expanded=False,
        )
        st.dataframe(
            [
                {
                    "id": row["id"],
                    "expected": row["label"],
                    "predicted": row["prediction"]["label"] if row["prediction"] else "FAILED",
                    "text": row["text"],
                    "error": row["error"],
                }
                for row in report["records"]
            ],
            hide_index=True,
            use_container_width=True,
        )
        # Export the slider's current policy, not a stale threshold from the original run.
        export = {**report, "summary": summary}
        st.download_button(
            "Download reproducible report",
            json.dumps(export, indent=2, allow_nan=False),
            file_name=f"{report['split']}-report.json",
            mime="application/json",
        )

with review_tab:
    st.subheader("Cases that need a person")
    queued = [
        (index, item)
        for index, item in enumerate(st.session_state.get("history", []))
        if item["status"] == "human_review" and "human_decision" not in item
    ]
    st.caption("Session-local queue. No tickets are sent to a team or persisted to a database.")
    if not queued:
        st.write("No cases waiting for review in this session.")
    for number, (index, item) in enumerate(queued, 1):
        with st.expander(f"Case {number} · {item['reason']}"):
            st.write(item["text"])
            if item["primary"]:
                st.write("Primary recommendation: " + item["primary"]["label"])
            st.json(item, expanded=False)
            department = st.selectbox(
                "Reviewer decision", list(LABELS), key=f"review-department-{index}"
            )
            if st.button("Record human decision", key=f"review-submit-{index}"):
                st.session_state["history"][index]["human_decision"] = department
                st.rerun()
    if queued:
        st.download_button(
            "Export review queue",
            json.dumps([item for _, item in queued], indent=2),
            "review-queue.json",
            "application/json",
        )
    resolved = [item for item in st.session_state.get("history", []) if "human_decision" in item]
    if resolved:
        st.subheader("Reviewed cases")
        st.dataframe(
            [
                {"ticket": item["text"], "human_decision": item["human_decision"]}
                for item in resolved
            ],
            hide_index=True,
        )
