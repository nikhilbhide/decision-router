"""Optional models are imported only when selected."""

import importlib.metadata
import json
import math
import os
import re
from time import perf_counter

import httpx

from .core import LABELS, QUESTIONS, Prediction

DEFAULT_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
HF_BASE_URL = "https://router.huggingface.co/v1"
HF_DEFAULT_MODEL = "Qwen/Qwen3.5-35B-A3B:deepinfra"
JEV_MODEL = "typesafe-ai/jev"
VERCEL_EVALUATE_URL = "https://ai-gateway.vercel.sh/v1/evaluate"
VERCEL_BASE_URL = "https://ai-gateway.vercel.sh/v1"
VERCEL_DEFAULT_MODEL = "alibaba/qwen-3-32b"


def normalize_sdk_probabilities(probabilities):
    """SDK rounds to four decimals; repair only the resulting small sum error."""
    if set(probabilities) != set(LABELS):
        raise ValueError("SDK distribution has unexpected labels")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()):
        raise ValueError("SDK probabilities are invalid")
    total = sum(probabilities.values())
    if total <= 0 or abs(total - 1) > len(LABELS) * 0.00005 + 1e-7:
        raise ValueError("SDK probability sum exceeds its rounding tolerance")
    return {label: p / total for label, p in probabilities.items()}


class Simulation:
    """Deterministic keyword rules with illustrative scores, never model evidence."""

    keywords = {
        "billing": ("invoice", "charged", "refund", "payment", "subscription", "billed"),
        "technical": ("bug", "error", "crash", "api", "integration", "outage", "slow"),
        "account": ("login", "log in", "password", "access", "authentication", "permission"),
        "sales": ("pricing", "demo", "purchase", "sales", "compare", "quote"),
    }

    def predict(self, text: str) -> Prediction:
        start = perf_counter()
        lower = text.lower()
        scores = {
            label: sum(word in lower for word in words) for label, words in self.keywords.items()
        }
        winners = [label for label, score in scores.items() if score > 0]
        probs = dict.fromkeys(LABELS, 0.025)
        if len(winners) == 1:
            label = winners[0]
            probs[label] = 0.90
        elif winners:
            probs = dict.fromkeys(LABELS, 0.02)
            for label in winners:
                probs[label] = (1 - 0.02 * (len(LABELS) - len(winners))) / len(winners)
            label = winners[0]
        else:
            label = "other"
            probs[label] = 0.90
        return Prediction(
            label, "simulation", (perf_counter() - start) * 1000, probs, "keyword-rules"
        )


class Laya:
    """Download one pinned checkpoint, then run it locally."""

    def __init__(self, checkpoint="typed-decisions", device="cpu", revision=None):
        if checkpoint not in {"english", "typed-decisions", "multilingual"}:
            raise ValueError("Unsupported Laya checkpoint")
        from huggingface_hub import snapshot_download
        from laya import load

        repo = "convaiinnovations/laya"
        self.revision = revision or DEFAULT_REVISION
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("Pass an immutable 40-character Hugging Face revision SHA")
        self.checkpoint = checkpoint
        prefix = "" if checkpoint == "english" else checkpoint + "/"
        folder = snapshot_download(
            repo,
            revision=self.revision,
            allow_patterns=[
                prefix + path
                for path in (
                    "rl_agent_config.json",
                    "model.safetensors",
                    "encoder/config.json",
                    "tokenizer/tokenizer.json",
                    "tokenizer/tokenizer_config.json",
                )
            ],
        )
        self.agent = load(str(os.path.join(folder, prefix)), device=device)
        self.sdk_version = importlib.metadata.version("laya")

    def predict(self, text: str) -> Prediction:
        # SDK otherwise truncates silently. Refuse tickets that don't fit this checkpoint.
        from laya.common import build_sequence

        question = self.agent._to_internal(QUESTIONS["department"])
        head_budget = self.agent.cfg.get("head_max_len", 192)
        limit = self.agent.cfg.get("max_len", 512)
        full_ids, _ = build_sequence(self.agent.tok, text, question, 1_000_000, head_budget)
        if len(full_ids) > limit:
            raise ValueError("Ticket exceeds checkpoint token budget; shorten or split it")
        start = perf_counter()
        result = self.agent.system_one(text, QUESTIONS)
        elapsed = (perf_counter() - start) * 1000
        answer = result["answers"]["department"]
        usage = result.get("usage", {})
        return Prediction(
            answer["choice"],
            "laya",
            elapsed,
            normalize_sdk_probabilities(answer["probabilities"]),
            f"{self.checkpoint} / laya {self.sdk_version}",
            self.revision,
            answer.get("confidence"),
            usage.get("input_tokens"),
            usage.get("output_tokens"),
        )


class Jev:
    """Hosted typed decisions through Vercel's evaluation API."""

    def __init__(self, api_key=None):
        self.key = api_key if api_key is not None else os.environ.get("AI_GATEWAY_API_KEY", "")
        if not self.key.strip():
            raise ValueError("Add your Vercel AI Gateway API key")
        self.model = JEV_MODEL

    def predict(self, text: str) -> Prediction:
        start = perf_counter()
        response = httpx.post(
            VERCEL_EVALUATE_URL,
            timeout=45,
            headers={"Authorization": f"Bearer {self.key}"},
            json={"model": self.model, "state": text, "questions": QUESTIONS},
        )
        response.raise_for_status()
        data = response.json()
        answer = data["answers"]["department"]
        if answer.get("type") != "choice":
            raise ValueError("Jev returned an unexpected answer type")
        probabilities = normalize_sdk_probabilities(answer["probabilities"])
        confidence = answer.get("confidence")
        if confidence is not None and (not math.isfinite(confidence) or not 0 <= confidence <= 1):
            raise ValueError("Jev returned invalid confidence")
        usage = data.get("usage", {})
        return Prediction(
            label=answer["choice"],
            provider="jev-vercel",
            latency_ms=(perf_counter() - start) * 1000,
            probabilities=probabilities,
            model=data.get("model", self.model),
            sdk_confidence=confidence,
            input_tokens=usage.get("inputTokens"),
            output_tokens=usage.get("outputTokens"),
        )


class HostedLLM:
    """OpenAI-compatible structured classification; no invented confidence scores."""

    def __init__(self, api_key=None, model=None, base_url=None):
        self.base = (base_url or os.environ.get("LLM_BASE_URL", HF_BASE_URL)).rstrip("/")
        if self.base == HF_BASE_URL:
            default_key = os.environ.get("HF_TOKEN", "") or os.environ.get("LLM_API_KEY", "")
            default_model = HF_DEFAULT_MODEL
        elif self.base == VERCEL_BASE_URL:
            default_key = os.environ.get("AI_GATEWAY_API_KEY", "")
            default_model = VERCEL_DEFAULT_MODEL
        else:
            default_key = os.environ.get("LLM_API_KEY", "")
            default_model = ""
        self.key = api_key if api_key is not None else default_key
        self.model = model if model is not None else os.environ.get("LLM_MODEL", default_model)
        if not self.key.strip() or not self.model.strip():
            raise ValueError("Configure the selected service's API key and LLM model")
        if not self.base.startswith("https://") and not re.match(
            r"^http://(localhost|127\.0\.0\.1)(:|/|$)", self.base
        ):
            raise ValueError("Use HTTPS or a local LLM endpoint")

    def predict(self, text: str) -> Prediction:
        schema = {
            "name": "department",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"department": {"type": "string", "enum": list(LABELS)}},
                "required": ["department"],
            },
        }
        start = perf_counter()
        response = httpx.post(
            self.base + "/chat/completions",
            timeout=45,
            headers={"Authorization": f"Bearer {self.key}"},
            json={
                "model": self.model,
                "temperature": 0,
                "max_tokens": 2048,
                "response_format": {"type": "json_schema", "json_schema": schema},
                "messages": [
                    {
                        "role": "system",
                        "content": QUESTIONS["department"]["instructions"]
                        + "\n"
                        + json.dumps(LABELS)
                        + '\nReturn only JSON with the key "department" and an allowed label.',
                    },
                    {"role": "user", "content": text},
                ],
            },
        )
        response.raise_for_status()
        data = response.json()
        value = json.loads(data["choices"][0]["message"]["content"])
        if set(value) != {"department"}:
            raise ValueError("LLM response must contain only department")
        usage = data.get("usage", {})
        return Prediction(
            value["department"],
            "hosted-llm",
            (perf_counter() - start) * 1000,
            model=self.model,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )


def make_provider(name, checkpoint="typed-decisions", device="cpu", revision=None):
    if name == "simulation":
        return Simulation()
    if name == "laya":
        return Laya(checkpoint, device, revision)
    if name == "jev":
        return Jev()
    if name == "llm":
        return HostedLLM()
    raise ValueError("Unknown provider")
