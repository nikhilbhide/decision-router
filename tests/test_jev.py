from dataclasses import asdict

import httpx
import pytest

from decision_router.core import LABELS, QUESTIONS, Policy, route
from decision_router.providers import JEV_MODEL, VERCEL_EVALUATE_URL, Jev, make_provider


def response_body():
    return {
        "model": JEV_MODEL,
        "answers": {
            "department": {
                "type": "choice",
                "choice": "billing",
                "probabilities": {label: 0.92 if label == "billing" else 0.02 for label in LABELS},
                "confidence": 0.7,
            }
        },
        "usage": {"inputTokens": 120, "outputTokens": 5},
    }


def stub(monkeypatch, body, status=200):
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return httpx.Response(status, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", post)
    return calls


def test_jev_contract_preserves_probabilities_and_uses_gateway_endpoint(monkeypatch):
    calls = stub(monkeypatch, response_body())
    result = route("charged twice", Jev(api_key="test-gateway-key"), Policy())
    assert result.status == "auto_routed"
    assert result.primary.provider == "jev-vercel"
    assert result.primary.input_tokens == 120
    assert result.primary.output_tokens == 5
    assert result.primary.sdk_confidence == 0.7
    assert result.primary.top_probability == pytest.approx(0.92)
    url, request = calls[0]
    assert url == VERCEL_EVALUATE_URL
    assert request["headers"]["Authorization"] == "Bearer test-gateway-key"
    assert request["json"] == {"model": JEV_MODEL, "state": "charged twice", "questions": QUESTIONS}
    assert "test-gateway-key" not in str(asdict(result.primary))


@pytest.mark.parametrize(
    "problem", ["type", "label", "missing_label", "sum", "nan", "confidence", "non_top"]
)
def test_invalid_jev_outputs_require_review(monkeypatch, problem):
    body = response_body()
    answer = body["answers"]["department"]
    if problem == "type":
        answer["type"] = "boolean"
    elif problem == "label":
        answer["choice"] = "invented"
    elif problem == "missing_label":
        del answer["probabilities"]["sales"]
    elif problem == "sum":
        answer["probabilities"]["billing"] = 0.5
    elif problem == "nan":
        answer["probabilities"]["billing"] = float("nan")
    elif problem == "confidence":
        answer["confidence"] = 2
    else:
        answer["choice"] = "account"
    stub(monkeypatch, body)
    result = route("ticket", Jev(api_key="test-key"), Policy())
    assert result.status == "human_review"
    assert result.primary is None


@pytest.mark.parametrize("status", [401, 402, 429, 500])
def test_gateway_failure_requires_review_and_does_not_expose_response(monkeypatch, status):
    stub(monkeypatch, {"error": "test-key must never appear in trace"}, status)
    result = route("ticket", Jev(api_key="test-key"), Policy())
    assert result.status == "human_review"
    assert "test-key" not in str(result.to_dict())


def test_explicit_empty_jev_key_does_not_fall_back_to_environment(monkeypatch):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-environment-key")
    assert make_provider("jev").key == "test-environment-key"
    with pytest.raises(ValueError, match="Gateway"):
        Jev(api_key="")


def test_missing_gateway_key_fails_before_network_call(monkeypatch):
    monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)
    calls = stub(monkeypatch, response_body())
    with pytest.raises(ValueError, match="Gateway"):
        Jev()
    assert not calls
