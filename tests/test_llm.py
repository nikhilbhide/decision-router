import json

import httpx
import pytest

from decision_router.providers import HF_BASE_URL, HF_DEFAULT_MODEL, HostedLLM


def configure(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-only-not-a-real-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")


def test_llm_requests_structured_output_and_does_not_invent_probabilities(monkeypatch):
    configure(monkeypatch)

    def fake_post(url, **kwargs):
        assert kwargs["json"]["response_format"]["json_schema"]["strict"] is True
        assert kwargs["json"]["messages"][-1]["role"] == "user"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps({"department": "billing"})}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 8},
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    prediction = HostedLLM().predict("charged twice")
    assert prediction.label == "billing"
    assert prediction.probabilities is None
    assert prediction.input_tokens == 100


def test_llm_invalid_label_is_rejected(monkeypatch):
    configure(monkeypatch)
    monkeypatch.setattr(
        httpx,
        "post",
        lambda url, **kwargs: httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"department":"invented-team"}'}}]},
            request=httpx.Request("POST", url),
        ),
    )
    with pytest.raises(ValueError, match="unsupported"):
        HostedLLM().predict("ticket")


def test_hugging_face_token_and_default_model(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.setenv("HF_TOKEN", "test-hf-not-a-real-token")
    client = HostedLLM()
    assert client.base == HF_BASE_URL
    assert client.model == HF_DEFAULT_MODEL
    assert client.key == "test-hf-not-a-real-token"


def test_explicit_empty_session_key_does_not_use_other_provider_credentials(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-other-provider")
    with pytest.raises(ValueError, match="Configure"):
        HostedLLM(api_key="", model=HF_DEFAULT_MODEL, base_url=HF_BASE_URL)
