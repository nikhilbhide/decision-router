from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.proto.TextInput_pb2 import TextInput
from streamlit.testing.v1 import AppTest


def element(collection, label):
    return next(item for item in collection if item.label == label)


def test_ui_routes_a_mock_ticket_and_previews_threshold_without_rerunning():
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=30).run()
    element(app.selectbox, "Decision engine").select("simulation").run()
    element(app.button, "Route ticket").click().run()
    assert not app.exception
    assert app.session_state["last_result"]["primary"]["provider"] == "simulation"
    assert app.session_state["last_result"]["status"] == "auto_routed"
    element(app.slider, "Automatic routing: top probability").set_value(1.0).run()
    assert not app.exception
    assert any(item.value == "Human review required" for item in app.warning)
    assert len(app.session_state["history"]) == 1


def test_ui_evaluation_exports_a_labeled_simulation_run():
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=30).run()
    element(app.selectbox, "Evaluate").select("simulation").run()
    element(app.button, "Run evaluation").click().run()
    assert not app.exception
    assert app.session_state["report"]["simulation"] is True
    assert len(app.session_state["report"]["records"]) == 20


def test_ui_human_can_resolve_a_review_case():
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=30).run()
    element(app.selectbox, "Decision engine").select("simulation").run()
    element(app.text_area, "Customer message").set_value("Something is wrong. Please help.").run()
    element(app.button, "Route ticket").click().run()
    assert app.session_state["last_result"]["status"] == "human_review"
    element(app.selectbox, "Reviewer decision").select("technical").run()
    element(app.button, "Record human decision").click().run()
    assert not app.exception
    assert app.session_state["history"][0]["human_decision"] == "technical"


@pytest.fixture
def model_calls(monkeypatch):
    from decision_router.core import LABELS, Prediction
    from decision_router.providers import HostedLLM, Jev, Laya

    calls = {"laya": [], "llm": [], "probability": 0.7}
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(Laya, "__init__", lambda self, *args, **kwargs: None)

    def predict(self, text):
        calls["laya"].append(text)
        p = calls["probability"]
        probabilities = dict.fromkeys(LABELS, (1 - p) / 4)
        probabilities["billing"] = p
        return Prediction("billing", "laya", 100, probabilities, "typed-decisions")

    def llm_predict(self, text):
        calls["llm"].append((self.key, self.base, self.model))
        return Prediction("billing", "hosted-llm", 500, model=self.model)

    def forbid_jev(*args, **kwargs):
        raise AssertionError("The app must not initialize Jev")

    monkeypatch.setattr(Laya, "predict", predict)
    monkeypatch.setattr(HostedLLM, "predict", llm_predict)
    monkeypatch.setattr(Jev, "__init__", forbid_jev)
    return calls


def test_ui_defaults_to_laya_and_hf_without_requesting_a_model(model_calls):
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert element(app.selectbox, "Decision engine").value == "laya"
    assert all("Jev" not in option for option in element(app.selectbox, "Decision engine").options)
    assert all("Jev" not in option for option in element(app.selectbox, "Evaluate").options)
    assert not any(item.label == "Jev connection" for item in app.expander)
    assert element(app.selectbox, "LLM service").value == "Hugging Face"
    assert element(app.text_input, "Hugging Face token").proto.type == TextInput.PASSWORD
    assert not element(app.button, "Route ticket").disabled
    assert not model_calls["laya"] and not model_calls["llm"]


def test_hf_token_alone_does_not_enable_fallback(model_calls):
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=30).run()
    element(app.text_input, "Hugging Face token").set_value("test-hf-token").run()
    assert not element(app.checkbox, "Ask an LLM when uncertain").value
    element(app.button, "Route ticket").click().run()
    assert not app.exception
    assert len(model_calls["laya"]) == 1
    assert not model_calls["llm"]
    assert app.session_state["last_result"]["status"] == "human_review"
    assert any("fallback was disabled" in item.value for item in app.caption)


def test_hf_fallback_calls_llm_only_when_laya_is_uncertain(model_calls):
    from decision_router.providers import HF_BASE_URL, HF_DEFAULT_MODEL

    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=30).run()
    element(app.text_input, "Hugging Face token").set_value("test-hf-token").run()
    element(app.checkbox, "Ask an LLM when uncertain").check().run()
    element(app.button, "Route ticket").click().run()
    assert not app.exception
    assert app.session_state["last_result"]["status"] == "auto_routed_after_agreement"
    assert model_calls["llm"] == [("test-hf-token", HF_BASE_URL, HF_DEFAULT_MODEL)]
    assert "test-hf-token" not in str(app.session_state["last_result"])

    model_calls["probability"] = 0.92
    element(app.button, "Route ticket").click().run()
    assert app.session_state["last_result"]["status"] == "auto_routed"
    assert len(model_calls["laya"]) == 2
    assert len(model_calls["llm"]) == 1
    assert any("passed the automatic routing threshold" in item.value for item in app.caption)

    element(app.button, "Run live comparison").click().run()
    assert not app.exception
    assert len(model_calls["llm"]) == 2
    assert app.session_state["comparison"]["decision"]["provider"] == "laya"
    assert len(app.session_state["history"]) == 2
    assert "test-hf-token" not in str(app.session_state["comparison"])


def test_llm_request_failure_shows_actionable_error_instead_of_uncertainty(
    monkeypatch, model_calls
):
    import httpx

    from decision_router.providers import HostedLLM

    def fail(self, text):
        httpx.Response(
            402,
            json={"error": "private response test-hf-token"},
            request=httpx.Request("POST", "https://router.huggingface.co/v1/chat/completions"),
        ).raise_for_status()

    monkeypatch.setattr(HostedLLM, "predict", fail)
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=30).run()
    element(app.text_input, "Hugging Face token").set_value("test-hf-token").run()
    element(app.checkbox, "Ask an LLM when uncertain").check().run()
    element(app.button, "Route ticket").click().run()
    assert not app.exception
    assert any("Model request failed" in item.value for item in app.error)
    assert not any(item.value == "Human review required" for item in app.warning)
    result = app.session_state["last_result"]
    assert result["status"] == "human_review"
    assert "HTTP 402" in result["reason"]
    assert "credits or billing" in result["reason"]
    assert "test-hf-token" not in str(result)
    assert "private response" not in str(result)
