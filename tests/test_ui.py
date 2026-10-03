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


def test_jev_ui_requires_key_before_enabling_paid_calls(monkeypatch):
    monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert element(app.selectbox, "Decision engine").value == "jev"
    assert element(app.button, "Route ticket").disabled
    assert element(app.button, "Run live comparison").disabled
    assert element(app.button, "Run evaluation").disabled
    assert element(app.text_input, "Vercel AI Gateway key").proto.type == TextInput.PASSWORD


def test_jev_ui_calls_selected_engine_once_without_local_model_download(monkeypatch):
    from decision_router.core import LABELS, Prediction
    from decision_router.providers import (
        VERCEL_BASE_URL,
        VERCEL_DEFAULT_MODEL,
        HostedLLM,
        Jev,
        Laya,
    )

    calls = []

    def predict(self, text):
        calls.append(self.key)
        return Prediction(
            "billing",
            "jev-vercel",
            150,
            {label: 0.92 if label == "billing" else 0.02 for label in LABELS},
            model=self.model,
        )

    def forbid_local(*args, **kwargs):
        raise AssertionError("Hosted Jev must not load Laya")

    llm_calls = []

    def llm_predict(self, text):
        llm_calls.append((self.key, self.base, self.model))
        return Prediction("billing", "hosted-llm", 500, model=self.model)

    monkeypatch.setattr(Jev, "predict", predict)
    monkeypatch.setattr(Laya, "__init__", forbid_local)
    monkeypatch.setattr(HostedLLM, "predict", llm_predict)
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=30).run()
    element(app.text_input, "Vercel AI Gateway key").set_value("test-session-key").run()
    element(app.checkbox, "Include the configured LLM").check().run()
    element(app.button, "Run live comparison").click().run()
    assert not app.exception
    assert calls == ["test-session-key"]
    assert llm_calls == [("test-session-key", VERCEL_BASE_URL, VERCEL_DEFAULT_MODEL)]
    assert "test-session-key" not in str(app.session_state["comparison"])
    assert app.session_state["comparison"]["decision"]["provider"] == "jev-vercel"
    element(app.button, "Route ticket").click().run()
    assert app.session_state["last_result"]["status"] == "auto_routed"
    assert "test-session-key" not in str(app.session_state["last_result"])
