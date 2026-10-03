from pathlib import Path

import pytest

pytest.importorskip("streamlit")
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
