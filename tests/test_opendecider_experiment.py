"""The experimental adapter rejects truncated or malformed decisions."""

import importlib.util
from pathlib import Path

import pytest

from decision_router.core import LABELS, QUESTIONS

spec = importlib.util.spec_from_file_location(
    "compare_decision_models",
    Path(__file__).resolve().parents[1] / "scripts" / "compare_decision_models.py",
)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def adapter(answer, warnings=None):
    class FakeModel:
        def system_one(self, text, questions):
            assert text == "Refund please"
            assert questions == QUESTIONS
            return {
                "answers": {"department": answer},
                "warnings": warnings,
                "usage": {"input_tokens": 120, "output_tokens": 0},
            }

    provider = object.__new__(experiment.OpenDeciderNano)
    provider.agent = FakeModel()
    provider.sdk_version = "0.8.1"
    provider.revision = experiment.NANO_REVISION
    return provider


def choice():
    probabilities = dict.fromkeys(LABELS, 0.025)
    probabilities["billing"] = 0.90
    return {
        "type": "choice",
        "choice": "billing",
        "probabilities": probabilities,
        "confidence": 0.9,
    }


def test_adapter_preserves_probabilities_and_revision():
    result = adapter(choice()).predict("Refund please")
    assert result.top_probability == 0.9
    assert result.revision == experiment.NANO_REVISION
    assert result.input_tokens == 120
    assert result.output_tokens == 0


def test_adapter_rejects_truncation():
    answer = choice()
    answer["truncated"] = True
    with pytest.raises(ValueError, match="truncated"):
        adapter(answer).predict("Refund please")


def test_adapter_rejects_invalid_distribution():
    answer = choice()
    answer["probabilities"]["billing"] = 0.5
    with pytest.raises(ValueError, match="sum to one"):
        adapter(answer).predict("Refund please")
