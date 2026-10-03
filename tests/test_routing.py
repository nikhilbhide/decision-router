import math

import pytest

from decision_router.core import LABELS, Policy, Prediction, route
from decision_router.providers import Simulation


class Fixed:
    def __init__(self, label="billing", probability=0.9):
        self.label = label
        self.probability = probability
        self.calls = 0

    def predict(self, text):
        self.calls += 1
        probabilities = dict.fromkeys(LABELS, (1 - self.probability) / 4)
        probabilities[self.label] = self.probability
        return Prediction(self.label, "test", 10, probabilities)


class Broken:
    def predict(self, text):
        raise RuntimeError("sensitive provider diagnostics")


def test_confident_primary_skips_fallback():
    fallback = Fixed()
    result = route("charged twice", Fixed(), Policy(), fallback)
    assert result.status == "auto_routed"
    assert result.department == "billing"
    assert fallback.calls == 0


def test_agreement_and_disagreement_have_different_review_paths():
    first = Fixed(probability=0.7)
    agreed = route("uncertain ticket", first, Policy(), Fixed())
    disagreed = route("uncertain ticket", first, Policy(), Fixed("technical"))
    assert agreed.status == "auto_routed_after_agreement"
    assert disagreed.status == "human_review"
    assert disagreed.department is None


def test_very_low_probability_and_other_always_need_review():
    assert route("uncertain", Fixed(probability=0.4), Policy(), Fixed()).status == "human_review"
    assert route("unrelated", Fixed("other"), Policy(), Fixed("other")).status == "human_review"


def test_provider_failures_and_empty_tickets_need_review_without_leaking_diagnostics():
    result = route("ticket", Broken(), Policy())
    assert result.status == "human_review"
    assert "sensitive" not in result.reason
    assert route("ticket", Fixed(probability=0.7), Policy(), Broken()).status == "human_review"
    model = Fixed()
    assert route("  ", model, Policy()).status == "human_review"
    assert model.calls == 0


@pytest.mark.parametrize("threshold", [-1, 2, math.nan, math.inf])
def test_invalid_policy_rejected(threshold):
    with pytest.raises(ValueError):
        Policy(threshold)


def test_label_only_predictions_cannot_auto_route():
    class LabelOnly:
        def predict(self, text):
            return Prediction("billing", "llm", 1)

    assert route("ticket", LabelOnly(), Policy()).status == "human_review"


@pytest.mark.parametrize(
    "probabilities",
    [
        {"billing": 1},
        dict.fromkeys(LABELS, 0.5),
        {**dict.fromkeys(LABELS, 0.0), "billing": math.nan},
        {**dict.fromkeys(LABELS, 0.0), "technical": 1},
    ],
)
def test_malformed_model_distributions_rejected(probabilities):
    with pytest.raises(ValueError):
        Prediction("billing", "test", 1, probabilities)


def test_simulation_is_identified_and_ambiguous_ticket_escalates():
    prediction = Simulation().predict("An invoice and an API error")
    assert prediction.provider == "simulation"
    assert sum(prediction.probabilities.values()) == pytest.approx(1)
    assert route("An invoice and an API error", Simulation(), Policy()).status == "human_review"
