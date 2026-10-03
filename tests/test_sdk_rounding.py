import pytest

from decision_router.providers import normalize_sdk_probabilities


def test_laya_four_decimal_rounding_is_normalized():
    rounded = {
        "billing": 0.7361,
        "technical": 0.0593,
        "account": 0.0548,
        "sales": 0.066,
        "other": 0.0837,
    }
    normalized = normalize_sdk_probabilities(rounded)
    assert sum(normalized.values()) == pytest.approx(1)
    assert normalized["billing"] == pytest.approx(0.7361 / 0.9999)


def test_bad_distribution_is_not_fixed_by_normalization():
    with pytest.raises(ValueError, match="tolerance"):
        normalize_sdk_probabilities(
            {"billing": 0.9, "technical": 0.1, "account": 0.1, "sales": 0.1, "other": 0.1}
        )
