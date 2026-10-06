"""Check cost arithmetic and capacity scaling, independently of model downloads."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "laya_economics", Path(__file__).resolve().parents[1] / "scripts/laya_economics.py"
)
economics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(economics)


def test_standard_prices_include_input_and_output():
    assert economics.api_cost_per_request("sonnet") * 1e6 == pytest.approx(1500)
    assert economics.api_cost_per_request("luna") * 1e6 == pytest.approx(75)


def test_full_fallback_adds_hosting_and_cannot_save():
    result = economics.scenario(1e6, 50, 25, 1)
    assert result["sonnet"]["cascade"] == pytest.approx(1550)
    assert result["luna"]["cascade"] == pytest.approx(125)
    assert result["luna"]["cascade_savings"] == pytest.approx(-50)


def test_replicas_round_up_and_include_peak_headroom():
    capacity = 25 * 0.6 * economics.SECONDS_PER_MONTH / 3
    assert economics.scenario(capacity, 50, 25, 0.1, peak_factor=3)["nodes"] == 1
    result = economics.scenario(capacity + 1, 50, 25, 0.1, peak_factor=3)
    assert result["nodes"] == 2
    assert result["laya_compute"] == 100


def test_cache_and_batch_discounts_apply_only_to_api():
    assert economics.api_cost_per_request("sonnet", cached_fraction=1, batch=True) == pytest.approx(
        (500 * 0.2 + 50 * 10) / 1e6 / 2
    )


@pytest.mark.parametrize("fallback", [-1, 1.1, float("nan")])
def test_invalid_scenarios_rejected(fallback):
    with pytest.raises(ValueError):
        economics.scenario(1e6, 50, 25, fallback)
