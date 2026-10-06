"""Independent arithmetic checks for price-only Jev projections."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "jev_economics", Path(__file__).resolve().parents[1] / "scripts/jev_economics.py"
)
economics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(economics)


def test_ten_rps_continuous_month():
    result = economics.project(10)
    assert result["requests_per_30_day_month"] == 25920000
    assert result["monthly_usd"] == pytest.approx({"jev": 544.32, "luna": 1944, "sonnet": 38880})


def test_jev_output_is_free_and_different_prompts_supported():
    assert economics.project(1, jev_input=1000, llm_output=500)["monthly_usd"][
        "jev"
    ] == pytest.approx(108.864)
    assert economics.project(1, llm_output=0)["monthly_usd"]["jev"] == pytest.approx(54.432)


def test_request_and_token_limits_both_apply():
    assert economics.project(80)["within_published_jev_limits"]
    assert not economics.project(80.1)["within_published_jev_limits"]
    assert not economics.project(80, jev_input=2000)["within_published_jev_limits"]


def test_duty_cycle_changes_volume_not_peak_rate_limit():
    assert economics.project(10, hours_per_day=8)["monthly_usd"]["jev"] == pytest.approx(181.44)
    assert not economics.project(100, hours_per_day=1)["within_published_jev_limits"]


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_traffic_rejected(value):
    with pytest.raises(ValueError):
        economics.project(value)
