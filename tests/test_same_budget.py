"""Fair same-budget QI baseline selection tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import pytest

from repro.baselines import BaselineRecord, best_same_budget_qi
from repro.cases import make_platoon_case
from repro.deployment import ArchitectureChoice
from repro.platoon_experiment import platoon_dense_architecture


def _choice() -> ArchitectureChoice:
    case = make_platoon_case()
    return ArchitectureChoice(
        case.fixed_eta,
        case.fixed_xi,
        platoon_dense_architecture(case),
    )


def _record(
    key: str,
    cost: float,
    performance: float,
    *,
    qi: bool = True,
    feasible: bool = True,
) -> BaselineRecord:
    return BaselineRecord(key, _choice(), cost, performance, qi, feasible)


def test_same_budget_searches_entire_family_not_exact_cost_or_scalarized_path() -> None:
    records = (
        _record("scalarized-low", 1.0, 9.0),
        _record("global-best-below-budget", 4.0, 1.0),
        _record("exact-budget", 5.0, 3.0),
        _record("scalarized-high", 8.0, 0.5),
        _record("non-qi", 3.0, 0.1, qi=False),
        _record("infeasible", 2.0, 0.01, feasible=False),
        _record("infinite-performance", 2.5, math.inf),
    )

    result = best_same_budget_qi(records, budget=5.0)

    assert result.best_performance == 1.0
    assert tuple(record.key for record in result.argmin_records) == (
        "global-best-below-budget",
    )
    assert result.argmin_records[0].architecture_cost < result.budget
    assert result.eligible_count == 3
    with pytest.raises(FrozenInstanceError):
        result.budget = 8.0  # type: ignore[misc]


def test_same_budget_returns_all_ties_and_uses_declared_tolerances() -> None:
    records = (
        _record("b", 2.0 + 5.0e-10, 1.0 + 5.0e-10),
        _record("a", 2.0, 1.0),
        _record("outside", 2.0 + 2.0e-9, 0.1),
    )

    result = best_same_budget_qi(
        records,
        budget=2.0,
        budget_tolerance=1.0e-9,
        tie_tolerance=1.0e-9,
    )

    assert tuple(record.key for record in result.argmin_records) == ("a", "b")
    assert result.eligible_count == 2


def test_same_budget_rejects_duplicate_keys_even_when_one_record_is_ineligible() -> None:
    records = (
        _record("duplicate", 1.0, 2.0),
        _record("duplicate", 9.0, 0.1, qi=False),
    )

    with pytest.raises(ValueError, match="BaselineRecord.key.*unique.*duplicate"):
        best_same_budget_qi(records, budget=2.0)


def test_same_budget_empty_eligible_family_is_explicit() -> None:
    result = best_same_budget_qi(
        (_record("non-qi", 1.0, 1.0, qi=False),),
        budget=1.0,
    )
    assert math.isinf(result.best_performance)
    assert result.argmin_records == ()
    assert result.eligible_count == 0


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    (
        ({"budget": True}, TypeError, "budget"),
        ({"budget": math.nan}, ValueError, "budget"),
        ({"budget": 1.0, "budget_tolerance": -1.0}, ValueError, "budget_tolerance"),
        ({"budget": 1.0, "tie_tolerance": math.nan}, ValueError, "tie_tolerance"),
    ),
)
def test_same_budget_validates_budget_fields(
    kwargs: dict[str, object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        best_same_budget_qi((_record("a", 1.0, 1.0),), **kwargs)  # type: ignore[arg-type]


def test_baseline_record_rejects_nan_bool_and_wrong_flag_types() -> None:
    choice = _choice()
    with pytest.raises(ValueError, match="performance"):
        BaselineRecord("x", choice, 1.0, math.nan, True, True)
    with pytest.raises(TypeError, match="architecture_cost"):
        BaselineRecord("x", choice, True, 1.0, True, True)
    with pytest.raises(TypeError, match="qi_compatible"):
        BaselineRecord("x", choice, 1.0, 1.0, 1, True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="feasible"):
        BaselineRecord("x", choice, 1.0, 1.0, True, 1)  # type: ignore[arg-type]
