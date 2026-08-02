"""QI criterion and finite-service baseline adapter tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import numpy as np
import pytest

from repro.baselines import (
    RFDPathPoint,
    canonical_service_templates,
    map_rfd_path_grid,
    platoon_controller_delays,
    platoon_plant_propagation_delays,
    platoon_qi_compatible,
    project_to_service_menu,
    qi_closure,
    qi_compatible,
    saturated_menu_qi_closure,
)
from repro.cases import make_platoon_case
from repro.deployment import ArchitectureChoice, architecture_cost
from repro.platoon_experiment import (
    iter_service_architectures,
    platoon_sparse_architecture,
)


def test_unsaturated_qi_criterion_uses_y_to_u_rows_and_handles_infinity() -> None:
    # D[i,j] is y_j -> u_i and P[r,l] is u_l -> y_r.
    plant = ((1, 2), (2, 1))
    qi_delays = ((0, 2), (2, 0))
    non_qi_delays = ((0, None), (None, 0))

    assert qi_compatible(qi_delays, plant)
    assert not qi_compatible(non_qi_delays, plant)

    # The violation is y_2 -> u_2 -> y_1 -> u_1: 0 + 2 + 0 < infinity.
    closure = qi_closure(non_qi_delays, plant)
    assert closure.delays == qi_delays
    assert closure.changed
    assert closure.iterations == 1
    assert qi_compatible(closure.delays, plant)
    assert qi_closure(closure.delays, plant).iterations == 0
    with pytest.raises(FrozenInstanceError):
        closure.iterations = 4  # type: ignore[misc]


def test_qi_inputs_reject_wrong_shapes_nan_bool_and_noninteger_delays() -> None:
    plant = ((1, 2), (2, 1))
    with pytest.raises(ValueError, match="compatible shapes"):
        qi_compatible(((0, 1, 2),), plant)
    with pytest.raises(ValueError, match="NaN"):
        qi_compatible(((0, math.nan), (2, 0)), plant)
    with pytest.raises(TypeError, match="non-boolean integer"):
        qi_compatible(((0, True), (2, 0)), plant)
    with pytest.raises(TypeError, match="non-boolean integer"):
        qi_compatible(((0, 1.5), (2, 0)), plant)


def test_delay_normalization_preserves_arbitrary_precision_integrals() -> None:
    beyond_exact_float = 2**53 + 1
    beyond_float_range = 10**400

    exact = qi_closure(
        ((0, None), (None, 0)),
        ((0, beyond_exact_float), (beyond_float_range, 0)),
    )

    assert exact.delays[0][1] == beyond_exact_float
    assert exact.delays[1][0] == beyond_float_range
    assert isinstance(exact.delays[0][1], int)
    assert isinstance(exact.delays[1][0], int)


def test_delay_normalization_accepts_integral_numeric_scalars_without_float_roundtrip() -> None:
    numpy_delay = np.int64(7)
    normalized = qi_closure(((numpy_delay,),), ((0,),)).delays
    integral_float = qi_closure(((1.0,),), ((0,),)).delays
    positive_infinity = qi_closure(((math.inf,),), ((0,),)).delays

    assert normalized == ((7,),)
    assert isinstance(normalized[0][0], int)
    assert integral_float == ((1,),)
    assert isinstance(integral_float[0][0], int)
    assert positive_infinity == ((None,),)


@pytest.mark.parametrize(
    ("bad_delay", "error"),
    (
        (1.5, TypeError),
        (math.nan, ValueError),
        (-1, ValueError),
        (-1.0, ValueError),
        (-math.inf, ValueError),
        (True, TypeError),
    ),
)
def test_delay_normalization_rejects_nonintegral_and_invalid_scalars(
    bad_delay: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        qi_closure(((bad_delay,),), ((0,),))


def test_platoon_propagation_direction_and_qi_count_match_submitted_family() -> None:
    case = make_platoon_case()
    propagation = platoon_plant_propagation_delays(case)

    # Rows are measured channels (leader,F1,F2,F3), columns are (u1,u2,u3).
    assert propagation == (
        (None, None, None),
        (1, None, None),
        (1, 1, None),
        (None, 1, 1),
    )
    architectures = tuple(iter_service_architectures(case))
    qi_count = sum(
        platoon_qi_compatible(case, architecture.service_delays, propagation)
        for architecture in architectures
    )
    assert len(architectures) == 8748
    assert qi_count == 99


def test_menu_projection_is_separate_from_exact_unsaturated_qi_closure() -> None:
    case = make_platoon_case()
    requested = (
        (2, 0, None, None),
        (None, None, 0, None),
        (None, None, None, 0),
    )
    projection = project_to_service_menu(case, requested, delay_policy="least_cost")

    assert projection.projected
    assert projection.controller_delays == requested
    assert projection.choice.service_delays[1][0] == 2
    assert projection.architecture_cost == pytest.approx(1.0e-3 / 3.0)
    case.deployment.validate_choice(projection.choice)

    faster = project_to_service_menu(case, requested, delay_policy="fastest")
    assert faster.choice.service_delays[1][0] == 1
    assert faster.controller_delays[0][0] == 1
    assert faster.architecture_cost == pytest.approx(1.0e-3 / 2.0)


def test_saturated_menu_repair_reports_projection_change_and_cost() -> None:
    case = make_platoon_case()
    raw_choice = ArchitectureChoice(
        case.fixed_eta,
        case.fixed_xi,
        platoon_sparse_architecture(case),
    )
    assert not platoon_qi_compatible(case, raw_choice)

    repair = saturated_menu_qi_closure(case, raw_choice)

    assert repair.projected
    assert repair.changed
    assert repair.qi_compatible
    assert platoon_qi_compatible(case, repair.repaired_choice)
    assert repair.repair_cost == pytest.approx(
        repair.repaired_cost - repair.raw_cost
    )
    assert repair.repair_cost >= -1.0e-15
    case.deployment.validate_choice(repair.repaired_choice)


def test_canonical_templates_have_declared_information_flows_and_exact_costs() -> None:
    case = make_platoon_case()
    templates = canonical_service_templates(case)

    assert tuple(templates) == (
        "dense",
        "submitted_sparse",
        "PF",
        "PLF",
        "BD",
        "BDL",
        "TPF",
        "TPLF",
    )
    pf = platoon_controller_delays(case, templates["PF"])
    assert pf == (
        (1, 0, None, None),
        (None, 1, 0, None),
        (None, None, 1, 0),
    )
    plf = platoon_controller_delays(case, templates["PLF"])
    assert plf[1][0] == 1 and plf[2][0] == 1
    bd = platoon_controller_delays(case, templates["BD"])
    assert bd[0][2] == 1 and bd[1][3] == 1
    tpf = platoon_controller_delays(case, templates["TPF"])
    assert tpf[2][1] == 2

    for choice in templates.values():
        case.deployment.validate_choice(choice)
        cost = architecture_cost(case.deployment, choice).total
        assert math.isfinite(cost) and cost >= 0.0


def test_rfd_adapter_requires_complete_grid_and_maps_raw_support_without_fabrication() -> None:
    case = make_platoon_case()
    local = (
        (False, True, False, False),
        (False, False, True, False),
        (False, False, False, True),
    )
    leader = (
        (True, True, False, False),
        (False, False, True, False),
        (False, False, False, True),
    )
    points = tuple(
        RFDPathPoint(regularization, threshold, leader if threshold == 0.1 else local)
        for regularization in (0.1, 1.0)
        for threshold in (0.01, 0.1)
    )

    mapped = map_rfd_path_grid(case, points, qi_repair=False)

    assert len(mapped) == 4
    selected = next(
        item
        for item in mapped
        if item.path_point.regularization == 0.1
        and item.path_point.threshold == 0.1
    )
    # An RFD support has no delay claim: least-cost mapping uses the slowest
    # physical tier that still provisions each requested active support.
    assert selected.choice.service_delays[1][0] == 2
    assert selected.qi_repaired is False

    with pytest.raises(ValueError, match="complete Cartesian grid"):
        map_rfd_path_grid(case, points[:-1])

    repaired = map_rfd_path_grid(case, points, qi_repair=True)
    assert all(item.qi_repaired for item in repaired)
    assert all(platoon_qi_compatible(case, item.choice) for item in repaired)


def test_rfd_path_point_validates_numeric_fields_and_support_shape() -> None:
    support = ((1, 0),)
    with pytest.raises(TypeError, match="regularization"):
        RFDPathPoint(True, 0.1, support)
    with pytest.raises(ValueError, match="threshold"):
        RFDPathPoint(1.0, math.nan, support)
    with pytest.raises(TypeError, match="binary"):
        RFDPathPoint(1.0, 0.1, ((0.5, 1),))
    with pytest.raises(ValueError, match="shape"):
        map_rfd_path_grid(
            make_platoon_case(),
            (RFDPathPoint(1.0, 0.1, support),),
        )
