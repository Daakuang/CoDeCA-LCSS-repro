"""Exact finite-catalog hardware-aware QI certificate tests."""

from __future__ import annotations

import itertools

import pytest

from repro.baselines import (
    platoon_plant_propagation_delays,
    platoon_qi_compatible,
)
from repro.cases import make_joint_platoon_case, make_platoon_case
from repro.deployment import DirectedServiceMenu, ServiceCatalog
from repro.joint_qi import (
    finite_catalog_qi_no_goods,
    fixed_service_qi_certificate,
    grouped_plant_propagation_delays,
    hardware_aware_qi_compatible,
    hardware_mask,
    hardware_vectors,
    hosted_controller_delays,
    iter_budget_feasible_qi_deployments,
    masked_controller_delays,
    satisfies_qi_no_goods,
    service_menu_qi_certificates,
    solve_hardware_aware_qi_budget_envelope,
    summarize_joint_qi,
)
from repro.joint_platoon_experiment import joint_solver_options, make_joint_problem
from repro.platoon_experiment import iter_service_architectures


PLATOON_SENSOR_GROUPS = (
    (0,),
    (1, 2),
    (3,),
    (4, 5),
    (6,),
    (7, 8),
    (9,),
)
PLATOON_SENSOR_DEVICE_SITES = (0, 1, 1, 2, 2, 3, 3)


def _small_catalog() -> ServiceCatalog:
    return ServiceCatalog(
        site_count=2,
        directed_menus=(
            DirectedServiceMenu(0, 0, (0,), mandatory_delay=0),
            DirectedServiceMenu(0, 1, (1, 2)),
            DirectedServiceMenu(1, 0, (1, 2)),
            DirectedServiceMenu(1, 1, (0,), mandatory_delay=0),
        ),
    )


def _small_service_menus():
    for zero_from_one, one_from_zero in itertools.product(
        (1, 2, None), repeat=2
    ):
        yield (
            (0, zero_from_one),
            (one_from_zero, 0),
        )


def test_fixed_service_hypergraph_and_catalog_no_goods_are_exhaustive() -> None:
    catalog = _small_catalog()
    actuator_sites = (0, 1)
    sensor_sites = (0, 1)
    propagation = ((1, 2), (2, 1))
    no_goods = finite_catalog_qi_no_goods(
        catalog, actuator_sites, sensor_sites, propagation
    )

    for service_delays in _small_service_menus():
        controller = hosted_controller_delays(
            service_delays, actuator_sites, sensor_sites
        )
        certificate = fixed_service_qi_certificate(controller, propagation)
        for mask in range(16):
            eta, xi = hardware_vectors(
                mask, actuator_count=2, sensor_count=2
            )
            direct = hardware_aware_qi_compatible(
                controller, propagation, eta, xi
            )
            assert certificate.is_compatible(mask) is direct
            assert (
                satisfies_qi_no_goods(no_goods, eta, xi, service_delays)
                is direct
            )


def test_hardware_mask_round_trip_and_inactive_rows_and_columns() -> None:
    eta = (1, 0, 1)
    xi = (0, 1, 0, 1)
    mask = hardware_mask(eta, xi)
    assert hardware_vectors(mask, actuator_count=3, sensor_count=4) == (eta, xi)

    all_active = (
        (0, 1, 2, None),
        (1, 0, None, 2),
        (2, None, 0, 1),
    )
    assert masked_controller_delays(all_active, eta, xi) == (
        (None, 1, None, None),
        (None, None, None, None),
        (None, None, None, 1),
    )


def test_budget_iterator_covers_exact_qi_product_without_duplicates() -> None:
    propagation = ((1, 2), (2, 1))
    menus = tuple(_small_service_menus())
    certificates = service_menu_qi_certificates(
        menus, (0, 1), (0, 1), propagation
    )
    service_costs = tuple(
        sum(delay is not None for row in menu for delay in row) - 2
        for menu in menus
    )
    budget = 3
    deployments = tuple(
        iter_budget_feasible_qi_deployments(
            certificates,
            service_costs,
            (1, 1),
            (1, 1),
            budget_units=budget,
        )
    )
    observed = {(item.service_index, item.hardware_mask) for item in deployments}
    expected = set()
    for service_index, certificate in enumerate(certificates):
        for mask in range(16):
            eta, xi = hardware_vectors(mask, actuator_count=2, sensor_count=2)
            hardware_cost = sum(eta) + sum(xi)
            if (
                certificate.is_compatible(mask)
                and service_costs[service_index] + hardware_cost <= budget
            ):
                expected.add((service_index, mask))
    assert observed == expected
    assert len(observed) == len(deployments)


def test_seven_sensor_package_platoon_qi_enumeration_is_exact_and_stable() -> None:
    case = make_platoon_case()
    propagation = grouped_plant_propagation_delays(
        case.plant.A,
        case.plant.B2,
        case.plant.C2,
        PLATOON_SENSOR_GROUPS,
    )
    assert propagation == (
        (None, None, None),
        (1, None, None),
        (1, None, None),
        (1, 1, None),
        (None, 1, None),
        (None, 1, 1),
        (None, None, 1),
    )

    records = tuple(iter_service_architectures(case))
    actuator_sites = case.deployment.layout.actuator_sites
    certificates = service_menu_qi_certificates(
        (record.service_delays for record in records),
        actuator_sites,
        PLATOON_SENSOR_DEVICE_SITES,
        propagation,
    )
    summary = summarize_joint_qi(certificates)

    assert summary.service_menu_count == 8748
    assert summary.hardware_pattern_count == 1024
    assert summary.joint_architecture_count == 8_957_952
    assert summary.joint_qi_count == 6_739_452
    assert summary.distinct_forbidden_pattern_count == 686
    assert len(summary.all_active_qi_menu_indices) == 99

    old_propagation = platoon_plant_propagation_delays(case)
    old_indices = tuple(
        index
        for index, record in enumerate(records)
        if platoon_qi_compatible(
            case, record.service_delays, old_propagation
        )
    )
    assert summary.all_active_qi_menu_indices == old_indices

    no_goods = finite_catalog_qi_no_goods(
        case.deployment.services,
        actuator_sites,
        PLATOON_SENSOR_DEVICE_SITES,
        propagation,
    )
    assert len(no_goods) == 134
    service_indices = (0, 1, 98, 99, 4373, 8747)
    hardware_masks = (0, 1, 7, 31, 341, 682, 1023)
    for service_index, mask in itertools.product(service_indices, hardware_masks):
        eta, xi = hardware_vectors(mask, actuator_count=3, sensor_count=7)
        direct = certificates[service_index].is_compatible(mask)
        assert (
            satisfies_qi_no_goods(
                no_goods,
                eta,
                xi,
                records[service_index].service_delays,
            )
            is direct
        )


def test_joint_qi_budget_wrapper_adds_exact_cuts_and_certifies_winners() -> None:
    pytest.importorskip("gurobipy")
    case = make_joint_platoon_case(seed=23)
    problem = make_joint_problem(case, horizon=10)
    propagation = grouped_plant_propagation_delays(
        case.plant.A,
        case.plant.B2,
        case.plant.C2,
        case.sensor_device_groups,
    )
    points = solve_hardware_aware_qi_budget_envelope(
        problem,
        propagation,
        (14.0, 27.0),
        base_options=joint_solver_options(architecture_budget=14.0),
    )

    assert tuple(point.budget for point in points) == (14.0, 27.0)
    expected_objectives = (7.803325988273029, 4.987445982902829)
    for point, expected_objective in zip(points, expected_objectives, strict=True):
        result = point.result
        assert result.no_good_count == 134
        assert result.verified_qi is True
        assert result.certified_optimal
        assert result.solve.status == "OPTIMAL"
        assert result.solve.mip_gap == 0.0
        assert result.solve.eta == (1, 1, 1)
        assert result.solve.xi is not None
        assert result.solve.xi[1] == result.solve.xi[3] == result.solve.xi[5] == 1
        assert result.solve.architecture_cost == pytest.approx(point.budget)
        assert result.solve.performance_objective == pytest.approx(
            expected_objective, abs=1.0e-8
        )
