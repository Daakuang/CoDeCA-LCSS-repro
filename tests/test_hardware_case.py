import hashlib

import numpy as np
from numpy.testing import assert_allclose
import pytest

from repro.cases import (
    hardware_case_matrix_sha256,
    make_hardware_selection_case,
    single_device_pbh_diagnostics,
)


def _controllability_rank(A: np.ndarray, b: np.ndarray) -> int:
    return int(np.linalg.matrix_rank(np.hstack((b, A @ b, A @ A @ b)), tol=1e-10))


def _observability_rank(A: np.ndarray, c: np.ndarray) -> int:
    return int(np.linalg.matrix_rank(np.vstack((c, c @ A, c @ A @ A)), tol=1e-10))


def _independent_pbh_statistics(
    A: np.ndarray, B2: np.ndarray, C2: np.ndarray, device: int
) -> tuple[int, int, float, float]:
    controllability_ranks: list[int] = []
    observability_ranks: list[int] = []
    controllability_conditions: list[float] = []
    observability_conditions: list[float] = []
    identity = np.eye(A.shape[0], dtype=np.complex128)
    complex_A = np.asarray(A, dtype=np.complex128)
    for eigenvalue in np.linalg.eigvals(A):
        control_pencil = np.hstack(
            (eigenvalue * identity - complex_A, B2[:, [device]])
        )
        observation_pencil = np.vstack(
            (eigenvalue * identity - complex_A, C2[[device], :])
        )
        controllability_ranks.append(
            int(np.linalg.matrix_rank(control_pencil, tol=1e-10))
        )
        observability_ranks.append(
            int(np.linalg.matrix_rank(observation_pencil, tol=1e-10))
        )
        controllability_conditions.append(float(np.linalg.cond(control_pencil)))
        observability_conditions.append(float(np.linalg.cond(observation_pencil)))
    return (
        min(controllability_ranks),
        min(observability_ranks),
        max(controllability_conditions),
        max(observability_conditions),
    )


def test_seed_23_chain_matches_frozen_legacy_provenance() -> None:
    case = make_hardware_selection_case(seed=23)
    plant = case.plant

    assert case.seed == 23
    assert case.provenance == "cat_sls.plants.make_chain_of_lqg"
    assert (plant.n, plant.m, plant.p, plant.nw, plant.nz) == (3, 3, 3, 6, 6)
    assert_allclose(
        plant.A,
        np.array(
            [
                [0.48, 0.07407259469380464, 0.0],
                [0.07297062263844285, 0.48, 0.06220152871067037],
                [0.0, 0.06188786905275792, 0.48],
            ]
        ),
        atol=0.0,
        rtol=1e-15,
    )
    assert_allclose(
        plant.B2,
        np.diag([0.9480018282001881, 0.9780185658947741, 0.880266870166061]),
        atol=0.0,
        rtol=1e-15,
    )
    assert hardware_case_matrix_sha256(case) == (
        "d889852e5ebb8d5b488943b776666f4637f331b8ffc1b6ec1f78d885f8364e62"
    )


def test_noise_and_regulated_output_channels_are_exact() -> None:
    plant = make_hardware_selection_case(seed=23).plant

    assert_allclose(plant.B1, np.hstack((np.eye(3), np.zeros((3, 3)))), atol=0.0)
    assert_allclose(plant.C2, np.eye(3), atol=0.0)
    assert_allclose(
        plant.D21, np.hstack((np.zeros((3, 3)), 0.20 * np.eye(3))), atol=0.0
    )
    assert_allclose(
        plant.C1, np.vstack((np.eye(3), np.zeros((3, 3)))), atol=0.0
    )
    assert_allclose(
        plant.D12, np.vstack((np.zeros((3, 3)), 0.50 * np.eye(3))), atol=0.0
    )
    assert_allclose(plant.D11, np.zeros((6, 6)), atol=0.0)
    assert np.max(np.abs(np.linalg.eigvals(plant.A))) == pytest.approx(
        0.5762010572653921, abs=1e-15
    )


def test_single_endpoint_devices_have_full_pbh_rank_but_middle_devices_do_not() -> None:
    plant = make_hardware_selection_case(seed=23).plant

    controllability = tuple(
        _controllability_rank(plant.A, plant.B2[:, [index]]) for index in range(3)
    )
    observability = tuple(
        _observability_rank(plant.A, plant.C2[[index], :]) for index in range(3)
    )

    assert controllability == (3, 2, 3)
    assert observability == (3, 2, 3)


def test_true_per_eigenvalue_pbh_pencils_are_frozen_and_reported_accurately() -> None:
    case = make_hardware_selection_case(seed=23)
    plant = case.plant
    independent = tuple(
        _independent_pbh_statistics(plant.A, plant.B2, plant.C2, device)
        for device in range(3)
    )
    reported = single_device_pbh_diagnostics(case)

    assert tuple(item[0] for item in independent) == (3, 2, 3)
    assert tuple(item[1] for item in independent) == (3, 2, 3)
    assert tuple(item[2] for item in independent) == pytest.approx(
        (16.15530443100607, 8.798734543113849e15, 18.499044226800542),
        rel=1e-12,
    )
    assert tuple(item[3] for item in independent) == pytest.approx(
        (16.846931063775035, 8.870030890816581e15, 21.126353238787644),
        rel=1e-12,
    )
    assert tuple(item.pbh_controllability_min_rank for item in reported) == (3, 2, 3)
    assert tuple(item.pbh_observability_min_rank for item in reported) == (3, 2, 3)
    for expected, actual in zip(independent, reported):
        assert actual.pbh_controllability_worst_condition == pytest.approx(
            expected[2], rel=1e-12
        )
        assert actual.pbh_observability_worst_condition == pytest.approx(
            expected[3], rel=1e-12
        )


def test_seed_drives_the_frozen_random_sequence_without_global_state() -> None:
    first = make_hardware_selection_case(seed=23)
    repeated = make_hardware_selection_case(seed=23)
    different = make_hardware_selection_case(seed=24)

    assert hardware_case_matrix_sha256(first) == hardware_case_matrix_sha256(repeated)
    assert hardware_case_matrix_sha256(first) != hardware_case_matrix_sha256(different)
    np.testing.assert_array_equal(first.plant.A, repeated.plant.A)
    np.testing.assert_array_equal(first.plant.B2, repeated.plant.B2)


def test_hardware_case_has_unit_device_costs_and_fixed_zero_delay_services() -> None:
    case = make_hardware_selection_case(seed=23)
    deployment = case.deployment

    assert deployment.layout.actuator_sites == (0, 1, 2)
    assert deployment.layout.sensor_sites == (0, 1, 2)
    assert deployment.layout.beta_sites == (0, 1, 2)
    assert deployment.layout.state_block_sizes == (1, 1, 1)
    assert_allclose(deployment.costs.actuator_costs, np.ones(3), atol=0.0)
    assert_allclose(deployment.costs.sensor_costs, np.ones(3), atol=0.0)
    assert case.fixed_service_delays == ((0, 0, 0),) * 3
    for destination in range(3):
        for source in range(3):
            menu = deployment.services.menu_for(destination=destination, source=source)
            assert menu is not None
            assert menu.finite_delays == (0,)
            assert menu.mandatory_delay == 0
            assert deployment.costs.service_costs_by_delay[0][destination, source] == 0.0
