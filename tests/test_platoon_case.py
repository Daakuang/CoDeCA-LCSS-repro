"""Exact-data checks for the submitted three-follower platoon."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.signal import cont2discrete

from repro.cases import (
    PLATOON_ACTUATOR_LAG_SECONDS,
    PLATOON_HEADWAY_SECONDS,
    PLATOON_SAMPLE_TIME_SECONDS,
    make_platoon_case,
    platoon_case_matrix_sha256,
)


def test_submitted_physics_dimensions_and_zoh_are_frozen() -> None:
    case = make_platoon_case()
    plant = case.plant

    assert PLATOON_HEADWAY_SECONDS == pytest.approx(0.6)
    assert PLATOON_ACTUATOR_LAG_SECONDS == pytest.approx(0.25)
    assert PLATOON_SAMPLE_TIME_SECONDS == pytest.approx(0.1)
    assert (plant.n, plant.m, plant.p, plant.nw, plant.nz) == (9, 3, 10, 4, 11)
    assert case.measurement_channel_count == 4
    assert case.measurement_block_sizes == (1, 3, 3, 3)
    assert case.disturbance_labels == ("alpha_0", "d_1", "d_2", "d_3")

    local = np.array(
        [[0.0, 1.0, -0.6], [0.0, 0.0, -1.0], [0.0, 0.0, -4.0]]
    )
    coupling = np.zeros((3, 3))
    coupling[1, 2] = 1.0
    shift = np.diag(np.ones(2), k=-1)
    expected_a_c = np.kron(np.eye(3), local) + np.kron(shift, coupling)
    expected_b2_c = np.kron(np.eye(3), np.array([[0.0], [0.0], [4.0]]))
    expected_b1_c = np.zeros((9, 4))
    expected_b1_c[:3, 0] = [0.0, 1.0, 0.0]
    for follower in range(3):
        expected_b1_c[3 * follower + 2, follower + 1] = 4.0

    assert_allclose(case.continuous_A, expected_a_c, rtol=0.0, atol=0.0)
    assert_allclose(case.continuous_B1, expected_b1_c, rtol=0.0, atol=0.0)
    assert_allclose(case.continuous_B2, expected_b2_c, rtol=0.0, atol=0.0)
    augmented = np.hstack((expected_b1_c, expected_b2_c))
    expected_a, expected_b, _, _, _ = cont2discrete(
        (expected_a_c, augmented, np.eye(9), np.zeros((9, 7))),
        0.1,
        method="zoh",
    )
    assert_allclose(plant.A, expected_a, rtol=0.0, atol=2.0e-15)
    assert_allclose(plant.B1, expected_b[:, :4], rtol=0.0, atol=2.0e-15)
    assert_allclose(plant.B2, expected_b[:, 4:], rtol=0.0, atol=2.0e-15)


def test_submitted_generalized_performance_and_measurement_channels_are_exact() -> None:
    case = make_platoon_case()
    plant = case.plant

    expected_c1 = np.zeros((11, 9))
    expected_c1[0:3, 0:9:3] = np.sqrt(10.0) * np.eye(3)
    expected_c1[3, [1, 4]] = [-1.0, 1.0]
    expected_c1[4, [4, 7]] = [-1.0, 1.0]
    expected_c1[5:8, 2:9:3] = np.sqrt(0.5) * np.eye(3)
    expected_d12 = np.zeros((11, 3))
    expected_d12[8:, :] = np.sqrt(0.1) * np.eye(3)
    expected_c2 = np.vstack((np.zeros((1, 9)), np.eye(9)))
    expected_d21 = np.zeros((10, 4))
    expected_d21[0, 0] = 1.0

    assert_allclose(plant.C1, expected_c1, rtol=0.0, atol=0.0)
    assert_allclose(plant.D12, expected_d12, rtol=0.0, atol=0.0)
    assert_allclose(plant.C2, expected_c2, rtol=0.0, atol=0.0)
    assert_allclose(plant.D21, expected_d21, rtol=0.0, atol=0.0)
    assert_allclose(plant.D11, np.zeros((11, 4)), rtol=0.0, atol=0.0)

    layout = case.deployment.layout
    assert layout.site_count == 4
    assert layout.actuator_sites == (1, 2, 3)
    assert layout.sensor_sites == (0, 1, 1, 1, 2, 2, 2, 3, 3, 3)
    assert layout.beta_sites == (1, 2, 3)
    assert layout.state_block_sizes == (3, 3, 3)
    assert case.fixed_eta == (1, 1, 1)
    assert case.fixed_xi == (1,) * 10


def test_platoon_matrix_hash_is_stable_and_arrays_are_immutable() -> None:
    case = make_platoon_case()
    digest = hashlib.sha256()
    for matrix in (
        case.continuous_A,
        case.continuous_B1,
        case.continuous_B2,
        case.plant.A,
        case.plant.B1,
        case.plant.B2,
        case.plant.C1,
        case.plant.D11,
        case.plant.D12,
        case.plant.C2,
        case.plant.D21,
    ):
        digest.update(np.asarray(matrix, dtype="<f8").tobytes(order="C"))
        with pytest.raises(ValueError, match="WRITEABLE|writable"):
            matrix.setflags(write=True)
    expected_hash = "dbbb5f1118a6ffc32a12f8f1d5293c6584ee69c872ee8d45facbc092a1c81d1e"
    assert digest.hexdigest() == expected_hash
    assert platoon_case_matrix_sha256(case) == expected_hash


@pytest.mark.parametrize("imaginary_part", [0.0, 1.0])
def test_platoon_continuous_matrix_rejects_every_complex_dtype_without_warning(
    imaginary_part: float,
) -> None:
    case = make_platoon_case()
    invalid = np.asarray(case.continuous_A, dtype=np.complex128).copy()
    invalid[0, 0] += 1.0j * imaginary_part

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="continuous_A.*finite real matrix"):
            replace(case, continuous_A=invalid)


def test_platoon_continuous_matrix_rejects_nonnumeric_object_with_clear_error() -> None:
    case = make_platoon_case()
    invalid = np.asarray(case.continuous_A, dtype=object).copy()
    invalid[0, 0] = "not-a-number"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="continuous_A.*finite real matrix"):
            replace(case, continuous_A=invalid)
