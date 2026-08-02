"""Independent algebraic and time-domain checks for the beta realization."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from repro import realization as realization_module
from repro.realization import FIRResponses, RealizationFilters, realization_filters


def _exact_fir_fixture() -> FIRResponses:
    """Return a noncommuting fixture with an exactly invertible FIR ``R``."""

    R = np.zeros((4, 2, 2), dtype=float)
    R[1] = np.eye(2)
    R[2] = [[0.0, 0.2], [0.0, 0.0]]

    M = np.zeros((4, 2, 2), dtype=float)
    M[1] = [[0.60, -0.20], [0.10, 0.50]]
    M[2] = [[-0.10, 0.30], [0.20, -0.40]]
    M[3] = [[0.05, 0.00], [-0.15, 0.10]]

    N = np.zeros((4, 2, 2), dtype=float)
    N[1] = [[0.40, 0.10], [-0.30, 0.20]]
    N[2] = [[0.00, -0.20], [0.25, 0.05]]
    N[3] = [[-0.10, 0.15], [0.00, -0.20]]

    L = np.zeros((4, 2, 2), dtype=float)
    L[0] = [[0.20, -0.05], [0.00, 0.10]]
    L[1] = [[-0.10, 0.20], [0.15, 0.00]]
    L[2] = [[0.05, 0.10], [-0.05, 0.12]]
    L[3] = [[0.00, -0.04], [0.08, 0.03]]
    return FIRResponses(R=R, M=M, N=N, L=L)


def _evaluate_raw(coefficients: np.ndarray, z_value: complex) -> np.ndarray:
    """Independent literal-lag polynomial evaluation used only by the oracle."""

    evaluated = np.zeros(coefficients.shape[1:], dtype=np.complex128)
    for lag, coefficient in enumerate(coefficients):
        evaluated += coefficient * z_value ** (-lag)
    return evaluated


def test_realization_elimination_recovers_raw_controller_at_complex_z_points() -> None:
    responses = _exact_fir_fixture()
    filters = realization_filters(responses)
    z_points = (0.8 + 0.4j, 1.2 - 0.3j, -0.7 + 1.1j)

    for z_value in z_points:
        R_z = _evaluate_raw(responses.R, z_value)
        M_z = _evaluate_raw(responses.M, z_value)
        N_z = _evaluate_raw(responses.N, z_value)
        L_z = _evaluate_raw(responses.L, z_value)
        expected = L_z - M_z @ np.linalg.solve(R_z, N_z)

        actual = realization_module.realization_controller_at_z(filters, z_value)

        assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.parametrize(
    "z_value",
    [0.0, np.nan, np.inf, -np.inf, True, np.bool_(False), [1.0], "1.0"],
)
def test_frequency_evaluation_rejects_non_numeric_non_scalar_or_nonfinite_z(
    z_value: object,
) -> None:
    filters = realization_filters(_exact_fir_fixture())

    with pytest.raises(ValueError, match="finite nonzero numeric scalar"):
        realization_module.realization_controller_at_z(
            filters, z_value  # type: ignore[arg-type]
        )


def test_frequency_evaluation_converts_tiny_z_overflow_to_clear_value_error() -> None:
    filters = realization_filters(_exact_fir_fixture())
    tiny_z = np.finfo(np.float64).tiny

    with pytest.raises(ValueError, match=r"m\(z\).*non-finite"):
        realization_module.realization_controller_at_z(filters, tiny_z)


def test_frequency_evaluation_returns_finite_complex_matrix() -> None:
    filters = realization_filters(_exact_fir_fixture())

    actual = realization_module.realization_controller_at_z(filters, 0.9 + 0.2j)

    assert actual.dtype == np.dtype(np.complex128)
    assert np.all(np.isfinite(actual))


def _conditioned_frequency_filters(small_singular_value: float) -> RealizationFilters:
    r_plus = np.zeros((2, 2, 2), dtype=float)
    r_plus[0] = np.diag([1.0 - small_singular_value, 0.0])
    m = np.zeros((3, 2, 2), dtype=float)
    n = np.zeros((3, 2, 2), dtype=float)
    l = np.zeros((4, 2, 2), dtype=float)
    m[0] = np.eye(2)
    n[0] = np.eye(2)
    return RealizationFilters(r_plus=r_plus, m=m, n=n, l=l)


def test_frequency_rcond_threshold_is_public_and_documented() -> None:
    assert realization_module.FREQUENCY_RCOND_MIN == pytest.approx(1.0e-12)
    assert "FREQUENCY_RCOND_MIN" in (
        realization_module.realization_controller_at_z.__doc__ or ""
    )


@pytest.mark.parametrize("small_singular_value", [0.0, 1.0e-14])
def test_frequency_evaluation_rejects_singular_or_ill_conditioned_elimination(
    small_singular_value: float,
) -> None:
    filters = _conditioned_frequency_filters(small_singular_value)

    with pytest.raises(
        ValueError,
        match=r"z=.*rcond=.*FREQUENCY_RCOND_MIN",
    ):
        realization_module.realization_controller_at_z(filters, 1.0)


def _two_lag_frequency_fixture() -> FIRResponses:
    responses = _exact_fir_fixture()
    R = responses.R.copy()
    R[3] = [[0.05, -0.04], [0.10, -0.03]]
    return FIRResponses(
        R=R,
        M=responses.M,
        N=responses.N,
        L=responses.L,
    )


def test_frequency_recovery_uses_two_noncommuting_nonzero_r_plus_lags() -> None:
    responses = _two_lag_frequency_fixture()
    filters = realization_filters(responses)
    first_lag = filters.r_plus[0]
    second_lag = filters.r_plus[1]
    assert np.linalg.norm(first_lag) > 0.0
    assert np.linalg.norm(second_lag) > 0.0
    assert not np.allclose(first_lag @ second_lag, second_lag @ first_lag)

    for z_value in (0.85 + 0.35j, 1.15 - 0.25j):
        expected = _evaluate_raw(responses.L, z_value) - _evaluate_raw(
            responses.M, z_value
        ) @ np.linalg.solve(
            _evaluate_raw(responses.R, z_value),
            _evaluate_raw(responses.N, z_value),
        )

        actual = realization_module.realization_controller_at_z(filters, z_value)

        assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def _centralized_fir_coefficients(responses: FIRResponses) -> np.ndarray:
    """Independently expand ``K`` using the fixture's nilpotent ``R[2]``."""

    raw_horizon = responses.R.shape[0] - 1
    nilpotent_coefficient = responses.R[2]
    assert_allclose(nilpotent_coefficient @ nilpotent_coefficient, 0.0)
    controller = np.zeros(
        (2 * raw_horizon + 1, responses.M.shape[1], responses.N.shape[2]),
        dtype=float,
    )
    controller[: responses.L.shape[0]] = responses.L
    for m_lag in range(1, raw_horizon + 1):
        for n_lag in range(1, raw_horizon + 1):
            controller[m_lag + n_lag - 1] -= (
                responses.M[m_lag] @ responses.N[n_lag]
            )
            controller[m_lag + n_lag] += (
                responses.M[m_lag]
                @ nilpotent_coefficient
                @ responses.N[n_lag]
            )
    return controller


def _causal_convolution(coefficients: np.ndarray, inputs: np.ndarray) -> np.ndarray:
    """Independent zero-prehistory FIR convolution oracle."""

    outputs = np.zeros((inputs.shape[0], coefficients.shape[1]), dtype=float)
    for time_index in range(inputs.shape[0]):
        for lag, coefficient in enumerate(coefficients):
            source_index = time_index - lag
            if source_index >= 0:
                outputs[time_index] += coefficient @ inputs[source_index]
    return outputs


def test_nodewise_rollout_matches_independent_centralized_fir_controller() -> None:
    responses = _exact_fir_fixture()
    filters = realization_filters(responses)
    measurements = np.array(
        [
            [0.30, -0.10],
            [0.00, 0.20],
            [-0.40, 0.50],
            [0.10, 0.00],
            [0.25, -0.30],
            [-0.15, 0.40],
            [0.05, -0.20],
            [0.35, 0.10],
        ],
        dtype=float,
    )

    rollout = realization_module.rollout_realization(filters, measurements)
    expected_u = _causal_convolution(
        _centralized_fir_coefficients(responses), measurements
    )
    expected_beta = np.zeros((measurements.shape[0] + 1, 2), dtype=float)
    for time_index in range(1, expected_beta.shape[0]):
        expected_beta[time_index] -= responses.R[2] @ expected_beta[time_index - 1]
        for raw_lag in range(1, responses.N.shape[0]):
            source_index = time_index - raw_lag
            if source_index >= 0:
                expected_beta[time_index] -= (
                    responses.N[raw_lag] @ measurements[source_index]
                )

    assert rollout.beta.shape == (measurements.shape[0] + 1, 2)
    assert rollout.u.shape == (measurements.shape[0], 2)
    assert_allclose(rollout.beta, expected_beta, rtol=0.0, atol=1.0e-12)
    assert np.max(np.abs(rollout.u - expected_u)) < 1.0e-9


@pytest.mark.parametrize(
    ("measurements", "message"),
    [
        (np.zeros((3,)), "two-dimensional"),
        (np.zeros((3, 3)), "input dimension"),
        (np.array([[np.nan, 0.0]]), "finite real"),
        (np.array([[1.0 + 1.0j, 0.0]]), "finite real"),
    ],
)
def test_rollout_rejects_invalid_measurement_trajectories(
    measurements: np.ndarray, message: str
) -> None:
    filters = realization_filters(_exact_fir_fixture())

    with pytest.raises(ValueError, match=message):
        realization_module.rollout_realization(filters, measurements)


@pytest.mark.parametrize(
    ("initial_history", "message"),
    [
        (np.zeros((2, 2)), "initial_beta_history.*shape"),
        (np.full((3, 2), np.inf), "initial_beta_history.*finite real"),
        (np.full((3, 2), 1.0j), "initial_beta_history.*finite real"),
    ],
)
def test_rollout_rejects_invalid_initial_beta_history(
    initial_history: np.ndarray, message: str
) -> None:
    filters = realization_filters(_exact_fir_fixture())

    with pytest.raises(ValueError, match=message):
        realization_module.rollout_realization(
            filters,
            np.zeros((4, 2)),
            initial_beta_history=initial_history,
        )


def _history_sensitive_filters() -> RealizationFilters:
    r_plus = np.zeros((2, 1, 1), dtype=float)
    r_plus[:, 0, 0] = [2.0, 3.0]
    m = np.zeros((3, 1, 1), dtype=float)
    m[:, 0, 0] = [5.0, 7.0, 11.0]
    n = np.zeros((3, 1, 1), dtype=float)
    l = np.zeros((4, 1, 1), dtype=float)
    return RealizationFilters(r_plus=r_plus, m=m, n=n, l=l)


def test_rollout_history_is_newest_first_for_m_and_delayed_r_plus() -> None:
    filters = _history_sensitive_filters()
    initial_history = np.array([[1.0], [2.0], [4.0]])

    rollout = realization_module.rollout_realization(
        filters,
        np.zeros((2, 1)),
        initial_beta_history=initial_history,
    )

    assert_allclose(rollout.beta[:, 0], [1.0, 8.0, 19.0])
    assert_allclose(rollout.u[:, 0], [63.0, 69.0])


def test_rollout_handles_empty_and_single_sample_measurement_sequences() -> None:
    responses = _exact_fir_fixture()
    filters = realization_filters(responses)

    empty = realization_module.rollout_realization(filters, np.zeros((0, 2)))
    measurement = np.array([[0.3, -0.1]])
    single = realization_module.rollout_realization(filters, measurement)

    assert empty.beta.shape == (1, 2)
    assert empty.u.shape == (0, 2)
    assert_allclose(empty.beta, 0.0)
    assert_allclose(single.beta[1], -responses.N[1] @ measurement[0])
    assert_allclose(single.u[0], responses.L[0] @ measurement[0])


def test_rollout_owns_results_independently_of_mutated_inputs() -> None:
    filters = _history_sensitive_filters()
    measurements = np.zeros((2, 1))
    initial_history = np.array([[1.0], [2.0], [4.0]])
    rollout = realization_module.rollout_realization(
        filters,
        measurements,
        initial_beta_history=initial_history,
    )

    measurements[:] = 99.0
    initial_history[:] = 99.0

    assert_allclose(rollout.beta[:, 0], [1.0, 8.0, 19.0])
    assert_allclose(rollout.u[:, 0], [63.0, 69.0])


@pytest.mark.parametrize("field", ["beta", "u"])
def test_rollout_outputs_cannot_reenable_writes(field: str) -> None:
    rollout = realization_module.rollout_realization(
        _history_sensitive_filters(), np.zeros((2, 1))
    )
    trajectory = getattr(rollout, field)

    with pytest.raises(ValueError, match="WRITEABLE|writable"):
        trajectory.setflags(write=True)
