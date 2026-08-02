from dataclasses import FrozenInstanceError

import numpy as np
from numpy.testing import assert_allclose
import pytest

from repro.realization import (
    FIRResponses,
    RAW_PADDING_ATOL,
    R1_IDENTITY_ATOL,
    RealizationFilters,
    realization_filters,
    service_lag_for_raw,
)


def _numbered_responses(*, horizon: int = 4) -> FIRResponses:
    """Return raw FIR arrays whose first axis is the literal lag index."""

    n_beta, n_u, n_y = 2, 2, 3
    R = np.zeros((horizon + 1, n_beta, n_beta), dtype=float)
    M = np.zeros((horizon + 1, n_u, n_beta), dtype=float)
    N = np.zeros((horizon + 1, n_beta, n_y), dtype=float)
    L = np.zeros((horizon + 1, n_u, n_y), dtype=float)

    R[1] = np.eye(n_beta)
    for raw_lag in range(2, horizon + 1):
        R[raw_lag] = 100.0 * raw_lag + np.arange(n_beta**2).reshape(
            n_beta, n_beta
        )
    for raw_lag in range(1, horizon + 1):
        M[raw_lag] = 10.0 * raw_lag + np.arange(n_u * n_beta).reshape(
            n_u, n_beta
        )
        N[raw_lag] = 20.0 * raw_lag + np.arange(n_beta * n_y).reshape(
            n_beta, n_y
        )
    for raw_lag in range(horizon + 1):
        L[raw_lag] = 30.0 * raw_lag + np.arange(n_u * n_y).reshape(n_u, n_y)

    return FIRResponses(R=R, M=M, N=N, L=L)


def test_realization_filters_apply_the_exact_ofsls_lag_shifts() -> None:
    responses = _numbered_responses()

    filters = realization_filters(responses)

    assert filters.m.shape[0] == responses.M.shape[0] - 1
    assert filters.n.shape[0] == responses.N.shape[0] - 1
    assert filters.r_plus.shape[0] == responses.R.shape[0] - 2
    for q in range(filters.m.shape[0]):
        assert_allclose(filters.m[q], responses.M[q + 1])
        assert_allclose(filters.n[q], -responses.N[q + 1])
    for q in range(filters.r_plus.shape[0]):
        assert_allclose(filters.r_plus[q], -responses.R[q + 2])
    assert_allclose(filters.l, responses.L)


def test_r1_identity_is_not_a_dynamic_communication_filter_coefficient() -> None:
    responses = _numbered_responses()

    filters = realization_filters(responses)

    assert service_lag_for_raw("R", 1) is None
    assert_allclose(filters.r_plus[0], -responses.R[2])
    assert filters.r_plus.shape[0] == responses.R.shape[0] - 2


@pytest.mark.parametrize(
    ("block", "raw_lag", "expected_deadline_index", "expected_allowed"),
    [
        ("L", 0, 0, False),
        ("L", 1, 1, False),
        ("L", 2, 2, True),
        ("M", 1, 0, False),
        ("M", 2, 1, False),
        ("M", 3, 2, True),
        ("N", 1, 1, False),
        ("N", 2, 2, True),
        ("R", 2, 1, False),
        ("R", 3, 2, True),
    ],
)
def test_delay_two_masks_raw_coefficients_at_the_message_deadline(
    block: str,
    raw_lag: int,
    expected_deadline_index: int,
    expected_allowed: bool,
) -> None:
    deadline_index = service_lag_for_raw(block, raw_lag)  # type: ignore[arg-type]

    assert deadline_index == expected_deadline_index
    assert (deadline_index >= 2) is expected_allowed


@pytest.mark.parametrize(
    ("block", "raw_lag"),
    [("L", -1), ("M", 0), ("N", 0), ("R", 0)],
)
def test_service_lag_rejects_coefficients_outside_each_raw_block_domain(
    block: str, raw_lag: int
) -> None:
    with pytest.raises(ValueError, match="raw_lag"):
        service_lag_for_raw(block, raw_lag)  # type: ignore[arg-type]


def test_service_lag_rejects_unknown_response_blocks() -> None:
    with pytest.raises(ValueError, match="block"):
        service_lag_for_raw("X", 1)  # type: ignore[arg-type]


def test_fir_responses_validate_spatial_and_horizon_dimensions() -> None:
    responses = _numbered_responses()

    with pytest.raises(ValueError, match="three-dimensional"):
        FIRResponses(
            R=responses.R[1],
            M=responses.M,
            N=responses.N,
            L=responses.L,
        )
    with pytest.raises(ValueError, match="R.*square"):
        FIRResponses(
            R=np.zeros((5, 2, 3)),
            M=responses.M,
            N=responses.N,
            L=responses.L,
        )
    with pytest.raises(ValueError, match="at least.*lag 2"):
        FIRResponses(
            R=responses.R[:2],
            M=responses.M[:2],
            N=responses.N[:2],
            L=responses.L[:2],
        )
    with pytest.raises(ValueError, match="coefficient axes"):
        FIRResponses(
            R=responses.R,
            M=responses.M[:-1],
            N=responses.N,
            L=responses.L,
        )
    with pytest.raises(ValueError, match="M"):
        FIRResponses(
            R=responses.R,
            M=np.zeros((5, 2, 3)),
            N=responses.N,
            L=responses.L,
        )
    with pytest.raises(ValueError, match="N"):
        FIRResponses(
            R=responses.R,
            M=responses.M,
            N=np.zeros((5, 3, 3)),
            L=responses.L,
        )
    with pytest.raises(ValueError, match="L"):
        FIRResponses(
            R=responses.R,
            M=responses.M,
            N=responses.N,
            L=np.zeros((5, 3, 3)),
        )


def test_fir_responses_require_r1_identity_with_the_declared_tolerance() -> None:
    responses = _numbered_responses()
    within_tolerance = responses.R.copy()
    within_tolerance[1, 0, 0] += 0.5 * R1_IDENTITY_ATOL

    accepted = FIRResponses(
        R=within_tolerance,
        M=responses.M,
        N=responses.N,
        L=responses.L,
    )

    assert_allclose(accepted.R[1], within_tolerance[1])
    outside_tolerance = responses.R.copy()
    outside_tolerance[1, 0, 0] += 2.0 * R1_IDENTITY_ATOL
    with pytest.raises(ValueError, match=r"R\[1\].*identity"):
        FIRResponses(
            R=outside_tolerance,
            M=responses.M,
            N=responses.N,
            L=responses.L,
        )


def test_realization_filters_validate_spatial_dimensions() -> None:
    with pytest.raises(ValueError, match="m"):
        RealizationFilters(
            r_plus=np.zeros((3, 2, 2)),
            m=np.zeros((4, 2, 3)),
            n=np.zeros((4, 2, 3)),
            l=np.zeros((5, 2, 3)),
        )


def test_realization_filters_require_the_minimum_horizon_implied_by_t_ge_two() -> None:
    with pytest.raises(ValueError, match="at least.*T >= 2"):
        RealizationFilters(
            r_plus=np.zeros((0, 2, 2)),
            m=np.zeros((1, 2, 2)),
            n=np.zeros((1, 2, 3)),
            l=np.zeros((2, 2, 3)),
        )


@pytest.mark.parametrize(
    ("r_count", "m_count", "n_count", "l_count", "message"),
    [
        (3, 4, 3, 5, "m and n"),
        (3, 4, 4, 4, "l must contain one more"),
        (2, 4, 4, 5, "r_plus must contain one fewer"),
    ],
)
def test_realization_filters_validate_every_filter_length_relationship(
    r_count: int,
    m_count: int,
    n_count: int,
    l_count: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RealizationFilters(
            r_plus=np.zeros((r_count, 2, 2)),
            m=np.zeros((m_count, 2, 2)),
            n=np.zeros((n_count, 2, 3)),
            l=np.zeros((l_count, 2, 3)),
        )


@pytest.mark.parametrize("block", ["R", "M", "N", "L"])
@pytest.mark.parametrize(
    ("bad_value", "bad_dtype"),
    [(np.nan, float), (np.inf, float), (1.0 + 2.0j, complex)],
    ids=["nan", "infinity", "complex"],
)
def test_fir_responses_reject_nonfinite_or_complex_coefficients(
    block: str, bad_value: complex, bad_dtype: type[float] | type[complex]
) -> None:
    responses = _numbered_responses()
    arrays = {
        name: getattr(responses, name).copy() for name in ("R", "M", "N", "L")
    }
    arrays[block] = np.asarray(arrays[block], dtype=bad_dtype)
    bad_lag = 2 if block == "R" else (0 if block == "L" else 1)
    arrays[block][bad_lag, 0, 0] = bad_value

    with pytest.raises(ValueError, match=rf"^{block}.*finite real"):
        FIRResponses(**arrays)


@pytest.mark.parametrize("block", ["r_plus", "m", "n", "l"])
@pytest.mark.parametrize(
    ("bad_value", "bad_dtype"),
    [(np.nan, float), (np.inf, float), (1.0 + 2.0j, complex)],
    ids=["nan", "infinity", "complex"],
)
def test_realization_filters_reject_nonfinite_or_complex_coefficients(
    block: str, bad_value: complex, bad_dtype: type[float] | type[complex]
) -> None:
    filters = realization_filters(_numbered_responses())
    arrays = {
        name: getattr(filters, name).copy() for name in ("r_plus", "m", "n", "l")
    }
    arrays[block] = np.asarray(arrays[block], dtype=bad_dtype)
    arrays[block][0, 0, 0] = bad_value

    with pytest.raises(ValueError, match=rf"^{block}.*finite real"):
        RealizationFilters(**arrays)


@pytest.mark.parametrize("block", ["R", "M", "N"])
def test_strictly_proper_raw_padding_must_be_zero_within_absolute_tolerance(
    block: str,
) -> None:
    responses = _numbered_responses()
    arrays = {
        name: getattr(responses, name).copy() for name in ("R", "M", "N", "L")
    }
    arrays[block][0, 0, 0] = 0.5 * RAW_PADDING_ATOL

    accepted = FIRResponses(**arrays)

    assert getattr(accepted, block)[0, 0, 0] == pytest.approx(
        0.5 * RAW_PADDING_ATOL
    )
    arrays[block][0, 0, 0] = 2.0 * RAW_PADDING_ATOL
    with pytest.raises(ValueError, match=rf"{block}\[0\].*zero.*tolerance"):
        FIRResponses(**arrays)


@pytest.mark.parametrize(
    ("container_name", "field"),
    [
        ("responses", "R"),
        ("responses", "M"),
        ("responses", "N"),
        ("responses", "L"),
        ("filters", "r_plus"),
        ("filters", "m"),
        ("filters", "n"),
        ("filters", "l"),
    ],
)
def test_coefficient_buffers_cannot_reenable_writes(
    container_name: str, field: str
) -> None:
    responses = _numbered_responses()
    containers = {
        "responses": responses,
        "filters": realization_filters(responses),
    }
    coefficient = getattr(containers[container_name], field)

    assert isinstance(coefficient, np.ndarray)
    assert coefficient.dtype == np.dtype(np.float64)
    with pytest.raises(ValueError, match="WRITEABLE|writable"):
        coefficient.setflags(write=True)


@pytest.mark.parametrize("raw_lag", [True, False, 1.0, np.float64(1.0), "1"])
def test_service_lag_rejects_nonintegral_and_boolean_lag_types(
    raw_lag: object,
) -> None:
    with pytest.raises(TypeError, match="non-boolean integer"):
        service_lag_for_raw("L", raw_lag)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("block", "raw_lag", "expected"),
    [
        ("L", np.int64(2), 2),
        ("M", np.int32(3), 2),
        ("N", np.int64(2), 2),
        ("R", np.int32(3), 2),
        ("R", np.int64(1), None),
    ],
)
def test_service_lag_accepts_numpy_integers_and_normalizes_python_ints(
    block: str, raw_lag: np.integer, expected: int | None
) -> None:
    actual = service_lag_for_raw(block, raw_lag)  # type: ignore[arg-type]

    assert actual == expected
    if actual is not None:
        assert type(actual) is int


def test_response_and_filter_containers_own_read_only_coefficients() -> None:
    R = _numbered_responses().R.copy()
    M = _numbered_responses().M.copy()
    N = _numbered_responses().N.copy()
    L = _numbered_responses().L.copy()
    responses = FIRResponses(R=R, M=M, N=N, L=L)
    filters = realization_filters(responses)

    R[1, 0, 0] = 99.0

    assert responses.R[1, 0, 0] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="read-only"):
        responses.R[1, 0, 0] = 2.0
    with pytest.raises(ValueError, match="read-only"):
        filters.m[0, 0, 0] = 2.0
    with pytest.raises(FrozenInstanceError):
        responses.R = responses.R.copy()
    with pytest.raises(FrozenInstanceError):
        filters.m = filters.m.copy()
