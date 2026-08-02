"""Coefficient conventions for the deployment-aware OF-SLS realization.

Raw response arrays use their literal lag as the first-axis index.  Thus
``R[t]``, ``M[t]``, and ``N[t]`` store the strictly proper raw coefficients at
lags ``t >= 1`` (index-zero padding is zero within ``RAW_PADDING_ATOL``), while
``L[t]`` is valid from lag zero.  All four arrays have shape
``(T + 1, ..., ...)`` for a common ``T >= 2``.

The returned realization-filter arrays use a zero-based filter lag ``q``:

``m[q] = M[q + 1]``, ``n[q] = -N[q + 1]``, and
``r_plus[q] = -R[q + 2]``.

Consequently, the structural coefficient ``R[1] = I`` cancels in
``z(I - zR)`` and is never treated as a communication-filter coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Number
from typing import Final, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray: TypeAlias = NDArray[np.float64]
ResponseBlock: TypeAlias = Literal["L", "M", "N", "R"]

# R[1] is a fixed structural coefficient; this absolute tolerance admits only
# floating-point assembly noise.  No relative tolerance is used.
R1_IDENTITY_ATOL: Final[float] = 1.0e-9
# Index-zero padding represents absent strictly proper coefficients and should
# be exactly zero; this tolerance admits only array-assembly roundoff.
RAW_PADDING_ATOL: Final[float] = 1.0e-12
# Frequency-domain elimination is refused when the reciprocal 2-norm
# condition estimate falls below this documented numerical-safety threshold.
FREQUENCY_RCOND_MIN: Final[float] = 1.0e-12


def _owned_read_only_3d(name: str, values: ArrayLike) -> FloatArray:
    array = np.asarray(values)
    if array.ndim != 3:
        raise ValueError(f"{name} must be a three-dimensional coefficient array")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise ValueError(f"{name} must contain only finite real numeric coefficients")

    normalized = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(normalized)):
        raise ValueError(f"{name} must contain only finite real numeric coefficients")

    # ``setflags(write=False)`` on an owning ndarray is reversible.  Back the
    # normalized ndarray by immutable ``bytes`` instead, so callers cannot
    # later re-enable WRITEABLE while shape, dtype, and vectorized arithmetic
    # remain standard NumPy behavior.
    immutable_buffer = normalized.tobytes(order="C")
    owned = np.frombuffer(immutable_buffer, dtype=np.float64).reshape(
        normalized.shape
    )
    return owned


@dataclass(frozen=True, slots=True, eq=False)
class FIRResponses:
    """Raw finite-horizon OF-SLS responses indexed by literal lag.

    ``R``, ``M``, and ``N`` have index-zero padding so that array index and raw
    lag agree.  That padding must be zero within absolute tolerance
    ``RAW_PADDING_ATOL`` (with zero relative tolerance).  ``L[0]`` is a genuine
    direct-feedthrough coefficient.  All blocks must be finite and real.  The
    arrays are defensively copied onto immutable buffers.
    """

    R: FloatArray
    M: FloatArray
    N: FloatArray
    L: FloatArray

    def __post_init__(self) -> None:
        for name in ("R", "M", "N", "L"):
            object.__setattr__(
                self,
                name,
                _owned_read_only_3d(name, cast(ArrayLike, getattr(self, name))),
            )

        if self.R.shape[1] != self.R.shape[2]:
            raise ValueError("R coefficient matrices must be square")

        coefficient_counts = {
            self.R.shape[0], self.M.shape[0], self.N.shape[0], self.L.shape[0]
        }
        if len(coefficient_counts) != 1:
            raise ValueError("R, M, N, and L coefficient axes must have equal length")
        if self.R.shape[0] < 3:
            raise ValueError("raw FIR arrays must include at least coefficients through lag 2")

        n_beta = self.R.shape[1]
        if self.M.shape[2] != n_beta:
            raise ValueError("M column dimension must equal the R state dimension")
        if self.N.shape[1] != n_beta:
            raise ValueError("N row dimension must equal the R state dimension")

        n_u = self.M.shape[1]
        n_y = self.N.shape[2]
        if self.L.shape[1:] != (n_u, n_y):
            raise ValueError("L spatial dimensions must agree with M outputs and N inputs")

        for name in ("R", "M", "N"):
            block = cast(FloatArray, getattr(self, name))
            if not np.allclose(
                block[0], 0.0, rtol=0.0, atol=RAW_PADDING_ATOL
            ):
                raise ValueError(
                    f"{name}[0] padding must be zero within absolute tolerance "
                    f"{RAW_PADDING_ATOL:g}"
                )

        identity = np.eye(n_beta, dtype=np.float64)
        if not np.allclose(
            self.R[1], identity, rtol=0.0, atol=R1_IDENTITY_ATOL
        ):
            raise ValueError(
                f"R[1] must equal the identity within absolute tolerance "
                f"{R1_IDENTITY_ATOL:g}"
            )


@dataclass(frozen=True, slots=True, eq=False)
class RealizationFilters:
    """Filters in ``z beta = r_plus beta + n y``, ``u = m beta + l y``.

    Every first axis is zero-based filter lag.  All blocks must be finite and
    real.  The arrays are defensively copied onto immutable buffers.
    """

    r_plus: FloatArray
    m: FloatArray
    n: FloatArray
    l: FloatArray

    def __post_init__(self) -> None:
        for name in ("r_plus", "m", "n", "l"):
            object.__setattr__(
                self,
                name,
                _owned_read_only_3d(name, cast(ArrayLike, getattr(self, name))),
            )

        if self.r_plus.shape[1] != self.r_plus.shape[2]:
            raise ValueError("r_plus coefficient matrices must be square")

        if (
            self.r_plus.shape[0] < 1
            or self.m.shape[0] < 2
            or self.n.shape[0] < 2
            or self.l.shape[0] < 3
        ):
            raise ValueError(
                "realization filters must include at least the coefficients "
                "implied by T >= 2"
            )

        n_beta = self.r_plus.shape[1]
        if self.m.shape[2] != n_beta:
            raise ValueError("m column dimension must equal the r_plus state dimension")
        if self.n.shape[1] != n_beta:
            raise ValueError("n row dimension must equal the r_plus state dimension")

        n_u = self.m.shape[1]
        n_y = self.n.shape[2]
        if self.l.shape[1:] != (n_u, n_y):
            raise ValueError("l spatial dimensions must agree with m outputs and n inputs")

        if self.m.shape[0] != self.n.shape[0]:
            raise ValueError("m and n coefficient axes must have equal length")
        if self.l.shape[0] != self.m.shape[0] + 1:
            raise ValueError("l must contain one more coefficient than m and n")
        if self.r_plus.shape[0] + 1 != self.m.shape[0]:
            raise ValueError("r_plus must contain one fewer coefficient than m and n")


@dataclass(frozen=True, slots=True, eq=False)
class RealizationRollout:
    """Complete nonnegative-time trajectory returned by the beta realization.

    ``beta`` contains ``beta[0]`` through ``beta[T]`` and ``u`` contains
    ``u[0]`` through ``u[T-1]``.  Both arrays own immutable finite-real data.
    Negative-time beta values, when supplied to :func:`rollout_realization`,
    are initial conditions and are not repeated in this trajectory.
    """

    beta: FloatArray
    u: FloatArray

    def __post_init__(self) -> None:
        for name in ("beta", "u"):
            values = np.asarray(cast(ArrayLike, getattr(self, name)))
            if values.ndim != 2:
                raise ValueError(f"{name} trajectory must be two-dimensional")
            if not np.issubdtype(values.dtype, np.number) or np.issubdtype(
                values.dtype, np.complexfloating
            ):
                raise ValueError(
                    f"{name} trajectory must contain only finite real values"
                )
            normalized = np.asarray(values, dtype=np.float64)
            if not np.all(np.isfinite(normalized)):
                raise ValueError(
                    f"{name} trajectory must contain only finite real values"
                )
            immutable_buffer = normalized.tobytes(order="C")
            immutable = np.frombuffer(
                immutable_buffer, dtype=np.float64
            ).reshape(normalized.shape)
            object.__setattr__(self, name, immutable)

        if self.beta.shape[0] != self.u.shape[0] + 1:
            raise ValueError("beta trajectory must contain one more sample than u")


def realization_filters(responses: FIRResponses) -> RealizationFilters:
    """Convert raw responses to the corrected fixed-host realization filters.

    With ``q`` denoting the zero-based filter lag, the conversion implements
    ``z M``, ``-z N``, and ``z(I-zR)`` exactly after the fixed ``R[1]=I`` term
    cancels.  In particular, ``R[1]`` is excluded from ``r_plus``.
    """

    return RealizationFilters(
        r_plus=-responses.R[2:],
        m=responses.M[1:],
        n=-responses.N[1:],
        l=responses.L,
    )


def _frequency_z(z_value: object) -> complex:
    if isinstance(z_value, (bool, np.bool_)) or not isinstance(z_value, Number):
        raise ValueError("z_value must be a finite nonzero numeric scalar")
    try:
        normalized_z = complex(z_value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "z_value must be a finite nonzero numeric scalar"
        ) from error
    if normalized_z == 0.0 or not (
        np.isfinite(normalized_z.real) and np.isfinite(normalized_z.imag)
    ):
        raise ValueError("z_value must be a finite nonzero numeric scalar")
    return normalized_z


def _filter_at_z(
    name: str, coefficients: FloatArray, z_value: complex
) -> NDArray[np.complex128]:
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            lags = np.arange(coefficients.shape[0], dtype=np.int64)
            powers = np.power(np.complex128(z_value), -lags)
            evaluated = np.tensordot(powers, coefficients, axes=(0, 0))
    except (FloatingPointError, OverflowError, ZeroDivisionError) as error:
        raise ValueError(
            f"{name}(z) frequency evaluation produced non-finite values "
            f"at z={z_value!r}"
        ) from error
    evaluated = np.asarray(evaluated, dtype=np.complex128)
    if not np.all(np.isfinite(evaluated)):
        raise ValueError(
            f"{name}(z) frequency evaluation produced non-finite values "
            f"at z={z_value!r}"
        )
    return evaluated


def realization_controller_at_z(
    filters: RealizationFilters, z_value: complex
) -> NDArray[np.complex128]:
    """Eliminate beta and evaluate the implemented controller at ``z_value``.

    A filter ``f`` is evaluated using the convention
    ``f(z) = sum_q f[q] z^-q``.  The realization equation is
    ``z beta = r_plus(z) beta + n(z) y``; hence this function returns

    ``l(z) + m(z) @ (z I - r_plus(z))^-1 @ n(z)``.

    With :func:`realization_filters`, this is exactly
    ``L(z) - M(z) R(z)^-1 N(z)`` because the stored shifts satisfy
    ``m=zM``, ``n=-zN``, and ``r_plus=z(I-zR)``.  The evaluation point must be
    a finite, nonzero numeric scalar.  Elimination is refused when the
    reciprocal 2-norm condition estimate of ``z I - r_plus(z)`` is below
    ``FREQUENCY_RCOND_MIN``.
    """

    normalized_z = _frequency_z(z_value)
    r_plus_z = _filter_at_z("r_plus", filters.r_plus, normalized_z)
    m_z = _filter_at_z("m", filters.m, normalized_z)
    n_z = _filter_at_z("n", filters.n, normalized_z)
    l_z = _filter_at_z("l", filters.l, normalized_z)
    dynamic_matrix = normalized_z * np.eye(
        filters.r_plus.shape[1], dtype=np.complex128
    ) - r_plus_z
    if not np.all(np.isfinite(dynamic_matrix)):
        raise ValueError(
            "z I - r_plus(z) produced non-finite values "
            f"at z={normalized_z!r}"
        )

    try:
        condition_number = float(np.linalg.cond(dynamic_matrix, p=2))
    except (np.linalg.LinAlgError, FloatingPointError, OverflowError):
        condition_number = np.inf
    reciprocal_condition = (
        0.0
        if not np.isfinite(condition_number) or condition_number <= 0.0
        else 1.0 / condition_number
    )
    conditioning_diagnostic = (
        f"at z={normalized_z!r}: rcond={reciprocal_condition:.3e}, "
        f"FREQUENCY_RCOND_MIN={FREQUENCY_RCOND_MIN:.3e}"
    )
    if reciprocal_condition < FREQUENCY_RCOND_MIN:
        raise ValueError(
            "z I - r_plus(z) is singular or numerically ill-conditioned "
            f"{conditioning_diagnostic}"
        )
    try:
        beta_to_y = np.linalg.solve(dynamic_matrix, n_z)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "failed to solve z I - r_plus(z) despite condition check "
            f"{conditioning_diagnostic}"
        ) from error
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            controller = l_z + m_z @ beta_to_y
    except (FloatingPointError, OverflowError) as error:
        raise ValueError(
            f"controller frequency response is non-finite at z={normalized_z!r}"
        ) from error
    controller = np.asarray(controller, dtype=np.complex128)
    if not np.all(np.isfinite(controller)):
        raise ValueError(
            f"controller frequency response is non-finite at z={normalized_z!r}"
        )
    return controller


def _finite_real_2d(name: str, values: ArrayLike) -> FloatArray:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain only finite real values") from error
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional trajectory")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise ValueError(f"{name} must contain only finite real values")
    normalized = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(normalized)):
        raise ValueError(f"{name} must contain only finite real values")
    return normalized.copy()


def rollout_realization(
    filters: RealizationFilters,
    measurements: ArrayLike,
    *,
    initial_beta_history: ArrayLike | None = None,
) -> RealizationRollout:
    """Roll out the causal nodewise FIR beta realization with explicit history.

    Coefficients use ``f(z) = sum_q f[q] z^-q``.  Thus the time-domain
    realization is

    ``beta[t+1] = sum_q r_plus[q] beta[t-q] + sum_q n[q] y[t-q]``

    ``u[t] = sum_q m[q] beta[t-q] + sum_q l[q] y[t-q]``.

    ``measurements[t]`` is ``y[t]`` and negative-time measurements are zero.
    The optional ``initial_beta_history`` has shape ``(len(m), n_beta)`` and
    is ordered newest first: row ``q`` is ``beta[-q]``.  It therefore contains
    ``beta[0], beta[-1], ...``; if omitted, all are zero.  History is copied
    into a function-local buffer, no state is retained between calls, and the
    returned object contains the complete computed ``beta`` and ``u``
    trajectories.
    """

    y = _finite_real_2d("measurements", measurements)
    n_y = filters.n.shape[2]
    if y.shape[1] != n_y:
        raise ValueError(
            f"measurements input dimension must be {n_y}, got {y.shape[1]}"
        )

    n_beta = filters.r_plus.shape[1]
    history_count = filters.m.shape[0]
    expected_history_shape = (history_count, n_beta)
    if initial_beta_history is None:
        beta_history = np.zeros(expected_history_shape, dtype=np.float64)
    else:
        beta_history = _finite_real_2d(
            "initial_beta_history", initial_beta_history
        )
        if beta_history.shape != expected_history_shape:
            raise ValueError(
                "initial_beta_history must have shape "
                f"{expected_history_shape}, got {beta_history.shape}"
            )

    sample_count = y.shape[0]
    beta = np.empty((sample_count + 1, n_beta), dtype=np.float64)
    u = np.zeros((sample_count, filters.m.shape[1]), dtype=np.float64)
    beta[0] = beta_history[0]

    for time_index in range(sample_count):
        for lag, coefficient in enumerate(filters.m):
            u[time_index] += coefficient @ beta_history[lag]
        for lag, coefficient in enumerate(filters.l):
            source_index = time_index - lag
            if source_index >= 0:
                u[time_index] += coefficient @ y[source_index]

        beta_next = np.zeros(n_beta, dtype=np.float64)
        for lag, coefficient in enumerate(filters.r_plus):
            beta_next += coefficient @ beta_history[lag]
        for lag, coefficient in enumerate(filters.n):
            source_index = time_index - lag
            if source_index >= 0:
                beta_next += coefficient @ y[source_index]
        beta[time_index + 1] = beta_next
        beta_history = np.vstack((beta_next, beta_history[:-1]))

    return RealizationRollout(beta=beta, u=u)


def service_lag_for_raw(block: ResponseBlock, raw_lag: int) -> int | None:
    """Return the latest admissible physical-service delay for a raw term.

    This is a message-deadline index, not the coefficient index ``q`` of every
    realization filter.  Terms used to form ``u[t]`` must arrive by ``t``, so
    ``L[t] -> t`` and ``M[t] -> t-1``.  Terms used to form ``beta[t+1]`` may
    arrive one sample later, so ``N[t] -> t`` and ``R[t] -> t-1`` for
    ``t >= 2``.  The filter algebra remains ``m[q] = M[q+1]``,
    ``n[q] = -N[q+1]``, and ``r_plus[q] = -R[q+2]``.  ``R[1]`` returns
    ``None`` because it is a structural identity coefficient rather than a
    transmitted message.
    """

    if isinstance(raw_lag, (bool, np.bool_)) or not isinstance(raw_lag, Integral):
        raise TypeError("raw_lag must be a non-boolean integer")
    raw_lag = int(raw_lag)

    if block == "L":
        if raw_lag < 0:
            raise ValueError("raw_lag for L must be nonnegative")
        return raw_lag
    if block == "M":
        if raw_lag < 1:
            raise ValueError(f"raw_lag for {block} must be at least 1")
        return raw_lag - 1
    if block == "N":
        if raw_lag < 1:
            raise ValueError(f"raw_lag for {block} must be at least 1")
        return raw_lag
    if block == "R":
        if raw_lag < 1:
            raise ValueError("raw_lag for R must be at least 1")
        if raw_lag == 1:
            return None
        return raw_lag - 1
    raise ValueError(f"unknown response block {block!r}")
