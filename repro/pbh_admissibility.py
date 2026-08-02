"""PBH certificates for admissible actuator and grouped-sensor selections."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal

import numpy as np


SelectionKind = Literal["stabilizability", "detectability"]


def _nonnegative_real(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a non-boolean real number")
    normalized = float(value)
    if not np.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return normalized


def _binary_vector(name: str, values: object, length: int) -> tuple[int, ...]:
    try:
        raw = tuple(values)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable") from error
    if len(raw) != length:
        raise ValueError(f"{name} must have length {length}")
    normalized: list[int] = []
    for index, value in enumerate(raw):
        if isinstance(value, (bool, np.bool_)):
            normalized.append(int(value))
        elif isinstance(value, Integral) and int(value) in (0, 1):
            normalized.append(int(value))
        else:
            raise TypeError(f"{name}[{index}] must be binary")
    return tuple(normalized)


def _state_matrix(A: object) -> np.ndarray:
    matrix = np.asarray(A, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("A must be a finite square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("A must be finite")
    return matrix


def _unstable_eigenvalues(
    A: np.ndarray,
    *,
    stability_radius: float,
    eigenvalue_tolerance: float,
) -> tuple[complex, ...]:
    candidates = tuple(
        complex(value)
        for value in np.linalg.eigvals(A)
        if abs(value) >= stability_radius - eigenvalue_tolerance
    )
    representatives: list[complex] = []
    for value in sorted(candidates, key=lambda item: (item.real, item.imag)):
        if any(
            abs(value - previous)
            <= eigenvalue_tolerance * max(1.0, abs(value), abs(previous))
            for previous in representatives
        ):
            continue
        representatives.append(value)
    return tuple(representatives)


@dataclass(frozen=True, slots=True)
class PBHModeEvidence:
    """Rank and singular-value evidence for one unstable PBH pencil."""

    eigenvalue_real: float
    eigenvalue_imag: float
    pencil_rows: int
    pencil_columns: int
    rank: int
    required_rank: int
    singular_values: tuple[float, ...]
    minimum_singular_value: float
    rank_tolerance: float

    @property
    def full_rank(self) -> bool:
        return self.rank == self.required_rank


@dataclass(frozen=True, slots=True)
class PBHSelectionCertificate:
    """Complete unstable-mode PBH certificate for one hardware pattern."""

    kind: SelectionKind
    selection: tuple[int, ...]
    selected_device_indices: tuple[int, ...]
    selected_matrix_indices: tuple[int, ...]
    stability_radius: float
    eigenvalue_tolerance: float
    rank_tolerance: float
    modes: tuple[PBHModeEvidence, ...]

    @property
    def admissible(self) -> bool:
        return all(mode.full_rank for mode in self.modes)

    @property
    def minimum_rank(self) -> int:
        return min((mode.rank for mode in self.modes), default=0)

    @property
    def worst_minimum_singular_value(self) -> float:
        return min(
            (mode.minimum_singular_value for mode in self.modes),
            default=float("inf"),
        )


def _mode_evidence(
    pencil: np.ndarray,
    eigenvalue: complex,
    *,
    required_rank: int,
    rank_tolerance: float,
) -> PBHModeEvidence:
    singular_values = tuple(
        float(value) for value in np.linalg.svd(pencil, compute_uv=False)
    )
    rank = sum(value > rank_tolerance for value in singular_values)
    return PBHModeEvidence(
        eigenvalue_real=float(eigenvalue.real),
        eigenvalue_imag=float(eigenvalue.imag),
        pencil_rows=pencil.shape[0],
        pencil_columns=pencil.shape[1],
        rank=rank,
        required_rank=required_rank,
        singular_values=singular_values,
        minimum_singular_value=singular_values[-1],
        rank_tolerance=rank_tolerance,
    )


def stabilizability_certificate(
    A: object,
    B: object,
    eta: object,
    *,
    stability_radius: float = 1.0,
    eigenvalue_tolerance: float = 1.0e-9,
    rank_tolerance: float = 1.0e-10,
) -> PBHSelectionCertificate:
    """Evaluate ``rank[lambda I-A, B_eta]=n`` at every unstable mode."""

    state_matrix = _state_matrix(A)
    input_matrix = np.asarray(B, dtype=float)
    if (
        input_matrix.ndim != 2
        or input_matrix.shape[0] != state_matrix.shape[0]
        or not np.all(np.isfinite(input_matrix))
    ):
        raise ValueError("B must be finite with shape (A.shape[0], actuator_count)")
    radius = _nonnegative_real("stability_radius", stability_radius)
    eigen_tolerance = _nonnegative_real(
        "eigenvalue_tolerance", eigenvalue_tolerance
    )
    rank_threshold = _nonnegative_real("rank_tolerance", rank_tolerance)
    selection = _binary_vector("eta", eta, input_matrix.shape[1])
    selected = tuple(index for index, value in enumerate(selection) if value)
    selected_matrix = input_matrix[:, selected] if selected else np.empty(
        (state_matrix.shape[0], 0), dtype=float
    )
    complex_A = np.asarray(state_matrix, dtype=np.complex128)
    identity = np.eye(state_matrix.shape[0], dtype=np.complex128)
    modes = tuple(
        _mode_evidence(
            np.hstack((eigenvalue * identity - complex_A, selected_matrix)),
            eigenvalue,
            required_rank=state_matrix.shape[0],
            rank_tolerance=rank_threshold,
        )
        for eigenvalue in _unstable_eigenvalues(
            state_matrix,
            stability_radius=radius,
            eigenvalue_tolerance=eigen_tolerance,
        )
    )
    return PBHSelectionCertificate(
        kind="stabilizability",
        selection=selection,
        selected_device_indices=selected,
        selected_matrix_indices=selected,
        stability_radius=radius,
        eigenvalue_tolerance=eigen_tolerance,
        rank_tolerance=rank_threshold,
        modes=modes,
    )


def detectability_certificate(
    A: object,
    C: object,
    sensor_groups: object,
    xi: object,
    *,
    stability_radius: float = 1.0,
    eigenvalue_tolerance: float = 1.0e-9,
    rank_tolerance: float = 1.0e-10,
) -> PBHSelectionCertificate:
    """Evaluate ``rank[lambda I-A; C_xi]=n`` at every unstable mode."""

    state_matrix = _state_matrix(A)
    output_matrix = np.asarray(C, dtype=float)
    if (
        output_matrix.ndim != 2
        or output_matrix.shape[1] != state_matrix.shape[0]
        or not np.all(np.isfinite(output_matrix))
    ):
        raise ValueError("C must be finite with shape (output_count, A.shape[0])")
    try:
        raw_groups = tuple(tuple(group) for group in sensor_groups)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError("sensor_groups must be a two-dimensional iterable") from error
    if not raw_groups or any(not group for group in raw_groups):
        raise ValueError("sensor_groups must contain nonempty groups")
    groups: list[tuple[int, ...]] = []
    for group_index, group in enumerate(raw_groups):
        normalized_group: list[int] = []
        for index, value in enumerate(group):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise TypeError(
                    f"sensor_groups[{group_index}][{index}] must be an integer"
                )
            normalized = int(value)
            if not 0 <= normalized < output_matrix.shape[0]:
                raise IndexError("sensor_groups contains an output index out of range")
            normalized_group.append(normalized)
        groups.append(tuple(normalized_group))
    flat = tuple(index for group in groups for index in group)
    if tuple(sorted(flat)) != tuple(range(output_matrix.shape[0])):
        raise ValueError("sensor_groups must partition the scalar rows of C")

    radius = _nonnegative_real("stability_radius", stability_radius)
    eigen_tolerance = _nonnegative_real(
        "eigenvalue_tolerance", eigenvalue_tolerance
    )
    rank_threshold = _nonnegative_real("rank_tolerance", rank_tolerance)
    selection = _binary_vector("xi", xi, len(groups))
    selected_devices = tuple(
        index for index, value in enumerate(selection) if value
    )
    selected_rows = tuple(
        row for device in selected_devices for row in groups[device]
    )
    selected_matrix = (
        output_matrix[selected_rows, :]
        if selected_rows
        else np.empty((0, state_matrix.shape[0]), dtype=float)
    )
    complex_A = np.asarray(state_matrix, dtype=np.complex128)
    identity = np.eye(state_matrix.shape[0], dtype=np.complex128)
    modes = tuple(
        _mode_evidence(
            np.vstack((eigenvalue * identity - complex_A, selected_matrix)),
            eigenvalue,
            required_rank=state_matrix.shape[0],
            rank_tolerance=rank_threshold,
        )
        for eigenvalue in _unstable_eigenvalues(
            state_matrix,
            stability_radius=radius,
            eigenvalue_tolerance=eigen_tolerance,
        )
    )
    return PBHSelectionCertificate(
        kind="detectability",
        selection=selection,
        selected_device_indices=selected_devices,
        selected_matrix_indices=selected_rows,
        stability_radius=radius,
        eigenvalue_tolerance=eigen_tolerance,
        rank_tolerance=rank_threshold,
        modes=modes,
    )


def enumerate_stabilizable_actuator_patterns(
    A: object,
    B: object,
    *,
    stability_radius: float = 1.0,
    eigenvalue_tolerance: float = 1.0e-9,
    rank_tolerance: float = 1.0e-10,
) -> tuple[PBHSelectionCertificate, ...]:
    """Exhaust all actuator subsets and return the admissible certificates."""

    input_matrix = np.asarray(B)
    if input_matrix.ndim != 2:
        raise ValueError("B must be two-dimensional")
    certificates = (
        stabilizability_certificate(
            A,
            B,
            tuple((mask >> index) & 1 for index in range(input_matrix.shape[1])),
            stability_radius=stability_radius,
            eigenvalue_tolerance=eigenvalue_tolerance,
            rank_tolerance=rank_tolerance,
        )
        for mask in range(1 << input_matrix.shape[1])
    )
    return tuple(item for item in certificates if item.admissible)


def enumerate_detectable_sensor_patterns(
    A: object,
    C: object,
    sensor_groups: object,
    *,
    stability_radius: float = 1.0,
    eigenvalue_tolerance: float = 1.0e-9,
    rank_tolerance: float = 1.0e-10,
) -> tuple[PBHSelectionCertificate, ...]:
    """Exhaust all grouped-sensor subsets and return admissible certificates."""

    try:
        groups = tuple(tuple(group) for group in sensor_groups)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError("sensor_groups must be a two-dimensional iterable") from error
    certificates = (
        detectability_certificate(
            A,
            C,
            groups,
            tuple((mask >> index) & 1 for index in range(len(groups))),
            stability_radius=stability_radius,
            eigenvalue_tolerance=eigenvalue_tolerance,
            rank_tolerance=rank_tolerance,
        )
        for mask in range(1 << len(groups))
    )
    return tuple(item for item in certificates if item.admissible)
