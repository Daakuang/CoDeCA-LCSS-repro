"""Independent NumPy diagnostics for B+ OF-SLS solutions.

This module deliberately reconstructs every residual and reported scalar from
the returned arrays and decoded architecture.  It does not access Gurobi model
expressions or variables.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import cast

import numpy as np

from .deployment import ArchitectureChoice, ArchitectureCostBreakdown
from .model import BPlusProblem, BPlusSolveResult, GeneralizedPlant, SolverSettings
from .realization import FIRResponses, service_lag_for_raw


_SOLUTION_BEARING_STATUSES: tuple[str, ...] = (
    "OPTIMAL",
    "SUBOPTIMAL",
    "TIME_LIMIT",
    "NODE_LIMIT",
    "ITERATION_LIMIT",
    "SOLUTION_LIMIT",
    "INTERRUPTED",
    "USER_OBJ_LIMIT",
    "WORK_LIMIT",
    "MEM_LIMIT",
)


def _max_abs(array: np.ndarray) -> float:
    return 0.0 if array.size == 0 else float(np.max(np.abs(array)))


@dataclass(frozen=True, slots=True)
class OFSLSResidualDiagnostics:
    left_r_max: float
    left_n_max: float
    right_r_max: float
    right_m_max: float
    terminal_max: float
    max_residual: float


@dataclass(frozen=True, slots=True)
class ObjectiveDiagnostics:
    recomputed: float
    reported: float
    absolute_error: float


@dataclass(frozen=True, slots=True)
class HardwareDiagnostics:
    violation_count: int
    max_abs: float


@dataclass(frozen=True, slots=True)
class ServiceDiagnostics:
    decoded_delays: tuple[tuple[int | None, ...], ...]
    menu_valid: bool
    menu_errors: tuple[str, ...]
    availability_valid: bool
    violation_count: int
    max_abs: float


@dataclass(frozen=True, slots=True)
class ArchitectureCostDiagnostics:
    recomputed: float
    reported: float
    absolute_error: float
    breakdown_max_error: float
    breakdown: ArchitectureCostBreakdown


@dataclass(frozen=True, slots=True)
class SolutionAudit:
    ofsls: OFSLSResidualDiagnostics
    objective: ObjectiveDiagnostics
    total_objective: ObjectiveDiagnostics
    hardware: HardwareDiagnostics
    service: ServiceDiagnostics
    architecture_cost: ArchitectureCostDiagnostics
    budget_violation: float
    certified: bool


def recompute_ofsls_residuals(
    plant: GeneralizedPlant, responses: FIRResponses
) -> OFSLSResidualDiagnostics:
    """Recompute both OF-SLS identities, including all four terminal rows."""

    horizon = responses.R.shape[0] - 1
    left_r: list[float] = []
    left_n: list[float] = []
    right_r: list[float] = []
    right_m: list[float] = []
    terminal: list[float] = []
    zero_nn = np.zeros((plant.n, plant.n))
    zero_mn = np.zeros((plant.m, plant.n))
    zero_np = np.zeros((plant.n, plant.p))
    for t in range(horizon + 1):
        R_next = responses.R[t + 1] if t < horizon else zero_nn
        M_next = responses.M[t + 1] if t < horizon else zero_mn
        N_next = responses.N[t + 1] if t < horizon else zero_np
        impulse = np.eye(plant.n) if t == 0 else zero_nn
        residuals = (
            R_next - plant.A @ responses.R[t] - plant.B2 @ responses.M[t] - impulse,
            N_next - plant.A @ responses.N[t] - plant.B2 @ responses.L[t],
            R_next - responses.R[t] @ plant.A - responses.N[t] @ plant.C2 - impulse,
            M_next - responses.M[t] @ plant.A - responses.L[t] @ plant.C2,
        )
        maxima = tuple(_max_abs(residual) for residual in residuals)
        left_r.append(maxima[0])
        left_n.append(maxima[1])
        right_r.append(maxima[2])
        right_m.append(maxima[3])
        if t == horizon:
            terminal.extend(maxima)
    component_maxima = (
        max(left_r, default=0.0),
        max(left_n, default=0.0),
        max(right_r, default=0.0),
        max(right_m, default=0.0),
    )
    terminal_max = max(terminal, default=0.0)
    return OFSLSResidualDiagnostics(
        left_r_max=component_maxima[0],
        left_n_max=component_maxima[1],
        right_r_max=component_maxima[2],
        right_m_max=component_maxima[3],
        terminal_max=terminal_max,
        max_residual=max(*component_maxima, terminal_max),
    )


def recompute_h2_objective(
    plant: GeneralizedPlant, responses: FIRResponses
) -> float:
    """Recompute the squared finite-horizon H2 objective from impulse blocks."""

    terms: list[float] = []
    for t in range(responses.R.shape[0]):
        direct = plant.D11 if t == 0 else np.zeros_like(plant.D11)
        x_response = responses.R[t] @ plant.B1 + responses.N[t] @ plant.D21
        u_response = responses.M[t] @ plant.B1 + responses.L[t] @ plant.D21
        impulse = direct + plant.C1 @ x_response + plant.D12 @ u_response
        terms.append(float(np.sum(impulse * impulse)))
    return float(math.fsum(terms))


def diagnose_hardware_gating(
    responses: FIRResponses,
    eta: tuple[int, ...],
    xi: tuple[int, ...],
    *,
    sensor_groups: tuple[tuple[int, ...], ...] | None = None,
    tolerance: float,
) -> HardwareDiagnostics:
    if len(eta) != responses.M.shape[1]:
        raise ValueError("eta length must equal the response input dimension")
    scalar_sensor_count = responses.N.shape[2]
    groups = (
        tuple((index,) for index in range(scalar_sensor_count))
        if sensor_groups is None
        else tuple(tuple(group) for group in sensor_groups)
    )
    if len(xi) != len(groups):
        raise ValueError("xi length must equal the physical sensor-group count")
    flattened = tuple(index for group in groups for index in group)
    if tuple(sorted(flattened)) != tuple(range(scalar_sensor_count)):
        raise ValueError("sensor_groups must cover every scalar output exactly once")
    mask_m = np.zeros(responses.M.shape, dtype=bool)
    mask_n = np.zeros(responses.N.shape, dtype=bool)
    mask_l = np.zeros(responses.L.shape, dtype=bool)
    for actuator, enabled in enumerate(eta):
        if enabled == 0:
            mask_m[:, actuator, :] = True
            mask_l[:, actuator, :] = True
    for sensor_device, enabled in enumerate(xi):
        if enabled == 0:
            for sensor in groups[sensor_device]:
                mask_n[:, :, sensor] = True
                mask_l[:, :, sensor] = True
    forbidden = (
        responses.M[mask_m],
        responses.N[mask_n],
        responses.L[mask_l],
    )
    values = np.concatenate(forbidden) if any(block.size for block in forbidden) else np.empty(0)
    absolute = np.abs(values)
    return HardwareDiagnostics(
        violation_count=int(np.count_nonzero(absolute > tolerance)),
        max_abs=_max_abs(values),
    )


def _beta_blocks(problem: BPlusProblem) -> tuple[int, ...]:
    blocks: list[int] = []
    for block, size in enumerate(problem.deployment.layout.state_block_sizes):
        blocks.extend(block for _ in range(size))
    return tuple(blocks)


def _forbidden_by_delay(delay: int | None, deadline_index: int) -> bool:
    return delay is None or deadline_index < delay


def _validated_service_data(
    problem: BPlusProblem,
    service_delays: object,
    service_availability: object,
) -> tuple[
    tuple[tuple[int | None, ...], ...],
    tuple[tuple[tuple[int, ...], ...], ...],
]:
    """Normalize decoded services or raise contextual ``ValueError``.

    Availability is stored on the global finite latency grid
    ``q = 0, ..., max(menu delay)``.  Omitted services and the selected off
    item therefore have an all-zero availability sequence.
    """

    site_count = problem.deployment.layout.site_count
    try:
        choice = ArchitectureChoice(
            eta=tuple(1 for _ in range(problem.plant.m)),
            xi=tuple(
                1 for _ in range(problem.deployment.layout.sensor_device_count)
            ),
            service_delays=cast(object, service_delays),
        )
        problem.deployment.validate_choice(choice)
    except (TypeError, ValueError) as error:
        message = str(error)
        if "shape" in message:
            raise ValueError(
                "service_delays shape must equal (site_count, site_count): "
                f"{message}"
            ) from error
        raise ValueError(f"service_delays are invalid: {message}") from error
    delays = choice.service_delays
    if len(delays) != site_count or any(len(row) != site_count for row in delays):
        raise ValueError(
            "service_delays shape must equal (site_count, site_count)"
        )

    maximum_delay = max(
        (
            delay
            for menu in problem.deployment.services.directed_menus
            for delay in menu.finite_delays
        ),
        default=0,
    )
    lag_count = maximum_delay + 1
    try:
        availability_rows = tuple(
            tuple(tuple(sequence) for sequence in row)
            for row in cast(object, service_availability)  # type: ignore[arg-type]
        )
    except TypeError as error:
        raise ValueError(
            "service_availability shape must equal "
            f"({site_count}, {site_count}, {lag_count})"
        ) from error
    if len(availability_rows) != site_count or any(
        len(row) != site_count for row in availability_rows
    ) or any(
        len(sequence) != lag_count
        for row in availability_rows
        for sequence in row
    ):
        raise ValueError(
            "service_availability shape must equal "
            f"({site_count}, {site_count}, {lag_count})"
        )

    normalized_rows: list[tuple[tuple[int, ...], ...]] = []
    for destination, row in enumerate(availability_rows):
        normalized_row: list[tuple[int, ...]] = []
        for source, sequence in enumerate(row):
            normalized_sequence: list[int] = []
            for lag, value in enumerate(sequence):
                if isinstance(value, (bool, np.bool_)):
                    normalized = int(value)
                elif isinstance(value, Integral) and int(value) in (0, 1):
                    normalized = int(value)
                else:
                    raise ValueError(
                        "service_availability entries must be binary; "
                        f"entry [{destination}][{source}][{lag}]={value!r}"
                    )
                normalized_sequence.append(normalized)
            selected_delay = delays[destination][source]
            expected = tuple(
                int(selected_delay is not None and selected_delay <= lag)
                for lag in range(lag_count)
            )
            if tuple(normalized_sequence) != expected:
                raise ValueError(
                    "service_availability is inconsistent with service_delays "
                    f"for destination {destination} <- source {source}: "
                    f"observed {tuple(normalized_sequence)}, expected {expected}"
                )
            normalized_row.append(tuple(normalized_sequence))
        normalized_rows.append(tuple(normalized_row))
    return delays, tuple(normalized_rows)


def diagnose_service_gating(
    problem: BPlusProblem,
    responses: FIRResponses,
    service_delays: tuple[tuple[int | None, ...], ...],
    service_availability: tuple[tuple[tuple[int, ...], ...], ...],
    *,
    tolerance: float,
) -> ServiceDiagnostics:
    """Audit shifted support; malformed decoded data raise ``ValueError``."""

    normalized_delays, _ = _validated_service_data(
        problem, service_delays, service_availability
    )

    layout = problem.deployment.layout
    beta_blocks = _beta_blocks(problem)
    forbidden_values: list[float] = []
    horizon = problem.horizon
    for t in range(horizon + 1):
        deadline_index = service_lag_for_raw("L", t)
        if deadline_index is None:
            raise RuntimeError("L coefficients must have a message deadline")
        for actuator in range(problem.plant.m):
            for sensor in range(problem.plant.p):
                delay = normalized_delays[layout.p_u(actuator)][layout.p_y(sensor)]
                if _forbidden_by_delay(delay, deadline_index):
                    forbidden_values.append(float(responses.L[t, actuator, sensor]))
    for t in range(1, horizon + 1):
        m_deadline = service_lag_for_raw("M", t)
        n_deadline = service_lag_for_raw("N", t)
        if m_deadline is None or n_deadline is None:
            raise RuntimeError("M and N coefficients must have message deadlines")
        for actuator in range(problem.plant.m):
            for state in range(problem.plant.n):
                delay = normalized_delays[layout.p_u(actuator)][
                    layout.beta_host(beta_blocks[state])
                ]
                if _forbidden_by_delay(delay, m_deadline):
                    forbidden_values.append(float(responses.M[t, actuator, state]))
        for state in range(problem.plant.n):
            for sensor in range(problem.plant.p):
                delay = normalized_delays[layout.beta_host(beta_blocks[state])][
                    layout.p_y(sensor)
                ]
                if _forbidden_by_delay(delay, n_deadline):
                    forbidden_values.append(float(responses.N[t, state, sensor]))
    for t in range(1, horizon + 1):
        deadline_index = service_lag_for_raw("R", t)
        if deadline_index is None:
            # R[1] is structural rather than a transmitted message.
            continue
        for row in range(problem.plant.n):
            for column in range(problem.plant.n):
                delay = normalized_delays[layout.beta_host(beta_blocks[row])][
                    layout.beta_host(beta_blocks[column])
                ]
                if _forbidden_by_delay(delay, deadline_index):
                    forbidden_values.append(float(responses.R[t, row, column]))
    values = np.asarray(forbidden_values, dtype=float)
    absolute = np.abs(values)
    return ServiceDiagnostics(
        decoded_delays=normalized_delays,
        menu_valid=True,
        menu_errors=(),
        availability_valid=True,
        violation_count=int(np.count_nonzero(absolute > tolerance)),
        max_abs=_max_abs(values),
    )


def recompute_architecture_cost(
    problem: BPlusProblem,
    eta: tuple[int, ...],
    xi: tuple[int, ...],
    service_delays: tuple[tuple[int | None, ...], ...],
) -> ArchitectureCostBreakdown:
    """Recompute explicit hardware and once-per-service deployment cost."""

    if len(eta) != problem.plant.m:
        raise ValueError("eta length must equal the plant input dimension")
    sensor_device_count = problem.deployment.layout.sensor_device_count
    if len(xi) != sensor_device_count:
        raise ValueError("xi length must equal the physical sensor-device count")
    actuator = math.fsum(
        float(problem.deployment.costs.actuator_costs[index])
        for index in range(problem.plant.m)
        if eta[index] == 1
    )
    sensor = math.fsum(
        float(problem.deployment.costs.sensor_costs[index])
        for index in range(sensor_device_count)
        if xi[index] == 1
    )
    service = math.fsum(
        float(problem.deployment.costs.service_costs_by_delay[delay][destination, source])
        for destination, row in enumerate(service_delays)
        for source, delay in enumerate(row)
        if delay is not None
    )
    return ArchitectureCostBreakdown(
        total=math.fsum((actuator, sensor, service)),
        actuator=actuator,
        sensor=sensor,
        service=service,
    )


def _validated_result_binaries(
    name: str, values: object, expected_length: int
) -> tuple[int, ...]:
    try:
        candidates = tuple(cast(object, values))  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(f"{name} must be an iterable of binary values") from error
    if len(candidates) != expected_length:
        raise ValueError(
            f"{name} length must equal {expected_length}, got {len(candidates)}"
        )
    normalized: list[int] = []
    for index, value in enumerate(candidates):
        if isinstance(value, (bool, np.bool_)):
            normalized.append(int(value))
        elif isinstance(value, Integral) and int(value) in (0, 1):
            normalized.append(int(value))
        else:
            raise ValueError(f"{name}[{index}] must be binary, got {value!r}")
    return tuple(normalized)


def _validate_responses(problem: BPlusProblem, responses: object, tolerance: float) -> FIRResponses:
    if not isinstance(responses, FIRResponses):
        raise ValueError("responses must be a validated FIRResponses container")
    expected_shapes = {
        "R": (problem.horizon + 1, problem.plant.n, problem.plant.n),
        "M": (problem.horizon + 1, problem.plant.m, problem.plant.n),
        "N": (problem.horizon + 1, problem.plant.n, problem.plant.p),
        "L": (problem.horizon + 1, problem.plant.m, problem.plant.p),
    }
    for name, expected in expected_shapes.items():
        observed = getattr(responses, name).shape
        if observed != expected:
            raise ValueError(
                "responses horizon/spatial shape mismatch: "
                f"{name} has shape {observed}, expected {expected}"
            )
    for index, fix in enumerate(problem.response_fixes):
        observed = float(getattr(responses, fix.block)[fix.lag, fix.row, fix.column])
        if abs(observed - fix.value) > tolerance:
            raise ValueError(
                "responses violate problem.response_fixes: "
                f"fix {index} requires {fix.block}[{fix.lag}][{fix.row},"
                f"{fix.column}]={fix.value}, observed {observed}"
            )
    return responses


def _reported_scalar(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        normalized = float(cast(object, value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real scalar") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be a finite real scalar")
    return normalized


def _validate_settings(settings: object) -> SolverSettings:
    if not isinstance(settings, SolverSettings):
        raise ValueError("settings must be a SolverSettings container")
    weight = _reported_scalar("settings.architecture_weight", settings.architecture_weight)
    if weight < 0.0:
        raise ValueError("settings.architecture_weight must be nonnegative")
    if settings.architecture_budget is not None:
        budget = _reported_scalar(
            "settings.architecture_budget", settings.architecture_budget
        )
        if budget < 0.0:
            raise ValueError("settings.architecture_budget must be nonnegative")
        if weight != 0.0:
            raise ValueError(
                "settings cannot combine architecture_weight with architecture_budget"
            )
    for name in ("hardware_budget", "service_budget"):
        value = getattr(settings, name)
        if value is not None:
            budget = _reported_scalar(f"settings.{name}", value)
            if budget < 0.0:
                raise ValueError(f"settings.{name} must be nonnegative")
    if settings.architecture_budget is not None and (
        settings.hardware_budget is not None or settings.service_budget is not None
    ):
        raise ValueError(
            "settings cannot combine architecture_budget with separate budget caps"
        )
    return settings


def _validate_solution_metadata(result: BPlusSolveResult) -> None:
    count = result.solution_count
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral):
        raise ValueError("result.solution_count must be a non-boolean integer")
    if int(count) <= 0:
        raise ValueError(
            "result.solution_count must be positive for a solution-bearing audit"
        )
    if not isinstance(result.status, str) or result.status not in _SOLUTION_BEARING_STATUSES:
        raise ValueError(
            "result.status is not solution-bearing for audit: "
            f"{result.status!r}"
        )


def audit_solution(
    problem: BPlusProblem,
    result: BPlusSolveResult,
    *,
    tolerance: float = 1.0e-8,
) -> SolutionAudit:
    """Return a structured independent audit for a solution-bearing result."""

    if not isinstance(problem, BPlusProblem):
        raise TypeError("problem must be a BPlusProblem")
    if not isinstance(result, BPlusSolveResult):
        raise TypeError("result must be a BPlusSolveResult")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")
    _validate_solution_metadata(result)
    if (
        result.responses is None
        or result.eta is None
        or result.xi is None
        or result.service_delays is None
        or result.service_availability is None
        or result.performance_objective is None
        or result.architecture_cost is None
        or result.architecture_breakdown is None
    ):
        raise ValueError("audit_solution requires a solution-bearing result")
    if result.objective_value is None:
        raise ValueError("result.objective_value is required for a certified audit")

    responses = _validate_responses(problem, result.responses, tolerance)
    eta = _validated_result_binaries("eta", result.eta, problem.plant.m)
    xi = _validated_result_binaries(
        "xi", result.xi, problem.deployment.layout.sensor_device_count
    )
    service_delays, service_availability = _validated_service_data(
        problem, result.service_delays, result.service_availability
    )
    try:
        decoded_choice = ArchitectureChoice(
            eta=eta, xi=xi, service_delays=service_delays
        )
        problem.deployment.validate_choice(decoded_choice)
    except (TypeError, ValueError) as error:
        raise ValueError(f"decoded architecture is invalid: {error}") from error
    if problem.fixed_eta is not None and eta != problem.fixed_eta:
        raise ValueError(
            f"eta disagrees with problem.fixed_eta: {eta} != {problem.fixed_eta}"
        )
    if problem.fixed_xi is not None and xi != problem.fixed_xi:
        raise ValueError(
            f"xi disagrees with problem.fixed_xi: {xi} != {problem.fixed_xi}"
        )
    if (
        problem.fixed_service_delays is not None
        and service_delays != problem.fixed_service_delays
    ):
        raise ValueError(
            "service_delays disagree with problem.fixed_service_delays: "
            f"{service_delays} != {problem.fixed_service_delays}"
        )

    settings = _validate_settings(result.settings)
    reported_performance = _reported_scalar(
        "result.performance_objective", result.performance_objective
    )
    reported_total = _reported_scalar("result.objective_value", result.objective_value)
    reported_cost = _reported_scalar(
        "result.architecture_cost", result.architecture_cost
    )
    if not isinstance(result.architecture_breakdown, ArchitectureCostBreakdown):
        raise ValueError(
            "result.architecture_breakdown must be an ArchitectureCostBreakdown"
        )
    reported_breakdown = result.architecture_breakdown
    for field_name in ("total", "actuator", "sensor", "service"):
        _reported_scalar(
            f"result.architecture_breakdown.{field_name}",
            getattr(reported_breakdown, field_name),
        )

    ofsls = recompute_ofsls_residuals(problem.plant, responses)
    objective = recompute_h2_objective(problem.plant, responses)
    hardware = diagnose_hardware_gating(
        responses,
        eta,
        xi,
        sensor_groups=problem.deployment.layout.sensor_groups,
        tolerance=tolerance,
    )
    service = diagnose_service_gating(
        problem,
        responses,
        service_delays,
        service_availability,
        tolerance=tolerance,
    )
    breakdown = recompute_architecture_cost(problem, eta, xi, service_delays)
    breakdown_max_error = max(
        abs(breakdown.total - reported_breakdown.total),
        abs(breakdown.actuator - reported_breakdown.actuator),
        abs(breakdown.sensor - reported_breakdown.sensor),
        abs(breakdown.service - reported_breakdown.service),
    )
    expected_total = objective + settings.architecture_weight * breakdown.total
    budget_violation = max(
        0.0,
        0.0
        if settings.architecture_budget is None
        else breakdown.total - settings.architecture_budget,
        0.0
        if settings.hardware_budget is None
        else breakdown.actuator + breakdown.sensor - settings.hardware_budget,
        0.0
        if settings.service_budget is None
        else breakdown.service - settings.service_budget,
    )
    objective_diagnostics = ObjectiveDiagnostics(
        recomputed=objective,
        reported=reported_performance,
        absolute_error=abs(objective - reported_performance),
    )
    total_diagnostics = ObjectiveDiagnostics(
        recomputed=expected_total,
        reported=reported_total,
        absolute_error=abs(expected_total - reported_total),
    )
    cost_diagnostics = ArchitectureCostDiagnostics(
        recomputed=breakdown.total,
        reported=reported_cost,
        absolute_error=abs(breakdown.total - reported_cost),
        breakdown_max_error=breakdown_max_error,
        breakdown=breakdown,
    )
    certified = all(
        (
            ofsls.max_residual <= tolerance,
            objective_diagnostics.absolute_error <= tolerance,
            total_diagnostics.absolute_error <= tolerance,
            hardware.max_abs <= tolerance,
            service.max_abs <= tolerance,
            service.menu_valid,
            service.availability_valid,
            cost_diagnostics.absolute_error <= tolerance,
            cost_diagnostics.breakdown_max_error <= tolerance,
            budget_violation <= tolerance,
        )
    )
    return SolutionAudit(
        ofsls=ofsls,
        objective=objective_diagnostics,
        total_objective=total_diagnostics,
        hardware=hardware,
        service=service,
        architecture_cost=cost_diagnostics,
        budget_violation=budget_violation,
        certified=certified,
    )
