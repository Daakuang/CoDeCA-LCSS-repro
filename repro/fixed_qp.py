"""Direct continuous fixed-architecture OF-SLS quadratic programs.

This module is the numerical certification path for an architecture whose
actuator, sensor-device, and directed-service decisions are already fixed.
Only the four raw response arrays ``R``, ``M``, ``N``, and ``L`` are decision
variables.  Hardware and service support is imposed by ordinary zero
equalities; no binary shell, MIP start, residual lifting, or additional cut is
introduced.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .deployment import ArchitectureChoice, architecture_cost
from .model import (
    AvailabilityMatrix,
    BPlusProblem,
    BPlusSolveResult,
    ResponseBlock,
    SolverOptions,
    _add_matrix_equality,
    _beta_block_for_coordinate,
    _coefficient_linear_expression,
    _gurobi,
    _numeric_h2,
    _scalar,
    _settings,
    _status_name,
)
from .realization import FIRResponses, service_lag_for_raw


def _require_fixed_architecture(problem: BPlusProblem) -> ArchitectureChoice:
    if (
        problem.fixed_eta is None
        or problem.fixed_xi is None
        or problem.fixed_service_delays is None
    ):
        raise ValueError(
            "direct fixed-QP solve requires fixed eta, xi, and service delays"
        )
    choice = ArchitectureChoice(
        eta=problem.fixed_eta,
        xi=problem.fixed_xi,
        service_delays=problem.fixed_service_delays,
    )
    problem.deployment.validate_choice(choice)
    return choice


def _fixed_availability(problem: BPlusProblem) -> tuple[AvailabilityMatrix, int]:
    if problem.fixed_service_delays is None:  # Guarded by the public entry point.
        raise RuntimeError("fixed service delays are unavailable")
    all_delays = [
        delay
        for menu in problem.deployment.services.directed_menus
        for delay in menu.finite_delays
    ]
    max_latency = max(all_delays, default=0)
    rows: list[tuple[tuple[int, ...], ...]] = []
    for destination in range(problem.deployment.layout.site_count):
        row: list[tuple[int, ...]] = []
        for source in range(problem.deployment.layout.site_count):
            delay = problem.fixed_service_delays[destination][source]
            row.append(
                tuple(
                    int(delay is not None and delay <= lag)
                    for lag in range(max_latency + 1)
                )
            )
        rows.append(tuple(row))
    return tuple(rows), max_latency


def _service_allows(
    problem: BPlusProblem,
    *,
    destination: int,
    source: int,
    deadline: int,
) -> bool:
    if problem.fixed_service_delays is None:  # Guarded by the public entry point.
        raise RuntimeError("fixed service delays are unavailable")
    delay = problem.fixed_service_delays[destination][source]
    return delay is not None and delay <= deadline


def _add_zero(model: Any, coefficient: Any, *, name: str) -> None:
    model.addConstr(_scalar(coefficient) == 0.0, name=name)


def _configure_model(model: Any, options: SolverOptions) -> None:
    model.Params.Seed = options.seed
    model.Params.Threads = options.threads
    model.Params.NumericFocus = options.numeric_focus
    model.Params.OutputFlag = options.output_flag
    model.Params.FeasibilityTol = options.feasibility_tolerance
    model.Params.OptimalityTol = options.optimality_tolerance
    model.Params.IntFeasTol = options.integer_feasibility_tolerance
    model.Params.MIPGap = options.mip_gap
    # Dual simplex is deterministic here and gives substantially tighter
    # equality residuals than solving a fully fixed mixed-integer shell.
    model.Params.Method = 1
    if options.time_limit is not None:
        model.Params.TimeLimit = options.time_limit


def _configure_numerical_fallback(model: Any) -> None:
    """Resolve a no-incumbent direct QP with an independent barrier path."""

    model.Params.DualReductions = 0
    # Retain conservative presolve so that explicit contradictory equalities
    # are classified as infeasible before the redundant OF-SLS system reaches
    # the barrier factorization.
    model.Params.Presolve = 1
    model.Params.Method = 2
    model.Params.Crossover = 1
    model.Params.BarConvTol = 1.0e-10


def _build_fixed_qp(
    problem: BPlusProblem, options: SolverOptions
) -> tuple[Any, dict[ResponseBlock, tuple[Any, ...]], ArchitectureChoice]:
    gp = _gurobi()
    choice = _require_fixed_architecture(problem)
    model = gp.Model("fixed_architecture_ofsls_qp")
    _configure_model(model, options)

    plant = problem.plant
    horizon = problem.horizon
    infinity = gp.GRB.INFINITY
    responses: dict[ResponseBlock, tuple[Any, ...]] = {
        "R": tuple(
            model.addMVar((plant.n, plant.n), lb=-infinity, name=f"R_{t}")
            for t in range(horizon + 1)
        ),
        "M": tuple(
            model.addMVar((plant.m, plant.n), lb=-infinity, name=f"M_{t}")
            for t in range(horizon + 1)
        ),
        "N": tuple(
            model.addMVar((plant.n, plant.p), lb=-infinity, name=f"N_{t}")
            for t in range(horizon + 1)
        ),
        "L": tuple(
            model.addMVar((plant.m, plant.p), lb=-infinity, name=f"L_{t}")
            for t in range(horizon + 1)
        ),
    }
    R, M, N, L = (
        responses["R"],
        responses["M"],
        responses["N"],
        responses["L"],
    )

    zero_nn = np.zeros((plant.n, plant.n))
    zero_mn = np.zeros((plant.m, plant.n))
    zero_np = np.zeros((plant.n, plant.p))
    _add_matrix_equality(model, R[0], zero_nn, name="raw_R0_padding")
    _add_matrix_equality(model, M[0], zero_mn, name="raw_M0_padding")
    _add_matrix_equality(model, N[0], zero_np, name="raw_N0_padding")

    # Literal raw-lag OF-SLS identities, including all four terminal rows.
    for t in range(horizon + 1):
        R_next = R[t + 1] if t < horizon else zero_nn
        M_next = M[t + 1] if t < horizon else zero_mn
        N_next = N[t + 1] if t < horizon else zero_np
        impulse = np.eye(plant.n) if t == 0 else zero_nn
        _add_matrix_equality(
            model,
            R_next - plant.A @ R[t] - plant.B2 @ M[t],
            impulse,
            name=f"ofsls_left_R_{t}",
        )
        _add_matrix_equality(
            model,
            N_next - plant.A @ N[t] - plant.B2 @ L[t],
            zero_np,
            name=f"ofsls_left_N_{t}",
        )
        _add_matrix_equality(
            model,
            R_next - R[t] @ plant.A - N[t] @ plant.C2,
            impulse,
            name=f"ofsls_right_R_{t}",
        )
        _add_matrix_equality(
            model,
            M_next - M[t] @ plant.A - L[t] @ plant.C2,
            zero_mn,
            name=f"ofsls_right_M_{t}",
        )

    # Fixed grouped hardware: eta gates rows of M/L; each xi device gates all
    # scalar output columns assigned to its physical sensor package.
    layout = problem.deployment.layout
    for actuator, enabled in enumerate(choice.eta):
        if enabled:
            continue
        for t in range(horizon + 1):
            for state in range(plant.n):
                _add_zero(
                    model,
                    M[t][actuator, state],
                    name=f"hardware_M_{t}_{actuator}_{state}",
                )
            for sensor in range(plant.p):
                _add_zero(
                    model,
                    L[t][actuator, sensor],
                    name=f"hardware_eta_L_{t}_{actuator}_{sensor}",
                )
    for sensor in range(plant.p):
        device = layout.sensor_device_for_output(sensor)
        if choice.xi[device]:
            continue
        for t in range(horizon + 1):
            for state in range(plant.n):
                _add_zero(
                    model,
                    N[t][state, sensor],
                    name=f"hardware_N_{t}_{state}_{sensor}",
                )
            for actuator in range(plant.m):
                _add_zero(
                    model,
                    L[t][actuator, sensor],
                    name=f"hardware_xi_L_{t}_{actuator}_{sensor}",
                )

    beta_blocks = _beta_block_for_coordinate(problem)

    def gate_raw(
        block: ResponseBlock,
        raw_lag: int,
        coefficient: Any,
        destination: int,
        source: int,
        name: str,
    ) -> None:
        deadline = service_lag_for_raw(block, raw_lag)
        if deadline is None:  # R[1] is the structural identity coefficient.
            return
        if not _service_allows(
            problem,
            destination=destination,
            source=source,
            deadline=deadline,
        ):
            _add_zero(model, coefficient, name=name)

    # These loop bounds and raw-lag offsets are intentionally identical to the
    # mixed-integer reference model: L/M/N/R use 0/1/0/1 service offsets.
    for t in range(horizon + 1):
        for actuator in range(plant.m):
            for sensor in range(plant.p):
                gate_raw(
                    "L",
                    t,
                    L[t][actuator, sensor],
                    layout.p_u(actuator),
                    layout.p_y(sensor),
                    f"service_L_{t}_{actuator}_{sensor}",
                )
    for t in range(1, horizon + 1):
        for actuator in range(plant.m):
            for state in range(plant.n):
                gate_raw(
                    "M",
                    t,
                    M[t][actuator, state],
                    layout.p_u(actuator),
                    layout.beta_host(beta_blocks[state]),
                    f"service_M_{t}_{actuator}_{state}",
                )
        for state in range(plant.n):
            for sensor in range(plant.p):
                gate_raw(
                    "N",
                    t,
                    N[t][state, sensor],
                    layout.beta_host(beta_blocks[state]),
                    layout.p_y(sensor),
                    f"service_N_{t}_{state}_{sensor}",
                )
        for row in range(plant.n):
            for column in range(plant.n):
                gate_raw(
                    "R",
                    t,
                    R[t][row, column],
                    layout.beta_host(beta_blocks[row]),
                    layout.beta_host(beta_blocks[column]),
                    f"service_R_{t}_{row}_{column}",
                )

    for index, fix in enumerate(problem.response_fixes):
        coefficient = responses[fix.block][fix.lag][fix.row, fix.column]
        model.addConstr(
            _scalar(coefficient) == fix.value,
            name=(
                f"response_fix_{index}_{fix.block}_{fix.lag}_"
                f"{fix.row}_{fix.column}"
            ),
        )

    breakdown = architecture_cost(problem.deployment, choice)
    architecture_constant = gp.LinExpr(float(breakdown.total))
    hardware_constant = gp.LinExpr(float(breakdown.actuator + breakdown.sensor))
    service_constant = gp.LinExpr(float(breakdown.service))
    if options.architecture_budget is not None:
        model.addConstr(
            architecture_constant <= options.architecture_budget,
            name="architecture_budget",
        )
    if options.hardware_budget is not None:
        model.addConstr(
            hardware_constant <= options.hardware_budget,
            name="hardware_budget",
        )
    if options.service_budget is not None:
        model.addConstr(
            service_constant <= options.service_budget,
            name="service_budget",
        )

    performance = gp.QuadExpr()
    for t in range(horizon + 1):
        terms = (
            (plant.C1, R[t], plant.B1),
            (plant.C1, N[t], plant.D21),
            (plant.D12, M[t], plant.B1),
            (plant.D12, L[t], plant.D21),
        )
        for row in range(plant.nz):
            for column in range(plant.nw):
                impulse = _coefficient_linear_expression(
                    gp,
                    constant=(
                        float(plant.D11[row, column]) if t == 0 else 0.0
                    ),
                    terms=terms,
                    row=row,
                    column=column,
                )
                performance += impulse * impulse
    model.setObjective(
        performance + float(options.architecture_weight) * breakdown.total,
        gp.GRB.MINIMIZE,
    )
    return model, responses, choice


def solve_fixed_architecture_qp(
    problem: BPlusProblem, options: SolverOptions | None = None
) -> BPlusSolveResult:
    """Solve one fully prescribed deployment as a direct continuous QP."""

    if not isinstance(problem, BPlusProblem):
        raise TypeError("problem must be a BPlusProblem")
    solver_options = SolverOptions() if options is None else options
    if not isinstance(solver_options, SolverOptions):
        raise TypeError("options must be SolverOptions")
    model, response_variables, choice = _build_fixed_qp(problem, solver_options)
    gp = _gurobi()
    model.optimize()
    runtime = float(model.Runtime)
    status = _status_name(gp, int(model.Status))
    solution_count = int(model.SolCount)
    if solution_count == 0:
        # A fixed OF-SLS QP has a nonnegative sum-of-squares objective.  A
        # direct INFEASIBLE/UNBOUNDED report can therefore be a numerical
        # classification failure in the redundant response equalities.  Use a
        # fresh model and an independent barrier path before accepting any
        # no-incumbent status.
        model.dispose()
        model, response_variables, choice = _build_fixed_qp(problem, solver_options)
        _configure_numerical_fallback(model)
        model.optimize()
        runtime += float(model.Runtime)
        status = _status_name(gp, int(model.Status))
        solution_count = int(model.SolCount)

    if solution_count == 0:
        result = BPlusSolveResult(
            status=status,
            responses=None,
            eta=None,
            xi=None,
            service_delays=None,
            service_availability=None,
            performance_objective=None,
            objective_value=None,
            architecture_cost=None,
            architecture_breakdown=None,
            best_bound=None,
            mip_gap=None,
            runtime=runtime,
            solution_count=solution_count,
            settings=_settings(solver_options),
        )
        model.dispose()
        return result

    responses = FIRResponses(
        R=np.stack(
            [np.asarray(block.X, dtype=float) for block in response_variables["R"]]
        ),
        M=np.stack(
            [np.asarray(block.X, dtype=float) for block in response_variables["M"]]
        ),
        N=np.stack(
            [np.asarray(block.X, dtype=float) for block in response_variables["N"]]
        ),
        L=np.stack(
            [np.asarray(block.X, dtype=float) for block in response_variables["L"]]
        ),
    )
    breakdown = architecture_cost(problem.deployment, choice)
    performance = _numeric_h2(problem.plant, responses)
    objective = float(model.ObjVal)
    recomputed = performance + solver_options.architecture_weight * breakdown.total
    objective_tolerance = max(
        1.0e-8,
        10.0
        * solver_options.optimality_tolerance
        * max(1.0, abs(objective), abs(recomputed)),
    )
    if abs(objective - recomputed) > objective_tolerance:
        raise RuntimeError(
            "fixed-QP objective does not match the independently decoded response "
            "and architecture cost"
        )
    availability, _ = _fixed_availability(problem)
    result = BPlusSolveResult(
        status=status,
        responses=responses,
        eta=choice.eta,
        xi=choice.xi,
        service_delays=choice.service_delays,
        service_availability=availability,
        performance_objective=performance,
        objective_value=objective,
        architecture_cost=breakdown.total,
        architecture_breakdown=breakdown,
        best_bound=None,
        mip_gap=None,
        runtime=runtime,
        solution_count=solution_count,
        settings=_settings(solver_options),
    )
    model.dispose()
    return result


def solve_fixed_qp(
    problem: BPlusProblem, options: SolverOptions | None = None
) -> BPlusSolveResult:
    """Compatibility alias for publication-pipeline solver injection."""

    return solve_fixed_architecture_qp(problem, options)
