"""Native joint hardware/service RFD scan with fixed-architecture re-synthesis."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Final, Iterable, Mapping, Sequence

import cvxpy as cp
import numpy as np

from .cases import JointPlatoonCase, make_joint_platoon_case, platoon_case_matrix_sha256
from .deployment import ArchitectureChoice, ServiceDelayMatrix, architecture_cost
from .diagnostics import audit_solution
from .fixed_qp import solve_fixed_qp
from .joint_platoon_experiment import joint_solver_options
from .joint_publication import _fixed_problem, _result_payload
from .joint_qi import grouped_plant_propagation_delays, hardware_aware_qi_compatible
from .model import BPlusProblem, BPlusSolveResult
from .platoon_experiment import iter_service_architectures
from .realization import FIRResponses, service_lag_for_raw
from .retry_qi_diagnostics import _barrier_variant


SEED: Final[int] = 23
THREADS: Final[int] = 1
HORIZON: Final[int] = 10
BUDGETS: Final[tuple[int, ...]] = tuple(range(14, 36))
REPORT_BUDGETS: Final[tuple[int, ...]] = (14, 24, 35)
LAMBDAS: Final[tuple[float, ...]] = (
    0.0,
    1.0e-4,
    3.0e-4,
    1.0e-3,
    3.0e-3,
    1.0e-2,
    3.0e-2,
    1.0e-1,
    3.0e-1,
    1.0,
    3.0,
    10.0,
    30.0,
    100.0,
    300.0,
    1000.0,
)
THRESHOLDS: Final[tuple[float, ...]] = (
    1.0e-8,
    1.0e-7,
    1.0e-6,
    1.0e-5,
    1.0e-4,
    1.0e-3,
    1.0e-2,
    1.0e-1,
    1.0,
    5.0,
    10.0,
)
DEFAULT_OUTPUT: Final[Path] = Path(
    "results/lcss_v2/joint_platoon/rfd_native_scan.json"
)
AUDIT_TOLERANCE: Final[float] = 2.0e-8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint(output: Path, payload: Mapping[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)


def _beta_coordinate_sites(case: JointPlatoonCase) -> tuple[int, ...]:
    layout = case.deployment.layout
    sites: list[int] = []
    for beta_site, block_size in zip(
        layout.beta_sites,
        layout.state_block_sizes,
        strict=True,
    ):
        sites.extend([beta_site] * block_size)
    return tuple(sites)


def _group_norm(expressions: Iterable[Any]) -> Any:
    pieces = [cp.reshape(item, (int(item.size),), order="F") for item in expressions]
    if not pieces:
        return 0.0
    return cp.norm(cp.hstack(pieces), 2)


def _physical_service_entries(
    case: JointPlatoonCase,
    variables: Mapping[str, Sequence[Any]],
) -> tuple[
    dict[tuple[int, int], list[tuple[int, Any]]],
    list[Any],
]:
    plant = case.plant
    layout = case.deployment.layout
    beta_sites = _beta_coordinate_sites(case)
    entries: dict[tuple[int, int], list[tuple[int, Any]]] = {}
    forbidden: list[Any] = []

    def add(block: str, raw_lag: int, value: Any, destination: int, source: int) -> None:
        deadline = service_lag_for_raw(block, raw_lag)
        if deadline is None:
            return
        menu = case.deployment.services.menu_for(
            destination=destination,
            source=source,
        )
        if menu is None or not any(delay <= deadline for delay in menu.finite_delays):
            forbidden.append(value)
            return
        entries.setdefault((destination, source), []).append((deadline, value))

    for time_index in range(HORIZON + 1):
        for actuator in range(plant.m):
            for sensor in range(plant.p):
                add(
                    "L",
                    time_index,
                    variables["L"][time_index][actuator, sensor],
                    layout.p_u(actuator),
                    layout.p_y(sensor),
                )
    for time_index in range(1, HORIZON + 1):
        for actuator in range(plant.m):
            for state in range(plant.n):
                add(
                    "M",
                    time_index,
                    variables["M"][time_index][actuator, state],
                    layout.p_u(actuator),
                    beta_sites[state],
                )
        for state in range(plant.n):
            for sensor in range(plant.p):
                add(
                    "N",
                    time_index,
                    variables["N"][time_index][state, sensor],
                    beta_sites[state],
                    layout.p_y(sensor),
                )
        for row in range(plant.n):
            for column in range(plant.n):
                add(
                    "R",
                    time_index,
                    variables["R"][time_index][row, column],
                    beta_sites[row],
                    beta_sites[column],
                )
    return entries, forbidden


def _service_regularizer(
    case: JointPlatoonCase,
    entries: Mapping[tuple[int, int], Sequence[tuple[int, Any]]],
) -> Any:
    expression: Any = 0.0
    costs = case.deployment.costs.service_costs_by_delay
    for pair, pair_entries in entries.items():
        menu = case.deployment.services.menu_for(
            destination=pair[0],
            source=pair[1],
        )
        if menu is None or menu.mandatory_delay is not None:
            continue
        delays = tuple(sorted(menu.finite_delays))
        slowest = delays[-1]
        slow_cost = float(costs[slowest][pair[0], pair[1]])
        expression += slow_cost * _group_norm(item[1] for item in pair_entries)
        for faster, slower in zip(delays[:-1], delays[1:], strict=True):
            increment = float(
                costs[faster][pair[0], pair[1]]
                - costs[slower][pair[0], pair[1]]
            )
            if increment <= 0.0:
                continue
            expression += increment * _group_norm(
                item[1] for item in pair_entries if item[0] < slower
            )
    return expression


def _build_rfd_problem(case: JointPlatoonCase) -> dict[str, Any]:
    plant = case.plant
    R = [cp.Variable((plant.n, plant.n), name=f"R_{t}") for t in range(HORIZON + 1)]
    M = [cp.Variable((plant.m, plant.n), name=f"M_{t}") for t in range(HORIZON + 1)]
    N = [cp.Variable((plant.n, plant.p), name=f"N_{t}") for t in range(HORIZON + 1)]
    L = [cp.Variable((plant.m, plant.p), name=f"L_{t}") for t in range(HORIZON + 1)]
    variables: dict[str, Sequence[Any]] = {"R": R, "M": M, "N": N, "L": L}
    constraints: list[Any] = [R[0] == 0.0, M[0] == 0.0, N[0] == 0.0]
    for time_index in range(HORIZON + 1):
        R_next = R[time_index + 1] if time_index < HORIZON else np.zeros((plant.n, plant.n))
        M_next = M[time_index + 1] if time_index < HORIZON else np.zeros((plant.m, plant.n))
        N_next = N[time_index + 1] if time_index < HORIZON else np.zeros((plant.n, plant.p))
        impulse = np.eye(plant.n) if time_index == 0 else np.zeros((plant.n, plant.n))
        constraints.extend(
            (
                R_next - plant.A @ R[time_index] - plant.B2 @ M[time_index] == impulse,
                N_next - plant.A @ N[time_index] - plant.B2 @ L[time_index] == 0.0,
                R_next - R[time_index] @ plant.A - N[time_index] @ plant.C2 == impulse,
                M_next - M[time_index] @ plant.A - L[time_index] @ plant.C2 == 0.0,
            )
        )
    service_entries, forbidden = _physical_service_entries(case, variables)
    constraints.extend(item == 0.0 for item in forbidden)

    h2: Any = 0.0
    for time_index in range(HORIZON + 1):
        direct = plant.D11 if time_index == 0 else np.zeros_like(plant.D11)
        residual = (
            direct
            + plant.C1 @ (R[time_index] @ plant.B1 + N[time_index] @ plant.D21)
            + plant.D12 @ (M[time_index] @ plant.B1 + L[time_index] @ plant.D21)
        )
        h2 += cp.sum_squares(residual)

    hardware: Any = 0.0
    for actuator in range(plant.m):
        hardware += _group_norm(
            [*(M[t][actuator, :] for t in range(HORIZON + 1)),
             *(L[t][actuator, :] for t in range(HORIZON + 1))]
        )
    for outputs in case.sensor_device_groups:
        hardware += _group_norm(
            [*(N[t][:, outputs] for t in range(HORIZON + 1)),
             *(L[t][:, outputs] for t in range(HORIZON + 1))]
        )
    regularizer = hardware + _service_regularizer(case, service_entries)
    weight = cp.Parameter(nonneg=True, value=0.0, name="lambda_rfd")
    problem = cp.Problem(cp.Minimize(h2 + weight * regularizer), constraints)
    return {
        "problem": problem,
        "variables": variables,
        "h2": h2,
        "regularizer": regularizer,
        "weight": weight,
        "service_entries": service_entries,
    }


def _responses(variables: Mapping[str, Sequence[Any]]) -> FIRResponses:
    values: dict[str, np.ndarray] = {}
    for name in ("R", "M", "N", "L"):
        blocks = [np.asarray(item.value, dtype=float) for item in variables[name]]
        if any(np.any(~np.isfinite(block)) for block in blocks):
            raise RuntimeError(f"RFD solver returned nonfinite {name} coefficients")
        values[name] = np.stack(blocks)
    for name in ("R", "M", "N"):
        padding_residual = float(np.max(np.abs(values[name][0])))
        if padding_residual > 2.0e-6:
            raise RuntimeError(
                f"RFD solver returned {name}[0] residual {padding_residual}"
            )
        values[name][0] = 0.0
    identity_residual = float(
        np.max(np.abs(values["R"][1] - np.eye(values["R"].shape[1])))
    )
    if identity_residual > 2.0e-6:
        raise RuntimeError(
            f"RFD solver returned R[1] identity residual {identity_residual}"
        )
    values["R"][1] = np.eye(values["R"].shape[1])
    return FIRResponses(**values)


def _ofsls_residual(case: JointPlatoonCase, responses: FIRResponses) -> float:
    plant = case.plant
    residuals: list[float] = []
    for time_index in range(HORIZON + 1):
        R_next = responses.R[time_index + 1] if time_index < HORIZON else np.zeros((plant.n, plant.n))
        M_next = responses.M[time_index + 1] if time_index < HORIZON else np.zeros((plant.m, plant.n))
        N_next = responses.N[time_index + 1] if time_index < HORIZON else np.zeros((plant.n, plant.p))
        impulse = np.eye(plant.n) if time_index == 0 else np.zeros((plant.n, plant.n))
        residuals.extend(
            (
                float(np.max(np.abs(R_next - plant.A @ responses.R[time_index] - plant.B2 @ responses.M[time_index] - impulse))),
                float(np.max(np.abs(N_next - plant.A @ responses.N[time_index] - plant.B2 @ responses.L[time_index]))),
                float(np.max(np.abs(R_next - responses.R[time_index] @ plant.A - responses.N[time_index] @ plant.C2 - impulse))),
                float(np.max(np.abs(M_next - responses.M[time_index] @ plant.A - responses.L[time_index] @ plant.C2))),
            )
        )
    return max(residuals)


def _hardware_norms(case: JointPlatoonCase, responses: FIRResponses) -> tuple[np.ndarray, np.ndarray]:
    actuator = np.asarray(
        [
            np.linalg.norm(
                np.concatenate((responses.M[:, index, :].ravel(), responses.L[:, index, :].ravel()))
            )
            for index in range(case.plant.m)
        ],
        dtype=float,
    )
    sensor = np.asarray(
        [
            np.linalg.norm(
                np.concatenate((responses.N[:, :, outputs].ravel(), responses.L[:, :, outputs].ravel()))
            )
            for outputs in case.sensor_device_groups
        ],
        dtype=float,
    )
    return actuator, sensor


def _response_entries(
    case: JointPlatoonCase,
    responses: FIRResponses,
) -> dict[tuple[int, int], list[tuple[int, float]]]:
    variables = {"R": responses.R, "M": responses.M, "N": responses.N, "L": responses.L}
    entries, _ = _physical_service_entries(case, variables)
    return {
        pair: [(deadline, abs(float(value))) for deadline, value in values]
        for pair, values in entries.items()
    }


def _decode_choice(
    case: JointPlatoonCase,
    responses: FIRResponses,
    threshold: float,
) -> ArchitectureChoice:
    actuator_norms, sensor_norms = _hardware_norms(case, responses)
    eta = tuple(int(value > threshold) for value in actuator_norms)
    xi = tuple(int(value > threshold) for value in sensor_norms)
    entries = _response_entries(case, responses)
    grid: list[list[int | None]] = [[None] * 4 for _ in range(4)]
    costs = case.deployment.costs.service_costs_by_delay
    for menu in case.deployment.services.directed_menus:
        destination, source = menu.destination_site, menu.source_site
        if menu.mandatory_delay is not None:
            grid[destination][source] = menu.mandatory_delay
            continue
        active_deadlines = [
            deadline
            for deadline, magnitude in entries.get((destination, source), [])
            if magnitude > threshold
        ]
        if not active_deadlines:
            continue
        deadline = min(active_deadlines)
        eligible = [delay for delay in menu.finite_delays if delay <= deadline]
        if not eligible:
            raise RuntimeError("RFD response violates the maximal physical service menu")
        grid[destination][source] = min(
            eligible,
            key=lambda delay: (
                float(costs[delay][destination, source]),
                -delay,
            ),
        )
    return ArchitectureChoice(eta=eta, xi=xi, service_delays=tuple(tuple(row) for row in grid))


def _choice_key(choice: ArchitectureChoice) -> str:
    return json.dumps(
        {
            "eta": choice.eta,
            "xi": choice.xi,
            "services": choice.service_delays,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _choice_payload(case: JointPlatoonCase, choice: ArchitectureChoice) -> dict[str, Any]:
    return {
        "eta": list(choice.eta),
        "xi": list(choice.xi),
        "service_delays": [list(row) for row in choice.service_delays],
        "architecture_breakdown": asdict(architecture_cost(case.deployment, choice)),
    }


def _choice_from_payload(payload: Mapping[str, Any]) -> ArchitectureChoice:
    return ArchitectureChoice(
        eta=tuple(int(value) for value in payload["eta"]),
        xi=tuple(int(value) for value in payload["xi"]),
        service_delays=tuple(
            tuple(None if value is None else int(value) for value in row)
            for row in payload["service_delays"]
        ),
    )


def _controller_delays(case: JointPlatoonCase, choice: ArchitectureChoice) -> tuple[tuple[int | None, ...], ...]:
    layout = case.deployment.layout
    return tuple(
        tuple(choice.service_delays[actuator_site][sensor_site] for sensor_site in layout.sensor_device_sites)
        for actuator_site in layout.actuator_sites
    )


def _is_completion(raw: ServiceDelayMatrix, candidate: ServiceDelayMatrix) -> bool:
    for raw_row, candidate_row in zip(raw, candidate, strict=True):
        for raw_delay, candidate_delay in zip(raw_row, candidate_row, strict=True):
            if raw_delay is not None and (candidate_delay is None or candidate_delay > raw_delay):
                return False
    return True


def _qi_completion(
    case: JointPlatoonCase,
    raw: ArchitectureChoice,
    compatible_services: Mapping[tuple[tuple[int, ...], tuple[int, ...]], Sequence[ServiceDelayMatrix]],
) -> ArchitectureChoice | None:
    candidates = compatible_services.get((raw.eta, raw.xi), ())
    completions = [service for service in candidates if _is_completion(raw.service_delays, service)]
    if not completions:
        return None
    best = min(
        completions,
        key=lambda service: (
            architecture_cost(case.deployment, ArchitectureChoice(raw.eta, raw.xi, service)).total,
            sum(
                raw.service_delays[i][j] != service[i][j]
                for i in range(4)
                for j in range(4)
            ),
            str(service),
        ),
    )
    return ArchitectureChoice(raw.eta, raw.xi, best)


def _compatible_service_cache(
    case: JointPlatoonCase,
    choices: Sequence[ArchitectureChoice],
) -> dict[tuple[tuple[int, ...], tuple[int, ...]], tuple[ServiceDelayMatrix, ...]]:
    propagation = grouped_plant_propagation_delays(
        case.plant.A,
        case.plant.B2,
        case.plant.C2,
        case.sensor_device_groups,
    )
    patterns = sorted({(choice.eta, choice.xi) for choice in choices})
    services = tuple(item.service_delays for item in iter_service_architectures(case.base))
    cache: dict[tuple[tuple[int, ...], tuple[int, ...]], tuple[ServiceDelayMatrix, ...]] = {}
    for eta, xi in patterns:
        cache[(eta, xi)] = tuple(
            service
            for service in services
            if hardware_aware_qi_compatible(
                tuple(
                    tuple(
                        service[actuator_site][sensor_site]
                        for sensor_site in case.deployment.layout.sensor_device_sites
                    )
                    for actuator_site in case.deployment.layout.actuator_sites
                ),
                propagation,
                eta,
                xi,
            )
        )
    return cache


def _fixed_resynthesis(
    case: JointPlatoonCase,
    choice: ArchitectureChoice,
) -> dict[str, Any]:
    cost = architecture_cost(case.deployment, choice).total
    problem = _fixed_problem(
        case,
        eta=choice.eta,
        xi=choice.xi,
        service_delays=choice.service_delays,
        horizon=HORIZON,
    )
    options = joint_solver_options(
        seed=SEED,
        threads=THREADS,
        architecture_budget=cost,
    )
    result = solve_fixed_qp(problem, options)
    payload = _result_payload(problem, result)
    payload["certified_feasible"] = (
        result.status == "OPTIMAL"
        and result.solution_count > 0
        and isinstance(payload.get("audit"), Mapping)
        and payload["audit"].get("certified") is True
    )
    if payload["certified_feasible"]:
        return payload
    barrier = _barrier_variant(
        problem,
        architecture_budget=cost,
        presolve=1,
    )
    payload["barrier_retry"] = barrier
    if barrier.get("certified_optimal_and_audited") is True:
        payload.update(
            {
                "status": "OPTIMAL",
                "solution_count": int(barrier["solution_count"]),
                "performance_h2": float(barrier["performance_h2"]),
                "objective_value": float(barrier["objective_value"]),
                "architecture_cost": cost,
                "architecture_breakdown": asdict(architecture_cost(case.deployment, choice)),
                "eta": list(choice.eta),
                "xi": list(choice.xi),
                "service_delays": [list(row) for row in choice.service_delays],
                "best_bound": None,
                "mip_gap": None,
                "runtime_seconds": float(barrier["runtime_seconds"]),
                "audit": barrier["audit"],
                "certified_feasible": True,
            }
        )
    return payload


def _envelope(
    candidates: Sequence[Mapping[str, Any]],
    budgets: Sequence[int],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for budget in budgets:
        eligible = [
            item
            for item in candidates
            if item["fixed_architecture_result"].get("certified_feasible") is True
            and float(item["choice"]["architecture_breakdown"]["total"]) <= budget + 1.0e-9
        ]
        best = min(
            eligible,
            key=lambda item: (
                float(item["fixed_architecture_result"]["performance_h2"]),
                float(item["choice"]["architecture_breakdown"]["total"]),
                str(item["key"]),
            ),
            default=None,
        )
        result.append(
            {
                "budget": int(budget),
                "eligible_certified_count": len(eligible),
                "argmin_key": None if best is None else best["key"],
                "realized_total_cost": None if best is None else float(best["choice"]["architecture_breakdown"]["total"]),
                "performance_h2": None if best is None else float(best["fixed_architecture_result"]["performance_h2"]),
            }
        )
    return result


def run_rfd(output: Path) -> dict[str, Any]:
    output = output.resolve()
    case = make_joint_platoon_case(seed=SEED)
    model = _build_rfd_problem(case)
    path_points: list[dict[str, Any]] = []
    response_by_lambda: dict[float, FIRResponses] = {}
    for regularization in LAMBDAS:
        model["weight"].value = regularization
        started = perf_counter()
        objective = model["problem"].solve(
            solver=cp.CLARABEL,
            warm_start=False,
            verbose=False,
            max_iter=500,
            tol_gap_abs=1.0e-9,
            tol_gap_rel=1.0e-9,
            tol_feas=1.0e-9,
        )
        wall = perf_counter() - started
        status = str(model["problem"].status)
        if status not in {"optimal", "optimal_inaccurate"}:
            raise RuntimeError(f"RFD SOCP failed at lambda={regularization}: {status}")
        responses = _responses(model["variables"])
        response_by_lambda[regularization] = responses
        residual = _ofsls_residual(case, responses)
        if residual > 2.0e-6:
            raise RuntimeError(
                f"RFD SOCP residual {residual} is too large at lambda={regularization}"
            )
        actuator_norms, sensor_norms = _hardware_norms(case, responses)
        for threshold in THRESHOLDS:
            choice = _decode_choice(case, responses, threshold)
            case.deployment.validate_choice(choice)
            path_points.append(
                {
                    "key": f"rfd_lambda_{regularization:.17g}_threshold_{threshold:.17g}",
                    "regularization": regularization,
                    "threshold": threshold,
                    "continuous_status": status,
                    "continuous_objective": float(objective),
                    "continuous_h2": float(model["h2"].value),
                    "continuous_regularizer": float(model["regularizer"].value),
                    "continuous_ofsls_residual_max": residual,
                    "continuous_wall_seconds": wall,
                    "actuator_group_norms": actuator_norms.tolist(),
                    "sensor_group_norms": sensor_norms.tolist(),
                    "choice": _choice_payload(case, choice),
                    "choice_key": _choice_key(choice),
                }
            )
        print(
            json.dumps(
                {
                    "family": "rfd_continuous_path",
                    "lambda": regularization,
                    "status": status,
                    "h2": float(model["h2"].value),
                    "regularizer": float(model["regularizer"].value),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    _checkpoint(
        output,
        {
            "schema_version": 1,
            "purpose": "native actuator/sensor/service RFD path with actual-cost re-synthesis",
            "seed": SEED,
            "threads": THREADS,
            "horizon": HORIZON,
            "matrix_sha256": platoon_case_matrix_sha256(case),
            "regularization_grid": list(LAMBDAS),
            "threshold_grid": list(THRESHOLDS),
            "path_point_count": len(path_points),
            "path_points": path_points,
            "stage": "continuous_path_complete",
            "complete": False,
        },
    )

    unique_raw: dict[str, ArchitectureChoice] = {}
    for point in path_points:
        unique_raw.setdefault(str(point["choice_key"]), _choice_from_payload(point["choice"]))
    compatible = _compatible_service_cache(case, tuple(unique_raw.values()))
    raw_candidates: list[dict[str, Any]] = []
    repaired_choices: dict[str, ArchitectureChoice] = {}
    for index, (choice_key, choice) in enumerate(sorted(unique_raw.items())):
        fixed = _fixed_resynthesis(case, choice)
        repaired = _qi_completion(case, choice, compatible)
        raw_qi = hardware_aware_qi_compatible(
            _controller_delays(case, choice),
            grouped_plant_propagation_delays(
                case.plant.A,
                case.plant.B2,
                case.plant.C2,
                case.sensor_device_groups,
            ),
            choice.eta,
            choice.xi,
        )
        record = {
            "key": f"rfd_raw_{index:03d}",
            "choice_key": choice_key,
            "choice": _choice_payload(case, choice),
            "source_path_keys": [point["key"] for point in path_points if point["choice_key"] == choice_key],
            "raw_hardware_aware_qi": raw_qi,
            "fixed_architecture_result": fixed,
            "qi_completion_choice_key": None if repaired is None else _choice_key(repaired),
        }
        raw_candidates.append(record)
        if repaired is not None:
            repaired_choices.setdefault(_choice_key(repaired), repaired)
        print(
            json.dumps(
                {
                    "family": "rfd_raw_fixed_qp",
                    "index": index,
                    "count": len(unique_raw),
                    "status": fixed["status"],
                    "certified": fixed["certified_feasible"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        _checkpoint(
            output,
            {
                "schema_version": 1,
                "purpose": "native actuator/sensor/service RFD path with actual-cost re-synthesis",
                "seed": SEED,
                "threads": THREADS,
                "horizon": HORIZON,
                "matrix_sha256": platoon_case_matrix_sha256(case),
                "regularization_grid": list(LAMBDAS),
                "threshold_grid": list(THRESHOLDS),
                "path_point_count": len(path_points),
                "path_points": path_points,
                "raw_unique_candidate_count": len(unique_raw),
                "raw_candidates": raw_candidates,
                "stage": "raw_fixed_qp_in_progress",
                "complete": False,
            },
        )

    repaired_candidates: list[dict[str, Any]] = []
    for index, (choice_key, choice) in enumerate(sorted(repaired_choices.items())):
        fixed = _fixed_resynthesis(case, choice)
        repaired_candidates.append(
            {
                "key": f"rfd_qi_repaired_{index:03d}",
                "choice_key": choice_key,
                "choice": _choice_payload(case, choice),
                "source_raw_keys": [
                    item["key"]
                    for item in raw_candidates
                    if item["qi_completion_choice_key"] == choice_key
                ],
                "verified_hardware_aware_qi": True,
                "fixed_architecture_result": fixed,
            }
        )
        print(
            json.dumps(
                {
                    "family": "rfd_qi_repaired_fixed_qp",
                    "index": index,
                    "count": len(repaired_choices),
                    "status": fixed["status"],
                    "certified": fixed["certified_feasible"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    payload = {
        "schema_version": 1,
        "purpose": "native actuator/sensor/service RFD path with actual-cost re-synthesis",
        "seed": SEED,
        "threads": THREADS,
        "horizon": HORIZON,
        "matrix_sha256": platoon_case_matrix_sha256(case),
        "regularization_grid": list(LAMBDAS),
        "threshold_grid": list(THRESHOLDS),
        "path_point_count": len(path_points),
        "path_points": path_points,
        "raw_unique_candidate_count": len(raw_candidates),
        "raw_candidates": raw_candidates,
        "raw_evaluated_budget_envelope": _envelope(raw_candidates, BUDGETS),
        "qi_repaired_unique_candidate_count": len(repaired_candidates),
        "qi_repaired_candidates": repaired_candidates,
        "qi_repaired_evaluated_budget_envelope": _envelope(repaired_candidates, BUDGETS),
        "report_budget_summary": {
            "raw": [item for item in _envelope(raw_candidates, BUDGETS) if item["budget"] in REPORT_BUDGETS],
            "qi_repaired": [item for item in _envelope(repaired_candidates, BUDGETS) if item["budget"] in REPORT_BUDGETS],
        },
        "complete": True,
    }
    _checkpoint(output, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    payload = run_rfd(arguments.output)
    print(
        json.dumps(
            {
                "output": str(arguments.output.resolve()),
                "path_point_count": payload["path_point_count"],
                "raw_unique_candidate_count": payload["raw_unique_candidate_count"],
                "qi_repaired_unique_candidate_count": payload["qi_repaired_unique_candidate_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
