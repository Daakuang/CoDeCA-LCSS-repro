"""Joint grouped-hardware and service optimization on the submitted platoon.

Run from the repository root with ``PYTHONPATH=LcssV1.1/V2``.  The default
solver seed and thread count are explicit publication settings; this runner
adds no screening, warm-start, or hidden acceleration route.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Final, Sequence

from .cases import (
    PLATOON_ARCHITECTURE_UNIT_SCALE,
    JointPlatoonCase,
    make_joint_platoon_case,
    platoon_case_matrix_sha256,
)
from .diagnostics import audit_solution
from .model import BPlusProblem, BPlusSolveResult, SolverOptions, solve_bplus


DEFAULT_SEED: Final[int] = 23
DEFAULT_THREADS: Final[int] = 1
DEFAULT_HORIZON: Final[int] = 10
DEFAULT_OUTPUT: Final[Path] = Path("results/lcss_v2/joint_platoon")


def make_joint_problem(
    case: JointPlatoonCase, *, horizon: int = DEFAULT_HORIZON
) -> BPlusProblem:
    """Build the free-device, free-service problem without fixing binaries."""

    if not isinstance(case, JointPlatoonCase):
        raise TypeError("case must be a JointPlatoonCase")
    return BPlusProblem(
        plant=case.plant,
        deployment=case.deployment,
        horizon=horizon,
        fixed_eta=None,
        fixed_xi=None,
        fixed_service_delays=None,
    )


def joint_solver_options(
    *,
    seed: int = DEFAULT_SEED,
    threads: int = DEFAULT_THREADS,
    architecture_budget: float | None = None,
    hardware_budget: float | None = None,
    service_budget: float | None = None,
) -> SolverOptions:
    """Return deterministic options for either combined or separate caps."""

    if architecture_budget is None and hardware_budget is None and service_budget is None:
        raise ValueError("at least one architecture or component budget is required")
    return SolverOptions(
        seed=seed,
        threads=threads,
        numeric_focus=3,
        output_flag=0,
        feasibility_tolerance=1.0e-9,
        optimality_tolerance=1.0e-9,
        integer_feasibility_tolerance=1.0e-9,
        mip_gap=1.0e-9,
        time_limit=None,
        architecture_weight=0.0,
        architecture_budget=architecture_budget,
        hardware_budget=hardware_budget,
        service_budget=service_budget,
    )


def _case_payload(case: JointPlatoonCase, *, horizon: int) -> dict[str, Any]:
    layout = case.deployment.layout
    return {
        "schema_version": 1,
        "seed": case.seed,
        "horizon": horizon,
        "matrix_sha256": platoon_case_matrix_sha256(case),
        "plant_dimensions": {
            "n": case.plant.n,
            "m": case.plant.m,
            "p": case.plant.p,
        },
        "sensor_device_labels": list(case.sensor_device_labels),
        "sensor_device_groups": [list(group) for group in case.sensor_device_groups],
        "sensor_device_sites": list(layout.sensor_device_sites),
        "actuator_device_labels": list(case.actuator_device_labels),
        "actuator_device_sites": list(layout.actuator_sites),
        "cost_units": {
            "hardware_per_device": 1.0,
            "service_original_cost_scale": PLATOON_ARCHITECTURE_UNIT_SCALE,
            "service_delay_1": 3.0,
            "service_delay_2": 2.0,
            "local_service": 0.0,
        },
    }


def _result_payload(
    problem: BPlusProblem,
    result: BPlusSolveResult,
    options: SolverOptions,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": result.status,
        "solution_count": result.solution_count,
        "runtime": result.runtime,
        "best_bound": result.best_bound,
        "mip_gap": result.mip_gap,
        "solver_settings": asdict(result.settings),
        "requested_budgets": {
            "architecture": options.architecture_budget,
            "hardware": options.hardware_budget,
            "service": options.service_budget,
        },
    }
    if result.solution_count == 0:
        return payload
    if (
        result.eta is None
        or result.xi is None
        or result.service_delays is None
        or result.performance_objective is None
        or result.objective_value is None
        or result.architecture_cost is None
        or result.architecture_breakdown is None
    ):
        raise RuntimeError("solution-bearing result has incomplete decoded data")
    audit = audit_solution(problem, result, tolerance=2.0e-8)
    payload.update(
        {
            "eta": list(result.eta),
            "xi": list(result.xi),
            "service_delays": [list(row) for row in result.service_delays],
            "performance_h2": result.performance_objective,
            "objective_value": result.objective_value,
            "architecture_cost": result.architecture_cost,
            "architecture_breakdown": asdict(result.architecture_breakdown),
            "audit": asdict(audit),
        }
    )
    return payload


def run_joint_platoon(
    *,
    horizon: int = DEFAULT_HORIZON,
    seed: int = DEFAULT_SEED,
    threads: int = DEFAULT_THREADS,
    architecture_budget: float | None = None,
    hardware_budget: float | None = None,
    service_budget: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Solve one declared budget point and return case/result payloads."""

    case = make_joint_platoon_case(seed=seed)
    problem = make_joint_problem(case, horizon=horizon)
    options = joint_solver_options(
        seed=seed,
        threads=threads,
        architecture_budget=architecture_budget,
        hardware_budget=hardware_budget,
        service_budget=service_budget,
    )
    result = solve_bplus(problem, options)
    return _case_payload(case, horizon=problem.horizon), _result_payload(
        problem, result, options
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Solve one joint platoon hardware-service budget point."
    )
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--architecture-budget", type=float)
    parser.add_argument("--hardware-budget", type=float)
    parser.add_argument("--service-budget", type=float)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    case_payload, result_payload = run_joint_platoon(
        horizon=args.horizon,
        seed=args.seed,
        threads=args.threads,
        architecture_budget=args.architecture_budget,
        hardware_budget=args.hardware_budget,
        service_budget=args.service_budget,
    )
    _write_json(args.output / "case.json", case_payload)
    _write_json(args.output / "result.json", result_payload)
    print(
        json.dumps(
            {
                "status": result_payload["status"],
                "solution_count": result_payload["solution_count"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0 if int(result_payload["solution_count"]) > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
