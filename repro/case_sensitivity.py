"""Recompute the fixed-hardware and delay-three sensitivity cases."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from .cases import JointPlatoonCase, make_joint_platoon_case, platoon_case_matrix_sha256
from .fixed_qp import solve_fixed_qp
from .joint_platoon_experiment import joint_solver_options, make_joint_problem
from .joint_publication import (
    _certificate_payload,
    _fixed_problem,
    _result_payload,
    solve_with_dual_reductions_disabled,
)
from .joint_qi import (
    grouped_plant_propagation_delays,
    solve_hardware_aware_qi,
)
from .model import BPlusProblem, BPlusSolveResult, SolverOptions, solve_bplus


SEED: Final[int] = 23
THREADS: Final[int] = 1
HORIZON: Final[int] = 10
BUDGETS: Final[tuple[int, ...]] = (14, 24, 35)
DEFAULT_OUTPUT: Final[Path] = Path(
    "results/lcss_v2/reproduced_sensitivity/case_sensitivity.json"
)


def _write(output: Path, payload: Mapping[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)


def _service_matrix_count(case: JointPlatoonCase) -> int:
    count = 1
    for menu in case.deployment.services.directed_menus:
        if menu.mandatory_delay is None:
            count *= len(menu.finite_delays) + 1
    return count


def _certify(
    case: JointPlatoonCase,
    problem: BPlusProblem,
    options: SolverOptions,
    incumbent: BPlusSolveResult,
    *,
    qi_no_good_count: int | None = None,
    verified_qi: bool | None = None,
) -> dict[str, Any]:
    incumbent_payload = _result_payload(problem, incumbent)
    fixed_result: BPlusSolveResult | None = None
    fixed_payload: dict[str, Any] | None = None
    if (
        incumbent.solution_count > 0
        and incumbent.eta is not None
        and incumbent.xi is not None
        and incumbent.service_delays is not None
    ):
        fixed_problem = _fixed_problem(
            case,
            eta=incumbent.eta,
            xi=incumbent.xi,
            service_delays=incumbent.service_delays,
            horizon=problem.horizon,
        )
        fixed_result = solve_fixed_qp(fixed_problem, options)
        fixed_payload = _result_payload(fixed_problem, fixed_result)
    payload: dict[str, Any] = {
        "incumbent": incumbent_payload,
        "fixed_architecture_resynthesis": fixed_payload,
        "certificate": _certificate_payload(
            incumbent,
            incumbent_payload,
            fixed_result,
            fixed_payload,
        ),
    }
    if qi_no_good_count is not None:
        payload["qi_no_good_count"] = qi_no_good_count
        payload["verified_qi"] = verified_qi
    return payload


def _solve_fixed_hardware() -> dict[str, Any]:
    case = make_joint_platoon_case(seed=SEED)
    points: list[dict[str, Any]] = []
    for budget in BUDGETS:
        problem = BPlusProblem(
            plant=case.plant,
            deployment=case.deployment,
            horizon=HORIZON,
            fixed_eta=(1, 1, 1),
            fixed_xi=(1,) * 7,
            fixed_service_delays=None,
        )
        options = joint_solver_options(
            seed=SEED,
            threads=THREADS,
            architecture_budget=budget,
        )
        incumbent = solve_bplus(problem, options)
        status_resolution = None
        if incumbent.status == "INF_OR_UNBD":
            original_status = incumbent.status
            incumbent = solve_with_dual_reductions_disabled(problem, options)
            status_resolution = {
                "original_status": original_status,
                "dual_reductions": 0,
                "resolved_status": incumbent.status,
                "certified_infeasible": (
                    incumbent.status == "INFEASIBLE"
                    and incumbent.solution_count == 0
                ),
            }
        point = {
            "budget": budget,
            **_certify(case, problem, options, incumbent),
        }
        if status_resolution is not None:
            point["status_resolution"] = status_resolution
        points.append(point)
    return {
        "fixed_eta": [1, 1, 1],
        "fixed_xi": [1] * 7,
        "matrix_sha256": platoon_case_matrix_sha256(case),
        "points": points,
    }


def _solve_three_tier() -> dict[str, Any]:
    case = make_joint_platoon_case(seed=SEED, service_horizon=3)
    problem = make_joint_problem(case, horizon=HORIZON)
    propagation = grouped_plant_propagation_delays(
        case.plant.A,
        case.plant.B2,
        case.plant.C2,
        case.sensor_device_groups,
    )
    proposed_points: list[dict[str, Any]] = []
    qi_points: list[dict[str, Any]] = []
    for budget in BUDGETS:
        options = joint_solver_options(
            seed=SEED,
            threads=THREADS,
            architecture_budget=budget,
        )
        proposed = solve_bplus(problem, options)
        proposed_points.append(
            {
                "budget": budget,
                **_certify(case, problem, options, proposed),
            }
        )
        qi = solve_hardware_aware_qi(problem, propagation, options)
        qi_points.append(
            {
                "budget": budget,
                **_certify(
                    case,
                    problem,
                    options,
                    qi.solve,
                    qi_no_good_count=qi.no_good_count,
                    verified_qi=qi.verified_qi,
                ),
            }
        )
    service_count = _service_matrix_count(case)
    return {
        "matrix_sha256": platoon_case_matrix_sha256(case),
        "service_matrix_count": service_count,
        "raw_hardware_service_candidate_count": service_count * 2**10,
        "pbh_admissible_hardware_service_candidate_count": service_count * 16,
        "menu": {
            "main_and_adjacent": [1, 2, 3, None],
            "distance_two": [2, 3, None],
            "service_cost_by_delay": {"1": 3.0, "2": 2.0, "3": 1.0},
        },
        "proposed_points": proposed_points,
        "qi_points": qi_points,
    }


def run_case_sensitivity(output: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "fixed-hardware ablation and delay-three service-menu sensitivity",
        "seed": SEED,
        "threads": THREADS,
        "horizon": HORIZON,
        "budgets": list(BUDGETS),
        "fixed_hardware_ablation": _solve_fixed_hardware(),
        "three_tier_latency_sensitivity": _solve_three_tier(),
        "complete": True,
    }
    _write(output.resolve(), payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    payload = run_case_sensitivity(arguments.output)
    print(
        json.dumps(
            {
                "output": str(arguments.output.resolve()),
                "budget_count": len(payload["budgets"]),
                "complete": payload["complete"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
