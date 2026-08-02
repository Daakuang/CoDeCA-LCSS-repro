"""Retry only the unresolved fixed-architecture QI publication points.

The retry uses the unchanged direct continuous OF-SLS QP, publication seed,
horizon, costs, masks, and audit tolerance.  It writes a separate diagnostic
artifact and never modifies the frozen publication result.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Final, Sequence

import numpy as np

from .cases import make_joint_platoon_case
from .deployment import architecture_cost
from .diagnostics import audit_solution
from .fixed_qp import _build_fixed_qp, _fixed_availability, solve_fixed_qp
from .joint_platoon_experiment import joint_solver_options
from .joint_publication import (
    DEFAULT_HORIZON,
    DEFAULT_SEED,
    DEFAULT_THREADS,
    _fixed_qi_point,
    _fixed_problem,
    enumerate_pbh_admissible_qi_deployments,
)
from .model import (
    BPlusProblem,
    BPlusSolveResult,
    _gurobi,
    _numeric_h2,
    _settings,
    _status_name,
)
from .realization import FIRResponses


DEFAULT_PUBLICATION: Final[Path] = Path(
    "results/lcss_v2/joint_platoon/publication.json"
)
DEFAULT_OUTPUT: Final[Path] = Path(
    "results/lcss_v2/joint_platoon/qi_fixed_qp_retry.json"
)
AUDIT_TOLERANCE: Final[float] = 2.0e-8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _barrier_variant(
    problem: BPlusProblem,
    *,
    architecture_budget: float,
    presolve: int,
) -> dict[str, object]:
    options = joint_solver_options(
        seed=DEFAULT_SEED,
        threads=DEFAULT_THREADS,
        architecture_budget=architecture_budget,
    )
    model, variables, choice = _build_fixed_qp(problem, options)
    gp = _gurobi()
    model.Params.DualReductions = 0
    model.Params.Presolve = presolve
    model.Params.Method = 2
    model.Params.BarHomogeneous = 1
    model.Params.Crossover = 1
    model.Params.BarConvTol = 1.0e-10
    model.optimize()

    status = _status_name(gp, int(model.Status))
    solution_count = int(model.SolCount)
    payload: dict[str, object] = {
        "presolve": presolve,
        "dual_reductions": 0,
        "method": "barrier",
        "barrier_homogeneous": 1,
        "crossover": 1,
        "barrier_convergence_tolerance": 1.0e-10,
        "status": status,
        "solution_count": solution_count,
        "runtime_seconds": float(model.Runtime),
        "barrier_iterations": int(model.BarIterCount),
        "certified_optimal_and_audited": False,
    }
    if solution_count > 0:
        responses = FIRResponses(
            R=np.stack([np.asarray(block.X, dtype=float) for block in variables["R"]]),
            M=np.stack([np.asarray(block.X, dtype=float) for block in variables["M"]]),
            N=np.stack([np.asarray(block.X, dtype=float) for block in variables["N"]]),
            L=np.stack([np.asarray(block.X, dtype=float) for block in variables["L"]]),
        )
        breakdown = architecture_cost(problem.deployment, choice)
        performance = _numeric_h2(problem.plant, responses)
        result = BPlusSolveResult(
            status=status,
            responses=responses,
            eta=choice.eta,
            xi=choice.xi,
            service_delays=choice.service_delays,
            service_availability=_fixed_availability(problem)[0],
            performance_objective=performance,
            objective_value=float(model.ObjVal),
            architecture_cost=breakdown.total,
            architecture_breakdown=breakdown,
            best_bound=None,
            mip_gap=None,
            runtime=float(model.Runtime),
            solution_count=solution_count,
            settings=_settings(options),
        )
        audit = audit_solution(problem, result, tolerance=AUDIT_TOLERANCE)
        payload.update(
            {
                "performance_h2": performance,
                "objective_value": float(model.ObjVal),
                "audit": asdict(audit),
                "certified_optimal_and_audited": (
                    status == "OPTIMAL" and audit.certified
                ),
            }
        )
    model.dispose()
    return payload


def retry_qi_diagnostics(publication: Path) -> dict[str, object]:
    source = json.loads(publication.read_text(encoding="utf-8"))
    unresolved = {
        str(point["key"])
        for point in source["qi"]["deployment_points"]
        if point.get("certified_feasible") is not True
    }
    case = make_joint_platoon_case(seed=DEFAULT_SEED)
    deployments, _, _ = enumerate_pbh_admissible_qi_deployments(
        case, source["pbh"]
    )
    selected = tuple(item for item in deployments if item.key in unresolved)
    if len(selected) != len(unresolved):
        missing = sorted(unresolved - {item.key for item in selected})
        raise RuntimeError(f"unresolved QI deployments are missing: {missing}")

    points: list[dict[str, object]] = []
    for deployment in selected:
        point = _fixed_qi_point(
            case,
            deployment,
            horizon=DEFAULT_HORIZON,
            seed=DEFAULT_SEED,
            threads=DEFAULT_THREADS,
            solver=solve_fixed_qp,
        )
        problem = _fixed_problem(
            case,
            eta=deployment.eta,
            xi=deployment.xi,
            service_delays=deployment.service_delays,
            horizon=DEFAULT_HORIZON,
        )
        point["barrier_variants"] = [
            _barrier_variant(
                problem,
                architecture_budget=deployment.total_cost,
                presolve=presolve,
            )
            for presolve in (0, 1)
        ]
        points.append(point)
    variant_optimal_count = sum(
        any(
            variant["certified_optimal_and_audited"] is True
            for variant in point["barrier_variants"]
        )
        for point in points
    )
    return {
        "schema_version": 1,
        "purpose": "independent retry of unresolved fixed-architecture QI QPs",
        "source_publication": publication.as_posix(),
        "source_publication_sha256": _sha256(publication),
        "seed": DEFAULT_SEED,
        "threads": DEFAULT_THREADS,
        "horizon": DEFAULT_HORIZON,
        "changed_model_constraints": False,
        "changed_support_or_tolerance": False,
        "requested_count": len(selected),
        "optimal_and_audited_count": sum(
            point["certified_feasible"] is True for point in points
        ),
        "barrier_variant_optimal_and_audited_count": variant_optimal_count,
        "points": points,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication", type=Path, default=DEFAULT_PUBLICATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    publication = arguments.publication.resolve()
    output = arguments.output.resolve()
    payload = retry_qi_diagnostics(publication)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output),
        "requested_count": payload["requested_count"],
        "optimal_and_audited_count": payload["optimal_and_audited_count"],
        "barrier_variant_optimal_and_audited_count": payload[
            "barrier_variant_optimal_and_audited_count"
        ],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
