"""Reproducible publication pipeline for joint platoon architecture selection.

The default run uses Seed 23, one solver thread, and an FIR horizon of 10.  It
solves the proposed and hardware-aware-QI MICPs at every integer combined
budget from 14 through 35, re-synthesizes every selected architecture with all
discrete choices fixed, evaluates the complete PBH-admissible QI deployment
catalog, and re-synthesizes the canonical and legacy-RFD service baselines with
free hardware under the same budgets.

No residual lifting, warm start, screening heuristic, or hidden cut is used.
The only additional constraints in the QI MICP are the exact finite-catalog QI
no-goods declared in :mod:`repro.joint_qi`.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Final, Iterable, Mapping, Sequence, TypeAlias

import numpy as np

from .cases import JointPlatoonCase, make_joint_platoon_case, platoon_case_matrix_sha256
from .deployment import ArchitectureChoice, ServiceDelayMatrix, architecture_cost
from .diagnostics import SolutionAudit, audit_solution
from .fixed_qp import solve_fixed_qp
from .joint_baselines import (
    JointBaselineOutcome,
    JointBaselineSpec,
    joint_canonical_baseline_specs,
    load_joint_rfd_baseline_specs,
    make_joint_baseline_problem,
    solve_joint_baselines,
)
from .joint_platoon_experiment import joint_solver_options, make_joint_problem
from .joint_qi import (
    FixedServiceQICertificate,
    grouped_plant_propagation_delays,
    hardware_mask,
    service_menu_qi_certificates,
    solve_hardware_aware_qi,
    summarize_admissible_joint_qi,
)
from .model import BPlusProblem, BPlusSolveResult, SolverOptions, solve_bplus
from .pbh_admissibility import (
    PBHSelectionCertificate,
    detectability_certificate,
    stabilizability_certificate,
)
from .platoon_experiment import (
    iter_service_architectures,
    platoon_dense_architecture,
    write_json_atomic,
)


DEFAULT_SEED: Final[int] = 23
DEFAULT_THREADS: Final[int] = 1
DEFAULT_HORIZON: Final[int] = 10
DEFAULT_BUDGETS: Final[tuple[int, ...]] = tuple(range(14, 36))
HORIZON_SENSITIVITY_BUDGETS: Final[tuple[int, ...]] = (14, 24, 35)
HORIZON_SENSITIVITY_HORIZONS: Final[tuple[int, ...]] = (8, 10, 12)
DEFAULT_DENSE_COST: Final[int] = 35
DEFAULT_OUTPUT: Final[Path] = Path(
    "results/lcss_v2/joint_platoon/publication.json"
)
SCHEMA_VERSION: Final[int] = 1
AUDIT_TOLERANCE: Final[float] = 2.0e-8
CERTIFICATE_ABSOLUTE_TOLERANCE: Final[float] = 5.0e-8
CERTIFICATE_RELATIVE_TOLERANCE: Final[float] = 1.0e-8
QI_CHECKPOINT_INTERVAL: Final[int] = 25
EXPECTED_SERVICE_MENU_COUNT: Final[int] = 8_748
EXPECTED_PBH_ACTUATOR_COUNT: Final[int] = 1
EXPECTED_PBH_SENSOR_COUNT: Final[int] = 16
EXPECTED_PBH_ADMISSIBLE_QI_COUNT: Final[int] = 2_736


SolveFunction: TypeAlias = Callable[
    [BPlusProblem, SolverOptions | None], BPlusSolveResult
]


@dataclass(frozen=True, slots=True)
class FixedQIDeployment:
    """One exact PBH-admissible and QI-compatible finite deployment."""

    key: str
    service_index: int
    hardware_mask: int
    eta: tuple[int, ...]
    xi: tuple[int, ...]
    service_delays: ServiceDelayMatrix
    actuator_cost: float
    sensor_cost: float
    service_cost: float
    total_cost: float


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validated_budgets(values: Iterable[object]) -> tuple[int, ...]:
    try:
        raw = tuple(values)
    except TypeError as error:
        raise TypeError("budgets must be an iterable") from error
    if not raw:
        raise ValueError("budgets must be nonempty")
    normalized: list[int] = []
    for index, value in enumerate(raw):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, int):
            raise TypeError(f"budgets[{index}] must be a non-boolean integer")
        if value < 0:
            raise ValueError(f"budgets[{index}] must be nonnegative")
        normalized.append(int(value))
    if any(right <= left for left, right in zip(normalized, normalized[1:])):
        raise ValueError("budgets must be strictly increasing")
    return tuple(normalized)


def _service_payload(
    values: ServiceDelayMatrix | None,
) -> list[list[int | None]] | None:
    return None if values is None else [list(row) for row in values]


def _pbh_certificate_payload(
    certificate: PBHSelectionCertificate,
) -> dict[str, Any]:
    return {
        "kind": certificate.kind,
        "selection": list(certificate.selection),
        "selected_device_indices": list(certificate.selected_device_indices),
        "selected_matrix_indices": list(certificate.selected_matrix_indices),
        "stability_radius": certificate.stability_radius,
        "eigenvalue_tolerance": certificate.eigenvalue_tolerance,
        "rank_tolerance": certificate.rank_tolerance,
        "admissible": certificate.admissible,
        "minimum_rank": certificate.minimum_rank,
        "worst_minimum_singular_value": (
            certificate.worst_minimum_singular_value
        ),
        "modes": [asdict(mode) for mode in certificate.modes],
    }


def exhaustive_pbh_certificate(case: JointPlatoonCase) -> dict[str, Any]:
    """Return all 8 actuator and 128 grouped-sensor PBH checks."""

    if not isinstance(case, JointPlatoonCase):
        raise TypeError("case must be a JointPlatoonCase")
    actuator_records = tuple(
        stabilizability_certificate(
            case.plant.A,
            case.plant.B2,
            tuple((mask >> index) & 1 for index in range(case.plant.m)),
        )
        for mask in range(1 << case.plant.m)
    )
    sensor_count = case.deployment.layout.sensor_device_count
    sensor_records = tuple(
        detectability_certificate(
            case.plant.A,
            case.plant.C2,
            case.sensor_device_groups,
            tuple((mask >> index) & 1 for index in range(sensor_count)),
        )
        for mask in range(1 << sensor_count)
    )
    admissible_actuators = tuple(
        item.selection for item in actuator_records if item.admissible
    )
    admissible_sensors = tuple(
        item.selection for item in sensor_records if item.admissible
    )
    return {
        "scope": "all_hardware_subsets_at_every_unstable_discrete_time_mode",
        "actuator_pattern_count": len(actuator_records),
        "sensor_pattern_count": len(sensor_records),
        "stabilizable_actuator_pattern_count": len(admissible_actuators),
        "detectable_sensor_pattern_count": len(admissible_sensors),
        "admissible_actuator_patterns": [list(item) for item in admissible_actuators],
        "admissible_sensor_patterns": [list(item) for item in admissible_sensors],
        "actuator_certificates": [
            _pbh_certificate_payload(item) for item in actuator_records
        ],
        "sensor_certificates": [
            _pbh_certificate_payload(item) for item in sensor_records
        ],
    }


def enumerate_pbh_admissible_qi_deployments(
    case: JointPlatoonCase,
    pbh_payload: Mapping[str, Any] | None = None,
) -> tuple[
    tuple[FixedQIDeployment, ...],
    tuple[FixedServiceQICertificate, ...],
    dict[str, Any],
]:
    """Enumerate the complete PBH-admissible QI product in stable order."""

    if not isinstance(case, JointPlatoonCase):
        raise TypeError("case must be a JointPlatoonCase")
    pbh = exhaustive_pbh_certificate(case) if pbh_payload is None else pbh_payload
    eta_patterns = tuple(
        tuple(int(value) for value in item)
        for item in pbh["admissible_actuator_patterns"]
    )
    xi_patterns = tuple(
        tuple(int(value) for value in item)
        for item in pbh["admissible_sensor_patterns"]
    )
    service_records = tuple(iter_service_architectures(case.base))
    if len(service_records) != EXPECTED_SERVICE_MENU_COUNT:
        raise RuntimeError(
            f"expected {EXPECTED_SERVICE_MENU_COUNT} service menus, "
            f"observed {len(service_records)}"
        )
    propagation = grouped_plant_propagation_delays(
        case.plant.A,
        case.plant.B2,
        case.plant.C2,
        case.sensor_device_groups,
    )
    certificates = service_menu_qi_certificates(
        (item.service_delays for item in service_records),
        case.deployment.layout.actuator_sites,
        case.deployment.layout.sensor_device_sites,
        propagation,
    )
    count_summary = summarize_admissible_joint_qi(
        certificates,
        eta_patterns[0] if len(eta_patterns) == 1 else (1,) * case.plant.m,
        xi_patterns,
    )
    if len(eta_patterns) != 1:
        raise RuntimeError(
            "the declared platoon publication case must have one PBH-admissible "
            "actuator pattern"
        )

    deployments: list[FixedQIDeployment] = []
    eta = eta_patterns[0]
    for service_index, (record, certificate) in enumerate(
        zip(service_records, certificates, strict=True)
    ):
        for xi in xi_patterns:
            mask = hardware_mask(eta, xi)
            if not certificate.is_compatible(mask):
                continue
            choice = ArchitectureChoice(eta, xi, record.service_delays)
            cost = architecture_cost(case.deployment, choice)
            sensor_mask = sum(value << index for index, value in enumerate(xi))
            deployments.append(
                FixedQIDeployment(
                    key=f"qi_service_{service_index:04d}_sensor_{sensor_mask:03d}",
                    service_index=service_index,
                    hardware_mask=mask,
                    eta=eta,
                    xi=xi,
                    service_delays=record.service_delays,
                    actuator_cost=cost.actuator,
                    sensor_cost=cost.sensor,
                    service_cost=cost.service,
                    total_cost=cost.total,
                )
            )
    if len({item.key for item in deployments}) != len(deployments):
        raise RuntimeError("QI deployment keys are not unique")
    if len(deployments) != count_summary.joint_qi_count:
        raise RuntimeError("QI deployment enumeration disagrees with count certificate")
    summary = {
        "service_menu_count": len(service_records),
        "pbh_admissible_actuator_pattern_count": len(eta_patterns),
        "pbh_admissible_sensor_pattern_count": len(xi_patterns),
        "pbh_admissible_service_hardware_candidate_count": (
            len(service_records) * len(eta_patterns) * len(xi_patterns)
        ),
        "pbh_admissible_qi_deployment_count": len(deployments),
        "count_certificate": {
            "service_menu_count": count_summary.service_menu_count,
            "admissible_sensor_pattern_count": (
                count_summary.admissible_sensor_pattern_count
            ),
            "joint_candidate_count": count_summary.joint_candidate_count,
            "joint_qi_count": count_summary.joint_qi_count,
            "eta": list(count_summary.eta),
            "per_sensor_pattern": [
                {
                    "sensor_mask": item.sensor_mask,
                    "xi": list(item.xi),
                    "qi_service_menu_count": item.qi_service_menu_count,
                }
                for item in count_summary.per_sensor_pattern
            ],
        },
    }
    return tuple(deployments), certificates, summary


def _fixed_problem(
    case: JointPlatoonCase,
    *,
    eta: tuple[int, ...],
    xi: tuple[int, ...],
    service_delays: ServiceDelayMatrix,
    horizon: int,
) -> BPlusProblem:
    return BPlusProblem(
        plant=case.plant,
        deployment=case.deployment,
        horizon=horizon,
        fixed_eta=eta,
        fixed_xi=xi,
        fixed_service_delays=service_delays,
    )


def _audit_payload(audit: SolutionAudit) -> dict[str, Any]:
    return asdict(audit)


def _result_payload(
    problem: BPlusProblem,
    result: BPlusSolveResult,
    *,
    audit_tolerance: float = AUDIT_TOLERANCE,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": result.status,
        "solution_count": result.solution_count,
        "performance_h2": result.performance_objective,
        "objective_value": result.objective_value,
        "architecture_cost": result.architecture_cost,
        "architecture_breakdown": (
            None
            if result.architecture_breakdown is None
            else asdict(result.architecture_breakdown)
        ),
        "eta": None if result.eta is None else list(result.eta),
        "xi": None if result.xi is None else list(result.xi),
        "service_delays": _service_payload(result.service_delays),
        "best_bound": result.best_bound,
        "mip_gap": result.mip_gap,
        "runtime_seconds": result.runtime,
        "solver_settings": asdict(result.settings),
        "audit": None,
    }
    if result.solution_count > 0:
        payload["audit"] = _audit_payload(
            audit_solution(problem, result, tolerance=audit_tolerance)
        )
    return payload


def _numeric_tolerance(lower: float, upper: float) -> float:
    return max(
        CERTIFICATE_ABSOLUTE_TOLERANCE,
        CERTIFICATE_RELATIVE_TOLERANCE
        * max(1.0, abs(lower), abs(upper)),
    )


def _certificate_payload(
    incumbent: BPlusSolveResult,
    incumbent_payload: Mapping[str, Any],
    fixed: BPlusSolveResult | None,
    fixed_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    lower = incumbent.best_bound
    upper = None if fixed is None else fixed.performance_objective
    if lower is None or upper is None:
        return {
            "certified": False,
            "lower_bound": lower,
            "fixed_qp_upper_bound": upper,
            "signed_upper_minus_lower": None,
            "tolerance": None,
            "reason": "missing_lower_or_upper_bound",
        }
    tolerance = _numeric_tolerance(lower, upper)
    incumbent_audit = incumbent_payload.get("audit")
    fixed_audit = None if fixed_payload is None else fixed_payload.get("audit")
    signed_gap = upper - lower
    certified = (
        incumbent.status == "OPTIMAL"
        and incumbent.solution_count > 0
        and incumbent.mip_gap is not None
        and incumbent.mip_gap <= incumbent.settings.target_mip_gap + 1.0e-15
        and isinstance(incumbent_audit, Mapping)
        and incumbent_audit.get("certified") is True
        and fixed is not None
        and fixed.status == "OPTIMAL"
        and fixed.solution_count > 0
        and isinstance(fixed_audit, Mapping)
        and fixed_audit.get("certified") is True
        and abs(signed_gap) <= tolerance
    )
    return {
        "certified": certified,
        "lower_bound": lower,
        "fixed_qp_upper_bound": upper,
        "signed_upper_minus_lower": signed_gap,
        "order_violation": max(0.0, lower - upper),
        "tolerance": tolerance,
        "reason": "certified_interval" if certified else "interval_or_audit_failed",
    }


def _fixed_resynthesis(
    case: JointPlatoonCase,
    incumbent: BPlusSolveResult,
    *,
    horizon: int,
    budget: float,
    seed: int,
    threads: int,
    solver: SolveFunction,
) -> tuple[BPlusProblem, BPlusSolveResult] | None:
    if (
        incumbent.solution_count <= 0
        or incumbent.eta is None
        or incumbent.xi is None
        or incumbent.service_delays is None
    ):
        return None
    problem = _fixed_problem(
        case,
        eta=incumbent.eta,
        xi=incumbent.xi,
        service_delays=incumbent.service_delays,
        horizon=horizon,
    )
    options = joint_solver_options(
        seed=seed,
        threads=threads,
        architecture_budget=budget,
    )
    return problem, solver(problem, options)


def solve_and_certify_incumbent(
    case: JointPlatoonCase,
    problem: BPlusProblem,
    options: SolverOptions,
    *,
    solver: SolveFunction = solve_bplus,
    fixed_solver: SolveFunction = solve_bplus,
) -> dict[str, Any]:
    """Solve one MICP and certify it with an audited fixed-architecture solve."""

    if options.architecture_budget is None:
        raise ValueError("publication MICP must declare a combined architecture budget")
    incumbent = solver(problem, options)
    incumbent_payload = _result_payload(problem, incumbent)
    fixed_pair = _fixed_resynthesis(
        case,
        incumbent,
        horizon=problem.horizon,
        budget=options.architecture_budget,
        seed=options.seed,
        threads=options.threads,
        solver=fixed_solver,
    )
    if fixed_pair is None:
        fixed_result = None
        fixed_payload = None
    else:
        fixed_problem, fixed_result = fixed_pair
        fixed_payload = _result_payload(fixed_problem, fixed_result)
    return {
        "incumbent": incumbent_payload,
        "fixed_architecture_resynthesis": fixed_payload,
        "certificate": _certificate_payload(
            incumbent,
            incumbent_payload,
            fixed_result,
            fixed_payload,
        ),
    }


def solve_with_dual_reductions_disabled(
    problem: BPlusProblem,
    options: SolverOptions | None,
) -> BPlusSolveResult:
    """Resolve an ambiguous Gurobi status without changing the base model."""

    from .model import _build_model, _solve_built_model

    solver_options = SolverOptions() if options is None else options
    if not isinstance(solver_options, SolverOptions):
        raise TypeError("options must be SolverOptions")
    built = _build_model(problem, solver_options)
    built.model.Params.DualReductions = 0
    return _solve_built_model(problem, solver_options, built)


def solve_fixed_continuous_qp(
    problem: BPlusProblem,
    options: SolverOptions | None,
) -> BPlusSolveResult:
    """Solve a fully fixed deployment as a genuine continuous convex QP.

    The native formulation uses binary variables even when equality constraints
    prescribe every device and service choice.  Gurobi can consequently report
    an integer optimum while the continuous response block is numerically
    inaccurate.  This route first solves that fully prescribed discrete shell,
    invokes Gurobi's exact ``convertToFixed`` transformation, and then
    re-optimizes the resulting continuous QP with dual simplex.  No MIP start,
    response warm start, residual lifting, or extra cut is supplied.
    """

    from .model import _build_model, _solve_built_model

    if (
        problem.fixed_eta is None
        or problem.fixed_xi is None
        or problem.fixed_service_delays is None
    ):
        raise ValueError(
            "continuous fixed-QP solve requires fixed eta, xi, and service delays"
        )
    solver_options = SolverOptions() if options is None else options
    if not isinstance(solver_options, SolverOptions):
        raise TypeError("options must be SolverOptions")
    built = _build_model(problem, solver_options)
    prescribed_shell = _solve_built_model(problem, solver_options, built)
    if prescribed_shell.solution_count <= 0 and prescribed_shell.status == "INF_OR_UNBD":
        built.model.reset()
        built.model.Params.DualReductions = 0
        prescribed_shell = _solve_built_model(problem, solver_options, built)
    if prescribed_shell.solution_count <= 0:
        return prescribed_shell
    built.model.convertToFixed()
    built.model.Params.Method = 1
    built.model.Params.NumericFocus = solver_options.numeric_focus
    built.model.Params.FeasibilityTol = solver_options.feasibility_tolerance
    built.model.Params.OptimalityTol = solver_options.optimality_tolerance
    return _solve_built_model(problem, solver_options, built)


def _architecture_key(payload: Mapping[str, Any]) -> str:
    fixed = payload.get("fixed_architecture_resynthesis")
    if not isinstance(fixed, Mapping):
        return "no-fixed-architecture"
    return json.dumps(
        {
            "eta": fixed.get("eta"),
            "xi": fixed.get("xi"),
            "service_delays": fixed.get("service_delays"),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def build_tolerance_audited_envelope(
    raw_points: Sequence[Mapping[str, Any]],
    budgets: Sequence[int],
    *,
    key_prefix: str,
) -> list[dict[str, Any]]:
    """Mechanically form the nonincreasing hard-budget upper envelope.

    At budget ``B`` the upper-bound candidate set contains every certified
    fixed-architecture result generated at a source budget no larger than
    ``B`` whose independently recomputed realized cost is also no larger than
    ``B``.  The lower bound is the MICP bound solved directly at ``B``.
    """

    budget_values = _validated_budgets(budgets)
    by_budget = {int(point["budget"]): point for point in raw_points}
    if set(by_budget) != set(budget_values):
        raise ValueError("raw_points must contain exactly one point per budget")
    envelope: list[dict[str, Any]] = []
    previous_upper = math.inf
    for budget in budget_values:
        source = by_budget[budget]
        candidates: list[tuple[float, float, int, str, Mapping[str, Any]]] = []
        for point in raw_points:
            source_budget = int(point["budget"])
            certificate = point.get("certificate")
            fixed = point.get("fixed_architecture_resynthesis")
            if (
                source_budget > budget
                or not isinstance(certificate, Mapping)
                or certificate.get("certified") is not True
                or not isinstance(fixed, Mapping)
            ):
                continue
            performance = fixed.get("performance_h2")
            realized_cost = fixed.get("architecture_cost")
            if not isinstance(performance, (int, float)) or not isinstance(
                realized_cost, (int, float)
            ):
                continue
            if float(realized_cost) > budget + CERTIFICATE_ABSOLUTE_TOLERANCE:
                continue
            candidates.append(
                (
                    float(performance),
                    float(realized_cost),
                    source_budget,
                    _architecture_key(point),
                    point,
                )
            )
        if not candidates:
            envelope.append(
                {
                    "key": f"{key_prefix}_B{budget:02d}",
                    "budget": budget,
                    "certified": False,
                    "reason": "no_certified_fixed_qp_candidate",
                }
            )
            continue
        candidates.sort(key=lambda item: item[:4])
        upper, realized_cost, winner_budget, architecture_identity, winner = candidates[0]
        source_certificate = source.get("certificate")
        lower = (
            source_certificate.get("lower_bound")
            if isinstance(source_certificate, Mapping)
            else None
        )
        if not isinstance(lower, (int, float)):
            envelope_certified = False
            tolerance = None
            signed_gap = None
        else:
            tolerance = _numeric_tolerance(float(lower), upper)
            signed_gap = upper - float(lower)
            envelope_certified = abs(signed_gap) <= tolerance
        if upper > previous_upper + (
            CERTIFICATE_ABSOLUTE_TOLERANCE
            if tolerance is None
            else tolerance
        ):
            raise RuntimeError("constructed budget envelope is not nonincreasing")
        previous_upper = min(previous_upper, upper)
        envelope.append(
            {
                "key": f"{key_prefix}_B{budget:02d}",
                "budget": budget,
                "certified": envelope_certified,
                "candidate_rule": (
                    "minimum audited fixed-QP upper bound among all certified "
                    "source budgets <= current budget"
                ),
                "eligible_candidate_count": len(candidates),
                "winner_source_budget": winner_budget,
                "winner_architecture_identity": architecture_identity,
                "realized_total_cost": realized_cost,
                "performance_h2": upper,
                "lower_bound_at_budget": lower,
                "signed_upper_minus_lower": signed_gap,
                "tolerance": tolerance,
                "eta": winner["fixed_architecture_resynthesis"].get("eta"),
                "xi": winner["fixed_architecture_resynthesis"].get("xi"),
                "service_delays": winner["fixed_architecture_resynthesis"].get(
                    "service_delays"
                ),
            }
        )
    return envelope


def _new_payload(
    case: JointPlatoonCase,
    *,
    budgets: tuple[int, ...],
    horizon: int,
    seed: int,
    threads: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "joint_platoon_publication_pipeline",
        "configuration": {
            "seed": seed,
            "threads": threads,
            "horizon": horizon,
            "budgets": list(budgets),
            "dense_total_cost": DEFAULT_DENSE_COST,
            "audit_tolerance": AUDIT_TOLERANCE,
            "certificate_absolute_tolerance": CERTIFICATE_ABSOLUTE_TOLERANCE,
            "certificate_relative_tolerance": CERTIFICATE_RELATIVE_TOLERANCE,
            "solver_acceleration": {
                "residual_lifting": False,
                "warm_start": False,
                "screening": False,
                "hidden_cuts": False,
                "qi_constraints": "exact finite-catalog no-goods only",
            },
            "fixed_resynthesis_route": (
                "direct continuous OF-SLS QP in R/M/N/L with fixed hardware "
                "and service zeros; no binary shell"
            ),
        },
        "case": {
            "matrix_sha256": platoon_case_matrix_sha256(case),
            "sensor_device_labels": list(case.sensor_device_labels),
            "sensor_device_groups": [
                list(group) for group in case.sensor_device_groups
            ],
            "sensor_device_sites": list(
                case.deployment.layout.sensor_device_sites
            ),
            "actuator_device_labels": list(case.actuator_device_labels),
            "actuator_device_sites": list(case.deployment.layout.actuator_sites),
        },
        "pbh": None,
        "dense_reference": None,
        "proposed": {"raw_budget_points": [], "envelope": []},
        "horizon_sensitivity": {
            "budgets": list(HORIZON_SENSITIVITY_BUDGETS),
            "horizons": list(HORIZON_SENSITIVITY_HORIZONS),
            "points": [],
            "interpretation_boundary": (
                "H=10 is the intermediate reporting horizon; this finite check "
                "tests sensitivity and is not a convergence claim."
            ),
        },
        "qi": {
            "catalog_summary": None,
            "deployment_points": [],
            "coverage": None,
            "raw_budget_points": [],
            "envelope": [],
        },
        "baselines": {
            "rfd_provenance": None,
            "identity_count": None,
            "points": [],
        },
        "validation": {},
        "updated_at_utc": _utc_now(),
    }


def _load_or_initialize(
    output: Path,
    expected: dict[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    if not resume or not output.is_file():
        return expected
    loaded = json.loads(output.read_text(encoding="utf-8"))
    for key in ("schema_version", "experiment", "configuration", "case"):
        if loaded.get(key) != expected.get(key):
            raise ValueError(f"existing publication checkpoint differs at {key!r}")
    return loaded


def _checkpoint(output: Path, payload: dict[str, Any]) -> None:
    payload["updated_at_utc"] = _utc_now()
    for attempt in range(12):
        try:
            write_json_atomic(output, payload)
            return
        except PermissionError:
            if attempt == 11:
                raise
            # OneDrive can transiently hold the destination while indexing the
            # previous atomic replacement.  Retrying preserves the same
            # strict-JSON payload and never reads a partial checkpoint.
            time.sleep(min(0.1 * (2**attempt), 2.0))


def _solve_dense_reference(
    case: JointPlatoonCase,
    *,
    horizon: int,
    seed: int,
    threads: int,
    solver: SolveFunction,
) -> dict[str, Any]:
    eta = (1,) * case.plant.m
    xi = (1,) * case.deployment.layout.sensor_device_count
    service = platoon_dense_architecture(case.base)
    problem = _fixed_problem(
        case,
        eta=eta,
        xi=xi,
        service_delays=service,
        horizon=horizon,
    )
    cost = architecture_cost(case.deployment, ArchitectureChoice(eta, xi, service))
    if not math.isclose(
        cost.total, DEFAULT_DENSE_COST, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise RuntimeError(
            f"dense architecture cost must equal {DEFAULT_DENSE_COST}, got {cost.total}"
        )
    result = solver(
        problem,
        joint_solver_options(
            seed=seed,
            threads=threads,
            architecture_budget=float(DEFAULT_DENSE_COST),
        ),
    )
    result_payload = _result_payload(problem, result)
    return {
        "definition": "all devices and every optional service at its minimum delay",
        "total_cost": cost.total,
        "fixed_architecture_result": result_payload,
        "certified_feasible": (
            result.status == "OPTIMAL"
            and result.solution_count > 0
            and isinstance(result_payload.get("audit"), Mapping)
            and result_payload["audit"].get("certified") is True
        ),
    }


def _run_proposed(
    payload: dict[str, Any],
    output: Path,
    case: JointPlatoonCase,
    *,
    budgets: tuple[int, ...],
    horizon: int,
    seed: int,
    threads: int,
    solver: SolveFunction,
    fixed_solver: SolveFunction,
) -> None:
    points = list(payload["proposed"].get("raw_budget_points", []))
    completed = {int(point["budget"]) for point in points}
    problem = make_joint_problem(case, horizon=horizon)
    for budget in budgets:
        if budget in completed:
            continue
        options = joint_solver_options(
            seed=seed, threads=threads, architecture_budget=float(budget)
        )
        point = solve_and_certify_incumbent(
            case,
            problem,
            options,
            solver=solver,
            fixed_solver=fixed_solver,
        )
        point.update({"key": f"proposed_B{budget:02d}", "budget": budget})
        points.append(point)
        points.sort(key=lambda item: int(item["budget"]))
        payload["proposed"]["raw_budget_points"] = points
        _checkpoint(output, payload)
        print(
            json.dumps(
                {
                    "family": "proposed",
                    "budget": budget,
                    "certified": point["certificate"]["certified"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    payload["proposed"]["envelope"] = build_tolerance_audited_envelope(
        points, budgets, key_prefix="proposed"
    )
    _checkpoint(output, payload)


def _fixed_qi_point(
    case: JointPlatoonCase,
    deployment: FixedQIDeployment,
    *,
    horizon: int,
    seed: int,
    threads: int,
    solver: SolveFunction,
) -> dict[str, Any]:
    problem = _fixed_problem(
        case,
        eta=deployment.eta,
        xi=deployment.xi,
        service_delays=deployment.service_delays,
        horizon=horizon,
    )
    result = solver(
        problem,
        joint_solver_options(
            seed=seed,
            threads=threads,
            architecture_budget=deployment.total_cost,
        ),
    )
    result_payload = _result_payload(problem, result)
    certified = (
        result.status == "OPTIMAL"
        and result.solution_count > 0
        and isinstance(result_payload.get("audit"), Mapping)
        and result_payload["audit"].get("certified") is True
    )
    return {
        "key": deployment.key,
        "service_index": deployment.service_index,
        "hardware_mask": deployment.hardware_mask,
        "eta": list(deployment.eta),
        "xi": list(deployment.xi),
        "service_delays": _service_payload(deployment.service_delays),
        "architecture_breakdown": {
            "actuator": deployment.actuator_cost,
            "sensor": deployment.sensor_cost,
            "service": deployment.service_cost,
            "total": deployment.total_cost,
        },
        "fixed_architecture_result": result_payload,
        "certified_feasible": certified,
    }


def _qi_feasibility_witness(
    target: FixedQIDeployment,
    deployments: Sequence[FixedQIDeployment],
    by_key: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Find a certified deployment whose response support is a target subset."""

    candidates: list[tuple[int, str, FixedQIDeployment, list[dict[str, Any]]]] = []
    for witness in deployments:
        point = by_key.get(witness.key)
        if (
            witness.key == target.key
            or witness.eta != target.eta
            or witness.xi != target.xi
            or point is None
            or point.get("certified_feasible") is not True
        ):
            continue
        differences: list[dict[str, Any]] = []
        contained = True
        for destination, (target_row, witness_row) in enumerate(
            zip(target.service_delays, witness.service_delays, strict=True)
        ):
            for source, (target_delay, witness_delay) in enumerate(
                zip(target_row, witness_row, strict=True)
            ):
                if witness_delay is not None and (
                    target_delay is None or target_delay > witness_delay
                ):
                    contained = False
                    break
                if target_delay != witness_delay:
                    differences.append(
                        {
                            "destination": destination,
                            "source": source,
                            "target_delay": target_delay,
                            "witness_delay": witness_delay,
                        }
                    )
            if not contained:
                break
        if contained:
            candidates.append((len(differences), witness.key, witness, differences))
    if not candidates:
        return None
    _, key, _, differences = min(candidates, key=lambda item: (item[0], item[1]))
    witness_point = by_key[key]
    witness_result = witness_point["fixed_architecture_result"]
    return {
        "key": key,
        "relation": "witness response support is a subset of target response support",
        "elementwise_delay_differences": differences,
        "witness_status": witness_result["status"],
        "witness_audit_certified": witness_result["audit"]["certified"],
        "witness_max_response_residual": witness_result["audit"]["ofsls"][
            "max_residual"
        ],
        "scope": "linear-feasibility evidence only; not a target optimality certificate",
    }


def _run_qi_family(
    payload: dict[str, Any],
    output: Path,
    case: JointPlatoonCase,
    *,
    horizon: int,
    seed: int,
    threads: int,
    solver: SolveFunction,
) -> None:
    deployments, _, summary = enumerate_pbh_admissible_qi_deployments(
        case, payload["pbh"]
    )
    if len(deployments) != EXPECTED_PBH_ADMISSIBLE_QI_COUNT:
        raise RuntimeError(
            f"expected {EXPECTED_PBH_ADMISSIBLE_QI_COUNT} PBH-admissible QI "
            f"deployments, observed {len(deployments)}"
        )
    payload["qi"]["catalog_summary"] = summary
    points = list(payload["qi"].get("deployment_points", []))
    by_key = {str(point["key"]): point for point in points}
    expected_keys = {item.key for item in deployments}
    if not set(by_key).issubset(expected_keys):
        raise ValueError("QI checkpoint contains an unknown deployment key")
    new_since_checkpoint = 0
    for index, deployment in enumerate(deployments, start=1):
        previous = by_key.get(deployment.key)
        if previous is not None and previous.get("certified_feasible") is True:
            continue
        point = _fixed_qi_point(
            case,
            deployment,
            horizon=horizon,
            seed=seed,
            threads=threads,
            solver=solver,
        )
        if previous is not None:
            previous_result = previous.get("fixed_architecture_result", {})
            resolved_result = point.get("fixed_architecture_result", {})
            witness = _qi_feasibility_witness(deployment, deployments, by_key)
            point["numerical_resolution"] = {
                "original_status": previous_result.get("status"),
                "resolved_status": resolved_result.get("status"),
                "fresh_model": True,
                "dual_reductions": 0,
                "presolve": 0,
                "method": "barrier",
                "crossover": 1,
                "barrier_convergence_tolerance": 1.0e-10,
                "feasibility_witness": witness,
            }
        by_key[deployment.key] = point
        new_since_checkpoint += 1
        if new_since_checkpoint >= QI_CHECKPOINT_INTERVAL:
            points = [by_key[item.key] for item in deployments if item.key in by_key]
            payload["qi"]["deployment_points"] = points
            payload["qi"]["coverage"] = _qi_coverage(points, len(deployments))
            _checkpoint(output, payload)
            new_since_checkpoint = 0
            print(
                json.dumps(
                    {
                        "family": "qi_deployments",
                        "evaluated": len(points),
                        "expected": len(deployments),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    points = [by_key[item.key] for item in deployments]
    payload["qi"]["deployment_points"] = points
    payload["qi"]["coverage"] = _qi_coverage(points, len(deployments))
    _checkpoint(output, payload)


def _run_horizon_sensitivity(
    payload: dict[str, Any],
    output: Path,
    case: JointPlatoonCase,
    *,
    seed: int,
    threads: int,
    solver: SolveFunction,
    fixed_solver: SolveFunction,
) -> None:
    """Evaluate the declared 3-by-3 finite-horizon sensitivity grid."""

    points = list(payload["horizon_sensitivity"].get("points", []))
    by_key = {
        (int(point["budget"]), int(point["horizon"])): point for point in points
    }
    proposed_by_budget = {
        int(point["budget"]): point
        for point in payload["proposed"].get("raw_budget_points", [])
    }
    for budget in HORIZON_SENSITIVITY_BUDGETS:
        for horizon in HORIZON_SENSITIVITY_HORIZONS:
            key = (budget, horizon)
            if key in by_key:
                continue
            if horizon == DEFAULT_HORIZON:
                source = proposed_by_budget.get(budget)
                if source is None:
                    raise RuntimeError(
                        f"main proposed solve is missing at sensitivity budget {budget}"
                    )
                point = {
                    **source,
                    "key": f"horizon_T{horizon:02d}_B{budget:02d}",
                    "horizon": horizon,
                    "reused_from_main_run": True,
                }
            else:
                problem = make_joint_problem(case, horizon=horizon)
                options = joint_solver_options(
                    seed=seed,
                    threads=threads,
                    architecture_budget=float(budget),
                )
                point = solve_and_certify_incumbent(
                    case,
                    problem,
                    options,
                    solver=solver,
                    fixed_solver=fixed_solver,
                )
                point.update(
                    {
                        "key": f"horizon_T{horizon:02d}_B{budget:02d}",
                        "budget": budget,
                        "horizon": horizon,
                        "reused_from_main_run": False,
                    }
                )
            by_key[key] = point
            points = [
                by_key[item]
                for item in sorted(by_key, key=lambda value: (value[0], value[1]))
            ]
            payload["horizon_sensitivity"]["points"] = points
            _checkpoint(output, payload)
            print(
                json.dumps(
                    {
                        "family": "horizon_sensitivity",
                        "budget": budget,
                        "horizon": horizon,
                        "certified": point["certificate"]["certified"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    expected = {
        (budget, horizon)
        for budget in HORIZON_SENSITIVITY_BUDGETS
        for horizon in HORIZON_SENSITIVITY_HORIZONS
    }
    if set(by_key) != expected:
        raise RuntimeError("horizon-sensitivity grid is incomplete")
    payload["horizon_sensitivity"]["all_points_certified"] = all(
        point["certificate"]["certified"] is True for point in points
    )
    _checkpoint(output, payload)


def _qi_coverage(points: Sequence[Mapping[str, Any]], expected: int) -> dict[str, Any]:
    evaluated = len(points)
    certified = sum(point.get("certified_feasible") is True for point in points)
    complete = evaluated == expected
    return {
        "catalog_identity": "PBH-admissible hardware and QI service deployments",
        "expected_deployment_count": expected,
        "evaluated_deployment_count": evaluated,
        "certified_feasible_count": certified,
        "nonperformance_status_count": evaluated - certified,
        "complete": complete,
        "plot_label": (
            f"feasible QI deployments ({certified} of {expected})"
            if complete
            else f"evaluated feasible QI deployments ({certified} of {evaluated})"
        ),
    }


def _run_qi_budget_envelope(
    payload: dict[str, Any],
    output: Path,
    case: JointPlatoonCase,
    *,
    budgets: tuple[int, ...],
    horizon: int,
    seed: int,
    threads: int,
    fixed_solver: SolveFunction,
) -> None:
    problem = make_joint_problem(case, horizon=horizon)
    propagation = grouped_plant_propagation_delays(
        case.plant.A,
        case.plant.B2,
        case.plant.C2,
        case.sensor_device_groups,
    )
    points = list(payload["qi"].get("raw_budget_points", []))
    completed = {int(point["budget"]) for point in points}
    for budget in budgets:
        if budget in completed:
            continue
        options = joint_solver_options(
            seed=seed, threads=threads, architecture_budget=float(budget)
        )
        qi_result = solve_hardware_aware_qi(problem, propagation, options)
        incumbent = qi_result.solve
        incumbent_payload = _result_payload(problem, incumbent)
        fixed_pair = _fixed_resynthesis(
            case,
            incumbent,
            horizon=horizon,
            budget=float(budget),
            seed=seed,
            threads=threads,
            solver=fixed_solver,
        )
        if fixed_pair is None:
            fixed_result = None
            fixed_payload = None
        else:
            fixed_problem, fixed_result = fixed_pair
            fixed_payload = _result_payload(fixed_problem, fixed_result)
        certificate = _certificate_payload(
            incumbent,
            incumbent_payload,
            fixed_result,
            fixed_payload,
        )
        certificate["certified"] = bool(
            certificate["certified"] and qi_result.verified_qi is True
        )
        point = {
            "key": f"qi_budget_B{budget:02d}",
            "budget": budget,
            "verified_hardware_aware_qi": qi_result.verified_qi,
            "exact_no_good_count": qi_result.no_good_count,
            "incumbent": incumbent_payload,
            "fixed_architecture_resynthesis": fixed_payload,
            "certificate": certificate,
        }
        points.append(point)
        points.sort(key=lambda item: int(item["budget"]))
        payload["qi"]["raw_budget_points"] = points
        _checkpoint(output, payload)
        print(
            json.dumps(
                {
                    "family": "hardware_aware_qi",
                    "budget": budget,
                    "certified": certificate["certified"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    payload["qi"]["envelope"] = build_tolerance_audited_envelope(
        points, budgets, key_prefix="qi"
    )
    _checkpoint(output, payload)


def _baseline_point(
    case: JointPlatoonCase,
    outcome: JointBaselineOutcome,
    *,
    seed: int,
    threads: int,
    fixed_solver: SolveFunction,
    cache: dict[tuple[Any, ...], tuple[BPlusSolveResult, dict[str, Any]]],
    ambiguity_solver: SolveFunction,
    status_cache: dict[tuple[Any, ...], tuple[BPlusSolveResult, dict[str, Any]]],
) -> dict[str, Any]:
    spec = outcome.spec
    problem = make_joint_baseline_problem(case, spec, horizon=outcome.horizon)
    incumbent = outcome.result
    incumbent_payload = _result_payload(problem, incumbent)
    fixed_result: BPlusSolveResult | None = None
    fixed_payload: dict[str, Any] | None = None
    if (
        incumbent.solution_count > 0
        and incumbent.eta is not None
        and incumbent.xi is not None
        and incumbent.service_delays is not None
    ):
        signature = (
            outcome.architecture_budget,
            incumbent.eta,
            incumbent.xi,
            incumbent.service_delays,
        )
        cached = cache.get(signature)
        if cached is None:
            pair = _fixed_resynthesis(
                case,
                incumbent,
                horizon=outcome.horizon,
                budget=outcome.architecture_budget,
                seed=seed,
                threads=threads,
                solver=fixed_solver,
            )
            if pair is None:
                raise RuntimeError("solution-bearing baseline could not be fixed")
            fixed_problem, fixed_result = pair
            fixed_payload = _result_payload(fixed_problem, fixed_result)
            cache[signature] = (fixed_result, fixed_payload)
        else:
            fixed_result, fixed_payload = cached
    certificate = _certificate_payload(
        incumbent,
        incumbent_payload,
        fixed_result,
        fixed_payload,
    )
    repaired_qi_valid = (
        spec.layer != "qi_repaired" or outcome.selected_hardware_qi is True
    )
    status_resolution: dict[str, Any] | None = None
    if spec.family == "rfd" and spec.layer == "raw":
        resolution_key = (
            outcome.architecture_budget,
            spec.service_delays,
        )
        if incumbent.status == "INF_OR_UNBD":
            resolved = status_cache.get(resolution_key)
            if resolved is None:
                options = joint_solver_options(
                    seed=seed,
                    threads=threads,
                    architecture_budget=outcome.architecture_budget,
                )
                resolved_result = ambiguity_solver(problem, options)
                resolved_payload = _result_payload(problem, resolved_result)
                status_cache[resolution_key] = (resolved_result, resolved_payload)
            else:
                resolved_result, resolved_payload = resolved
            status_resolution = {
                "performed": True,
                "method": "Gurobi DualReductions=0 on the unchanged fixed-service problem",
                "original_status": incumbent.status,
                "resolved_status": resolved_result.status,
                "resolved_result": resolved_payload,
                "certified_infeasible": resolved_result.status == "INFEASIBLE",
            }
        else:
            status_resolution = {
                "performed": False,
                "method": None,
                "original_status": incumbent.status,
                "resolved_status": incumbent.status,
                "resolved_result": None,
                "certified_infeasible": incumbent.status == "INFEASIBLE",
            }
    plot_eligible = bool(
        certificate["certified"]
        and repaired_qi_valid
        and not (spec.family == "rfd" and spec.layer == "raw")
    )
    return {
        "key": f"{spec.key}:B{int(outcome.architecture_budget):02d}",
        "budget": int(outcome.architecture_budget),
        "identity": spec.key,
        "source_identity": spec.source_key,
        "family": spec.family,
        "layer": spec.layer,
        "fixed_service_delays": _service_payload(spec.service_delays),
        "regularization": spec.regularization,
        "threshold": spec.threshold,
        "repair_changed": spec.repair_changed,
        "repair_iterations": spec.repair_iterations,
        "selected_hardware_qi": outcome.selected_hardware_qi,
        "raw_rfd_policy": (
            (
                "certified_infeasible_status_only"
                if status_resolution is not None
                and status_resolution["certified_infeasible"] is True
                else "no_incumbent_status_only"
            )
            if spec.family == "rfd" and spec.layer == "raw"
            else None
        ),
        "status_resolution": status_resolution,
        "incumbent": incumbent_payload,
        "fixed_architecture_resynthesis": fixed_payload,
        "certificate": certificate,
        "plot_eligible": plot_eligible,
    }


def _run_baselines(
    payload: dict[str, Any],
    output: Path,
    repository_root: Path,
    case: JointPlatoonCase,
    *,
    budgets: tuple[int, ...],
    horizon: int,
    seed: int,
    threads: int,
    solver: SolveFunction,
    fixed_solver: SolveFunction,
    ambiguity_solver: SolveFunction,
) -> None:
    canonical = joint_canonical_baseline_specs(case)
    rfd, provenance = load_joint_rfd_baseline_specs(case, repository_root)
    specs: tuple[JointBaselineSpec, ...] = (*canonical, *rfd)
    payload["baselines"]["rfd_provenance"] = provenance
    payload["baselines"]["identity_count"] = len(specs)
    points = list(payload["baselines"].get("points", []))
    completed = {
        int(budget)
        for budget in budgets
        if sum(int(point["budget"]) == budget for point in points) == len(specs)
    }
    fixed_cache: dict[
        tuple[Any, ...], tuple[BPlusSolveResult, dict[str, Any]]
    ] = {}
    status_cache: dict[
        tuple[Any, ...], tuple[BPlusSolveResult, dict[str, Any]]
    ] = {}
    for budget in budgets:
        if budget in completed:
            continue
        options = joint_solver_options(
            seed=seed, threads=threads, architecture_budget=float(budget)
        )
        outcomes = solve_joint_baselines(
            case,
            specs,
            architecture_budget=float(budget),
            horizon=horizon,
            options=options,
            solver=solver,
            strict_repaired_qi=True,
        )
        budget_points = [
            _baseline_point(
                case,
                outcome,
                seed=seed,
                threads=threads,
                fixed_solver=fixed_solver,
                cache=fixed_cache,
                ambiguity_solver=ambiguity_solver,
                status_cache=status_cache,
            )
            for outcome in outcomes
        ]
        points.extend(budget_points)
        points.sort(key=lambda item: (int(item["budget"]), str(item["identity"])))
        payload["baselines"]["points"] = points
        _checkpoint(output, payload)
        print(
            json.dumps(
                {
                    "family": "canonical_rfd",
                    "budget": budget,
                    "identity_count": len(budget_points),
                    "certified_count": sum(
                        point["certificate"]["certified"]
                        for point in budget_points
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )


def _best_qi_family_by_budget(
    points: Sequence[Mapping[str, Any]], budgets: Sequence[int]
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for budget in budgets:
        eligible: list[tuple[float, float, str]] = []
        for point in points:
            if point.get("certified_feasible") is not True:
                continue
            breakdown = point.get("architecture_breakdown")
            fixed = point.get("fixed_architecture_result")
            if not isinstance(breakdown, Mapping) or not isinstance(fixed, Mapping):
                continue
            cost = breakdown.get("total")
            performance = fixed.get("performance_h2")
            if (
                isinstance(cost, (int, float))
                and isinstance(performance, (int, float))
                and float(cost) <= budget + CERTIFICATE_ABSOLUTE_TOLERANCE
            ):
                eligible.append((float(performance), float(cost), str(point["key"])))
        eligible.sort()
        results.append(
            {
                "budget": budget,
                "eligible_certified_count": len(eligible),
                "best_performance_h2": None if not eligible else eligible[0][0],
                "realized_total_cost": None if not eligible else eligible[0][1],
                "argmin_key": None if not eligible else eligible[0][2],
            }
        )
    return results


def _final_validation(payload: dict[str, Any], budgets: tuple[int, ...]) -> None:
    dense = payload.get("dense_reference")
    if not isinstance(dense, Mapping) or dense.get("certified_feasible") is not True:
        raise RuntimeError("dense fixed-architecture reference is not audited feasible")
    dense_result = dense.get("fixed_architecture_result")
    if not isinstance(dense_result, Mapping) or not isinstance(
        dense_result.get("performance_h2"), (int, float)
    ):
        raise RuntimeError("dense reference has no performance value")
    dense_h2 = float(dense_result["performance_h2"])
    comparisons: list[dict[str, Any]] = []
    for family in ("proposed", "qi"):
        envelope = payload[family]["envelope"]
        if len(envelope) != len(budgets):
            raise RuntimeError(f"{family} envelope is incomplete")
        if any(point.get("certified") is not True for point in envelope):
            raise RuntimeError(f"{family} envelope contains an uncertified budget")
        for point in envelope:
            signed_loss = float(point["performance_h2"]) - dense_h2
            tolerance = _numeric_tolerance(float(point["performance_h2"]), dense_h2)
            if signed_loss < -tolerance:
                raise RuntimeError(
                    f"{family} point is materially better than the dense reference"
                )
            point["signed_dense_reference_loss"] = signed_loss
            point["plot_delta_J_perf"] = (
                0.0 if abs(signed_loss) <= tolerance else signed_loss
            )
            point["dense_difference_clamped_to_zero"] = abs(signed_loss) <= tolerance
            point["rho"] = float(point["realized_total_cost"]) / DEFAULT_DENSE_COST

    qi_points = payload["qi"]["deployment_points"]
    family_optima = _best_qi_family_by_budget(qi_points, budgets)
    coverage = payload["qi"]["coverage"]
    complete_coverage = isinstance(coverage, Mapping) and coverage.get("complete") is True
    if complete_coverage:
        for catalog, envelope in zip(
            family_optima, payload["qi"]["envelope"], strict=True
        ):
            if catalog["best_performance_h2"] is None:
                raise RuntimeError("complete QI catalog has no feasible budget winner")
            difference = float(catalog["best_performance_h2"]) - float(
                envelope["performance_h2"]
            )
            tolerance = _numeric_tolerance(
                float(catalog["best_performance_h2"]),
                float(envelope["performance_h2"]),
            )
            if abs(difference) > tolerance:
                raise RuntimeError(
                    "complete fixed-QI catalog disagrees with the exact QI MICP envelope"
                )
            comparisons.append(
                {
                    "budget": catalog["budget"],
                    "catalog_minus_qi_micp": difference,
                    "tolerance": tolerance,
                    "consistent": True,
                }
            )
    payload["qi"]["catalog_budget_optima"] = family_optima
    horizon_points = payload["horizon_sensitivity"].get("points", [])
    if len(horizon_points) != (
        len(HORIZON_SENSITIVITY_BUDGETS) * len(HORIZON_SENSITIVITY_HORIZONS)
    ):
        raise RuntimeError("horizon-sensitivity grid is incomplete")
    baseline_points = payload["baselines"].get("points", [])
    expected_baseline_count = (
        int(payload["baselines"]["identity_count"]) * len(budgets)
    )
    if len(baseline_points) != expected_baseline_count:
        raise RuntimeError("canonical/RFD common-budget sweep is incomplete")
    raw_rfd = [
        point
        for point in baseline_points
        if point.get("family") == "rfd" and point.get("layer") == "raw"
    ]
    raw_rfd_resolved_infeasible = sum(
        isinstance(point.get("status_resolution"), Mapping)
        and point["status_resolution"].get("certified_infeasible") is True
        for point in raw_rfd
    )
    raw_rfd_no_incumbent = len(raw_rfd) - raw_rfd_resolved_infeasible
    payload["validation"] = {
        "complete": True,
        "dense_reference_h2": dense_h2,
        "proposed_budget_count": len(payload["proposed"]["envelope"]),
        "qi_budget_count": len(payload["qi"]["envelope"]),
        "qi_catalog_complete": complete_coverage,
        "qi_catalog_vs_exact_qi_micp": comparisons,
        "horizon_sensitivity_point_count": len(horizon_points),
        "horizon_sensitivity_all_certified": all(
            point["certificate"]["certified"] is True for point in horizon_points
        ),
        "baseline_point_count": len(baseline_points),
        "raw_rfd_identity_count": len(
            {str(point["source_identity"]) for point in raw_rfd}
        ),
        "raw_rfd_budget_status_count": len(raw_rfd),
        "raw_rfd_resolved_infeasible_count": raw_rfd_resolved_infeasible,
        "raw_rfd_no_incumbent_count": raw_rfd_no_incumbent,
        "raw_rfd_points_plotted": 0,
        "numeric_policy": (
            "signed dense-reference differences with magnitude no larger than "
            "the declared mixed absolute-relative tolerance are plotted at zero"
        ),
    }


def freeze_existing_publication(output: Path) -> dict[str, Any]:
    """Freeze conservative status metadata without launching another solver."""

    output_path = Path(output).resolve()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    case = make_joint_platoon_case(seed=DEFAULT_SEED)
    deployments, _, _ = enumerate_pbh_admissible_qi_deployments(
        case, payload["pbh"]
    )
    by_key = {
        str(point["key"]): point for point in payload["qi"]["deployment_points"]
    }
    deployment_by_key = {deployment.key: deployment for deployment in deployments}
    diagnostics = [
        point for point in by_key.values() if point.get("certified_feasible") is not True
    ]
    for point in diagnostics:
        key = str(point["key"])
        resolution = dict(point.get("numerical_resolution") or {})
        resolution.setdefault(
            "original_status", point["fixed_architecture_result"]["status"]
        )
        resolution.setdefault(
            "resolved_status", point["fixed_architecture_result"]["status"]
        )
        resolution["feasibility_witness"] = _qi_feasibility_witness(
            deployment_by_key[key], deployments, by_key
        )
        resolution["additional_optimality_resolve"] = {
            "execution_status": "NOT_EXECUTED_EXECUTION_WINDOW_LIMIT",
            "requested_qp_variants": [
                {
                    "dual_reductions": 0,
                    "presolve": presolve,
                    "method": "barrier",
                    "barrier_homogeneous": 1,
                    "crossover": 1,
                    "barrier_convergence_tolerance": 1.0e-10,
                }
                for presolve in (0, 1)
            ],
            "policy": (
                "diagnostic only; excluded from performance plots and budget "
                "optima unless a later OPTIMAL-and-audited result is recorded"
            ),
        }
        point["numerical_resolution"] = resolution
    coverage = _qi_coverage(payload["qi"]["deployment_points"], len(deployments))
    payload["qi"]["coverage"] = coverage
    validation = dict(payload["validation"])
    validation.update(
        {
            "qi_logical_catalog_count": len(deployments),
            "qi_fixed_qp_optimal_and_audited_count": coverage[
                "certified_feasible_count"
            ],
            "qi_fixed_qp_infeasible_count": sum(
                point["fixed_architecture_result"]["status"] == "INFEASIBLE"
                for point in diagnostics
            ),
            "qi_fixed_qp_numeric_or_suboptimal_count": len(diagnostics),
            "qi_diagnostic_feasibility_witness_count": sum(
                point["numerical_resolution"]["feasibility_witness"] is not None
                for point in diagnostics
            ),
            "qi_hard_budget_all_certified": all(
                point["certificate"]["certified"] is True
                for point in payload["qi"]["raw_budget_points"]
            ),
            "qi_additional_anomaly_resolve_status": (
                "NOT_EXECUTED_EXECUTION_WINDOW_LIMIT"
            ),
        }
    )
    payload["validation"] = validation
    _checkpoint(output_path, payload)
    return payload


def run_joint_publication(
    *,
    repository_root: Path,
    output: Path = DEFAULT_OUTPUT,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    horizon: int = DEFAULT_HORIZON,
    seed: int = DEFAULT_SEED,
    threads: int = DEFAULT_THREADS,
    resume: bool = True,
    solver: SolveFunction = solve_bplus,
    fixed_solver: SolveFunction = solve_fixed_qp,
    ambiguity_solver: SolveFunction = solve_with_dual_reductions_disabled,
) -> dict[str, Any]:
    """Run or resume the complete joint-platoon publication computation."""

    root = Path(repository_root).resolve()
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = root / output_path
    budget_values = _validated_budgets(budgets)
    if (seed, threads, horizon) != (
        DEFAULT_SEED,
        DEFAULT_THREADS,
        DEFAULT_HORIZON,
    ):
        raise ValueError("publication settings must be Seed=23, Threads=1, H=10")
    if budget_values != DEFAULT_BUDGETS:
        raise ValueError("publication budgets must contain every integer from 14 to 35")

    case = make_joint_platoon_case(seed=seed)
    expected = _new_payload(
        case,
        budgets=budget_values,
        horizon=horizon,
        seed=seed,
        threads=threads,
    )
    payload = _load_or_initialize(output_path, expected, resume=resume)
    if payload.get("pbh") is None:
        payload["pbh"] = exhaustive_pbh_certificate(case)
        if (
            payload["pbh"]["stabilizable_actuator_pattern_count"]
            != EXPECTED_PBH_ACTUATOR_COUNT
            or payload["pbh"]["detectable_sensor_pattern_count"]
            != EXPECTED_PBH_SENSOR_COUNT
        ):
            raise RuntimeError("PBH-admissible hardware counts changed")
        _checkpoint(output_path, payload)
    if payload.get("dense_reference") is None:
        payload["dense_reference"] = _solve_dense_reference(
            case,
            horizon=horizon,
            seed=seed,
            threads=threads,
            solver=fixed_solver,
        )
        _checkpoint(output_path, payload)

    _run_proposed(
        payload,
        output_path,
        case,
        budgets=budget_values,
        horizon=horizon,
        seed=seed,
        threads=threads,
        solver=solver,
        fixed_solver=fixed_solver,
    )
    _run_horizon_sensitivity(
        payload,
        output_path,
        case,
        seed=seed,
        threads=threads,
        solver=solver,
        fixed_solver=fixed_solver,
    )
    _run_qi_family(
        payload,
        output_path,
        case,
        horizon=horizon,
        seed=seed,
        threads=threads,
        solver=fixed_solver,
    )
    _run_qi_budget_envelope(
        payload,
        output_path,
        case,
        budgets=budget_values,
        horizon=horizon,
        seed=seed,
        threads=threads,
        fixed_solver=fixed_solver,
    )
    _run_baselines(
        payload,
        output_path,
        root,
        case,
        budgets=budget_values,
        horizon=horizon,
        seed=seed,
        threads=threads,
        solver=solver,
        fixed_solver=fixed_solver,
        ambiguity_solver=ambiguity_solver,
    )
    _final_validation(payload, budget_values)
    _checkpoint(output_path, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.repository_root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    run_joint_publication(
        repository_root=root,
        output=output,
        horizon=args.horizon,
        seed=args.seed,
        threads=args.threads,
        resume=not args.no_resume,
    )
    if not args.no_plot:
        from .joint_plotting import generate_joint_publication_figure

        generate_joint_publication_figure(
            publication_source=output,
            output_directory=output.parent,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
