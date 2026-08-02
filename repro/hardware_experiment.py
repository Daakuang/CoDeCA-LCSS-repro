"""Exhaustive and MICP validation for independent hardware selection.

Run from the repository root with ``PYTHONPATH=LcssV1.1/V2``::

    python -m repro.hardware_experiment --seed 23 --threads 1 \
        --output results/lcss_v2/hardware_selection

The exhaustive order is lexicographic in the six-bit tuple
``eta[0:3] + xi[0:3]``.  Each fixed-device solve uses the same native B+
model as the free-device MICP; the former is a QP after its binary decisions
are fixed by equality constraints.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import io
import itertools
import json
import math
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence

import numpy as np

from .cases import (
    HARDWARE_MATRIX_ORDER,
    HardwareSelectionCase,
    hardware_case_matrix_sha256,
    make_hardware_selection_case,
    single_device_pbh_diagnostics,
)
from .diagnostics import SolutionAudit, audit_solution
from .model import BPlusProblem, BPlusSolveResult, SolverOptions, solve_bplus


PUBLICATION_HORIZONS: Final[tuple[int, ...]] = (8, 10, 12)
HARDWARE_BUDGETS: Final[tuple[int, ...]] = (2, 3, 4, 5, 6)
VALIDATION_BUDGETS: Final[tuple[int, ...]] = (2, 4, 6)
OBJECTIVE_TOLERANCE: Final[float] = 1.0e-8
AUDIT_TOLERANCE: Final[float] = 1.0e-8
DECLARED_MIP_GAP: Final[float] = 1.0e-9


@dataclass(frozen=True, slots=True, order=True)
class HardwareChoice:
    """Independent actuator and sensor decisions in canonical bit order."""

    eta: tuple[int, int, int]
    xi: tuple[int, int, int]

    def __post_init__(self) -> None:
        for name, values in (("eta", self.eta), ("xi", self.xi)):
            if len(values) != 3 or any(value not in (0, 1) for value in values):
                raise ValueError(f"{name} must contain exactly three binary values")

    @property
    def hardware_cost(self) -> int:
        return int(sum(self.eta) + sum(self.xi))


@dataclass(frozen=True, slots=True)
class AuditCertificate:
    certified: bool
    ofsls: float
    objective_error: float
    total_objective_error: float
    hardware: float
    service: float
    architecture_cost_error: float
    architecture_breakdown_error: float
    budget_violation: float
    relative_ofsls: float
    response_max: float


@dataclass(frozen=True, slots=True)
class EnumerationRecord:
    horizon: int
    eta: tuple[int, int, int]
    xi: tuple[int, int, int]
    hardware_cost: int
    service_delays: tuple[tuple[int, ...], ...]
    feasible: bool
    status: str
    h2_objective: float | None
    total_objective: float | None
    architecture_cost: float | None
    best_bound: float | None
    mip_gap: float | None
    runtime: float
    solution_count: int
    audit: AuditCertificate | None

    @property
    def choice(self) -> HardwareChoice:
        return HardwareChoice(eta=self.eta, xi=self.xi)


@dataclass(frozen=True, slots=True)
class BudgetPoint:
    horizon: int
    gamma: int
    best_h2: float | None
    argmin_choices: tuple[HardwareChoice, ...]


@dataclass(frozen=True, slots=True)
class MICPValidation:
    horizon: int
    gamma: int
    status: str
    eta: tuple[int, int, int]
    xi: tuple[int, int, int]
    hardware_cost: int
    h2_objective: float
    enumeration_best_h2: float
    objective_mismatch: float
    returned_choice_in_argmin: bool
    best_bound: float | None
    mip_gap: float | None
    runtime: float
    audit_certified: bool
    audit: AuditCertificate
    passed: bool


@dataclass(frozen=True, slots=True)
class HardwareExperimentRun:
    enumerations: Mapping[int, tuple[EnumerationRecord, ...]]
    envelopes: Mapping[int, Mapping[int, BudgetPoint]]
    validations: Mapping[int, MICPValidation]
    total_fixed_solve_runtime: float
    total_micp_runtime: float


def hardware_choices() -> tuple[HardwareChoice, ...]:
    """Return all 64 choices, lexicographic in ``eta + xi`` (zero first)."""

    choices: list[HardwareChoice] = []
    for bits in itertools.product((0, 1), repeat=6):
        choices.append(
            HardwareChoice(
                eta=(bits[0], bits[1], bits[2]),
                xi=(bits[3], bits[4], bits[5]),
            )
        )
    return tuple(choices)


def _solver_options(
    *, seed: int, threads: int, architecture_budget: float | None = None
) -> SolverOptions:
    return SolverOptions(
        seed=seed,
        threads=threads,
        numeric_focus=3,
        output_flag=0,
        feasibility_tolerance=1.0e-9,
        optimality_tolerance=1.0e-9,
        integer_feasibility_tolerance=1.0e-9,
        mip_gap=DECLARED_MIP_GAP,
        architecture_budget=architecture_budget,
    )


def _problem(
    case: HardwareSelectionCase,
    *,
    horizon: int,
    choice: HardwareChoice | None,
) -> BPlusProblem:
    return BPlusProblem(
        plant=case.plant,
        deployment=case.deployment,
        horizon=horizon,
        fixed_eta=None if choice is None else choice.eta,
        fixed_xi=None if choice is None else choice.xi,
        fixed_service_delays=case.fixed_service_delays,
    )


def _max_abs(values: np.ndarray) -> float:
    return 0.0 if values.size == 0 else float(np.max(np.abs(values)))


def _response_scale_diagnostics(
    problem: BPlusProblem, result: BPlusSolveResult
) -> tuple[float, float]:
    """Return max response magnitude and a term-scaled recursion residual.

    For each of the four OF-SLS recursions at every lag, the relative residual
    is ``max_abs(residual) / max(1, sum(max_abs(each equation term)))``.  This
    is evaluated independently from the Gurobi constraint objects.
    """

    if result.responses is None:
        raise ValueError("response diagnostics require an incumbent")
    response = result.responses
    plant = problem.plant
    response_max = max(
        _max_abs(response.R),
        _max_abs(response.M),
        _max_abs(response.N),
        _max_abs(response.L),
    )
    ratios: list[float] = []
    zero_nn = np.zeros((plant.n, plant.n))
    zero_mn = np.zeros((plant.m, plant.n))
    zero_np = np.zeros((plant.n, plant.p))
    for lag in range(problem.horizon + 1):
        R_next = response.R[lag + 1] if lag < problem.horizon else zero_nn
        M_next = response.M[lag + 1] if lag < problem.horizon else zero_mn
        N_next = response.N[lag + 1] if lag < problem.horizon else zero_np
        impulse = np.eye(plant.n) if lag == 0 else zero_nn
        equations = (
            (R_next, plant.A @ response.R[lag], plant.B2 @ response.M[lag], impulse),
            (N_next, plant.A @ response.N[lag], plant.B2 @ response.L[lag]),
            (R_next, response.R[lag] @ plant.A, response.N[lag] @ plant.C2, impulse),
            (M_next, response.M[lag] @ plant.A, response.L[lag] @ plant.C2),
        )
        signs = ((1.0, -1.0, -1.0, -1.0),) * 4
        for terms, equation_signs in zip(equations, signs):
            residual = sum(
                (sign * term for sign, term in zip(equation_signs, terms)),
                np.zeros_like(terms[0]),
            )
            scale = max(1.0, math.fsum(_max_abs(term) for term in terms))
            ratios.append(_max_abs(residual) / scale)
    return response_max, max(ratios, default=0.0)


def _certificate(
    audit: SolutionAudit, problem: BPlusProblem, result: BPlusSolveResult
) -> AuditCertificate:
    response_max, relative_ofsls = _response_scale_diagnostics(problem, result)
    return AuditCertificate(
        certified=audit.certified,
        ofsls=audit.ofsls.max_residual,
        objective_error=audit.objective.absolute_error,
        total_objective_error=audit.total_objective.absolute_error,
        hardware=audit.hardware.max_abs,
        service=audit.service.max_abs,
        architecture_cost_error=audit.architecture_cost.absolute_error,
        architecture_breakdown_error=audit.architecture_cost.breakdown_max_error,
        budget_violation=audit.budget_violation,
        relative_ofsls=relative_ofsls,
        response_max=response_max,
    )


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    normalized = float(value)
    return normalized if math.isfinite(normalized) else None


def _three_bits(values: tuple[int, ...] | None, *, name: str) -> tuple[int, int, int]:
    if values is None or len(values) != 3:
        raise RuntimeError(f"solution-bearing result has invalid {name}")
    normalized = (int(values[0]), int(values[1]), int(values[2]))
    if any(value not in (0, 1) for value in normalized):
        raise RuntimeError(f"solution-bearing result has nonbinary {name}")
    return normalized


def solve_fixed_hardware(
    case: HardwareSelectionCase,
    choice: HardwareChoice,
    *,
    horizon: int,
    seed: int = 23,
    threads: int = 1,
) -> EnumerationRecord:
    """Solve and independently audit one fixed-hardware FIR problem."""

    problem = _problem(case, horizon=horizon, choice=choice)
    result = solve_bplus(problem, _solver_options(seed=seed, threads=threads))
    feasible = result.solution_count > 0
    certificate: AuditCertificate | None = None
    if feasible:
        certificate = _certificate(
            audit_solution(problem, result, tolerance=AUDIT_TOLERANCE),
            problem,
            result,
        )
    return EnumerationRecord(
        horizon=horizon,
        eta=choice.eta,
        xi=choice.xi,
        hardware_cost=choice.hardware_cost,
        service_delays=case.fixed_service_delays,
        feasible=feasible,
        status=result.status,
        h2_objective=_finite_or_none(result.performance_objective),
        total_objective=_finite_or_none(result.objective_value),
        architecture_cost=_finite_or_none(result.architecture_cost),
        best_bound=_finite_or_none(result.best_bound),
        mip_gap=_finite_or_none(result.mip_gap),
        runtime=float(result.runtime),
        solution_count=result.solution_count,
        audit=certificate,
    )


def enumerate_hardware(
    case: HardwareSelectionCase,
    *,
    horizon: int,
    seed: int = 23,
    threads: int = 1,
) -> tuple[EnumerationRecord, ...]:
    """Solve all fixed-device cases in the frozen lexicographic order."""

    return tuple(
        solve_fixed_hardware(
            case, choice, horizon=horizon, seed=seed, threads=threads
        )
        for choice in hardware_choices()
    )


def budget_envelope(
    records: Sequence[EnumerationRecord],
    *,
    budgets: Sequence[int] = HARDWARE_BUDGETS,
    objective_tolerance: float = OBJECTIVE_TOLERANCE,
) -> Mapping[int, BudgetPoint]:
    """Return best feasible performance under ``hardware_cost <= gamma``."""

    if not records:
        raise ValueError("records must not be empty")
    if isinstance(objective_tolerance, (bool, np.bool_)) or not isinstance(
        objective_tolerance, Real
    ):
        raise TypeError("objective_tolerance must be a finite nonnegative real")
    normalized_tolerance = float(objective_tolerance)
    if not math.isfinite(normalized_tolerance) or normalized_tolerance < 0.0:
        raise ValueError("objective_tolerance must be a finite nonnegative real")
    try:
        raw_budgets = tuple(budgets)
    except TypeError as error:
        raise TypeError("budgets must be an iterable of nonnegative integers") from error
    if not raw_budgets:
        raise ValueError("budgets must not be empty")
    normalized_budgets: list[int] = []
    for index, value in enumerate(raw_budgets):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise TypeError(f"budgets[{index}] must be a nonnegative integer")
        gamma = int(value)
        if gamma < 0:
            raise ValueError(f"budgets[{index}] must be nonnegative")
        if gamma in normalized_budgets:
            raise ValueError(f"budgets contains duplicate value {gamma}")
        normalized_budgets.append(gamma)
    for index, record in enumerate(records):
        if not isinstance(record, EnumerationRecord):
            raise TypeError("records must contain EnumerationRecord values")
        if record.feasible or record.solution_count > 0:
            value = record.h2_objective
            prefix = f"feasible/solution-bearing record {index} h2_objective"
            if value is None:
                raise ValueError(f"{prefix} must not be None")
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise TypeError(f"{prefix} must be a finite real")
            if not math.isfinite(float(value)):
                raise ValueError(f"{prefix} must be a finite real")
    horizons = {record.horizon for record in records}
    if len(horizons) != 1:
        raise ValueError("records must share one FIR horizon")
    horizon = next(iter(horizons))
    points: dict[int, BudgetPoint] = {}
    for gamma in normalized_budgets:
        eligible = tuple(
            record
            for record in records
            if record.feasible
            and record.h2_objective is not None
            and record.hardware_cost <= gamma
        )
        if not eligible:
            points[gamma] = BudgetPoint(
                horizon=horizon,
                gamma=gamma,
                best_h2=None,
                argmin_choices=(),
            )
            continue
        best_h2 = min(float(record.h2_objective) for record in eligible)
        argmin = tuple(
            sorted(
                record.choice
                for record in eligible
                if abs(float(record.h2_objective) - best_h2)
                <= normalized_tolerance
            )
        )
        points[gamma] = BudgetPoint(
            horizon=horizon,
            gamma=gamma,
            best_h2=best_h2,
            argmin_choices=argmin,
        )
    return MappingProxyType(points)


def _solve_budget_micp(
    case: HardwareSelectionCase,
    envelope: Mapping[int, BudgetPoint],
    *,
    horizon: int,
    gamma: int,
    seed: int,
    threads: int,
) -> MICPValidation:
    point = envelope[gamma]
    if point.best_h2 is None or not point.argmin_choices:
        raise RuntimeError(f"enumeration has no feasible point for Gamma={gamma}")
    problem = _problem(case, horizon=horizon, choice=None)
    result = solve_bplus(
        problem,
        _solver_options(
            seed=seed, threads=threads, architecture_budget=float(gamma)
        ),
    )
    if result.solution_count <= 0 or result.performance_objective is None:
        raise RuntimeError(
            f"budget MICP Gamma={gamma} returned no incumbent: {result.status}"
        )
    eta = _three_bits(result.eta, name="eta")
    xi = _three_bits(result.xi, name="xi")
    choice = HardwareChoice(eta=eta, xi=xi)
    audit = _certificate(
        audit_solution(problem, result, tolerance=AUDIT_TOLERANCE), problem, result
    )
    h2 = float(result.performance_objective)
    mismatch = abs(h2 - point.best_h2)
    in_argmin = choice in point.argmin_choices
    gap = _finite_or_none(result.mip_gap)
    passed = bool(
        mismatch <= OBJECTIVE_TOLERANCE
        and in_argmin
        and choice.hardware_cost <= gamma
        and gap is not None
        and gap <= DECLARED_MIP_GAP
        and audit.certified
    )
    return MICPValidation(
        horizon=horizon,
        gamma=gamma,
        status=result.status,
        eta=eta,
        xi=xi,
        hardware_cost=choice.hardware_cost,
        h2_objective=h2,
        enumeration_best_h2=point.best_h2,
        objective_mismatch=mismatch,
        returned_choice_in_argmin=in_argmin,
        best_bound=_finite_or_none(result.best_bound),
        mip_gap=gap,
        runtime=float(result.runtime),
        audit_certified=audit.certified,
        audit=audit,
        passed=passed,
    )


def _record_json(record: EnumerationRecord) -> dict[str, Any]:
    return {
        "horizon": record.horizon,
        "eta": list(record.eta),
        "xi": list(record.xi),
        "hardware_cost": record.hardware_cost,
        "service_delays": [list(row) for row in record.service_delays],
        "feasible": record.feasible,
        "status": record.status,
        "h2_objective": record.h2_objective,
        "total_objective": record.total_objective,
        "architecture_cost": record.architecture_cost,
        "best_bound": record.best_bound,
        "mip_gap": record.mip_gap,
        "runtime": record.runtime,
        "solution_count": record.solution_count,
        "audit": None if record.audit is None else asdict(record.audit),
    }


def _validation_json(validation: MICPValidation) -> dict[str, Any]:
    values = asdict(validation)
    values["eta"] = list(validation.eta)
    values["xi"] = list(validation.xi)
    return values


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _write_json(path: Path, data: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _case_artifact(case: HardwareSelectionCase) -> dict[str, Any]:
    plant = case.plant
    pbh = [asdict(item) for item in single_device_pbh_diagnostics(case)]
    return {
        "schema_version": 1,
        "name": "three_node_scalar_coupled_lqg_chain",
        "seed": case.seed,
        "provenance": case.provenance,
        "matrix_order": list(HARDWARE_MATRIX_ORDER),
        "matrix_sha256": hardware_case_matrix_sha256(case),
        "dimensions": {
            "n": plant.n,
            "m": plant.m,
            "p": plant.p,
            "nw": plant.nw,
            "nz": plant.nz,
        },
        "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(plant.A)))),
        "matrices": {
            name: np.asarray(getattr(plant, name), dtype=float).tolist()
            for name in HARDWARE_MATRIX_ORDER
        },
        "layout": {
            "actuator_sites": list(case.deployment.layout.actuator_sites),
            "sensor_sites": list(case.deployment.layout.sensor_sites),
            "beta_sites": list(case.deployment.layout.beta_sites),
            "state_block_sizes": list(case.deployment.layout.state_block_sizes),
        },
        "actuator_costs": case.deployment.costs.actuator_costs.tolist(),
        "sensor_costs": case.deployment.costs.sensor_costs.tolist(),
        "fixed_service_delays": [list(row) for row in case.fixed_service_delays],
        "service_cost": 0.0,
        "single_device_pbh": pbh,
    }


def _budget_rows(
    envelopes: Mapping[int, Mapping[int, BudgetPoint]]
) -> tuple[dict[str, str | int | float], ...]:
    rows: list[dict[str, str | int | float]] = []
    for horizon in sorted(envelopes):
        points = envelopes[horizon]
        dense = points[6].best_h2
        for gamma in sorted(points):
            point = points[gamma]
            canonical = point.argmin_choices[0] if point.argmin_choices else None
            loss = (
                None
                if point.best_h2 is None or dense is None
                else 100.0 * (point.best_h2 / dense - 1.0)
            )
            rows.append(
                {
                    "horizon": horizon,
                    "gamma": gamma,
                    "best_h2": "" if point.best_h2 is None else point.best_h2,
                    "dense_h2": "" if dense is None else dense,
                    "performance_loss_percent": "" if loss is None else loss,
                    "argmin_count": len(point.argmin_choices),
                    "canonical_eta": "" if canonical is None else "".join(map(str, canonical.eta)),
                    "canonical_xi": "" if canonical is None else "".join(map(str, canonical.xi)),
                    "argmin_choices": "|".join(
                        f"{''.join(map(str, choice.eta))}/{''.join(map(str, choice.xi))}"
                        for choice in point.argmin_choices
                    ),
                }
            )
    return tuple(rows)


def _write_budget_csv(
    path: Path, envelopes: Mapping[int, Mapping[int, BudgetPoint]]
) -> None:
    fieldnames = [
        "horizon",
        "gamma",
        "best_h2",
        "dense_h2",
        "performance_loss_percent",
        "argmin_count",
        "canonical_eta",
        "canonical_xi",
        "argmin_choices",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(_budget_rows(envelopes))
    _atomic_write_text(path, stream.getvalue())


def _audit_maxima(
    certificates: Sequence[AuditCertificate],
) -> dict[str, float]:
    fields = (
        "ofsls",
        "objective_error",
        "total_objective_error",
        "hardware",
        "service",
        "architecture_cost_error",
        "architecture_breakdown_error",
        "budget_violation",
        "relative_ofsls",
        "response_max",
    )
    return {
        field: max((float(getattr(item, field)) for item in certificates), default=0.0)
        for field in fields
    }


def _validated_certificates(
    enumerations: Mapping[int, tuple[EnumerationRecord, ...]],
    envelopes: Mapping[int, Mapping[int, BudgetPoint]],
    validations: Mapping[int, MICPValidation],
) -> tuple[AuditCertificate, ...]:
    certificates: list[AuditCertificate] = []
    for horizon, points in envelopes.items():
        reported_choices = {
            choice for point in points.values() for choice in point.argmin_choices
        }
        certificates.extend(
            record.audit
            for record in enumerations[horizon]
            if record.choice in reported_choices and record.audit is not None
        )
    certificates.extend(validation.audit for validation in validations.values())
    return tuple(certificates)


def _write_artifacts(
    *,
    output: Path,
    case: HardwareSelectionCase,
    horizons: tuple[int, ...],
    seed: int,
    threads: int,
    validation_horizon: int,
    enumerations: Mapping[int, tuple[EnumerationRecord, ...]],
    envelopes: Mapping[int, Mapping[int, BudgetPoint]],
    validations: Mapping[int, MICPValidation],
) -> None:
    import gurobipy as gp

    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "case.json", _case_artifact(case))
    _write_json(
        output / "enumeration.json",
        {
            "schema_version": 1,
            "choice_order": "lexicographic eta[0:3] then xi[0:3], zero before one",
            "horizons": list(horizons),
            "records": [
                _record_json(record)
                for horizon in horizons
                for record in enumerations[horizon]
            ],
        },
    )
    _write_budget_csv(output / "budget_summary.csv", envelopes)
    _write_json(
        output / "solver_manifest.json",
        {
            "schema_version": 1,
            "seed": seed,
            "threads": threads,
            "horizons": list(horizons),
            "validation_horizon": validation_horizon,
            "fixed_qp_count": sum(len(records) for records in enumerations.values()),
            "micp_count": len(validations),
            "gurobi_version": ".".join(str(part) for part in gp.gurobi.version()),
            "settings": {
                "NumericFocus": 3,
                "FeasibilityTol": 1.0e-9,
                "OptimalityTol": 1.0e-9,
                "IntFeasTol": 1.0e-9,
                "MIPGap": DECLARED_MIP_GAP,
            },
        },
    )
    all_certificates = tuple(
        record.audit
        for records in enumerations.values()
        for record in records
        if record.audit is not None
    ) + tuple(validation.audit for validation in validations.values())
    validated_certificates = _validated_certificates(
        enumerations, envelopes, validations
    )
    audit_maxima = _audit_maxima(all_certificates)
    validated_audit_maxima = _audit_maxima(validated_certificates)
    _write_json(
        output / "validation.json",
        {
            "schema_version": 1,
            "validation_horizon": validation_horizon,
            "validated_budgets": list(VALIDATION_BUDGETS),
            "objective_tolerance": OBJECTIVE_TOLERANCE,
            "declared_mip_gap": DECLARED_MIP_GAP,
            "validations": [
                _validation_json(validations[gamma])
                for gamma in VALIDATION_BUDGETS
            ],
            "mismatch_max": max(
                validation.objective_mismatch for validation in validations.values()
            ),
            "audit_maxima": audit_maxima,
            "validated_audit_maxima": validated_audit_maxima,
            "all_passed": bool(
                all(validation.passed for validation in validations.values())
                and all(certificate.certified for certificate in validated_certificates)
            ),
        },
    )


def run_hardware_experiment(
    *,
    seed: int,
    threads: int,
    output: Path,
    horizons: Sequence[int] = PUBLICATION_HORIZONS,
    validation_horizon: int = 8,
) -> HardwareExperimentRun:
    """Run exhaustive fixed QPs and the three budget-constrained MICP checks."""

    normalized_horizons = tuple(int(horizon) for horizon in horizons)
    if not normalized_horizons or len(set(normalized_horizons)) != len(normalized_horizons):
        raise ValueError("horizons must be a nonempty sequence of unique values")
    if any(horizon < 2 for horizon in normalized_horizons):
        raise ValueError("each horizon must be at least two")
    if validation_horizon not in normalized_horizons:
        raise ValueError("validation_horizon must belong to horizons")
    case = make_hardware_selection_case(seed=seed)
    enumeration_data = {
        horizon: enumerate_hardware(
            case, horizon=horizon, seed=seed, threads=threads
        )
        for horizon in normalized_horizons
    }
    envelope_data = {
        horizon: budget_envelope(enumeration_data[horizon])
        for horizon in normalized_horizons
    }
    validation_data = {
        gamma: _solve_budget_micp(
            case,
            envelope_data[validation_horizon],
            horizon=validation_horizon,
            gamma=gamma,
            seed=seed,
            threads=threads,
        )
        for gamma in VALIDATION_BUDGETS
    }
    enumerations = MappingProxyType(enumeration_data)
    envelopes = MappingProxyType(envelope_data)
    validations = MappingProxyType(validation_data)
    _write_artifacts(
        output=Path(output),
        case=case,
        horizons=normalized_horizons,
        seed=seed,
        threads=threads,
        validation_horizon=validation_horizon,
        enumerations=enumerations,
        envelopes=envelopes,
        validations=validations,
    )
    return HardwareExperimentRun(
        enumerations=enumerations,
        envelopes=envelopes,
        validations=validations,
        total_fixed_solve_runtime=math.fsum(
            record.runtime
            for records in enumerations.values()
            for record in records
        ),
        total_micp_runtime=math.fsum(
            validation.runtime for validation in validations.values()
        ),
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--threads", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    run = run_hardware_experiment(
        seed=arguments.seed,
        threads=arguments.threads,
        output=arguments.output,
    )
    if not all(validation.passed for validation in run.validations.values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
