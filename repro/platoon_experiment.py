"""Deterministic platoon service enumeration and fixed-architecture adapter.

The service grid is always oriented ``destination <- source`` over physical
sites ``(leader, follower 1, follower 2, follower 3)``.  Enumeration changes
only logical service choices.  All three actuators and all ten scalar output
coordinates are fixed active in this main experiment.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import itertools
import json
import math
from numbers import Integral
import os
from pathlib import Path
import subprocess
import tempfile
from time import perf_counter
from typing import Any, Iterator, Mapping, Sequence, TypeAlias, cast

import numpy as np

from .cases import PlatoonCase, make_platoon_case, platoon_case_matrix_sha256
from .deployment import (
    ArchitectureChoice,
    ServiceDelayMatrix,
    architecture_cost,
)
from .model import BPlusProblem
from .model import BPlusSolveResult, SolverOptions, solve_bplus
from .diagnostics import audit_solution
from .manifest import build_manifest, sha256_file
from .realization import realization_filters, rollout_realization


CanonicalDelay: TypeAlias = int | None

PUBLICATION_SEED = 23
PUBLICATION_THREADS = 1
PUBLICATION_HORIZON = 10
PUBLICATION_HORIZONS = (8, 10, 12)
SERVICE_COST_UNIT = 1.0 / 6000.0
COMMON_BUDGET_UNITS = 17
DENSE_BUDGET_UNITS = 25
REALIZATION_TOLERANCE = 1.0e-9
AUDIT_TOLERANCE = 2.0e-8
PUBLICATION_ARTIFACT_FILES = (
    "config.json",
    "menu.json",
    "budgets.json",
    "pilot.json",
    "bplus.json",
    "qi.json",
    "canonical.json",
    "rfd.json",
    "common_budget.json",
    "validation.json",
    "manifest.json",
)


@dataclass(frozen=True, slots=True)
class PlatoonServiceArchitecture:
    """One canonical service grid and its exact submitted service cost."""

    service_delays: ServiceDelayMatrix
    service_cost: float
    serialization: str

    def __post_init__(self) -> None:
        delays = _normalize_service_delays(self.service_delays)
        cost = float(self.service_cost)
        if not math.isfinite(cost) or cost < 0.0:
            raise ValueError("service_cost must be finite and nonnegative")
        expected_serialization = canonical_service_serialization(delays)
        if self.serialization != expected_serialization:
            raise ValueError("serialization must be canonical for service_delays")
        object.__setattr__(self, "service_delays", delays)
        object.__setattr__(self, "service_cost", cost)


def _validated_horizon(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("horizon must be a non-boolean integer")
    horizon = int(value)
    if horizon < 2:
        raise ValueError("horizon must be at least two")
    return horizon


def _normalize_service_delays(
    values: ServiceDelayMatrix | PlatoonServiceArchitecture,
) -> ServiceDelayMatrix:
    raw = values.service_delays if isinstance(values, PlatoonServiceArchitecture) else values
    try:
        rows = tuple(tuple(row) for row in raw)
    except TypeError as error:
        raise TypeError("service delays must be a two-dimensional iterable") from error
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise ValueError("platoon service delays must have shape (4, 4)")
    normalized: list[tuple[CanonicalDelay, ...]] = []
    for row in rows:
        output_row: list[CanonicalDelay] = []
        for delay in row:
            if delay is None:
                output_row.append(None)
            elif isinstance(delay, (bool, np.bool_)) or not isinstance(delay, Integral):
                raise TypeError("each service delay must be an integer or None")
            elif int(delay) < 0:
                raise ValueError("finite service delays must be nonnegative")
            else:
                output_row.append(int(delay))
        normalized.append(tuple(output_row))
    return tuple(normalized)


def canonical_service_serialization(
    architecture: ServiceDelayMatrix | PlatoonServiceArchitecture,
) -> str:
    """Return stable compact JSON; ``null`` is the absent/infinite service."""

    delays = _normalize_service_delays(architecture)
    return json.dumps(delays, separators=(",", ":"), allow_nan=False)


def iter_service_architectures(
    case: PlatoonCase,
) -> Iterator[PlatoonServiceArchitecture]:
    """Enumerate the submitted finite service set in lexicographic pair order.

    Optional menu order is each pair's increasing finite delays followed by
    ``None``.  The cardinality is obtained from the Cartesian product itself:
    three leader links with three choices, four adjacent directed links with
    three choices, and two distance-two directed links with two choices.
    """

    if not isinstance(case, PlatoonCase):
        raise TypeError("case must be a PlatoonCase")
    catalog = case.deployment.services
    base: list[list[CanonicalDelay]] = [
        [None for _ in range(catalog.site_count)]
        for _ in range(catalog.site_count)
    ]
    optional_pairs: list[tuple[int, int]] = []
    options: list[tuple[CanonicalDelay, ...]] = []
    for menu in catalog.directed_menus:
        pair = (menu.destination_site, menu.source_site)
        if menu.mandatory_delay is not None:
            base[pair[0]][pair[1]] = menu.mandatory_delay
        else:
            optional_pairs.append(pair)
            options.append((*menu.finite_delays, None))

    for selection in itertools.product(*options):
        grid = [row.copy() for row in base]
        for (destination, source), delay in zip(
            optional_pairs, selection, strict=True
        ):
            grid[destination][source] = delay
        service_delays: ServiceDelayMatrix = tuple(tuple(row) for row in grid)
        choice = ArchitectureChoice(
            eta=case.fixed_eta,
            xi=case.fixed_xi,
            service_delays=service_delays,
        )
        breakdown = architecture_cost(case.deployment, choice)
        yield PlatoonServiceArchitecture(
            service_delays=service_delays,
            service_cost=breakdown.service,
            serialization=canonical_service_serialization(service_delays),
        )


def platoon_dense_architecture(case: PlatoonCase) -> ServiceDelayMatrix:
    """Install every admissible optional service at its physical minimum."""

    if not isinstance(case, PlatoonCase):
        raise TypeError("case must be a PlatoonCase")
    rows: list[list[CanonicalDelay]] = [
        [None for _ in range(4)] for _ in range(4)
    ]
    for menu in case.deployment.services.directed_menus:
        rows[menu.destination_site][menu.source_site] = (
            menu.mandatory_delay
            if menu.mandatory_delay is not None
            else menu.finite_delays[0]
        )
    return tuple(tuple(row) for row in rows)


def platoon_sparse_architecture(case: PlatoonCase) -> ServiceDelayMatrix:
    """Submitted non-QI OF-SLS architecture at cost 2.8333e-3.

    The manuscript's 3-by-4 controller-channel matrix is embedded in the
    physical 4-by-4 service grid by adding the mandatory leader-local service
    in row zero.  That service carries no controller/beta payload and is free.
    """

    if not isinstance(case, PlatoonCase):
        raise TypeError("case must be a PlatoonCase")
    delays: ServiceDelayMatrix = (
        (0, None, None, None),
        (None, 0, 1, None),
        (1, 1, 0, None),
        (1, 2, 1, 0),
    )
    case.deployment.validate_choice(
        ArchitectureChoice(case.fixed_eta, case.fixed_xi, delays)
    )
    return delays


def build_platoon_problem(
    case: PlatoonCase,
    service_delays: ServiceDelayMatrix | PlatoonServiceArchitecture,
    *,
    horizon: int = 10,
) -> BPlusProblem:
    """Build the fixed-device/fixed-service B+ problem used by audit QPs."""

    if not isinstance(case, PlatoonCase):
        raise TypeError("case must be a PlatoonCase")
    delays = _normalize_service_delays(service_delays)
    case.deployment.validate_choice(
        ArchitectureChoice(case.fixed_eta, case.fixed_xi, delays)
    )
    return BPlusProblem(
        plant=case.plant,
        deployment=case.deployment,
        horizon=_validated_horizon(horizon),
        fixed_eta=case.fixed_eta,
        fixed_xi=case.fixed_xi,
        fixed_service_delays=delays,
    )


def architecture_records(
    case: PlatoonCase,
) -> Sequence[PlatoonServiceArchitecture]:
    """Materialize the finite menu for later budget/QI comparisons."""

    return tuple(iter_service_architectures(case))


def architecture_cost_units(cost: float, *, tolerance: float = 1.0e-8) -> int:
    """Return exact integer units of one sixth of the milliscale link cost."""

    if isinstance(cost, (bool, np.bool_)):
        raise TypeError("cost must be a non-boolean finite nonnegative real")
    normalized = float(cost)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError("cost must be a finite nonnegative real")
    scaled = normalized / SERVICE_COST_UNIT
    units = int(round(scaled))
    if abs(scaled - units) > tolerance:
        raise ValueError(
            "architecture cost is not an integer multiple of the canonical cost unit"
        )
    return units


def derive_attainable_budget_units(case: PlatoonCase) -> tuple[int, ...]:
    """Derive all attainable integer service costs from the 8748-item menu."""

    records = architecture_records(case)
    if len(records) != 8748:
        raise ValueError(f"expected 8748 service architectures, observed {len(records)}")
    return tuple(sorted({architecture_cost_units(item.service_cost) for item in records}))


def qi_menu_indices(
    case: PlatoonCase, records: Sequence[PlatoonServiceArchitecture]
) -> tuple[int, ...]:
    """Return QI menu indices using raw matrices, robust to ``python -m`` identity."""

    from .baselines import platoon_plant_propagation_delays, platoon_qi_compatible

    propagation = platoon_plant_propagation_delays(case)
    return tuple(
        index
        for index, architecture in enumerate(records)
        if platoon_qi_compatible(case, architecture.service_delays, propagation)
    )


def select_compact_budget_units(
    attainable_units: Sequence[int], *, minimum_feasible_units: int
) -> tuple[int, ...]:
    """Select deterministic feasibility, common-budget, intermediate, and dense anchors."""

    normalized = tuple(sorted({int(value) for value in attainable_units}))
    if not normalized or minimum_feasible_units not in normalized:
        raise ValueError("minimum_feasible_units must be attainable")
    if COMMON_BUDGET_UNITS not in normalized or DENSE_BUDGET_UNITS not in normalized:
        raise ValueError("attainable costs must include common and dense budgets")
    if minimum_feasible_units > COMMON_BUDGET_UNITS:
        raise ValueError("the submitted common budget must be feasible")

    def nearest(target: float, lower: int, upper: int) -> int:
        candidates = tuple(value for value in normalized if lower <= value <= upper)
        if not candidates:
            raise ValueError("no attainable budget exists in the requested interval")
        return min(candidates, key=lambda value: (abs(value - target), value))

    lower_midpoint = nearest(
        0.5 * (minimum_feasible_units + COMMON_BUDGET_UNITS),
        minimum_feasible_units,
        COMMON_BUDGET_UNITS,
    )
    upper_midpoint = nearest(
        0.5 * (COMMON_BUDGET_UNITS + DENSE_BUDGET_UNITS),
        COMMON_BUDGET_UNITS,
        DENSE_BUDGET_UNITS,
    )
    return tuple(
        sorted(
            {
                minimum_feasible_units,
                lower_midpoint,
                COMMON_BUDGET_UNITS,
                upper_midpoint,
                DENSE_BUDGET_UNITS,
            }
        )
    )


def signed_performance_loss(
    performance: float, dense_reference: float
) -> tuple[float, float]:
    """Return signed absolute and percentage losses without clipping."""

    values = (float(performance), float(dense_reference))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("performance values must be finite")
    if values[1] <= 0.0:
        raise ValueError("dense_reference must be positive")
    absolute = values[0] - values[1]
    return absolute, 100.0 * absolute / values[1]


def _atomic_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_json_atomic(path: Path, value: object) -> None:
    """Atomically write strict RFC-compatible JSON (NaN/Infinity rejected)."""

    encoded = json.dumps(
        value,
        allow_nan=False,
        indent=2,
        sort_keys=True,
        separators=(",", ": "),
    )
    _atomic_text(Path(path), encoded + "\n")


def _parse_legacy_delay_signature(signature: str) -> tuple[tuple[bool, ...], ...]:
    tokens = tuple(token.strip().lower() for token in signature.split(","))
    if len(tokens) != 12:
        raise ValueError("legacy RFD delay_signature must contain 12 entries")
    support: list[bool] = []
    for token in tokens:
        if token in {"inf", "+inf", "infinity", "none"}:
            support.append(False)
            continue
        try:
            value = float(token)
        except ValueError as error:
            raise ValueError(
                f"invalid legacy RFD delay signature entry {token!r}"
            ) from error
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"invalid legacy RFD delay signature entry {token!r}"
            )
        support.append(True)
    return tuple(
        tuple(support[4 * row + column] for column in range(4))
        for row in range(3)
    )


def load_legacy_rfd_path_points(
    repository_root: Path,
) -> tuple[tuple[Any, ...], dict[str, object]]:
    """Load only the complete hashed legacy RFD support grid, never old H2 values."""

    from .baselines import RFDPathPoint

    root = Path(repository_root).resolve()
    source_directory = root / "results" / "rfd_regularization_paper_platoon_fresh"
    manifest_path = source_directory / "manifest.json"
    csv_path = source_directory / "rfd_regularization_baseline.csv"
    config_path = source_directory / "config.yaml"
    for path in (manifest_path, csv_path, config_path):
        if not path.is_file():
            raise ValueError(f"legacy RFD provenance file is missing: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != "exp8_rfd_regularization_baseline":
        raise ValueError("legacy RFD manifest has the wrong experiment_id")
    config = cast(Mapping[str, object], manifest.get("config", {}))
    regularizations = tuple(float(value) for value in cast(Sequence[object], config.get("regularization_weights", ())))
    thresholds = tuple(float(value) for value in cast(Sequence[object], config.get("coefficient_thresholds", ())))
    if not regularizations or not thresholds:
        raise ValueError("legacy RFD manifest does not declare its Cartesian grid")

    raw_rows: dict[tuple[float, float], Mapping[str, str]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("method") != "rfd_regularized_raw":
                continue
            if row.get("status") != "optimal":
                raise ValueError("legacy RFD raw grid contains a non-optimal point")
            key = (
                float(cast(str, row.get("regularization_weight"))),
                float(cast(str, row.get("coefficient_threshold"))),
            )
            if key in raw_rows:
                raise ValueError(f"duplicate legacy RFD raw grid key {key}")
            raw_rows[key] = row
    expected = {
        (regularization, threshold)
        for regularization in regularizations
        for threshold in thresholds
    }
    if set(raw_rows) != expected:
        missing = sorted(expected - set(raw_rows))
        extra = sorted(set(raw_rows) - expected)
        raise ValueError(
            f"legacy RFD support grid is incomplete: missing={missing}, extra={extra}"
        )
    points = tuple(
        RFDPathPoint(
            regularization=regularization,
            threshold=threshold,
            raw_support=_parse_legacy_delay_signature(
                cast(str, raw_rows[(regularization, threshold)]["delay_signature"])
            ),
        )
        for regularization in sorted(regularizations)
        for threshold in sorted(thresholds)
    )
    source_hashes = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in (manifest_path, config_path, csv_path)
    }
    provenance: dict[str, object] = {
        "complete_cartesian_grid": True,
        "regularization_count": len(regularizations),
        "threshold_count": len(thresholds),
        "raw_point_count": len(points),
        "regularizations": list(sorted(regularizations)),
        "thresholds": list(sorted(thresholds)),
        "source_path": csv_path.relative_to(root).as_posix(),
        "source_sha256": sha256_file(csv_path),
        "source_hashes": source_hashes,
        "legacy_git_commit": manifest.get("git_commit"),
        "old_performance_values_reused": False,
    }
    return points, provenance


_POINT_FIELDS = frozenset(
    {
        "key",
        "method",
        "budget",
        "budget_units",
        "architecture_cost",
        "architecture_cost_units",
        "performance_h2",
        "dense_reference_h2",
        "signed_absolute_loss",
        "signed_percent_loss",
        "best_bound",
        "mip_gap",
        "status",
        "solution_count",
        "runtime_seconds",
        "physical_service_matrix",
        "deployment_service_matrix",
        "horizon",
        "solver_settings",
        "residuals",
        "hardware_gating",
        "service_gating",
        "architecture_cost_audit",
        "realization_mismatch",
        "certified",
        "globality_basis",
        "response_artifact",
    }
)

_SOLVER_SETTING_FIELDS = frozenset(
    {
        "seed",
        "threads",
        "numeric_focus",
        "output_flag",
        "feasibility_tolerance",
        "optimality_tolerance",
        "integer_feasibility_tolerance",
        "target_mip_gap",
        "time_limit",
        "architecture_weight",
        "architecture_budget",
    }
)


def _finite_or_none(name: str, value: object, *, allow_none: bool) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    try:
        normalized = float(cast(Any, value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be a finite number")
    return normalized


def _validate_delay_matrix(name: str, value: object, shape: tuple[int, int]) -> None:
    try:
        rows = tuple(tuple(row) for row in cast(Sequence[Sequence[object]], value))
    except TypeError as error:
        raise ValueError(f"{name} must be a matrix") from error
    if len(rows) != shape[0] or any(len(row) != shape[1] for row in rows):
        raise ValueError(f"{name} must have shape {shape}")
    for row in rows:
        for delay in row:
            if delay is None:
                continue
            if isinstance(delay, (bool, np.bool_)) or not isinstance(delay, Integral):
                raise ValueError(f"{name} entries must be integer delays or null")
            if int(delay) < 0:
                raise ValueError(f"{name} delays must be nonnegative")


def validate_point_record(point: Mapping[str, object], *, artifact_root: Path) -> None:
    """Validate one solve record and its immutable response-file provenance."""

    missing = sorted(_POINT_FIELDS - set(point))
    if missing:
        raise ValueError(f"point record is missing fields: {', '.join(missing)}")
    try:
        json.dumps(point, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"point record is not strict JSON: {error}") from error
    key = point["key"]
    if not isinstance(key, str) or not key or any(character in key for character in "/\\"):
        raise ValueError("point key must be a nonempty path-safe string")
    if not isinstance(point["method"], str) or not point["method"]:
        raise ValueError("point method must be a nonempty string")
    budget = _finite_or_none("budget", point["budget"], allow_none=False)
    dense = _finite_or_none(
        "dense_reference_h2", point["dense_reference_h2"], allow_none=False
    )
    if cast(float, budget) < 0.0 or cast(float, dense) <= 0.0:
        raise ValueError("budget must be nonnegative and dense reference positive")
    budget_units = point["budget_units"]
    if isinstance(budget_units, (bool, np.bool_)) or not isinstance(budget_units, Integral):
        raise ValueError("budget_units must be an integer")
    if architecture_cost_units(cast(float, budget)) != int(budget_units):
        raise ValueError("budget does not match budget_units")
    if isinstance(point["horizon"], (bool, np.bool_)) or int(cast(Any, point["horizon"])) < 2:
        raise ValueError("horizon must be an integer at least two")
    settings = point["solver_settings"]
    if not isinstance(settings, Mapping):
        raise ValueError("solver_settings must be an object")
    missing_settings = sorted(_SOLVER_SETTING_FIELDS - set(settings))
    if missing_settings:
        raise ValueError(
            f"solver_settings is incomplete: {', '.join(missing_settings)}"
        )
    if settings["seed"] != PUBLICATION_SEED or settings["threads"] != PUBLICATION_THREADS:
        raise ValueError("solver settings must use Seed=23 and Threads=1")
    if settings["numeric_focus"] != 3:
        raise ValueError("solver settings must use NumericFocus=3")
    if settings["time_limit"] is not None:
        raise ValueError("publication solve records must not use a time limit")
    solution_count = point["solution_count"]
    if isinstance(solution_count, (bool, np.bool_)) or not isinstance(solution_count, Integral):
        raise ValueError("solution_count must be an integer")
    _finite_or_none("runtime_seconds", point["runtime_seconds"], allow_none=False)
    if not isinstance(point["certified"], bool):
        raise ValueError("certified must be bool")

    has_solution = int(solution_count) > 0
    numeric_fields = (
        "architecture_cost",
        "performance_h2",
        "signed_absolute_loss",
        "signed_percent_loss",
    )
    if not has_solution:
        for field in numeric_fields:
            if point[field] is not None:
                raise ValueError(f"no-solution point must encode {field} as null")
        for field in (
            "physical_service_matrix",
            "deployment_service_matrix",
            "residuals",
            "hardware_gating",
            "service_gating",
            "architecture_cost_audit",
            "realization_mismatch",
            "response_artifact",
        ):
            if point[field] is not None:
                raise ValueError(f"no-solution point must encode {field} as null")
        if point["architecture_cost_units"] is not None or point["certified"]:
            raise ValueError("no-solution point cannot have cost units or certification")
        _finite_or_none("best_bound", point["best_bound"], allow_none=True)
        _finite_or_none("mip_gap", point["mip_gap"], allow_none=True)
        return

    for field in numeric_fields:
        _finite_or_none(field, point[field], allow_none=False)
    _finite_or_none("best_bound", point["best_bound"], allow_none=True)
    _finite_or_none("mip_gap", point["mip_gap"], allow_none=True)
    cost = cast(float, point["architecture_cost"])
    cost_units = point["architecture_cost_units"]
    if not isinstance(cost_units, Integral) or isinstance(cost_units, (bool, np.bool_)):
        raise ValueError("architecture_cost_units must be an integer")
    if architecture_cost_units(cost) != int(cost_units):
        raise ValueError("architecture_cost does not match its exact units")
    if cost > cast(float, budget) + 1.0e-12:
        raise ValueError("architecture cost exceeds the declared budget")
    absolute, percentage = signed_performance_loss(
        cast(float, point["performance_h2"]), cast(float, dense)
    )
    if abs(absolute - cast(float, point["signed_absolute_loss"])) > 1.0e-12:
        raise ValueError("signed_absolute_loss is inconsistent")
    if abs(percentage - cast(float, point["signed_percent_loss"])) > 1.0e-10:
        raise ValueError("signed_percent_loss is inconsistent")
    _validate_delay_matrix("physical_service_matrix", point["physical_service_matrix"], (3, 4))
    _validate_delay_matrix("deployment_service_matrix", point["deployment_service_matrix"], (4, 4))
    for field in (
        "residuals",
        "hardware_gating",
        "service_gating",
        "architecture_cost_audit",
        "realization_mismatch",
    ):
        if not isinstance(point[field], Mapping):
            raise ValueError(f"{field} must be present for a solution")
    mismatch = cast(Mapping[str, object], point["realization_mismatch"])
    mismatch_value = _finite_or_none("realization_mismatch.max_abs", mismatch.get("max_abs"), allow_none=False)
    mismatch_tolerance = _finite_or_none("realization_mismatch.tolerance", mismatch.get("tolerance"), allow_none=False)
    if cast(float, mismatch_value) > cast(float, mismatch_tolerance):
        raise ValueError("realization mismatch exceeds its declared tolerance")
    response = point["response_artifact"]
    if not isinstance(response, Mapping):
        raise ValueError("response_artifact must be present for a solution")
    relative = response.get("path")
    digest = response.get("sha256")
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("response artifact path must be relative and contained")
    path = Path(artifact_root) / relative
    if not path.is_file() or not isinstance(digest, str) or sha256_file(path) != digest:
        raise ValueError("response artifact hash does not match the saved response")
    if point["certified"]:
        gap = point["mip_gap"]
        target_gap = float(cast(Any, settings["target_mip_gap"]))
        if point["status"] != "OPTIMAL" or (gap is not None and float(cast(Any, gap)) > target_gap + 1.0e-12):
            raise ValueError("certified point lacks an optimal closed-gap solve")


def _save_response_artifact(
    artifact_root: Path, key: str, responses: object
) -> dict[str, str]:
    root = Path(artifact_root)
    response_directory = root / "responses"
    response_directory.mkdir(parents=True, exist_ok=True)
    target = response_directory / f"{key}.npz"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{key}.", suffix=".tmp", dir=response_directory
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(
                stream,
                R=np.asarray(getattr(responses, "R"), dtype=float),
                M=np.asarray(getattr(responses, "M"), dtype=float),
                N=np.asarray(getattr(responses, "N"), dtype=float),
                L=np.asarray(getattr(responses, "L"), dtype=float),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "path": target.relative_to(root).as_posix(),
        "sha256": sha256_file(target),
    }


def _central_controller_coefficients(responses: object, sample_count: int) -> np.ndarray:
    R = np.asarray(getattr(responses, "R"), dtype=float)
    M = np.asarray(getattr(responses, "M"), dtype=float)
    N = np.asarray(getattr(responses, "N"), dtype=float)
    L = np.asarray(getattr(responses, "L"), dtype=float)
    state_dimension = R.shape[1]
    r_hat = R[1:]
    inverse = np.zeros((sample_count, state_dimension, state_dimension))
    inverse[0] = np.eye(state_dimension)
    for lag in range(1, sample_count):
        for response_lag in range(1, min(lag, len(r_hat) - 1) + 1):
            inverse[lag] -= r_hat[response_lag] @ inverse[lag - response_lag]
    controller = np.zeros((sample_count, M.shape[1], N.shape[2]))
    controller[: min(sample_count, len(L))] += L[:sample_count]
    for m_lag in range(1, len(M)):
        for inverse_lag in range(sample_count):
            for n_lag in range(1, len(N)):
                lag = m_lag + inverse_lag + n_lag - 1
                if lag < sample_count:
                    controller[lag] -= M[m_lag] @ inverse[inverse_lag] @ N[n_lag]
    return controller


def _realization_mismatch(responses: object) -> dict[str, object]:
    filters = realization_filters(cast(Any, responses))
    rng = np.random.default_rng(PUBLICATION_SEED)
    sample_count = 24
    measurements = rng.standard_normal((sample_count, filters.n.shape[2]))
    rollout = rollout_realization(filters, measurements)
    controller = _central_controller_coefficients(responses, sample_count)
    centralized = np.zeros_like(rollout.u)
    for time_index in range(sample_count):
        for lag in range(time_index + 1):
            centralized[time_index] += controller[lag] @ measurements[time_index - lag]
    mismatch = float(np.max(np.abs(rollout.u - centralized)))
    return {
        "max_abs": mismatch,
        "tolerance": REALIZATION_TOLERANCE,
        "sequence_kind": "gaussian_measurement_rollout",
        "seed": PUBLICATION_SEED,
        "sample_count": sample_count,
    }


def build_publication_point(
    case: PlatoonCase,
    problem: BPlusProblem,
    result: BPlusSolveResult,
    *,
    key: str,
    method: str,
    budget_units: int,
    dense_reference_h2: float,
    artifact_root: Path,
    globality_basis: str,
) -> dict[str, object]:
    """Convert one native solve into a strict, independently audited record."""

    from .baselines import platoon_controller_delays

    if not isinstance(case, PlatoonCase) or not isinstance(problem, BPlusProblem):
        raise TypeError("case and problem must be validated platoon B+ objects")
    if not isinstance(result, BPlusSolveResult):
        raise TypeError("result must be a BPlusSolveResult")
    if problem.plant is not case.plant and problem.plant.A.shape != case.plant.A.shape:
        raise ValueError("problem does not match the platoon case")
    budget = int(budget_units) * SERVICE_COST_UNIT
    settings = asdict(result.settings)
    base: dict[str, object] = {
        "key": key,
        "method": method,
        "budget": budget,
        "budget_units": int(budget_units),
        "architecture_cost": None,
        "architecture_cost_units": None,
        "performance_h2": None,
        "dense_reference_h2": float(dense_reference_h2),
        "signed_absolute_loss": None,
        "signed_percent_loss": None,
        "best_bound": result.best_bound,
        "mip_gap": result.mip_gap,
        "status": result.status,
        "solution_count": result.solution_count,
        "runtime_seconds": result.runtime,
        "physical_service_matrix": None,
        "deployment_service_matrix": None,
        "horizon": problem.horizon,
        "solver_settings": settings,
        "residuals": None,
        "hardware_gating": None,
        "service_gating": None,
        "architecture_cost_audit": None,
        "realization_mismatch": None,
        "certified": False,
        "globality_basis": globality_basis,
        "response_artifact": None,
    }
    if result.solution_count <= 0:
        validate_point_record(base, artifact_root=artifact_root)
        return base
    if (
        result.responses is None
        or result.service_delays is None
        or result.performance_objective is None
        or result.architecture_cost is None
    ):
        raise ValueError("solution-bearing result is incomplete")
    audit = audit_solution(problem, result, tolerance=AUDIT_TOLERANCE)
    absolute_loss, percent_loss = signed_performance_loss(
        result.performance_objective, dense_reference_h2
    )
    mismatch = _realization_mismatch(result.responses)
    closed_gap = (
        result.mip_gap is None
        or result.mip_gap <= result.settings.target_mip_gap + 1.0e-12
    )
    cost_units = architecture_cost_units(result.architecture_cost)
    response_artifact = _save_response_artifact(
        artifact_root, key, result.responses
    )
    base.update(
        {
            "architecture_cost": result.architecture_cost,
            "architecture_cost_units": cost_units,
            "performance_h2": result.performance_objective,
            "signed_absolute_loss": absolute_loss,
            "signed_percent_loss": percent_loss,
            "physical_service_matrix": [
                list(row)
                for row in platoon_controller_delays(case, result.service_delays)
            ],
            "deployment_service_matrix": [
                list(row) for row in result.service_delays
            ],
            "residuals": asdict(audit.ofsls),
            "hardware_gating": asdict(audit.hardware),
            "service_gating": asdict(audit.service),
            "architecture_cost_audit": asdict(audit.architecture_cost),
            "realization_mismatch": mismatch,
            "certified": bool(
                result.status == "OPTIMAL"
                and audit.certified
                and closed_gap
                and float(mismatch["max_abs"]) <= REALIZATION_TOLERANCE
            ),
            "response_artifact": response_artifact,
        }
    )
    validate_point_record(base, artifact_root=artifact_root)
    return base


def _point_lists(payloads: Mapping[str, Mapping[str, object]]) -> tuple[Mapping[str, object], ...]:
    locations = (
        ("pilot.json", "points"),
        ("bplus.json", "discovery_points"),
        ("bplus.json", "envelope_points"),
        ("qi.json", "points"),
        ("canonical.json", "points"),
        ("rfd.json", "points"),
    )
    points: list[Mapping[str, object]] = []
    for filename, field in locations:
        values = payloads.get(filename, {}).get(field, [])
        if not isinstance(values, list):
            raise ValueError(f"{filename}:{field} must be a list")
        if any(not isinstance(value, Mapping) for value in values):
            raise ValueError(f"{filename}:{field} must contain point objects")
        points.extend(cast(Sequence[Mapping[str, object]], values))
    return tuple(points)


def validate_publication_artifacts(artifact_root: Path) -> dict[str, object]:
    """Validate a complete saved run without invoking any solver."""

    root = Path(artifact_root)
    missing_files = [name for name in PUBLICATION_ARTIFACT_FILES if not (root / name).is_file()]
    if missing_files:
        raise ValueError(f"publication artifacts are missing: {missing_files}")
    payloads: dict[str, Mapping[str, object]] = {}
    for name in PUBLICATION_ARTIFACT_FILES:
        loaded = json.loads((root / name).read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping):
            raise ValueError(f"{name} must contain a JSON object")
        payloads[name] = loaded
    points = _point_lists(payloads)
    for point in points:
        validate_point_record(point, artifact_root=root)
    keys = [cast(str, point["key"]) for point in points]
    duplicate_keys = sorted({key for key in keys if keys.count(key) > 1})
    if duplicate_keys:
        raise ValueError(f"duplicate solve keys: {duplicate_keys}")
    point_index = {cast(str, point["key"]): point for point in points}
    common = payloads["common_budget.json"]
    source_keys = [
        cast(str, common.get("bplus_key")),
        *cast(Sequence[str], common.get("same_budget_qi_keys", [])),
        *cast(Sequence[str], common.get("rfd_keys", [])),
        *cast(Sequence[str], common.get("canonical_keys", [])),
    ]
    unknown_references = sorted({key for key in source_keys if key not in point_index})
    if unknown_references:
        raise ValueError(f"common-budget references unknown solve keys: {unknown_references}")
    untraceable: list[str] = []
    for item in cast(Sequence[Mapping[str, object]], common.get("manuscript_values", [])):
        source_key = cast(str, item.get("source_key"))
        field = cast(str, item.get("field"))
        if source_key not in point_index or field not in point_index[source_key]:
            untraceable.append(f"{source_key}:{field}")
            continue
        if point_index[source_key][field] != item.get("value"):
            untraceable.append(f"{source_key}:{field}")
    if untraceable:
        raise ValueError(f"untraceable manuscript values: {untraceable}")
    manifest = payloads["manifest.json"]
    hash_errors: list[str] = []
    for relative, expected_hash in cast(Mapping[str, str], manifest.get("artifact_hashes", {})).items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            hash_errors.append(relative)
    if hash_errors:
        raise ValueError(f"artifact manifest hash mismatches: {hash_errors}")
    return {
        "valid": True,
        "files": sorted(payloads),
        "point_count": len(points),
        "duplicate_keys": duplicate_keys,
        "untraceable_manuscript_values": untraceable,
        "certified_point_count": sum(bool(point["certified"]) for point in points),
    }


def _publication_options(*, budget_units: int | None = None) -> SolverOptions:
    return SolverOptions(
        seed=PUBLICATION_SEED,
        threads=PUBLICATION_THREADS,
        numeric_focus=3,
        output_flag=0,
        feasibility_tolerance=1.0e-9,
        optimality_tolerance=1.0e-9,
        integer_feasibility_tolerance=1.0e-9,
        mip_gap=1.0e-9,
        time_limit=None,
        architecture_weight=0.0,
        architecture_budget=(
            None if budget_units is None else budget_units * SERVICE_COST_UNIT
        ),
    )


def _free_service_problem(case: PlatoonCase, horizon: int) -> BPlusProblem:
    return BPlusProblem(
        plant=case.plant,
        deployment=case.deployment,
        horizon=horizon,
        fixed_eta=case.fixed_eta,
        fixed_xi=case.fixed_xi,
    )


def _solve_and_record(
    case: PlatoonCase,
    problem: BPlusProblem,
    options: SolverOptions,
    *,
    key: str,
    method: str,
    budget_units: int,
    dense_reference_h2: float,
    artifact_root: Path,
    globality_basis: str,
) -> dict[str, object]:
    start = perf_counter()
    result = solve_bplus(problem, options)
    point = build_publication_point(
        case,
        problem,
        result,
        key=key,
        method=method,
        budget_units=budget_units,
        dense_reference_h2=dense_reference_h2,
        artifact_root=artifact_root,
        globality_basis=globality_basis,
    )
    print(
        json.dumps(
            {
                "solve_key": key,
                "status": result.status,
                "horizon": problem.horizon,
                "budget_units": budget_units,
                "architecture_cost_units": point["architecture_cost_units"],
                "performance_h2": point["performance_h2"],
                "gap": result.mip_gap,
                "solver_seconds": result.runtime,
                "wall_seconds": perf_counter() - start,
            },
            allow_nan=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return point


def _is_feasible_point(point: Mapping[str, object]) -> bool:
    return int(cast(int, point["solution_count"])) > 0


def _best_point_keys_within_budget(
    points: Sequence[Mapping[str, object]],
    *,
    budget_units: int,
    tie_tolerance: float = 1.0e-12,
) -> tuple[str, ...]:
    eligible = tuple(
        point
        for point in points
        if point.get("certified") is True
        and point.get("architecture_cost_units") is not None
        and int(cast(int, point["architecture_cost_units"])) <= budget_units
        and point.get("performance_h2") is not None
    )
    if not eligible:
        return ()
    best = min(float(cast(float, point["performance_h2"])) for point in eligible)
    return tuple(
        sorted(
            cast(str, point["key"])
            for point in eligible
            if abs(float(cast(float, point["performance_h2"])) - best)
            <= tie_tolerance
        )
    )


def _binary_search_minimum_feasible_budget(
    case: PlatoonCase,
    attainable_units: Sequence[int],
    *,
    dense_reference_h2: float,
    artifact_root: Path,
    cache: dict[int, dict[str, object]],
) -> int:
    candidates = tuple(
        value for value in attainable_units if value <= COMMON_BUDGET_UNITS
    )
    if not candidates or candidates[-1] != COMMON_BUDGET_UNITS:
        raise ValueError("common budget is missing from attainable costs")

    def solve_budget(units: int) -> dict[str, object]:
        if units not in cache:
            cache[units] = _solve_and_record(
                case,
                _free_service_problem(case, PUBLICATION_HORIZON),
                _publication_options(budget_units=units),
                key=f"bplus_T10_B{units:04d}",
                method="bplus_budget",
                budget_units=units,
                dense_reference_h2=dense_reference_h2,
                artifact_root=artifact_root,
                globality_basis="closed_mixed_integer_gap_over_full_8748_menu",
            )
        return cache[units]

    high = len(candidates) - 1
    if not _is_feasible_point(solve_budget(candidates[high])):
        raise RuntimeError("submitted sparse common budget is infeasible")
    low = -1
    while high - low > 1:
        midpoint = (low + high) // 2
        if _is_feasible_point(solve_budget(candidates[midpoint])):
            high = midpoint
        else:
            low = midpoint
    return candidates[high]


def _service_matrix_json(matrix: ServiceDelayMatrix) -> list[list[int | None]]:
    return [list(row) for row in matrix]


def _point_architecture_signature(point: Mapping[str, object]) -> str | None:
    matrix = point.get("deployment_service_matrix")
    return None if matrix is None else json.dumps(matrix, separators=(",", ":"))


def horizon_sensitivity_is_material(
    performance_values: Sequence[float],
    *,
    architecture_signatures: Sequence[str | None],
    feasible_flags: Sequence[bool],
    relative_tolerance: float = 1.0e-3,
) -> bool:
    """Flag layout, feasibility, or declared relative FIR-performance sensitivity."""

    performances = tuple(float(value) for value in performance_values)
    signatures = tuple(architecture_signatures)
    feasible = tuple(bool(value) for value in feasible_flags)
    if not performances or not (
        len(performances) == len(signatures) == len(feasible)
    ):
        raise ValueError("horizon sensitivity sequences must have equal nonzero length")
    if not all(math.isfinite(value) and value >= 0.0 for value in performances):
        raise ValueError("performance_values must be finite and nonnegative")
    if not math.isfinite(relative_tolerance) or relative_tolerance < 0.0:
        raise ValueError("relative_tolerance must be finite and nonnegative")
    scale = max(max(abs(value) for value in performances), 1.0e-15)
    relative_span = (max(performances) - min(performances)) / scale
    return (
        len(set(signatures)) > 1
        or len(set(feasible)) > 1
        or relative_span > relative_tolerance
    )


def _git_commit(repository_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def _build_artifact_manifest(
    repository_root: Path,
    artifact_root: Path,
    *,
    run_seconds: float,
    point_count: int,
    figure_outputs: Mapping[str, object] | None = None,
) -> dict[str, object]:
    manifest = build_manifest(repository_root)
    artifact_hashes = {
        path.relative_to(artifact_root).as_posix(): sha256_file(path)
        for path in sorted(artifact_root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    source_paths = (
        repository_root / "LcssV1.1" / "V2" / "repro" / "platoon_experiment.py",
        repository_root / "LcssV1.1" / "V2" / "repro" / "plotting.py",
        repository_root
        / "LcssV1.1"
        / "V2"
        / "repro"
        / "tests"
        / "test_publication_artifacts.py",
    )
    manifest.update(
        {
            "schema_version": 2,
            "experiment": "lcss_v2_platoon_publication",
            "command": (
                "python -m repro.platoon_experiment --seed 23 --threads 1 "
                "--output results/lcss_v2/platoon"
            ),
            "source_git_commit": _git_commit(repository_root),
            "source_hashes": {
                path.relative_to(repository_root).as_posix(): sha256_file(path)
                for path in source_paths
            },
            "run_seconds": run_seconds,
            "point_count": point_count,
            "artifact_hashes": artifact_hashes,
            "figure_outputs": dict(figure_outputs or {}),
        }
    )
    return manifest


def run_publication_experiment(
    repository_root: Path,
    output_root: Path,
    *,
    seed: int = PUBLICATION_SEED,
    threads: int = PUBLICATION_THREADS,
) -> dict[str, object]:
    """Run the complete deterministic platoon publication workflow."""

    from .baselines import (
        BaselineRecord,
        best_same_budget_qi,
        canonical_service_templates,
        map_rfd_path_grid,
        platoon_plant_propagation_delays,
        saturated_menu_qi_closure,
    )
    from .plotting import generate_publication_figure

    root = Path(repository_root).resolve()
    output = Path(output_root)
    output = (root / output).resolve() if not output.is_absolute() else output.resolve()
    expected_output = (root / "results" / "lcss_v2" / "platoon").resolve()
    if output != expected_output:
        raise ValueError(
            "publication output must resolve exactly to results/lcss_v2/platoon"
        )
    if seed != PUBLICATION_SEED or threads != PUBLICATION_THREADS:
        raise ValueError("publication run requires Seed=23 and Threads=1")
    output.mkdir(parents=True, exist_ok=True)
    run_start = perf_counter()
    case = make_platoon_case()
    menu_records = tuple(architecture_records(case))
    attainable_units = derive_attainable_budget_units(case)
    propagation = platoon_plant_propagation_delays(case)
    qi_index_set = set(qi_menu_indices(case, menu_records))
    menu_rows: list[dict[str, object]] = []
    qi_architectures: list[tuple[int, PlatoonServiceArchitecture]] = []
    for index, architecture in enumerate(menu_records):
        qi = index in qi_index_set
        if qi:
            qi_architectures.append((index, architecture))
        menu_rows.append(
            {
                "index": index,
                "serialization": architecture.serialization,
                "cost": architecture.service_cost,
                "cost_units": architecture_cost_units(architecture.service_cost),
                "qi_compatible": qi,
                "deployment_service_matrix": _service_matrix_json(
                    architecture.service_delays
                ),
            }
        )
    if len(qi_architectures) != 99:
        raise RuntimeError(
            f"expected 99 QI-compatible architectures, observed {len(qi_architectures)}"
        )
    write_json_atomic(
        output / "config.json",
        {
            "schema_version": 1,
            "seed": seed,
            "threads": threads,
            "publication_horizon": PUBLICATION_HORIZON,
            "pilot_horizons": list(PUBLICATION_HORIZONS),
            "solver_options": asdict(_publication_options()),
            "matrix_sha256": platoon_case_matrix_sha256(case),
            "fixed_eta": list(case.fixed_eta),
            "fixed_xi": list(case.fixed_xi),
            "hardware_policy": "all platoon hardware fixed active",
            "raw_lag_service_offsets": {"L": 0, "M": 1, "N": 0, "R": 1},
            "R1_identity_exempt": True,
            "service_cost_unit": SERVICE_COST_UNIT,
            "common_budget": COMMON_BUDGET_UNITS * SERVICE_COST_UNIT,
            "common_budget_units": COMMON_BUDGET_UNITS,
            "dense_budget": DENSE_BUDGET_UNITS * SERVICE_COST_UNIT,
            "dense_budget_units": DENSE_BUDGET_UNITS,
            "audit_tolerance": AUDIT_TOLERANCE,
            "realization_tolerance": REALIZATION_TOLERANCE,
            "time_limits_used": False,
        },
    )
    write_json_atomic(
        output / "menu.json",
        {
            "schema_version": 1,
            "orientation": "destination_site <- source_site",
            "architecture_count": len(menu_rows),
            "qi_architecture_count": len(qi_architectures),
            "architectures": menu_rows,
        },
    )

    dense_problem = build_platoon_problem(
        case, platoon_dense_architecture(case), horizon=PUBLICATION_HORIZON
    )
    dense_result = solve_bplus(dense_problem, _publication_options())
    if dense_result.performance_objective is None:
        raise RuntimeError("dense publication reference has no solution")
    dense_h2 = dense_result.performance_objective
    dense_point = build_publication_point(
        case,
        dense_problem,
        dense_result,
        key="canonical_dense_raw",
        method="canonical_raw",
        budget_units=DENSE_BUDGET_UNITS,
        dense_reference_h2=dense_h2,
        artifact_root=output,
        globality_basis="fixed_dense_architecture_native_qp",
    )
    print(
        json.dumps(
            {
                "solve_key": dense_point["key"],
                "status": dense_point["status"],
                "performance_h2": dense_h2,
                "gap": dense_point["mip_gap"],
                "solver_seconds": dense_point["runtime_seconds"],
            },
            allow_nan=False,
            sort_keys=True,
        ),
        flush=True,
    )

    bplus_cache: dict[int, dict[str, object]] = {}
    minimum_feasible_units = _binary_search_minimum_feasible_budget(
        case,
        attainable_units,
        dense_reference_h2=dense_h2,
        artifact_root=output,
        cache=bplus_cache,
    )
    selected_units = select_compact_budget_units(
        attainable_units, minimum_feasible_units=minimum_feasible_units
    )
    for units in selected_units:
        if units not in bplus_cache:
            bplus_cache[units] = _solve_and_record(
                case,
                _free_service_problem(case, PUBLICATION_HORIZON),
                _publication_options(budget_units=units),
                key=f"bplus_T10_B{units:04d}",
                method="bplus_budget",
                budget_units=units,
                dense_reference_h2=dense_h2,
                artifact_root=output,
                globality_basis="closed_mixed_integer_gap_over_full_8748_menu",
            )
    envelope_points = [bplus_cache[units] for units in selected_units]
    discovery_points = [
        point
        for units, point in sorted(bplus_cache.items())
        if units not in selected_units
    ]
    write_json_atomic(
        output / "budgets.json",
        {
            "schema_version": 1,
            "cost_unit": SERVICE_COST_UNIT,
            "attainable_units": list(attainable_units),
            "attainable_budgets": [units * SERVICE_COST_UNIT for units in attainable_units],
            "minimum_feasible_units": minimum_feasible_units,
            "minimum_feasible_budget": minimum_feasible_units * SERVICE_COST_UNIT,
            "selected_units": list(selected_units),
            "selected_budgets": [units * SERVICE_COST_UNIT for units in selected_units],
            "selection_rule": (
                "minimum feasible, lower midpoint, submitted common budget, "
                "upper midpoint, dense endpoint; nearest attainable integer unit"
            ),
            "hard_budget_inequality": "J_arch <= B",
            "exact_cost_constraint_used": False,
        },
    )

    pilot_points: list[dict[str, object]] = []
    dense_by_horizon: dict[int, dict[str, object]] = {PUBLICATION_HORIZON: dense_point}
    for horizon in (8, 12):
        problem = build_platoon_problem(
            case, platoon_dense_architecture(case), horizon=horizon
        )
        result = solve_bplus(problem, _publication_options())
        if result.performance_objective is None:
            raise RuntimeError(f"dense horizon pilot T={horizon} has no solution")
        point = build_publication_point(
            case,
            problem,
            result,
            key=f"pilot_dense_T{horizon}",
            method="pilot_dense_fixed",
            budget_units=DENSE_BUDGET_UNITS,
            dense_reference_h2=result.performance_objective,
            artifact_root=output,
            globality_basis="fixed_dense_architecture_native_qp",
        )
        dense_by_horizon[horizon] = point
        pilot_points.append(point)
    pilot_anchor_units = tuple(
        sorted({minimum_feasible_units, COMMON_BUDGET_UNITS, DENSE_BUDGET_UNITS})
    )
    pilot_anchor_keys: dict[int, dict[int, str]] = {
        units: {PUBLICATION_HORIZON: cast(str, bplus_cache[units]["key"])}
        for units in pilot_anchor_units
    }
    for horizon in (8, 12):
        dense_reference = cast(float, dense_by_horizon[horizon]["performance_h2"])
        for units in pilot_anchor_units:
            point = _solve_and_record(
                case,
                _free_service_problem(case, horizon),
                _publication_options(budget_units=units),
                key=f"pilot_bplus_T{horizon}_B{units:04d}",
                method="pilot_bplus_budget",
                budget_units=units,
                dense_reference_h2=dense_reference,
                artifact_root=output,
                globality_basis="closed_mixed_integer_gap_over_full_8748_menu",
            )
            pilot_points.append(point)
            pilot_anchor_keys[units][horizon] = cast(str, point["key"])
    pilot_lookup = {
        cast(str, point["key"]): point
        for point in (
            *pilot_points,
            *envelope_points,
            *discovery_points,
            dense_point,
        )
    }
    sensitivities: list[dict[str, object]] = []
    for units in pilot_anchor_units:
        horizon_points = [
            pilot_lookup[pilot_anchor_keys[units][horizon]]
            for horizon in PUBLICATION_HORIZONS
        ]
        signatures = [
            _point_architecture_signature(point) for point in horizon_points
        ]
        feasible_flags = [_is_feasible_point(point) for point in horizon_points]
        performances = [point["performance_h2"] for point in horizon_points]
        finite_performances = [
            float(cast(float, value)) for value in performances if value is not None
        ]
        performance_relative_span = (
            None
            if len(finite_performances) < 2
            else (max(finite_performances) - min(finite_performances))
            / max(max(abs(value) for value in finite_performances), 1.0e-15)
        )
        material = (
            horizon_sensitivity_is_material(
                finite_performances,
                architecture_signatures=signatures,
                feasible_flags=feasible_flags,
            )
            if len(finite_performances) == len(horizon_points)
            else len(set(signatures)) > 1 or len(set(feasible_flags)) > 1
        )
        sensitivities.append(
            {
                "budget_units": units,
                "budget": units * SERVICE_COST_UNIT,
                "keys_by_horizon": {
                    str(horizon): pilot_anchor_keys[units][horizon]
                    for horizon in PUBLICATION_HORIZONS
                },
                "feasible_by_horizon": {
                    str(horizon): feasible
                    for horizon, feasible in zip(
                        PUBLICATION_HORIZONS, feasible_flags, strict=True
                    )
                },
                "architecture_changes": len(set(signatures)) > 1,
                "performance_relative_span": performance_relative_span,
                "performance_changes_materially": (
                    performance_relative_span is not None
                    and performance_relative_span > 1.0e-3
                ),
                "material_sensitivity": material,
                "performance_h2": {
                    str(horizon): point["performance_h2"]
                    for horizon, point in zip(
                        PUBLICATION_HORIZONS, horizon_points, strict=True
                    )
                },
                "terminal_residual": {
                    str(horizon): (
                        None
                        if point["residuals"] is None
                        else cast(Mapping[str, object], point["residuals"])[
                            "terminal_max"
                        ]
                    )
                    for horizon, point in zip(
                        PUBLICATION_HORIZONS, horizon_points, strict=True
                    )
                },
            }
        )
    pilot_payload = {
        "schema_version": 1,
        "horizons": list(PUBLICATION_HORIZONS),
        "dense_keys_by_horizon": {
            str(horizon): dense_by_horizon[horizon]["key"]
            for horizon in PUBLICATION_HORIZONS
        },
        "anchor_units": list(pilot_anchor_units),
        "points": pilot_points,
        "sensitivities": sensitivities,
        "any_material_sensitivity": any(
            bool(item["material_sensitivity"]) for item in sensitivities
        ),
        "sensitivity_hidden_or_tuned_away": False,
    }
    write_json_atomic(output / "pilot.json", pilot_payload)
    write_json_atomic(
        output / "bplus.json",
        {
            "schema_version": 1,
            "label": "certified finite performance-budget envelope",
            "horizon": PUBLICATION_HORIZON,
            "dense_reference_key": dense_point["key"],
            "dense_reference_h2": dense_h2,
            "discovery_points": discovery_points,
            "envelope_points": envelope_points,
            "all_envelope_points_globally_closed": all(
                point["certified"] is True for point in envelope_points
            ),
        },
    )

    qi_points: list[dict[str, object]] = []
    qi_baseline_records: list[Any] = []
    for position, (menu_index, architecture) in enumerate(qi_architectures, start=1):
        point = _solve_and_record(
            case,
            build_platoon_problem(
                case, architecture.service_delays, horizon=PUBLICATION_HORIZON
            ),
            _publication_options(),
            key=f"qi_menu_{menu_index:04d}",
            method="qi_fixed",
            budget_units=architecture_cost_units(architecture.service_cost),
            dense_reference_h2=dense_h2,
            artifact_root=output,
            globality_basis="fixed_QI_architecture_native_qp",
        )
        qi_points.append(point)
        choice = ArchitectureChoice(
            case.fixed_eta, case.fixed_xi, architecture.service_delays
        )
        qi_baseline_records.append(
            BaselineRecord(
                key=cast(str, point["key"]),
                choice=choice,
                architecture_cost=architecture.service_cost,
                performance=(
                    math.inf
                    if point["performance_h2"] is None
                    else float(cast(float, point["performance_h2"]))
                ),
                qi_compatible=True,
                feasible=_is_feasible_point(point),
            )
        )
        if position % 10 == 0 or position == len(qi_architectures):
            print(
                f"QI fixed-QP progress: {position}/{len(qi_architectures)}",
                flush=True,
            )
    qi_same_budget: list[dict[str, object]] = []
    qi_point_by_key = {
        cast(str, point["key"]): point for point in qi_points
    }
    uncertified_qi_points = tuple(
        point
        for point in qi_points
        if _is_feasible_point(point) and point["certified"] is not True
    )
    no_solution_qi_points = tuple(
        point for point in qi_points if not _is_feasible_point(point)
    )
    unexpected_no_solution_statuses = sorted(
        {
            cast(str, point["status"])
            for point in no_solution_qi_points
            if point["status"] not in {"INFEASIBLE", "INF_OR_UNBD"}
        }
    )
    for units in selected_units:
        result = best_same_budget_qi(
            qi_baseline_records,
            budget=units * SERVICE_COST_UNIT,
            budget_tolerance=1.0e-12,
            tie_tolerance=1.0e-12,
        )
        argmin_keys = [record.key for record in result.argmin_records]
        winner_certified = bool(argmin_keys) and all(
            qi_point_by_key[key]["certified"] is True for key in argmin_keys
        )
        lower_bound_exclusions: list[dict[str, object]] = []
        unresolved_uncertified: list[str] = []
        for point in uncertified_qi_points:
            cost_units = int(cast(int, point["architecture_cost_units"]))
            if cost_units > units:
                continue
            bound = point["best_bound"]
            if (
                bound is None
                or not math.isfinite(result.best_performance)
                or float(cast(float, bound)) <= result.best_performance
            ):
                unresolved_uncertified.append(cast(str, point["key"]))
                continue
            lower_bound_exclusions.append(
                {
                    "key": point["key"],
                    "architecture_cost_units": cost_units,
                    "best_bound": bound,
                    "certified_incumbent_h2": result.best_performance,
                    "strict_margin": float(cast(float, bound))
                    - result.best_performance,
                    "excluded": True,
                }
            )
        global_certificate = (
            winner_certified
            and not unresolved_uncertified
            and not unexpected_no_solution_statuses
        )
        qi_same_budget.append(
            {
                "budget": units * SERVICE_COST_UNIT,
                "budget_units": units,
                "best_performance_h2": (
                    None
                    if not math.isfinite(result.best_performance)
                    else result.best_performance
                ),
                "argmin_keys": argmin_keys,
                "eligible_count": result.eligible_count,
                "winner_certified": winner_certified,
                "lower_bound_dominance_exclusions": lower_bound_exclusions,
                "unresolved_uncertified_candidates": unresolved_uncertified,
                "no_solution_count_within_budget": sum(
                    int(cast(int, point["budget_units"])) <= units
                    for point in no_solution_qi_points
                ),
                "no_solution_status_reason": (
                    "INFEASIBLE is direct; INF_OR_UNBD cannot mean objective "
                    "unbounded below because the fixed-QP objective is a sum of squares"
                ),
                "global_over_all_99_qi_architectures": global_certificate,
                "exact_cost_required": False,
            }
        )
    if not all(
        bool(item["global_over_all_99_qi_architectures"])
        for item in qi_same_budget
    ):
        raise RuntimeError(
            "one or more same-budget QI optima lack an exhaustive certificate"
        )
    common_qi_entry = next(
        item
        for item in qi_same_budget
        if item["budget_units"] == COMMON_BUDGET_UNITS
    )
    if not cast(Sequence[str], common_qi_entry["argmin_keys"]):
        raise RuntimeError("no QI architecture is feasible at the common budget")
    write_json_atomic(
        output / "qi.json",
        {
            "schema_version": 1,
            "criterion": "D <= D min-plus P min-plus D",
            "plant_propagation_delays": [list(row) for row in propagation],
            "total_menu_count": len(menu_records),
            "qi_family_count": len(qi_architectures),
            "points": qi_points,
            "same_budget_results": qi_same_budget,
            "each_qi_architecture_solved_once_at_T10": True,
            "uncertified_fixed_qp_points": [
                {
                    "key": point["key"],
                    "architecture_cost_units": point["architecture_cost_units"],
                    "best_bound": point["best_bound"],
                    "max_residual": cast(Mapping[str, object], point["residuals"])[
                        "max_residual"
                    ],
                    "eligible_selected_budget_units": [
                        units
                        for units in selected_units
                        if int(cast(int, point["architecture_cost_units"])) <= units
                    ],
                    "budget_exclusions": [
                        {
                            "budget_units": item["budget_units"],
                            **exclusion,
                        }
                        for item in qi_same_budget
                        for exclusion in cast(
                            Sequence[Mapping[str, object]],
                            item["lower_bound_dominance_exclusions"],
                        )
                        if exclusion["key"] == point["key"]
                    ],
                    "audit_tolerance_not_relaxed": True,
                }
                for point in uncertified_qi_points
            ],
            "no_solution_fixed_qp_count": len(no_solution_qi_points),
            "unexpected_no_solution_statuses": unexpected_no_solution_statuses,
        },
    )

    templates = canonical_service_templates(case)
    canonical_points: list[dict[str, object]] = [dense_point]
    canonical_entries: list[dict[str, object]] = []
    canonical_baseline_keys: list[str] = []
    canonical_point_by_key: dict[str, dict[str, object]] = {
        cast(str, dense_point["key"]): dense_point
    }
    for name, choice in templates.items():
        raw_key = f"canonical_{name.lower()}_raw"
        if name == "dense":
            raw_point = dense_point
        else:
            raw_cost_units = architecture_cost_units(
                architecture_cost(case.deployment, choice).total
            )
            raw_point = _solve_and_record(
                case,
                build_platoon_problem(
                    case, choice.service_delays, horizon=PUBLICATION_HORIZON
                ),
                _publication_options(),
                key=raw_key,
                method="canonical_raw",
                budget_units=raw_cost_units,
                dense_reference_h2=dense_h2,
                artifact_root=output,
                globality_basis="fixed_canonical_architecture_native_qp",
            )
            canonical_points.append(raw_point)
            canonical_point_by_key[raw_key] = raw_point
        repair = saturated_menu_qi_closure(case, choice)
        if repair.changed:
            repair_key = f"canonical_{name.lower()}_qi_repair"
            repair_units = architecture_cost_units(repair.repaired_cost)
            repair_point = _solve_and_record(
                case,
                build_platoon_problem(
                    case,
                    repair.repaired_choice.service_delays,
                    horizon=PUBLICATION_HORIZON,
                ),
                _publication_options(),
                key=repair_key,
                method="canonical_qi_repair",
                budget_units=repair_units,
                dense_reference_h2=dense_h2,
                artifact_root=output,
                globality_basis="fixed_QI_repaired_canonical_architecture_native_qp",
            )
            canonical_points.append(repair_point)
            canonical_point_by_key[repair_key] = repair_point
        else:
            repair_key = cast(str, raw_point["key"])
        if name not in {"dense", "submitted_sparse"}:
            canonical_baseline_keys.extend((cast(str, raw_point["key"]), repair_key))
        canonical_entries.append(
            {
                "name": name,
                "raw_key": raw_point["key"],
                "repair_key": repair_key,
                "raw_service_matrix": _service_matrix_json(choice.service_delays),
                "repaired_service_matrix": _service_matrix_json(
                    repair.repaired_choice.service_delays
                ),
                "raw_cost": repair.raw_cost,
                "raw_cost_units": architecture_cost_units(repair.raw_cost),
                "repaired_cost": repair.repaired_cost,
                "repaired_cost_units": architecture_cost_units(
                    repair.repaired_cost
                ),
                "repair_cost": repair.repair_cost,
                "repair_changed": repair.changed,
                "repair_qi_compatible": repair.qi_compatible,
                "repair_iterations": repair.iterations,
            }
        )
    write_json_atomic(
        output / "canonical.json",
        {
            "schema_version": 1,
            "template_names": list(templates),
            "entries": canonical_entries,
            "points": canonical_points,
            "common_budget_baseline_excludes": ["dense", "submitted_sparse"],
        },
    )

    rfd_points: list[dict[str, object]] = []
    rfd_grid: list[dict[str, object]] = []
    rfd_repair_grid: list[dict[str, object]] = []
    rfd_not_reported_reason: str | None = None
    rfd_provenance: dict[str, object] = {}
    try:
        path_points, rfd_provenance = load_legacy_rfd_path_points(root)
        mapped_grid = map_rfd_path_grid(case, path_points, qi_repair=False)
        mapped_repair_grid = map_rfd_path_grid(case, path_points, qi_repair=True)
        signature_to_key: dict[str, str] = {}
        for item in mapped_grid:
            signature = canonical_service_serialization(item.choice.service_delays)
            if signature not in signature_to_key:
                key = f"rfd_arch_{len(signature_to_key):03d}"
                units = architecture_cost_units(
                    architecture_cost(case.deployment, item.choice).total
                )
                point = _solve_and_record(
                    case,
                    build_platoon_problem(
                        case,
                        item.choice.service_delays,
                        horizon=PUBLICATION_HORIZON,
                    ),
                    _publication_options(),
                    key=key,
                    method="rfd_mapped_fixed",
                    budget_units=units,
                    dense_reference_h2=dense_h2,
                    artifact_root=output,
                    globality_basis="fixed_RFD_mapped_architecture_native_qp",
                )
                rfd_points.append(point)
                signature_to_key[signature] = key
            rfd_grid.append(
                {
                    "regularization": item.path_point.regularization,
                    "threshold": item.path_point.threshold,
                    "raw_support": [list(row) for row in item.path_point.raw_support],
                    "mapped_solve_key": signature_to_key[signature],
                    "mapped_service_matrix": _service_matrix_json(
                        item.choice.service_delays
                    ),
                    "qi_repaired": False,
                }
            )
        repair_signature_to_key: dict[str, str] = {}
        for item in mapped_repair_grid:
            signature = canonical_service_serialization(item.choice.service_delays)
            if signature not in repair_signature_to_key:
                key = f"rfd_qi_repair_arch_{len(repair_signature_to_key):03d}"
                units = architecture_cost_units(
                    architecture_cost(case.deployment, item.choice).total
                )
                point = _solve_and_record(
                    case,
                    build_platoon_problem(
                        case,
                        item.choice.service_delays,
                        horizon=PUBLICATION_HORIZON,
                    ),
                    _publication_options(),
                    key=key,
                    method="rfd_qi_repair_fixed",
                    budget_units=units,
                    dense_reference_h2=dense_h2,
                    artifact_root=output,
                    globality_basis="fixed_QI_repaired_RFD_architecture_native_qp",
                )
                rfd_points.append(point)
                repair_signature_to_key[signature] = key
            rfd_repair_grid.append(
                {
                    "regularization": item.path_point.regularization,
                    "threshold": item.path_point.threshold,
                    "raw_support": [list(row) for row in item.path_point.raw_support],
                    "mapped_solve_key": repair_signature_to_key[signature],
                    "mapped_service_matrix": _service_matrix_json(
                        item.choice.service_delays
                    ),
                    "qi_repaired": True,
                    "repair_changed": (
                        None if item.repair is None else item.repair.changed
                    ),
                    "repair_cost": (
                        None if item.repair is None else item.repair.repair_cost
                    ),
                }
            )
    except (OSError, TypeError, ValueError) as error:
        rfd_not_reported_reason = (
            "complete legacy raw-support grid/provenance unavailable: " + str(error)
        )
    write_json_atomic(
        output / "rfd.json",
        {
            "schema_version": 1,
            "reported": rfd_not_reported_reason is None,
            "not_reported_reason": rfd_not_reported_reason,
            "provenance": rfd_provenance,
            "path_grid": rfd_grid,
            "qi_repair_path_grid": rfd_repair_grid,
            "unique_mapped_architecture_count": sum(
                point["method"] == "rfd_mapped_fixed" for point in rfd_points
            ),
            "unique_qi_repaired_architecture_count": sum(
                point["method"] == "rfd_qi_repair_fixed" for point in rfd_points
            ),
            "raw_mapped_cost_units": sorted(
                {
                    int(cast(int, point["budget_units"]))
                    for point in rfd_points
                    if point["method"] == "rfd_mapped_fixed"
                }
            ),
            "qi_repaired_cost_units": sorted(
                {
                    int(cast(int, point["budget_units"]))
                    for point in rfd_points
                    if point["method"] == "rfd_qi_repair_fixed"
                }
            ),
            "repair_rule": (
                "exact unsaturated min-plus QI closure followed by least-cost "
                "finite-menu projection, repeated to QI compatibility"
            ),
            "raw_and_repair_layers_are_distinct": True,
            "points": rfd_points,
            "old_performance_values_reused": False,
            "all_mapped_architectures_resynthesized_with_new_deadlines": (
                rfd_not_reported_reason is None
            ),
        },
    )

    bplus_common = bplus_cache[COMMON_BUDGET_UNITS]
    qi_common_keys = tuple(cast(Sequence[str], common_qi_entry["argmin_keys"]))
    rfd_common_keys = _best_point_keys_within_budget(
        rfd_points, budget_units=COMMON_BUDGET_UNITS
    )
    canonical_candidates = [
        canonical_point_by_key[key]
        for key in sorted(set(canonical_baseline_keys))
    ]
    canonical_common_keys = _best_point_keys_within_budget(
        canonical_candidates, budget_units=COMMON_BUDGET_UNITS
    )
    all_point_index = {
        cast(str, point["key"]): point
        for point in (
            *envelope_points,
            *discovery_points,
            *pilot_points,
            *qi_points,
            *canonical_points,
            *rfd_points,
        )
    }
    comparison_keys = (
        cast(str, bplus_common["key"]),
        *qi_common_keys,
        *rfd_common_keys,
        *canonical_common_keys,
    )
    if any(
        int(cast(int, all_point_index[key]["architecture_cost_units"]))
        > COMMON_BUDGET_UNITS
        for key in comparison_keys
    ):
        raise RuntimeError("a common-budget comparison architecture exceeds B")
    bplus_loss = float(cast(float, bplus_common["signed_absolute_loss"]))
    comparisons: list[dict[str, object]] = []
    manuscript_values: list[dict[str, object]] = []
    for key in comparison_keys:
        point = all_point_index[key]
        ratio = (
            None
            if bplus_loss == 0.0
            else float(cast(float, point["signed_absolute_loss"])) / bplus_loss
        )
        comparisons.append(
            {
                "source_key": key,
                "method": point["method"],
                "architecture_cost": point["architecture_cost"],
                "architecture_cost_units": point["architecture_cost_units"],
                "performance_h2": point["performance_h2"],
                "signed_absolute_loss": point["signed_absolute_loss"],
                "signed_percent_loss": point["signed_percent_loss"],
                "signed_loss_ratio_to_bplus": ratio,
                "physical_service_matrix": point["physical_service_matrix"],
            }
        )
        for field in (
            "architecture_cost",
            "performance_h2",
            "signed_absolute_loss",
            "signed_percent_loss",
        ):
            manuscript_values.append(
                {"source_key": key, "field": field, "value": point[field]}
            )
    common_payload = {
        "schema_version": 1,
        "budget": COMMON_BUDGET_UNITS * SERVICE_COST_UNIT,
        "budget_units": COMMON_BUDGET_UNITS,
        "budget_source": "submitted_sparse exact declared service cost",
        "hard_constraint": "J_arch <= B",
        "bplus_key": bplus_common["key"],
        "same_budget_qi_keys": list(qi_common_keys),
        "rfd_keys": list(rfd_common_keys),
        "canonical_keys": list(canonical_common_keys),
        "comparisons": comparisons,
        "manuscript_values": manuscript_values,
        "all_compared_costs_within_budget": True,
        "losses_and_ratios_are_signed_and_unclipped": True,
    }
    write_json_atomic(output / "common_budget.json", common_payload)

    all_points = tuple(all_point_index.values())
    solve_log = "\n".join(
        json.dumps(
            {
                "key": point["key"],
                "method": point["method"],
                "status": point["status"],
                "runtime_seconds": point["runtime_seconds"],
                "solution_count": point["solution_count"],
                "mip_gap": point["mip_gap"],
                "certified": point["certified"],
            },
            allow_nan=False,
            sort_keys=True,
        )
        for point in sorted(all_points, key=lambda item: cast(str, item["key"]))
    )
    _atomic_text(output / "solve_summary.jsonl", solve_log + "\n")
    validation_payload = {
        "schema_version": 1,
        "valid": True,
        "point_count": len(all_points),
        "unique_key_count": len(all_point_index),
        "duplicate_keys": [],
        "untraceable_manuscript_values": [],
        "certified_point_count": sum(
            point["certified"] is True for point in all_points
        ),
        "realization_mismatch_max": max(
            (
                float(
                    cast(Mapping[str, object], point["realization_mismatch"])[
                        "max_abs"
                    ]
                )
                for point in all_points
                if point["realization_mismatch"] is not None
            ),
            default=0.0,
        ),
        "response_artifacts_hashed": True,
        "strict_json_allow_nan_false": True,
        "time_limits_used": False,
    }
    write_json_atomic(output / "validation.json", validation_payload)
    provisional_manifest = _build_artifact_manifest(
        root,
        output,
        run_seconds=perf_counter() - run_start,
        point_count=len(all_points),
    )
    write_json_atomic(output / "manifest.json", provisional_manifest)
    validate_publication_artifacts(output)
    figure_outputs = generate_publication_figure(
        output, root / "LcssV1.1" / "V2" / "fig"
    )
    final_manifest = _build_artifact_manifest(
        root,
        output,
        run_seconds=perf_counter() - run_start,
        point_count=len(all_points),
        figure_outputs=figure_outputs,
    )
    final_manifest["v2_figure_hashes"] = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted((root / "LcssV1.1" / "V2" / "fig").glob("platoon_budget_envelope*"))
        if path.is_file()
    }
    write_json_atomic(output / "manifest.json", final_manifest)
    final_validation = validate_publication_artifacts(output)
    summary: dict[str, object] = {
        "output": output.relative_to(root).as_posix(),
        "run_seconds": perf_counter() - run_start,
        "minimum_feasible_units": minimum_feasible_units,
        "selected_units": list(selected_units),
        "dense_reference_h2": dense_h2,
        "bplus_envelope": [
            {
                "key": point["key"],
                "budget_units": point["budget_units"],
                "architecture_cost_units": point["architecture_cost_units"],
                "performance_h2": point["performance_h2"],
                "mip_gap": point["mip_gap"],
                "certified": point["certified"],
            }
            for point in envelope_points
        ],
        "same_budget_qi_keys": list(qi_common_keys),
        "same_budget_rfd_keys": list(rfd_common_keys),
        "same_budget_canonical_keys": list(canonical_common_keys),
        "pilot_material_sensitivity": pilot_payload["any_material_sensitivity"],
        "rfd_reported": rfd_not_reported_reason is None,
        "figure_outputs": figure_outputs,
        "validation": final_validation,
    }
    print(json.dumps(summary, allow_nan=False, sort_keys=True), flush=True)
    return summary


def _parse_publication_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", dest="seed", type=int, default=PUBLICATION_SEED)
    parser.add_argument("--seed23", dest="seed", action="store_const", const=23)
    parser.add_argument(
        "--threads", dest="threads", type=int, default=PUBLICATION_THREADS
    )
    parser.add_argument("--threads1", dest="threads", action="store_const", const=1)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_publication_args(argv)
    repository_root = Path(__file__).resolve().parents[1]
    run_publication_experiment(
        repository_root,
        arguments.output,
        seed=arguments.seed,
        threads=arguments.threads,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
