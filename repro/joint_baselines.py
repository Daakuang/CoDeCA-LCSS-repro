"""Joint-hardware re-synthesis of canonical and RFD service baselines.

The baseline identity fixes only the site-level service matrix.  Actuator and
grouped-sensor binaries remain optimization variables, and every baseline is
solved under the same combined hardware-plus-service budget.  This separates
the service support inherited from a named baseline from the hardware choice
made by the B+ problem.

Raw RFD supports and their deterministic finite-menu QI repairs are retained as
distinct records.  A raw point with no solver incumbent is therefore reported
as infeasible/no-incumbent, never as a performance point.  QI repair is first
performed for the four physical measurement stations with all hardware active.
The repaired matrix is then checked on the seven grouped sensor devices, and a
solution-bearing repaired point is checked again after applying its selected
``eta`` and ``xi`` hardware mask.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Final, Literal, Sequence, TypeAlias

from .baselines import (
    Delay,
    DelayMatrix,
    MenuQIRepairResult,
    RFDPathPoint,
    SupportMatrix,
    canonical_service_templates,
    map_rfd_path_grid,
    platoon_plant_propagation_delays,
    platoon_qi_compatible,
    saturated_menu_qi_closure,
)
from .cases import JointPlatoonCase
from .deployment import ServiceDelayMatrix
from .joint_qi import (
    grouped_plant_propagation_delays,
    hardware_aware_qi_compatible,
    hosted_controller_delays,
)
from .model import BPlusProblem, BPlusSolveResult, SolverOptions, solve_bplus
from .platoon_experiment import load_legacy_rfd_path_points


BaselineFamily: TypeAlias = Literal["canonical", "rfd"]
BaselineLayer: TypeAlias = Literal["raw", "qi_repaired"]
OutcomeDisposition: TypeAlias = Literal[
    "certified_feasible",
    "feasible_uncertified",
    "infeasible",
    "infeasible_or_unbounded",
    "no_incumbent",
]
SolveFunction: TypeAlias = Callable[
    [BPlusProblem, SolverOptions | None], BPlusSolveResult
]

CANONICAL_BASELINE_NAMES: Final[tuple[str, ...]] = (
    "PF",
    "PLF",
    "BD",
    "BDL",
    "TPF",
    "TPLF",
)
EXPECTED_LEGACY_RFD_POINT_COUNT: Final[int] = 45
DEFAULT_BASELINE_HORIZON: Final[int] = 10


def _require_joint_case(case: object) -> JointPlatoonCase:
    if not isinstance(case, JointPlatoonCase):
        raise TypeError("case must be a JointPlatoonCase")
    return case


def _finite_nonnegative(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite nonnegative real number")
    try:
        normalized = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"{name} must be a finite nonnegative real number"
        ) from error
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative real number")
    return normalized


def _device_to_physical_block(case: JointPlatoonCase) -> tuple[int, ...]:
    """Map each grouped sensor device to its original physical station."""

    block_for_scalar: list[int] = []
    for block, size in enumerate(case.base.measurement_block_sizes):
        block_for_scalar.extend(block for _ in range(size))
    if len(block_for_scalar) != case.plant.p:
        raise RuntimeError("measurement block sizes do not cover the plant outputs")

    device_blocks: list[int] = []
    for device, group in enumerate(case.sensor_device_groups):
        blocks = {block_for_scalar[output] for output in group}
        if len(blocks) != 1:
            raise ValueError(
                f"sensor device {device} crosses physical measurement blocks"
            )
        device_blocks.append(next(iter(blocks)))
    return tuple(device_blocks)


def _delay_not_earlier(left: Delay, right: Delay) -> bool:
    """Return ``left <= right`` in the delay order with ``None = infinity``."""

    if left is None:
        return right is None
    return right is None or left <= right


@dataclass(frozen=True, slots=True)
class GroupedQIValidation:
    """Four-station and seven-device QI checks for one service matrix."""

    device_to_physical_block: tuple[int, ...]
    four_block_plant_delays: DelayMatrix
    seven_group_plant_delays: DelayMatrix
    stationwise_conservative: bool
    four_block_all_hardware_qi: bool
    seven_group_all_hardware_qi: bool


def grouped_qi_validation(
    case: JointPlatoonCase,
    service_delays: ServiceDelayMatrix,
) -> GroupedQIValidation:
    """Validate the conservative four-block-to-seven-group QI reduction.

    Each grouped sensor is a subset of one original station block.  Hence the
    earliest Markov lag of the original block is no later than the lag of any
    group inside it (``None`` is infinity).  Groups at one station inherit the
    same controller-delay column.  Consequently, splitting a station cannot
    create an earlier ``D P D`` path: four-block all-hardware QI is sufficient
    for seven-group all-hardware QI.  Both sides are nevertheless evaluated
    directly here so that this implication is an executable certificate.
    """

    joint = _require_joint_case(case)
    device_blocks = _device_to_physical_block(joint)
    four_delays = platoon_plant_propagation_delays(joint.base)
    seven_delays = grouped_plant_propagation_delays(
        joint.plant.A,
        joint.plant.B2,
        joint.plant.C2,
        joint.sensor_device_groups,
    )
    device_sites = joint.deployment.layout.sensor_device_sites
    stationwise_conservative = (
        len(device_blocks) == len(seven_delays)
        and all(
            device_sites[device] == block
            and all(
                _delay_not_earlier(
                    four_delays[block][actuator],
                    seven_delays[device][actuator],
                )
                for actuator in range(joint.plant.m)
            )
            for device, block in enumerate(device_blocks)
        )
    )

    actuator_sites = joint.deployment.layout.actuator_sites
    controller_delays = hosted_controller_delays(
        service_delays,
        actuator_sites,
        device_sites,
    )
    four_qi = platoon_qi_compatible(joint.base, service_delays, four_delays)
    seven_qi = hardware_aware_qi_compatible(
        controller_delays,
        seven_delays,
        (1,) * joint.plant.m,
        (1,) * len(device_sites),
    )
    if stationwise_conservative and four_qi and not seven_qi:
        raise RuntimeError(
            "four-block QI did not imply seven-group QI despite the verified "
            "stationwise delay ordering"
        )
    return GroupedQIValidation(
        device_to_physical_block=device_blocks,
        four_block_plant_delays=four_delays,
        seven_group_plant_delays=seven_delays,
        stationwise_conservative=stationwise_conservative,
        four_block_all_hardware_qi=four_qi,
        seven_group_all_hardware_qi=seven_qi,
    )


@dataclass(frozen=True, slots=True)
class JointBaselineSpec:
    """One named fixed-service baseline with free grouped hardware."""

    key: str
    source_key: str
    family: BaselineFamily
    layer: BaselineLayer
    service_delays: ServiceDelayMatrix
    qi_validation: GroupedQIValidation
    repair_changed: bool = False
    repair_iterations: int = 0
    regularization: float | None = None
    threshold: float | None = None
    raw_support: SupportMatrix | None = None

    def __post_init__(self) -> None:
        if not self.key or not self.source_key:
            raise ValueError("key and source_key must be nonempty")
        if self.family not in ("canonical", "rfd"):
            raise ValueError("family must be 'canonical' or 'rfd'")
        if self.layer not in ("raw", "qi_repaired"):
            raise ValueError("layer must be 'raw' or 'qi_repaired'")
        if self.repair_iterations < 0:
            raise ValueError("repair_iterations must be nonnegative")
        if self.layer == "raw" and (self.repair_changed or self.repair_iterations):
            raise ValueError("raw baselines cannot carry QI-repair metadata")
        if self.family == "rfd":
            if (
                self.regularization is None
                or self.threshold is None
                or self.raw_support is None
            ):
                raise ValueError("RFD baselines require path-grid metadata")
        elif any(
            value is not None
            for value in (self.regularization, self.threshold, self.raw_support)
        ):
            raise ValueError("canonical baselines cannot carry RFD metadata")


def _rfd_source_key(point: RFDPathPoint) -> str:
    return f"rfd:{point.regularization:.17g}:{point.threshold:.17g}"


def _make_spec(
    *,
    case: JointPlatoonCase,
    source_key: str,
    family: BaselineFamily,
    layer: BaselineLayer,
    service_delays: ServiceDelayMatrix,
    repair_changed: bool = False,
    repair_iterations: int = 0,
    point: RFDPathPoint | None = None,
) -> JointBaselineSpec:
    validation = grouped_qi_validation(case, service_delays)
    if layer == "qi_repaired" and not (
        validation.stationwise_conservative
        and validation.four_block_all_hardware_qi
        and validation.seven_group_all_hardware_qi
    ):
        raise RuntimeError(
            f"QI-repaired service baseline {source_key!r} failed its all-hardware "
            "four-block/seven-group certificate"
        )
    return JointBaselineSpec(
        key=f"{source_key}:{layer}",
        source_key=source_key,
        family=family,
        layer=layer,
        service_delays=service_delays,
        qi_validation=validation,
        repair_changed=repair_changed,
        repair_iterations=repair_iterations,
        regularization=None if point is None else point.regularization,
        threshold=None if point is None else point.threshold,
        raw_support=None if point is None else point.raw_support,
    )


def joint_canonical_baseline_specs(
    case: JointPlatoonCase,
    *,
    names: Sequence[str] = CANONICAL_BASELINE_NAMES,
    include_qi_repair: bool = True,
) -> tuple[JointBaselineSpec, ...]:
    """Build canonical fixed-service identities for joint re-synthesis."""

    joint = _require_joint_case(case)
    requested_names = tuple(names)
    if not requested_names or len(set(requested_names)) != len(requested_names):
        raise ValueError("names must be nonempty and unique")
    templates = canonical_service_templates(joint.base)
    unknown = tuple(name for name in requested_names if name not in templates)
    if unknown:
        raise ValueError(f"unknown canonical baseline names: {unknown}")

    specs: list[JointBaselineSpec] = []
    for name in requested_names:
        raw_choice = templates[name]
        source_key = f"canonical:{name}"
        specs.append(
            _make_spec(
                case=joint,
                source_key=source_key,
                family="canonical",
                layer="raw",
                service_delays=raw_choice.service_delays,
            )
        )
        if include_qi_repair:
            repair = saturated_menu_qi_closure(joint.base, raw_choice)
            specs.append(
                _make_spec(
                    case=joint,
                    source_key=source_key,
                    family="canonical",
                    layer="qi_repaired",
                    service_delays=repair.repaired_choice.service_delays,
                    repair_changed=repair.changed,
                    repair_iterations=repair.iterations,
                )
            )
    return tuple(specs)


def joint_rfd_baseline_specs(
    case: JointPlatoonCase,
    path_points: Sequence[RFDPathPoint],
    *,
    include_qi_repair: bool = True,
) -> tuple[JointBaselineSpec, ...]:
    """Map a complete RFD grid to raw and deterministic QI-repaired specs."""

    joint = _require_joint_case(case)
    raw_points = map_rfd_path_grid(joint.base, path_points, qi_repair=False)
    repair_cache: dict[ServiceDelayMatrix, MenuQIRepairResult] = {}
    specs: list[JointBaselineSpec] = []
    for mapped in raw_points:
        point = mapped.path_point
        source_key = _rfd_source_key(point)
        raw_delays = mapped.choice.service_delays
        specs.append(
            _make_spec(
                case=joint,
                source_key=source_key,
                family="rfd",
                layer="raw",
                service_delays=raw_delays,
                point=point,
            )
        )
        if include_qi_repair:
            repair = repair_cache.get(raw_delays)
            if repair is None:
                repair = saturated_menu_qi_closure(joint.base, mapped.choice)
                repair_cache[raw_delays] = repair
            specs.append(
                _make_spec(
                    case=joint,
                    source_key=source_key,
                    family="rfd",
                    layer="qi_repaired",
                    service_delays=repair.repaired_choice.service_delays,
                    repair_changed=repair.changed,
                    repair_iterations=repair.iterations,
                    point=point,
                )
            )
    keys = tuple(spec.key for spec in specs)
    if len(set(keys)) != len(keys):
        raise RuntimeError("joint RFD baseline keys are not unique")
    return tuple(specs)


def load_joint_rfd_baseline_specs(
    case: JointPlatoonCase,
    repository_root: Path,
    *,
    include_qi_repair: bool = True,
) -> tuple[tuple[JointBaselineSpec, ...], dict[str, object]]:
    """Load the hashed 45-point legacy grid and build joint baseline specs."""

    points, provenance = load_legacy_rfd_path_points(Path(repository_root))
    if len(points) != EXPECTED_LEGACY_RFD_POINT_COUNT:
        raise RuntimeError(
            "the declared legacy RFD baseline must contain exactly "
            f"{EXPECTED_LEGACY_RFD_POINT_COUNT} raw points"
        )
    specs = joint_rfd_baseline_specs(
        case,
        points,  # type: ignore[arg-type]
        include_qi_repair=include_qi_repair,
    )
    return specs, provenance


def make_joint_baseline_problem(
    case: JointPlatoonCase,
    spec: JointBaselineSpec,
    *,
    horizon: int = DEFAULT_BASELINE_HORIZON,
) -> BPlusProblem:
    """Fix one service matrix while leaving all hardware binaries free."""

    joint = _require_joint_case(case)
    if not isinstance(spec, JointBaselineSpec):
        raise TypeError("spec must be a JointBaselineSpec")
    return BPlusProblem(
        plant=joint.plant,
        deployment=joint.deployment,
        horizon=horizon,
        fixed_eta=None,
        fixed_xi=None,
        fixed_service_delays=spec.service_delays,
    )


@dataclass(frozen=True, slots=True)
class JointBaselineOutcome:
    """One joint re-synthesis result with explicit publication eligibility."""

    spec: JointBaselineSpec
    architecture_budget: float
    horizon: int
    result: BPlusSolveResult
    selected_hardware_qi: bool | None

    @property
    def disposition(self) -> OutcomeDisposition:
        if self.result.solution_count > 0:
            if self.result.status == "OPTIMAL":
                return "certified_feasible"
            return "feasible_uncertified"
        if self.result.status == "INFEASIBLE":
            return "infeasible"
        if self.result.status == "INF_OR_UNBD":
            return "infeasible_or_unbounded"
        return "no_incumbent"

    @property
    def publishable(self) -> bool:
        if self.disposition != "certified_feasible":
            return False
        return self.spec.layer != "qi_repaired" or self.selected_hardware_qi is True

    @property
    def performance_point(self) -> float | None:
        """Return H2 only for an optimal, publication-eligible result."""

        return self.result.performance_objective if self.publishable else None


def _validated_solver_options(
    architecture_budget: float,
    options: SolverOptions | None,
) -> SolverOptions:
    if options is None:
        return SolverOptions(
            architecture_weight=0.0,
            architecture_budget=architecture_budget,
        )
    if not isinstance(options, SolverOptions):
        raise TypeError("options must be SolverOptions")
    if options.architecture_budget is None or not math.isclose(
        options.architecture_budget,
        architecture_budget,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("options must impose the declared combined architecture budget")
    if (
        options.architecture_weight != 0.0
        or options.hardware_budget is not None
        or options.service_budget is not None
    ):
        raise ValueError(
            "baseline re-synthesis requires only one combined hard architecture budget"
        )
    return options


def _validate_incumbent(
    *,
    case: JointPlatoonCase,
    spec: JointBaselineSpec,
    budget: float,
    result: BPlusSolveResult,
) -> bool | None:
    if result.solution_count == 0:
        return None
    if result.solution_count < 0:
        raise RuntimeError("solver returned a negative solution count")
    if result.eta is None or result.xi is None:
        raise RuntimeError("solution-bearing baseline result is missing eta or xi")
    if result.service_delays != spec.service_delays:
        raise RuntimeError("solver changed the fixed baseline service matrix")
    if result.performance_objective is None or result.architecture_cost is None:
        raise RuntimeError("solution-bearing baseline result is incomplete")
    if result.architecture_cost > budget + 1.0e-8:
        raise RuntimeError("solver incumbent exceeds the common architecture budget")
    controller_delays = hosted_controller_delays(
        result.service_delays,
        case.deployment.layout.actuator_sites,
        case.deployment.layout.sensor_device_sites,
    )
    return hardware_aware_qi_compatible(
        controller_delays,
        spec.qi_validation.seven_group_plant_delays,
        result.eta,
        result.xi,
    )


def solve_joint_baselines(
    case: JointPlatoonCase,
    specs: Sequence[JointBaselineSpec],
    *,
    architecture_budget: float,
    horizon: int = DEFAULT_BASELINE_HORIZON,
    options: SolverOptions | None = None,
    solver: SolveFunction = solve_bplus,
    strict_repaired_qi: bool = True,
) -> tuple[JointBaselineOutcome, ...]:
    """Re-synthesize every fixed-service baseline under one combined budget.

    Duplicate service matrices share a solver call, but retain separate
    canonical/RFD and raw/repaired identities in the returned sequence.
    """

    joint = _require_joint_case(case)
    budget = _finite_nonnegative("architecture_budget", architecture_budget)
    baseline_specs = tuple(specs)
    if not baseline_specs:
        raise ValueError("specs must be nonempty")
    if any(not isinstance(spec, JointBaselineSpec) for spec in baseline_specs):
        raise TypeError("specs must contain JointBaselineSpec values")
    keys = tuple(spec.key for spec in baseline_specs)
    if len(set(keys)) != len(keys):
        raise ValueError("spec keys must be unique")
    solver_options = _validated_solver_options(budget, options)
    if not callable(solver):
        raise TypeError("solver must be callable")
    if not isinstance(strict_repaired_qi, bool):
        raise TypeError("strict_repaired_qi must be bool")

    result_cache: dict[ServiceDelayMatrix, BPlusSolveResult] = {}
    outcomes: list[JointBaselineOutcome] = []
    for spec in baseline_specs:
        result = result_cache.get(spec.service_delays)
        if result is None:
            problem = make_joint_baseline_problem(joint, spec, horizon=horizon)
            result = solver(problem, solver_options)
            if not isinstance(result, BPlusSolveResult):
                raise TypeError("solver must return BPlusSolveResult")
            result_cache[spec.service_delays] = result
        selected_qi = _validate_incumbent(
            case=joint,
            spec=spec,
            budget=budget,
            result=result,
        )
        if (
            strict_repaired_qi
            and spec.layer == "qi_repaired"
            and result.solution_count > 0
            and selected_qi is not True
        ):
            raise RuntimeError(
                f"QI-repaired baseline {spec.key!r} failed the selected-hardware "
                "seven-group QI check"
            )
        outcomes.append(
            JointBaselineOutcome(
                spec=spec,
                architecture_budget=budget,
                horizon=horizon,
                result=result,
                selected_hardware_qi=selected_qi,
            )
        )
    return tuple(outcomes)
