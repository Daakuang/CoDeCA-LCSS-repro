"""Fair QI, RFD-derived, and canonical platoon baseline adapters.

Delay matrices use the controller convention ``destination actuator <- source
measurement``: ``D[i][j]`` is the earliest allowed coefficient of
``y_j -> u_i``.  Plant propagation matrices use ``P[r][l]`` for
``u_l -> y_r``.  The unsaturated delay-domain QI condition is therefore

``D[i,j] <= D[i,r] + P[r,l] + D[l,j]`` for every ``i,j,r,l``,

or, equivalently, ``D <= D (min,+) P (min,+) D``.  ``None`` denotes infinity.
The exact unsaturated closure below is deliberately separate from projection
onto the prescribed finite service menu.  A projected menu repair is a
deployment adapter; it is not claimed to be a mathematically minimal QI
closure.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Iterable, Mapping, Sequence, TypeAlias, cast

import numpy as np

from .cases import PlatoonCase
from .deployment import (
    ArchitectureChoice,
    ServiceDelayMatrix,
    architecture_cost,
)
from .platoon_experiment import (
    PlatoonServiceArchitecture,
    platoon_dense_architecture,
    platoon_sparse_architecture,
)


Delay: TypeAlias = int | None
DelayMatrix: TypeAlias = tuple[tuple[Delay, ...], ...]
SupportMatrix: TypeAlias = tuple[tuple[bool, ...], ...]


def _nonnegative_real(
    name: str,
    value: object,
    *,
    finite: bool = True,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a non-boolean real number")
    normalized = float(value)
    if math.isnan(normalized) or (finite and not math.isfinite(normalized)):
        qualifier = "finite " if finite else ""
        raise ValueError(f"{name} must be a {qualifier}nonnegative real number")
    if normalized < 0.0 or normalized == -math.inf:
        raise ValueError(f"{name} must be a nonnegative real number")
    return normalized


def _nonnegative_integer(name: str, value: object) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a non-boolean integer")
    if isinstance(value, Integral):
        normalized = int(value)
    elif isinstance(value, Real):
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise TypeError(f"{name} must be a non-boolean integer")
        normalized = int(numeric)
    else:
        raise TypeError(f"{name} must be a non-boolean integer")
    if normalized < 0:
        raise ValueError(f"{name} must be nonnegative")
    return normalized


def _delay_entry(name: str, value: object) -> Delay:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a non-boolean integer or positive infinity")
    if isinstance(value, Integral):
        return _nonnegative_integer(name, value)
    if isinstance(value, Real):
        numeric = float(value)
        if math.isnan(numeric):
            raise ValueError(f"{name} must not be NaN")
        if numeric == math.inf:
            return None
        if numeric == -math.inf:
            raise ValueError(f"{name} must not be negative infinity")
    return _nonnegative_integer(name, value)


def _delay_matrix(
    name: str,
    values: object,
    *,
    shape: tuple[int, int] | None = None,
) -> DelayMatrix:
    try:
        raw_rows = tuple(tuple(row) for row in cast(Iterable[object], values))
    except TypeError as error:
        raise TypeError(f"{name} must be a two-dimensional iterable") from error
    if not raw_rows or not raw_rows[0]:
        raise ValueError(f"{name} must be nonempty")
    column_count = len(raw_rows[0])
    if any(len(row) != column_count for row in raw_rows):
        raise ValueError(f"{name} must be rectangular")
    normalized = tuple(
        tuple(
            _delay_entry(f"{name}[{row_index}][{column_index}]", value)
            for column_index, value in enumerate(row)
        )
        for row_index, row in enumerate(raw_rows)
    )
    if shape is not None and (len(normalized), column_count) != shape:
        raise ValueError(
            f"{name} must have shape {shape}, got {(len(normalized), column_count)}"
        )
    return normalized


def _support_matrix(name: str, values: object) -> SupportMatrix:
    try:
        rows = tuple(tuple(row) for row in cast(Iterable[object], values))
    except TypeError as error:
        raise TypeError(f"{name} must be a two-dimensional iterable") from error
    if not rows or not rows[0]:
        raise ValueError(f"{name} must be nonempty")
    column_count = len(rows[0])
    if any(len(row) != column_count for row in rows):
        raise ValueError(f"{name} must be rectangular")
    normalized: list[tuple[bool, ...]] = []
    for row_index, row in enumerate(rows):
        output_row: list[bool] = []
        for column_index, value in enumerate(row):
            if isinstance(value, (bool, np.bool_)):
                output_row.append(bool(value))
            elif isinstance(value, Integral) and int(value) in (0, 1):
                output_row.append(bool(value))
            else:
                raise TypeError(
                    f"{name}[{row_index}][{column_index}] must be binary"
                )
        normalized.append(tuple(output_row))
    return tuple(normalized)


def _delay_sum(*values: Delay) -> Delay:
    if any(value is None for value in values):
        return None
    return sum(cast(int, value) for value in values)


def _minimum_delay(values: Iterable[Delay]) -> Delay:
    finite = tuple(value for value in values if value is not None)
    return None if not finite else min(finite)


def _delay_leq(left: Delay, right: Delay) -> bool:
    if left is None:
        return right is None
    return right is None or left <= right


def _min_plus_qi_term(controller: DelayMatrix, plant: DelayMatrix) -> DelayMatrix:
    actuator_count = len(controller)
    measurement_count = len(controller[0])
    return tuple(
        tuple(
            _minimum_delay(
                _delay_sum(
                    controller[output_actuator][plant_measurement],
                    plant[plant_measurement][plant_actuator],
                    controller[plant_actuator][source_measurement],
                )
                for plant_measurement in range(measurement_count)
                for plant_actuator in range(actuator_count)
            )
            for source_measurement in range(measurement_count)
        )
        for output_actuator in range(actuator_count)
    )


def _validated_qi_pair(
    controller_delays: object, plant_propagation_delays: object
) -> tuple[DelayMatrix, DelayMatrix]:
    controller = _delay_matrix("controller_delays", controller_delays)
    plant = _delay_matrix("plant_propagation_delays", plant_propagation_delays)
    if len(plant) != len(controller[0]) or len(plant[0]) != len(controller):
        raise ValueError(
            "controller_delays and plant_propagation_delays must have compatible "
            "shapes (actuators, measurements) and (measurements, actuators)"
        )
    return controller, plant


@dataclass(frozen=True, slots=True)
class DelayChange:
    row: int
    column: int
    before: Delay
    after: Delay


def _delay_changes(before: DelayMatrix, after: DelayMatrix) -> tuple[DelayChange, ...]:
    if len(before) != len(after) or len(before[0]) != len(after[0]):
        raise ValueError("delay matrices must have matching shapes")
    return tuple(
        DelayChange(row, column, before[row][column], after[row][column])
        for row in range(len(before))
        for column in range(len(before[0]))
        if before[row][column] != after[row][column]
    )


@dataclass(frozen=True, slots=True)
class QIClosureResult:
    """Exact unsaturated min-plus QI closure on integer/infinite delays."""

    delays: DelayMatrix
    iterations: int
    changes: tuple[DelayChange, ...]

    @property
    def changed(self) -> bool:
        return bool(self.changes)


def qi_compatible(
    controller_delays: object,
    plant_propagation_delays: object,
) -> bool:
    """Return whether ``D <= D (min,+) P (min,+) D`` elementwise."""

    controller, plant = _validated_qi_pair(
        controller_delays, plant_propagation_delays
    )
    indirect = _min_plus_qi_term(controller, plant)
    return all(
        _delay_leq(controller[row][column], indirect[row][column])
        for row in range(len(controller))
        for column in range(len(controller[0]))
    )


def qi_closure(
    controller_delays: object,
    plant_propagation_delays: object,
) -> QIClosureResult:
    """Return the exact unsaturated fixed point ``D <- min(D, D P D)``.

    Termination follows from the discrete domain.  In a finite matrix, each
    infinity can become finite at most once; thereafter every changed entry is
    a strictly decreasing nonnegative integer and can change only finitely many
    times.
    """

    original, plant = _validated_qi_pair(
        controller_delays, plant_propagation_delays
    )
    current = original
    iterations = 0
    while True:
        indirect = _min_plus_qi_term(current, plant)
        next_value = tuple(
            tuple(
                _minimum_delay((current[row][column], indirect[row][column]))
                for column in range(len(current[0]))
            )
            for row in range(len(current))
        )
        if next_value == current:
            return QIClosureResult(
                delays=current,
                iterations=iterations,
                changes=_delay_changes(original, current),
            )
        current = next_value
        iterations += 1


def _require_platoon_case(case: object) -> PlatoonCase:
    if not isinstance(case, PlatoonCase):
        raise TypeError("case must be a PlatoonCase")
    return case


def platoon_plant_propagation_delays(
    case: PlatoonCase,
    *,
    tolerance: float = 1.0e-12,
    max_lag: int | None = None,
) -> DelayMatrix:
    """Aggregate ``C2 A^(t-1) B2`` into four-by-three physical channels."""

    platoon = _require_platoon_case(case)
    threshold = _nonnegative_real("tolerance", tolerance)
    lag_limit = (
        platoon.plant.A.shape[0] + 1
        if max_lag is None
        else _nonnegative_integer("max_lag", max_lag)
    )
    if lag_limit < 1:
        raise ValueError("max_lag must be at least one")

    block_offsets = np.cumsum((0, *platoon.measurement_block_sizes))
    delays: list[list[Delay]] = [
        [None for _ in range(platoon.plant.m)]
        for _ in range(platoon.measurement_channel_count)
    ]
    transition_power = np.eye(platoon.plant.n)
    for lag in range(1, lag_limit + 1):
        markov = platoon.plant.C2 @ transition_power @ platoon.plant.B2
        for measurement in range(platoon.measurement_channel_count):
            row_start = int(block_offsets[measurement])
            row_stop = int(block_offsets[measurement + 1])
            for actuator in range(platoon.plant.m):
                if delays[measurement][actuator] is not None:
                    continue
                block = markov[row_start:row_stop, actuator : actuator + 1]
                if float(np.max(np.abs(block))) > threshold:
                    delays[measurement][actuator] = lag
        transition_power = transition_power @ platoon.plant.A
    return tuple(tuple(row) for row in delays)


def _platoon_choice(
    case: PlatoonCase,
    architecture: ArchitectureChoice
    | PlatoonServiceArchitecture
    | ServiceDelayMatrix,
) -> ArchitectureChoice:
    platoon = _require_platoon_case(case)
    if isinstance(architecture, ArchitectureChoice):
        choice = architecture
    else:
        service_delays = (
            architecture.service_delays
            if isinstance(architecture, PlatoonServiceArchitecture)
            else architecture
        )
        choice = ArchitectureChoice(
            platoon.fixed_eta,
            platoon.fixed_xi,
            service_delays,
        )
    if choice.eta != platoon.fixed_eta or choice.xi != platoon.fixed_xi:
        raise ValueError("platoon baseline adapters require the fixed active hardware")
    platoon.deployment.validate_choice(choice)
    return choice


def platoon_controller_delays(
    case: PlatoonCase,
    architecture: ArchitectureChoice
    | PlatoonServiceArchitecture
    | ServiceDelayMatrix,
) -> DelayMatrix:
    """Extract the three-actuator by four-measurement controller delay grid."""

    platoon = _require_platoon_case(case)
    choice = _platoon_choice(platoon, architecture)
    return tuple(
        tuple(
            choice.service_delays[actuator_site][measurement_site]
            for measurement_site in range(platoon.measurement_channel_count)
        )
        for actuator_site in platoon.deployment.layout.actuator_sites
    )


def platoon_qi_compatible(
    case: PlatoonCase,
    architecture: ArchitectureChoice
    | PlatoonServiceArchitecture
    | ServiceDelayMatrix,
    plant_propagation_delays: object | None = None,
) -> bool:
    """Test QI of the controller-delay pattern induced by one service menu."""

    platoon = _require_platoon_case(case)
    propagation = (
        platoon_plant_propagation_delays(platoon)
        if plant_propagation_delays is None
        else plant_propagation_delays
    )
    return qi_compatible(
        platoon_controller_delays(platoon, architecture), propagation
    )


@dataclass(frozen=True, slots=True)
class ServiceProjectionResult:
    choice: ArchitectureChoice
    requested_controller_delays: DelayMatrix
    controller_delays: DelayMatrix
    architecture_cost: float
    projected: bool
    changes: tuple[DelayChange, ...]

    @property
    def changed(self) -> bool:
        return bool(self.changes)


def _service_cost(case: PlatoonCase, destination: int, source: int, delay: int) -> float:
    layers = case.deployment.costs.service_costs_by_delay
    if delay >= len(layers):
        raise ValueError(f"service cost is not defined for delay {delay}")
    return float(layers[delay][destination, source])


def project_to_service_menu(
    case: PlatoonCase,
    controller_delays: object,
    *,
    delay_policy: str = "least_cost",
) -> ServiceProjectionResult:
    """Provision physical services no later than requested controller delays.

    ``least_cost`` minimizes the declared per-service cost among eligible tiers
    (breaking ties toward the slower tier).  ``fastest`` selects the smallest
    eligible delay.  Mandatory services remain installed even if the requested
    controller support omits them; this difference is reported in ``changes``.
    """

    platoon = _require_platoon_case(case)
    requested = _delay_matrix(
        "controller_delays",
        controller_delays,
        shape=(platoon.plant.m, platoon.measurement_channel_count),
    )
    if delay_policy not in ("least_cost", "fastest"):
        raise ValueError("delay_policy must be 'least_cost' or 'fastest'")

    site_count = platoon.deployment.layout.site_count
    grid: list[list[Delay]] = [
        [None for _ in range(site_count)] for _ in range(site_count)
    ]
    for menu in platoon.deployment.services.directed_menus:
        if menu.mandatory_delay is not None:
            grid[menu.destination_site][menu.source_site] = menu.mandatory_delay

    for actuator, destination in enumerate(
        platoon.deployment.layout.actuator_sites
    ):
        for source in range(platoon.measurement_channel_count):
            deadline = requested[actuator][source]
            if deadline is None:
                continue
            menu = platoon.deployment.services.menu_for(
                destination=destination, source=source
            )
            if menu is None:
                raise ValueError(
                    f"no physical service menu for controller channel ({actuator}, {source})"
                )
            eligible = tuple(delay for delay in menu.finite_delays if delay <= deadline)
            if not eligible:
                raise ValueError(
                    f"no service tier meets controller delay {deadline} for "
                    f"channel ({actuator}, {source})"
                )
            if delay_policy == "fastest":
                selected = min(eligible)
            else:
                selected = min(
                    eligible,
                    key=lambda delay: (
                        _service_cost(platoon, destination, source, delay),
                        -delay,
                    ),
                )
            if menu.mandatory_delay is not None:
                selected = menu.mandatory_delay
                if selected > deadline:
                    raise ValueError(
                        f"mandatory service delay {selected} misses controller delay "
                        f"{deadline} for channel ({actuator}, {source})"
                    )
            grid[destination][source] = selected

    service_delays: ServiceDelayMatrix = tuple(tuple(row) for row in grid)
    choice = ArchitectureChoice(
        platoon.fixed_eta,
        platoon.fixed_xi,
        service_delays,
    )
    platoon.deployment.validate_choice(choice)
    projected_delays = platoon_controller_delays(platoon, choice)
    return ServiceProjectionResult(
        choice=choice,
        requested_controller_delays=requested,
        controller_delays=projected_delays,
        architecture_cost=architecture_cost(platoon.deployment, choice).total,
        projected=True,
        changes=_delay_changes(requested, projected_delays),
    )


@dataclass(frozen=True, slots=True)
class MenuQIRepairResult:
    """Finite-menu projection/repair, distinct from exact QI closure."""

    raw_choice: ArchitectureChoice
    repaired_choice: ArchitectureChoice
    controller_delays: DelayMatrix
    iterations: int
    exact_closure_iterations: int
    projected: bool
    changes: tuple[DelayChange, ...]
    raw_cost: float
    repaired_cost: float
    repair_cost: float
    qi_compatible: bool

    @property
    def changed(self) -> bool:
        return bool(self.changes)


def saturated_menu_qi_closure(
    case: PlatoonCase,
    architecture: ArchitectureChoice
    | PlatoonServiceArchitecture
    | ServiceDelayMatrix,
    plant_propagation_delays: object | None = None,
    *,
    delay_policy: str = "least_cost",
) -> MenuQIRepairResult:
    """Iterate exact closure and finite-menu projection to a QI menu pattern.

    Projection can make a requested delay earlier than the exact unsaturated
    closure, which may create new indirect paths.  The operation is therefore
    repeated.  The finite menu and elementwise nonincreasing delay sequence
    guarantee termination.  This deterministic repair is not asserted to be a
    globally least-cost or nearest QI projection.
    """

    platoon = _require_platoon_case(case)
    raw_choice = _platoon_choice(platoon, architecture)
    propagation = _delay_matrix(
        "plant_propagation_delays",
        platoon_plant_propagation_delays(platoon)
        if plant_propagation_delays is None
        else plant_propagation_delays,
        shape=(platoon.measurement_channel_count, platoon.plant.m),
    )
    original = platoon_controller_delays(platoon, raw_choice)
    current_choice = raw_choice
    current = original
    iterations = 0
    exact_iterations = 0
    finite_state_bound = 1 + sum(
        len(menu.finite_delays) + int(menu.mandatory_delay is None)
        for menu in platoon.deployment.services.directed_menus
    )
    while not qi_compatible(current, propagation):
        exact = qi_closure(current, propagation)
        exact_iterations += exact.iterations
        projection = project_to_service_menu(
            platoon, exact.delays, delay_policy=delay_policy
        )
        next_delays = projection.controller_delays
        if next_delays == current:
            raise RuntimeError("finite-menu QI projection stalled before compatibility")
        current_choice = projection.choice
        current = next_delays
        iterations += 1
        if iterations > finite_state_bound:
            raise RuntimeError("finite-menu QI projection exceeded its finite-state bound")

    raw_cost = architecture_cost(platoon.deployment, raw_choice).total
    repaired_cost = architecture_cost(platoon.deployment, current_choice).total
    return MenuQIRepairResult(
        raw_choice=raw_choice,
        repaired_choice=current_choice,
        controller_delays=current,
        iterations=iterations,
        exact_closure_iterations=exact_iterations,
        projected=iterations > 0,
        changes=_delay_changes(original, current),
        raw_cost=raw_cost,
        repaired_cost=repaired_cost,
        repair_cost=repaired_cost - raw_cost,
        qi_compatible=qi_compatible(current, propagation),
    )


def _template_choice(
    case: PlatoonCase, selected: set[tuple[int, int]]
) -> ArchitectureChoice:
    requested: list[list[Delay]] = [
        [None for _ in range(case.measurement_channel_count)]
        for _ in range(case.plant.m)
    ]
    for actuator, destination in enumerate(case.deployment.layout.actuator_sites):
        for source in range(case.measurement_channel_count):
            menu = case.deployment.services.menu_for(
                destination=destination, source=source
            )
            if menu is not None and menu.mandatory_delay is not None:
                requested[actuator][source] = menu.mandatory_delay
    for actuator, source in selected:
        destination = case.deployment.layout.actuator_sites[actuator]
        menu = case.deployment.services.menu_for(
            destination=destination, source=source
        )
        if menu is None:
            raise ValueError(
                f"canonical template requests forbidden channel ({actuator}, {source})"
            )
        requested[actuator][source] = min(menu.finite_delays)
    return project_to_service_menu(
        case, requested, delay_policy="fastest"
    ).choice


def canonical_service_templates(
    case: PlatoonCase,
) -> dict[str, ArchitectureChoice]:
    """Return fresh physical-menu choices for the declared platoon templates."""

    platoon = _require_platoon_case(case)
    follower_count = platoon.plant.m
    pf = {(actuator, actuator) for actuator in range(follower_count)}
    plf = pf | {(actuator, 0) for actuator in range(follower_count)}
    bd = pf | {
        (actuator, actuator + 2) for actuator in range(follower_count - 1)
    }
    bdl = bd | {(actuator, 0) for actuator in range(follower_count)}
    tpf: set[tuple[int, int]] = set()
    for actuator in range(follower_count):
        vehicle_position = actuator + 1
        for predecessor_position in (
            vehicle_position - 1,
            vehicle_position - 2,
        ):
            if predecessor_position >= 0:
                tpf.add((actuator, predecessor_position))
    tplf = tpf | {(actuator, 0) for actuator in range(follower_count)}
    return {
        "dense": ArchitectureChoice(
            platoon.fixed_eta,
            platoon.fixed_xi,
            platoon_dense_architecture(platoon),
        ),
        "submitted_sparse": ArchitectureChoice(
            platoon.fixed_eta,
            platoon.fixed_xi,
            platoon_sparse_architecture(platoon),
        ),
        "PF": _template_choice(platoon, pf),
        "PLF": _template_choice(platoon, plf),
        "BD": _template_choice(platoon, bd),
        "BDL": _template_choice(platoon, bdl),
        "TPF": _template_choice(platoon, tpf),
        "TPLF": _template_choice(platoon, tplf),
    }


@dataclass(frozen=True, slots=True)
class RFDPathPoint:
    """One caller-supplied point from a complete RFD threshold grid."""

    regularization: float
    threshold: float
    raw_support: SupportMatrix
    specified_delays: DelayMatrix | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "regularization",
            _nonnegative_real("regularization", self.regularization),
        )
        object.__setattr__(
            self, "threshold", _nonnegative_real("threshold", self.threshold)
        )
        support = _support_matrix("raw_support", self.raw_support)
        object.__setattr__(self, "raw_support", support)
        if self.specified_delays is not None:
            delays = _delay_matrix(
                "specified_delays",
                self.specified_delays,
                shape=(len(support), len(support[0])),
            )
            for row in range(len(support)):
                for column in range(len(support[0])):
                    if support[row][column] != (delays[row][column] is not None):
                        raise ValueError(
                            "specified_delays must be finite exactly on raw_support"
                        )
            object.__setattr__(self, "specified_delays", delays)


@dataclass(frozen=True, slots=True)
class RFDServicePoint:
    path_point: RFDPathPoint
    choice: ArchitectureChoice
    projection: ServiceProjectionResult
    qi_repaired: bool
    repair: MenuQIRepairResult | None


def _rfd_requested_delays(case: PlatoonCase, point: RFDPathPoint) -> DelayMatrix:
    if point.specified_delays is not None:
        return point.specified_delays
    requested: list[list[Delay]] = [
        [None for _ in range(case.measurement_channel_count)]
        for _ in range(case.plant.m)
    ]
    for actuator in range(case.plant.m):
        destination = case.deployment.layout.actuator_sites[actuator]
        for source in range(case.measurement_channel_count):
            if not point.raw_support[actuator][source]:
                continue
            menu = case.deployment.services.menu_for(
                destination=destination, source=source
            )
            if menu is None:
                raise ValueError(
                    f"RFD support requests forbidden channel ({actuator}, {source})"
                )
            requested[actuator][source] = max(menu.finite_delays)
    return tuple(tuple(row) for row in requested)


def map_rfd_path_grid(
    case: PlatoonCase,
    path_points: Sequence[RFDPathPoint],
    *,
    qi_repair: bool = False,
    plant_propagation_delays: object | None = None,
) -> tuple[RFDServicePoint, ...]:
    """Map a complete caller-supplied RFD grid to physical services.

    This function performs no RFD optimization and invents no missing path
    points.  The distinct regularization and threshold values must form a
    complete Cartesian grid, with exactly one supplied record per pair.
    Returned choices are intended for re-synthesis through
    :func:`repro.platoon_experiment.build_platoon_problem`.
    """

    platoon = _require_platoon_case(case)
    if not isinstance(qi_repair, bool):
        raise TypeError("qi_repair must be bool")
    try:
        points = tuple(path_points)
    except TypeError as error:
        raise TypeError("path_points must be an iterable of RFDPathPoint") from error
    if not points:
        raise ValueError("path_points must contain a complete Cartesian grid")
    if any(not isinstance(point, RFDPathPoint) for point in points):
        raise TypeError("path_points must contain RFDPathPoint values")
    expected_shape = (platoon.plant.m, platoon.measurement_channel_count)
    for point in points:
        if (len(point.raw_support), len(point.raw_support[0])) != expected_shape:
            raise ValueError(
                f"raw_support must have shape {expected_shape}, got "
                f"{(len(point.raw_support), len(point.raw_support[0]))}"
            )
    regularizations = tuple(sorted({point.regularization for point in points}))
    thresholds = tuple(sorted({point.threshold for point in points}))
    keys = tuple((point.regularization, point.threshold) for point in points)
    if len(set(keys)) != len(keys) or set(keys) != {
        (regularization, threshold)
        for regularization in regularizations
        for threshold in thresholds
    }:
        raise ValueError(
            "path_points must contain exactly one complete Cartesian grid"
        )

    by_key: Mapping[tuple[float, float], RFDPathPoint] = {
        (point.regularization, point.threshold): point for point in points
    }
    mapped: list[RFDServicePoint] = []
    for regularization in regularizations:
        for threshold in thresholds:
            point = by_key[(regularization, threshold)]
            projection = project_to_service_menu(
                platoon,
                _rfd_requested_delays(platoon, point),
                delay_policy="least_cost",
            )
            if qi_repair:
                repair = saturated_menu_qi_closure(
                    platoon,
                    projection.choice,
                    plant_propagation_delays,
                )
                choice = repair.repaired_choice
            else:
                repair = None
                choice = projection.choice
            mapped.append(
                RFDServicePoint(
                    path_point=point,
                    choice=choice,
                    projection=projection,
                    qi_repaired=qi_repair,
                    repair=repair,
                )
            )
    return tuple(mapped)


@dataclass(frozen=True, slots=True)
class BaselineRecord:
    """One fully enumerated, fixed-architecture re-synthesis record."""

    key: str
    choice: ArchitectureChoice
    architecture_cost: float
    performance: float
    qi_compatible: bool
    feasible: bool

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("key must be a nonempty string")
        if not isinstance(self.choice, ArchitectureChoice):
            raise TypeError("choice must be an ArchitectureChoice")
        object.__setattr__(
            self,
            "architecture_cost",
            _nonnegative_real("architecture_cost", self.architecture_cost),
        )
        object.__setattr__(
            self,
            "performance",
            _nonnegative_real("performance", self.performance, finite=False),
        )
        if not isinstance(self.qi_compatible, bool):
            raise TypeError("qi_compatible must be bool")
        if not isinstance(self.feasible, bool):
            raise TypeError("feasible must be bool")


@dataclass(frozen=True, slots=True)
class SameBudgetResult:
    budget: float
    budget_tolerance: float
    tie_tolerance: float
    best_performance: float
    argmin_records: tuple[BaselineRecord, ...]
    eligible_count: int


def best_same_budget_qi(
    records: Iterable[BaselineRecord],
    *,
    budget: float,
    budget_tolerance: float = 1.0e-12,
    tie_tolerance: float = 1.0e-12,
) -> SameBudgetResult:
    """Globally minimize performance over all eligible enumerated QI records.

    Eligibility means QI-compatible, feasible, finite performance, and
    ``architecture_cost <= budget + budget_tolerance``.  This is neither an
    exact-cost comparison nor a search restricted to scalarized-path points.
    """

    normalized_budget = _nonnegative_real("budget", budget)
    normalized_budget_tolerance = _nonnegative_real(
        "budget_tolerance", budget_tolerance
    )
    normalized_tie_tolerance = _nonnegative_real(
        "tie_tolerance", tie_tolerance
    )
    try:
        family = tuple(records)
    except TypeError as error:
        raise TypeError("records must be an iterable of BaselineRecord") from error
    if any(not isinstance(record, BaselineRecord) for record in family):
        raise TypeError("records must contain BaselineRecord values")
    seen_keys: set[str] = set()
    duplicate_key_set: set[str] = set()
    for record in family:
        if record.key in seen_keys:
            duplicate_key_set.add(record.key)
        seen_keys.add(record.key)
    duplicate_keys = tuple(sorted(duplicate_key_set))
    if duplicate_keys:
        formatted = ", ".join(duplicate_keys)
        raise ValueError(
            "BaselineRecord.key values must be globally unique; "
            f"duplicate key(s): {formatted}"
        )
    eligible = tuple(
        record
        for record in family
        if record.qi_compatible
        and record.feasible
        and math.isfinite(record.performance)
        and record.architecture_cost
        <= normalized_budget + normalized_budget_tolerance
    )
    if not eligible:
        return SameBudgetResult(
            budget=normalized_budget,
            budget_tolerance=normalized_budget_tolerance,
            tie_tolerance=normalized_tie_tolerance,
            best_performance=math.inf,
            argmin_records=(),
            eligible_count=0,
        )
    best = min(record.performance for record in eligible)
    argmin = tuple(
        sorted(
            (
                record
                for record in eligible
                if abs(record.performance - best) <= normalized_tie_tolerance
            ),
            key=lambda record: record.key,
        )
    )
    return SameBudgetResult(
        budget=normalized_budget,
        budget_tolerance=normalized_budget_tolerance,
        tie_tolerance=normalized_tie_tolerance,
        best_performance=best,
        argmin_records=argmin,
        eligible_count=len(eligible),
    )
