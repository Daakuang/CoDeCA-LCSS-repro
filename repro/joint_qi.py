"""Exact hardware-aware QI certificates for finite service catalogs.

Controller delays use the convention ``D[i][j]`` for measurement block
``j`` to actuator ``i``.  Plant propagation delays use ``P[r][l]`` for
actuator ``l`` to measurement block ``r``.  ``None`` denotes infinity.

For a fixed service menu, deactivating an actuator or sensor replaces the
corresponding row or column of ``D`` by infinity.  Every all-active QI
violation therefore defines a forbidden set of at most two actuator devices
and two sensor devices.  A hardware selection is QI exactly when it contains
none of these forbidden sets.  This hypergraph certificate avoids repeated
min-plus tests over the complete hardware power set.

For a variable finite service menu, :func:`finite_catalog_qi_no_goods`
enumerates the exact conjunctions of hardware and one-hot service states that
violate QI.  Adding one linear no-good inequality for each conjunction gives
an exact QI-restricted MICP over the prescribed finite catalog.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import itertools
from numbers import Integral, Real
from typing import TYPE_CHECKING, Iterable, Iterator, Sequence, TypeAlias

import numpy as np

from .baselines import Delay, DelayMatrix, qi_compatible
from .deployment import ServiceCatalog, ServiceDelayMatrix

if TYPE_CHECKING:
    from .model import BPlusProblem, BPlusSolveResult, SolverOptions


BinaryVector: TypeAlias = tuple[int, ...]


def _nonnegative_integer(name: str, value: object) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a non-boolean integer")
    if isinstance(value, Integral):
        normalized = int(value)
    elif isinstance(value, Real):
        numeric = float(value)
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise TypeError(f"{name} must be a non-boolean integer")
        normalized = int(numeric)
    else:
        raise TypeError(f"{name} must be a non-boolean integer")
    if normalized < 0:
        raise ValueError(f"{name} must be nonnegative")
    return normalized


def _delay(name: str, value: object) -> Delay:
    if value is None:
        return None
    if isinstance(value, Real) and not isinstance(value, (bool, np.bool_)):
        numeric = float(value)
        if np.isposinf(numeric):
            return None
        if np.isnan(numeric):
            raise ValueError(f"{name} must not be NaN")
    return _nonnegative_integer(name, value)


def _delay_matrix(name: str, values: object) -> DelayMatrix:
    try:
        rows = tuple(tuple(row) for row in values)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(f"{name} must be a two-dimensional iterable") from error
    if not rows or not rows[0]:
        raise ValueError(f"{name} must be nonempty")
    column_count = len(rows[0])
    if any(len(row) != column_count for row in rows):
        raise ValueError(f"{name} must be rectangular")
    return tuple(
        tuple(
            _delay(f"{name}[{row_index}][{column_index}]", value)
            for column_index, value in enumerate(row)
        )
        for row_index, row in enumerate(rows)
    )


def _binary_vector(name: str, values: object, length: int) -> BinaryVector:
    try:
        raw = tuple(values)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable") from error
    if len(raw) != length:
        raise ValueError(f"{name} must have length {length}")
    normalized: list[int] = []
    for index, value in enumerate(raw):
        if isinstance(value, (bool, np.bool_)):
            normalized.append(int(value))
        elif isinstance(value, Integral) and int(value) in (0, 1):
            normalized.append(int(value))
        else:
            raise TypeError(f"{name}[{index}] must be binary")
    return tuple(normalized)


def _site_vector(
    name: str, values: object, *, site_count: int | None = None
) -> tuple[int, ...]:
    try:
        raw = tuple(values)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable") from error
    if not raw:
        raise ValueError(f"{name} must be nonempty")
    normalized = tuple(
        _nonnegative_integer(f"{name}[{index}]", value)
        for index, value in enumerate(raw)
    )
    if site_count is not None and any(site >= site_count for site in normalized):
        raise IndexError(f"{name} contains a site outside 0:{site_count}")
    return normalized


def _plant_delays(
    values: object, *, measurement_count: int, actuator_count: int
) -> DelayMatrix:
    normalized = _delay_matrix("plant_propagation_delays", values)
    shape = (len(normalized), len(normalized[0]))
    expected = (measurement_count, actuator_count)
    if shape != expected:
        raise ValueError(
            f"plant_propagation_delays must have shape {expected}, got {shape}"
        )
    return normalized


def grouped_plant_propagation_delays(
    A: object,
    B2: object,
    C2: object,
    sensor_groups: object,
    *,
    tolerance: float = 1.0e-12,
    max_lag: int | None = None,
) -> DelayMatrix:
    """Compute earliest ``C2 A^(t-1) B2`` support for sensor packages.

    ``sensor_groups`` must partition the scalar rows of ``C2``.  A package-to-
    actuator entry is finite at the first lag for which any scalar coordinate
    in that package has magnitude greater than ``tolerance``.
    """

    transition = np.asarray(A, dtype=float)
    input_matrix = np.asarray(B2, dtype=float)
    output_matrix = np.asarray(C2, dtype=float)
    if transition.ndim != 2 or transition.shape[0] != transition.shape[1]:
        raise ValueError("A must be square")
    state_count = transition.shape[0]
    if input_matrix.ndim != 2 or input_matrix.shape[0] != state_count:
        raise ValueError("B2 must have shape (A.shape[0], actuator_count)")
    if output_matrix.ndim != 2 or output_matrix.shape[1] != state_count:
        raise ValueError("C2 must have shape (measurement_count, A.shape[0])")
    if isinstance(tolerance, (bool, np.bool_)) or not isinstance(tolerance, Real):
        raise TypeError("tolerance must be a non-boolean real number")
    threshold = float(tolerance)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")

    try:
        raw_groups = tuple(tuple(group) for group in sensor_groups)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError("sensor_groups must be a two-dimensional iterable") from error
    if not raw_groups or any(not group for group in raw_groups):
        raise ValueError("sensor_groups must contain nonempty groups")
    groups = tuple(
        tuple(
            _nonnegative_integer(f"sensor_groups[{group_index}][{index}]", value)
            for index, value in enumerate(group)
        )
        for group_index, group in enumerate(raw_groups)
    )
    flat = tuple(index for group in groups for index in group)
    if tuple(sorted(flat)) != tuple(range(output_matrix.shape[0])):
        raise ValueError("sensor_groups must partition the scalar rows of C2")

    lag_limit = state_count + 1 if max_lag is None else _nonnegative_integer(
        "max_lag", max_lag
    )
    if lag_limit < 1:
        raise ValueError("max_lag must be at least one")
    delays: list[list[Delay]] = [
        [None for _ in range(input_matrix.shape[1])] for _ in groups
    ]
    transition_power = np.eye(state_count)
    for lag in range(1, lag_limit + 1):
        markov = output_matrix @ transition_power @ input_matrix
        for measurement, group in enumerate(groups):
            for actuator in range(input_matrix.shape[1]):
                if delays[measurement][actuator] is not None:
                    continue
                block = markov[np.asarray(group), actuator]
                if float(np.max(np.abs(block))) > threshold:
                    delays[measurement][actuator] = lag
        transition_power = transition_power @ transition
    return tuple(tuple(row) for row in delays)


def hosted_controller_delays(
    service_delays: object,
    actuator_sites: object,
    sensor_sites: object,
) -> DelayMatrix:
    """Map a site-level service menu to the all-active controller delays."""

    services = _delay_matrix("service_delays", service_delays)
    if len(services) != len(services[0]):
        raise ValueError("service_delays must be square")
    actuators = _site_vector(
        "actuator_sites", actuator_sites, site_count=len(services)
    )
    sensors = _site_vector("sensor_sites", sensor_sites, site_count=len(services))
    return tuple(
        tuple(services[actuator_site][sensor_site] for sensor_site in sensors)
        for actuator_site in actuators
    )


def masked_controller_delays(
    all_active_controller_delays: object,
    eta: object,
    xi: object,
) -> DelayMatrix:
    """Set every inactive actuator row or sensor column to infinity."""

    controller = _delay_matrix(
        "all_active_controller_delays", all_active_controller_delays
    )
    actuator_active = _binary_vector("eta", eta, len(controller))
    sensor_active = _binary_vector("xi", xi, len(controller[0]))
    return tuple(
        tuple(
            controller[actuator][sensor]
            if actuator_active[actuator] and sensor_active[sensor]
            else None
            for sensor in range(len(sensor_active))
        )
        for actuator in range(len(actuator_active))
    )


def hardware_aware_qi_compatible(
    all_active_controller_delays: object,
    plant_propagation_delays: object,
    eta: object,
    xi: object,
) -> bool:
    """Test QI after applying independent actuator and sensor selections."""

    controller = _delay_matrix(
        "all_active_controller_delays", all_active_controller_delays
    )
    plant = _plant_delays(
        plant_propagation_delays,
        measurement_count=len(controller[0]),
        actuator_count=len(controller),
    )
    return qi_compatible(masked_controller_delays(controller, eta, xi), plant)


def hardware_mask(eta: object, xi: object) -> int:
    """Pack actuator bits first and sensor bits second into one integer."""

    try:
        actuator_count = len(eta)  # type: ignore[arg-type]
        sensor_count = len(xi)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError("eta and xi must be sized iterables") from error
    actuator_active = _binary_vector("eta", eta, actuator_count)
    sensor_active = _binary_vector("xi", xi, sensor_count)
    mask = sum(value << index for index, value in enumerate(actuator_active))
    return mask | sum(
        value << (actuator_count + index)
        for index, value in enumerate(sensor_active)
    )


def hardware_vectors(
    mask: object, *, actuator_count: int, sensor_count: int
) -> tuple[BinaryVector, BinaryVector]:
    """Unpack a validated integer hardware mask into ``(eta, xi)``."""

    actuators = _nonnegative_integer("actuator_count", actuator_count)
    sensors = _nonnegative_integer("sensor_count", sensor_count)
    if actuators < 1 or sensors < 1:
        raise ValueError("actuator_count and sensor_count must be positive")
    normalized = _nonnegative_integer("mask", mask)
    if normalized >= 1 << (actuators + sensors):
        raise ValueError("mask contains bits outside the declared hardware dimensions")
    eta = tuple((normalized >> index) & 1 for index in range(actuators))
    xi = tuple(
        (normalized >> (actuators + index)) & 1 for index in range(sensors)
    )
    return eta, xi


@dataclass(frozen=True, slots=True)
class QIViolation:
    """One strict min-plus QI violation under all-active hardware."""

    target_actuator: int
    target_sensor: int
    intermediate_sensor: int
    intermediate_actuator: int
    target_delay: Delay
    first_controller_delay: int
    plant_delay: int
    second_controller_delay: int
    required_hardware_mask: int

    @property
    def indirect_delay(self) -> int:
        return (
            self.first_controller_delay
            + self.plant_delay
            + self.second_controller_delay
        )


def all_active_qi_violations(
    controller_delays: object,
    plant_propagation_delays: object,
) -> tuple[QIViolation, ...]:
    """Return every strict QI inequality violation for an all-active menu."""

    controller = _delay_matrix("controller_delays", controller_delays)
    actuator_count = len(controller)
    sensor_count = len(controller[0])
    plant = _plant_delays(
        plant_propagation_delays,
        measurement_count=sensor_count,
        actuator_count=actuator_count,
    )
    violations: list[QIViolation] = []
    for target_actuator in range(actuator_count):
        for target_sensor in range(sensor_count):
            target = controller[target_actuator][target_sensor]
            for intermediate_sensor in range(sensor_count):
                first = controller[target_actuator][intermediate_sensor]
                if first is None:
                    continue
                for intermediate_actuator in range(actuator_count):
                    propagation = plant[intermediate_sensor][intermediate_actuator]
                    second = controller[intermediate_actuator][target_sensor]
                    if propagation is None or second is None:
                        continue
                    indirect = first + propagation + second
                    if target is not None and indirect >= target:
                        continue
                    required = (
                        (1 << target_actuator)
                        | (1 << intermediate_actuator)
                        | (1 << (actuator_count + intermediate_sensor))
                        | (1 << (actuator_count + target_sensor))
                    )
                    violations.append(
                        QIViolation(
                            target_actuator=target_actuator,
                            target_sensor=target_sensor,
                            intermediate_sensor=intermediate_sensor,
                            intermediate_actuator=intermediate_actuator,
                            target_delay=target,
                            first_controller_delay=first,
                            plant_delay=propagation,
                            second_controller_delay=second,
                            required_hardware_mask=required,
                        )
                    )
    return tuple(violations)


def _minimal_masks(values: Iterable[int]) -> tuple[int, ...]:
    unique = tuple(sorted(set(values), key=lambda value: (value.bit_count(), value)))
    return tuple(
        value
        for index, value in enumerate(unique)
        if not any((other & value) == other for other in unique[:index])
    )


@dataclass(frozen=True, slots=True)
class FixedServiceQICertificate:
    """Exact forbidden-hardware certificate for one fixed service menu."""

    actuator_count: int
    sensor_count: int
    forbidden_hardware_masks: tuple[int, ...]

    def __post_init__(self) -> None:
        actuator_count = _nonnegative_integer("actuator_count", self.actuator_count)
        sensor_count = _nonnegative_integer("sensor_count", self.sensor_count)
        if actuator_count < 1 or sensor_count < 1:
            raise ValueError("actuator_count and sensor_count must be positive")
        object.__setattr__(self, "actuator_count", actuator_count)
        object.__setattr__(self, "sensor_count", sensor_count)
        full_limit = 1 << (actuator_count + sensor_count)
        masks = tuple(self.forbidden_hardware_masks)
        if any(mask <= 0 or mask >= full_limit for mask in masks):
            raise ValueError("forbidden_hardware_masks contains an invalid mask")
        minimal = _minimal_masks(masks)
        if masks != minimal:
            raise ValueError(
                "forbidden_hardware_masks must be unique, minimal, and canonical"
            )

    @property
    def hardware_pattern_count(self) -> int:
        return 1 << (self.actuator_count + self.sensor_count)

    @property
    def all_active_compatible(self) -> bool:
        return not self.forbidden_hardware_masks

    def is_compatible(self, mask: object) -> bool:
        normalized = _nonnegative_integer("mask", mask)
        if normalized >= self.hardware_pattern_count:
            raise ValueError("mask contains bits outside the certificate dimensions")
        return not any(
            normalized & forbidden == forbidden
            for forbidden in self.forbidden_hardware_masks
        )

    def iter_compatible_masks(self) -> Iterator[int]:
        return (
            mask
            for mask in range(self.hardware_pattern_count)
            if self.is_compatible(mask)
        )

    @property
    def compatible_hardware_count(self) -> int:
        return sum(1 for _ in self.iter_compatible_masks())


def fixed_service_qi_certificate(
    controller_delays: object,
    plant_propagation_delays: object,
) -> FixedServiceQICertificate:
    """Build the exact minimal forbidden-hardware certificate for one menu."""

    controller = _delay_matrix("controller_delays", controller_delays)
    violations = all_active_qi_violations(
        controller, plant_propagation_delays
    )
    return FixedServiceQICertificate(
        actuator_count=len(controller),
        sensor_count=len(controller[0]),
        forbidden_hardware_masks=_minimal_masks(
            violation.required_hardware_mask for violation in violations
        ),
    )


def service_menu_qi_certificates(
    service_menus: Iterable[ServiceDelayMatrix],
    actuator_sites: object,
    sensor_sites: object,
    plant_propagation_delays: object,
) -> tuple[FixedServiceQICertificate, ...]:
    """Build fixed-service certificates in the supplied deterministic order."""

    actuators = _site_vector("actuator_sites", actuator_sites)
    sensors = _site_vector("sensor_sites", sensor_sites)
    plant = _plant_delays(
        plant_propagation_delays,
        measurement_count=len(sensors),
        actuator_count=len(actuators),
    )
    return tuple(
        fixed_service_qi_certificate(
            hosted_controller_delays(menu, actuators, sensors), plant
        )
        for menu in service_menus
    )


@dataclass(frozen=True, slots=True)
class JointQIEnumerationSummary:
    """Cardinality certificate for a complete service-by-hardware product."""

    service_menu_count: int
    hardware_pattern_count: int
    joint_architecture_count: int
    joint_qi_count: int
    all_active_qi_menu_indices: tuple[int, ...]
    distinct_forbidden_pattern_count: int


def summarize_joint_qi(
    certificates: Sequence[FixedServiceQICertificate],
) -> JointQIEnumerationSummary:
    """Count all QI pairs exactly, caching only within this call."""

    normalized = tuple(certificates)
    if not normalized:
        raise ValueError("certificates must be nonempty")
    dimensions = (
        normalized[0].actuator_count,
        normalized[0].sensor_count,
    )
    if any(
        (item.actuator_count, item.sensor_count) != dimensions
        for item in normalized
    ):
        raise ValueError("all certificates must use the same hardware dimensions")

    count_by_pattern: dict[tuple[int, ...], int] = {}
    joint_qi_count = 0
    all_active_indices: list[int] = []
    for index, certificate in enumerate(normalized):
        pattern = certificate.forbidden_hardware_masks
        if pattern not in count_by_pattern:
            count_by_pattern[pattern] = certificate.compatible_hardware_count
        joint_qi_count += count_by_pattern[pattern]
        if certificate.all_active_compatible:
            all_active_indices.append(index)

    hardware_count = normalized[0].hardware_pattern_count
    return JointQIEnumerationSummary(
        service_menu_count=len(normalized),
        hardware_pattern_count=hardware_count,
        joint_architecture_count=len(normalized) * hardware_count,
        joint_qi_count=joint_qi_count,
        all_active_qi_menu_indices=tuple(all_active_indices),
        distinct_forbidden_pattern_count=len(count_by_pattern),
    )


@dataclass(frozen=True, slots=True)
class AdmissibleSensorPatternQICount:
    """QI service count for one PBH-admissible sensor mask."""

    sensor_mask: int
    xi: BinaryVector
    qi_service_menu_count: int


@dataclass(frozen=True, slots=True)
class AdmissibleJointQISummary:
    """Exact QI count restricted to fixed actuators and sensor patterns."""

    service_menu_count: int
    admissible_sensor_pattern_count: int
    joint_candidate_count: int
    joint_qi_count: int
    eta: BinaryVector
    per_sensor_pattern: tuple[AdmissibleSensorPatternQICount, ...]


def summarize_admissible_joint_qi(
    certificates: Sequence[FixedServiceQICertificate],
    eta: object,
    admissible_sensor_patterns: Iterable[object],
) -> AdmissibleJointQISummary:
    """Count QI pairs over an explicitly supplied admissible hardware set."""

    normalized = tuple(certificates)
    if not normalized:
        raise ValueError("certificates must be nonempty")
    actuator_count = normalized[0].actuator_count
    sensor_count = normalized[0].sensor_count
    if any(
        (item.actuator_count, item.sensor_count)
        != (actuator_count, sensor_count)
        for item in normalized
    ):
        raise ValueError("all certificates must use the same hardware dimensions")
    actuator_active = _binary_vector("eta", eta, actuator_count)
    try:
        raw_patterns = tuple(admissible_sensor_patterns)
    except TypeError as error:
        raise TypeError("admissible_sensor_patterns must be an iterable") from error
    patterns = tuple(
        _binary_vector(f"admissible_sensor_patterns[{index}]", item, sensor_count)
        for index, item in enumerate(raw_patterns)
    )
    if len(set(patterns)) != len(patterns):
        raise ValueError("admissible_sensor_patterns must be unique")

    per_pattern: list[AdmissibleSensorPatternQICount] = []
    for xi in sorted(patterns, key=lambda value: hardware_mask((), value)):
        mask = hardware_mask(actuator_active, xi)
        per_pattern.append(
            AdmissibleSensorPatternQICount(
                sensor_mask=hardware_mask((), xi),
                xi=xi,
                qi_service_menu_count=sum(
                    certificate.is_compatible(mask) for certificate in normalized
                ),
            )
        )
    return AdmissibleJointQISummary(
        service_menu_count=len(normalized),
        admissible_sensor_pattern_count=len(per_pattern),
        joint_candidate_count=len(normalized) * len(per_pattern),
        joint_qi_count=sum(item.qi_service_menu_count for item in per_pattern),
        eta=actuator_active,
        per_sensor_pattern=tuple(per_pattern),
    )


@dataclass(frozen=True, slots=True)
class BudgetFeasibleQIDeployment:
    """One exact service/hardware index within an integer architecture budget."""

    service_index: int
    hardware_mask: int
    service_cost_units: int
    hardware_cost_units: int

    @property
    def total_cost_units(self) -> int:
        return self.service_cost_units + self.hardware_cost_units


def hardware_cost_units(
    mask: object,
    actuator_cost_units: object,
    sensor_cost_units: object,
) -> int:
    """Evaluate independent nonnegative integer device costs for one mask."""

    try:
        raw_actuator_costs = tuple(actuator_cost_units)  # type: ignore[arg-type]
        raw_sensor_costs = tuple(sensor_cost_units)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError("device costs must be iterables") from error
    actuator_costs = tuple(
        _nonnegative_integer(f"actuator_cost_units[{index}]", value)
        for index, value in enumerate(raw_actuator_costs)
    )
    sensor_costs = tuple(
        _nonnegative_integer(f"sensor_cost_units[{index}]", value)
        for index, value in enumerate(raw_sensor_costs)
    )
    eta, xi = hardware_vectors(
        mask,
        actuator_count=len(actuator_costs),
        sensor_count=len(sensor_costs),
    )
    return sum(
        cost * selected for cost, selected in zip(actuator_costs, eta, strict=True)
    ) + sum(
        cost * selected for cost, selected in zip(sensor_costs, xi, strict=True)
    )


def iter_budget_feasible_qi_deployments(
    certificates: Sequence[FixedServiceQICertificate],
    service_cost_units: object,
    actuator_cost_units: object,
    sensor_cost_units: object,
    *,
    budget_units: int,
) -> Iterator[BudgetFeasibleQIDeployment]:
    """Enumerate the complete QI deployment set under one integer budget."""

    normalized = tuple(certificates)
    try:
        raw_service_costs = tuple(service_cost_units)  # type: ignore[arg-type]
        raw_actuator_costs = tuple(actuator_cost_units)  # type: ignore[arg-type]
        raw_sensor_costs = tuple(sensor_cost_units)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError("cost inputs must be iterables") from error
    if len(normalized) != len(raw_service_costs):
        raise ValueError("service_cost_units must match the certificate count")
    services = tuple(
        _nonnegative_integer(f"service_cost_units[{index}]", value)
        for index, value in enumerate(raw_service_costs)
    )
    actuator_costs = tuple(
        _nonnegative_integer(f"actuator_cost_units[{index}]", value)
        for index, value in enumerate(raw_actuator_costs)
    )
    sensor_costs = tuple(
        _nonnegative_integer(f"sensor_cost_units[{index}]", value)
        for index, value in enumerate(raw_sensor_costs)
    )
    budget = _nonnegative_integer("budget_units", budget_units)
    if normalized:
        dimensions = (
            normalized[0].actuator_count,
            normalized[0].sensor_count,
        )
        if dimensions != (len(actuator_costs), len(sensor_costs)):
            raise ValueError("device cost dimensions must match the QI certificates")
        if any(
            (item.actuator_count, item.sensor_count) != dimensions
            for item in normalized
        ):
            raise ValueError("all certificates must use the same dimensions")

    hardware_cost_by_mask = tuple(
        hardware_cost_units(mask, actuator_costs, sensor_costs)
        for mask in range(1 << (len(actuator_costs) + len(sensor_costs)))
    )
    return (
        BudgetFeasibleQIDeployment(
            service_index=service_index,
            hardware_mask=mask,
            service_cost_units=services[service_index],
            hardware_cost_units=hardware_cost_by_mask[mask],
        )
        for service_index, certificate in enumerate(normalized)
        for mask in certificate.iter_compatible_masks()
        if services[service_index] + hardware_cost_by_mask[mask] <= budget
    )


@dataclass(frozen=True, slots=True, order=True)
class ServiceStateLiteral:
    """One optional one-hot service state ``destination <- source``."""

    destination_site: int
    source_site: int
    delay: Delay


@dataclass(frozen=True, slots=True)
class QINoGood:
    """A conjunction that must be excluded to impose finite-catalog QI."""

    actuator_devices: tuple[int, ...]
    sensor_devices: tuple[int, ...]
    service_states: tuple[ServiceStateLiteral, ...]

    @property
    def literal_count(self) -> int:
        return (
            len(self.actuator_devices)
            + len(self.sensor_devices)
            + len(self.service_states)
        )

    def is_triggered(
        self, eta: object, xi: object, service_delays: object
    ) -> bool:
        try:
            actuator_count = len(eta)  # type: ignore[arg-type]
            sensor_count = len(xi)  # type: ignore[arg-type]
        except TypeError as error:
            raise TypeError("eta and xi must be sized iterables") from error
        actuator_active = _binary_vector("eta", eta, actuator_count)
        sensor_active = _binary_vector("xi", xi, sensor_count)
        services = _delay_matrix("service_delays", service_delays)
        return (
            all(actuator_active[index] for index in self.actuator_devices)
            and all(sensor_active[index] for index in self.sensor_devices)
            and all(
                services[state.destination_site][state.source_site] == state.delay
                for state in self.service_states
            )
        )


def _no_good_literals(item: QINoGood) -> frozenset[tuple[object, ...]]:
    return frozenset(
        itertools.chain(
            (("actuator", index) for index in item.actuator_devices),
            (("sensor", index) for index in item.sensor_devices),
            (
                (
                    "service",
                    state.destination_site,
                    state.source_site,
                    state.delay,
                )
                for state in item.service_states
            ),
        )
    )


def _minimal_no_goods(values: Iterable[QINoGood]) -> tuple[QINoGood, ...]:
    unique = tuple(set(values))
    ordered = tuple(
        sorted(
            unique,
            key=lambda item: (
                item.literal_count,
                item.actuator_devices,
                item.sensor_devices,
                tuple(
                    (
                        state.destination_site,
                        state.source_site,
                        state.delay is None,
                        -1 if state.delay is None else state.delay,
                    )
                    for state in item.service_states
                ),
            ),
        )
    )
    literal_sets = tuple(_no_good_literals(item) for item in ordered)
    return tuple(
        item
        for index, item in enumerate(ordered)
        if not any(other < literal_sets[index] for other in literal_sets[:index])
    )


def finite_catalog_qi_no_goods(
    catalog: ServiceCatalog,
    actuator_sites: object,
    sensor_sites: object,
    plant_propagation_delays: object,
) -> tuple[QINoGood, ...]:
    """Enumerate an exact finite no-good representation of hardware-aware QI.

    For each strict inequality

    ``D[i,r] + P[r,l] + D[l,j] < D[i,j]``,

    the returned conjunction contains the distinct required hardware binaries
    and the distinct optional one-hot service states.  Mandatory and forbidden
    service states are structural constants and are omitted.  If the service
    states for repeated site pairs conflict, that conjunction is impossible
    and is not emitted.
    """

    if not isinstance(catalog, ServiceCatalog):
        raise TypeError("catalog must be a ServiceCatalog")
    actuators = _site_vector(
        "actuator_sites", actuator_sites, site_count=catalog.site_count
    )
    sensors = _site_vector(
        "sensor_sites", sensor_sites, site_count=catalog.site_count
    )
    plant = _plant_delays(
        plant_propagation_delays,
        measurement_count=len(sensors),
        actuator_count=len(actuators),
    )

    def menu_for(pair: tuple[int, int]):
        return catalog.menu_for(destination=pair[0], source=pair[1])

    def states(pair: tuple[int, int], *, finite_only: bool) -> tuple[Delay, ...]:
        menu = menu_for(pair)
        if menu is None:
            return () if finite_only else (None,)
        available: tuple[Delay, ...]
        if menu.mandatory_delay is not None:
            available = (menu.mandatory_delay,)
        else:
            available = (*menu.finite_delays, None)
        return (
            tuple(delay for delay in available if delay is not None)
            if finite_only
            else available
        )

    def optional(pair: tuple[int, int]) -> bool:
        menu = menu_for(pair)
        return menu is not None and menu.mandatory_delay is None

    no_goods: list[QINoGood] = []
    for target_actuator in range(len(actuators)):
        for target_sensor in range(len(sensors)):
            target_pair = (
                actuators[target_actuator],
                sensors[target_sensor],
            )
            for intermediate_sensor in range(len(sensors)):
                first_pair = (
                    actuators[target_actuator],
                    sensors[intermediate_sensor],
                )
                for intermediate_actuator in range(len(actuators)):
                    propagation = plant[intermediate_sensor][intermediate_actuator]
                    if propagation is None:
                        continue
                    second_pair = (
                        actuators[intermediate_actuator],
                        sensors[target_sensor],
                    )
                    selections = itertools.product(
                        states(first_pair, finite_only=True),
                        states(second_pair, finite_only=True),
                        states(target_pair, finite_only=False),
                    )
                    for first, second, target in selections:
                        indirect = first + propagation + second
                        if target is not None and indirect >= target:
                            continue
                        assignments: dict[tuple[int, int], Delay] = {}
                        consistent = True
                        for pair, delay in (
                            (first_pair, first),
                            (second_pair, second),
                            (target_pair, target),
                        ):
                            if pair in assignments and assignments[pair] != delay:
                                consistent = False
                                break
                            assignments[pair] = delay
                        if not consistent:
                            continue
                        service_states = tuple(
                            sorted(
                                (
                                    ServiceStateLiteral(
                                        destination_site=pair[0],
                                        source_site=pair[1],
                                        delay=delay,
                                    )
                                    for pair, delay in assignments.items()
                                    if optional(pair)
                                ),
                                key=lambda state: (
                                    state.destination_site,
                                    state.source_site,
                                    state.delay is None,
                                    -1 if state.delay is None else state.delay,
                                ),
                            )
                        )
                        no_goods.append(
                            QINoGood(
                                actuator_devices=tuple(
                                    sorted(
                                        {target_actuator, intermediate_actuator}
                                    )
                                ),
                                sensor_devices=tuple(
                                    sorted({target_sensor, intermediate_sensor})
                                ),
                                service_states=service_states,
                            )
                        )
    return _minimal_no_goods(no_goods)


def satisfies_qi_no_goods(
    no_goods: Iterable[QINoGood],
    eta: object,
    xi: object,
    service_delays: object,
) -> bool:
    """Return whether a deployment satisfies every generated QI no-good."""

    return not any(
        item.is_triggered(eta, xi, service_delays) for item in no_goods
    )


def add_qi_no_good_constraints(
    built_model: object,
    no_goods: Iterable[QINoGood],
    *,
    name_prefix: str = "hardware_aware_qi",
) -> int:
    """Add exact QI inequalities to a built B+ MICP without name lookup.

    ``built_model`` is expected to expose ``model``, ``variables.eta``,
    ``variables.xi``, and the nested one-hot ``selectors`` mapping.  These are
    structural fields of the native B+ builder; Gurobi variable-name strings
    are never inspected.
    """

    if not isinstance(name_prefix, str) or not name_prefix:
        raise ValueError("name_prefix must be a nonempty string")
    try:
        model = built_model.model  # type: ignore[attr-defined]
        variables = built_model.variables  # type: ignore[attr-defined]
        selectors = built_model.selectors  # type: ignore[attr-defined]
        eta = variables.eta
        xi = variables.xi
    except AttributeError as error:
        raise TypeError(
            "built_model must expose model, variables.eta/xi, and selectors"
        ) from error
    normalized = tuple(no_goods)
    if any(not isinstance(item, QINoGood) for item in normalized):
        raise TypeError("no_goods must contain QINoGood values")
    for index, no_good in enumerate(normalized):
        literals = [eta[item] for item in no_good.actuator_devices]
        literals.extend(xi[item] for item in no_good.sensor_devices)
        for state in no_good.service_states:
            pair = (state.destination_site, state.source_site)
            try:
                literals.append(selectors[pair][state.delay])
            except KeyError as error:
                raise ValueError(
                    "QI no-good refers to a service state absent from the built model: "
                    f"{pair} at delay {state.delay}"
                ) from error
        if len(literals) != no_good.literal_count or not literals:
            raise RuntimeError("QI no-good literal accounting is inconsistent")
        expression = sum(literals[1:], literals[0])
        model.addConstr(
            expression <= len(literals) - 1,
            name=f"{name_prefix}_{index}",
        )
    return len(normalized)


@dataclass(frozen=True, slots=True)
class HardwareAwareQISolveResult:
    """Native B+ solve plus direct finite-catalog QI verification."""

    solve: BPlusSolveResult
    no_good_count: int
    verified_qi: bool | None

    @property
    def certified_optimal(self) -> bool:
        return (
            self.solve.status == "OPTIMAL"
            and self.solve.solution_count > 0
            and self.solve.mip_gap == 0.0
            and self.verified_qi is True
        )


def solve_hardware_aware_qi(
    problem: BPlusProblem,
    plant_propagation_delays: object,
    options: SolverOptions | None = None,
) -> HardwareAwareQISolveResult:
    """Solve one exact QI-restricted finite-catalog B+ MICP.

    The returned optimum is certified over the prescribed FIR horizon,
    deployment class, and finite service catalog when ``certified_optimal`` is
    true.  This function does not turn that finite certificate into a global
    statement about arbitrary controllers or unrestricted network designs.
    """

    from .model import (
        BPlusProblem,
        SolverOptions,
        _build_model,
        _solve_built_model,
    )

    if not isinstance(problem, BPlusProblem):
        raise TypeError("problem must be a BPlusProblem")
    solver_options = SolverOptions() if options is None else options
    if not isinstance(solver_options, SolverOptions):
        raise TypeError("options must be SolverOptions")
    layout = problem.deployment.layout
    no_goods = finite_catalog_qi_no_goods(
        problem.deployment.services,
        layout.actuator_sites,
        layout.sensor_device_sites,
        plant_propagation_delays,
    )
    built = _build_model(problem, solver_options)
    count = add_qi_no_good_constraints(built, no_goods)
    solve = _solve_built_model(problem, solver_options, built)

    verified: bool | None = None
    if solve.solution_count > 0:
        if solve.eta is None or solve.xi is None or solve.service_delays is None:
            raise RuntimeError("solution-bearing QI result has incomplete architecture data")
        controller = hosted_controller_delays(
            solve.service_delays,
            layout.actuator_sites,
            layout.sensor_device_sites,
        )
        verified = hardware_aware_qi_compatible(
            controller,
            plant_propagation_delays,
            solve.eta,
            solve.xi,
        )
        if not verified or not satisfies_qi_no_goods(
            no_goods, solve.eta, solve.xi, solve.service_delays
        ):
            raise RuntimeError("decoded QI solve violates its exact no-good system")
    return HardwareAwareQISolveResult(
        solve=solve,
        no_good_count=count,
        verified_qi=verified,
    )


@dataclass(frozen=True, slots=True)
class QIBudgetSolvePoint:
    """One combined-budget point on the certified QI envelope."""

    budget: float
    result: HardwareAwareQISolveResult


def solve_hardware_aware_qi_budget_envelope(
    problem: BPlusProblem,
    plant_propagation_delays: object,
    budgets: Iterable[object],
    *,
    base_options: SolverOptions | None = None,
) -> tuple[QIBudgetSolvePoint, ...]:
    """Solve an increasing sequence of combined architecture budgets exactly."""

    from .model import SolverOptions

    options = SolverOptions() if base_options is None else base_options
    if not isinstance(options, SolverOptions):
        raise TypeError("base_options must be SolverOptions")
    if options.architecture_weight != 0.0:
        raise ValueError("QI budget envelopes require architecture_weight=0")
    if options.hardware_budget is not None or options.service_budget is not None:
        raise ValueError("combined QI budget envelopes cannot use separate budget caps")
    try:
        raw_budgets = tuple(budgets)
    except TypeError as error:
        raise TypeError("budgets must be an iterable") from error
    normalized: list[float] = []
    for index, value in enumerate(raw_budgets):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(f"budgets[{index}] must be a non-boolean real number")
        numeric = float(value)
        if not np.isfinite(numeric) or numeric < 0.0:
            raise ValueError(f"budgets[{index}] must be finite and nonnegative")
        normalized.append(numeric)
    if not normalized:
        raise ValueError("budgets must be nonempty")
    if any(right <= left for left, right in zip(normalized, normalized[1:])):
        raise ValueError("budgets must be strictly increasing")

    return tuple(
        QIBudgetSolvePoint(
            budget=budget,
            result=solve_hardware_aware_qi(
                problem,
                plant_propagation_delays,
                replace(options, architecture_budget=budget),
            ),
        )
        for budget in normalized
    )
