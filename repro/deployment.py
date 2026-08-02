"""Immutable data model for the prescribed fixed-host B+ deployment class.

Every service index is oriented as ``destination r <- source s``.  A service
choice is stored in ``service_delays[r][s]``; ``None`` is the canonical
representation of an unselected/forbidden infinite-delay service.  Device
activation and service provisioning are deliberately separate decisions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray: TypeAlias = NDArray[np.float64]
ServiceDelay: TypeAlias = int | None
ServiceDelayMatrix: TypeAlias = tuple[tuple[ServiceDelay, ...], ...]


def _integer(name: str, value: object, *, positive: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        qualifier = "positive " if positive else "nonnegative "
        raise TypeError(f"{name} must be a {qualifier}integer")
    normalized = int(value)
    lower_bound = 1 if positive else 0
    if normalized < lower_bound:
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {qualifier}")
    return normalized


def _index_tuple(name: str, values: object, site_count: int) -> tuple[int, ...]:
    try:
        candidates = tuple(cast(object, values))  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable of site indices") from error
    normalized: list[int] = []
    for position, value in enumerate(candidates):
        index = _integer(f"{name}[{position}]", value)
        if index >= site_count:
            raise ValueError(
                f"{name}[{position}]={index} is outside site_count={site_count}"
            )
        normalized.append(index)
    return tuple(normalized)


def _immutable_nonnegative_real_array(
    name: str, values: ArrayLike, *, dimensions: int
) -> FloatArray:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite nonnegative real values") from error
    if array.ndim != dimensions:
        dimension_name = "one-dimensional" if dimensions == 1 else "two-dimensional"
        raise ValueError(f"{name} must be {dimension_name}")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise ValueError(f"{name} must contain finite nonnegative real values")
    normalized = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(normalized)) or np.any(normalized < 0.0):
        raise ValueError(f"{name} must contain finite nonnegative real values")

    immutable_buffer = normalized.tobytes(order="C")
    return np.frombuffer(immutable_buffer, dtype=np.float64).reshape(normalized.shape)


def _binary_tuple(name: str, values: object) -> tuple[int, ...]:
    try:
        candidates = tuple(cast(object, values))  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable of binary values") from error

    normalized: list[int] = []
    for position, value in enumerate(candidates):
        if isinstance(value, (bool, np.bool_)):
            normalized.append(int(value))
            continue
        if not isinstance(value, Integral):
            raise TypeError(
                f"{name}[{position}] must be bool or an integer equal to 0 or 1"
            )
        binary = int(value)
        if binary not in (0, 1):
            raise ValueError(f"{name}[{position}] must equal 0 or 1")
        normalized.append(binary)
    return tuple(normalized)


def _service_delay_matrix(values: object) -> ServiceDelayMatrix:
    try:
        rows = tuple(tuple(row) for row in cast(object, values))  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError("service_delays must be a two-dimensional iterable") from error

    normalized_rows: list[tuple[ServiceDelay, ...]] = []
    for destination, row in enumerate(rows):
        normalized_row: list[ServiceDelay] = []
        for source, delay in enumerate(row):
            if delay is None:
                normalized_row.append(None)
                continue
            normalized_row.append(
                _integer(f"service_delays[{destination}][{source}]", delay)
            )
        normalized_rows.append(tuple(normalized_row))
    return tuple(normalized_rows)


@dataclass(frozen=True, slots=True)
class FixedHostLayout:
    """Fixed output, device, and realization-state hosting maps.

    ``sensor_sites`` is indexed by scalar measured-output coordinates.  The
    optional ``sensor_groups`` partitions those coordinates into physical
    sensing devices.  Device binaries and sensor costs are indexed by groups,
    while service routing continues to use the host of each scalar output.
    Omitting ``sensor_groups`` preserves the legacy one-device-per-output
    convention.
    """

    actuator_sites: tuple[int, ...]
    sensor_sites: tuple[int, ...]
    beta_sites: tuple[int, ...]
    state_block_sizes: tuple[int, ...]
    site_count: int
    sensor_groups: tuple[tuple[int, ...], ...] | None = None

    def __post_init__(self) -> None:
        site_count = _integer("site_count", self.site_count, positive=True)
        object.__setattr__(self, "site_count", site_count)
        object.__setattr__(
            self,
            "actuator_sites",
            _index_tuple("actuator_sites", self.actuator_sites, site_count),
        )
        sensor_sites = _index_tuple("sensor_sites", self.sensor_sites, site_count)
        object.__setattr__(
            self,
            "sensor_sites",
            sensor_sites,
        )
        beta_sites = _index_tuple("beta_sites", self.beta_sites, site_count)
        object.__setattr__(self, "beta_sites", beta_sites)

        try:
            block_sizes_raw = tuple(self.state_block_sizes)
        except TypeError as error:
            raise TypeError("state_block_sizes must be an iterable") from error
        block_sizes = tuple(
            _integer(f"state_block_sizes[{index}]", size, positive=True)
            for index, size in enumerate(block_sizes_raw)
        )
        if len(beta_sites) != len(block_sizes):
            raise ValueError(
                "beta_sites and state_block_sizes must have the same length"
            )
        if not beta_sites:
            raise ValueError("at least one hosted beta block is required")
        object.__setattr__(self, "state_block_sizes", block_sizes)

        scalar_sensor_count = len(sensor_sites)
        if self.sensor_groups is None:
            sensor_groups = tuple((index,) for index in range(scalar_sensor_count))
        else:
            try:
                raw_groups = tuple(tuple(group) for group in self.sensor_groups)
            except TypeError as error:
                raise TypeError(
                    "sensor_groups must be a two-dimensional iterable of output indices"
                ) from error
            if not raw_groups:
                raise ValueError("sensor_groups must contain at least one group")
            normalized_groups: list[tuple[int, ...]] = []
            for group_index, group in enumerate(raw_groups):
                if not group:
                    raise ValueError(f"sensor_groups[{group_index}] must not be empty")
                normalized_group: list[int] = []
                for member_index, member in enumerate(group):
                    output_index = _integer(
                        f"sensor_groups[{group_index}][{member_index}]", member
                    )
                    if output_index >= scalar_sensor_count:
                        raise ValueError(
                            "sensor group output index is outside sensor_sites: "
                            f"sensor_groups[{group_index}][{member_index}]={output_index}"
                        )
                    normalized_group.append(output_index)
                if len(set(normalized_group)) != len(normalized_group):
                    raise ValueError(
                        f"sensor_groups[{group_index}] contains duplicate output indices"
                    )
                group_sites = {sensor_sites[index] for index in normalized_group}
                if len(group_sites) != 1:
                    raise ValueError(
                        f"sensor_groups[{group_index}] must be hosted at one site"
                    )
                normalized_groups.append(tuple(normalized_group))
            sensor_groups = tuple(normalized_groups)

        flattened = tuple(index for group in sensor_groups for index in group)
        expected_outputs = tuple(range(scalar_sensor_count))
        if len(flattened) != scalar_sensor_count or tuple(sorted(flattened)) != expected_outputs:
            raise ValueError(
                "sensor_groups must cover every sensor output exactly once"
            )
        object.__setattr__(self, "sensor_groups", sensor_groups)

    @property
    def beta_dimension(self) -> int:
        return sum(self.state_block_sizes)

    @property
    def sensor_device_count(self) -> int:
        if self.sensor_groups is None:  # Defensive; normalized in __post_init__.
            raise RuntimeError("sensor_groups were not normalized")
        return len(self.sensor_groups)

    @property
    def sensor_device_sites(self) -> tuple[int, ...]:
        if self.sensor_groups is None:  # Defensive; normalized in __post_init__.
            raise RuntimeError("sensor_groups were not normalized")
        return tuple(self.sensor_sites[group[0]] for group in self.sensor_groups)

    def sensor_device_for_output(self, sensor: int) -> int:
        output_index = _bounded_component_index(
            "sensor", sensor, len(self.sensor_sites)
        )
        if self.sensor_groups is None:  # Defensive; normalized in __post_init__.
            raise RuntimeError("sensor_groups were not normalized")
        for device, group in enumerate(self.sensor_groups):
            if output_index in group:
                return device
        raise RuntimeError(f"sensor output {output_index} has no physical device group")

    def p_u(self, actuator: int) -> int:
        return self.actuator_sites[
            _bounded_component_index("actuator", actuator, len(self.actuator_sites))
        ]

    def p_y(self, sensor: int) -> int:
        return self.sensor_sites[
            _bounded_component_index("sensor", sensor, len(self.sensor_sites))
        ]

    def beta_host(self, beta_block: int) -> int:
        return self.beta_sites[
            _bounded_component_index("beta_block", beta_block, len(self.beta_sites))
        ]


def _bounded_component_index(name: str, value: object, count: int) -> int:
    index = _integer(name, value)
    if index >= count:
        raise IndexError(f"{name} index {index} is outside the valid range 0:{count}")
    return index


@dataclass(frozen=True, slots=True)
class DirectedServiceMenu:
    """Finite delay choices for one directed service ``destination <- source``."""

    destination_site: int
    source_site: int
    finite_delays: tuple[int, ...]
    mandatory_delay: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "destination_site",
            _integer("destination_site", self.destination_site),
        )
        object.__setattr__(
            self, "source_site", _integer("source_site", self.source_site)
        )
        try:
            delay_values = tuple(self.finite_delays)
        except TypeError as error:
            raise TypeError("finite_delays must be an iterable") from error
        delays = tuple(
            _integer(f"finite_delays[{index}]", delay)
            for index, delay in enumerate(delay_values)
        )
        if not delays:
            raise ValueError("finite_delays must contain at least one delay")
        if tuple(sorted(set(delays))) != delays:
            raise ValueError("finite_delays must be strictly increasing and unique")
        object.__setattr__(self, "finite_delays", delays)

        if self.mandatory_delay is not None:
            mandatory_delay = _integer("mandatory_delay", self.mandatory_delay)
            if mandatory_delay not in delays:
                raise ValueError("mandatory_delay must belong to finite_delays")
            object.__setattr__(self, "mandatory_delay", mandatory_delay)


@dataclass(frozen=True, slots=True)
class ServiceCatalog:
    """Finite directed-service menus with constant-time oriented-pair lookup.

    Omitted nonlocal pairs are forbidden.  Construction freezes a private
    tuple-of-tuples lookup grid so repeated ``destination <- source`` queries
    do not rescan ``directed_menus``.
    """

    site_count: int
    directed_menus: tuple[DirectedServiceMenu, ...]
    _menu_grid: tuple[tuple[DirectedServiceMenu | None, ...], ...] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        site_count = _integer("site_count", self.site_count, positive=True)
        object.__setattr__(self, "site_count", site_count)
        try:
            menus = tuple(self.directed_menus)
        except TypeError as error:
            raise TypeError("directed_menus must be an iterable") from error

        menu_grid: list[list[DirectedServiceMenu | None]] = [
            [None for _ in range(site_count)] for _ in range(site_count)
        ]
        for menu in menus:
            if not isinstance(menu, DirectedServiceMenu):
                raise TypeError("directed_menus must contain DirectedServiceMenu values")
            if menu.destination_site >= site_count or menu.source_site >= site_count:
                raise ValueError(
                    "directed service site index is outside the catalog site count"
                )
            pair = (menu.destination_site, menu.source_site)
            if menu_grid[pair[0]][pair[1]] is not None:
                raise ValueError(f"duplicate directed service menu for pair {pair}")
            menu_grid[pair[0]][pair[1]] = menu

        immutable_menu_grid = tuple(tuple(row) for row in menu_grid)
        for site in range(site_count):
            local = immutable_menu_grid[site][site]
            if (
                local is None
                or local.finite_delays != (0,)
                or local.mandatory_delay != 0
            ):
                raise ValueError(
                    f"local service ({site}, {site}) must be mandatory at delay zero"
                )

        object.__setattr__(
            self,
            "directed_menus",
            tuple(
                sorted(
                    menus,
                    key=lambda menu: (menu.destination_site, menu.source_site),
                )
            ),
        )
        object.__setattr__(self, "_menu_grid", immutable_menu_grid)

    def _validated_pair(self, destination: object, source: object) -> tuple[int, int]:
        destination_index = _integer("destination", destination)
        source_index = _integer("source", source)
        if destination_index >= self.site_count or source_index >= self.site_count:
            raise IndexError("service pair site index is outside the catalog")
        return destination_index, source_index

    def menu_for(
        self, *, destination: int, source: int
    ) -> DirectedServiceMenu | None:
        destination_index, source_index = self._validated_pair(
            destination, source
        )
        return self._menu_grid[destination_index][source_index]

    def finite_delays(self, *, destination: int, source: int) -> tuple[int, ...]:
        menu = self.menu_for(destination=destination, source=source)
        return () if menu is None else menu.finite_delays

    def is_forbidden(self, *, destination: int, source: int) -> bool:
        return self.menu_for(destination=destination, source=source) is None


@dataclass(frozen=True, slots=True, eq=False)
class DeploymentCostSpec:
    """Independent device costs and per-delay directed-service cost layers."""

    actuator_costs: FloatArray
    sensor_costs: FloatArray
    service_costs_by_delay: tuple[FloatArray, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "actuator_costs",
            _immutable_nonnegative_real_array(
                "actuator_costs", self.actuator_costs, dimensions=1
            ),
        )
        object.__setattr__(
            self,
            "sensor_costs",
            _immutable_nonnegative_real_array(
                "sensor_costs", self.sensor_costs, dimensions=1
            ),
        )
        try:
            cost_layers_raw = tuple(self.service_costs_by_delay)
        except TypeError as error:
            raise TypeError("service_costs_by_delay must be an iterable") from error
        if not cost_layers_raw:
            raise ValueError("service_costs_by_delay must contain the delay-zero layer")

        cost_layers = tuple(
            _immutable_nonnegative_real_array(
                f"service_costs_by_delay[{delay}]", values, dimensions=2
            )
            for delay, values in enumerate(cost_layers_raw)
        )
        first_shape = cost_layers[0].shape
        if len(first_shape) != 2 or first_shape[0] != first_shape[1]:
            raise ValueError("service cost layers must be square")
        if any(layer.shape != first_shape for layer in cost_layers[1:]):
            raise ValueError("service cost layers must all have the same square shape")
        object.__setattr__(self, "service_costs_by_delay", cost_layers)


@dataclass(frozen=True, slots=True)
class ArchitectureChoice:
    """Explicit hardware binaries and directed service-delay selections.

    Boolean binaries and integral 0/1 values are accepted, then normalized to
    plain Python integers.  ``service_delays[r][s]`` is either a finite
    nonnegative integer or ``None`` for infinite delay/unselected service.
    """

    eta: tuple[int, ...]
    xi: tuple[int, ...]
    service_delays: ServiceDelayMatrix

    def __post_init__(self) -> None:
        object.__setattr__(self, "eta", _binary_tuple("eta", self.eta))
        object.__setattr__(self, "xi", _binary_tuple("xi", self.xi))
        object.__setattr__(
            self, "service_delays", _service_delay_matrix(self.service_delays)
        )


@dataclass(frozen=True, slots=True)
class DeploymentSpec:
    """Cross-validated fixed hosts, service menus, and architecture costs."""

    layout: FixedHostLayout
    services: ServiceCatalog
    costs: DeploymentCostSpec

    def __post_init__(self) -> None:
        if not isinstance(self.layout, FixedHostLayout):
            raise TypeError("layout must be a FixedHostLayout")
        if not isinstance(self.services, ServiceCatalog):
            raise TypeError("services must be a ServiceCatalog")
        if not isinstance(self.costs, DeploymentCostSpec):
            raise TypeError("costs must be a DeploymentCostSpec")

        if self.layout.site_count != self.services.site_count:
            raise ValueError("layout and service catalog site counts must agree")
        if self.costs.actuator_costs.shape != (len(self.layout.actuator_sites),):
            raise ValueError("actuator cost count must match actuator_sites")
        if self.costs.sensor_costs.shape != (self.layout.sensor_device_count,):
            raise ValueError("sensor cost count must match sensor_device_count")
        expected_service_shape = (self.layout.site_count, self.layout.site_count)
        if self.costs.service_costs_by_delay[0].shape != expected_service_shape:
            raise ValueError("service cost layer shape must match the deployment site count")

        delay_layer_count = len(self.costs.service_costs_by_delay)
        for menu in self.services.directed_menus:
            for delay in menu.finite_delays:
                if delay >= delay_layer_count:
                    raise ValueError(
                        f"service menu delay {delay} has no corresponding cost layer"
                    )
        delay_zero_cost = self.costs.service_costs_by_delay[0]
        for site in range(self.layout.site_count):
            if delay_zero_cost[site, site] != 0.0:
                raise ValueError("mandatory local services must be free")

    def validate_choice(self, choice: ArchitectureChoice) -> None:
        if not isinstance(choice, ArchitectureChoice):
            raise TypeError("choice must be an ArchitectureChoice")
        if len(choice.eta) != len(self.layout.actuator_sites):
            raise ValueError("eta length must match actuator_sites")
        if len(choice.xi) != self.layout.sensor_device_count:
            raise ValueError("xi length must match sensor_device_count")

        site_count = self.layout.site_count
        if len(choice.service_delays) != site_count or any(
            len(row) != site_count for row in choice.service_delays
        ):
            raise ValueError(
                "service_delays must have shape (site_count, site_count)"
            )

        for destination in range(site_count):
            for source in range(site_count):
                selected_delay = choice.service_delays[destination][source]
                menu = self.services.menu_for(
                    destination=destination, source=source
                )
                if menu is None:
                    if selected_delay is not None:
                        raise ValueError(
                            "selected service is forbidden by the finite menu: "
                            f"destination {destination} <- source {source}"
                        )
                    continue
                if menu.mandatory_delay is not None:
                    if selected_delay != menu.mandatory_delay:
                        raise ValueError(
                            "mandatory service must select delay "
                            f"{menu.mandatory_delay}: destination {destination} "
                            f"<- source {source}"
                        )
                    continue
                if selected_delay is not None and selected_delay not in menu.finite_delays:
                    raise ValueError(
                        f"selected delay {selected_delay} is outside the finite menu "
                        f"for destination {destination} <- source {source}"
                    )


@dataclass(frozen=True, slots=True)
class ArchitectureCostBreakdown:
    total: float
    actuator: float
    sensor: float
    service: float


def _finite_cost_sum(component: str, terms: list[float]) -> float:
    try:
        total = math.fsum(terms)
    except OverflowError as error:
        raise OverflowError(
            f"{component} architecture cost overflow from finite selected terms"
        ) from error
    if not math.isfinite(total):
        raise OverflowError(
            f"{component} architecture cost overflow from finite selected terms"
        )
    return total


def architecture_cost(
    deployment: DeploymentSpec, choice: ArchitectureChoice
) -> ArchitectureCostBreakdown:
    """Compute cost only from explicit devices and selected logical services.

    Each selected directed service is charged exactly once, independent of its
    payload or of whether a sensor/actuator at either endpoint is active.
    """

    deployment.validate_choice(choice)
    actuator_terms = [
        float(cost)
        for cost, enabled in zip(deployment.costs.actuator_costs, choice.eta)
        if enabled == 1
    ]
    sensor_terms = [
        float(cost)
        for cost, enabled in zip(deployment.costs.sensor_costs, choice.xi)
        if enabled == 1
    ]
    service_terms = [
        float(deployment.costs.service_costs_by_delay[delay][destination, source])
        for destination, row in enumerate(choice.service_delays)
        for source, delay in enumerate(row)
        if delay is not None
    ]
    actuator_cost = _finite_cost_sum("actuator component", actuator_terms)
    sensor_cost = _finite_cost_sum("sensor component", sensor_terms)
    service_cost = _finite_cost_sum("service component", service_terms)
    total_cost = _finite_cost_sum(
        "total", [actuator_cost, sensor_cost, service_cost]
    )
    return ArchitectureCostBreakdown(
        total=total_cost,
        actuator=actuator_cost,
        sensor=sensor_cost,
        service=service_cost,
    )
