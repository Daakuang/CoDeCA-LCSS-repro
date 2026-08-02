from dataclasses import FrozenInstanceError, fields
import math

import numpy as np
import pytest

from repro.deployment import (
    ArchitectureChoice,
    DeploymentCostSpec,
    DeploymentSpec,
    DirectedServiceMenu,
    FixedHostLayout,
    ServiceCatalog,
    architecture_cost,
)


def _layout() -> FixedHostLayout:
    return FixedHostLayout(
        actuator_sites=(0, 1),
        sensor_sites=(0, 1),
        beta_sites=(0, 1),
        state_block_sizes=(2, 1),
        site_count=2,
    )


def _catalog() -> ServiceCatalog:
    return ServiceCatalog(
        site_count=2,
        directed_menus=(
            DirectedServiceMenu(0, 0, (0,), mandatory_delay=0),
            DirectedServiceMenu(1, 1, (0,), mandatory_delay=0),
            DirectedServiceMenu(1, 0, (1, 2)),
            DirectedServiceMenu(0, 1, (1,)),
        ),
    )


def _costs() -> DeploymentCostSpec:
    return DeploymentCostSpec(
        actuator_costs=np.array([5.0, 7.0]),
        sensor_costs=np.array([11.0, 13.0]),
        service_costs_by_delay=(
            np.zeros((2, 2)),
            np.array([[0.0, 17.0], [19.0, 0.0]]),
            np.array([[0.0, 0.0], [23.0, 0.0]]),
        ),
    )


def _deployment() -> DeploymentSpec:
    return DeploymentSpec(layout=_layout(), services=_catalog(), costs=_costs())


def _choice(
    *,
    eta: tuple[int, ...] = (1, 1),
    xi: tuple[int, ...] = (1, 1),
    service_delays: tuple[tuple[int | None, ...], ...] = ((0, 1), (1, 0)),
) -> ArchitectureChoice:
    return ArchitectureChoice(
        eta=eta,
        xi=xi,
        service_delays=service_delays,
    )


def test_fixed_layout_exposes_the_three_independent_host_maps() -> None:
    layout = _layout()

    assert layout.p_u(1) == 1
    assert layout.p_y(0) == 0
    assert layout.beta_host(0) == 0
    assert layout.beta_host(1) == 1
    assert layout.beta_dimension == 3
    assert layout.sensor_groups == ((0,), (1,))
    assert layout.sensor_device_count == 2
    assert layout.sensor_device_sites == (0, 1)
    assert layout.sensor_device_for_output(1) == 1


def test_fixed_layout_groups_colocated_outputs_into_physical_sensor_devices() -> None:
    layout = FixedHostLayout(
        actuator_sites=(0,),
        sensor_sites=(0, 1, 1),
        beta_sites=(0,),
        state_block_sizes=(1,),
        site_count=2,
        sensor_groups=((0,), (1, 2)),
    )

    assert layout.sensor_device_count == 2
    assert layout.sensor_device_sites == (0, 1)
    assert layout.sensor_device_for_output(0) == 0
    assert layout.sensor_device_for_output(1) == 1
    assert layout.sensor_device_for_output(2) == 1


@pytest.mark.parametrize(
    "sensor_groups",
    [
        (),
        ((0,),),
        ((0, 1), (1, 2)),
        ((0,), (1, 3)),
        ((0,), (1,), (2, 2)),
        ((0, 1), (2,)),
    ],
)
def test_fixed_layout_rejects_invalid_or_cross_site_sensor_groups(
    sensor_groups: tuple[tuple[int, ...], ...],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        FixedHostLayout(
            actuator_sites=(0,),
            sensor_sites=(0, 1, 1),
            beta_sites=(0,),
            state_block_sizes=(1,),
            site_count=2,
            sensor_groups=sensor_groups,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"site_count": 0},
        {"site_count": True},
        {"actuator_sites": (0, 2)},
        {"sensor_sites": (-1, 1)},
        {"beta_sites": (0, 2)},
        {"beta_sites": (0,), "state_block_sizes": (2, 1)},
        {"state_block_sizes": (2, 0)},
        {"state_block_sizes": (2, True)},
    ],
)
def test_fixed_layout_rejects_invalid_counts_indices_and_block_sizes(
    kwargs: dict[str, object],
) -> None:
    arguments: dict[str, object] = {
        "actuator_sites": (0, 1),
        "sensor_sites": (0, 1),
        "beta_sites": (0, 1),
        "state_block_sizes": (2, 1),
        "site_count": 2,
    }
    arguments.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        FixedHostLayout(**arguments)  # type: ignore[arg-type]


def test_service_catalog_is_oriented_destination_from_source_and_forbids_omissions() -> None:
    catalog = _catalog()

    assert catalog.finite_delays(destination=1, source=0) == (1, 2)
    assert catalog.finite_delays(destination=0, source=1) == (1,)
    assert catalog.finite_delays(destination=0, source=0) == (0,)
    assert catalog.finite_delays(destination=1, source=1) == (0,)
    assert catalog.is_forbidden(destination=0, source=0) is False

    forbidden_catalog = ServiceCatalog(
        site_count=2,
        directed_menus=(
            DirectedServiceMenu(0, 0, (0,), mandatory_delay=0),
            DirectedServiceMenu(1, 1, (0,), mandatory_delay=0),
        ),
    )
    assert forbidden_catalog.finite_delays(destination=1, source=0) == ()
    assert forbidden_catalog.is_forbidden(destination=1, source=0) is True


def test_service_catalog_builds_a_deeply_immutable_direct_lookup_grid() -> None:
    catalog = _catalog()
    grid_field = next(
        catalog_field
        for catalog_field in fields(ServiceCatalog)
        if catalog_field.name == "_menu_grid"
    )

    assert isinstance(catalog._menu_grid, tuple)
    assert all(isinstance(row, tuple) for row in catalog._menu_grid)
    assert len(catalog._menu_grid) == catalog.site_count
    assert all(len(row) == catalog.site_count for row in catalog._menu_grid)
    assert catalog._menu_grid[1][0] is catalog.menu_for(destination=1, source=0)
    assert catalog._menu_grid[0][1] is catalog.menu_for(destination=0, source=1)
    assert grid_field.init is False
    assert grid_field.repr is False
    assert grid_field.compare is False
    assert "_menu_grid" not in repr(catalog)
    with pytest.raises(TypeError):
        catalog._menu_grid[0][0] = None  # type: ignore[index]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"finite_delays": (1, 1)},
        {"finite_delays": (2, 1)},
        {"finite_delays": (-1,)},
        {"finite_delays": (True,)},
        {"finite_delays": (1,), "mandatory_delay": 2},
    ],
)
def test_directed_service_menu_rejects_invalid_delay_menus(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        DirectedServiceMenu(
            destination_site=0,
            source_site=1,
            **kwargs,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "directed_menus",
    [
        (
            DirectedServiceMenu(0, 0, (0,), mandatory_delay=0),
            DirectedServiceMenu(0, 0, (0,), mandatory_delay=0),
            DirectedServiceMenu(1, 1, (0,), mandatory_delay=0),
        ),
        (
            DirectedServiceMenu(0, 0, (0,), mandatory_delay=0),
            DirectedServiceMenu(1, 1, (0,), mandatory_delay=0),
            DirectedServiceMenu(2, 0, (1,)),
        ),
        (DirectedServiceMenu(0, 0, (0,), mandatory_delay=0),),
        (
            DirectedServiceMenu(0, 0, (0,), mandatory_delay=0),
            DirectedServiceMenu(1, 1, (0,)),
        ),
        (
            DirectedServiceMenu(0, 0, (1,), mandatory_delay=1),
            DirectedServiceMenu(1, 1, (0,), mandatory_delay=0),
        ),
    ],
)
def test_catalog_rejects_duplicate_invalid_or_nonmandatory_local_services(
    directed_menus: tuple[DirectedServiceMenu, ...],
) -> None:
    with pytest.raises(ValueError):
        ServiceCatalog(site_count=2, directed_menus=directed_menus)


@pytest.mark.parametrize(
    "field,bad_values",
    [
        ("actuator_costs", np.array([1.0, -1.0])),
        ("sensor_costs", np.array([1.0, np.nan])),
        ("sensor_costs", np.array([1.0, np.inf])),
        ("actuator_costs", np.array([1.0 + 1.0j, 2.0])),
        ("actuator_costs", np.zeros((1, 2))),
    ],
)
def test_cost_spec_requires_finite_nonnegative_real_vectors(
    field: str, bad_values: np.ndarray,
) -> None:
    arguments = {
        "actuator_costs": np.array([5.0, 7.0]),
        "sensor_costs": np.array([11.0, 13.0]),
        "service_costs_by_delay": (np.zeros((2, 2)),),
    }
    arguments[field] = bad_values

    with pytest.raises(ValueError, match="finite nonnegative real|one-dimensional"):
        DeploymentCostSpec(**arguments)


@pytest.mark.parametrize(
    "service_costs",
    [
        (),
        (np.zeros((2, 3)),),
        (np.zeros((2, 2)), np.zeros((3, 3))),
        (np.array([[0.0, -1.0], [0.0, 0.0]]),),
        (np.array([[0.0, np.inf], [0.0, 0.0]]),),
    ],
)
def test_cost_spec_requires_a_consistent_finite_square_delay_stack(
    service_costs: tuple[np.ndarray, ...],
) -> None:
    with pytest.raises(ValueError):
        DeploymentCostSpec(
            actuator_costs=np.ones(2),
            sensor_costs=np.ones(2),
            service_costs_by_delay=service_costs,
        )


def test_deployment_cross_validates_layout_menu_cost_shapes_and_free_local_services() -> None:
    with pytest.raises(ValueError, match="actuator"):
        DeploymentSpec(
            layout=_layout(),
            services=_catalog(),
            costs=DeploymentCostSpec(
                actuator_costs=np.ones(1),
                sensor_costs=np.ones(2),
                service_costs_by_delay=_costs().service_costs_by_delay,
            ),
        )
    with pytest.raises(ValueError, match="site count"):
        DeploymentSpec(
            layout=_layout(),
            services=ServiceCatalog(
                site_count=1,
                directed_menus=(
                    DirectedServiceMenu(0, 0, (0,), mandatory_delay=0),
                ),
            ),
            costs=DeploymentCostSpec(
                actuator_costs=np.ones(2),
                sensor_costs=np.ones(2),
                service_costs_by_delay=(np.zeros((1, 1)),),
            ),
        )
    nonfree_local = list(_costs().service_costs_by_delay)
    nonfree_local[0] = np.eye(2)
    with pytest.raises(ValueError, match="local.*free"):
        DeploymentSpec(
            layout=_layout(),
            services=_catalog(),
            costs=DeploymentCostSpec(
                actuator_costs=np.ones(2),
                sensor_costs=np.ones(2),
                service_costs_by_delay=tuple(nonfree_local),
            ),
        )


def test_deployment_rejects_menu_delays_without_a_cost_layer() -> None:
    with pytest.raises(ValueError, match="delay 2"):
        DeploymentSpec(
            layout=_layout(),
            services=_catalog(),
            costs=DeploymentCostSpec(
                actuator_costs=np.ones(2),
                sensor_costs=np.ones(2),
                service_costs_by_delay=(np.zeros((2, 2)), np.zeros((2, 2))),
            ),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"eta": (1,)},
        {"xi": (1,)},
        {"eta": (1, 2)},
        {"eta": (1, -1)},
        {"eta": (1, 1.0)},
        {"service_delays": ((0, 1),)},
        {"service_delays": ((0, 1), (1, None))},
        {"service_delays": ((0, 2), (1, 0))},
        {"service_delays": ((0, 1), (3, 0))},
    ],
)
def test_deployment_rejects_invalid_binary_shapes_delays_and_mandatory_services(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        arguments: dict[str, object] = {
            "eta": (1, 1),
            "xi": (1, 1),
            "service_delays": ((0, 1), (1, 0)),
        }
        arguments.update(kwargs)
        choice = ArchitectureChoice(**arguments)  # type: ignore[arg-type]
        _deployment().validate_choice(choice)


def test_deployment_rejects_a_selected_service_omitted_from_the_finite_menu() -> None:
    services = ServiceCatalog(
        site_count=2,
        directed_menus=(
            DirectedServiceMenu(0, 0, (0,), mandatory_delay=0),
            DirectedServiceMenu(1, 1, (0,), mandatory_delay=0),
        ),
    )
    deployment = DeploymentSpec(
        layout=_layout(),
        services=services,
        costs=DeploymentCostSpec(
            actuator_costs=np.ones(2),
            sensor_costs=np.ones(2),
            service_costs_by_delay=(np.zeros((2, 2)), np.ones((2, 2))),
        ),
    )

    with pytest.raises(ValueError, match="forbidden"):
        deployment.validate_choice(
            _choice(service_delays=((0, None), (1, 0)))
        )


def test_architecture_cost_uses_explicit_devices_and_charges_each_service_once() -> None:
    breakdown = architecture_cost(
        _deployment(),
        _choice(eta=(1, 0), xi=(0, 1), service_delays=((0, 1), (2, 0))),
    )

    assert breakdown.actuator == pytest.approx(5.0)
    assert breakdown.sensor == pytest.approx(13.0)
    assert breakdown.service == pytest.approx(17.0 + 23.0)
    assert breakdown.total == pytest.approx(58.0)


def test_architecture_cost_uses_stable_summation_across_large_and_small_terms() -> None:
    deployment = DeploymentSpec(
        layout=FixedHostLayout(
            actuator_sites=(0, 0, 1),
            sensor_sites=(0, 1),
            beta_sites=(0, 1),
            state_block_sizes=(2, 1),
            site_count=2,
        ),
        services=_catalog(),
        costs=DeploymentCostSpec(
            actuator_costs=np.array([1.0e16, 1.0, 1.0]),
            sensor_costs=np.array([1.0, 0.0]),
            service_costs_by_delay=(
                np.zeros((2, 2)),
                np.array([[0.0, 1.0], [0.0, 0.0]]),
                np.zeros((2, 2)),
            ),
        ),
    )

    breakdown = architecture_cost(
        deployment,
        ArchitectureChoice(
            eta=(1, 1, 1),
            xi=(1, 0),
            service_delays=((0, 1), (None, 0)),
        ),
    )

    expected_actuator = math.fsum([1.0e16, 1.0, 1.0])
    expected_total = math.fsum([expected_actuator, 1.0, 1.0])
    assert breakdown.actuator == expected_actuator
    assert breakdown.total == expected_total
    assert math.isfinite(breakdown.total)


def test_architecture_cost_rejects_component_overflow_from_finite_active_costs() -> None:
    deployment = DeploymentSpec(
        layout=_layout(),
        services=_catalog(),
        costs=DeploymentCostSpec(
            actuator_costs=np.array([1.0e308, 1.0e308]),
            sensor_costs=np.zeros(2),
            service_costs_by_delay=(
                np.zeros((2, 2)),
                np.zeros((2, 2)),
                np.zeros((2, 2)),
            ),
        ),
    )

    with pytest.raises(OverflowError, match="actuator.*overflow"):
        architecture_cost(
            deployment,
            _choice(
                eta=(1, 1),
                xi=(0, 0),
                service_delays=((0, None), (None, 0)),
            ),
        )


def test_architecture_cost_rejects_total_overflow_with_finite_components() -> None:
    deployment = DeploymentSpec(
        layout=_layout(),
        services=_catalog(),
        costs=DeploymentCostSpec(
            actuator_costs=np.array([1.0e308, 0.0]),
            sensor_costs=np.array([1.0e308, 0.0]),
            service_costs_by_delay=(
                np.zeros((2, 2)),
                np.zeros((2, 2)),
                np.zeros((2, 2)),
            ),
        ),
    )

    with pytest.raises(OverflowError, match="total.*overflow"):
        architecture_cost(
            deployment,
            _choice(
                eta=(1, 0),
                xi=(1, 0),
                service_delays=((0, None), (None, 0)),
            ),
        )


def test_inactive_sensor_does_not_disable_or_pay_for_beta_service_at_its_site() -> None:
    breakdown = architecture_cost(
        _deployment(),
        _choice(eta=(0, 0), xi=(0, 0), service_delays=((0, None), (1, 0))),
    )

    assert breakdown.sensor == pytest.approx(0.0)
    assert breakdown.actuator == pytest.approx(0.0)
    assert breakdown.service == pytest.approx(19.0)
    assert breakdown.total == pytest.approx(19.0)


def test_inactive_actuator_does_not_disable_or_pay_for_beta_service_at_its_site() -> None:
    breakdown = architecture_cost(
        _deployment(),
        _choice(eta=(0, 0), xi=(0, 0), service_delays=((0, 1), (None, 0))),
    )

    assert breakdown.actuator == pytest.approx(0.0)
    assert breakdown.sensor == pytest.approx(0.0)
    assert breakdown.service == pytest.approx(17.0)
    assert breakdown.total == pytest.approx(17.0)


def test_cost_arrays_and_all_frozen_containers_are_immutable() -> None:
    actuator_costs = np.array([5.0, 7.0])
    sensor_costs = np.array([11.0, 13.0])
    service_costs = np.zeros((2, 2))
    costs = DeploymentCostSpec(
        actuator_costs=actuator_costs,
        sensor_costs=sensor_costs,
        service_costs_by_delay=(service_costs,),
    )
    actuator_costs[0] = 99.0
    sensor_costs[0] = 99.0
    service_costs[0, 0] = 99.0

    assert costs.actuator_costs[0] == pytest.approx(5.0)
    assert costs.sensor_costs[0] == pytest.approx(11.0)
    assert costs.service_costs_by_delay[0][0, 0] == pytest.approx(0.0)
    for cost_array in (
        costs.actuator_costs,
        costs.sensor_costs,
        costs.service_costs_by_delay[0],
    ):
        with pytest.raises(ValueError, match="WRITEABLE|writable"):
            cost_array.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        costs.actuator_costs = np.ones(2)
    with pytest.raises(FrozenInstanceError):
        _layout().site_count = 3
    with pytest.raises(FrozenInstanceError):
        _choice().eta = (0, 0)


def test_choices_normalize_boolean_binaries_to_plain_integers() -> None:
    choice = ArchitectureChoice(
        eta=(True, np.bool_(False)),
        xi=(np.int64(1), np.int32(0)),
        service_delays=((0, None), (1, 0)),
    )

    assert choice.eta == (1, 0)
    assert choice.xi == (1, 0)
    assert all(type(value) is int for value in choice.eta + choice.xi)
