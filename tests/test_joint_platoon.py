import numpy as np
from numpy.testing import assert_allclose
import pytest

from repro.cases import (
    PLATOON_SENSOR_DEVICE_GROUPS,
    PLATOON_SENSOR_DEVICE_LABELS,
    PLATOON_SENSOR_DEVICE_SITES,
    make_joint_platoon_case,
    make_platoon_case,
    platoon_case_matrix_sha256,
)
from repro.deployment import (
    ArchitectureChoice,
    DeploymentCostSpec,
    DeploymentSpec,
    DirectedServiceMenu,
    FixedHostLayout,
    ServiceCatalog,
    architecture_cost,
)
from repro.diagnostics import diagnose_hardware_gating
from repro.joint_platoon_experiment import joint_solver_options, make_joint_problem
from repro.model import (
    BPlusProblem,
    GeneralizedPlant,
    ResponseFix,
    SolverOptions,
    solve_bplus,
)
from repro.realization import FIRResponses


DENSE_SERVICES = (
    (0, None, None, None),
    (1, 0, 1, 2),
    (1, 1, 0, 1),
    (1, 2, 1, 0),
)


def test_joint_platoon_preserves_plant_and_groups_seven_physical_sensors() -> None:
    base = make_platoon_case()
    joint = make_joint_platoon_case(seed=23)

    assert platoon_case_matrix_sha256(joint) == platoon_case_matrix_sha256(base)
    assert_allclose(joint.plant.A, base.plant.A, atol=0.0)
    assert_allclose(joint.plant.B1, base.plant.B1, atol=0.0)
    assert_allclose(joint.plant.B2, base.plant.B2, atol=0.0)
    assert_allclose(joint.plant.C2, base.plant.C2, atol=0.0)
    assert joint.sensor_device_groups == PLATOON_SENSOR_DEVICE_GROUPS
    assert joint.sensor_device_labels == PLATOON_SENSOR_DEVICE_LABELS
    assert joint.deployment.layout.sensor_device_sites == PLATOON_SENSOR_DEVICE_SITES
    assert joint.deployment.layout.sensor_device_count == 7
    assert joint.fixed_eta is None
    assert joint.fixed_xi is None
    assert base.fixed_eta == (1, 1, 1)
    assert base.fixed_xi == (1,) * 10


def test_joint_platoon_uses_transparent_normalized_hardware_and_service_units() -> None:
    joint = make_joint_platoon_case(seed=23)
    costs = joint.deployment.costs

    assert_allclose(costs.actuator_costs, np.ones(3), atol=0.0)
    assert_allclose(costs.sensor_costs, np.ones(7), atol=0.0)
    assert costs.service_costs_by_delay[1][1, 0] == pytest.approx(3.0)
    assert costs.service_costs_by_delay[2][1, 0] == pytest.approx(2.0)
    assert costs.service_costs_by_delay[0][1, 1] == pytest.approx(0.0)

    breakdown = architecture_cost(
        joint.deployment,
        ArchitectureChoice(
            eta=(1, 1, 1),
            xi=(1, 1, 1, 1, 1, 1, 1),
            service_delays=DENSE_SERVICES,
        ),
    )
    assert breakdown.actuator == pytest.approx(3.0)
    assert breakdown.sensor == pytest.approx(7.0)
    assert breakdown.service == pytest.approx(25.0)
    assert breakdown.total == pytest.approx(35.0)


def test_three_tier_sensitivity_extends_only_the_optional_service_menus() -> None:
    joint = make_joint_platoon_case(seed=23, service_horizon=3)
    catalog = joint.deployment.services

    for destination in (1, 2, 3):
        leader = catalog.menu_for(destination=destination, source=0)
        assert leader is not None
        assert leader.finite_delays == (1, 2, 3)
    for destination, source in ((1, 2), (2, 1), (2, 3), (3, 2)):
        adjacent = catalog.menu_for(destination=destination, source=source)
        assert adjacent is not None
        assert adjacent.finite_delays == (1, 2, 3)
    for destination, source in ((1, 3), (3, 1)):
        distance_two = catalog.menu_for(destination=destination, source=source)
        assert distance_two is not None
        assert distance_two.finite_delays == (2, 3)
    for site in range(4):
        local = catalog.menu_for(destination=site, source=site)
        assert local is not None
        assert local.finite_delays == (0,)
        assert local.mandatory_delay == 0

    costs = joint.deployment.costs.service_costs_by_delay
    assert len(costs) == 4
    assert costs[1][1, 0] == pytest.approx(3.0)
    assert costs[2][1, 0] == pytest.approx(2.0)
    assert costs[3][1, 0] == pytest.approx(1.0)
    assert 4**7 * 3**2 == 147456


def test_joint_problem_leaves_all_hardware_and_service_decisions_free() -> None:
    problem = make_joint_problem(make_joint_platoon_case(seed=23), horizon=10)

    assert problem.fixed_eta is None
    assert problem.fixed_xi is None
    assert problem.fixed_service_delays is None
    assert problem.plant.p == 10
    assert problem.deployment.layout.sensor_device_count == 7


def test_solver_options_support_combined_or_separate_caps_but_not_both() -> None:
    combined = joint_solver_options(architecture_budget=30.0)
    assert combined.architecture_budget == pytest.approx(30.0)
    assert combined.hardware_budget is None
    assert combined.service_budget is None

    separate = joint_solver_options(hardware_budget=8.0, service_budget=17.0)
    assert separate.architecture_budget is None
    assert separate.hardware_budget == pytest.approx(8.0)
    assert separate.service_budget == pytest.approx(17.0)

    with pytest.raises(ValueError, match="either a combined"):
        joint_solver_options(
            architecture_budget=30.0,
            hardware_budget=8.0,
            service_budget=17.0,
        )
    with pytest.raises(ValueError, match="at least one"):
        joint_solver_options()


def test_grouped_hardware_diagnostic_gates_every_scalar_output_in_the_group() -> None:
    R = np.zeros((3, 1, 1))
    R[1, 0, 0] = 1.0
    M = np.zeros((3, 1, 1))
    N = np.zeros((3, 1, 10))
    L = np.zeros((3, 1, 10))
    N[1, 0, 1] = 2.0
    N[1, 0, 2] = 3.0
    responses = FIRResponses(R=R, M=M, N=N, L=L)
    xi = (1, 0, 1, 1, 1, 1, 1)

    report = diagnose_hardware_gating(
        responses,
        eta=(1,),
        xi=xi,
        sensor_groups=PLATOON_SENSOR_DEVICE_GROUPS,
        tolerance=1.0e-9,
    )

    assert report.violation_count == 2
    assert report.max_abs == pytest.approx(3.0)


def _grouped_gate_problem(*, output: int) -> BPlusProblem:
    deployment = DeploymentSpec(
        layout=FixedHostLayout(
            actuator_sites=(0,),
            sensor_sites=(0, 0),
            beta_sites=(0,),
            state_block_sizes=(1,),
            site_count=1,
            sensor_groups=((0, 1),),
        ),
        services=ServiceCatalog(
            site_count=1,
            directed_menus=(
                DirectedServiceMenu(0, 0, (0,), mandatory_delay=0),
            ),
        ),
        costs=DeploymentCostSpec(
            actuator_costs=np.ones(1),
            sensor_costs=np.ones(1),
            service_costs_by_delay=(np.zeros((1, 1)),),
        ),
    )
    plant = GeneralizedPlant(
        A=np.zeros((1, 1)),
        B2=np.ones((1, 1)),
        C2=np.ones((2, 1)),
        B1=np.ones((1, 1)),
        C1=np.zeros((1, 1)),
        D12=np.zeros((1, 1)),
        D21=np.zeros((2, 1)),
        D11=np.zeros((1, 1)),
    )
    return BPlusProblem(
        plant=plant,
        deployment=deployment,
        horizon=2,
        fixed_eta=(1,),
        fixed_xi=(0,),
        fixed_service_delays=((0,),),
        response_fixes=(ResponseFix("L", lag=0, row=0, column=output, value=1.0),),
    )


@pytest.mark.parametrize("output", [0, 1])
def test_grouped_sensor_gate_is_enforced_on_each_model_column(output: int) -> None:
    pytest.importorskip("gurobipy")
    result = solve_bplus(_grouped_gate_problem(output=output), SolverOptions())

    assert result.status in {"INFEASIBLE", "INF_OR_UNBD"}
    assert result.solution_count == 0
