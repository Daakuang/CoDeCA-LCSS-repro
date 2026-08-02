from dataclasses import FrozenInstanceError, replace

import numpy as np
from numpy.testing import assert_allclose
import pytest

gp = pytest.importorskip("gurobipy")

from repro.deployment import (
    ArchitectureCostBreakdown,
    DeploymentCostSpec,
    DeploymentSpec,
    DirectedServiceMenu,
    FixedHostLayout,
    ServiceCatalog,
)
from repro.diagnostics import audit_solution
from repro.model import (
    BPlusProblem,
    BPlusSolveResult,
    GeneralizedPlant,
    SolverOptions,
    _build_model,
    solve_bplus,
)
from repro.realization import FIRResponses


def _zero_delay_deployment() -> DeploymentSpec:
    menus = tuple(
        DirectedServiceMenu(destination, source, (0,), mandatory_delay=0)
        for destination in range(2)
        for source in range(2)
    )
    return DeploymentSpec(
        layout=FixedHostLayout(
            actuator_sites=(0, 1),
            sensor_sites=(0, 1),
            beta_sites=(0, 1),
            state_block_sizes=(1, 1),
            site_count=2,
        ),
        services=ServiceCatalog(site_count=2, directed_menus=menus),
        costs=DeploymentCostSpec(
            actuator_costs=np.array([1.0, 2.0]),
            sensor_costs=np.array([3.0, 4.0]),
            service_costs_by_delay=(np.zeros((2, 2)),),
        ),
    )


def _tiny_plant() -> GeneralizedPlant:
    return GeneralizedPlant(
        A=np.diag([0.2, 0.35]),
        B2=np.eye(2),
        C2=np.eye(2),
        B1=np.eye(2),
        C1=np.vstack((np.eye(2), np.zeros((2, 2)))),
        D12=np.vstack((np.zeros((2, 2)), np.sqrt(0.2) * np.eye(2))),
        D21=0.1 * np.eye(2),
        D11=np.array(
            [[0.10, 0.00], [0.00, -0.05], [0.02, 0.00], [0.00, 0.03]]
        ),
    )


def _independent_fixed_fir_qp(plant: GeneralizedPlant, horizon: int) -> float:
    """Hand-build the dense FIR QP without calling the revision model."""

    model = gp.Model("independent_tiny_fixed_fir_qp")
    model.Params.OutputFlag = 0
    model.Params.Seed = 23
    model.Params.Threads = 1
    model.Params.FeasibilityTol = 1.0e-8
    model.Params.OptimalityTol = 1.0e-8
    infinity = gp.GRB.INFINITY
    R = tuple(
        model.addMVar((2, 2), lb=-infinity, name=f"ind_R_{t}")
        for t in range(horizon + 1)
    )
    M = tuple(
        model.addMVar((2, 2), lb=-infinity, name=f"ind_M_{t}")
        for t in range(horizon + 1)
    )
    N = tuple(
        model.addMVar((2, 2), lb=-infinity, name=f"ind_N_{t}")
        for t in range(horizon + 1)
    )
    L = tuple(
        model.addMVar((2, 2), lb=-infinity, name=f"ind_L_{t}")
        for t in range(horizon + 1)
    )

    for block in (R[0], M[0], N[0]):
        model.addConstr(block == np.zeros((2, 2)))
    zero = np.zeros((2, 2))
    for t in range(horizon + 1):
        R_next = R[t + 1] if t < horizon else zero
        M_next = M[t + 1] if t < horizon else zero
        N_next = N[t + 1] if t < horizon else zero
        impulse = np.eye(2) if t == 0 else zero
        model.addConstr(R_next - plant.A @ R[t] - plant.B2 @ M[t] == impulse)
        model.addConstr(N_next - plant.A @ N[t] - plant.B2 @ L[t] == zero)
        model.addConstr(R_next - R[t] @ plant.A - N[t] @ plant.C2 == impulse)
        model.addConstr(M_next - M[t] @ plant.A - L[t] @ plant.C2 == zero)

    objective = gp.QuadExpr()
    for t in range(horizon + 1):
        direct = plant.D11 if t == 0 else np.zeros_like(plant.D11)
        x_response = R[t] @ plant.B1 + N[t] @ plant.D21
        u_response = M[t] @ plant.B1 + L[t] @ plant.D21
        impulse = direct + plant.C1 @ x_response + plant.D12 @ u_response
        for row in range(impulse.shape[0]):
            for column in range(impulse.shape[1]):
                objective += impulse[row, column] * impulse[row, column]
    model.setObjective(objective, gp.GRB.MINIMIZE)
    model.optimize()

    assert model.Status == gp.GRB.OPTIMAL
    return float(model.ObjVal)


def test_native_model_matches_an_independently_built_fixed_fir_qp() -> None:
    plant = _tiny_plant()
    problem = BPlusProblem(
        plant=plant,
        deployment=_zero_delay_deployment(),
        horizon=2,
        fixed_eta=(1, 1),
        fixed_xi=(1, 1),
        fixed_service_delays=((0, 0), (0, 0)),
    )

    expected = _independent_fixed_fir_qp(plant, horizon=2)
    result = solve_bplus(problem, SolverOptions())

    assert result.status == "OPTIMAL"
    assert result.responses is not None
    assert result.performance_objective == pytest.approx(expected, abs=1.0e-9)
    assert result.architecture_cost == pytest.approx(10.0)
    assert result.eta == (1, 1)
    assert result.xi == (1, 1)
    assert result.service_delays == ((0, 0), (0, 0))
    assert result.settings.seed == 23
    assert result.settings.threads == 1

    report = audit_solution(problem, result, tolerance=1.0e-7)
    assert report.ofsls.max_residual <= 1.0e-7
    assert report.objective.absolute_error <= 1.0e-9
    assert report.hardware.max_abs <= 1.0e-9
    assert report.service.max_abs <= 1.0e-9
    assert report.architecture_cost.absolute_error <= 1.0e-12

    manual = 0.0
    response = result.responses
    for t in range(problem.horizon + 1):
        direct = plant.D11 if t == 0 else np.zeros_like(plant.D11)
        x_response = response.R[t] @ plant.B1 + response.N[t] @ plant.D21
        u_response = response.M[t] @ plant.B1 + response.L[t] @ plant.D21
        impulse = direct + plant.C1 @ x_response + plant.D12 @ u_response
        manual += float(np.sum(impulse**2))
    assert result.performance_objective == pytest.approx(manual, abs=1.0e-10)
    assert_allclose(response.R[1], np.eye(2), atol=1.0e-9)


def test_separate_hardware_and_service_caps_are_enforced_and_audited() -> None:
    base = _zero_delay_deployment()
    service_layers = (np.array([[0.0, 1.0], [1.0, 0.0]]),)
    deployment = DeploymentSpec(
        layout=base.layout,
        services=base.services,
        costs=DeploymentCostSpec(
            actuator_costs=base.costs.actuator_costs,
            sensor_costs=base.costs.sensor_costs,
            service_costs_by_delay=service_layers,
        ),
    )
    problem = BPlusProblem(
        plant=_tiny_plant(),
        deployment=deployment,
        horizon=2,
        fixed_eta=(1, 1),
        fixed_xi=(1, 1),
        fixed_service_delays=((0, 0), (0, 0)),
    )

    result = solve_bplus(
        problem,
        SolverOptions(hardware_budget=10.0, service_budget=2.0),
    )

    assert result.status == "OPTIMAL"
    assert result.architecture_breakdown is not None
    assert result.architecture_breakdown.actuator == pytest.approx(3.0)
    assert result.architecture_breakdown.sensor == pytest.approx(7.0)
    assert result.architecture_breakdown.service == pytest.approx(2.0)
    report = audit_solution(problem, result, tolerance=1.0e-7)
    assert report.budget_violation == pytest.approx(0.0)
    assert report.certified

    hardware_infeasible = solve_bplus(
        problem,
        SolverOptions(hardware_budget=9.0, service_budget=2.0),
    )
    service_infeasible = solve_bplus(
        problem,
        SolverOptions(hardware_budget=10.0, service_budget=1.0),
    )
    assert hardware_infeasible.solution_count == 0
    assert service_infeasible.solution_count == 0


def test_generalized_plant_owns_immutable_finite_real_matrices() -> None:
    A = np.diag([0.2, 0.35])
    plant = _tiny_plant()
    independent = GeneralizedPlant(
        A=A,
        B2=plant.B2,
        C2=plant.C2,
        B1=plant.B1,
        C1=plant.C1,
        D12=plant.D12,
        D21=plant.D21,
        D11=plant.D11,
        D22=np.zeros((2, 2)),
    )
    A[0, 0] = 99.0

    assert independent.A[0, 0] == pytest.approx(0.2)
    with pytest.raises(ValueError, match="WRITEABLE|writable"):
        independent.A.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        independent.A = np.eye(2)
    with pytest.raises(ValueError, match="D22.*zero"):
        GeneralizedPlant(
            A=plant.A,
            B2=plant.B2,
            C2=plant.C2,
            B1=plant.B1,
            C1=plant.C1,
            D12=plant.D12,
            D21=plant.D21,
            D11=plant.D11,
            D22=np.eye(2),
        )


def test_solver_options_default_to_numeric_focus_three_and_allow_zero() -> None:
    assert SolverOptions().numeric_focus == 3
    assert SolverOptions(numeric_focus=0).numeric_focus == 0


@pytest.mark.parametrize("value", [True, False, 1.0, "3"])
def test_solver_options_reject_noninteger_numeric_focus(value: object) -> None:
    with pytest.raises(TypeError, match="numeric_focus must be a non-boolean integer"):
        SolverOptions(numeric_focus=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [-1, 4])
def test_solver_options_reject_out_of_range_numeric_focus(value: int) -> None:
    with pytest.raises(ValueError, match="numeric_focus must be between 0 and 3"):
        SolverOptions(numeric_focus=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("numeric_focus", [3, 0])
def test_numeric_focus_is_applied_to_gurobi_and_recorded_in_result_settings(
    numeric_focus: int,
) -> None:
    problem = BPlusProblem(
        plant=_tiny_plant(),
        deployment=_zero_delay_deployment(),
        horizon=2,
        fixed_eta=(1, 1),
        fixed_xi=(1, 1),
        fixed_service_delays=((0, 0), (0, 0)),
    )
    options = SolverOptions(numeric_focus=numeric_focus)
    built = _build_model(problem, options)

    assert int(built.model.Params.NumericFocus) == numeric_focus

    result = solve_bplus(problem, options)

    assert result.status == "OPTIMAL"
    assert result.settings.numeric_focus == numeric_focus


@pytest.fixture(scope="module")
def tiny_solution() -> tuple[BPlusProblem, BPlusSolveResult]:
    problem = BPlusProblem(
        plant=_tiny_plant(),
        deployment=_zero_delay_deployment(),
        horizon=2,
        fixed_eta=(1, 1),
        fixed_xi=(1, 1),
        fixed_service_delays=((0, 0), (0, 0)),
    )
    result = solve_bplus(problem, SolverOptions())
    assert result.status == "OPTIMAL"
    return problem, result


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("eta", (2, 1), "eta"),
        ("eta", (1,), "eta.*length"),
        ("xi", (1, 2), "xi"),
        ("xi", (1,), "xi.*length"),
    ],
)
def test_audit_rejects_malformed_decoded_device_binaries(
    tiny_solution: tuple[BPlusProblem, BPlusSolveResult],
    field: str,
    value: tuple[int, ...],
    message: str,
) -> None:
    problem, result = tiny_solution

    with pytest.raises(ValueError, match=message):
        audit_solution(problem, replace(result, **{field: value}))


def test_audit_rejects_device_decisions_that_disagree_with_problem_fixes(
    tiny_solution: tuple[BPlusProblem, BPlusSolveResult],
) -> None:
    problem, result = tiny_solution

    with pytest.raises(ValueError, match="fixed_eta"):
        audit_solution(problem, replace(result, eta=(0, 1)))
    with pytest.raises(ValueError, match="fixed_xi"):
        audit_solution(problem, replace(result, xi=(1, 0)))


def test_audit_marks_forged_total_and_performance_objectives_uncertified(
    tiny_solution: tuple[BPlusProblem, BPlusSolveResult],
) -> None:
    problem, result = tiny_solution
    assert result.objective_value is not None
    assert result.performance_objective is not None

    forged_total = audit_solution(
        problem, replace(result, objective_value=result.objective_value + 1.0)
    )
    forged_performance = audit_solution(
        problem,
        replace(
            result,
            performance_objective=result.performance_objective + 2.0,
            objective_value=result.objective_value + 2.0,
        ),
    )

    assert forged_total.total_objective.absolute_error == pytest.approx(1.0)
    assert forged_total.certified is False
    assert forged_performance.objective.absolute_error == pytest.approx(2.0)
    assert forged_performance.total_objective.absolute_error == pytest.approx(2.0)
    assert forged_performance.certified is False
    with pytest.raises(ValueError, match="objective_value"):
        audit_solution(problem, replace(result, objective_value=None))


def test_audit_recomputes_cost_even_if_reported_cost_fields_are_forged_consistently(
    tiny_solution: tuple[BPlusProblem, BPlusSolveResult],
) -> None:
    problem, result = tiny_solution
    forged_breakdown = ArchitectureCostBreakdown(
        total=40.0, actuator=10.0, sensor=20.0, service=10.0
    )

    report = audit_solution(
        problem,
        replace(
            result,
            architecture_cost=40.0,
            architecture_breakdown=forged_breakdown,
        ),
    )

    assert report.architecture_cost.recomputed == pytest.approx(10.0)
    assert report.architecture_cost.absolute_error == pytest.approx(30.0)
    assert report.architecture_cost.breakdown_max_error >= 10.0
    assert report.certified is False


def test_audit_rejects_a_response_with_the_wrong_problem_horizon(
    tiny_solution: tuple[BPlusProblem, BPlusSolveResult],
) -> None:
    problem, result = tiny_solution
    assert result.responses is not None
    response = result.responses
    wrong_horizon = FIRResponses(
        R=np.concatenate((response.R, np.zeros_like(response.R[:1])), axis=0),
        M=np.concatenate((response.M, np.zeros_like(response.M[:1])), axis=0),
        N=np.concatenate((response.N, np.zeros_like(response.N[:1])), axis=0),
        L=np.concatenate((response.L, np.zeros_like(response.L[:1])), axis=0),
    )

    with pytest.raises(ValueError, match="responses.*horizon"):
        audit_solution(problem, replace(result, responses=wrong_horizon))


@pytest.mark.parametrize(
    ("status", "solution_count", "message"),
    [
        ("INFEASIBLE", 0, "solution_count|status"),
        ("INFEASIBLE", 1, "status"),
        ("TIME_LIMIT", 0, "solution_count"),
        ("OPTIMAL", True, "solution_count"),
        ("STATUS_999", 1, "status"),
    ],
)
def test_audit_rejects_non_solution_bearing_status_and_count_combinations(
    tiny_solution: tuple[BPlusProblem, BPlusSolveResult],
    status: str,
    solution_count: int | bool,
    message: str,
) -> None:
    problem, result = tiny_solution

    with pytest.raises(ValueError, match=message):
        audit_solution(
            problem,
            replace(result, status=status, solution_count=solution_count),
        )


def test_audit_allows_a_limit_status_only_when_an_incumbent_is_present(
    tiny_solution: tuple[BPlusProblem, BPlusSolveResult],
) -> None:
    problem, result = tiny_solution

    report = audit_solution(
        problem, replace(result, status="TIME_LIMIT", solution_count=1)
    )

    assert report.certified
