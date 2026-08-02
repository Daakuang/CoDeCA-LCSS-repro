from dataclasses import replace

import numpy as np
from numpy.testing import assert_allclose
import pytest

pytest.importorskip("gurobipy")

from repro.deployment import (
    ArchitectureCostBreakdown,
    DeploymentCostSpec,
    DeploymentSpec,
    DirectedServiceMenu,
    FixedHostLayout,
    ServiceCatalog,
)
from repro.diagnostics import audit_solution, diagnose_hardware_gating
from repro.model import (
    BPlusProblem,
    GeneralizedPlant,
    ResponseFix,
    SolverOptions,
    solve_bplus,
)
from repro.realization import FIRResponses


def _problem(*, eta: tuple[int, int], xi: tuple[int, int]) -> BPlusProblem:
    deployment = DeploymentSpec(
        layout=FixedHostLayout(
            actuator_sites=(0, 1),
            sensor_sites=(0, 1),
            beta_sites=(0, 1),
            state_block_sizes=(1, 1),
            site_count=2,
        ),
        services=ServiceCatalog(
            site_count=2,
            directed_menus=tuple(
                DirectedServiceMenu(r, s, (0,), mandatory_delay=0)
                for r in range(2)
                for s in range(2)
            ),
        ),
        costs=DeploymentCostSpec(
            actuator_costs=np.ones(2),
            sensor_costs=np.ones(2),
            service_costs_by_delay=(np.zeros((2, 2)),),
        ),
    )
    plant = GeneralizedPlant(
        A=np.zeros((2, 2)),
        B2=np.eye(2),
        C2=np.eye(2),
        B1=np.eye(2),
        C1=np.eye(2),
        D12=np.zeros((2, 2)),
        D21=np.zeros((2, 2)),
        D11=np.zeros((2, 2)),
    )
    return BPlusProblem(
        plant=plant,
        deployment=deployment,
        horizon=2,
        fixed_eta=eta,
        fixed_xi=xi,
        fixed_service_delays=((0, 0), (0, 0)),
    )


def test_inactive_actuator_zeros_every_M_and_L_row_without_masking_R1() -> None:
    problem = _problem(eta=(0, 1), xi=(1, 1))

    result = solve_bplus(problem, SolverOptions())

    assert result.status == "OPTIMAL"
    assert result.responses is not None
    assert_allclose(result.responses.M[:, 0, :], 0.0, atol=1.0e-9)
    assert_allclose(result.responses.L[:, 0, :], 0.0, atol=1.0e-9)
    assert_allclose(result.responses.R[1], np.eye(2), atol=1.0e-9)
    report = audit_solution(problem, result)
    assert report.hardware.violation_count == 0
    assert report.hardware.max_abs <= 1.0e-9


def test_inactive_sensor_zeros_every_N_and_L_column_without_masking_R1() -> None:
    problem = _problem(eta=(1, 1), xi=(0, 1))

    result = solve_bplus(problem, SolverOptions())

    assert result.status == "OPTIMAL"
    assert result.responses is not None
    assert_allclose(result.responses.N[:, :, 0], 0.0, atol=1.0e-9)
    assert_allclose(result.responses.L[:, :, 0], 0.0, atol=1.0e-9)
    assert_allclose(result.responses.R[1], np.eye(2), atol=1.0e-9)
    report = audit_solution(problem, result)
    assert report.hardware.violation_count == 0
    assert report.hardware.max_abs <= 1.0e-9


def test_hardware_diagnostic_counts_a_doubly_gated_L_entry_only_once() -> None:
    R = np.zeros((3, 2, 2))
    R[1] = np.eye(2)
    M = np.zeros((3, 2, 2))
    N = np.zeros((3, 2, 2))
    L = np.zeros((3, 2, 2))
    L[0, 0, 0] = 1.0
    responses = FIRResponses(R=R, M=M, N=N, L=L)

    report = diagnose_hardware_gating(
        responses, eta=(0, 1), xi=(0, 1), tolerance=1.0e-9
    )

    assert report.violation_count == 1
    assert report.max_abs == pytest.approx(1.0)


def test_hard_budget_with_free_hardware_decodes_all_selectors_and_is_audited() -> None:
    fixed = _problem(eta=(1, 1), xi=(1, 1))
    problem = BPlusProblem(
        plant=fixed.plant,
        deployment=fixed.deployment,
        horizon=fixed.horizon,
        fixed_service_delays=fixed.fixed_service_delays,
    )

    result = solve_bplus(problem, SolverOptions(architecture_budget=0.0))

    assert result.status == "OPTIMAL"
    assert result.eta == (0, 0)
    assert result.xi == (0, 0)
    assert result.architecture_cost == pytest.approx(0.0)
    assert result.objective_value == pytest.approx(result.performance_objective)
    report = audit_solution(problem, result)
    assert report.budget_violation == pytest.approx(0.0)
    assert report.certified

    forged_breakdown = ArchitectureCostBreakdown(
        total=1.0, actuator=1.0, sensor=0.0, service=0.0
    )
    over_budget = audit_solution(
        problem,
        replace(
            result,
            eta=(1, 0),
            architecture_cost=1.0,
            architecture_breakdown=forged_breakdown,
        ),
    )
    assert over_budget.budget_violation == pytest.approx(1.0)
    assert over_budget.certified is False


def test_infeasible_model_returns_no_incumbent_and_audit_rejects_it() -> None:
    fixed = _problem(eta=(0, 0), xi=(0, 0))
    problem = BPlusProblem(
        plant=fixed.plant,
        deployment=fixed.deployment,
        horizon=fixed.horizon,
        fixed_eta=fixed.fixed_eta,
        fixed_xi=fixed.fixed_xi,
        fixed_service_delays=fixed.fixed_service_delays,
        response_fixes=(ResponseFix("L", lag=0, row=0, column=0, value=1.0),),
    )

    result = solve_bplus(problem, SolverOptions())

    assert result.status in {"INFEASIBLE", "INF_OR_UNBD"}
    assert result.solution_count == 0
    assert result.responses is None
    assert result.eta is None
    assert result.objective_value is None
    with pytest.raises(ValueError, match="solution-bearing"):
        audit_solution(problem, result)
