from dataclasses import replace

import numpy as np
from numpy.testing import assert_allclose
import pytest

pytest.importorskip("gurobipy")

from repro.deployment import (
    DeploymentCostSpec,
    DeploymentSpec,
    DirectedServiceMenu,
    FixedHostLayout,
    ServiceCatalog,
)
from repro.diagnostics import audit_solution, diagnose_service_gating
from repro.model import (
    BPlusProblem,
    BPlusSolveResult,
    GeneralizedPlant,
    ResponseFix,
    SolverOptions,
    solve_bplus,
)
from repro.realization import FIRResponses


def _fixed_delay_deployment(
    selected_delays: tuple[tuple[int | None, ...], ...],
) -> DeploymentSpec:
    site_count = len(selected_delays)
    finite_delays = [
        delay
        for destination, row in enumerate(selected_delays)
        for source, delay in enumerate(row)
        if destination != source and delay is not None
    ]
    menus = [
        DirectedServiceMenu(site, site, (0,), mandatory_delay=0)
        for site in range(site_count)
    ]
    menus.extend(
        DirectedServiceMenu(destination, source, (delay,))
        for destination, row in enumerate(selected_delays)
        for source, delay in enumerate(row)
        if destination != source and delay is not None
    )
    maximum_delay = max(finite_delays, default=0)
    return DeploymentSpec(
        layout=FixedHostLayout(
            actuator_sites=tuple(range(site_count)),
            sensor_sites=tuple(range(site_count)),
            beta_sites=tuple(range(site_count)),
            state_block_sizes=tuple(1 for _ in range(site_count)),
            site_count=site_count,
        ),
        services=ServiceCatalog(site_count=site_count, directed_menus=tuple(menus)),
        costs=DeploymentCostSpec(
            actuator_costs=np.zeros(site_count),
            sensor_costs=np.zeros(site_count),
            service_costs_by_delay=tuple(
                np.zeros((site_count, site_count))
                for _ in range(maximum_delay + 1)
            ),
        ),
    )


def _zero_performance_plant(
    A: np.ndarray, *, B2: np.ndarray | None = None, C2: np.ndarray | None = None
) -> GeneralizedPlant:
    state_count = A.shape[0]
    return GeneralizedPlant(
        A=A,
        B2=np.zeros_like(A) if B2 is None else B2,
        C2=np.zeros_like(A) if C2 is None else C2,
        B1=np.eye(state_count),
        C1=np.zeros((1, state_count)),
        D12=np.zeros((1, state_count)),
        D21=np.zeros((state_count, state_count)),
        D11=np.zeros((1, state_count)),
    )


def _delay_two_problem() -> BPlusProblem:
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
            directed_menus=(
                DirectedServiceMenu(0, 0, (0,), mandatory_delay=0),
                DirectedServiceMenu(1, 1, (0,), mandatory_delay=0),
                DirectedServiceMenu(0, 1, (2,)),
                DirectedServiceMenu(1, 0, (2,)),
            ),
        ),
        costs=DeploymentCostSpec(
            actuator_costs=np.zeros(2),
            sensor_costs=np.zeros(2),
            service_costs_by_delay=(
                np.zeros((2, 2)),
                np.zeros((2, 2)),
                np.array([[0.0, 5.0], [7.0, 0.0]]),
            ),
        ),
    )
    plant = GeneralizedPlant(
        A=np.zeros((2, 2)),
        B2=np.eye(2),
        C2=np.eye(2),
        B1=np.eye(2),
        C1=np.zeros((1, 2)),
        D12=np.zeros((1, 2)),
        D21=np.zeros((2, 2)),
        D11=np.zeros((1, 2)),
    )
    return BPlusProblem(
        plant=plant,
        deployment=deployment,
        horizon=4,
        fixed_eta=(1, 1),
        fixed_xi=(1, 1),
        fixed_service_delays=((0, 2), (None, 0)),
        response_fixes=(ResponseFix("L", lag=2, row=0, column=1, value=0.5),),
    )


@pytest.fixture(scope="module")
def delay_two_solution() -> tuple[BPlusProblem, BPlusSolveResult]:
    problem = _delay_two_problem()
    result = solve_bplus(problem, SolverOptions())

    assert result.status == "OPTIMAL"
    return problem, result


def test_delay_two_uses_exact_zero_one_zero_one_deadline_indices(
    delay_two_solution: tuple[BPlusProblem, BPlusSolveResult],
) -> None:
    problem, result = delay_two_solution

    assert result.status == "OPTIMAL"
    assert result.responses is not None
    response = result.responses
    assert result.service_delays == ((0, 2), (None, 0))
    assert result.service_availability[0][1] == (0, 0, 1)
    assert result.service_availability[1][0] == (0, 0, 0)

    # For destination site 0 <- source site 1 and d=2, the first permitted raw
    # coefficients are L[2], M[3], N[2], and R[3].  This fixture fixes L[2],
    # which propagates one algebraic step later to N[3], M[3], and R[4].
    assert_allclose(response.L[:2, 0, 1], 0.0, atol=1.0e-9)
    assert_allclose(response.M[1:3, 0, 1], 0.0, atol=1.0e-9)
    assert response.N[1, 0, 1] == pytest.approx(0.0, abs=1.0e-9)
    assert response.R[2, 0, 1] == pytest.approx(0.0, abs=1.0e-9)
    assert response.L[2, 0, 1] == pytest.approx(0.5, abs=1.0e-9)
    assert response.M[3, 0, 1] == pytest.approx(0.5, abs=1.0e-9)
    assert response.N[3, 0, 1] == pytest.approx(0.5, abs=1.0e-9)
    assert response.R[4, 0, 1] == pytest.approx(0.5, abs=1.0e-9)
    assert_allclose(response.R[1], np.eye(2), atol=1.0e-9)

    report = audit_solution(problem, result, tolerance=1.0e-7)
    assert report.ofsls.max_residual <= 1.0e-7
    assert report.service.violation_count == 0
    assert report.service.max_abs <= 1.0e-9
    assert report.service.menu_valid
    assert report.architecture_cost.recomputed == pytest.approx(5.0)
    assert report.certified


def test_delay_two_allows_cross_host_N2_but_rejects_early_N1() -> None:
    selected_delays = ((0, 2), (None, 0))
    problem = BPlusProblem(
        plant=_zero_performance_plant(
            np.zeros((2, 2)), B2=np.array([[0.0, 1.0], [1.0, 0.0]])
        ),
        deployment=_fixed_delay_deployment(selected_delays),
        horizon=2,
        fixed_eta=(1, 1),
        fixed_xi=(1, 1),
        fixed_service_delays=selected_delays,
        response_fixes=(ResponseFix("L", lag=1, row=1, column=1, value=0.5),),
    )

    result = solve_bplus(problem, SolverOptions())

    assert result.status == "OPTIMAL"
    assert result.responses is not None
    assert result.responses.N[1, 0, 1] == pytest.approx(0.0, abs=1.0e-9)
    assert result.responses.N[2, 0, 1] == pytest.approx(0.5, abs=1.0e-9)
    report = audit_solution(problem, result, tolerance=1.0e-8)
    assert report.service.violation_count == 0
    assert report.certified

    early_N = result.responses.N.copy()
    early_N[1, 0, 1] = 0.25
    early_response = FIRResponses(
        R=result.responses.R,
        M=result.responses.M,
        N=early_N,
        L=result.responses.L,
    )
    early_report = audit_solution(
        problem, replace(result, responses=early_response), tolerance=1.0e-8
    )
    assert early_report.service.violation_count >= 1
    assert early_report.certified is False


def test_delay_two_allows_distance_two_R3_but_rejects_early_R2() -> None:
    selected_delays = (
        (0, 1, 2),
        (None, 0, 1),
        (None, None, 0),
    )
    A = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )
    problem = BPlusProblem(
        plant=_zero_performance_plant(A),
        deployment=_fixed_delay_deployment(selected_delays),
        horizon=3,
        fixed_eta=(1, 1, 1),
        fixed_xi=(1, 1, 1),
        fixed_service_delays=selected_delays,
    )

    result = solve_bplus(problem, SolverOptions())

    assert result.status == "OPTIMAL"
    assert result.responses is not None
    assert result.responses.R[2, 0, 2] == pytest.approx(0.0, abs=1.0e-9)
    assert result.responses.R[3, 0, 2] == pytest.approx(1.0, abs=1.0e-9)
    report = audit_solution(problem, result, tolerance=1.0e-8)
    assert report.service.violation_count == 0
    assert report.certified

    early_R = result.responses.R.copy()
    early_R[2, 0, 2] = 0.25
    early_response = FIRResponses(
        R=early_R,
        M=result.responses.M,
        N=result.responses.N,
        L=result.responses.L,
    )
    early_report = audit_solution(
        problem, replace(result, responses=early_response), tolerance=1.0e-8
    )
    assert early_report.service.violation_count >= 1
    assert early_report.certified is False


def test_delay_one_allows_cross_A_response_at_R2() -> None:
    selected_delays = ((0, 1), (None, 0))
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    problem = BPlusProblem(
        plant=_zero_performance_plant(A),
        deployment=_fixed_delay_deployment(selected_delays),
        horizon=2,
        fixed_eta=(1, 1),
        fixed_xi=(1, 1),
        fixed_service_delays=selected_delays,
    )

    result = solve_bplus(problem, SolverOptions())

    assert result.status == "OPTIMAL"
    assert result.responses is not None
    assert result.responses.R[2, 0, 1] == pytest.approx(1.0, abs=1.0e-9)
    report = audit_solution(problem, result, tolerance=1.0e-8)
    assert report.certified


def test_audit_rejects_malformed_service_delays_and_availability(
    delay_two_solution: tuple[BPlusProblem, BPlusSolveResult],
) -> None:
    problem, result = delay_two_solution
    assert result.service_availability is not None
    malformed_binary = list(list(row) for row in result.service_availability)
    malformed_binary[0] = list(malformed_binary[0])
    malformed_binary[0][1] = (0, 2, 1)
    inconsistent = list(list(row) for row in result.service_availability)
    inconsistent[0] = list(inconsistent[0])
    inconsistent[0][1] = (0, 0, 0)

    with pytest.raises(ValueError, match="service_delays.*shape"):
        audit_solution(problem, replace(result, service_delays=((0,),)))
    with pytest.raises(ValueError, match="service_availability.*shape"):
        audit_solution(problem, replace(result, service_availability=(((0,),),)))
    with pytest.raises(ValueError, match="service_availability.*binary"):
        audit_solution(
            problem,
            replace(
                result,
                service_availability=tuple(
                    tuple(tuple(entry) for entry in row) for row in malformed_binary
                ),
            ),
        )
    with pytest.raises(ValueError, match="service_availability.*inconsistent"):
        audit_solution(
            problem,
            replace(
                result,
                service_availability=tuple(
                    tuple(tuple(entry) for entry in row) for row in inconsistent
                ),
            ),
        )
    with pytest.raises(ValueError, match="service_delays.*invalid"):
        audit_solution(
            problem,
            replace(result, service_delays=((0, 1), (None, 0))),
        )


def test_low_level_service_diagnostic_rejects_malformed_inputs_without_index_error(
    delay_two_solution: tuple[BPlusProblem, BPlusSolveResult],
) -> None:
    problem, result = delay_two_solution
    assert result.responses is not None
    assert result.service_availability is not None

    with pytest.raises(ValueError, match="service_delays.*shape"):
        diagnose_service_gating(
            problem,
            result.responses,
            ((0,),),
            result.service_availability,
            tolerance=1.0e-8,
        )


def test_audit_rejects_fixed_service_and_response_fix_mismatches(
    delay_two_solution: tuple[BPlusProblem, BPlusSolveResult],
) -> None:
    problem, result = delay_two_solution
    assert result.responses is not None
    off_availability = (
        ((1, 1, 1), (0, 0, 0)),
        ((0, 0, 0), (1, 1, 1)),
    )
    with pytest.raises(ValueError, match="fixed_service_delays"):
        audit_solution(
            problem,
            replace(
                result,
                service_delays=((0, None), (None, 0)),
                service_availability=off_availability,
            ),
        )

    L = result.responses.L.copy()
    L[2, 0, 1] = 0.25
    tampered_response = FIRResponses(
        R=result.responses.R,
        M=result.responses.M,
        N=result.responses.N,
        L=L,
    )
    with pytest.raises(ValueError, match="response_fixes"):
        audit_solution(problem, replace(result, responses=tampered_response))


def test_scalarization_with_free_services_decodes_selected_and_off_menu_items() -> None:
    fixed = _delay_two_problem()
    problem = BPlusProblem(
        plant=fixed.plant,
        deployment=fixed.deployment,
        horizon=fixed.horizon,
        fixed_eta=fixed.fixed_eta,
        fixed_xi=fixed.fixed_xi,
        response_fixes=fixed.response_fixes,
    )

    result = solve_bplus(problem, SolverOptions(architecture_weight=1.0))

    assert result.status == "OPTIMAL"
    assert result.service_delays == ((0, 2), (None, 0))
    assert result.architecture_cost == pytest.approx(5.0)
    assert result.performance_objective == pytest.approx(0.0)
    assert result.objective_value == pytest.approx(5.0)
    report = audit_solution(problem, result)
    assert report.total_objective.absolute_error <= 1.0e-9
    assert report.certified
