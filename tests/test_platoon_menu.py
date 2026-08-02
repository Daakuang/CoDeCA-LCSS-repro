"""Service-menu, fixed-QP, and realization checks for the platoon."""

from __future__ import annotations

import itertools

import gurobipy as gp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from repro.cases import make_platoon_case
from repro.deployment import ArchitectureChoice, architecture_cost
from repro.diagnostics import audit_solution
from repro.model import SolverOptions, solve_bplus
from repro.platoon_experiment import (
    build_platoon_problem,
    canonical_service_serialization,
    iter_service_architectures,
    platoon_dense_architecture,
    platoon_sparse_architecture,
)
from repro.realization import realization_filters, rollout_realization


def test_submitted_directed_service_menu_costs_and_count_are_exact() -> None:
    case = make_platoon_case()
    catalog = case.deployment.services
    for site in range(4):
        menu = catalog.menu_for(destination=site, source=site)
        assert menu is not None
        assert menu.finite_delays == (0,)
        assert menu.mandatory_delay == 0
    for destination in (1, 2, 3):
        leader = catalog.menu_for(destination=destination, source=0)
        assert leader is not None
        assert leader.finite_delays == (1, 2)
        assert leader.mandatory_delay is None
    for destination, source in itertools.permutations((1, 2, 3), 2):
        menu = catalog.menu_for(destination=destination, source=source)
        assert menu is not None
        expected = (1, 2) if abs(destination - source) == 1 else (2,)
        assert menu.finite_delays == expected
        assert menu.mandatory_delay is None
    for source in (1, 2, 3):
        assert catalog.is_forbidden(destination=0, source=source)

    costs = case.deployment.costs.service_costs_by_delay
    assert len(costs) == 3
    for destination in (1, 2, 3):
        for source in (0, 1, 2, 3):
            if destination == source:
                assert costs[0][destination, source] == 0.0
            else:
                assert costs[1][destination, source] == pytest.approx(0.001 / 2.0)
                assert costs[2][destination, source] == pytest.approx(0.001 / 3.0)

    architectures = tuple(iter_service_architectures(case))
    assert len(architectures) == 8748
    assert len({canonical_service_serialization(item) for item in architectures}) == 8748
    expected_product = 3**3 * 3**4 * 2**2
    assert len(architectures) == expected_product


def test_dense_and_sparse_architectures_have_submitted_costs() -> None:
    case = make_platoon_case()
    dense = platoon_dense_architecture(case)
    sparse = platoon_sparse_architecture(case)
    assert dense == (
        (0, None, None, None),
        (1, 0, 1, 2),
        (1, 1, 0, 1),
        (1, 2, 1, 0),
    )
    assert sparse == (
        (0, None, None, None),
        (None, 0, 1, None),
        (1, 1, 0, None),
        (1, 2, 1, 0),
    )
    dense_cost = architecture_cost(
        case.deployment,
        ArchitectureChoice(case.fixed_eta, case.fixed_xi, dense),
    )
    sparse_cost = architecture_cost(
        case.deployment,
        ArchitectureChoice(case.fixed_eta, case.fixed_xi, sparse),
    )
    assert dense_cost.total == pytest.approx(0.004166666666666667)
    assert sparse_cost.total == pytest.approx(0.0028333333333333335)
    assert dense_cost.actuator == dense_cost.sensor == 0.0


def _hosted_state_sites() -> tuple[int, ...]:
    return (1, 1, 1, 2, 2, 2, 3, 3, 3)


def _service_allows(delays: tuple[tuple[int | None, ...], ...], r: int, s: int, lag: int) -> bool:
    delay = delays[r][s]
    return delay is not None and delay <= lag


def _independent_fixed_qp(
    case: object,
    delays: tuple[tuple[int | None, ...], ...],
    horizon: int,
) -> tuple[float, dict[str, np.ndarray]]:
    """Literal Gurobi QP oracle; intentionally imports no model/objective helper."""

    plant = case.plant
    model = gp.Model("independent_platoon_fixed_qp")
    model.Params.OutputFlag = 0
    model.Params.Seed = 23
    model.Params.Threads = 1
    model.Params.NumericFocus = 3
    model.Params.FeasibilityTol = 1.0e-9
    model.Params.OptimalityTol = 1.0e-9
    infinity = gp.GRB.INFINITY
    R = tuple(model.addMVar((9, 9), lb=-infinity) for _ in range(horizon + 1))
    M = tuple(model.addMVar((3, 9), lb=-infinity) for _ in range(horizon + 1))
    N = tuple(model.addMVar((9, 10), lb=-infinity) for _ in range(horizon + 1))
    L = tuple(model.addMVar((3, 10), lb=-infinity) for _ in range(horizon + 1))
    model.addConstr(R[0] == np.zeros((9, 9)))
    model.addConstr(M[0] == np.zeros((3, 9)))
    model.addConstr(N[0] == np.zeros((9, 10)))
    for t in range(horizon + 1):
        rn = R[t + 1] if t < horizon else np.zeros((9, 9))
        mn = M[t + 1] if t < horizon else np.zeros((3, 9))
        nn = N[t + 1] if t < horizon else np.zeros((9, 10))
        impulse = np.eye(9) if t == 0 else np.zeros((9, 9))
        model.addConstr(rn - plant.A @ R[t] - plant.B2 @ M[t] == impulse)
        model.addConstr(nn - plant.A @ N[t] - plant.B2 @ L[t] == np.zeros((9, 10)))
        model.addConstr(rn - R[t] @ plant.A - N[t] @ plant.C2 == impulse)
        model.addConstr(mn - M[t] @ plant.A - L[t] @ plant.C2 == np.zeros((3, 9)))

    state_sites = _hosted_state_sites()
    actuator_sites = (1, 2, 3)
    sensor_sites = (0, 1, 1, 1, 2, 2, 2, 3, 3, 3)
    # Hand-written deadline table, independent of repro.realization helpers:
    # L[t] -> t, M[t] -> t-1, N[t] -> t, R[t] -> t-1 (R[1] exempt).
    for t in range(horizon + 1):
        for i, destination in enumerate(actuator_sites):
            for j, source in enumerate(sensor_sites):
                if not _service_allows(delays, destination, source, t):
                    model.addConstr(L[t][i, j] == 0.0)
    for t in range(1, horizon + 1):
        for i, destination in enumerate(actuator_sites):
            for j, source in enumerate(state_sites):
                if not _service_allows(delays, destination, source, t - 1):
                    model.addConstr(M[t][i, j] == 0.0)
        for i, destination in enumerate(state_sites):
            for j, source in enumerate(sensor_sites):
                # N[t] contributes to beta[t+1], so y[t-t] may arrive at t+1.
                if not _service_allows(delays, destination, source, t):
                    model.addConstr(N[t][i, j] == 0.0)
    for t in range(2, horizon + 1):
        for i, destination in enumerate(state_sites):
            for j, source in enumerate(state_sites):
                # R[t] is Rplus[t-2] in the beta[t+1] update: deadline t-1.
                if not _service_allows(delays, destination, source, t - 1):
                    model.addConstr(R[t][i, j] == 0.0)

    objective = gp.QuadExpr()
    for t in range(horizon + 1):
        impulse = (
            (plant.D11 if t == 0 else np.zeros_like(plant.D11))
            + plant.C1 @ (R[t] @ plant.B1 + N[t] @ plant.D21)
            + plant.D12 @ (M[t] @ plant.B1 + L[t] @ plant.D21)
        )
        for row in range(11):
            for column in range(4):
                objective += impulse[row, column] * impulse[row, column]
    model.setObjective(objective, gp.GRB.MINIMIZE)
    model.optimize()
    assert model.Status == gp.GRB.OPTIMAL
    objective_value = float(model.ObjVal)
    response_values = {
        "R": np.stack([np.asarray(block.X) for block in R]),
        "M": np.stack([np.asarray(block.X) for block in M]),
        "N": np.stack([np.asarray(block.X) for block in N]),
        "L": np.stack([np.asarray(block.X) for block in L]),
    }
    model.dispose()
    return objective_value, response_values


def _central_controller_coefficients(responses: object, sample_count: int) -> np.ndarray:
    Rhat = responses.R[1:]
    inverse = np.zeros((sample_count, 9, 9))
    inverse[0] = np.eye(9)
    for lag in range(1, sample_count):
        for rlag in range(1, min(lag, len(Rhat) - 1) + 1):
            inverse[lag] -= Rhat[rlag] @ inverse[lag - rlag]
    controller = np.zeros((sample_count, 3, 10))
    controller[: min(sample_count, len(responses.L))] += responses.L[:sample_count]
    for mlag in range(1, len(responses.M)):
        for ilag in range(sample_count):
            for nlag in range(1, len(responses.N)):
                lag = mlag + ilag + nlag - 1
                if lag < sample_count:
                    controller[lag] -= responses.M[mlag] @ inverse[ilag] @ responses.N[nlag]
    return controller


def _independent_ofsls_residual_max(case: object, responses: object) -> float:
    """Literal four-equation/terminal residual oracle."""

    plant = case.plant
    horizon = responses.R.shape[0] - 1
    residuals: list[float] = []
    for t in range(horizon + 1):
        R_next = responses.R[t + 1] if t < horizon else np.zeros((9, 9))
        M_next = responses.M[t + 1] if t < horizon else np.zeros((3, 9))
        N_next = responses.N[t + 1] if t < horizon else np.zeros((9, 10))
        impulse = np.eye(9) if t == 0 else np.zeros((9, 9))
        residuals.extend(
            (
                float(
                    np.max(
                        np.abs(
                            R_next
                            - plant.A @ responses.R[t]
                            - plant.B2 @ responses.M[t]
                            - impulse
                        )
                    )
                ),
                float(
                    np.max(
                        np.abs(
                            N_next
                            - plant.A @ responses.N[t]
                            - plant.B2 @ responses.L[t]
                        )
                    )
                ),
                float(
                    np.max(
                        np.abs(
                            R_next
                            - responses.R[t] @ plant.A
                            - responses.N[t] @ plant.C2
                            - impulse
                        )
                    )
                ),
                float(
                    np.max(
                        np.abs(
                            M_next
                            - responses.M[t] @ plant.A
                            - responses.L[t] @ plant.C2
                        )
                    )
                ),
            )
        )
    return max(residuals)


def _closed_loop_responses(case: object, responses: object) -> tuple[np.ndarray, np.ndarray]:
    plant = case.plant
    x_response = np.stack(
        [
            responses.R[t] @ plant.B1 + responses.N[t] @ plant.D21
            for t in range(responses.R.shape[0])
        ]
    )
    u_response = np.stack(
        [
            responses.M[t] @ plant.B1 + responses.L[t] @ plant.D21
            for t in range(responses.R.shape[0])
        ]
    )
    return x_response, u_response


@pytest.mark.parametrize("architecture_name", ["dense", "sparse"])
def test_fixed_bplus_matches_independent_qp_support_and_nodewise_realization(
    architecture_name: str,
) -> None:
    case = make_platoon_case()
    delays = (
        platoon_dense_architecture(case)
        if architecture_name == "dense"
        else platoon_sparse_architecture(case)
    )
    problem = build_platoon_problem(case, delays, horizon=10)
    result = solve_bplus(problem, SolverOptions())
    assert result.status == "OPTIMAL"
    assert result.responses is not None
    audit = audit_solution(problem, result, tolerance=2.0e-8)
    assert audit.certified
    assert audit.ofsls.max_residual <= 2.0e-8
    assert audit.service.violation_count == 0
    assert audit.hardware.violation_count == 0
    oracle_objective, oracle_responses = _independent_fixed_qp(case, delays, 10)
    assert abs(float(result.performance_objective) - oracle_objective) <= 2.0e-7

    responses = result.responses
    assert_allclose(responses.R[1], np.eye(9), rtol=0.0, atol=2.0e-8)
    oracle = type("OracleResponses", (), oracle_responses)()
    # Raw OF-SLS responses are not unique on unexcited disturbance directions.
    # Compare the physical w->(x,u) responses, and independently require both
    # raw tuples to satisfy all four affine recursions and terminal equations.
    actual_x, actual_u = _closed_loop_responses(case, responses)
    oracle_x, oracle_u = _closed_loop_responses(case, oracle)
    assert_allclose(actual_x, oracle_x, rtol=0.0, atol=2.0e-8)
    assert_allclose(actual_u, oracle_u, rtol=0.0, atol=2.0e-8)
    assert _independent_ofsls_residual_max(case, responses) <= 2.0e-8
    assert _independent_ofsls_residual_max(case, oracle) <= 2.0e-8
    # A delay-two service first permits L[2], M[3], N[2], and R[3].
    # Allowed blocks need not be nonzero at an optimum; only early zeros are
    # asserted.  Pair: follower 3 <- follower 1.
    assert_allclose(responses.L[:2, 2, 1:4], 0.0, atol=2.0e-8)
    assert_allclose(responses.M[1:3, 2, 0:3], 0.0, atol=2.0e-8)
    assert_allclose(responses.N[1, 6:9, 1:4], 0.0, atol=2.0e-8)
    assert_allclose(responses.R[2, 6:9, 0:3], 0.0, atol=2.0e-8)

    filters = realization_filters(responses)
    rng = np.random.default_rng(23)
    measurements = rng.standard_normal((24, 10))
    rollout = rollout_realization(filters, measurements)
    controller = _central_controller_coefficients(responses, len(measurements))
    centralized = np.zeros_like(rollout.u)
    for t in range(len(measurements)):
        for lag in range(t + 1):
            centralized[t] += controller[lag] @ measurements[t - lag]
    assert np.max(np.abs(rollout.u - centralized)) < 1.0e-9
