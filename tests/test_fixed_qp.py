"""Direct continuous fixed-architecture OF-SLS QP checks."""

from __future__ import annotations

import pytest

pytest.importorskip("gurobipy")

from repro.cases import make_joint_platoon_case
from repro.diagnostics import audit_solution
from repro.fixed_qp import _build_fixed_qp, solve_fixed_architecture_qp, solve_fixed_qp
from repro.model import BPlusProblem, ResponseFix, SolverOptions
from repro.platoon_experiment import platoon_dense_architecture
from test_model_tiny import (
    _independent_fixed_fir_qp,
    _tiny_plant,
    _zero_delay_deployment,
)


def _tiny_problem() -> BPlusProblem:
    return BPlusProblem(
        plant=_tiny_plant(),
        deployment=_zero_delay_deployment(),
        horizon=2,
        fixed_eta=(1, 1),
        fixed_xi=(1, 1),
        fixed_service_delays=((0, 0), (0, 0)),
    )


def test_direct_builder_contains_only_continuous_response_variables() -> None:
    problem = _tiny_problem()
    model, variables, choice = _build_fixed_qp(problem, SolverOptions())
    model.update()
    try:
        assert model.NumBinVars == 0
        assert model.NumIntVars == 0
        assert model.NumVars > 0
        assert set(variables) == {"R", "M", "N", "L"}
        assert choice.eta == (1, 1)
        assert choice.xi == (1, 1)
    finally:
        model.dispose()


def test_direct_qp_matches_existing_independent_dense_oracle() -> None:
    problem = _tiny_problem()
    expected = _independent_fixed_fir_qp(problem.plant, problem.horizon)
    result = solve_fixed_architecture_qp(problem, SolverOptions())
    audit = audit_solution(problem, result, tolerance=2.0e-8)

    assert result.status == "OPTIMAL"
    assert result.performance_objective == pytest.approx(expected, abs=1.0e-9)
    assert result.architecture_cost == pytest.approx(10.0)
    assert result.best_bound is None
    assert result.mip_gap is None
    assert audit.certified
    assert audit.ofsls.max_residual <= 2.0e-8


@pytest.mark.parametrize(
    ("name", "delays", "xi", "budget", "expected_objective"),
    (
        (
            "proposed_B20",
            (
                (0, None, None, None),
                (None, 0, 2, None),
                (None, 1, 0, None),
                (1, 2, 1, 0),
            ),
            (1, 1, 0, 1, 0, 1, 0),
            20.0,
            5.253928415821751,
        ),
        (
            "qi_service_0018_sensor_043",
            (
                (0, None, None, None),
                (1, 0, 1, 2),
                (1, 1, 0, 2),
                (1, 2, 1, 0),
            ),
            (1, 1, 0, 1, 0, 1, 0),
            31.0,
            4.9874406072352055,
        ),
    ),
)
def test_joint_fixed_qp_certifies_difficult_architectures(
    name: str,
    delays: tuple[tuple[int | None, ...], ...],
    xi: tuple[int, ...],
    budget: float,
    expected_objective: float,
) -> None:
    del name  # Kept in parametrization so failures identify the architecture.
    case = make_joint_platoon_case(seed=23)
    problem = BPlusProblem(
        plant=case.plant,
        deployment=case.deployment,
        horizon=10,
        fixed_eta=(1, 1, 1),
        fixed_xi=xi,
        fixed_service_delays=delays,
    )
    result = solve_fixed_qp(
        problem, SolverOptions(architecture_budget=budget)
    )
    audit = audit_solution(problem, result, tolerance=2.0e-8)

    assert result.status == "OPTIMAL"
    assert result.performance_objective == pytest.approx(
        expected_objective, abs=2.0e-10
    )
    assert result.architecture_cost == pytest.approx(budget)
    assert audit.certified
    assert audit.ofsls.max_residual <= 2.0e-8


def test_joint_dense_qp_matches_publication_reference() -> None:
    case = make_joint_platoon_case(seed=23)
    problem = BPlusProblem(
        plant=case.plant,
        deployment=case.deployment,
        horizon=10,
        fixed_eta=(1, 1, 1),
        fixed_xi=(1, 1, 1, 1, 1, 1, 1),
        fixed_service_delays=platoon_dense_architecture(case.base),
    )
    result = solve_fixed_qp(
        problem, SolverOptions(architecture_budget=35.0)
    )
    audit = audit_solution(problem, result, tolerance=2.0e-8)

    assert result.status == "OPTIMAL"
    assert result.performance_objective == pytest.approx(
        4.987426757285976, abs=2.0e-10
    )
    assert audit.certified
    assert audit.ofsls.max_residual <= 2.0e-8


def test_fixed_architecture_budget_and_scalarized_objective_semantics() -> None:
    problem = _tiny_problem()
    infeasible = solve_fixed_qp(
        problem, SolverOptions(architecture_budget=9.0)
    )
    assert infeasible.status == "INFEASIBLE"
    assert infeasible.solution_count == 0
    assert infeasible.responses is None

    scalarized = solve_fixed_qp(
        problem, SolverOptions(architecture_weight=0.25)
    )
    assert scalarized.performance_objective is not None
    assert scalarized.objective_value == pytest.approx(
        scalarized.performance_objective + 0.25 * 10.0, abs=1.0e-9
    )
    assert audit_solution(problem, scalarized, tolerance=2.0e-8).certified


def test_response_fixes_and_separate_budget_caps_are_preserved() -> None:
    base = _tiny_problem()
    fixed = BPlusProblem(
        plant=base.plant,
        deployment=base.deployment,
        horizon=base.horizon,
        fixed_eta=base.fixed_eta,
        fixed_xi=base.fixed_xi,
        fixed_service_delays=base.fixed_service_delays,
        response_fixes=(ResponseFix("R", 1, 0, 0, 1.0),),
    )
    result = solve_fixed_qp(
        fixed, SolverOptions(hardware_budget=10.0, service_budget=0.0)
    )
    assert result.status == "OPTIMAL"
    assert audit_solution(fixed, result, tolerance=2.0e-8).certified

    contradictory = BPlusProblem(
        plant=base.plant,
        deployment=base.deployment,
        horizon=base.horizon,
        fixed_eta=base.fixed_eta,
        fixed_xi=base.fixed_xi,
        fixed_service_delays=base.fixed_service_delays,
        response_fixes=(ResponseFix("R", 1, 0, 0, 0.0),),
    )
    infeasible = solve_fixed_qp(contradictory, SolverOptions())
    assert infeasible.status == "INFEASIBLE"
    assert infeasible.solution_count == 0


def test_direct_qp_rejects_an_incompletely_fixed_architecture() -> None:
    problem = BPlusProblem(
        plant=_tiny_plant(),
        deployment=_zero_delay_deployment(),
        horizon=2,
        fixed_eta=(1, 1),
        fixed_xi=None,
        fixed_service_delays=((0, 0), (0, 0)),
    )
    with pytest.raises(ValueError, match="requires fixed eta, xi, and service delays"):
        solve_fixed_qp(problem, SolverOptions())
