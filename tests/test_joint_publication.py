"""Publication-pipeline invariants that do not require a Gurobi license."""

from __future__ import annotations

from dataclasses import dataclass

from repro.cases import make_joint_platoon_case
from repro.joint_publication import (
    DEFAULT_BUDGETS,
    DEFAULT_HORIZON,
    DEFAULT_SEED,
    DEFAULT_THREADS,
    EXPECTED_PBH_ADMISSIBLE_QI_COUNT,
    HORIZON_SENSITIVITY_BUDGETS,
    HORIZON_SENSITIVITY_HORIZONS,
    build_tolerance_audited_envelope,
    enumerate_pbh_admissible_qi_deployments,
    exhaustive_pbh_certificate,
    solve_fixed_continuous_qp,
    solve_with_dual_reductions_disabled,
)
from repro.joint_platoon_experiment import make_joint_problem
from repro.model import BPlusProblem, SolverOptions
from repro.platoon_experiment import platoon_dense_architecture


def _raw_point(
    budget: int,
    performance: float,
    lower_bound: float,
    cost: float,
) -> dict[str, object]:
    return {
        "budget": budget,
        "certificate": {"certified": True, "lower_bound": lower_bound},
        "fixed_architecture_resynthesis": {
            "performance_h2": performance,
            "architecture_cost": cost,
            "eta": [1, 1, 1],
            "xi": [0, 1, 0, 1, 0, 1, 0],
            "service_delays": [[0]],
        },
    }


def test_publication_grid_and_solver_settings_are_explicit() -> None:
    assert DEFAULT_SEED == 23
    assert DEFAULT_THREADS == 1
    assert DEFAULT_HORIZON == 10
    assert DEFAULT_BUDGETS == tuple(range(14, 36))
    assert HORIZON_SENSITIVITY_BUDGETS == (14, 24, 35)
    assert HORIZON_SENSITIVITY_HORIZONS == (8, 10, 12)


def test_pbh_and_qi_catalog_have_the_exact_declared_identity_count() -> None:
    case = make_joint_platoon_case(seed=23)
    pbh = exhaustive_pbh_certificate(case)

    assert pbh["actuator_pattern_count"] == 8
    assert pbh["sensor_pattern_count"] == 128
    assert pbh["stabilizable_actuator_pattern_count"] == 1
    assert pbh["detectable_sensor_pattern_count"] == 16
    assert pbh["admissible_actuator_patterns"] == [[1, 1, 1]]

    deployments, certificates, summary = enumerate_pbh_admissible_qi_deployments(
        case, pbh
    )
    assert len(certificates) == 8_748
    assert len(deployments) == EXPECTED_PBH_ADMISSIBLE_QI_COUNT == 2_736
    assert len({item.key for item in deployments}) == len(deployments)
    assert {item.eta for item in deployments} == {(1, 1, 1)}
    assert len({item.xi for item in deployments}) == 16
    assert min(item.total_cost for item in deployments) >= 0.0
    assert max(item.total_cost for item in deployments) == 35.0
    assert summary["pbh_admissible_qi_deployment_count"] == 2_736


def test_budget_envelope_uses_all_prior_certified_upper_bounds() -> None:
    raw = (
        _raw_point(14, 10.0, 10.0, 14.0),
        _raw_point(15, 10.0 + 1.0e-10, 10.0, 15.0),
        _raw_point(16, 9.0, 9.0, 16.0),
    )
    envelope = build_tolerance_audited_envelope(
        raw, (14, 15, 16), key_prefix="test"
    )

    assert [item["performance_h2"] for item in envelope] == [10.0, 10.0, 9.0]
    assert [item["winner_source_budget"] for item in envelope] == [14, 14, 16]
    assert all(item["certified"] for item in envelope)
    assert [item["eligible_candidate_count"] for item in envelope] == [1, 2, 3]


def test_dual_reductions_diagnostic_sets_only_the_declared_parameter(
    monkeypatch,
) -> None:
    case = make_joint_platoon_case(seed=23)
    problem = make_joint_problem(case, horizon=10)
    options = SolverOptions(architecture_budget=14.0)

    @dataclass
    class Params:
        DualReductions: int = 1

    @dataclass
    class Model:
        Params: Params

    @dataclass
    class Built:
        model: Model

    built = Built(Model(Params()))
    sentinel = object()
    monkeypatch.setattr("repro.model._build_model", lambda *_args: built)

    def fake_solve(_problem, _options, observed):
        assert observed is built
        assert observed.model.Params.DualReductions == 0
        return sentinel

    monkeypatch.setattr("repro.model._solve_built_model", fake_solve)
    assert solve_with_dual_reductions_disabled(problem, options) is sentinel


def test_fixed_qp_route_converts_the_prescribed_shell_before_final_solve(
    monkeypatch,
) -> None:
    case = make_joint_platoon_case(seed=23)
    problem = BPlusProblem(
        plant=case.plant,
        deployment=case.deployment,
        horizon=10,
        fixed_eta=(1, 1, 1),
        fixed_xi=(1, 1, 1, 1, 1, 1, 1),
        fixed_service_delays=platoon_dense_architecture(case.base),
    )
    options = SolverOptions(architecture_budget=35.0)

    class Params:
        Method = -1
        NumericFocus = 0
        FeasibilityTol = 1.0e-6
        OptimalityTol = 1.0e-6
        DualReductions = 1

    class Model:
        def __init__(self) -> None:
            self.Params = Params()
            self.converted = False

        def convertToFixed(self) -> None:
            self.converted = True

        def reset(self) -> None:
            raise AssertionError("the optimal shell should not be reset")

    class Built:
        def __init__(self) -> None:
            self.model = Model()

    class Shell:
        solution_count = 1
        status = "OPTIMAL"

    built = Built()
    final = object()
    calls = 0
    monkeypatch.setattr("repro.model._build_model", lambda *_args: built)

    def fake_solve(_problem, _options, observed):
        nonlocal calls
        calls += 1
        assert observed is built
        if calls == 1:
            assert not built.model.converted
            return Shell()
        assert built.model.converted
        assert built.model.Params.Method == 1
        return final

    monkeypatch.setattr("repro.model._solve_built_model", fake_solve)
    assert solve_fixed_continuous_qp(problem, options) is final
    assert calls == 2
