"""Joint canonical/RFD baseline adapters and QI publication gates."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from repro.cases import make_joint_platoon_case
from repro.deployment import ArchitectureChoice, architecture_cost
from repro.joint_baselines import (
    CANONICAL_BASELINE_NAMES,
    EXPECTED_LEGACY_RFD_POINT_COUNT,
    JointBaselineSpec,
    grouped_qi_validation,
    joint_canonical_baseline_specs,
    load_joint_rfd_baseline_specs,
    make_joint_baseline_problem,
    solve_joint_baselines,
)
from repro.model import BPlusProblem, BPlusSolveResult, SolverOptions, SolverSettings


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _settings(options: SolverOptions) -> SolverSettings:
    return SolverSettings(
        seed=options.seed,
        threads=options.threads,
        numeric_focus=options.numeric_focus,
        output_flag=options.output_flag,
        feasibility_tolerance=options.feasibility_tolerance,
        optimality_tolerance=options.optimality_tolerance,
        integer_feasibility_tolerance=options.integer_feasibility_tolerance,
        target_mip_gap=options.mip_gap,
        time_limit=options.time_limit,
        architecture_weight=options.architecture_weight,
        architecture_budget=options.architecture_budget,
        hardware_budget=options.hardware_budget,
        service_budget=options.service_budget,
    )


def _no_incumbent(options: SolverOptions, *, status: str) -> BPlusSolveResult:
    return BPlusSolveResult(
        status=status,
        responses=None,
        eta=None,
        xi=None,
        service_delays=None,
        service_availability=None,
        performance_objective=None,
        objective_value=None,
        architecture_cost=None,
        architecture_breakdown=None,
        best_bound=None,
        mip_gap=None,
        runtime=0.0,
        solution_count=0,
        settings=_settings(options),
    )


def _all_hardware_incumbent(
    problem: BPlusProblem,
    options: SolverOptions,
    *,
    performance: float,
) -> BPlusSolveResult:
    assert problem.fixed_service_delays is not None
    eta = (1,) * problem.plant.m
    xi = (1,) * problem.deployment.layout.sensor_device_count
    choice = ArchitectureChoice(eta, xi, problem.fixed_service_delays)
    cost = architecture_cost(problem.deployment, choice)
    return BPlusSolveResult(
        status="OPTIMAL",
        responses=None,
        eta=eta,
        xi=xi,
        service_delays=problem.fixed_service_delays,
        service_availability=None,
        performance_objective=performance,
        objective_value=performance,
        architecture_cost=cost.total,
        architecture_breakdown=cost,
        best_bound=performance,
        mip_gap=0.0,
        runtime=0.0,
        solution_count=1,
        settings=_settings(options),
    )


def _one_changed_rfd_pair() -> tuple[JointBaselineSpec, JointBaselineSpec]:
    case = make_joint_platoon_case(seed=23)
    specs, _ = load_joint_rfd_baseline_specs(case, REPOSITORY_ROOT)
    by_source: dict[str, dict[str, JointBaselineSpec]] = {}
    for spec in specs:
        by_source.setdefault(spec.source_key, {})[spec.layer] = spec
    for layers in by_source.values():
        raw = layers["raw"]
        repaired = layers["qi_repaired"]
        if raw.service_delays != repaired.service_delays:
            return raw, repaired
    raise AssertionError("the legacy RFD grid contains no changed QI repair")


def test_four_station_propagation_is_conservative_for_seven_groups() -> None:
    case = make_joint_platoon_case(seed=23)
    dense = joint_canonical_baseline_specs(
        case,
        names=("TPLF",),
        include_qi_repair=False,
    )[0]
    certificate = grouped_qi_validation(case, dense.service_delays)

    assert certificate.device_to_physical_block == (0, 1, 1, 2, 2, 3, 3)
    assert certificate.stationwise_conservative
    for device, block in enumerate(certificate.device_to_physical_block):
        for actuator in range(case.plant.m):
            aggregate = certificate.four_block_plant_delays[block][actuator]
            grouped = certificate.seven_group_plant_delays[device][actuator]
            assert aggregate is not None or grouped is None
            assert grouped is None or aggregate is None or aggregate <= grouped


def test_canonical_and_legacy_rfd_repair_specs_have_both_qi_certificates() -> None:
    case = make_joint_platoon_case(seed=23)
    canonical = joint_canonical_baseline_specs(case)
    assert len(canonical) == 2 * len(CANONICAL_BASELINE_NAMES)
    assert len({spec.key for spec in canonical}) == len(canonical)
    assert {spec.layer for spec in canonical} == {"raw", "qi_repaired"}

    rfd, provenance = load_joint_rfd_baseline_specs(case, REPOSITORY_ROOT)
    assert provenance["raw_point_count"] == EXPECTED_LEGACY_RFD_POINT_COUNT
    assert len(rfd) == 2 * EXPECTED_LEGACY_RFD_POINT_COUNT
    assert len({spec.key for spec in rfd}) == len(rfd)
    assert sum(spec.layer == "raw" for spec in rfd) == EXPECTED_LEGACY_RFD_POINT_COUNT
    assert all(
        spec.regularization is not None
        and spec.threshold is not None
        and spec.raw_support is not None
        for spec in rfd
    )

    repaired = tuple(
        spec for spec in (*canonical, *rfd) if spec.layer == "qi_repaired"
    )
    assert repaired
    assert all(spec.qi_validation.stationwise_conservative for spec in repaired)
    assert all(spec.qi_validation.four_block_all_hardware_qi for spec in repaired)
    assert all(spec.qi_validation.seven_group_all_hardware_qi for spec in repaired)


def test_baseline_problem_fixes_only_service_and_keeps_hardware_free() -> None:
    case = make_joint_platoon_case(seed=23)
    spec = joint_canonical_baseline_specs(
        case,
        names=("PF",),
        include_qi_repair=False,
    )[0]
    problem = make_joint_baseline_problem(case, spec, horizon=10)

    assert problem.fixed_service_delays == spec.service_delays
    assert problem.fixed_eta is None
    assert problem.fixed_xi is None
    assert problem.deployment.layout.sensor_device_count == 7
    assert problem.plant.m == 3


def test_raw_infeasible_and_repaired_feasible_remain_distinct() -> None:
    case = make_joint_platoon_case(seed=23)
    raw, repaired = _one_changed_rfd_pair()
    budget = 100.0

    def fake_solver(
        problem: BPlusProblem, options: SolverOptions | None
    ) -> BPlusSolveResult:
        assert options is not None
        assert options.architecture_budget == budget
        assert problem.fixed_eta is None and problem.fixed_xi is None
        if problem.fixed_service_delays == raw.service_delays:
            return _no_incumbent(options, status="INFEASIBLE")
        assert problem.fixed_service_delays == repaired.service_delays
        return _all_hardware_incumbent(problem, options, performance=7.5)

    outcomes = solve_joint_baselines(
        case,
        (raw, repaired),
        architecture_budget=budget,
        solver=fake_solver,
    )

    assert outcomes[0].spec.layer == "raw"
    assert outcomes[0].disposition == "infeasible"
    assert outcomes[0].performance_point is None
    assert outcomes[0].selected_hardware_qi is None
    assert outcomes[1].spec.layer == "qi_repaired"
    assert outcomes[1].disposition == "certified_feasible"
    assert outcomes[1].performance_point == 7.5
    assert outcomes[1].selected_hardware_qi is True


def test_selected_hardware_qi_recheck_is_a_strict_publication_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = make_joint_platoon_case(seed=23)
    repaired = joint_canonical_baseline_specs(
        case,
        names=("PF",),
    )[1]
    budget = 100.0

    def fake_solver(
        problem: BPlusProblem, options: SolverOptions | None
    ) -> BPlusSolveResult:
        assert options is not None
        return _all_hardware_incumbent(problem, options, performance=8.0)

    monkeypatch.setattr(
        "repro.joint_baselines.hardware_aware_qi_compatible",
        lambda *_args, **_kwargs: False,
    )
    with pytest.raises(RuntimeError, match="selected-hardware seven-group QI"):
        solve_joint_baselines(
            case,
            (repaired,),
            architecture_budget=budget,
            solver=fake_solver,
        )

    diagnostic = solve_joint_baselines(
        case,
        (repaired,),
        architecture_budget=budget,
        solver=fake_solver,
        strict_repaired_qi=False,
    )[0]
    assert diagnostic.selected_hardware_qi is False
    assert not diagnostic.publishable
    assert diagnostic.performance_point is None


def test_duplicate_service_matrices_share_solver_call_but_keep_identities() -> None:
    case = make_joint_platoon_case(seed=23)
    raw, repaired = joint_canonical_baseline_specs(case, names=("TPLF",))
    assert raw.service_delays == repaired.service_delays
    calls = 0

    def fake_solver(
        problem: BPlusProblem, options: SolverOptions | None
    ) -> BPlusSolveResult:
        nonlocal calls
        calls += 1
        assert options is not None
        return _all_hardware_incumbent(problem, options, performance=6.0)

    outcomes = solve_joint_baselines(
        case,
        (raw, repaired),
        architecture_budget=100.0,
        solver=fake_solver,
    )
    assert calls == 1
    assert tuple(outcome.spec.layer for outcome in outcomes) == (
        "raw",
        "qi_repaired",
    )
    assert all(outcome.performance_point == 6.0 for outcome in outcomes)


def test_solver_options_must_use_the_same_combined_budget() -> None:
    case = make_joint_platoon_case(seed=23)
    spec = joint_canonical_baseline_specs(
        case, names=("PF",), include_qi_repair=False
    )[0]
    wrong = SolverOptions(architecture_budget=9.0)
    with pytest.raises(ValueError, match="combined architecture budget"):
        solve_joint_baselines(
            case,
            (spec,),
            architecture_budget=10.0,
            options=wrong,
            solver=lambda _problem, options: _no_incumbent(
                options or wrong, status="INFEASIBLE"
            ),
        )

    split = replace(
        SolverOptions(),
        hardware_budget=5.0,
        service_budget=5.0,
    )
    with pytest.raises(ValueError, match="combined architecture budget"):
        solve_joint_baselines(
            case,
            (spec,),
            architecture_budget=10.0,
            options=split,
            solver=lambda _problem, options: _no_incumbent(
                options or split, status="INFEASIBLE"
            ),
        )
