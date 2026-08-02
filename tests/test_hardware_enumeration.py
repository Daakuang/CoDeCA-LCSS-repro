import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gurobipy")

from repro.hardware_experiment import (
    PUBLICATION_HORIZONS,
    EnumerationRecord,
    HardwareChoice,
    budget_envelope,
    hardware_choices,
    run_hardware_experiment,
)


def _synthetic_record() -> EnumerationRecord:
    return EnumerationRecord(
        horizon=8,
        eta=(1, 0, 0),
        xi=(1, 0, 0),
        hardware_cost=2,
        service_delays=((0, 0, 0),) * 3,
        feasible=True,
        status="OPTIMAL",
        h2_objective=1.0,
        total_objective=1.0,
        architecture_cost=2.0,
        best_bound=1.0,
        mip_gap=0.0,
        runtime=0.0,
        solution_count=1,
        audit=None,
    )


@pytest.fixture(scope="session")
def hardware_t8_run(tmp_path_factory: pytest.TempPathFactory):
    output = tmp_path_factory.mktemp("hardware_selection")
    run = run_hardware_experiment(
        seed=23,
        threads=1,
        output=output,
        horizons=(8,),
        validation_horizon=8,
    )
    return output, run


def test_lexicographic_hardware_enumeration_has_64_unique_choices() -> None:
    choices = hardware_choices()

    assert len(choices) == 64
    assert len(set(choices)) == 64
    assert choices == tuple(sorted(choices))
    assert choices[0] == HardwareChoice(eta=(0, 0, 0), xi=(0, 0, 0))
    assert choices[-1] == HardwareChoice(eta=(1, 1, 1), xi=(1, 1, 1))
    assert choices[1] == HardwareChoice(eta=(0, 0, 0), xi=(0, 0, 1))


def test_publication_horizon_contract_is_frozen() -> None:
    assert PUBLICATION_HORIZONS == (8, 10, 12)


@pytest.mark.parametrize("tolerance", [True, -1.0, float("nan"), float("inf"), -float("inf")])
def test_budget_envelope_rejects_invalid_objective_tolerance(tolerance: object) -> None:
    with pytest.raises((TypeError, ValueError), match="objective_tolerance"):
        budget_envelope((_synthetic_record(),), objective_tolerance=tolerance)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "budgets",
    [(True,), (2.9,), (-1,), (2, 2)],
)
def test_budget_envelope_rejects_nonintegral_negative_or_duplicate_budgets(
    budgets: tuple[object, ...],
) -> None:
    with pytest.raises((TypeError, ValueError), match="budget|budgets"):
        budget_envelope((_synthetic_record(),), budgets=budgets)  # type: ignore[arg-type]


def test_budget_envelope_normalizes_numpy_integer_budgets() -> None:
    envelope = budget_envelope(
        (_synthetic_record(),), budgets=(np.int64(2), np.int32(3))
    )

    assert tuple(envelope) == (2, 3)
    assert all(type(gamma) is int for gamma in envelope)


@pytest.mark.parametrize("h2_objective", [None, float("nan"), float("inf"), -float("inf"), True])
def test_budget_envelope_rejects_invalid_h2_on_feasible_records(
    h2_objective: object,
) -> None:
    record = replace(_synthetic_record(), h2_objective=h2_objective)

    with pytest.raises((TypeError, ValueError), match="h2_objective"):
        budget_envelope((record,))


def test_budget_envelope_rejects_invalid_h2_on_solution_bearing_records() -> None:
    record = replace(
        _synthetic_record(), feasible=False, solution_count=1, h2_objective=None
    )

    with pytest.raises(ValueError, match="solution-bearing.*h2_objective"):
        budget_envelope((record,))


def test_t8_exhaustive_solve_records_every_case_without_nonfinite_json(
    hardware_t8_run,
) -> None:
    output, run = hardware_t8_run
    records = run.enumerations[8]

    assert len(records) == 64
    assert {(record.eta, record.xi) for record in records} == {
        (choice.eta, choice.xi) for choice in hardware_choices()
    }
    assert all(record.horizon == 8 for record in records)
    assert all(record.service_delays == ((0, 0, 0),) * 3 for record in records)
    assert any(record.feasible for record in records)
    assert any(not record.feasible for record in records)
    encoded = (output / "enumeration.json").read_text(encoding="utf-8")
    assert "NaN" not in encoded
    assert "Infinity" not in encoded
    decoded = json.loads(encoded)
    assert decoded["horizons"] == [8]
    assert len(decoded["records"]) == 64


def test_budget_envelope_uses_at_most_semantics_and_micp_matches_argmin(
    hardware_t8_run,
) -> None:
    _, run = hardware_t8_run
    envelope = run.envelopes[8]

    assert tuple(envelope) == (2, 3, 4, 5, 6)
    previous = float("inf")
    for gamma, point in envelope.items():
        assert point.gamma == gamma
        assert point.best_h2 is not None
        assert point.best_h2 <= previous + 1e-10
        assert point.argmin_choices
        assert all(choice.hardware_cost <= gamma for choice in point.argmin_choices)
        previous = point.best_h2

    assert tuple(run.validations) == (2, 4, 6)
    for gamma, validation in run.validations.items():
        assert validation.horizon == 8
        assert validation.gamma == gamma
        assert validation.objective_mismatch <= 1e-8
        assert validation.returned_choice_in_argmin
        assert validation.hardware_cost <= gamma
        assert validation.mip_gap is not None
        assert validation.mip_gap <= 1e-9
        assert validation.audit_certified


def test_runner_writes_five_stable_publication_artifact_schemas(hardware_t8_run) -> None:
    output, run = hardware_t8_run
    expected = {
        "case.json",
        "enumeration.json",
        "budget_summary.csv",
        "solver_manifest.json",
        "validation.json",
    }
    assert {path.name for path in output.iterdir()} == expected

    case = json.loads((output / "case.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "solver_manifest.json").read_text(encoding="utf-8"))
    validation = json.loads((output / "validation.json").read_text(encoding="utf-8"))
    assert case["schema_version"] == 1
    assert case["seed"] == 23
    assert case["matrix_sha256"]
    assert manifest["seed"] == 23
    assert manifest["threads"] == 1
    assert manifest["horizons"] == [8]
    assert manifest["fixed_qp_count"] == 64
    assert manifest["micp_count"] == 3
    assert manifest["settings"]["NumericFocus"] == 3
    assert validation["validation_horizon"] == 8
    assert validation["objective_tolerance"] == 1e-8
    assert validation["all_passed"] is True
    assert validation["validated_audit_maxima"]["ofsls"] <= 1e-8
    assert validation["validated_audit_maxima"]["relative_ofsls"] <= 1e-8
    assert validation["audit_maxima"]["response_max"] >= 1.0
    assert [
        item["pbh_controllability_min_rank"] for item in case["single_device_pbh"]
    ] == [
        3,
        2,
        3,
    ]
    assert [
        item["pbh_observability_min_rank"] for item in case["single_device_pbh"]
    ] == [
        3,
        2,
        3,
    ]
    assert case["single_device_pbh"][0][
        "pbh_controllability_worst_condition"
    ] == pytest.approx(16.15530443100607, rel=1e-12)
    assert case["single_device_pbh"][1][
        "pbh_observability_worst_condition"
    ] == pytest.approx(8.870030890816581e15, rel=1e-12)
    assert "controllability_rank" not in case["single_device_pbh"][0]

    with (output / "budget_summary.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == [
            "horizon",
            "gamma",
            "best_h2",
            "dense_h2",
            "performance_loss_percent",
            "argmin_count",
            "canonical_eta",
            "canonical_xi",
            "argmin_choices",
        ]
        rows = list(reader)
    assert len(rows) == 5
    assert {int(row["gamma"]) for row in rows} == {2, 3, 4, 5, 6}
    assert run.total_fixed_solve_runtime >= 0.0
    assert run.total_micp_runtime >= 0.0
