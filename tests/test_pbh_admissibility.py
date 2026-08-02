"""PBH admissible hardware-set and admissible joint-QI count tests."""

from __future__ import annotations

from repro.cases import make_joint_platoon_case
from repro.joint_qi import (
    grouped_plant_propagation_delays,
    service_menu_qi_certificates,
    summarize_admissible_joint_qi,
)
from repro.pbh_admissibility import (
    detectability_certificate,
    enumerate_detectable_sensor_patterns,
    enumerate_stabilizable_actuator_patterns,
    stabilizability_certificate,
)
from repro.platoon_experiment import iter_service_architectures


def _selection_mask(selection: tuple[int, ...]) -> int:
    return sum(value << index for index, value in enumerate(selection))


def test_platoon_pbh_requires_all_actuators_and_three_position_velocity_packages() -> None:
    case = make_joint_platoon_case(seed=23)
    actuator_certificates = enumerate_stabilizable_actuator_patterns(
        case.plant.A, case.plant.B2
    )
    assert tuple(item.selection for item in actuator_certificates) == ((1, 1, 1),)
    actuator_mode = actuator_certificates[0].modes[0]
    assert actuator_mode.rank == actuator_mode.required_rank == case.plant.n
    assert actuator_mode.minimum_singular_value > actuator_mode.rank_tolerance

    sensor_certificates = enumerate_detectable_sensor_patterns(
        case.plant.A,
        case.plant.C2,
        case.sensor_device_groups,
    )
    observed_masks = tuple(_selection_mask(item.selection) for item in sensor_certificates)
    assert observed_masks == (
        42,
        43,
        46,
        47,
        58,
        59,
        62,
        63,
        106,
        107,
        110,
        111,
        122,
        123,
        126,
        127,
    )
    assert all(
        item.selection[1] == item.selection[3] == item.selection[5] == 1
        for item in sensor_certificates
    )
    assert all(
        mode.rank == case.plant.n
        and mode.minimum_singular_value > mode.rank_tolerance
        for item in sensor_certificates
        for mode in item.modes
    )


def test_pbh_evidence_records_rank_failure_and_full_singular_spectrum() -> None:
    case = make_joint_platoon_case(seed=23)
    missing_last_actuator = stabilizability_certificate(
        case.plant.A, case.plant.B2, (1, 1, 0)
    )
    assert not missing_last_actuator.admissible
    assert missing_last_actuator.modes[0].rank == case.plant.n - 1
    assert len(missing_last_actuator.modes[0].singular_values) == case.plant.n
    assert (
        missing_last_actuator.modes[0].minimum_singular_value
        < missing_last_actuator.modes[0].rank_tolerance
    )

    missing_middle_position_velocity = detectability_certificate(
        case.plant.A,
        case.plant.C2,
        case.sensor_device_groups,
        (0, 1, 0, 0, 0, 1, 0),
    )
    assert not missing_middle_position_velocity.admissible
    assert missing_middle_position_velocity.modes[0].rank < case.plant.n
    assert len(missing_middle_position_velocity.modes[0].singular_values) == case.plant.n


def test_pbh_admissible_joint_candidate_and_qi_counts_exclude_empty_hardware() -> None:
    case = make_joint_platoon_case(seed=23)
    sensor_certificates = enumerate_detectable_sensor_patterns(
        case.plant.A,
        case.plant.C2,
        case.sensor_device_groups,
    )
    propagation = grouped_plant_propagation_delays(
        case.plant.A,
        case.plant.B2,
        case.plant.C2,
        case.sensor_device_groups,
    )
    records = tuple(iter_service_architectures(case.base))
    qi_certificates = service_menu_qi_certificates(
        (record.service_delays for record in records),
        case.deployment.layout.actuator_sites,
        case.deployment.layout.sensor_device_sites,
        propagation,
    )
    summary = summarize_admissible_joint_qi(
        qi_certificates,
        (1, 1, 1),
        (item.selection for item in sensor_certificates),
    )

    assert summary.service_menu_count == 8748
    assert summary.admissible_sensor_pattern_count == 16
    assert summary.joint_candidate_count == 139_968
    assert summary.joint_qi_count == 2_736
    assert tuple(item.sensor_mask for item in summary.per_sensor_pattern) == (
        42,
        43,
        46,
        47,
        58,
        59,
        62,
        63,
        106,
        107,
        110,
        111,
        122,
        123,
        126,
        127,
    )
    assert tuple(
        item.qi_service_menu_count for item in summary.per_sensor_pattern
    ) == (243, 99, 243, 99, 243, 99, 243, 99, 243, 99, 243, 99, 243, 99, 243, 99)
