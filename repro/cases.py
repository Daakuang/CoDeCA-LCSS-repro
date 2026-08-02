"""Frozen publication cases for the LCSS V2 revision.

The hardware-selection plant below ports only the scalar, three-node instance
of :func:`cat_sls.plants.make_chain_of_lqg`.  It intentionally does not import
that legacy builder: the random-number call order and numerical constants are
spelled out here so the revision case remains stable while retaining explicit
provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from numbers import Integral
from typing import Final

import numpy as np
from scipy.signal import cont2discrete

from .deployment import (
    DeploymentCostSpec,
    DeploymentSpec,
    DirectedServiceMenu,
    FixedHostLayout,
    ServiceCatalog,
    ServiceDelayMatrix,
)
from .model import GeneralizedPlant


HARDWARE_CASE_PROVENANCE: Final[str] = "cat_sls.plants.make_chain_of_lqg"
HARDWARE_SITE_COUNT: Final[int] = 3
PBH_RANK_TOLERANCE: Final[float] = 1.0e-10
HARDWARE_MATRIX_ORDER: Final[tuple[str, ...]] = (
    "A",
    "B1",
    "B2",
    "C1",
    "D11",
    "D12",
    "C2",
    "D21",
)

PLATOON_HEADWAY_SECONDS: Final[float] = 0.6
PLATOON_ACTUATOR_LAG_SECONDS: Final[float] = 0.25
PLATOON_SAMPLE_TIME_SECONDS: Final[float] = 0.1
PLATOON_FOLLOWER_COUNT: Final[int] = 3
PLATOON_SITE_COUNT: Final[int] = 4
PLATOON_COMMUNICATION_WEIGHT: Final[float] = 1.0e-3
PLATOON_MEASUREMENT_BLOCK_SIZES: Final[tuple[int, ...]] = (1, 3, 3, 3)
PLATOON_SENSOR_DEVICE_GROUPS: Final[tuple[tuple[int, ...], ...]] = (
    (0,),
    (1, 2),
    (3,),
    (4, 5),
    (6,),
    (7, 8),
    (9,),
)
PLATOON_SENSOR_DEVICE_LABELS: Final[tuple[str, ...]] = (
    "leader_alpha0",
    "follower_1_range_range_rate",
    "follower_1_acceleration",
    "follower_2_range_range_rate",
    "follower_2_acceleration",
    "follower_3_range_range_rate",
    "follower_3_acceleration",
)
PLATOON_SENSOR_DEVICE_SITES: Final[tuple[int, ...]] = (0, 1, 1, 2, 2, 3, 3)
PLATOON_ACTUATOR_DEVICE_LABELS: Final[tuple[str, ...]] = (
    "follower_1_actuator",
    "follower_2_actuator",
    "follower_3_actuator",
)
PLATOON_ARCHITECTURE_UNIT_SCALE: Final[float] = 6000.0
PLATOON_DISTURBANCE_LABELS: Final[tuple[str, ...]] = (
    "alpha_0",
    "d_1",
    "d_2",
    "d_3",
)
PLATOON_MATRIX_ORDER: Final[tuple[str, ...]] = (
    "continuous_A",
    "continuous_B1",
    "continuous_B2",
    "A",
    "B1",
    "B2",
    "C1",
    "D11",
    "D12",
    "C2",
    "D21",
)


def _immutable_finite_real_matrix(name: str, values: object) -> np.ndarray:
    """Validate before conversion so complex/object inputs cannot be truncated."""

    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real matrix") from error
    if (
        array.ndim != 2
        or not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
    ):
        raise ValueError(f"{name} must be a finite real matrix")
    normalized = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(normalized)):
        raise ValueError(f"{name} must be a finite real matrix")
    return np.frombuffer(
        normalized.tobytes(order="C"), dtype=np.float64
    ).reshape(normalized.shape)


@dataclass(frozen=True, slots=True)
class HardwareSelectionCase:
    """Three-node scalar LQG chain with fixed zero-delay communication."""

    plant: GeneralizedPlant
    deployment: DeploymentSpec
    fixed_service_delays: ServiceDelayMatrix
    seed: int
    provenance: str = HARDWARE_CASE_PROVENANCE


@dataclass(frozen=True, slots=True, eq=False)
class PlatoonCase:
    """Submitted three-follower constant-time-headway generalized plant.

    ``plant.p == 10`` counts scalar measured outputs.  The four physical
    measurement channels are ``(leader, follower 1, follower 2, follower 3)``
    with scalar block sizes ``(1, 3, 3, 3)``.  Since hardware is mandatory in
    this main experiment, the ten scalar ``xi`` entries are fixed to one; they
    are not interpreted as ten independently deployable sensor devices.

    The shared disturbance is ``w = (alpha_0, d_1, d_2, d_3)``.  ``D21``
    exposes ``alpha_0`` as the leader measurement and introduces no separate
    measurement-noise coordinates.
    """

    plant: GeneralizedPlant
    deployment: DeploymentSpec
    continuous_A: np.ndarray
    continuous_B1: np.ndarray
    continuous_B2: np.ndarray
    measurement_block_sizes: tuple[int, ...]
    disturbance_labels: tuple[str, ...]
    fixed_eta: tuple[int, ...]
    fixed_xi: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in ("continuous_A", "continuous_B1", "continuous_B2"):
            object.__setattr__(
                self,
                name,
                _immutable_finite_real_matrix(name, getattr(self, name)),
            )

    @property
    def measurement_channel_count(self) -> int:
        return len(self.measurement_block_sizes)


@dataclass(frozen=True, slots=True)
class JointPlatoonCase:
    """Same platoon plant with free grouped hardware and normalized costs."""

    base: PlatoonCase
    deployment: DeploymentSpec
    seed: int
    sensor_device_labels: tuple[str, ...] = PLATOON_SENSOR_DEVICE_LABELS
    actuator_device_labels: tuple[str, ...] = PLATOON_ACTUATOR_DEVICE_LABELS
    fixed_eta: None = None
    fixed_xi: None = None

    @property
    def plant(self) -> GeneralizedPlant:
        return self.base.plant

    @property
    def continuous_A(self) -> np.ndarray:
        return self.base.continuous_A

    @property
    def continuous_B1(self) -> np.ndarray:
        return self.base.continuous_B1

    @property
    def continuous_B2(self) -> np.ndarray:
        return self.base.continuous_B2

    @property
    def measurement_block_sizes(self) -> tuple[int, ...]:
        return self.base.measurement_block_sizes

    @property
    def disturbance_labels(self) -> tuple[str, ...]:
        return self.base.disturbance_labels

    @property
    def sensor_device_groups(self) -> tuple[tuple[int, ...], ...]:
        groups = self.deployment.layout.sensor_groups
        if groups is None:  # Defensive; normalized by FixedHostLayout.
            raise RuntimeError("joint platoon sensor groups were not normalized")
        return groups


@dataclass(frozen=True, slots=True)
class SingleDevicePBHDiagnostics:
    """Worst per-eigenvalue PBH rank and rectangular-pencil conditioning."""

    site: int
    pbh_controllability_min_rank: int
    pbh_observability_min_rank: int
    pbh_controllability_worst_condition: float | None
    pbh_observability_worst_condition: float | None


def _validated_seed(seed: object) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
        raise TypeError("seed must be a non-boolean integer")
    normalized = int(seed)
    if normalized < 0:
        raise ValueError("seed must be nonnegative")
    return normalized


def _normalize_spectral_radius(
    matrix: np.ndarray, *, target_radius: float
) -> np.ndarray:
    radius = float(np.max(np.abs(np.linalg.eigvals(matrix))))
    if radius <= 0.0 or radius <= target_radius:
        return matrix
    return matrix * (target_radius / radius)


def _hardware_generalized_plant(seed: int) -> GeneralizedPlant:
    """Port the exact legacy scalar-chain construction and RNG call order."""

    rng = np.random.default_rng(seed)
    A = 0.48 * np.eye(HARDWARE_SITE_COUNT, dtype=float)
    for left, right in ((0, 1), (1, 2)):
        gain_left_to_right = 0.07 * (0.85 + 0.3 * rng.random())
        gain_right_to_left = 0.07 * (0.85 + 0.3 * rng.random())
        A[left, right] += gain_left_to_right
        A[right, left] += gain_right_to_left
    A = _normalize_spectral_radius(A, target_radius=0.90)

    B2 = np.eye(HARDWARE_SITE_COUNT, dtype=float)
    for index in range(HARDWARE_SITE_COUNT):
        B2[index, index] = 0.85 + 0.15 * rng.random()

    identity = np.eye(HARDWARE_SITE_COUNT, dtype=float)
    zeros = np.zeros((HARDWARE_SITE_COUNT, HARDWARE_SITE_COUNT), dtype=float)
    return GeneralizedPlant(
        A=A,
        B1=np.hstack((identity, zeros)),
        B2=B2,
        C1=np.vstack((identity, zeros)),
        D11=np.zeros((2 * HARDWARE_SITE_COUNT, 2 * HARDWARE_SITE_COUNT)),
        D12=np.vstack((zeros, 0.50 * identity)),
        C2=identity,
        D21=np.hstack((zeros, 0.20 * identity)),
    )


def _hardware_deployment() -> tuple[DeploymentSpec, ServiceDelayMatrix]:
    sites = tuple(range(HARDWARE_SITE_COUNT))
    menus = tuple(
        DirectedServiceMenu(
            destination_site=destination,
            source_site=source,
            finite_delays=(0,),
            mandatory_delay=0,
        )
        for destination in sites
        for source in sites
    )
    deployment = DeploymentSpec(
        layout=FixedHostLayout(
            actuator_sites=sites,
            sensor_sites=sites,
            beta_sites=sites,
            state_block_sizes=(1, 1, 1),
            site_count=HARDWARE_SITE_COUNT,
        ),
        services=ServiceCatalog(
            site_count=HARDWARE_SITE_COUNT,
            directed_menus=menus,
        ),
        costs=DeploymentCostSpec(
            actuator_costs=np.ones(HARDWARE_SITE_COUNT),
            sensor_costs=np.ones(HARDWARE_SITE_COUNT),
            service_costs_by_delay=(
                np.zeros((HARDWARE_SITE_COUNT, HARDWARE_SITE_COUNT)),
            ),
        ),
    )
    fixed_services: ServiceDelayMatrix = tuple(
        tuple(0 for _ in sites) for _ in sites
    )
    return deployment, fixed_services


def make_hardware_selection_case(*, seed: int = 23) -> HardwareSelectionCase:
    """Return the deterministic secondary hardware-selection experiment."""

    normalized_seed = _validated_seed(seed)
    deployment, fixed_services = _hardware_deployment()
    return HardwareSelectionCase(
        plant=_hardware_generalized_plant(normalized_seed),
        deployment=deployment,
        fixed_service_delays=fixed_services,
        seed=normalized_seed,
    )


def hardware_case_matrix_sha256(case: HardwareSelectionCase) -> str:
    """Hash all generalized-plant matrices in a declared, endian-stable order."""

    if not isinstance(case, HardwareSelectionCase):
        raise TypeError("case must be a HardwareSelectionCase")
    digest = hashlib.sha256()
    for name in HARDWARE_MATRIX_ORDER:
        matrix = np.asarray(getattr(case.plant, name), dtype="<f8")
        digest.update(matrix.tobytes(order="C"))
    return digest.hexdigest()


def _finite_condition_or_none(values: tuple[float, ...]) -> float | None:
    worst = max(values)
    return worst if np.isfinite(worst) else None


def single_device_pbh_diagnostics(
    case: HardwareSelectionCase,
) -> tuple[SingleDevicePBHDiagnostics, ...]:
    """Evaluate true PBH pencils for every candidate actuator and sensor.

    Every eigenvalue returned by ``numpy.linalg.eigvals`` is scanned directly;
    repeated or nearly repeated eigenvalues need no ordering or deduplication
    because only the minimum rank and maximum condition are retained.  Ranks
    use the explicit absolute tolerance ``PBH_RANK_TOLERANCE = 1e-10``.
    Conditions are NumPy 2-norm condition numbers of the rectangular pencils
    ``[lambda I - A, b_i]`` and ``[lambda I - A; c_i]``.  A nonfinite worst
    condition is represented as ``None`` so publication JSON remains strict.
    """

    if not isinstance(case, HardwareSelectionCase):
        raise TypeError("case must be a HardwareSelectionCase")
    plant = case.plant
    if plant.m != plant.p:
        raise ValueError("single-device PBH diagnostics require paired device counts")
    identity = np.eye(plant.n, dtype=np.complex128)
    complex_A = np.asarray(plant.A, dtype=np.complex128)
    diagnostics: list[SingleDevicePBHDiagnostics] = []
    for site in range(plant.m):
        controllability_ranks: list[int] = []
        observability_ranks: list[int] = []
        controllability_conditions: list[float] = []
        observability_conditions: list[float] = []
        for eigenvalue in np.linalg.eigvals(plant.A):
            control_pencil = np.hstack(
                (eigenvalue * identity - complex_A, plant.B2[:, [site]])
            )
            observation_pencil = np.vstack(
                (eigenvalue * identity - complex_A, plant.C2[[site], :])
            )
            controllability_ranks.append(
                int(
                    np.linalg.matrix_rank(
                        control_pencil, tol=PBH_RANK_TOLERANCE
                    )
                )
            )
            observability_ranks.append(
                int(
                    np.linalg.matrix_rank(
                        observation_pencil, tol=PBH_RANK_TOLERANCE
                    )
                )
            )
            controllability_conditions.append(float(np.linalg.cond(control_pencil)))
            observability_conditions.append(float(np.linalg.cond(observation_pencil)))
        diagnostics.append(
            SingleDevicePBHDiagnostics(
                site=site,
                pbh_controllability_min_rank=min(controllability_ranks),
                pbh_observability_min_rank=min(observability_ranks),
                pbh_controllability_worst_condition=_finite_condition_or_none(
                    tuple(controllability_conditions)
                ),
                pbh_observability_worst_condition=_finite_condition_or_none(
                    tuple(observability_conditions)
                ),
            )
        )
    return tuple(diagnostics)


def _platoon_continuous_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the continuous dynamics stated in the submitted manuscript."""

    follower_count = PLATOON_FOLLOWER_COUNT
    local_dimension = 3
    inverse_lag = 1.0 / PLATOON_ACTUATOR_LAG_SECONDS
    local = np.array(
        [
            [0.0, 1.0, -PLATOON_HEADWAY_SECONDS],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -inverse_lag],
        ],
        dtype=float,
    )
    preceding_acceleration = np.zeros((local_dimension, local_dimension))
    preceding_acceleration[1, 2] = 1.0
    predecessor_shift = np.diag(np.ones(follower_count - 1), k=-1)
    A = np.kron(np.eye(follower_count), local) + np.kron(
        predecessor_shift, preceding_acceleration
    )

    local_actuator = np.array([[0.0], [0.0], [inverse_lag]])
    B2 = np.kron(np.eye(follower_count), local_actuator)
    B1 = np.zeros((local_dimension * follower_count, follower_count + 1))
    B1[:local_dimension, 0] = (0.0, 1.0, 0.0)
    for follower in range(follower_count):
        B1[local_dimension * follower + 2, follower + 1] = inverse_lag
    return A, B1, B2


def _platoon_performance_matrices() -> tuple[np.ndarray, np.ndarray]:
    """Encode the published quadratic weights as H2 output factors."""

    state_dimension = 3 * PLATOON_FOLLOWER_COUNT
    input_dimension = PLATOON_FOLLOWER_COUNT
    spacing = np.zeros((PLATOON_FOLLOWER_COUNT, state_dimension))
    spacing[:, 0:state_dimension:3] = np.sqrt(10.0) * np.eye(
        PLATOON_FOLLOWER_COUNT
    )
    relative_velocity_differences = np.zeros(
        (PLATOON_FOLLOWER_COUNT - 1, state_dimension)
    )
    for row in range(PLATOON_FOLLOWER_COUNT - 1):
        relative_velocity_differences[row, 1 + 3 * row] = -1.0
        relative_velocity_differences[row, 1 + 3 * (row + 1)] = 1.0
    acceleration = np.zeros((PLATOON_FOLLOWER_COUNT, state_dimension))
    acceleration[:, 2:state_dimension:3] = np.sqrt(0.5) * np.eye(
        PLATOON_FOLLOWER_COUNT
    )
    C1 = np.vstack(
        (
            spacing,
            relative_velocity_differences,
            acceleration,
            np.zeros((input_dimension, state_dimension)),
        )
    )
    D12 = np.vstack(
        (
            np.zeros((C1.shape[0] - input_dimension, input_dimension)),
            np.sqrt(0.1) * np.eye(input_dimension),
        )
    )
    return C1, D12


def _platoon_generalized_plant(
    A_c: np.ndarray, B1_c: np.ndarray, B2_c: np.ndarray
) -> GeneralizedPlant:
    state_dimension = A_c.shape[0]
    disturbance_dimension = B1_c.shape[1]
    input_dimension = B2_c.shape[1]
    augmented_B = np.hstack((B1_c, B2_c))
    A_d, B_d, _, _, _ = cont2discrete(
        (
            A_c,
            augmented_B,
            np.eye(state_dimension),
            np.zeros(
                (state_dimension, disturbance_dimension + input_dimension)
            ),
        ),
        PLATOON_SAMPLE_TIME_SECONDS,
        method="zoh",
    )
    C1, D12 = _platoon_performance_matrices()
    C2 = np.vstack((np.zeros((1, state_dimension)), np.eye(state_dimension)))
    D21 = np.zeros((1 + state_dimension, disturbance_dimension))
    D21[0, 0] = 1.0
    return GeneralizedPlant(
        A=A_d,
        B1=B_d[:, :disturbance_dimension],
        B2=B_d[:, disturbance_dimension:],
        C1=C1,
        D11=np.zeros((C1.shape[0], disturbance_dimension)),
        D12=D12,
        C2=C2,
        D21=D21,
    )


def _validated_service_horizon(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("service_horizon must be a non-boolean integer")
    if value not in (2, 3):
        raise ValueError("service_horizon must equal 2 or 3")
    return value


def _platoon_deployment(*, service_horizon: int = 2) -> DeploymentSpec:
    latency_horizon = _validated_service_horizon(service_horizon)
    menus: list[DirectedServiceMenu] = [
        DirectedServiceMenu(
            destination_site=site,
            source_site=site,
            finite_delays=(0,),
            mandatory_delay=0,
        )
        for site in range(PLATOON_SITE_COUNT)
    ]
    for destination in range(1, PLATOON_SITE_COUNT):
        menus.append(
            DirectedServiceMenu(
                destination_site=destination,
                source_site=0,
                finite_delays=tuple(range(1, latency_horizon + 1)),
            )
        )
    for destination in range(1, PLATOON_SITE_COUNT):
        for source in range(1, PLATOON_SITE_COUNT):
            if destination == source:
                continue
            separation = abs(destination - source)
            menus.append(
                DirectedServiceMenu(
                    destination_site=destination,
                    source_site=source,
                    finite_delays=(
                        tuple(range(1, latency_horizon + 1))
                        if separation == 1
                        else tuple(range(2, latency_horizon + 1))
                    ),
                )
            )

    cost_layers: list[np.ndarray] = []
    for delay in range(latency_horizon + 1):
        layer = np.zeros((PLATOON_SITE_COUNT, PLATOON_SITE_COUNT))
        if delay > 0:
            unscaled_cost = (
                PLATOON_COMMUNICATION_WEIGHT / (1.0 + delay)
                if delay <= 2
                else 1.0 / PLATOON_ARCHITECTURE_UNIT_SCALE
            )
            layer[1:, :] = unscaled_cost
            for site in range(1, PLATOON_SITE_COUNT):
                layer[site, site] = 0.0
        cost_layers.append(layer)

    return DeploymentSpec(
        layout=FixedHostLayout(
            actuator_sites=(1, 2, 3),
            # Scalar-output hosts; this repeats each follower channel host for
            # its three state coordinates without creating extra devices.
            sensor_sites=(0, 1, 1, 1, 2, 2, 2, 3, 3, 3),
            beta_sites=(1, 2, 3),
            state_block_sizes=(3, 3, 3),
            site_count=PLATOON_SITE_COUNT,
        ),
        services=ServiceCatalog(
            site_count=PLATOON_SITE_COUNT,
            directed_menus=tuple(menus),
        ),
        costs=DeploymentCostSpec(
            actuator_costs=np.zeros(PLATOON_FOLLOWER_COUNT),
            sensor_costs=np.zeros(sum(PLATOON_MEASUREMENT_BLOCK_SIZES)),
            service_costs_by_delay=tuple(cost_layers),
        ),
    )


def _joint_platoon_deployment(base: DeploymentSpec) -> DeploymentSpec:
    """Return the same fixed hosts/menu with seven device groups and unit costs."""

    if not isinstance(base, DeploymentSpec):
        raise TypeError("base must be a DeploymentSpec")
    layout = base.layout
    if PLATOON_COMMUNICATION_WEIGHT * PLATOON_ARCHITECTURE_UNIT_SCALE != 6.0:
        raise RuntimeError("platoon architecture-unit scaling must equal six")
    service_cost_layers: list[np.ndarray] = []
    for delay in range(len(base.costs.service_costs_by_delay)):
        layer = np.zeros((layout.site_count, layout.site_count))
        if delay > 0:
            layer = (
                np.asarray(base.costs.service_costs_by_delay[delay], dtype=float)
                * PLATOON_ARCHITECTURE_UNIT_SCALE
            )
            for site in range(1, layout.site_count):
                layer[site, site] = 0.0
        service_cost_layers.append(layer)
    return DeploymentSpec(
        layout=FixedHostLayout(
            actuator_sites=layout.actuator_sites,
            sensor_sites=layout.sensor_sites,
            beta_sites=layout.beta_sites,
            state_block_sizes=layout.state_block_sizes,
            site_count=layout.site_count,
            sensor_groups=PLATOON_SENSOR_DEVICE_GROUPS,
        ),
        services=base.services,
        costs=DeploymentCostSpec(
            actuator_costs=np.ones(PLATOON_FOLLOWER_COUNT),
            sensor_costs=np.ones(len(PLATOON_SENSOR_DEVICE_GROUPS)),
            service_costs_by_delay=tuple(service_cost_layers),
        ),
    )


def make_platoon_case(*, service_horizon: int = 2) -> PlatoonCase:
    """Return the exact deterministic submitted platoon in the B+ host model."""

    A_c, B1_c, B2_c = _platoon_continuous_matrices()
    return PlatoonCase(
        plant=_platoon_generalized_plant(A_c, B1_c, B2_c),
        deployment=_platoon_deployment(service_horizon=service_horizon),
        continuous_A=A_c,
        continuous_B1=B1_c,
        continuous_B2=B2_c,
        measurement_block_sizes=PLATOON_MEASUREMENT_BLOCK_SIZES,
        disturbance_labels=PLATOON_DISTURBANCE_LABELS,
        fixed_eta=(1, 1, 1),
        fixed_xi=(1,) * sum(PLATOON_MEASUREMENT_BLOCK_SIZES),
    )


def make_joint_platoon_case(
    *, seed: int = 23, service_horizon: int = 2
) -> JointPlatoonCase:
    """Return the submitted platoon with free grouped hardware and services."""

    normalized_seed = _validated_seed(seed)
    base = make_platoon_case(service_horizon=service_horizon)
    joint = JointPlatoonCase(
        base=base,
        deployment=_joint_platoon_deployment(base.deployment),
        seed=normalized_seed,
    )
    if joint.deployment.layout.sensor_device_sites != PLATOON_SENSOR_DEVICE_SITES:
        raise RuntimeError("joint platoon sensor-device hosts are inconsistent")
    return joint


def platoon_case_matrix_sha256(case: PlatoonCase | JointPlatoonCase) -> str:
    """Hash continuous and generalized-plant matrices in a declared order."""

    if not isinstance(case, (PlatoonCase, JointPlatoonCase)):
        raise TypeError("case must be a PlatoonCase or JointPlatoonCase")
    digest = hashlib.sha256()
    for name in PLATOON_MATRIX_ORDER:
        matrix = (
            getattr(case, name)
            if name.startswith("continuous_")
            else getattr(case.plant, name)
        )
        digest.update(np.asarray(matrix, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()
