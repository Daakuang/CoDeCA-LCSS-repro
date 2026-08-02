"""Auditable native Gurobi model for the finite B+ OF-SLS co-design class.

The generalized plant is restricted to ``D22 = 0``.  Raw response arrays use
literal lag indices, exactly as in :mod:`repro.realization`: ``R/M/N`` have
zero index padding, ``R[1] = I``, and ``L[0]`` is a genuine feedthrough term.
No big-M support constraints or legacy acceleration cuts are used.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Any, Final, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .deployment import (
    ArchitectureChoice,
    ArchitectureCostBreakdown,
    DeploymentSpec,
    ServiceDelayMatrix,
    architecture_cost,
)
from .realization import FIRResponses, service_lag_for_raw


FloatArray: TypeAlias = NDArray[np.float64]
ResponseBlock: TypeAlias = Literal["R", "M", "N", "L"]
DEFAULT_SEED: Final[int] = 23
DEFAULT_THREADS: Final[int] = 1


def _immutable_real_matrix(name: str, values: ArrayLike) -> FloatArray:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite real values") from error
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise ValueError(f"{name} must contain finite real values")
    normalized = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(normalized)):
        raise ValueError(f"{name} must contain finite real values")
    return np.frombuffer(normalized.tobytes(order="C"), dtype=np.float64).reshape(
        normalized.shape
    )


def _positive_integer(name: str, value: object, *, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-boolean integer")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return normalized


def _finite_nonnegative(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a finite nonnegative real scalar")
    try:
        normalized = float(cast(Any, value))
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a finite nonnegative real scalar") from error
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative real scalar")
    return normalized


@dataclass(frozen=True, slots=True, eq=False)
class GeneralizedPlant:
    """Immutable real generalized plant with zero ``D22``.

    The disturbance ``w`` is shared by the state and measurement channels:
    ``x+ = A x + B1 w + B2 u`` and ``y = C2 x + D21 w``.  Supplying ``D22``
    is optional, but any supplied matrix must have shape ``(p, m)`` and be
    identically zero within exact floating-point comparison.
    """

    A: FloatArray
    B2: FloatArray
    C2: FloatArray
    B1: FloatArray
    C1: FloatArray
    D12: FloatArray
    D21: FloatArray
    D11: FloatArray
    D22: FloatArray | None = None

    def __post_init__(self) -> None:
        for name in ("A", "B2", "C2", "B1", "C1", "D12", "D21", "D11"):
            object.__setattr__(
                self,
                name,
                _immutable_real_matrix(name, cast(ArrayLike, getattr(self, name))),
            )

        n = self.A.shape[0]
        if self.A.shape != (n, n):
            raise ValueError("A must be square")
        m = self.B2.shape[1]
        p = self.C2.shape[0]
        nw = self.B1.shape[1]
        nz = self.C1.shape[0]
        expected = {
            "B2": (n, m),
            "C2": (p, n),
            "B1": (n, nw),
            "C1": (nz, n),
            "D12": (nz, m),
            "D21": (p, nw),
            "D11": (nz, nw),
        }
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(
                    f"{name} has shape {getattr(self, name).shape}, expected {shape}"
                )

        if self.D22 is not None:
            d22 = _immutable_real_matrix("D22", self.D22)
            if d22.shape != (p, m):
                raise ValueError(f"D22 has shape {d22.shape}, expected {(p, m)}")
            if np.any(d22 != 0.0):
                raise ValueError("D22 must be zero for the B+ reference model")
            object.__setattr__(self, "D22", d22)

    @property
    def n(self) -> int:
        return int(self.A.shape[0])

    @property
    def m(self) -> int:
        return int(self.B2.shape[1])

    @property
    def p(self) -> int:
        return int(self.C2.shape[0])

    @property
    def nw(self) -> int:
        return int(self.B1.shape[1])

    @property
    def nz(self) -> int:
        return int(self.C1.shape[0])


@dataclass(frozen=True, slots=True)
class ResponseFix:
    """Optional exact raw-coefficient constraint for audit fixtures/fixed QPs."""

    block: ResponseBlock
    lag: int
    row: int
    column: int
    value: float

    def __post_init__(self) -> None:
        if self.block not in ("R", "M", "N", "L"):
            raise ValueError(f"unknown response block {self.block!r}")
        object.__setattr__(self, "lag", _positive_integer("lag", self.lag, minimum=0))
        object.__setattr__(self, "row", _positive_integer("row", self.row, minimum=0))
        object.__setattr__(
            self, "column", _positive_integer("column", self.column, minimum=0)
        )
        try:
            value = float(self.value)
        except (TypeError, ValueError) as error:
            raise TypeError("value must be a finite real scalar") from error
        if not math.isfinite(value):
            raise ValueError("value must be a finite real scalar")
        object.__setattr__(self, "value", value)


def _fixed_binary_vector(
    name: str, values: tuple[int, ...] | None, expected_length: int
) -> tuple[int, ...] | None:
    if values is None:
        return None
    try:
        candidates = tuple(values)
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable of binary values") from error
    if len(candidates) != expected_length:
        raise ValueError(f"{name} length must be {expected_length}")
    normalized: list[int] = []
    for index, value in enumerate(candidates):
        if isinstance(value, (bool, np.bool_)):
            normalized.append(int(value))
        elif isinstance(value, Integral) and int(value) in (0, 1):
            normalized.append(int(value))
        else:
            raise ValueError(f"{name}[{index}] must equal 0 or 1")
    return tuple(normalized)


def _fixed_service_matrix(
    values: ServiceDelayMatrix | None, site_count: int
) -> ServiceDelayMatrix | None:
    if values is None:
        return None
    try:
        rows = tuple(tuple(row) for row in values)
    except TypeError as error:
        raise TypeError("fixed_service_delays must be a two-dimensional iterable") from error
    if len(rows) != site_count or any(len(row) != site_count for row in rows):
        raise ValueError(
            "fixed_service_delays must have shape (site_count, site_count)"
        )
    normalized: list[tuple[int | None, ...]] = []
    for destination, row in enumerate(rows):
        normalized_row: list[int | None] = []
        for source, delay in enumerate(row):
            if delay is None:
                normalized_row.append(None)
                continue
            if isinstance(delay, (bool, np.bool_)) or not isinstance(delay, Integral):
                raise TypeError(
                    f"fixed_service_delays[{destination}][{source}] must be "
                    "a nonnegative integer or None"
                )
            normalized_delay = int(delay)
            if normalized_delay < 0:
                raise ValueError("fixed service delays must be nonnegative")
            normalized_row.append(normalized_delay)
        normalized.append(tuple(normalized_row))
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class BPlusProblem:
    """Validated finite-horizon deployment-response optimization instance."""

    plant: GeneralizedPlant
    deployment: DeploymentSpec
    horizon: int
    fixed_eta: tuple[int, ...] | None = None
    fixed_xi: tuple[int, ...] | None = None
    fixed_service_delays: ServiceDelayMatrix | None = None
    response_fixes: tuple[ResponseFix, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.plant, GeneralizedPlant):
            raise TypeError("plant must be a GeneralizedPlant")
        if not isinstance(self.deployment, DeploymentSpec):
            raise TypeError("deployment must be a DeploymentSpec")
        object.__setattr__(
            self, "horizon", _positive_integer("horizon", self.horizon, minimum=2)
        )
        layout = self.deployment.layout
        if layout.beta_dimension != self.plant.n:
            raise ValueError("hosted beta dimension must equal the plant state dimension")
        if len(layout.actuator_sites) != self.plant.m:
            raise ValueError("actuator host count must equal the plant input dimension")
        if len(layout.sensor_sites) != self.plant.p:
            raise ValueError("sensor host count must equal the plant output dimension")

        fixed_eta = _fixed_binary_vector("fixed_eta", self.fixed_eta, self.plant.m)
        fixed_xi = _fixed_binary_vector(
            "fixed_xi", self.fixed_xi, layout.sensor_device_count
        )
        fixed_delays = _fixed_service_matrix(
            self.fixed_service_delays, layout.site_count
        )
        object.__setattr__(self, "fixed_eta", fixed_eta)
        object.__setattr__(self, "fixed_xi", fixed_xi)
        object.__setattr__(self, "fixed_service_delays", fixed_delays)
        if fixed_delays is not None:
            self.deployment.validate_choice(
                ArchitectureChoice(
                    eta=tuple(1 for _ in range(self.plant.m)),
                    xi=tuple(1 for _ in range(layout.sensor_device_count)),
                    service_delays=fixed_delays,
                )
            )

        try:
            fixes = tuple(self.response_fixes)
        except TypeError as error:
            raise TypeError("response_fixes must be an iterable") from error
        dimensions = {
            "R": (self.plant.n, self.plant.n),
            "M": (self.plant.m, self.plant.n),
            "N": (self.plant.n, self.plant.p),
            "L": (self.plant.m, self.plant.p),
        }
        seen: set[tuple[str, int, int, int]] = set()
        for fix in fixes:
            if not isinstance(fix, ResponseFix):
                raise TypeError("response_fixes must contain ResponseFix values")
            rows, columns = dimensions[fix.block]
            if fix.lag > self.horizon or fix.row >= rows or fix.column >= columns:
                raise ValueError(f"response fix {fix} is outside the raw response shape")
            key = (fix.block, fix.lag, fix.row, fix.column)
            if key in seen:
                raise ValueError(f"duplicate response fix for {key}")
            seen.add(key)
        object.__setattr__(self, "response_fixes", fixes)


@dataclass(frozen=True, slots=True)
class SolverOptions:
    """Deterministic publication settings and architecture tradeoff mode."""

    seed: int = DEFAULT_SEED
    threads: int = DEFAULT_THREADS
    numeric_focus: int = 3
    output_flag: int = 0
    feasibility_tolerance: float = 1.0e-9
    optimality_tolerance: float = 1.0e-9
    integer_feasibility_tolerance: float = 1.0e-9
    mip_gap: float = 1.0e-9
    time_limit: float | None = None
    architecture_weight: float = 0.0
    architecture_budget: float | None = None
    hardware_budget: float | None = None
    service_budget: float | None = None

    def __post_init__(self) -> None:
        seed = _positive_integer("seed", self.seed, minimum=0)
        threads = _positive_integer("threads", self.threads)
        if seed != DEFAULT_SEED:
            raise ValueError(f"publication Seed must equal {DEFAULT_SEED}")
        if threads != DEFAULT_THREADS:
            raise ValueError(f"publication Threads must equal {DEFAULT_THREADS}")
        if isinstance(self.numeric_focus, (bool, np.bool_)) or not isinstance(
            self.numeric_focus, Integral
        ):
            raise TypeError("numeric_focus must be a non-boolean integer")
        numeric_focus = int(self.numeric_focus)
        if not 0 <= numeric_focus <= 3:
            raise ValueError("numeric_focus must be between 0 and 3")
        if self.output_flag not in (0, 1):
            raise ValueError("output_flag must equal 0 or 1")
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "threads", threads)
        object.__setattr__(self, "numeric_focus", numeric_focus)
        for name in (
            "feasibility_tolerance",
            "optimality_tolerance",
            "integer_feasibility_tolerance",
            "mip_gap",
            "architecture_weight",
        ):
            object.__setattr__(
                self, name, _finite_nonnegative(name, getattr(self, name))
            )
        if self.feasibility_tolerance < 1.0e-9:
            raise ValueError("feasibility_tolerance must be at least 1e-9")
        if self.optimality_tolerance < 1.0e-9:
            raise ValueError("optimality_tolerance must be at least 1e-9")
        if self.integer_feasibility_tolerance < 1.0e-9:
            raise ValueError("integer_feasibility_tolerance must be at least 1e-9")
        if self.time_limit is not None:
            time_limit = _finite_nonnegative("time_limit", self.time_limit)
            if time_limit == 0.0:
                raise ValueError("time_limit must be positive when supplied")
            object.__setattr__(self, "time_limit", time_limit)
        if self.architecture_budget is not None:
            object.__setattr__(
                self,
                "architecture_budget",
                _finite_nonnegative("architecture_budget", self.architecture_budget),
            )
        if self.hardware_budget is not None:
            object.__setattr__(
                self,
                "hardware_budget",
                _finite_nonnegative("hardware_budget", self.hardware_budget),
            )
        if self.service_budget is not None:
            object.__setattr__(
                self,
                "service_budget",
                _finite_nonnegative("service_budget", self.service_budget),
            )
        if self.architecture_budget is not None and self.architecture_weight != 0.0:
            raise ValueError(
                "choose either architecture_weight scalarization or a hard "
                "architecture_budget"
            )
        if self.architecture_budget is not None and (
            self.hardware_budget is not None or self.service_budget is not None
        ):
            raise ValueError(
                "choose either a combined architecture_budget or separate "
                "hardware_budget/service_budget caps"
            )


@dataclass(frozen=True, slots=True)
class SolverSettings:
    seed: int
    threads: int
    numeric_focus: int
    output_flag: int
    feasibility_tolerance: float
    optimality_tolerance: float
    integer_feasibility_tolerance: float
    target_mip_gap: float
    time_limit: float | None
    architecture_weight: float
    architecture_budget: float | None
    hardware_budget: float | None
    service_budget: float | None


AvailabilityMatrix: TypeAlias = tuple[tuple[tuple[int, ...], ...], ...]


@dataclass(frozen=True, slots=True)
class BPlusSolveResult:
    status: str
    responses: FIRResponses | None
    eta: tuple[int, ...] | None
    xi: tuple[int, ...] | None
    service_delays: ServiceDelayMatrix | None
    service_availability: AvailabilityMatrix | None
    performance_objective: float | None
    objective_value: float | None
    architecture_cost: float | None
    architecture_breakdown: ArchitectureCostBreakdown | None
    best_bound: float | None
    mip_gap: float | None
    runtime: float
    solution_count: int
    settings: SolverSettings


@dataclass(frozen=True, slots=True)
class _ModelVariables:
    R: tuple[Any, ...]
    M: tuple[Any, ...]
    N: tuple[Any, ...]
    L: tuple[Any, ...]
    eta: Any
    xi: Any


@dataclass(frozen=True, slots=True)
class _BuiltModel:
    model: Any
    variables: _ModelVariables
    selectors: dict[tuple[int, int], dict[int | None, Any]]
    availability: dict[tuple[int, int], tuple[Any, ...]]
    max_latency: int
    performance_expression: Any
    architecture_expression: Any
    hardware_expression: Any
    service_expression: Any
    performance_residual_variables: tuple[tuple[int, int, int, Any], ...]


def _gurobi() -> Any:
    import gurobipy as gp

    return gp


def _scalar(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def _add_matrix_equality(model: Any, left: Any, right: Any, *, name: str) -> None:
    rows, columns = left.shape
    for row in range(rows):
        for column in range(columns):
            right_entry = right[row, column] if hasattr(right, "shape") else right
            model.addConstr(
                _scalar(left[row, column]) == _scalar(right_entry),
                name=f"{name}_{row}_{column}",
            )


def _coefficient_linear_expression(
    gp: Any,
    *,
    constant: float,
    terms: tuple[tuple[FloatArray, Any, FloatArray], ...],
    row: int,
    column: int,
) -> Any:
    expression = gp.LinExpr(float(constant))
    for left, variable, right in terms:
        for inner_row in np.flatnonzero(left[row, :]):
            left_value = float(left[row, inner_row])
            for inner_column in np.flatnonzero(right[:, column]):
                coefficient = left_value * float(right[inner_column, column])
                if coefficient != 0.0:
                    expression.add(
                        _scalar(variable[int(inner_row), int(inner_column)]),
                        coefficient,
                    )
    return expression


def _beta_block_for_coordinate(problem: BPlusProblem) -> tuple[int, ...]:
    blocks: list[int] = []
    for block, size in enumerate(problem.deployment.layout.state_block_sizes):
        blocks.extend(block for _ in range(size))
    return tuple(blocks)


def _add_zero_indicator(model: Any, binary: Any, coefficient: Any, name: str) -> None:
    model.addGenConstrIndicator(
        _scalar(binary), False, _scalar(coefficient) == 0.0, name=name
    )


def _status_name(gp: Any, status: int) -> str:
    names = {
        gp.GRB.LOADED: "LOADED",
        gp.GRB.OPTIMAL: "OPTIMAL",
        gp.GRB.INFEASIBLE: "INFEASIBLE",
        gp.GRB.INF_OR_UNBD: "INF_OR_UNBD",
        gp.GRB.UNBOUNDED: "UNBOUNDED",
        gp.GRB.CUTOFF: "CUTOFF",
        gp.GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
        gp.GRB.NODE_LIMIT: "NODE_LIMIT",
        gp.GRB.TIME_LIMIT: "TIME_LIMIT",
        gp.GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        gp.GRB.INTERRUPTED: "INTERRUPTED",
        gp.GRB.NUMERIC: "NUMERIC",
        gp.GRB.SUBOPTIMAL: "SUBOPTIMAL",
        gp.GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
        gp.GRB.WORK_LIMIT: "WORK_LIMIT",
        gp.GRB.MEM_LIMIT: "MEM_LIMIT",
    }
    return names.get(int(status), f"STATUS_{int(status)}")


def _build_model(
    problem: BPlusProblem,
    options: SolverOptions,
    *,
    residual_objective_variables: bool = False,
) -> _BuiltModel:
    if not isinstance(residual_objective_variables, bool):
        raise TypeError("residual_objective_variables must be bool")
    gp = _gurobi()
    model = gp.Model("bplus_deployment_ofsls")
    model.Params.Seed = options.seed
    model.Params.Threads = options.threads
    model.Params.NumericFocus = options.numeric_focus
    model.Params.OutputFlag = options.output_flag
    model.Params.FeasibilityTol = options.feasibility_tolerance
    model.Params.OptimalityTol = options.optimality_tolerance
    model.Params.IntFeasTol = options.integer_feasibility_tolerance
    model.Params.MIPGap = options.mip_gap
    if options.time_limit is not None:
        model.Params.TimeLimit = options.time_limit

    plant = problem.plant
    horizon = problem.horizon
    infinity = gp.GRB.INFINITY
    R = tuple(
        model.addMVar((plant.n, plant.n), lb=-infinity, name=f"R_{t}")
        for t in range(horizon + 1)
    )
    M = tuple(
        model.addMVar((plant.m, plant.n), lb=-infinity, name=f"M_{t}")
        for t in range(horizon + 1)
    )
    N = tuple(
        model.addMVar((plant.n, plant.p), lb=-infinity, name=f"N_{t}")
        for t in range(horizon + 1)
    )
    L = tuple(
        model.addMVar((plant.m, plant.p), lb=-infinity, name=f"L_{t}")
        for t in range(horizon + 1)
    )
    eta = model.addMVar((plant.m,), vtype=gp.GRB.BINARY, name="eta")
    sensor_device_count = problem.deployment.layout.sensor_device_count
    xi = model.addMVar((sensor_device_count,), vtype=gp.GRB.BINARY, name="xi")
    variables = _ModelVariables(R=R, M=M, N=N, L=L, eta=eta, xi=xi)

    zero_nn = np.zeros((plant.n, plant.n))
    zero_mn = np.zeros((plant.m, plant.n))
    zero_np = np.zeros((plant.n, plant.p))
    _add_matrix_equality(model, R[0], zero_nn, name="raw_R0_padding")
    _add_matrix_equality(model, M[0], zero_mn, name="raw_M0_padding")
    _add_matrix_equality(model, N[0], zero_np, name="raw_N0_padding")
    for t in range(horizon + 1):
        R_next = R[t + 1] if t < horizon else zero_nn
        M_next = M[t + 1] if t < horizon else zero_mn
        N_next = N[t + 1] if t < horizon else zero_np
        impulse = np.eye(plant.n) if t == 0 else zero_nn
        _add_matrix_equality(
            model,
            R_next - plant.A @ R[t] - plant.B2 @ M[t],
            impulse,
            name=f"ofsls_left_R_{t}",
        )
        _add_matrix_equality(
            model,
            N_next - plant.A @ N[t] - plant.B2 @ L[t],
            zero_np,
            name=f"ofsls_left_N_{t}",
        )
        _add_matrix_equality(
            model,
            R_next - R[t] @ plant.A - N[t] @ plant.C2,
            impulse,
            name=f"ofsls_right_R_{t}",
        )
        _add_matrix_equality(
            model,
            M_next - M[t] @ plant.A - L[t] @ plant.C2,
            zero_mn,
            name=f"ofsls_right_M_{t}",
        )

    if problem.fixed_eta is not None:
        for index, value in enumerate(problem.fixed_eta):
            model.addConstr(_scalar(eta[index]) == value, name=f"fixed_eta_{index}")
    if problem.fixed_xi is not None:
        for index, value in enumerate(problem.fixed_xi):
            model.addConstr(_scalar(xi[index]) == value, name=f"fixed_xi_{index}")

    for actuator in range(plant.m):
        for t in range(horizon + 1):
            for state in range(plant.n):
                _add_zero_indicator(
                    model,
                    eta[actuator],
                    M[t][actuator, state],
                    f"hardware_M_{t}_{actuator}_{state}",
                )
            for sensor in range(plant.p):
                _add_zero_indicator(
                    model,
                    eta[actuator],
                    L[t][actuator, sensor],
                    f"hardware_eta_L_{t}_{actuator}_{sensor}",
                )
    layout = problem.deployment.layout
    for sensor in range(plant.p):
        sensor_device = layout.sensor_device_for_output(sensor)
        for t in range(horizon + 1):
            for state in range(plant.n):
                _add_zero_indicator(
                    model,
                    xi[sensor_device],
                    N[t][state, sensor],
                    f"hardware_N_{t}_{state}_{sensor}",
                )
            for actuator in range(plant.m):
                _add_zero_indicator(
                    model,
                    xi[sensor_device],
                    L[t][actuator, sensor],
                    f"hardware_xi_L_{t}_{actuator}_{sensor}",
                )

    all_delays = [
        delay
        for menu in problem.deployment.services.directed_menus
        for delay in menu.finite_delays
    ]
    max_latency = max(all_delays, default=0)
    selectors: dict[tuple[int, int], dict[int | None, Any]] = {}
    availability: dict[tuple[int, int], tuple[Any, ...]] = {}
    for menu in problem.deployment.services.directed_menus:
        destination, source = menu.destination_site, menu.source_site
        pair = (destination, source)
        pair_selectors: dict[int | None, Any] = {
            None: model.addVar(vtype=gp.GRB.BINARY, name=f"service_{destination}_{source}_off")
        }
        for delay in menu.finite_delays:
            pair_selectors[delay] = model.addVar(
                vtype=gp.GRB.BINARY,
                name=f"service_{destination}_{source}_d{delay}",
            )
        model.addConstr(
            gp.quicksum(pair_selectors.values()) == 1,
            name=f"service_one_hot_{destination}_{source}",
        )
        if menu.mandatory_delay is not None:
            model.addConstr(
                pair_selectors[menu.mandatory_delay] == 1,
                name=f"service_mandatory_{destination}_{source}",
            )
        if problem.fixed_service_delays is not None:
            selected = problem.fixed_service_delays[destination][source]
            model.addConstr(
                pair_selectors[selected] == 1,
                name=f"service_fixed_{destination}_{source}",
            )

        pair_availability = tuple(
            model.addVar(
                vtype=gp.GRB.BINARY,
                name=f"s_{destination}_{source}_{lag}",
            )
            for lag in range(max_latency + 1)
        )
        for lag, available in enumerate(pair_availability):
            model.addConstr(
                available
                == gp.quicksum(
                    selector
                    for delay, selector in pair_selectors.items()
                    if delay is not None and delay <= lag
                ),
                name=f"service_cumulative_{destination}_{source}_{lag}",
            )
            if lag > 0:
                model.addConstr(
                    pair_availability[lag - 1] <= available,
                    name=f"service_monotone_{destination}_{source}_{lag}",
                )
        selectors[pair] = pair_selectors
        availability[pair] = pair_availability

    beta_blocks = _beta_block_for_coordinate(problem)

    def availability_var(destination: int, source: int, lag: int) -> Any | None:
        pair_values = availability.get((destination, source))
        if pair_values is None:
            return None
        return pair_values[min(lag, max_latency)]

    def gate(
        coefficient: Any,
        destination: int,
        source: int,
        service_lag: int,
        name: str,
    ) -> None:
        support = availability_var(destination, source, service_lag)
        if support is None:
            model.addConstr(_scalar(coefficient) == 0.0, name=f"{name}_forbidden")
        else:
            _add_zero_indicator(model, support, coefficient, name)

    def gate_raw(
        block: ResponseBlock,
        raw_lag: int,
        coefficient: Any,
        destination: int,
        source: int,
        name: str,
    ) -> None:
        deadline_index = service_lag_for_raw(block, raw_lag)
        if deadline_index is None:
            return
        gate(coefficient, destination, source, deadline_index, name)

    for t in range(horizon + 1):
        for actuator in range(plant.m):
            for sensor in range(plant.p):
                gate_raw(
                    "L",
                    t,
                    L[t][actuator, sensor],
                    layout.p_u(actuator),
                    layout.p_y(sensor),
                    f"service_L_{t}_{actuator}_{sensor}",
                )
    for t in range(1, horizon + 1):
        for actuator in range(plant.m):
            for state in range(plant.n):
                gate_raw(
                    "M",
                    t,
                    M[t][actuator, state],
                    layout.p_u(actuator),
                    layout.beta_host(beta_blocks[state]),
                    f"service_M_{t}_{actuator}_{state}",
                )
        for state in range(plant.n):
            for sensor in range(plant.p):
                gate_raw(
                    "N",
                    t,
                    N[t][state, sensor],
                    layout.beta_host(beta_blocks[state]),
                    layout.p_y(sensor),
                    f"service_N_{t}_{state}_{sensor}",
                )
    for t in range(1, horizon + 1):
        for row in range(plant.n):
            for column in range(plant.n):
                gate_raw(
                    "R",
                    t,
                    R[t][row, column],
                    layout.beta_host(beta_blocks[row]),
                    layout.beta_host(beta_blocks[column]),
                    f"service_R_{t}_{row}_{column}",
                )

    response_variables = {"R": R, "M": M, "N": N, "L": L}
    for index, fix in enumerate(problem.response_fixes):
        coefficient = response_variables[fix.block][fix.lag][fix.row, fix.column]
        model.addConstr(
            _scalar(coefficient) == fix.value,
            name=f"response_fix_{index}_{fix.block}_{fix.lag}_{fix.row}_{fix.column}",
        )

    hardware_expression = gp.LinExpr()
    for actuator, cost in enumerate(problem.deployment.costs.actuator_costs):
        hardware_expression.add(_scalar(eta[actuator]), float(cost))
    for sensor, cost in enumerate(problem.deployment.costs.sensor_costs):
        hardware_expression.add(_scalar(xi[sensor]), float(cost))
    service_expression = gp.LinExpr()
    for (destination, source), pair_selectors in selectors.items():
        for delay, selector in pair_selectors.items():
            if delay is not None:
                cost = float(
                    problem.deployment.costs.service_costs_by_delay[delay][
                        destination, source
                    ]
                )
                service_expression.add(selector, cost)
    architecture_expression = hardware_expression + service_expression
    if options.architecture_budget is not None:
        model.addConstr(
            architecture_expression <= options.architecture_budget,
            name="architecture_budget",
        )
    if options.hardware_budget is not None:
        model.addConstr(
            hardware_expression <= options.hardware_budget,
            name="hardware_budget",
        )
    if options.service_budget is not None:
        model.addConstr(
            service_expression <= options.service_budget,
            name="service_budget",
        )

    performance_expression = gp.QuadExpr()
    performance_residual_variables: list[tuple[int, int, int, Any]] = []
    for t in range(horizon + 1):
        terms = (
            (plant.C1, R[t], plant.B1),
            (plant.C1, N[t], plant.D21),
            (plant.D12, M[t], plant.B1),
            (plant.D12, L[t], plant.D21),
        )
        for row in range(plant.nz):
            for column in range(plant.nw):
                residual = _coefficient_linear_expression(
                    gp,
                    constant=float(plant.D11[row, column]) if t == 0 else 0.0,
                    terms=terms,
                    row=row,
                    column=column,
                )
                if residual_objective_variables:
                    residual_variable = model.addVar(
                        lb=-infinity,
                        name=f"h2_residual_{t}_{row}_{column}",
                    )
                    model.addConstr(
                        residual_variable == residual,
                        name=f"h2_residual_definition_{t}_{row}_{column}",
                    )
                    performance_residual_variables.append(
                        (t, row, column, residual_variable)
                    )
                    performance_expression += residual_variable * residual_variable
                else:
                    performance_expression += residual * residual
    model.setObjective(
        performance_expression
        + float(options.architecture_weight) * architecture_expression,
        gp.GRB.MINIMIZE,
    )
    return _BuiltModel(
        model=model,
        variables=variables,
        selectors=selectors,
        availability=availability,
        max_latency=max_latency,
        performance_expression=performance_expression,
        architecture_expression=architecture_expression,
        hardware_expression=hardware_expression,
        service_expression=service_expression,
        performance_residual_variables=tuple(performance_residual_variables),
    )


def _decode_binary(value: float, tolerance: float, *, name: str) -> int:
    nearest = int(round(value))
    if nearest not in (0, 1) or abs(value - nearest) > tolerance:
        raise RuntimeError(
            f"{name}={value!r} is not binary within declared tolerance {tolerance:g}"
        )
    return nearest


def _numeric_h2(plant: GeneralizedPlant, responses: FIRResponses) -> float:
    terms: list[float] = []
    for t in range(responses.R.shape[0]):
        direct = plant.D11 if t == 0 else np.zeros_like(plant.D11)
        x_response = responses.R[t] @ plant.B1 + responses.N[t] @ plant.D21
        u_response = responses.M[t] @ plant.B1 + responses.L[t] @ plant.D21
        impulse = direct + plant.C1 @ x_response + plant.D12 @ u_response
        terms.append(float(np.sum(impulse * impulse)))
    return float(math.fsum(terms))


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


def _solve_built_model(
    problem: BPlusProblem,
    solver_options: SolverOptions,
    built: _BuiltModel,
) -> BPlusSolveResult:
    """Optimize and decode one already-built native B+ model."""

    gp = _gurobi()
    built.model.optimize()
    status = _status_name(gp, int(built.model.Status))
    solution_count = int(built.model.SolCount)
    runtime = float(built.model.Runtime)
    best_bound = (
        float(built.model.ObjBound)
        if bool(built.model.IsMIP) and math.isfinite(float(built.model.ObjBound))
        else None
    )
    mip_gap = (
        float(built.model.MIPGap)
        if bool(built.model.IsMIP) and solution_count > 0
        else None
    )
    if solution_count == 0:
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
            best_bound=best_bound,
            mip_gap=mip_gap,
            runtime=runtime,
            solution_count=solution_count,
            settings=_settings(solver_options),
        )

    variables = built.variables
    responses = FIRResponses(
        R=np.stack([np.asarray(block.X, dtype=float) for block in variables.R]),
        M=np.stack([np.asarray(block.X, dtype=float) for block in variables.M]),
        N=np.stack([np.asarray(block.X, dtype=float) for block in variables.N]),
        L=np.stack([np.asarray(block.X, dtype=float) for block in variables.L]),
    )
    binary_tolerance = 10.0 * solver_options.integer_feasibility_tolerance
    eta = tuple(
        _decode_binary(float(value), binary_tolerance, name=f"eta[{index}]")
        for index, value in enumerate(np.asarray(variables.eta.X, dtype=float))
    )
    xi = tuple(
        _decode_binary(float(value), binary_tolerance, name=f"xi[{index}]")
        for index, value in enumerate(np.asarray(variables.xi.X, dtype=float))
    )
    site_count = problem.deployment.layout.site_count
    delay_rows: list[tuple[int | None, ...]] = []
    availability_rows: list[tuple[tuple[int, ...], ...]] = []
    for destination in range(site_count):
        delay_row: list[int | None] = []
        availability_row: list[tuple[int, ...]] = []
        for source in range(site_count):
            selectors = built.selectors.get((destination, source))
            if selectors is None:
                delay_row.append(None)
                availability_row.append(tuple(0 for _ in range(built.max_latency + 1)))
                continue
            decoded_selectors = {
                delay: _decode_binary(
                    float(selector.X),
                    binary_tolerance,
                    name=f"service[{destination},{source},{delay}]",
                )
                for delay, selector in selectors.items()
            }
            selected = [delay for delay, enabled in decoded_selectors.items() if enabled]
            if len(selected) != 1:
                raise RuntimeError(
                    f"service ({destination},{source}) did not decode to one menu item"
                )
            delay_row.append(selected[0])
            availability_row.append(
                tuple(
                    _decode_binary(
                        float(variable.X),
                        binary_tolerance,
                        name=f"s[{destination},{source},{lag}]",
                    )
                    for lag, variable in enumerate(
                        built.availability[(destination, source)]
                    )
                )
            )
        delay_rows.append(tuple(delay_row))
        availability_rows.append(tuple(availability_row))
    service_delays: ServiceDelayMatrix = tuple(delay_rows)
    service_availability: AvailabilityMatrix = tuple(availability_rows)
    choice = ArchitectureChoice(eta=eta, xi=xi, service_delays=service_delays)
    breakdown = architecture_cost(problem.deployment, choice)
    performance = _numeric_h2(problem.plant, responses)
    objective_value = float(built.model.ObjVal)
    recomputed_objective = performance + solver_options.architecture_weight * breakdown.total
    objective_tolerance = max(
        1.0e-8,
        10.0 * solver_options.optimality_tolerance * max(1.0, abs(objective_value)),
    )
    if abs(objective_value - recomputed_objective) > objective_tolerance:
        raise RuntimeError(
            "solver objective does not match the independently decoded response and "
            "architecture cost"
        )
    return BPlusSolveResult(
        status=status,
        responses=responses,
        eta=eta,
        xi=xi,
        service_delays=service_delays,
        service_availability=service_availability,
        performance_objective=performance,
        objective_value=objective_value,
        architecture_cost=breakdown.total,
        architecture_breakdown=breakdown,
        best_bound=best_bound,
        mip_gap=mip_gap,
        runtime=runtime,
        solution_count=solution_count,
        settings=_settings(solver_options),
    )


def solve_bplus(
    problem: BPlusProblem, options: SolverOptions | None = None
) -> BPlusSolveResult:
    """Build and solve one native B+ MIQP without suppressing solver errors."""

    if not isinstance(problem, BPlusProblem):
        raise TypeError("problem must be a BPlusProblem")
    solver_options = SolverOptions() if options is None else options
    if not isinstance(solver_options, SolverOptions):
        raise TypeError("options must be SolverOptions")
    built = _build_model(problem, solver_options)
    return _solve_built_model(problem, solver_options, built)
