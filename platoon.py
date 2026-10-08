"""The paper's three-follower plant and finite deployment catalogue.

Delay matrices use [destination, source]; infinity means no service.
States are (spacing error, relative velocity, acceleration) per follower.
"""

from dataclasses import dataclass
from itertools import product
from typing import Any, Iterator

import numpy as np
from scipy.signal import cont2discrete


SENSOR_GROUPS = ((0,), (1, 2), (3,), (4, 5), (6,), (7, 8), (9,))
SENSOR_SITES = (0, 1, 1, 2, 2, 3, 3)
STATE_SITES = (1, 1, 1, 2, 2, 2, 3, 3, 3)
OUTPUT_SITES = (0, *STATE_SITES)
ACTUATOR_SITES = (1, 2, 3)
SHAPES = (("R", (9, 9)), ("M", (3, 9)), ("N", (9, 10)), ("L", (3, 10)))


@dataclass
class Plant:
    A: np.ndarray
    B1: np.ndarray
    B2: np.ndarray
    C1: np.ndarray
    D12: np.ndarray
    C2: np.ndarray
    D21: np.ndarray


def plant() -> Plant:
    """ZOH discretization at 0.1 s, headway 0.6 s, actuator lag 0.25 s."""
    local = np.array([[0., 1., -.6], [0., 0., -1.], [0., 0., -4.]])
    coupling = np.zeros((3, 3))
    coupling[1, 2] = 1.
    Ac = np.kron(np.eye(3), local) + np.kron(np.diag([1., 1.], -1), coupling)
    B2c = np.kron(np.eye(3), np.array([[0.], [0.], [4.]]))
    B1c = np.column_stack((np.eye(9)[:, 1], B2c))
    A, B, _, _, _ = cont2discrete(
        (Ac, np.hstack((B1c, B2c)), np.eye(9), np.zeros((9, 7))), .1
    )
    C1 = np.zeros((11, 9))
    C1[:3, ::3] = np.sqrt(10.) * np.eye(3)
    C1[3, [1, 4]] = (-1., 1.)
    C1[4, [4, 7]] = (-1., 1.)
    C1[5:8, 2::3] = np.sqrt(.5) * np.eye(3)
    D12 = np.vstack((np.zeros((8, 3)), np.sqrt(.1) * np.eye(3)))
    C2 = np.vstack((np.zeros((1, 9)), np.eye(9)))
    D21 = np.zeros((10, 4))
    D21[0, 0] = 1.
    return Plant(A, B[:, :4], B[:, 4:], C1, D12, C2, D21)


def menus(tiers: int = 2) -> dict[tuple[int, int], tuple[float, ...]]:
    """Local services are free; remote delays 1, 2, 3 cost 3, 2, 1."""
    result = {(i, i): (0.,) for i in range(4)}
    for destination in range(1, 4):
        for source in range(4):
            if source != destination:
                first = 2 if source and abs(destination - source) == 2 else 1
                result[destination, source] = (*range(first, tiers + 1), np.inf)
    return result


def service_cost(delays: np.ndarray) -> float:
    remote = delays[np.isfinite(delays) & (delays > 0)]
    return float(np.sum(4 - remote))


def architecture(eta: Any, xi: Any, delays: np.ndarray) -> dict:
    """JSON representation shared by the solvers and saved results."""
    return {
        "eta": [int(x) for x in eta], "xi": [int(x) for x in xi],
        "delays": [[int(d) if np.isfinite(d) else None for d in row] for row in delays],
        "cost": float(sum(eta) + sum(xi) + service_cost(delays)),
    }


def delay_array(choice: dict) -> np.ndarray:
    return np.array([[np.inf if d is None else d for d in row] for row in choice["delays"]])


def service_catalog(tiers: int = 2) -> Iterator[np.ndarray]:
    optional = [(pair, states) for pair, states in menus(tiers).items() if pair[0] != pair[1]]
    for states in product(*(states for _, states in optional)):
        delays = np.full((4, 4), np.inf)
        np.fill_diagonal(delays, 0)
        for (pair, _), delay in zip(optional, states):
            delays[pair] = delay
        yield delays


def sls_residuals(p: Plant, response: dict) -> Iterator[Any]:
    """Both OF-SLS affine identities, including the FIR terminal equations."""
    R, M, N, L = (response[name] for name in ("R", "M", "N", "L"))
    yield R[0]
    yield M[0]
    yield N[0]
    for t in range(len(R)):
        Rn, Mn, Nn = (x[t + 1] if t + 1 < len(R) else np.zeros(x[0].shape) for x in (R, M, N))
        impulse = np.eye(9) if t == 0 else np.zeros((9, 9))
        yield Rn - p.A @ R[t] - p.B2 @ M[t] - impulse
        yield Nn - p.A @ N[t] - p.B2 @ L[t]
        yield Rn - R[t] @ p.A - N[t] @ p.C2 - impulse
        yield Mn - M[t] @ p.A - L[t] @ p.C2


def performance_blocks(p: Plant, response: dict) -> Iterator[Any]:
    """Squared Frobenius norms sum to the paper's H2 performance objective."""
    for R, M, N, L in zip(*(response[name] for name in ("R", "M", "N", "L"))):
        yield p.C1 @ (R @ p.B1 + N @ p.D21) + p.D12 @ (M @ p.B1 + L @ p.D21)


def service_entries(response: dict) -> Iterator[tuple[tuple[int, int], int, Any]]:
    """Physical communication deadlines of each FIR coefficient (B+ timing)."""
    for name, destinations, sources in (
        ("R", STATE_SITES, STATE_SITES), ("M", ACTUATOR_SITES, STATE_SITES),
        ("N", STATE_SITES, OUTPUT_SITES), ("L", ACTUATOR_SITES, OUTPUT_SITES),
    ):
        first = {"R": 2, "M": 1, "N": 1, "L": 0}[name]
        for t in range(first, len(response[name])):
            deadline = t - 1 if name in ("R", "M") else t
            for row, destination in enumerate(destinations):
                for col, source in enumerate(sources):
                    yield (destination, source), deadline, response[name][t][row, col]


def hardware_groups(response: dict) -> tuple[list, list]:
    actuators = [[x[i, :] for name in ("M", "L") for x in response[name]] for i in range(3)]
    sensors = [[x[:, group] for name in ("N", "L") for x in response[name]] for group in SENSOR_GROUPS]
    return actuators, sensors


def propagation(p: Plant) -> np.ndarray:
    """First nonzero Markov lag from each actuator to each sensor package."""
    delays = np.full((7, 3), np.inf)
    power = np.eye(9)
    for lag in range(1, 11):
        markov = p.C2 @ power @ p.B2
        for sensor, group in enumerate(SENSOR_GROUPS):
            for actuator in range(3):
                if np.isinf(delays[sensor, actuator]) and np.max(np.abs(markov[list(group), actuator])) > 1e-12:
                    delays[sensor, actuator] = lag
        power = power @ p.A
    return delays


def qi_compatible(delays: np.ndarray, eta: Any, xi: Any, P: np.ndarray) -> bool:
    """D[i,j] <= D[i,r] + P[r,l] + D[l,j], restricted to installed devices."""
    D = delays[np.ix_(ACTUATOR_SITES, SENSOR_SITES)].copy()
    D[np.asarray(eta) == 0, :] = np.inf
    D[:, np.asarray(xi) == 0] = np.inf
    indirect = D[:, :, None, None] + P[None, :, :, None] + D[None, None, :, :]
    return bool(np.all(D <= np.min(indirect, axis=(1, 2))))


def admissible_hardware(p: Plant) -> tuple[list[tuple], list[tuple]]:
    """Exhaustive PBH tests on unstable/marginal modes, before service enumeration."""
    modes = [v for v in np.linalg.eigvals(p.A) if abs(v) >= 1 - 1e-9]
    actuators, sensors = [], []
    for eta in product((0, 1), repeat=3):
        B = p.B2[:, np.asarray(eta, dtype=bool)]
        if all(np.linalg.matrix_rank(np.hstack((v * np.eye(9) - p.A, B)), tol=1e-9) == 9 for v in modes):
            actuators.append(eta)
    for xi in product((0, 1), repeat=7):
        rows = [j for on, group in zip(xi, SENSOR_GROUPS) if on for j in group]
        C = p.C2[rows, :]
        if all(np.linalg.matrix_rank(np.vstack((v * np.eye(9) - p.A, C)), tol=1e-9) == 9 for v in modes):
            sensors.append(xi)
    return actuators, sensors


def qi_forbidden_patterns(P: np.ndarray, tiers: int = 2) -> list[frozenset]:
    """Finite no-good inequalities for QI; hardware and one-hot service literals."""
    menu = menus(tiers)
    patterns = set()
    for i, j, r, l in product(range(3), range(7), range(7), range(3)):
        if not np.isfinite(P[r, l]):
            continue
        pairs = ((i + 1, SENSOR_SITES[r]), (l + 1, SENSOR_SITES[j]), (i + 1, SENSOR_SITES[j]))
        for d1, d2, target in product(*(menu.get(pair, (np.inf,)) for pair in pairs)):
            if d1 + P[r, l] + d2 >= target:
                continue
            states = dict(zip(pairs, (d1, d2, target)))
            if any(states[pair] != d for pair, d in zip(pairs, (d1, d2, target))):
                continue
            literals = {("eta", i), ("eta", l), ("xi", j), ("xi", r)}
            literals.update(("service", *pair, d) for pair, d in states.items() if pair[0] != pair[1])
            patterns.add(frozenset(literals))
    minimal = []
    for pattern in sorted(patterns, key=lambda x: (len(x), repr(sorted(x)))):
        if not any(other < pattern for other in minimal):
            minimal.append(pattern)
    return minimal


def canonical_services(P: np.ndarray) -> dict[str, np.ndarray]:
    """Six standard communication patterns and their finite-menu QI repairs."""
    pf = {(i + 1, i) for i in range(3)}
    leader = {(i, 0) for i in range(1, 4)}
    bd = pf | {(i, i + 1) for i in range(1, 3)}
    tpf = {(i, j) for i in range(1, 4) for j in (i - 1, i - 2) if j >= 0}
    supports = {"PF": pf, "PLF": pf | leader, "BD": bd, "BDL": bd | leader, "TPF": tpf, "TPLF": tpf | leader}
    result = {}
    for name, support in supports.items():
        D = np.full((4, 4), np.inf)
        np.fill_diagonal(D, 0)
        for pair in support:
            D[pair] = min(menus()[pair])
        result[name] = D.copy()
        # Closure and menu projection can introduce further paths; iterate.
        while not qi_compatible(D, (1,) * 3, (1,) * 7, P):
            controller = D[np.ix_(ACTUATOR_SITES, SENSOR_SITES)]
            indirect = controller[:, :, None, None] + P[None, :, :, None] + controller[None, None, :, :]
            requested = np.minimum(controller, np.min(indirect, axis=(1, 2)))
            for i, site in enumerate(ACTUATOR_SITES):
                for source in range(4):
                    deadline = np.min(requested[i, np.array(SENSOR_SITES) == source])
                    D[site, source] = max(d for d in menus()[site, source] if d <= deadline)
        result[name + "+QI"] = D
    return result
