"""Native actuator/sensor/service regularization for design (RFD).

The SOCP proposes deployments. All reported performance values come from
unregularized fixed-deployment QPs, charged at their actual deployment cost.
"""

from collections import defaultdict
from typing import Any, Iterator

import cvxpy as cp
import numpy as np

from platoon import (
    SHAPES, Plant, architecture, delay_array, hardware_groups, menus,
    performance_blocks, propagation, qi_compatible, service_catalog,
    service_entries, sls_residuals,
)


LAMBDAS = (0., 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, .1, .3, 1., 3., 10., 30., 100., 300., 1000.)
THRESHOLDS = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, .1, 1., 5., 10.)


def group_norm(blocks: list) -> Any:
    return cp.norm(cp.hstack([cp.reshape(x, (x.size,), order="F") for x in blocks]), 2) if blocks else 0.


def build_rfd(p: Plant, horizon: int = 10) -> tuple:
    response = {name: [cp.Variable(shape) for _ in range(horizon + 1)] for name, shape in SHAPES}
    constraints = [residual == 0 for residual in sls_residuals(p, response)]
    groups = defaultdict(list)
    for pair, deadline, coefficient in service_entries(response):
        if min(menus().get(pair, (np.inf,))) > deadline:
            constraints.append(coefficient == 0)
        else:
            groups[pair].append((deadline, coefficient))
    penalty = sum(group_norm(group) for family in hardware_groups(response) for group in family)
    for pair, entries in groups.items():
        if pair[0] == pair[1]:
            continue
        delays = [d for d in menus()[pair] if np.isfinite(d)]
        penalty += (4 - delays[-1]) * group_norm([x for _, x in entries])
        for faster, slower in zip(delays[:-1], delays[1:]):
            penalty += (slower - faster) * group_norm([x for deadline, x in entries if deadline < slower])
    h2 = sum(cp.sum_squares(x) for x in performance_blocks(p, response))
    weight = cp.Parameter(nonneg=True, value=0.)
    return cp.Problem(cp.Minimize(h2 + weight * penalty), constraints), response, weight, h2, penalty


def decode(response: dict, threshold: float) -> dict:
    masks = []
    for family in hardware_groups(response):
        masks.append([int(np.linalg.norm(np.concatenate([x.ravel() for x in group])) > threshold) for group in family])
    active = defaultdict(list)
    for pair, deadline, coefficient in service_entries(response):
        if min(menus().get(pair, (np.inf,))) <= deadline and abs(float(coefficient)) > threshold:
            active[pair].append(deadline)
    delays = np.full((4, 4), np.inf)
    np.fill_diagonal(delays, 0)
    for pair, deadlines in active.items():
        eligible = [d for d in menus().get(pair, ()) if d <= min(deadlines)]
        if not eligible:
            raise RuntimeError(f"RFD coefficient requires an unavailable service: {pair}")
        delays[pair] = max(eligible)
    return architecture(*masks, delays)


def rfd_path(p: Plant) -> Iterator[dict]:
    problem, variables, weight, h2, penalty = build_rfd(p)
    for value in LAMBDAS:
        weight.value = value
        problem.solve(solver=cp.CLARABEL, warm_start=False, max_iter=500,
                      tol_gap_abs=1e-9, tol_gap_rel=1e-9, tol_feas=1e-9)
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"RFD lambda={value}: {problem.status}")
        response = {name: [np.asarray(x.value) for x in blocks] for name, blocks in variables.items()}
        if not all(np.all(np.isfinite(x)) for blocks in response.values() for x in blocks):
            raise RuntimeError("RFD returned nonfinite coefficients")
        padding = max(np.max(np.abs(response[name][0])) for name in ("R", "M", "N"))
        identity = np.max(np.abs(response["R"][1] - np.eye(9)))
        if max(padding, identity) > 2e-6:
            raise RuntimeError("RFD impulse/padding residual exceeds 2e-6")
        for name in ("R", "M", "N"):
            response[name][0].fill(0)
        response["R"][1] = np.eye(9)
        residual = max(float(np.max(np.abs(x))) for x in sls_residuals(p, response))
        if residual > 2e-6:
            raise RuntimeError("RFD affine residual exceeds 2e-6")
        for threshold in THRESHOLDS:
            yield {"lambda": value, "threshold": threshold, "status": problem.status,
                   "regularized_J": float(problem.value), "h2": float(h2.value),
                   "penalty": float(penalty.value), "residual": residual,
                   "choice": decode(response, threshold)}


def qi_completions(p: Plant, choices: list[dict]) -> list[dict | None]:
    """Least-cost QI completion of each native candidate at fixed hardware."""
    P = propagation(p)
    catalog = list(service_catalog())
    cache = {}
    completions = []
    for choice in choices:
        hardware = (tuple(choice["eta"]), tuple(choice["xi"]))
        if hardware not in cache:
            cache[hardware] = [D for D in catalog if qi_compatible(D, *hardware, P)]
        original = delay_array(choice)
        eligible = [architecture(*hardware, D) for D in cache[hardware] if np.all(D <= original)]
        completions.append(min(eligible, default=None, key=lambda item: (
            item["cost"], int(np.sum(delay_array(item) != original)),
            str(tuple(tuple(row) for row in item["delays"])),
        )))
    return completions
