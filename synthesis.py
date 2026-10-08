"""Budgeted OF-SLS mixed-integer QP and fixed-deployment QP.

Gurobi indicator constraints enforce exact zeros: no artificial big-M bound.
The fixed problem uses ordinary equalities and contains no binary variables.
"""

from typing import Any

import gurobipy as gp
import numpy as np

from platoon import (
    SHAPES, Plant, architecture, delay_array, hardware_groups, menus,
    performance_blocks, propagation, qi_compatible, qi_forbidden_patterns,
    service_cost, service_entries, sls_residuals,
)


def tolerance(a: float, b: float) -> float:
    return max(5e-8, 1e-8 * max(1., abs(a), abs(b)))


def build_model(
    p: Plant, budget: float, horizon: int = 10, tiers: int = 2,
    eta: Any = None, xi: Any = None, delays: np.ndarray | None = None,
    qi: bool = False,
) -> tuple:
    model = gp.Model("platoon_ofsls")
    for name, value in {
        "OutputFlag": 0, "Seed": 23, "Threads": 1, "NumericFocus": 3,
        "FeasibilityTol": 1e-9, "OptimalityTol": 1e-9,
        "IntFeasTol": 1e-9, "MIPGap": 1e-9,
    }.items():
        model.setParam(name, value)
    fixed = eta is not None and xi is not None and delays is not None
    if fixed:
        model.Params.Method = 1
    response = {
        name: [model.addMVar(shape, lb=-gp.GRB.INFINITY, name=f"{name}_{t}") for t in range(horizon + 1)]
        for name, shape in SHAPES
    }
    for residual in sls_residuals(p, response):
        model.addConstr(residual == 0)
    hardware = {
        "eta": tuple(eta) if eta is not None else tuple(
            model.addVar(vtype=gp.GRB.BINARY, name=f"eta_{i}") for i in range(3)
        ),
        "xi": tuple(xi) if xi is not None else tuple(
            model.addVar(vtype=gp.GRB.BINARY, name=f"xi_{i}") for i in range(7)
        ),
    }

    def gate(enabled: Any, value: Any) -> None:
        if isinstance(enabled, gp.Var):
            model.addGenConstrIndicator(enabled, 0, value == 0)
        elif not enabled:
            model.addConstr(value == 0)

    for name, groups in zip(("eta", "xi"), hardware_groups(response)):
        for enabled, group in zip(hardware[name], groups):
            for block in group:
                gate(enabled, block)
    states, available = {}, {}
    cost = gp.quicksum((*hardware["eta"], *hardware["xi"]))
    if delays is None:
        for pair, options in menus(tiers).items():
            if pair[0] == pair[1]:
                continue
            for delay in options:
                states[*pair, delay] = model.addVar(
                    vtype=gp.GRB.BINARY, name=f"service_{pair[0]}_{pair[1]}_d{delay}"
                )
            model.addConstr(gp.quicksum(states[*pair, d] for d in options) == 1)
            cost += gp.quicksum((4 - d) * states[*pair, d] for d in options if np.isfinite(d))
            for lag in range(tiers + 1):
                available[*pair, lag] = model.addVar(
                    vtype=gp.GRB.BINARY, name=f"available_{pair[0]}_{pair[1]}_{lag}"
                )
                model.addConstr(available[*pair, lag] == gp.quicksum(states[*pair, d] for d in options if d <= lag))
    else:
        cost += service_cost(delays)
    model.addConstr(cost <= budget)
    for pair, deadline, coefficient in service_entries(response):
        if delays is not None:
            enabled = int(delays[pair] <= deadline)
        elif pair[0] == pair[1]:
            enabled = 1
        else:
            enabled = available.get((*pair, min(deadline, tiers)), 0)
        gate(enabled, coefficient)
    if qi:
        for pattern in qi_forbidden_patterns(propagation(p), tiers):
            literals = []
            for literal in sorted(pattern):
                if literal[0] == "service":
                    _, destination, source, delay = literal
                    literals.append(
                        states[destination, source, delay] if delays is None
                        else int(delays[destination, source] == delay)
                    )
                else:
                    literals.append(hardware[literal[0]][literal[1]])
            model.addConstr(gp.quicksum(literals) <= len(literals) - 1)
    objective = sum((block * block).sum() for block in performance_blocks(p, response))
    model.setObjective(objective, gp.GRB.MINIMIZE)
    model.update()
    return model, response, hardware, states


def audit(p: Plant, response: dict, choice: dict, budget: float, tiers: int) -> dict:
    """Evaluate equations, hardware zeros and service timing independently."""
    def norm(x: Any) -> float:
        return float(np.max(np.abs(x)))

    sls = max(norm(x) for x in sls_residuals(p, response))
    zero = 0.
    for mask, groups in zip((choice["eta"], choice["xi"]), hardware_groups(response)):
        for enabled, group in zip(mask, groups):
            if not enabled:
                zero = max(zero, *(norm(x) for x in group))
    D = delay_array(choice)
    for pair, deadline, coefficient in service_entries(response):
        if D[pair] > deadline:
            zero = max(zero, abs(float(coefficient)))
    valid_menu = all(D[i, j] in menus(tiers).get((i, j), (np.inf,)) for i in range(4) for j in range(4))
    finite = all(np.all(np.isfinite(x)) for blocks in response.values() for x in blocks)
    return {
        "sls_residual": sls, "zero_residual": zero,
        "certified": bool(finite and valid_menu and max(sls, zero) <= 2e-8 and choice["cost"] <= budget + 2e-8),
    }


def _read_solution(
    model: Any, response: dict, hardware: dict, states: dict,
    delays: Any, p: Plant, budget: float, tiers: int,
) -> dict:
    names = ("OPTIMAL", "INFEASIBLE", "INF_OR_UNBD", "NUMERIC", "TIME_LIMIT", "INTERRUPTED", "SUBOPTIMAL")
    result = {"status": next((name for name in names if getattr(gp.GRB, name) == model.Status), str(model.Status)),
              "certified": False, "runtime": float(model.Runtime)}
    if not model.SolCount:
        return result
    raw = [[x.X if isinstance(x, gp.Var) else x for x in hardware[name]] for name in ("eta", "xi")]
    discrete = [*raw[0], *raw[1], *(x.X for x in states.values())]
    integer_error = max(abs(x - round(x)) for x in discrete)
    if delays is None:
        delays = np.full((4, 4), np.inf)
        np.fill_diagonal(delays, 0)
        for (i, j, d), value in states.items():
            if value.X > .5:
                delays[i, j] = d
    choice = architecture(*(tuple(round(x) for x in row) for row in raw), delays)
    numeric = {name: [x.X for x in blocks] for name, blocks in response.items()}
    J = float(sum(np.sum(x * x) for x in performance_blocks(p, numeric)))
    check = audit(p, numeric, choice, budget, tiers)
    check["integer_residual"] = integer_error
    check["objective_error"] = abs(J - model.ObjVal)
    ok = check["certified"] and integer_error <= 1e-9 and check["objective_error"] <= 2e-8
    result.update(choice=choice, J=J, audit=check, certified=bool(ok and model.Status == gp.GRB.OPTIMAL))
    if model.IsMIP:
        result.update(bound=float(model.ObjBound), gap=float(model.MIPGap))
        result["certified"] &= model.MIPGap <= 1e-9 + 1e-15
    return result


def solve(
    p: Plant, budget: float, horizon: int = 10, tiers: int = 2,
    eta: Any = None, xi: Any = None, delays: np.ndarray | None = None,
    qi: bool = False,
) -> dict:
    """Solve once; certify an integer choice by independent fixed-QP synthesis."""
    model, response, hardware, states = build_model(p, budget, horizon, tiers, eta, xi, delays, qi)
    fixed = eta is not None and xi is not None and delays is not None
    try:
        model.optimize()
        if model.Status == gp.GRB.INF_OR_UNBD:
            model.Params.DualReductions = 0
            model.optimize()
        result = _read_solution(model, response, hardware, states, delays, p, budget, tiers)
        if fixed and not result["certified"]:
            # Redundant FIR equalities occasionally need the barrier path.
            attempts = [result]
            for presolve in (0, 1):
                model.reset()
                for name, value in {"DualReductions": 0, "Presolve": presolve, "Method": 2,
                                    "BarHomogeneous": 1, "Crossover": 1, "BarConvTol": 1e-10}.items():
                    model.setParam(name, value)
                model.optimize()
                attempts.append(_read_solution(model, response, hardware, states, delays, p, budget, tiers))
            certified = [x for x in attempts if x["certified"]]
            if certified:
                result = min(certified, key=lambda x: x["J"]).copy()
                result["certified"] = all(
                    abs(x["J"] - result["J"]) <= tolerance(x["J"], result["J"])
                    for x in certified
                )
            else:
                result = attempts[-1].copy()
            result["attempts"] = attempts
    finally:
        model.dispose()
    if not fixed and "choice" in result:
        choice = result["choice"]
        upper = solve(p, budget, horizon, tiers, choice["eta"], choice["xi"], delay_array(choice))
        result["fixed_qp"] = upper
        lower = result.get("bound")
        result["certified"] = bool(result["certified"] and upper["certified"] and lower is not None
                                   and abs(upper["J"] - lower) <= tolerance(upper["J"], lower))
        if upper["certified"]:
            result["incumbent_J"], result["J"] = result["J"], upper["J"]
    if qi and "choice" in result:
        choice = result["choice"]
        result["verified_qi"] = qi_compatible(delay_array(choice), choice["eta"], choice["xi"], propagation(p))
        result["certified"] &= result["verified_qi"]
    return result
