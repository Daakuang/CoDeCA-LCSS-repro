"""The paper's experiment grid. Each finished solve is checkpointed to JSON."""

import json
import os
from pathlib import Path
from typing import Iterator

import numpy as np

from platoon import (
    admissible_hardware, canonical_services, delay_array, menus, plant,
    propagation, qi_compatible, service_catalog,
)
from synthesis import solve


def save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def jobs(section: str, p) -> Iterator[tuple[str, dict, dict]]:
    """Yield (result key, labels, solver arguments); no computation on import."""
    budgets = range(14, 36)
    if section in ("all", "codesign"):
        dense = np.full((4, 4), np.inf)
        for pair, options in menus().items():
            dense[pair] = min(options)
        yield "dense", {"family": "dense"}, dict(budget=35, eta=(1,) * 3, xi=(1,) * 7, delays=dense)
        for budget in budgets:
            yield f"codesign:{budget}", {"family": "codesign", "budget": budget}, dict(budget=budget)
    if section in ("all", "qi"):
        eta_patterns, xi_patterns = admissible_hardware(p)
        P = propagation(p)
        count = 0
        for delays in service_catalog():
            for eta in eta_patterns:
                for xi in xi_patterns:
                    if qi_compatible(delays, eta, xi, P):
                        yield f"qi:{count}", {"family": "qi"}, dict(budget=35, eta=eta, xi=xi, delays=delays)
                        count += 1
        if count != 2736:
            raise RuntimeError(f"QI catalogue differs from the declared 2736 deployments: {count}")
    if section in ("all", "canonical"):
        for name, delays in canonical_services(propagation(p)).items():
            for budget in budgets:
                labels = {"family": "canonical", "name": name, "budget": budget}
                yield f"canonical:{name}:{budget}", labels, dict(budget=budget, delays=delays)
    if section in ("all", "sensitivity"):
        for budget in (14, 24, 35):
            for horizon in (8, 10, 12):
                label = {"family": "horizon", "budget": budget, "horizon": horizon}
                yield f"horizon:{horizon}:{budget}", label, dict(budget=budget, horizon=horizon)
            labels = {"family": "hardware", "budget": budget}
            yield f"hardware:{budget}", labels, dict(budget=budget, eta=(1,) * 3, xi=(1,) * 7)
            for qi in (False, True):
                label = {"family": "three_tier_qi" if qi else "three_tier", "budget": budget}
                yield f"{label['family']}:{budget}", label, dict(budget=budget, tiers=3, qi=qi)


def run(output: Path, section: str) -> Path:
    from rfd import qi_completions, rfd_path

    path = output / "computed.json"
    config = {"seed": 23, "threads": 1, "horizon": 10, "budgets": list(range(14, 36)), "implementation": 2}
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "format": "codeca-minimal-1", "configuration": config, "records": {}, "completed_sections": [],
    }
    if payload.get("configuration") != config:
        raise ValueError("Existing results use different settings; choose another --output folder")
    payload["complete"] = False
    p = plant()

    def execute(key: str, labels: dict, arguments: dict) -> None:
        old = payload["records"].get(key)
        allow_infeasible = labels["family"] in ("canonical", "rfd", "rfd_qi", "hardware")
        if old and (old["certified"] or (old["status"] == "INFEASIBLE" and allow_infeasible)):
            return
        print(key, flush=True)
        payload["records"][key] = {**labels, **solve(p, **arguments)}
        save(path, payload)

    for key, labels, arguments in jobs(section, p):
        if labels["family"] == "horizon" and labels["horizon"] == 10:
            source = payload["records"].get(f"codesign:{labels['budget']}")
            if source and source["certified"]:
                payload["records"][key] = {**source, **labels}
                save(path, payload)
                continue
        execute(key, labels, arguments)
    if section in ("all", "rfd"):
        if len(payload.get("rfd_path", [])) != 176:
            payload["rfd_path"] = list(rfd_path(p))
            save(path, payload)
        unique = {}
        for point in payload["rfd_path"]:
            choice = point["choice"]
            unique[json.dumps(choice, sort_keys=True)] = choice
        raw = [unique[key] for key in sorted(unique)]
        repaired = qi_completions(p, raw)
        for family, choices in (("rfd", raw), ("rfd_qi", repaired)):
            unique = {json.dumps(x, sort_keys=True): x for x in choices if x is not None}
            for i, key in enumerate(sorted(unique)):
                choice = unique[key]
                execute(f"{family}:{i}", {"family": family}, dict(
                    budget=choice["cost"], eta=choice["eta"], xi=choice["xi"], delays=delay_array(choice),
                ))
    sections = ("codesign", "qi", "canonical", "rfd", "sensitivity") if section == "all" else (section,)
    payload["completed_sections"] = sorted(set(payload["completed_sections"]) | set(sections))
    unresolved = [
        key for key, row in payload["records"].items()
        if not row["certified"] and row["status"] != "INFEASIBLE"
    ]
    # In this plant every PBH-admissible QI deployment has a feasible FIR witness.
    required_feasible = ("dense", "codesign", "qi", "horizon", "three_tier", "three_tier_qi")
    unresolved += [
        key for key, row in payload["records"].items()
        if row["family"] in required_feasible and not row["certified"]
    ]
    payload["unresolved"] = sorted(set(unresolved))
    payload["complete"] = len(payload["completed_sections"]) == 5 and not unresolved
    save(path, payload)
    if unresolved:
        raise RuntimeError(f"{len(set(unresolved))} unresolved results retained in {path}; rerun to retry them")
    return path
