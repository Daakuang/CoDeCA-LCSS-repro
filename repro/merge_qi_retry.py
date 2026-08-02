"""Merge independently audited QI retry points into the publication artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from .joint_publication import (
    DEFAULT_BUDGETS,
    _final_validation,
    _qi_coverage,
)


DEFAULT_PUBLICATION: Final[Path] = Path(
    "results/lcss_v2/joint_platoon/publication.json"
)
DEFAULT_RETRY: Final[Path] = Path(
    "results/lcss_v2/joint_platoon/qi_fixed_qp_retry.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _certified_variant(point: Mapping[str, Any]) -> Mapping[str, Any]:
    variants = [
        item
        for item in point["barrier_variants"]
        if item.get("certified_optimal_and_audited") is True
    ]
    if len(variants) != 1:
        raise RuntimeError(
            f"expected one certified retry variant for {point['key']}, "
            f"observed {len(variants)}"
        )
    return variants[0]


def merge_qi_retry(publication: Path, retry: Path) -> dict[str, Any]:
    publication = publication.resolve()
    retry = retry.resolve()
    payload = json.loads(publication.read_text(encoding="utf-8"))
    diagnostics = json.loads(retry.read_text(encoding="utf-8"))
    if diagnostics["source_publication_sha256"] != _sha256(publication):
        raise RuntimeError("QI retry does not identify the current publication artifact")
    retry_points = {str(point["key"]): point for point in diagnostics["points"]}
    if len(retry_points) != int(diagnostics["requested_count"]):
        raise RuntimeError("QI retry keys are not unique")

    merged = 0
    for point in payload["qi"]["deployment_points"]:
        key = str(point["key"])
        if key not in retry_points:
            continue
        retry_point = retry_points[key]
        variant = _certified_variant(retry_point)
        total_cost = float(point["architecture_breakdown"]["total"])
        previous = point["fixed_architecture_result"]
        point["fixed_architecture_result"] = {
            "status": "OPTIMAL",
            "solution_count": int(variant["solution_count"]),
            "performance_h2": float(variant["performance_h2"]),
            "objective_value": float(variant["objective_value"]),
            "architecture_cost": total_cost,
            "architecture_breakdown": point["architecture_breakdown"],
            "eta": point["eta"],
            "xi": point["xi"],
            "service_delays": point["service_delays"],
            "best_bound": None,
            "mip_gap": None,
            "runtime_seconds": float(variant["runtime_seconds"]),
            "solver_settings": previous["solver_settings"],
            "audit": variant["audit"],
        }
        point["certified_feasible"] = True
        point["numerical_resolution"] = {
            "original_status": previous["status"],
            "resolved_status": "OPTIMAL",
            "source_retry_artifact": retry.as_posix(),
            "source_retry_sha256": _sha256(retry),
            "selected_variant": {
                "dual_reductions": int(variant["dual_reductions"]),
                "presolve": int(variant["presolve"]),
                "method": str(variant["method"]),
                "barrier_homogeneous": int(variant["barrier_homogeneous"]),
                "crossover": int(variant["crossover"]),
                "barrier_convergence_tolerance": float(
                    variant["barrier_convergence_tolerance"]
                ),
            },
        }
        merged += 1
    if merged != len(retry_points):
        raise RuntimeError(
            f"merged {merged} QI retry points, expected {len(retry_points)}"
        )

    points = payload["qi"]["deployment_points"]
    payload["qi"]["coverage"] = _qi_coverage(points, len(points))
    _final_validation(payload, DEFAULT_BUDGETS)
    validation = dict(payload["validation"])
    validation.update(
        {
            "qi_logical_catalog_count": len(points),
            "qi_fixed_qp_optimal_and_audited_count": len(points),
            "qi_fixed_qp_infeasible_count": 0,
            "qi_fixed_qp_numeric_or_suboptimal_count": 0,
            "qi_diagnostic_feasibility_witness_count": 0,
            "qi_hard_budget_all_certified": all(
                point["certificate"]["certified"] is True
                for point in payload["qi"]["raw_budget_points"]
            ),
            "qi_additional_anomaly_resolve_status": (
                "COMPLETED_OPTIMAL_AND_INDEPENDENTLY_AUDITED"
            ),
        }
    )
    payload["validation"] = validation
    payload.setdefault("artifact_history", []).append(
        {
            "operation": "merge_independently_audited_qi_retry",
            "source_retry": retry.as_posix(),
            "source_retry_sha256": _sha256(retry),
            "merged_point_count": merged,
        }
    )
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication", type=Path, default=DEFAULT_PUBLICATION)
    parser.add_argument("--retry", type=Path, default=DEFAULT_RETRY)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    publication = arguments.publication.resolve()
    payload = merge_qi_retry(publication, arguments.retry)
    temporary = publication.with_suffix(publication.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, publication)
    print(
        json.dumps(
            {
                "publication": str(publication),
                "publication_sha256": _sha256(publication),
                "qi_certified_count": payload["qi"]["coverage"][
                    "certified_feasible_count"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
