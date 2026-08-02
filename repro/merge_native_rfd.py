"""Merge the native RFD path into a joint-publication result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Final, Sequence


DEFAULT_PUBLICATION: Final[Path] = Path(
    "results/lcss_v2/reproduced_joint/publication.json"
)
DEFAULT_RFD: Final[Path] = Path(
    "results/lcss_v2/reproduced_joint/rfd_native_scan.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def merge_native_rfd(publication_path: Path, rfd_path: Path) -> dict[str, Any]:
    """Return a publication payload with the independently computed RFD scan."""

    publication = json.loads(publication_path.read_text(encoding="utf-8"))
    rfd = json.loads(rfd_path.read_text(encoding="utf-8"))
    if rfd.get("complete") is not True:
        raise RuntimeError("the native RFD artifact is incomplete")
    if rfd.get("matrix_sha256") != publication["case"]["matrix_sha256"]:
        raise RuntimeError("the RFD artifact uses another plant")

    validation = dict(publication["validation"])
    publication["baselines"]["legacy_rfd_adapter_diagnostics"] = {
        "identity": "station-support adapter excluded from the final comparison",
        "raw_rfd_identity_count": validation.pop("raw_rfd_identity_count", None),
        "raw_rfd_budget_status_count": validation.pop(
            "raw_rfd_budget_status_count", None
        ),
        "raw_rfd_resolved_infeasible_count": validation.pop(
            "raw_rfd_resolved_infeasible_count", None
        ),
        "raw_rfd_no_incumbent_count": validation.pop(
            "raw_rfd_no_incumbent_count", None
        ),
        "raw_rfd_points_plotted": validation.pop("raw_rfd_points_plotted", None),
    }
    publication["rfd_native"] = rfd
    validation.update(
        {
            "rfd_path_point_count": int(rfd["path_point_count"]),
            "rfd_raw_unique_candidate_count": int(
                rfd["raw_unique_candidate_count"]
            ),
            "rfd_raw_certified_candidate_count": sum(
                item["fixed_architecture_result"]["certified_feasible"] is True
                for item in rfd["raw_candidates"]
            ),
            "rfd_qi_repaired_unique_candidate_count": int(
                rfd["qi_repaired_unique_candidate_count"]
            ),
            "rfd_qi_repaired_certified_candidate_count": sum(
                item["fixed_architecture_result"]["certified_feasible"] is True
                for item in rfd["qi_repaired_candidates"]
            ),
            "rfd_raw_budget_envelope_budget_count": sum(
                item.get("argmin_key") is not None
                for item in rfd["raw_evaluated_budget_envelope"]
            ),
            "complete": True,
        }
    )
    publication["validation"] = validation
    publication.setdefault("artifact_history", []).append(
        {
            "operation": "merge_native_rfd",
            "rfd_native": rfd_path.as_posix(),
            "rfd_native_sha256": _sha256(rfd_path),
        }
    )
    return publication


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication", type=Path, default=DEFAULT_PUBLICATION)
    parser.add_argument("--rfd", type=Path, default=DEFAULT_RFD)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    publication = arguments.publication.resolve()
    rfd = arguments.rfd.resolve()
    payload = merge_native_rfd(publication, rfd)
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
                "rfd_path_point_count": payload["validation"][
                    "rfd_path_point_count"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
