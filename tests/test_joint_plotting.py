"""Static publication-figure schema and rendering checks."""

from __future__ import annotations

import json
from pathlib import Path

from repro.joint_plotting import (
    FIGURE_STEM,
    build_joint_figure,
    generate_joint_publication_figure,
    load_joint_figure_data,
)


def _envelope_point(key: str, budget: int, cost: float, performance: float):
    return {
        "key": key,
        "budget": budget,
        "certified": True,
        "realized_total_cost": cost,
        "performance_h2": performance,
    }


def _baseline_point(
    *, family: str, layer: str, identity: str, cost: float, performance: float
):
    return {
        "key": f"{identity}:B14",
        "source_identity": identity,
        "family": family,
        "layer": layer,
        "plot_eligible": not (family == "rfd" and layer == "raw"),
        "certificate": {"certified": True},
        "fixed_architecture_resynthesis": {
            "architecture_cost": cost,
            "performance_h2": performance,
            "eta": [1, 1, 1],
            "xi": [0, 1, 0, 1, 0, 1, 0],
            "service_delays": [[0, None], [1, 0]],
        },
    }


def _payload() -> dict[str, object]:
    dense = 5.0
    return {
        "validation": {
            "complete": True,
            "dense_reference_h2": dense,
            "raw_rfd_identity_count": 45,
            "raw_rfd_resolved_infeasible_count": 990,
            "raw_rfd_no_incumbent_count": 0,
        },
        "proposed": {
            "envelope": [
                _envelope_point("p14", 14, 14.0, 8.0),
                _envelope_point("p35", 35, 35.0, 5.0 + 1.0e-10),
            ]
        },
        "qi": {
            "coverage": {
                "expected_deployment_count": 2736,
                "complete": False,
                "plot_label": "Evaluated PBH-admissible QI deployments",
            },
            "deployment_points": [
                {
                    "key": "q1",
                    "certified_feasible": True,
                    "architecture_breakdown": {"total": 17.0},
                    "fixed_architecture_result": {"performance_h2": 7.0},
                }
            ],
            "envelope": [
                _envelope_point("q14", 14, 14.0, 8.2),
                _envelope_point("q35", 35, 35.0, 5.0),
            ],
        },
        "baselines": {
            "points": [
                _baseline_point(
                    family="canonical",
                    layer="raw",
                    identity="canonical:PF",
                    cost=16.0,
                    performance=7.8,
                ),
                _baseline_point(
                    family="canonical",
                    layer="qi_repaired",
                    identity="canonical:PF",
                    cost=18.0,
                    performance=7.2,
                ),
                _baseline_point(
                    family="rfd",
                    layer="qi_repaired",
                    identity="rfd:1:0.1",
                    cost=20.0,
                    performance=7.5,
                ),
                _baseline_point(
                    family="rfd",
                    layer="raw",
                    identity="rfd:1:0.1",
                    cost=14.0,
                    performance=9.0,
                ),
            ]
        },
    }


def test_loader_clamps_only_tolerance_level_dense_differences(tmp_path: Path) -> None:
    source = tmp_path / "publication.json"
    source.write_text(json.dumps(_payload()), encoding="utf-8")
    data = load_joint_figure_data(source)

    assert data.proposed[-1].signed_loss > 0.0
    assert data.proposed[-1].plotted_loss == 0.0
    assert len(data.qi_deployments) == 1
    assert len(data.rfd_architectures) == 1
    assert data.raw_rfd_identity_count == 45


def test_figure_exports_matplotlib_png_pdf_and_strict_source_json(
    tmp_path: Path,
) -> None:
    source = tmp_path / "publication.json"
    source.write_text(json.dumps(_payload()), encoding="utf-8")
    manifest = generate_joint_publication_figure(
        publication_source=source,
        output_directory=tmp_path,
    )

    assert (tmp_path / f"{FIGURE_STEM}.png").is_file()
    assert (tmp_path / f"{FIGURE_STEM}.pdf").is_file()
    source_data = json.loads(
        (tmp_path / f"{FIGURE_STEM}_source_data.json").read_text(encoding="utf-8")
    )
    assert source_data["raw_rfd"]["performance_points_plotted"] == 0
    assert source_data["dense_total_cost"] == 35
    assert manifest["renderer"] == "matplotlib"


def test_figure_uses_locked_five_series_labels_and_explicit_y_ticks(
    tmp_path: Path,
) -> None:
    source = tmp_path / "publication.json"
    source.write_text(json.dumps(_payload()), encoding="utf-8")
    figure = build_joint_figure(load_joint_figure_data(source))
    axis = figure.axes[0]

    assert [text.get_text() for text in axis.get_legend().get_texts()] == [
        "evaluated feasible QI deployments (1 of 1 evaluated)",
        "QI hard-budget optima",
        "joint-MICP hard-budget optima",
        "RFD architectures",
        "canonical architectures",
    ]
    assert [text.get_text() for text in axis.get_yticklabels()] == [
        r"$0$",
        r"$10^{-7}$",
        r"$10^{-5}$",
        r"$10^{-3}$",
        r"$10^{-1}$",
        r"$10^{0}$",
    ]
    # QI family, RFD, and corrected canonical only.  The retained
    # QI-repaired canonical data are not a sixth plotted series.
    assert len(axis.collections) == 3
