"""Matplotlib publication figure for the joint platoon experiment."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import matplotlib
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

from .joint_publication import (
    CERTIFICATE_ABSOLUTE_TOLERANCE,
    CERTIFICATE_RELATIVE_TOLERANCE,
    DEFAULT_DENSE_COST,
    EXPECTED_PBH_ADMISSIBLE_QI_COUNT,
)
from .platoon_experiment import write_json_atomic


FIGURE_STEM: Final[str] = "platoon_budget_comparison_joint"
FIGURE_SIZE_INCHES: Final[tuple[float, float]] = (3.45, 2.72)
PNG_DPI: Final[int] = 300


@dataclass(frozen=True, slots=True)
class PlotPoint:
    key: str
    label: str
    family: str
    total_cost: float
    performance_h2: float
    signed_loss: float
    plotted_loss: float

    @property
    def rho(self) -> float:
        return self.total_cost / DEFAULT_DENSE_COST


@dataclass(frozen=True, slots=True)
class JointFigureData:
    dense_reference_h2: float
    proposed: tuple[PlotPoint, ...]
    qi_deployments: tuple[PlotPoint, ...]
    qi_coverage_label: str
    qi_expected_count: int
    qi_evaluated_count: int
    qi_coverage_complete: bool
    qi_budget_optima: tuple[PlotPoint, ...]
    canonical_raw: tuple[PlotPoint, ...]
    canonical_repaired: tuple[PlotPoint, ...]
    rfd_architectures: tuple[PlotPoint, ...]
    rfd_path_point_count: int
    rfd_raw_candidate_count: int
    rfd_qi_repaired_candidate_count: int
    raw_rfd_identity_count: int
    raw_rfd_resolved_infeasible_count: int
    raw_rfd_no_incumbent_count: int


def _style() -> dict[str, object]:
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 7.0,
        "axes.labelsize": 7.2,
        "axes.linewidth": 0.55,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "xtick.major.width": 0.45,
        "ytick.major.width": 0.45,
        "legend.fontsize": 5.15,
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "pdf.compression": 9,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _objects(value: object, name: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"{name} must be a list of JSON objects")
    return tuple(value)


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _numeric_tolerance(left: float, right: float) -> float:
    return max(
        CERTIFICATE_ABSOLUTE_TOLERANCE,
        CERTIFICATE_RELATIVE_TOLERANCE * max(1.0, abs(left), abs(right)),
    )


def _point(
    *,
    key: str,
    label: str,
    family: str,
    cost: object,
    performance: object,
    dense: float,
) -> PlotPoint:
    total_cost = _finite(cost, f"{key}.total_cost")
    objective = _finite(performance, f"{key}.performance_h2")
    signed = objective - dense
    tolerance = _numeric_tolerance(objective, dense)
    if signed < -tolerance:
        raise ValueError(f"{key} is materially better than the dense reference")
    return PlotPoint(
        key=key,
        label=label,
        family=family,
        total_cost=total_cost,
        performance_h2=objective,
        signed_loss=signed,
        plotted_loss=0.0 if abs(signed) <= tolerance else signed,
    )


def _envelope_points(
    values: object,
    *,
    family: str,
    dense: float,
) -> tuple[PlotPoint, ...]:
    points: list[PlotPoint] = []
    for row in _objects(values, f"{family}.envelope"):
        if row.get("certified") is not True:
            raise ValueError(f"{family} envelope contains an uncertified point")
        budget = int(_finite(row.get("budget"), "budget"))
        points.append(
            _point(
                key=str(row["key"]),
                label=f"B={budget}",
                family=family,
                cost=row.get("realized_total_cost"),
                performance=row.get("performance_h2"),
                dense=dense,
            )
        )
    return tuple(points)


def _architecture_signature(row: Mapping[str, Any]) -> str:
    fixed = _object(row.get("fixed_architecture_resynthesis"), "fixed resynthesis")
    return json.dumps(
        {
            "eta": fixed.get("eta"),
            "xi": fixed.get("xi"),
            "service_delays": fixed.get("service_delays"),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _deduplicate_points(points: Sequence[PlotPoint]) -> tuple[PlotPoint, ...]:
    by_identity: dict[tuple[str, str], PlotPoint] = {}
    for point in points:
        key = (
            f"{point.total_cost:.12g}",
            f"{point.performance_h2:.15g}",
        )
        previous = by_identity.get(key)
        if previous is None or point.key < previous.key:
            by_identity[key] = point
    return tuple(
        sorted(
            by_identity.values(),
            key=lambda item: (item.total_cost, item.performance_h2, item.key),
        )
    )


def _pareto(points: Sequence[PlotPoint]) -> tuple[PlotPoint, ...]:
    unique = _deduplicate_points(points)
    retained: list[PlotPoint] = []
    best = math.inf
    for point in unique:
        tolerance = _numeric_tolerance(best, point.performance_h2) if math.isfinite(best) else 0.0
        if point.performance_h2 < best - tolerance:
            retained.append(point)
            best = point.performance_h2
    return tuple(retained)


def _baseline_points(
    rows: Sequence[Mapping[str, Any]],
    *,
    family: str,
    layer: str,
    dense: float,
    global_pareto: bool,
) -> tuple[PlotPoint, ...]:
    by_architecture: dict[str, PlotPoint] = {}
    for row in rows:
        if (
            row.get("family") != family
            or row.get("layer") != layer
            or row.get("plot_eligible") is not True
        ):
            continue
        certificate = _object(row.get("certificate"), "baseline.certificate")
        if certificate.get("certified") is not True:
            continue
        fixed = _object(
            row.get("fixed_architecture_resynthesis"),
            "baseline.fixed_architecture_resynthesis",
        )
        identity = str(row["source_identity"])
        signature = _architecture_signature(row)
        point = _point(
            key=str(row["key"]),
            label=identity.split(":")[-1],
            family=f"{family}_{layer}",
            cost=fixed.get("architecture_cost"),
            performance=fixed.get("performance_h2"),
            dense=dense,
        )
        previous = by_architecture.get(signature)
        if previous is None or point.performance_h2 < previous.performance_h2:
            by_architecture[signature] = point
    values = tuple(by_architecture.values())
    if global_pareto:
        return _pareto(values)
    by_identity: dict[str, list[PlotPoint]] = {}
    for point in values:
        by_identity.setdefault(point.label, []).append(point)
    return tuple(
        point
        for identity in sorted(by_identity)
        for point in _pareto(by_identity[identity])
    )


def _native_rfd_points(
    rfd: Mapping[str, Any],
    *,
    dense: float,
) -> tuple[PlotPoint, ...]:
    points: list[PlotPoint] = []
    rows = _objects(
        rfd.get("raw_evaluated_budget_envelope"),
        "rfd_native.raw_evaluated_budget_envelope",
    )
    for row in rows:
        if row.get("argmin_key") is None:
            continue
        points.append(
            _point(
                key=str(row["argmin_key"]),
                label=f"B={int(_finite(row.get('budget'), 'RFD budget'))}",
                family="rfd",
                cost=row.get("realized_total_cost"),
                performance=row.get("performance_h2"),
                dense=dense,
            )
        )
    return _pareto(points)


def load_joint_figure_data(publication_source: Path) -> JointFigureData:
    payload = _object(
        json.loads(Path(publication_source).read_text(encoding="utf-8")),
        "publication",
    )
    validation = _object(payload.get("validation"), "validation")
    if validation.get("complete") is not True:
        raise ValueError("publication computation is not complete")
    dense = _finite(validation.get("dense_reference_h2"), "dense_reference_h2")
    proposed = _object(payload.get("proposed"), "proposed")
    qi = _object(payload.get("qi"), "qi")
    coverage = _object(qi.get("coverage"), "qi.coverage")
    expected = int(_finite(coverage.get("expected_deployment_count"), "expected count"))
    if expected != EXPECTED_PBH_ADMISSIBLE_QI_COUNT:
        raise ValueError("QI catalog count does not equal the declared 2736 deployments")

    qi_deployments: list[PlotPoint] = []
    deployment_rows = _objects(qi.get("deployment_points"), "qi.deployment_points")
    if coverage.get("complete") is True and len(deployment_rows) != expected:
        raise ValueError("complete QI coverage has the wrong deployment record count")
    for row in deployment_rows:
        if row.get("certified_feasible") is not True:
            continue
        fixed = _object(row.get("fixed_architecture_result"), "QI fixed result")
        breakdown = _object(row.get("architecture_breakdown"), "QI architecture cost")
        qi_deployments.append(
            _point(
                key=str(row["key"]),
                label=str(row["key"]),
                family="qi_deployments",
                cost=breakdown.get("total"),
                performance=fixed.get("performance_h2"),
                dense=dense,
            )
        )

    baseline_root = _object(payload.get("baselines"), "baselines")
    baseline_rows = _objects(baseline_root.get("points"), "baselines.points")
    canonical_raw = _baseline_points(
        baseline_rows,
        family="canonical",
        layer="raw",
        dense=dense,
        global_pareto=False,
    )
    canonical_repaired = _baseline_points(
        baseline_rows,
        family="canonical",
        layer="qi_repaired",
        dense=dense,
        global_pareto=False,
    )
    rfd_native_value = payload.get("rfd_native")
    if isinstance(rfd_native_value, Mapping):
        rfd_native = _object(rfd_native_value, "rfd_native")
        if rfd_native.get("complete") is not True:
            raise ValueError("native RFD scan is not complete")
        rfd_architectures = _native_rfd_points(rfd_native, dense=dense)
        rfd_path_point_count = int(
            _finite(rfd_native.get("path_point_count"), "RFD path point count")
        )
        rfd_raw_candidate_count = int(
            _finite(
                rfd_native.get("raw_unique_candidate_count"),
                "RFD raw candidate count",
            )
        )
        rfd_qi_repaired_candidate_count = int(
            _finite(
                rfd_native.get("qi_repaired_unique_candidate_count"),
                "RFD repaired candidate count",
            )
        )
    else:
        rfd_architectures = _baseline_points(
            baseline_rows,
            family="rfd",
            layer="qi_repaired",
            dense=dense,
            global_pareto=True,
        )
        rfd_path_point_count = int(validation.get("raw_rfd_identity_count", 0))
        rfd_raw_candidate_count = 0
        rfd_qi_repaired_candidate_count = len(rfd_architectures)
    return JointFigureData(
        dense_reference_h2=dense,
        proposed=_envelope_points(
            proposed.get("envelope"), family="proposed", dense=dense
        ),
        qi_deployments=tuple(qi_deployments),
        qi_coverage_label=str(coverage.get("plot_label")),
        qi_expected_count=expected,
        qi_evaluated_count=len(deployment_rows),
        qi_coverage_complete=coverage.get("complete") is True,
        qi_budget_optima=_envelope_points(
            qi.get("envelope"), family="qi_budget_optima", dense=dense
        ),
        canonical_raw=canonical_raw,
        canonical_repaired=canonical_repaired,
        rfd_architectures=rfd_architectures,
        rfd_path_point_count=rfd_path_point_count,
        rfd_raw_candidate_count=rfd_raw_candidate_count,
        rfd_qi_repaired_candidate_count=rfd_qi_repaired_candidate_count,
        raw_rfd_identity_count=int(validation.get("raw_rfd_identity_count", 0)),
        raw_rfd_resolved_infeasible_count=int(
            validation.get("raw_rfd_resolved_infeasible_count", 0)
        ),
        raw_rfd_no_incumbent_count=int(
            validation.get("raw_rfd_no_incumbent_count", 0)
        ),
    )


def _xy(points: Sequence[PlotPoint]) -> tuple[list[float], list[float]]:
    return [point.rho for point in points], [point.plotted_loss for point in points]


def build_joint_figure(data: JointFigureData) -> plt.Figure:
    """Build the mother-like two-root-plus-neutral comparison figure."""

    colors = {
        "blue": "#1F5A99",
        "blue_open": "#9CB9D5",
        "orange": "#C77718",
        "ink": "#2C3136",
        "neutral": "#727B84",
        "pale": "#B9C1C8",
    }
    with matplotlib.rc_context(_style()):
        figure, axis = plt.subplots(figsize=FIGURE_SIZE_INCHES)
        figure.subplots_adjust(left=0.16, right=0.985, bottom=0.18, top=0.70)
        axis.scatter(
            *_xy(data.qi_deployments),
            s=7,
            marker="o",
            facecolors="none",
            edgecolors=colors["pale"],
            alpha=0.32,
            linewidths=0.35,
            zorder=1,
        )
        axis.plot(
            *_xy(data.qi_budget_optima),
            color=colors["orange"],
            marker="s",
            markerfacecolor="white",
            markeredgecolor=colors["orange"],
            linestyle=(0, (3.0, 1.8)),
            markersize=3.0,
            linewidth=0.95,
            zorder=4,
        )
        axis.plot(
            *_xy(data.proposed),
            color=colors["blue"],
            marker="o",
            markerfacecolor=colors["blue"],
            markeredgecolor="white",
            markeredgewidth=0.35,
            linestyle="-",
            markersize=3.25,
            linewidth=1.05,
            zorder=5,
        )
        axis.scatter(
            *_xy(data.rfd_architectures),
            s=19,
            marker="^",
            facecolors="none",
            edgecolors=colors["neutral"],
            linewidths=0.65,
            zorder=3,
        )
        axis.scatter(
            *_xy(data.canonical_raw),
            s=22,
            marker="x",
            color=colors["ink"],
            linewidths=0.75,
            zorder=6,
        )
        axis.axhline(0.0, color=colors["neutral"], linewidth=0.45, zorder=0)
        axis.set_yscale("symlog", base=10, linthresh=1.0e-8, linscale=0.65)
        all_points = (
            *data.proposed,
            *data.qi_deployments,
            *data.qi_budget_optima,
            *data.canonical_raw,
            *data.rfd_architectures,
        )
        min_rho = min((point.rho for point in all_points), default=0.35)
        axis.set_xlim(max(0.0, min_rho - 0.04), 1.035)
        max_loss = max((point.plotted_loss for point in all_points), default=1.0)
        axis.set_ylim(-4.0e-10, max(1.0, 1.25 * max_loss))
        axis.set_yticks([0.0, 1.0e-7, 1.0e-5, 1.0e-3, 1.0e-1, 1.0])
        axis.set_yticklabels(
            [
                r"$0$",
                r"$10^{-7}$",
                r"$10^{-5}$",
                r"$10^{-3}$",
                r"$10^{-1}$",
                r"$10^{0}$",
            ]
        )
        axis.set_xlabel(r"normalized realized cost $\rho=J_{\mathrm{arch}}/35$")
        axis.set_ylabel(r"$\Delta J_{\mathrm{perf}}$")
        axis.grid(True, axis="y", linewidth=0.28, alpha=0.25)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(length=2.4, color=colors["neutral"])

        handles: list[object] = [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markersize=3.2,
                markerfacecolor="none",
                markeredgecolor=colors["pale"],
            ),
            Line2D(
                [],
                [],
                color=colors["orange"],
                marker="s",
                markerfacecolor="white",
                linestyle=(0, (3.0, 1.8)),
                markersize=3.2,
                linewidth=0.95,
            ),
            Line2D(
                [],
                [],
                color=colors["blue"],
                marker="o",
                markersize=3.3,
                linewidth=1.05,
            ),
            Line2D(
                [],
                [],
                marker="^",
                linestyle="",
                markersize=3.8,
                markerfacecolor="none",
                markeredgecolor=colors["neutral"],
            ),
            Line2D(
                [],
                [],
                marker="x",
                linestyle="",
                markersize=3.8,
                color=colors["ink"],
            ),
        ]
        if data.qi_coverage_complete:
            qi_family_label = (
                f"feasible QI deployments ({len(data.qi_deployments)} of "
                f"{data.qi_expected_count})"
            )
        else:
            qi_family_label = (
                f"evaluated feasible QI deployments ({len(data.qi_deployments)} of "
                f"{data.qi_evaluated_count} evaluated)"
            )
        labels = [
            qi_family_label,
            "QI hard-budget optima",
            "joint-MICP hard-budget optima",
            "RFD architectures",
            "canonical architectures",
        ]
        axis.legend(
            handles,
            labels,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.01, 1.0, 0.22),
            mode="expand",
            frameon=False,
            ncol=2,
            handlelength=1.15,
            handletextpad=0.35,
            borderaxespad=0.15,
            columnspacing=0.7,
            labelspacing=0.25,
        )
    return figure


def _point_records(points: Sequence[PlotPoint]) -> list[dict[str, Any]]:
    return [
        {**asdict(point), "rho": point.rho}
        for point in points
    ]


def generate_joint_publication_figure(
    *,
    publication_source: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Validate source data, export PNG/PDF, and save source/manifest JSON."""

    source = Path(publication_source).resolve()
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = load_joint_figure_data(source)
    figure = build_joint_figure(data)
    png = output / f"{FIGURE_STEM}.png"
    pdf = output / f"{FIGURE_STEM}.pdf"
    source_data = output / f"{FIGURE_STEM}_source_data.json"
    manifest_path = output / f"{FIGURE_STEM}_figure_manifest.json"
    try:
        # PDF font embedding is resolved at save time, after the figure has
        # been built.  Keep the export inside the same rc policy so the
        # publication PDF retains TrueType text instead of Type 3 glyphs.
        with matplotlib.rc_context(_style()):
            figure.savefig(png, dpi=PNG_DPI, bbox_inches="tight")
            figure.savefig(pdf, bbox_inches="tight")
    finally:
        plt.close(figure)

    source_payload = {
        "schema_version": 1,
        "semantics": (
            "joint hardware-service comparison under common combined hard budgets"
        ),
        "dense_reference_h2": data.dense_reference_h2,
        "dense_total_cost": DEFAULT_DENSE_COST,
        "x_axis": "rho = realized total architecture cost / 35",
        "y_axis": "Delta J_perf relative to the fixed-architecture dense H=10 reference",
        "numeric_zero_policy": (
            "dense-reference differences within the declared mixed tolerance are "
            "plotted at zero and retain their signed value in this JSON"
        ),
        "proposed": _point_records(data.proposed),
        "qi_deployments": _point_records(data.qi_deployments),
        "qi_coverage_label": data.qi_coverage_label,
        "qi_budget_optima": _point_records(data.qi_budget_optima),
        "canonical_raw": _point_records(data.canonical_raw),
        "canonical_repaired": _point_records(data.canonical_repaired),
        "rfd_architectures": _point_records(data.rfd_architectures),
        "rfd_scan": {
            "performance_points_plotted": len(data.rfd_architectures),
            "plotted_layer": (
                "native_raw"
                if data.rfd_raw_candidate_count > 0
                else "legacy_qi_repaired"
            ),
            "path_point_count": data.rfd_path_point_count,
            "raw_unique_candidate_count": data.rfd_raw_candidate_count,
            "qi_repaired_unique_candidate_count": (
                data.rfd_qi_repaired_candidate_count
            ),
        },
        "raw_rfd": {
            "performance_points_plotted": 0,
            "identity_count": data.raw_rfd_identity_count,
            "resolved_infeasible_budget_status_count": (
                data.raw_rfd_resolved_infeasible_count
            ),
            "unresolved_no_incumbent_budget_status_count": (
                data.raw_rfd_no_incumbent_count
            ),
        },
    }
    write_json_atomic(source_data, source_payload)
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "renderer": "matplotlib",
        "palette_policy": "two non-neutral roots plus neutral tones",
        "non_color_redundancy": (
            "solid circles, open dashed squares, pale open dots, triangles, "
            "and x markers"
        ),
        "publication_source": str(source),
        "outputs": {
            "png": {"path": str(png), "sha256": _sha256(png)},
            "pdf": {"path": str(pdf), "sha256": _sha256(pdf)},
            "source_data": {
                "path": str(source_data),
                "sha256": _sha256(source_data),
            },
        },
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--publication-source",
        type=Path,
        default=Path("results/lcss_v2/joint_platoon/publication.json"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("results/lcss_v2/joint_platoon"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    generate_joint_publication_figure(
        publication_source=args.publication_source,
        output_directory=args.output_directory,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
