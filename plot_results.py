"""Figure 1 and numerical summaries; only matplotlib and saved JSON are needed."""

import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def tolerance(a: float, b: float) -> float:
    return max(5e-8, 1e-8 * max(1., abs(a), abs(b)))


def envelope(rows: list[dict]) -> list[dict]:
    points = []
    for budget in range(14, 36):
        eligible = [
            r for r in rows if r["certified"]
            and r["choice"]["cost"] <= budget + 1e-9
        ]
        if eligible:
            best = min(eligible, key=lambda r: (r["J"], r["choice"]["cost"]))
            points.append({**best, "budget": budget})
    return points


def pareto(rows: list[dict]) -> list[dict]:
    best, points = math.inf, []
    for row in sorted((r for r in rows if r["certified"]), key=lambda r: (r["choice"]["cost"], r["J"])):
        if not points or row["J"] < best - tolerance(best, row["J"]):
            points.append(row)
            best = row["J"]
    return points


def plot_data(payload: dict) -> dict:
    if payload.get("format") != "codeca-minimal-1":
        raise ValueError("Use data/reference.json or the new computed.json format")
    records = list(payload["records"].values())
    families = {
        name: [r for r in records if r["family"] == name]
        for name in ("dense", "codesign", "qi", "canonical", "rfd")
    }
    for name, count in (("dense", 1), ("codesign", 22), ("qi", 2736)):
        if len(families[name]) != count or not all(r["certified"] for r in families[name]):
            raise ValueError(f"Incomplete or uncertified {name} results; full figure requires {count} records")
    if len(families["canonical"]) != 264 or len(payload.get("rfd_path", [])) != 176 or not families["rfd"]:
        raise ValueError("Complete the canonical and native RFD comparisons before plotting")
    curves = {name: payload.get("envelopes", {}).get(name, envelope(families[name])) for name in ("codesign", "qi")}
    if any(len(rows) != 22 or not all(r["certified"] for r in rows) for rows in curves.values()):
        raise ValueError("Budget envelopes are incomplete")
    lower_bounds = {r["budget"]: r.get("bound") for r in families["codesign"]}
    for point in curves["codesign"]:
        lower = lower_bounds[point["budget"]]
        if lower is None or abs(point["J"] - lower) > tolerance(point["J"], lower):
            raise ValueError("Co-design envelope disagrees with the budget's lower bound")
    canonical = []
    for name in ("PF", "PLF", "BD", "BDL", "TPF", "TPLF"):
        canonical.extend(pareto([r for r in families["canonical"] if r["name"] == name]))
    return {**families, **curves, "qi_cloud": families["qi"], "canonical": canonical,
            "rfd": pareto(envelope(families["rfd"])), "dense_J": families["dense"][0]["J"]}


def draw(data: dict):
    dense = data["dense_J"]

    def xy(rows):
        costs, losses = [], []
        for row in rows:
            J, cost = row["J"], row["choice"]["cost"]
            if not (math.isfinite(J) and math.isfinite(cost)) or J < dense - tolerance(J, dense):
                raise ValueError("Invalid performance relative to the dense reference")
            costs.append(cost / 35)
            losses.append(0. if abs(J - dense) <= tolerance(J, dense) else J - dense)
        return costs, losses

    figure, axis = plt.subplots(figsize=(3.45, 2.72))
    figure.subplots_adjust(left=.16, right=.985, bottom=.18, top=.70)
    blue, orange, ink, neutral, pale = "#1F5A99", "#C77718", "#2C3136", "#727B84", "#B9C1C8"
    axis.scatter(*xy(data["qi_cloud"]), s=7, marker="o", facecolors="none",
                 edgecolors=pale, alpha=.32, linewidths=.35, zorder=1)
    axis.plot(*xy(data["qi"]), color=orange, marker="s", markerfacecolor="white", markeredgecolor=orange,
              linestyle=(0, (3., 1.8)), markersize=3., linewidth=.95, zorder=4)
    axis.plot(*xy(data["codesign"]), color=blue, marker="o", markerfacecolor=blue, markeredgecolor="white",
              markeredgewidth=.35, markersize=3.25, linewidth=1.05, zorder=5)
    axis.scatter(*xy(data["rfd"]), s=19, marker="^", facecolors="none", edgecolors=neutral, linewidths=.65, zorder=3)
    axis.scatter(*xy(data["canonical"]), s=22, marker="x", color=ink, linewidths=.75, zorder=6)
    axis.axhline(0., color=neutral, linewidth=.45, zorder=0)
    axis.set_yscale("symlog", base=10, linthresh=1e-8, linscale=.65)
    all_points = sum((data[name] for name in ("codesign", "qi", "qi_cloud", "canonical", "rfd")), [])
    costs, losses = xy(all_points)
    axis.set_xlim(max(0., min(costs) - .04), 1.035)
    axis.set_ylim(-4e-10, max(1., 1.25 * max(losses)))
    axis.set_yticks(
        [0., 1e-7, 1e-5, 1e-3, .1, 1.],
        [r"$0$", r"$10^{-7}$", r"$10^{-5}$", r"$10^{-3}$", r"$10^{-1}$", r"$10^{0}$"],
    )
    axis.set_xlabel(r"normalized realized cost $\rho=J_{\mathrm{arch}}/35$")
    axis.set_ylabel(r"$\Delta J_{\mathrm{perf}}$")
    axis.grid(True, axis="y", linewidth=.28, alpha=.25)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(length=2.4, color=neutral)
    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=3.2, markerfacecolor="none", markeredgecolor=pale),
        Line2D([], [], color=orange, marker="s", markerfacecolor="white",
               linestyle=(0, (3., 1.8)), markersize=3.2, linewidth=.95),
        Line2D([], [], color=blue, marker="o", markersize=3.3, linewidth=1.05),
        Line2D([], [], marker="^", linestyle="", markersize=3.8, markerfacecolor="none", markeredgecolor=neutral),
        Line2D([], [], marker="x", linestyle="", markersize=3.8, color=ink),
    ]
    labels = [
        "feasible QI deployments (2736 of 2736)", "QI hard-budget optima",
        "joint-MICP hard-budget optima", "RFD architectures", "canonical architectures",
    ]
    axis.legend(handles, labels, loc="lower left", bbox_to_anchor=(0., 1.01, 1., .22), mode="expand", frameon=False,
                ncol=2, handlelength=1.15, handletextpad=.35, borderaxespad=.15, columnspacing=.7, labelspacing=.25)
    return figure


def export(source: Path, output: Path) -> None:
    payload = json.loads(source.read_text(encoding="utf-8"))
    data = plot_data(payload)
    output.mkdir(parents=True, exist_ok=True)
    style = {"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 7.,
             "axes.labelsize": 7.2, "axes.linewidth": .55, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
             "xtick.major.width": .45, "ytick.major.width": .45, "legend.fontsize": 5.15,
             "mathtext.fontset": "dejavusans", "pdf.fonttype": 42, "ps.fonttype": 42, "pdf.compression": 9}
    with matplotlib.rc_context(style):
        figure = draw(data)
        for suffix in ("png", "pdf"):
            figure.savefig(output / f"figure1.{suffix}", dpi=300, bbox_inches="tight")
        plt.close(figure)
    with (output / "budget_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("method", "budget", "realized_cost", "J", "J_minus_dense"))
        for family in ("codesign", "qi", "rfd", "canonical"):
            for row in envelope(data[family]):
                if row["budget"] in (14, 24, 35):
                    writer.writerow((
                        family, row["budget"], row["choice"]["cost"], row["J"], row["J"] - data["dense_J"],
                    ))
    with (output / "sensitivity.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("case", "horizon", "budget", "status", "certified", "realized_cost", "J"))
        for row in payload["records"].values():
            if row["family"] in ("horizon", "hardware", "three_tier", "three_tier_qi"):
                writer.writerow((row["family"], row.get("horizon", 10), row["budget"], row["status"], row["certified"],
                                 row.get("choice", {}).get("cost"), row.get("J")))
    print(f"Figure and summaries: {output.resolve()}")
