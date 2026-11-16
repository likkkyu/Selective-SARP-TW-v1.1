"""Publication-ready road example and quality-time trade-off figure."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LNS_DIR = ROOT / "experiments" / "hard4_lns_20261106"
sys.path[:0] = [str(HERE), str(LNS_DIR)]

from greedy_hard4 import make_start
from road_network import load_overpass
from road_route_engine import RoadRouteEngine


TOL_MUTED = ["#CC6677", "#332288", "#DDCC77", "#117733", "#88CCEE", "#882255", "#44AA99", "#999933"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    output_dir = HERE / "figures"
    output_dir.mkdir(exist_ok=True)
    stem = output_dir / "real_road_validation_v1"
    targets = [stem.with_suffix(ext) for ext in (".pdf", ".png")]
    targets += [output_dir / "real_road_validation_v1_source.csv", output_dir / "real_road_validation_v1_manifest.json", output_dir / "real_road_validation_v1_alt_text.md"]
    if any(path.exists() for path in targets):
        raise FileExistsError("Refusing to overwrite an existing real-road figure deliverable")

    analysis_path = HERE / "road_native_formal_test_v1_analysis.json"
    pool_path = HERE / "road_native_formal_test_v1_seed2026111401_n128.pt"
    graph_path = HERE / "data" / "futian_luohu_current.json"
    plan_path = HERE / "formal_p1_v1" / "plan.json"
    analysis = json.loads(analysis_path.read_text())
    plan = json.loads(plan_path.read_text())
    pool = torch.load(pool_path, map_location="cpu", weights_only=True)
    sample = pool["data"][0]
    graph = load_overpass(graph_path)
    engine = RoadRouteEngine(sample, plan["physical_config"], plan["normalization_profile"])
    routes = make_start(engine)
    if engine.cost(routes) is None:
        raise AssertionError("Example strict-insertion routes are invalid")

    methods = [
        ("Neural", "neural", "#0072B2", "o"),
        ("Neural + LNS", "hybrid", "#D55E00", "s"),
        ("Insertion", "insertion", "#009E73", "^"),
        ("Insertion + LNS", "lns", "#CC79A7", "D"),
    ]
    source_rows = []
    for label, key, color, marker in methods:
        row = analysis["method_summary"][key]
        source_rows.append({
            "method": label,
            "service_rate_percent_mean": row["service_rate_percent"]["mean"],
            "service_rate_percent_ci_low": row["service_rate_percent"]["bootstrap_95_ci_for_mean"][0],
            "service_rate_percent_ci_high": row["service_rate_percent"]["bootstrap_95_ci_for_mean"][1],
            "algorithm_seconds_mean": row["algorithm_seconds"]["mean"],
            "algorithm_seconds_ci_low": row["algorithm_seconds"]["bootstrap_95_ci_for_mean"][0],
            "algorithm_seconds_ci_high": row["algorithm_seconds"]["bootstrap_95_ci_for_mean"][1],
            "color": color,
            "marker": marker,
        })
    with (output_dir / "real_road_validation_v1_source.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.7,
    })
    fig, (ax_map, ax_trade) = plt.subplots(
        1,
        2,
        figsize=(180 / 25.4, 78 / 25.4),
        gridspec_kw={"width_ratios": [1.22, 1]},
        layout="constrained",
    )
    fig.patch.set_facecolor("white")

    # Panel A: actual shortest paths on the frozen directed Shenzhen OSM graph.
    for u, v in graph.edges():
        x = [graph.nodes[u]["lon"], graph.nodes[v]["lon"]]
        y = [graph.nodes[u]["lat"], graph.nodes[v]["lat"]]
        ax_map.plot(x, y, color="#D9D9D9", linewidth=0.18, alpha=0.55, zorder=1)
    terminal_ids = sample["road_node_ids"].tolist()
    for route_index, route in enumerate(routes):
        color = TOL_MUTED[route_index % len(TOL_MUTED)]
        full = [0] + list(route) + [0]
        for left, right in zip(full[:-1], full[1:]):
            path = nx.shortest_path(graph, int(terminal_ids[left]), int(terminal_ids[right]), weight="length_km")
            ax_map.plot(
                [graph.nodes[node]["lon"] for node in path],
                [graph.nodes[node]["lat"] for node in path],
                color=color,
                linewidth=1.05,
                alpha=0.9,
                zorder=2,
                solid_capstyle="round",
            )
        pickup_nodes = [terminal_ids[node] for node in route if node <= 50]
        delivery_nodes = [terminal_ids[node] for node in route if node > 50]
        ax_map.scatter(
            [graph.nodes[node]["lon"] for node in pickup_nodes],
            [graph.nodes[node]["lat"] for node in pickup_nodes],
            s=8,
            marker="o",
            facecolors="white",
            edgecolors=color,
            linewidths=0.65,
            zorder=3,
        )
        ax_map.scatter(
            [graph.nodes[node]["lon"] for node in delivery_nodes],
            [graph.nodes[node]["lat"] for node in delivery_nodes],
            s=9,
            marker="^",
            facecolors=color,
            edgecolors=color,
            linewidths=0.45,
            zorder=3,
        )
    depot = int(terminal_ids[0])
    ax_map.scatter(graph.nodes[depot]["lon"], graph.nodes[depot]["lat"], s=42, marker="*", color="#000000", zorder=4, label="Depot")
    ax_map.scatter([], [], s=12, marker="o", facecolors="white", edgecolors="#333333", linewidths=0.7, label="Pickup")
    ax_map.scatter([], [], s=12, marker="^", color="#333333", label="Delivery")
    ax_map.set_title("(a) Frozen Shenzhen road instance and routes", loc="left", fontweight="bold")
    ax_map.set_xlabel("Longitude (°E)")
    ax_map.set_ylabel("Latitude (°N)")
    ax_map.set_aspect(1 / np.cos(np.deg2rad(float(sample["road_lonlat"][:, 1].mean()))))
    ax_map.legend(frameon=False, ncol=3, loc="lower left", handletextpad=0.35, columnspacing=0.8)
    ax_map.spines[["top", "right"]].set_visible(False)

    # Panel B: common-scale point estimates and paired-instance bootstrap CIs.
    for row in source_rows:
        x = row["algorithm_seconds_mean"]
        y = row["service_rate_percent_mean"]
        xerr = [[x - row["algorithm_seconds_ci_low"]], [row["algorithm_seconds_ci_high"] - x]]
        yerr = [[y - row["service_rate_percent_ci_low"]], [row["service_rate_percent_ci_high"] - y]]
        ax_trade.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            marker=row["marker"],
            markersize=5.5,
            markerfacecolor="white",
            markeredgewidth=1.2,
            color=row["color"],
            linewidth=1.1,
            capsize=2.2,
            label=row["method"],
            zorder=3,
        )
    by_name = {row["method"]: row for row in source_rows}
    for initial, improved, color in (("Neural", "Neural + LNS", "#666666"), ("Insertion", "Insertion + LNS", "#999999")):
        a, b = by_name[initial], by_name[improved]
        ax_trade.annotate(
            "",
            xy=(b["algorithm_seconds_mean"], b["service_rate_percent_mean"]),
            xytext=(a["algorithm_seconds_mean"], a["service_rate_percent_mean"]),
            arrowprops={"arrowstyle": "->", "color": color, "linewidth": 0.8, "linestyle": "--"},
            zorder=2,
        )
    ax_trade.set_title("(b) Service quality–time trade-off", loc="left", fontweight="bold")
    ax_trade.set_xlabel("Algorithm time per instance (s)")
    ax_trade.set_ylabel("Served orders (%)")
    ax_trade.set_xlim(0, 3.55)
    ax_trade.set_ylim(88, 101)
    ax_trade.set_xticks(np.arange(0, 3.6, 0.5))
    ax_trade.grid(axis="both", color="#E6E6E6", linewidth=0.5, zorder=0)
    ax_trade.spines[["top", "right"]].set_visible(False)
    ax_trade.legend(frameon=False, loc="lower right", handletextpad=0.5)
    ax_trade.text(
        0.02,
        0.98,
        "Mean and 95% paired-instance bootstrap CI\n(n = 128 road scenarios)",
        transform=ax_trade.transAxes,
        va="top",
        ha="left",
        fontsize=6.8,
        color="#333333",
    )

    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=450, facecolor="white")
    plt.close(fig)

    alt_text = (
        "Two-panel scientific figure. Panel a maps one frozen 50-order scenario on the actual "
        "Futian–Luohu directed OSM driving graph, with depot, pickup and delivery nodes and eight "
        "strict-insertion vehicle routes drawn along road shortest paths. Panel b plots mean served "
        "orders against algorithm time for neural construction, neural plus three-second LNS, strict "
        "insertion, and insertion plus three-second LNS. Pure neural construction has the lowest "
        "service rate; both LNS variants approach 100 percent service at approximately 3.1–3.25 seconds."
    )
    (output_dir / "real_road_validation_v1_alt_text.md").write_text(alt_text + "\n")
    manifest = {
        "status": "created",
        "purpose": "provisional manuscript figure; WTC final-size requirements pending live verification",
        "figure_width_mm": 180,
        "figure_height_mm": 78,
        "panel_a": "actual directed OSM graph and actual shortest-path route geometry for frozen test instance 0",
        "panel_b": "method means and 95% paired-instance bootstrap confidence intervals; n=128 scenarios",
        "transformations": ["three neural training seeds averaged within each test instance", "no smoothing", "no exclusions"],
        "sources": {
            str(analysis_path.relative_to(ROOT)): digest(analysis_path),
            str(pool_path.relative_to(ROOT)): digest(pool_path),
            str(graph_path.relative_to(ROOT)): digest(graph_path),
            str(plan_path.relative_to(ROOT)): digest(plan_path),
        },
        "outputs": {path.name: digest(path) for path in targets if path.exists()},
        "software": {"python": platform.python_version(), "matplotlib": mpl.__version__, "networkx": nx.__version__},
        "integrity": "road geometry and results were not smoothed, filtered, or selectively altered",
    }
    (output_dir / "real_road_validation_v1_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "created", "routes": len(routes), "outputs": [str(path) for path in targets]}, indent=2))


if __name__ == "__main__":
    main()
