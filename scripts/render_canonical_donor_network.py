#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch


ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = ROOT / "canonical_features.csv"
EXPLANATIONS_CSV = ROOT / "canonical_donor_taxon_explanations.csv"
EDGES_CSV = ROOT / "canonical_donor_edges.csv"
OUTPUT_PNG = ROOT / "canonical_donor_network.png"

FIG_WIDTH = 24
FIG_HEIGHT = 34
FIG_DPI = 200

TOPOLOGY_COLORS = {
    "transformer": "#4C78A8",
    "hybrid_attention_linear": "#72B7B2",
    "hybrid_attention_ssm": "#54A24B",
    "ssm_only": "#F58518",
}
DEFAULT_NODE_COLOR = "#7F7F7F"
BACKBONE_EDGE_COLOR = "#111111"
DONOR_EDGE_COLOR = "#D95F02"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def parse_date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return float(stripped)


def build_node_index(
    features: list[dict[str, str]],
    explanations: list[dict[str, str]],
) -> tuple[list[str], dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    feature_index = {row["taxon"]: row for row in features}
    explanation_index = {row["taxon"]: row for row in explanations}
    taxa = sorted(
        explanation_index,
        key=lambda taxon: (
            parse_date(explanation_index[taxon]["resolved_date"]),
            taxon,
        ),
    )
    return taxa, feature_index, explanation_index


def build_primary_children(
    explanation_index: dict[str, dict[str, str]],
) -> tuple[list[str], dict[str, list[str]]]:
    children: dict[str, list[str]] = defaultdict(list)
    roots: list[str] = []

    for taxon, row in explanation_index.items():
        parent = row.get("primary_donor", "")
        if parent:
            children[parent].append(taxon)
        else:
            roots.append(taxon)

    def sort_key(taxon: str) -> tuple[dt.date, str]:
        return parse_date(explanation_index[taxon]["resolved_date"]), taxon

    roots.sort(key=sort_key)
    for child_list in children.values():
        child_list.sort(key=sort_key)
    return roots, dict(children)


def assign_tree_x_positions(
    roots: list[str],
    children: dict[str, list[str]],
) -> dict[str, float]:
    positions: dict[str, float] = {}
    next_leaf = 0.0

    def walk(taxon: str) -> float:
        nonlocal next_leaf
        child_names = children.get(taxon, [])
        if not child_names:
            x = next_leaf
            next_leaf += 1.0
            positions[taxon] = x
            return x

        child_x = [walk(child) for child in child_names]
        x = sum(child_x) / len(child_x)
        positions[taxon] = x
        return x

    for index, root in enumerate(roots):
        walk(root)
        if index < len(roots) - 1:
            next_leaf += 1.5

    return positions


def node_color(taxon: str, feature_index: dict[str, dict[str, str]]) -> str:
    topology = feature_index.get(taxon, {}).get("stack_topology", "")
    return TOPOLOGY_COLORS.get(topology, DEFAULT_NODE_COLOR)


def node_size(
    taxon: str,
    incoming_weight: dict[str, float],
    outgoing_weight: dict[str, float],
) -> float:
    signal = incoming_weight.get(taxon, 0.0) + outgoing_weight.get(taxon, 0.0)
    return 34.0 + 9.0 * math.sqrt(signal + 1.0)


def render(
    features: list[dict[str, str]],
    explanations: list[dict[str, str]],
    edges: list[dict[str, str]],
    output_path: Path,
    title: str,
) -> None:
    taxa, feature_index, explanation_index = build_node_index(features, explanations)
    roots, children = build_primary_children(explanation_index)
    x_positions = assign_tree_x_positions(roots, children)
    y_positions = {
        taxon: mdates.date2num(parse_date(explanation_index[taxon]["resolved_date"]))
        for taxon in taxa
    }

    incoming_weight: dict[str, float] = {}
    outgoing_weight: dict[str, float] = {}
    for edge in edges:
        weight = float(edge["borrowed_trait_count"])
        incoming_weight[edge["recipient"]] = incoming_weight.get(edge["recipient"], 0.0) + weight
        outgoing_weight[edge["donor"]] = outgoing_weight.get(edge["donor"], 0.0) + weight

    xmin = min(x_positions.values())
    xmax = max(x_positions.values())
    ymin = min(y_positions.values())
    ymax = max(y_positions.values())
    xpad = max(1.8, (xmax - xmin) * 0.07)
    ypad = max(25.0, (ymax - ymin) * 0.02)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI)
    ax.set_facecolor("#FBFAF7")
    fig.patch.set_facecolor("#FBFAF7")

    for taxon, row in explanation_index.items():
        parent = row.get("primary_donor", "")
        if not parent:
            continue
        support = parse_optional_float(row.get("primary_donor_support"))
        line_width = 1.3 if support is None else 0.7 + 1.9 * support
        alpha = 0.7 if support is None else 0.18 + 0.78 * support
        ax.plot(
            [x_positions[parent], x_positions[taxon]],
            [y_positions[parent], y_positions[taxon]],
            color=BACKBONE_EDGE_COLOR,
            linewidth=line_width,
            alpha=alpha,
            zorder=1,
        )

    for edge in edges:
        if edge.get("is_primary_donor") == "1":
            continue
        donor = edge["donor"]
        recipient = edge["recipient"]
        if donor not in x_positions or recipient not in x_positions:
            continue
        weight = float(edge["borrowed_trait_count"])
        linewidth = 0.7 + 0.22 * weight
        dx = x_positions[recipient] - x_positions[donor]
        if abs(dx) < 0.3:
            rad = 0.24
        else:
            rad = 0.16 if dx >= 0 else -0.16
        patch = FancyArrowPatch(
            (x_positions[donor], y_positions[donor]),
            (x_positions[recipient], y_positions[recipient]),
            arrowstyle="-|>",
            mutation_scale=7,
            linewidth=linewidth,
            linestyle=(0, (4, 3)),
            color=DONOR_EDGE_COLOR,
            alpha=0.22,
            connectionstyle=f"arc3,rad={rad}",
            zorder=2,
        )
        ax.add_patch(patch)

    for taxon in taxa:
        ax.scatter(
            x_positions[taxon],
            y_positions[taxon],
            s=node_size(taxon, incoming_weight, outgoing_weight),
            color=node_color(taxon, feature_index),
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )

    for taxon in taxa:
        child_count = len(children.get(taxon, []))
        align_right = child_count > 0
        offset = -0.18 if align_right else 0.18
        ax.text(
            x_positions[taxon] + offset,
            y_positions[taxon],
            taxon,
            fontsize=5.7,
            ha="right" if align_right else "left",
            va="center",
            color="#1C1C1C",
            zorder=4,
        )

    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)
    ax.yaxis.set_major_locator(mdates.YearLocator())
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", color="#D9D3C7", linewidth=0.7, alpha=0.7)
    ax.set_xticks([])
    ax.set_ylabel("Release Date", fontsize=11)
    ax.set_title(title, fontsize=16, pad=18)

    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#6E6250")

    legend_handles = [
        Line2D([0], [0], color=BACKBONE_EDGE_COLOR, lw=1.6, label="Primary donor"),
        Line2D(
            [0],
            [0],
            color=DONOR_EDGE_COLOR,
            lw=1.6,
            linestyle=(0, (4, 3)),
            label="Other donor",
        )
    ]
    for topology, color in TOPOLOGY_COLORS.items():
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=color,
                markeredgecolor="white",
                markersize=8,
                label=topology,
            )
        )
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=DEFAULT_NODE_COLOR,
            markeredgecolor="white",
            markersize=8,
            label="other topology",
        )
    )
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=8,
        title="Edge / Node Color",
        title_fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the donor-only canonical network as a time-scaled PNG."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=FEATURES_CSV,
        help=f"Canonical feature matrix (default: {FEATURES_CSV})",
    )
    parser.add_argument(
        "--explanations",
        type=Path,
        default=EXPLANATIONS_CSV,
        help=f"Donor explanation table (default: {EXPLANATIONS_CSV})",
    )
    parser.add_argument(
        "--edges",
        type=Path,
        default=EDGES_CSV,
        help=f"Donor edge table (default: {EDGES_CSV})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PNG,
        help=f"PNG output path (default: {OUTPUT_PNG})",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Canonical Donor Network",
        help="Figure title",
    )
    args = parser.parse_args()

    render(
        features=load_csv(args.features),
        explanations=load_csv(args.explanations),
        edges=load_csv(args.edges),
        output_path=args.output,
        title=args.title,
    )
    print(f"Wrote PNG: {args.output}")


if __name__ == "__main__":
    main()
