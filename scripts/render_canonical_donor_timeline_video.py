#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import math
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.patches import FancyArrowPatch

from render_canonical_donor_network import (
    DONOR_EDGE_COLOR,
    FEATURES_CSV,
    assign_tree_x_positions,
    build_node_index,
    build_primary_children,
    load_csv,
    parse_date,
    parse_optional_float,
)


ROOT = Path(__file__).resolve().parent.parent
EXPLANATIONS_CSV = ROOT / "canonical_donor_consensus_explanations.csv"
EDGES_CSV = ROOT / "canonical_donor_consensus_edges.csv"
OUTPUT_MP4 = ROOT / "canonical_donor_consensus_evolution.mp4"

FIG_WIDTH = 16
FIG_HEIGHT = 12
FIG_DPI = 180
FPS = 15
FRAME_COUNT = 900
START_HOLD_FRAMES = 0
END_HOLD_FRAMES = 36
LABEL_FADE_SECONDS = 0.5
SETTLE_DAYS = 45
DONOR_EVENT_WINDOW_DAYS = 45
MAX_VISIBLE_DONOR_EVENTS = 8
TIME_WARP_BLEND = 0.28
TIME_WARP_SIGMA_DAYS = 75
DISPLAY_END_DATE = dt.date(2026, 6, 30)
ROOT_LABEL_X_OFFSET = 0.028
ROOT_LABEL_FONT = 7.0
ROOT_LABEL_HEIGHT = 0.88
ROOT_LABEL_BASE_WIDTH = 0.022
ROOT_LABEL_WIDTH_PER_CHAR = 0.0042
ROOT_LABEL_PADDING_X = 0.007
ROOT_LABEL_PADDING_Y = 0.10
ROOT_LABEL_SEARCH_RADIUS = 12
ROOT_LABEL_BASE_OFFSET = 0.62
TEXT_STROKE = [pe.withStroke(linewidth=3.2, foreground="#050505")]

LINE_PALETTE = [
    "#4CC9F0",
    "#F72585",
    "#B8F200",
    "#F77F00",
    "#80ED99",
    "#9B5DE5",
    "#FFCA3A",
    "#00BBF9",
    "#E5383B",
    "#06D6A0",
    "#FF99C8",
    "#90BE6D",
    "#C77DFF",
    "#FFD166",
    "#EF476F",
    "#118AB2",
]


def ease_in_out(progress: float) -> float:
    clamped = max(0.0, min(1.0, progress))
    return 3.0 * clamped**2 - 2.0 * clamped**3


class TimeWarp:
    def __init__(
        self,
        first_date: dt.date,
        last_date: dt.date,
        daily_event_weights: dict[dt.date, float],
        blend: float = TIME_WARP_BLEND,
        sigma_days: float = TIME_WARP_SIGMA_DAYS,
    ) -> None:
        self.first_date = first_date
        self.last_date = last_date
        total_days = max(1, (last_date - first_date).days)
        self.total_days = total_days
        daily = np.zeros(total_days + 1, dtype=float)
        for date, weight in daily_event_weights.items():
            index = max(0, min(total_days, (date - first_date).days))
            daily[index] += weight

        radius = max(1, int(sigma_days * 3))
        xs = np.arange(-radius, radius + 1, dtype=float)
        kernel = np.exp(-0.5 * (xs / sigma_days) ** 2)
        kernel /= kernel.sum()
        smooth = np.convolve(daily, kernel, mode="same")
        smooth_norm = smooth / max(np.mean(smooth), 1e-9)
        density = blend + (1.0 - blend) * smooth_norm
        cumulative = np.cumsum(density)
        cumulative -= cumulative[0]
        self.positions = cumulative / max(cumulative[-1], 1e-9)

    def transform(self, value: dt.date) -> float:
        if value <= self.first_date:
            return 0.0
        if value >= self.last_date:
            return 1.0
        day_float = (value - self.first_date).days
        lower = int(math.floor(day_float))
        upper = min(self.total_days, lower + 1)
        frac = day_float - lower
        return float(self.positions[lower] + frac * (self.positions[upper] - self.positions[lower]))

    def inverse(self, position: float) -> dt.date:
        clamped = max(0.0, min(1.0, position))
        index = int(np.searchsorted(self.positions, clamped, side="left"))
        if index <= 0:
            return self.first_date
        if index >= len(self.positions):
            return self.last_date
        lower_pos = self.positions[index - 1]
        upper_pos = self.positions[index]
        frac = 0.0 if upper_pos <= lower_pos else (clamped - lower_pos) / (upper_pos - lower_pos)
        day_offset = (index - 1) + frac
        return self.first_date + dt.timedelta(days=day_offset)

    def transform_extended(self, value: dt.date) -> float:
        if self.first_date <= value <= self.last_date:
            return self.transform(value)
        if value < self.first_date:
            return -(self.first_date - value).days / max(1.0, float(self.total_days))
        return 1.0 + (value - self.last_date).days / max(1.0, float(self.total_days))


def event_weight(row: dict[str, str]) -> float:
    donor_count = float(row.get("secondary_donor_count", "0") or 0.0)
    innovation_count = float(row.get("innovation_count", "0") or 0.0)
    return 1.0 + 0.30 * donor_count + 0.03 * innovation_count


def cutoff_for_frame(frame_index: int) -> float:
    if frame_index < START_HOLD_FRAMES:
        return 0.0
    if frame_index >= FRAME_COUNT - END_HOLD_FRAMES:
        return 1.0
    active_frame = frame_index - START_HOLD_FRAMES
    active_span = FRAME_COUNT - START_HOLD_FRAMES - END_HOLD_FRAMES - 1
    return active_frame / active_span if active_span > 0 else 1.0


def active_span_frames() -> int:
    return max(1, FRAME_COUNT - START_HOLD_FRAMES - END_HOLD_FRAMES - 1)


def label_fade_frames() -> int:
    return max(1, int(round(FPS * LABEL_FADE_SECONDS)))


def appearance_frame_for_position(position: float) -> float:
    if position <= 0.0:
        return 0.0
    if position >= 1.0:
        return float(FRAME_COUNT - END_HOLD_FRAMES - 1)
    return float(START_HOLD_FRAMES) + position * float(active_span_frames())


def label_alpha_for_frame(frame_index: int, release_x: float) -> float:
    appear_frame = appearance_frame_for_position(release_x)
    return max(0.0, min(1.0, (frame_index - appear_frame) / float(label_fade_frames())))


def line_color(taxon: str, y_rank: dict[str, float]) -> str:
    return LINE_PALETTE[int(y_rank[taxon]) % len(LINE_PALETTE)]


def outgoing_weights(edges: list[dict[str, str]]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for edge in edges:
        donor = edge["donor"]
        weights[donor] = weights.get(donor, 0.0) + float(edge["borrowed_trait_count"])
    return weights


def child_counts(explanations: dict[str, dict[str, str]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in explanations.values():
        parent = row.get("primary_donor", "")
        if parent:
            counts[parent] += 1
    return counts


def primary_edge_cost(order: list[str], explanation_index: dict[str, dict[str, str]]) -> float:
    index = {taxon: position for position, taxon in enumerate(order)}
    cost = 0.0
    for taxon, row in explanation_index.items():
        parent = row.get("primary_donor", "")
        if not parent:
            continue
        support = parse_optional_float(row.get("primary_donor_support")) or 0.0
        cost += (1.0 + support) * abs(index[taxon] - index[parent])
    return cost


def optimize_taxon_order(initial_order: list[str], explanation_index: dict[str, dict[str, str]]) -> list[str]:
    order = list(initial_order)
    if len(order) < 3:
        return order

    improved = True
    while improved:
        improved = False
        baseline = primary_edge_cost(order, explanation_index)
        for index in range(len(order) - 1):
            swapped = list(order)
            swapped[index], swapped[index + 1] = swapped[index + 1], swapped[index]
            new_cost = primary_edge_cost(swapped, explanation_index)
            if new_cost + 1e-9 < baseline:
                order = swapped
                baseline = new_cost
                improved = True
    return order


def branch_path(
    taxon: str,
    cutoff_position: float,
    y_rank: dict[str, float],
    explanation_index: dict[str, dict[str, str]],
    time_warp: TimeWarp,
) -> tuple[list[float], list[float]] | None:
    release_date = parse_date(explanation_index[taxon]["resolved_date"])
    release_x = time_warp.transform(release_date)
    if cutoff_position < release_x:
        return None

    current_x = cutoff_position
    y_self = y_rank[taxon]
    parent = explanation_index[taxon].get("primary_donor", "")
    if not parent:
        return [release_x, current_x], [y_self, y_self]

    y_parent = y_rank[parent]
    return [release_x, release_x, current_x], [y_parent, y_self, y_self]


def donor_event_alpha(cutoff: dt.date, release_date: dt.date) -> float:
    delta = (cutoff - release_date).days
    if delta < 0 or delta > DONOR_EVENT_WINDOW_DAYS:
        return 0.0
    return 1.0 - delta / DONOR_EVENT_WINDOW_DAYS


def visible_taxa_at(cutoff_date: dt.date, taxa: list[str], explanation_index: dict[str, dict[str, str]]) -> list[str]:
    return [
        taxon
        for taxon in taxa
        if parse_date(explanation_index[taxon]["resolved_date"]) <= cutoff_date
    ]


def release_progress(taxon: str, explanation_index: dict[str, dict[str, str]], time_warp: TimeWarp) -> float:
    return time_warp.transform(parse_date(explanation_index[taxon]["resolved_date"]))


def label_rect(label_x: float, label_y: float, side: str, taxon: str) -> tuple[float, float, float, float]:
    width = ROOT_LABEL_BASE_WIDTH + ROOT_LABEL_WIDTH_PER_CHAR * len(taxon)
    if side == "right":
        x0 = label_x - 0.002
        x1 = label_x + width
    else:
        x0 = label_x - width
        x1 = label_x + 0.002
    y0 = label_y - ROOT_LABEL_HEIGHT / 2.0
    y1 = label_y + ROOT_LABEL_HEIGHT / 2.0
    return x0, x1, y0, y1


def rects_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    ax0, ax1, ay0, ay1 = a
    bx0, bx1, by0, by1 = b
    return not (
        ax1 + ROOT_LABEL_PADDING_X < bx0
        or ax0 > bx1 + ROOT_LABEL_PADDING_X
        or ay1 + ROOT_LABEL_PADDING_Y < by0
        or ay0 > by1 + ROOT_LABEL_PADDING_Y
    )


def label_y_offsets(radius: int) -> list[float]:
    offsets: list[float] = []
    for step in range(1, radius + 1):
        distance = ROOT_LABEL_BASE_OFFSET + float(step - 1)
        offsets.append(distance)
        offsets.append(-distance)
    offsets.append(0.0)
    return offsets


def compute_release_label_positions(
    taxa: list[str],
    explanation_index: dict[str, dict[str, str]],
    y_rank: dict[str, float],
    time_warp: TimeWarp,
) -> dict[str, tuple[float, float, str]]:
    positions: dict[str, tuple[float, float, str]] = {}
    for taxon in taxa:
        base_x = release_progress(taxon, explanation_index, time_warp)
        base_y = y_rank[taxon]
        side = "right" if base_x < 0.90 else "left"
        label_x = base_x + 0.003 if side == "right" else base_x - 0.003
        positions[taxon] = (label_x, base_y, side)
    return positions


def latest_release(cutoff: dt.date, taxa: list[str], explanation_index: dict[str, dict[str, str]]) -> str | None:
    visible = [
        taxon
        for taxon in taxa
        if parse_date(explanation_index[taxon]["resolved_date"]) <= cutoff
    ]
    if not visible:
        return None
    return max(visible, key=lambda taxon: (parse_date(explanation_index[taxon]["resolved_date"]), taxon))


def render_frame(
    frame_path: Path,
    frame_index: int,
    cutoff_date: dt.date,
    cutoff_position: float,
    first_date: dt.date,
    last_date: dt.date,
    display_end_date: dt.date,
    taxa: list[str],
    feature_index: dict[str, dict[str, str]],
    explanation_index: dict[str, dict[str, str]],
    y_rank: dict[str, float],
    edges: list[dict[str, str]],
    influence_score: dict[str, float],
    child_count: Counter[str],
    time_warp: TimeWarp,
    release_label_positions: dict[str, tuple[float, float, str]],
) -> None:
    cutoff_x = cutoff_position
    first_x = -0.03
    last_axis_x = time_warp.transform_extended(display_end_date)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI)
    ax.set_facecolor("#050505")
    fig.patch.set_facecolor("#050505")

    ax.axvline(cutoff_x, color="#E9C46A", linewidth=2.0, alpha=0.9, zorder=1)
    ax.axvspan(cutoff_x, last_axis_x, color="#050505", alpha=0.80, zorder=0)

    visible_taxa = 0
    active_secondary_edges: list[dict[str, object]] = []

    for taxon in taxa:
        path = branch_path(taxon, cutoff_position, y_rank, explanation_index, time_warp)
        if path is None:
            continue
        visible_taxa += 1
        xs, ys = path
        support = parse_optional_float(explanation_index[taxon].get("primary_donor_support")) or 0.0
        color = line_color(taxon, y_rank)
        width = 0.9 + 1.8 * support
        alpha = 0.38 + 0.44 * support
        ax.plot(xs, ys, color=color, linewidth=width, alpha=alpha, zorder=2)
        ax.scatter(xs[-1], ys[-1], s=20 + 18 * support, color=color, edgecolor="#111111", linewidth=0.4, zorder=4)

    for edge in edges:
        if edge.get("is_primary_donor") == "1":
            continue
        recipient = edge["recipient"]
        donor = edge["donor"]
        release_date = parse_date(explanation_index[recipient]["resolved_date"])
        alpha = donor_event_alpha(cutoff_date, release_date)
        if alpha <= 0.0:
            continue
        active_secondary_edges.append(
            {
                "edge": edge,
                "alpha": alpha,
                "score": alpha * float(edge["borrowed_trait_count"]),
            }
        )

    active_secondary_edges.sort(key=lambda item: (-float(item["score"]), item["edge"]["recipient"], item["edge"]["donor"]))
    for item in active_secondary_edges[:MAX_VISIBLE_DONOR_EVENTS]:
        edge = item["edge"]
        alpha = float(item["alpha"])
        recipient = edge["recipient"]
        donor = edge["donor"]
        release_date = parse_date(explanation_index[recipient]["resolved_date"])
        start = (time_warp.transform(release_date), y_rank[donor])
        end = (time_warp.transform(min(release_date + dt.timedelta(days=18), last_date)), y_rank[recipient])
        dy = y_rank[recipient] - y_rank[donor]
        rad = max(-0.35, min(0.35, 0.015 * dy))
        patch = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.5 + 0.12 * float(edge["borrowed_trait_count"]),
            linestyle=(0, (4, 3)),
            color=DONOR_EDGE_COLOR,
            alpha=alpha * 0.30,
            connectionstyle=f"arc3,rad={rad}",
            zorder=3,
        )
        ax.add_patch(patch)

    visible = visible_taxa_at(cutoff_date, taxa, explanation_index)
    for taxon in visible:
        color = line_color(taxon, y_rank)
        release_x = release_progress(taxon, explanation_index, time_warp)
        label_alpha = label_alpha_for_frame(frame_index, release_x)
        if label_alpha <= 0.0:
            continue
        label_x, label_y, side = release_label_positions[taxon]
        ax.text(
            label_x,
            label_y,
            taxon,
            fontsize=ROOT_LABEL_FONT,
            fontweight="bold",
            ha="left" if side == "right" else "right",
            va="center",
            color=(0.964, 0.964, 0.964, label_alpha),
            zorder=6,
            path_effects=TEXT_STROKE,
            bbox={
                "boxstyle": "round,pad=0.14,rounding_size=0.14",
                "facecolor": (0.05, 0.05, 0.05, 0.82 * label_alpha),
                "edgecolor": (*mcolors.to_rgb(color), 0.90 * label_alpha),
                "linewidth": 0.75,
            },
        )

    ax.text(
        0.985,
        0.965,
        cutoff_date.strftime("%b %Y"),
        transform=ax.transAxes,
        fontsize=23,
        fontweight="bold",
        ha="right",
        va="top",
        color="#F4F1DE",
        zorder=7,
        path_effects=TEXT_STROKE,
    )

    y_values = list(y_rank.values())
    ax.set_xlim(first_x, last_axis_x)
    ax.set_ylim(max(y_values) + 4.0, min(y_values) - 8.0)
    tick_dates = [
        dt.date(year, 1, 1)
        for year in range(first_date.year, display_end_date.year + 2)
        if dt.date(year, 1, 1) <= display_end_date
    ]
    ax.set_xticks([time_warp.transform_extended(value) for value in tick_dates], [str(value.year) for value in tick_dates])
    ax.tick_params(axis="x", colors="#D7D7D7", labelsize=10)
    ax.set_yticks([])
    ax.grid(axis="x", color="#363636", linewidth=0.8, alpha=0.65)
    ax.grid(axis="y", color="#151515", linewidth=0.4, alpha=0.25)

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#5A5A5A")

    fig.tight_layout(pad=1.0)
    fig.savefig(frame_path, dpi=FIG_DPI, facecolor=fig.get_facecolor())
    plt.close(fig)


def encode_video(frames_dir: Path, output_path: Path, fps: int) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame-%04d.png"),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a YouTube-style timeline video for the canonical donor consensus network."
    )
    parser.add_argument("--features", type=Path, default=FEATURES_CSV)
    parser.add_argument("--explanations", type=Path, default=EXPLANATIONS_CSV)
    parser.add_argument("--edges", type=Path, default=EDGES_CSV)
    parser.add_argument("--output", type=Path, default=OUTPUT_MP4)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--frames", type=int, default=FRAME_COUNT)
    args = parser.parse_args()

    features = load_csv(args.features)
    explanations = load_csv(args.explanations)
    edges = load_csv(args.edges)
    taxa, feature_index, explanation_index = build_node_index(features, explanations)
    roots, children = build_primary_children(explanation_index)
    scaffold_x = assign_tree_x_positions(roots, children)
    initial_order = sorted(
        taxa,
        key=lambda taxon: (
            scaffold_x[taxon],
            parse_date(explanation_index[taxon]["resolved_date"]),
            taxon,
        ),
    )
    ordered_taxa = optimize_taxon_order(initial_order, explanation_index)
    y_rank = {taxon: float(index) for index, taxon in enumerate(ordered_taxa)}
    influence = outgoing_weights(edges)
    counts = child_counts(explanation_index)
    for taxon in taxa:
        influence[taxon] = influence.get(taxon, 0.0) + 4.0 * counts.get(taxon, 0)

    all_dates = [parse_date(row["resolved_date"]) for row in explanation_index.values()]
    first_date = min(all_dates)
    last_date = max(all_dates)
    display_end_date = max(last_date, DISPLAY_END_DATE)
    daily_event_weights = {
        parse_date(row["resolved_date"]): 0.0
        for row in explanation_index.values()
    }
    for row in explanation_index.values():
        release_date = parse_date(row["resolved_date"])
        daily_event_weights[release_date] = daily_event_weights.get(release_date, 0.0) + event_weight(row)
    time_warp = TimeWarp(first_date, last_date, daily_event_weights)
    release_label_positions = compute_release_label_positions(ordered_taxa, explanation_index, y_rank, time_warp)

    with tempfile.TemporaryDirectory(prefix="canonical-donor-timeline-") as temp_dir:
        frames_dir = Path(temp_dir)
        for frame_index in range(args.frames):
            cutoff_position = cutoff_for_frame(frame_index)
            cutoff_date = time_warp.inverse(cutoff_position)
            frame_path = frames_dir / f"frame-{frame_index:04d}.png"
            render_frame(
                frame_path=frame_path,
                frame_index=frame_index,
                cutoff_date=cutoff_date,
                cutoff_position=cutoff_position,
                first_date=first_date,
                last_date=last_date,
                display_end_date=display_end_date,
                taxa=ordered_taxa,
                feature_index=feature_index,
                explanation_index=explanation_index,
                y_rank=y_rank,
                edges=edges,
                influence_score=influence,
                child_count=counts,
                time_warp=time_warp,
                release_label_positions=release_label_positions,
            )
            if (frame_index + 1) % 20 == 0 or frame_index + 1 == args.frames:
                print(f"Rendered frame {frame_index + 1}/{args.frames}")
        encode_video(frames_dir, args.output, args.fps)

    print(f"Wrote MP4: {args.output}")


if __name__ == "__main__":
    main()
