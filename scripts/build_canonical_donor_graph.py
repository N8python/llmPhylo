#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp


ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT / "canonical_features.csv"
EXPLANATIONS_OUTPUT = ROOT / "canonical_donor_taxon_explanations.csv"
EDGES_OUTPUT = ROOT / "canonical_donor_edges.csv"
SUMMARY_OUTPUT = ROOT / "canonical_donor_summary.md"

DEFAULT_DONOR_COST = 2.0
DEFAULT_BORROW_COST = 0.10
DEFAULT_INNOVATION_COST = 1.0

IDENTIFIER_FIELDS = {"taxon", "release_date"}

MOE_DEPENDENT_FIELDS = {
    "moe_distribution",
    "moe_schedule_type",
    "moe_subtype",
    "router_activation",
    "router_activation_configurable",
    "has_shared_expert",
    "shared_branch_merge_mode",
    "shared_expert_configurable",
    "expert_bias_correction",
    "zero_expert_option",
    "coefficient_mix_shared_routed",
    "grouped_expert_selection",
}

SHARED_EXPERT_DEPENDENT_FIELDS = {
    "shared_branch_merge_mode",
    "coefficient_mix_shared_routed",
}

ROPE_DEPENDENT_FIELDS = {
    "windowed_rope_scope",
    "rope_partition_detail",
    "rope_parameterization",
    "periodic_no_rope_layers",
    "optional_nope",
}

QK_NORM_DEPENDENT_FIELDS = {
    "qk_norm_detail",
}


@dataclass(frozen=True)
class Taxon:
    taxon: str
    raw_date: str
    resolved_date: dt.date
    date_precision: str
    traits: dict[str, str]


@dataclass
class TaxonExplanation:
    taxon: Taxon
    applicable_traits: tuple[str, ...]
    primary_donor: str | None = None
    donor_traits: dict[str, list[str]] = field(default_factory=dict)
    innovations: list[str] = field(default_factory=list)
    total_score: float = 0.0

    @property
    def donor_count(self) -> int:
        return len(self.donor_traits)

    @property
    def borrowed_trait_count(self) -> int:
        return sum(len(traits) for traits in self.donor_traits.values())

    @property
    def innovation_count(self) -> int:
        return len(self.innovations)


def resolve_release_date(raw: str) -> tuple[dt.date, str]:
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        return dt.date.fromisoformat(raw), "exact_day"
    if re.fullmatch(r"\d{4}-\d{2}", raw):
        year, month = map(int, raw.split("-"))
        return dt.date(year, month, 15), "month_midpoint"
    if re.fullmatch(r"\d{4}", raw):
        return dt.date(int(raw), 7, 2), "year_midpoint"
    raise ValueError(f"Unsupported date format: {raw}")


def load_taxa(path: Path) -> tuple[list[Taxon], list[str]]:
    with path.open() as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if not reader.fieldnames:
            raise ValueError(f"No CSV header found in {path}")
        trait_fields = [field for field in reader.fieldnames if field not in IDENTIFIER_FIELDS]

    taxa: list[Taxon] = []
    for row in rows:
        resolved_date, precision = resolve_release_date(row["release_date"])
        taxa.append(
            Taxon(
                taxon=row["taxon"].strip(),
                raw_date=row["release_date"].strip(),
                resolved_date=resolved_date,
                date_precision=precision,
                traits={field: row[field].strip() for field in trait_fields},
            )
        )

    taxa.sort(key=lambda item: (item.resolved_date, item.taxon))
    return taxa, trait_fields


def applicable_traits(taxon: Taxon, trait_fields: list[str]) -> tuple[str, ...]:
    has_moe = taxon.traits.get("has_moe") == "1"
    has_shared_expert = taxon.traits.get("has_shared_expert") == "1"
    uses_rope = "rope" in taxon.traits.get("positional_encoding", "")
    has_qk_norm = taxon.traits.get("qk_norm") not in {"", "?", "none"}

    applicable: list[str] = []
    for field in trait_fields:
        value = taxon.traits[field]
        if value in {"", "?"}:
            continue
        if field in MOE_DEPENDENT_FIELDS and not has_moe:
            continue
        if field in SHARED_EXPERT_DEPENDENT_FIELDS and not has_shared_expert:
            continue
        if field in ROPE_DEPENDENT_FIELDS and not uses_rope:
            continue
        if field in QK_NORM_DEPENDENT_FIELDS and not has_qk_norm:
            continue
        applicable.append(field)
    return tuple(applicable)


def older_taxa_for(child: Taxon, taxa: list[Taxon]) -> list[Taxon]:
    return [taxon for taxon in taxa if taxon.resolved_date < child.resolved_date]


def greedy_donor_assignment(
    child: Taxon,
    older_taxa: list[Taxon],
    applicable: tuple[str, ...],
    donor_cost: float,
    borrow_cost: float,
    innovation_cost: float,
) -> tuple[dict[str, list[str]], list[str]]:
    uncovered = set(applicable)
    donor_traits: dict[str, list[str]] = {}

    while uncovered:
        best_donor: Taxon | None = None
        best_cover: list[str] = []
        best_savings = 0.0

        for donor in older_taxa:
            cover = sorted(
                field
                for field in uncovered
                if donor.traits.get(field, "?") == child.traits.get(field, "?")
            )
            if not cover:
                continue

            savings = len(cover) * (innovation_cost - borrow_cost) - donor_cost
            if savings > best_savings + 1e-12:
                best_donor = donor
                best_cover = cover
                best_savings = savings
                continue

            if best_donor is None or abs(savings - best_savings) > 1e-12:
                continue

            if len(cover) > len(best_cover):
                best_donor = donor
                best_cover = cover
                continue

            if len(cover) < len(best_cover):
                continue

            current_gap = (child.resolved_date - best_donor.resolved_date).days
            candidate_gap = (child.resolved_date - donor.resolved_date).days
            if candidate_gap < current_gap:
                best_donor = donor
                best_cover = cover
            elif candidate_gap == current_gap and donor.taxon < best_donor.taxon:
                best_donor = donor
                best_cover = cover

        if best_donor is None or best_savings <= 1e-12:
            break

        donor_traits[best_donor.taxon] = best_cover
        uncovered.difference_update(best_cover)

    return donor_traits, sorted(uncovered)


def exact_donor_assignment(
    child: Taxon,
    older_taxa: list[Taxon],
    applicable: tuple[str, ...],
    donor_cost: float,
    borrow_cost: float,
    innovation_cost: float,
) -> tuple[dict[str, list[str]], list[str]]:
    if not applicable:
        return {}, []

    candidate_donors: list[Taxon] = []
    donor_covers: list[tuple[str, ...]] = []
    for donor in older_taxa:
        cover = tuple(
            field
            for field in applicable
            if donor.traits.get(field, "?") == child.traits.get(field, "?")
        )
        if cover:
            candidate_donors.append(donor)
            donor_covers.append(cover)

    if not candidate_donors:
        return {}, list(applicable)

    trait_to_idx = {field: idx for idx, field in enumerate(applicable)}
    n_donors = len(candidate_donors)
    n_traits = len(applicable)

    assignment_pairs: list[tuple[int, int]] = []
    assignments_by_trait: list[list[int]] = [[] for _ in range(n_traits)]
    for donor_idx, cover in enumerate(donor_covers):
        for field in cover:
            trait_idx = trait_to_idx[field]
            assignment_idx = len(assignment_pairs)
            assignment_pairs.append((donor_idx, trait_idx))
            assignments_by_trait[trait_idx].append(assignment_idx)

    n_assignments = len(assignment_pairs)
    n_vars = n_donors + n_assignments + n_traits

    objective = np.zeros(n_vars, dtype=float)
    objective[:n_donors] = donor_cost
    objective[n_donors : n_donors + n_assignments] = borrow_cost
    objective[n_donors + n_assignments :] = innovation_cost

    n_rows = n_traits + n_assignments
    matrix = sparse.lil_array((n_rows, n_vars), dtype=float)
    lower = np.full(n_rows, -np.inf, dtype=float)
    upper = np.full(n_rows, np.inf, dtype=float)

    for trait_idx in range(n_traits):
        row_idx = trait_idx
        for assignment_idx in assignments_by_trait[trait_idx]:
            matrix[row_idx, n_donors + assignment_idx] = 1.0
        matrix[row_idx, n_donors + n_assignments + trait_idx] = 1.0
        lower[row_idx] = 1.0
        upper[row_idx] = 1.0

    for assignment_idx, (donor_idx, _trait_idx) in enumerate(assignment_pairs):
        row_idx = n_traits + assignment_idx
        matrix[row_idx, n_donors + assignment_idx] = 1.0
        matrix[row_idx, donor_idx] = -1.0
        upper[row_idx] = 0.0

    result = milp(
        c=objective,
        integrality=np.ones(n_vars, dtype=int),
        bounds=Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars)),
        constraints=LinearConstraint(matrix.tocsr(), lb=lower, ub=upper),
    )
    if result.x is None or not result.success:
        raise RuntimeError(
            f"Exact donor solve failed for {child.taxon}: status={result.status}, message={result.message}"
        )

    values = result.x
    donor_traits: dict[str, list[str]] = {}
    for assignment_idx, (donor_idx, trait_idx) in enumerate(assignment_pairs):
        if values[n_donors + assignment_idx] <= 0.5:
            continue
        donor_name = candidate_donors[donor_idx].taxon
        donor_traits.setdefault(donor_name, []).append(applicable[trait_idx])

    for donor_name in donor_traits:
        donor_traits[donor_name].sort()

    innovations = [
        applicable[trait_idx]
        for trait_idx in range(n_traits)
        if values[n_donors + n_assignments + trait_idx] > 0.5
    ]
    return donor_traits, innovations


def solve_greedy(
    taxa: list[Taxon],
    trait_fields: list[str],
    donor_cost: float,
    borrow_cost: float,
    innovation_cost: float,
) -> tuple[dict[str, TaxonExplanation], float]:
    explanations: dict[str, TaxonExplanation] = {}
    total_score = 0.0
    taxon_index = {taxon.taxon: taxon for taxon in taxa}

    for taxon in taxa:
        applicable = applicable_traits(taxon, trait_fields)
        older = older_taxa_for(taxon, taxa)
        donor_traits, innovations = greedy_donor_assignment(
            child=taxon,
            older_taxa=older,
            applicable=applicable,
            donor_cost=donor_cost,
            borrow_cost=borrow_cost,
            innovation_cost=innovation_cost,
        )
        primary_donor = choose_primary_donor(taxon, donor_traits, taxon_index)
        explanation = TaxonExplanation(
            taxon=taxon,
            applicable_traits=applicable,
            primary_donor=primary_donor,
            donor_traits=donor_traits,
            innovations=innovations,
            total_score=(
                donor_cost * len(donor_traits)
                + borrow_cost * sum(len(traits) for traits in donor_traits.values())
                + innovation_cost * len(innovations)
            ),
        )
        explanations[taxon.taxon] = explanation
        total_score += explanation.total_score

    return explanations, total_score


def solve_exact(
    taxa: list[Taxon],
    trait_fields: list[str],
    donor_cost: float,
    borrow_cost: float,
    innovation_cost: float,
) -> tuple[dict[str, TaxonExplanation], float]:
    explanations: dict[str, TaxonExplanation] = {}
    total_score = 0.0
    taxon_index = {taxon.taxon: taxon for taxon in taxa}

    for taxon in taxa:
        applicable = applicable_traits(taxon, trait_fields)
        older = older_taxa_for(taxon, taxa)
        donor_traits, innovations = exact_donor_assignment(
            child=taxon,
            older_taxa=older,
            applicable=applicable,
            donor_cost=donor_cost,
            borrow_cost=borrow_cost,
            innovation_cost=innovation_cost,
        )
        primary_donor = choose_primary_donor(taxon, donor_traits, taxon_index)
        explanation = TaxonExplanation(
            taxon=taxon,
            applicable_traits=applicable,
            primary_donor=primary_donor,
            donor_traits=donor_traits,
            innovations=innovations,
            total_score=(
                donor_cost * len(donor_traits)
                + borrow_cost * sum(len(traits) for traits in donor_traits.values())
                + innovation_cost * len(innovations)
            ),
        )
        explanations[taxon.taxon] = explanation
        total_score += explanation.total_score

    return explanations, total_score


def choose_primary_donor(
    child: Taxon,
    donor_traits: dict[str, list[str]],
    taxon_index: dict[str, Taxon],
) -> str | None:
    if not donor_traits:
        return None
    ranked = sorted(
        donor_traits.items(),
        key=lambda item: (
            -len(item[1]),
            (child.resolved_date - taxon_index[item[0]].resolved_date).days,
            item[0],
        ),
    )
    return ranked[0][0]


def format_donor_map(donor_traits: dict[str, list[str]]) -> str:
    if not donor_traits:
        return ""
    return ";".join(
        f"{donor}:{'|'.join(traits)}" for donor, traits in sorted(donor_traits.items())
    )


def explanation_score_delta(
    explanation: TaxonExplanation,
    baseline: TaxonExplanation | None,
) -> tuple[str, str]:
    if baseline is None:
        return "", ""
    improvement = baseline.total_score - explanation.total_score
    return f"{baseline.total_score:.2f}", f"{improvement:.2f}"


def write_explanations(
    taxa: list[Taxon],
    explanations: dict[str, TaxonExplanation],
    greedy_explanations: dict[str, TaxonExplanation],
    output_path: Path,
    donor_cost: float,
    borrow_cost: float,
    innovation_cost: float,
) -> None:
    fieldnames = [
        "taxon",
        "raw_date",
        "resolved_date",
        "date_precision",
        "older_taxon_count",
        "applicable_trait_count",
        "primary_donor",
        "primary_donor_date",
        "primary_donor_gap_days",
        "primary_donor_borrowed_trait_count",
        "secondary_donor_count",
        "borrowed_trait_count",
        "innovation_count",
        "secondary_donor_cost_total",
        "trait_borrow_cost_total",
        "innovation_cost_total",
        "total_score",
        "greedy_total_score",
        "score_improvement_vs_greedy",
        "innovations",
        "donors",
    ]
    taxon_index = {taxon.taxon: taxon for taxon in taxa}
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for taxon in taxa:
            explanation = explanations[taxon.taxon]
            greedy_explanation = greedy_explanations.get(taxon.taxon)
            greedy_total_score, score_improvement = explanation_score_delta(
                explanation,
                greedy_explanation,
            )
            if explanation.primary_donor is None:
                primary_donor = ""
                primary_donor_date = ""
                primary_donor_gap_days = ""
                primary_donor_borrowed_trait_count = ""
            else:
                donor_taxon = taxon_index[explanation.primary_donor]
                primary_donor = donor_taxon.taxon
                primary_donor_date = donor_taxon.resolved_date.isoformat()
                primary_donor_gap_days = (taxon.resolved_date - donor_taxon.resolved_date).days
                primary_donor_borrowed_trait_count = len(
                    explanation.donor_traits[explanation.primary_donor]
                )
            writer.writerow(
                {
                    "taxon": taxon.taxon,
                    "raw_date": taxon.raw_date,
                    "resolved_date": taxon.resolved_date.isoformat(),
                    "date_precision": taxon.date_precision,
                    "older_taxon_count": len(older_taxa_for(taxon, taxa)),
                    "applicable_trait_count": len(explanation.applicable_traits),
                    "primary_donor": primary_donor,
                    "primary_donor_date": primary_donor_date,
                    "primary_donor_gap_days": primary_donor_gap_days,
                    "primary_donor_borrowed_trait_count": primary_donor_borrowed_trait_count,
                    "secondary_donor_count": explanation.donor_count,
                    "borrowed_trait_count": explanation.borrowed_trait_count,
                    "innovation_count": explanation.innovation_count,
                    "secondary_donor_cost_total": f"{explanation.donor_count * donor_cost:.2f}",
                    "trait_borrow_cost_total": f"{explanation.borrowed_trait_count * borrow_cost:.2f}",
                    "innovation_cost_total": f"{explanation.innovation_count * innovation_cost:.2f}",
                    "total_score": f"{explanation.total_score:.2f}",
                    "greedy_total_score": greedy_total_score,
                    "score_improvement_vs_greedy": score_improvement,
                    "innovations": "|".join(explanation.innovations),
                    "donors": format_donor_map(explanation.donor_traits),
                }
            )


def write_edges(
    taxa: list[Taxon],
    explanations: dict[str, TaxonExplanation],
    output_path: Path,
) -> None:
    fieldnames = [
        "donor",
        "recipient",
        "donor_date",
        "recipient_date",
        "gap_days",
        "is_primary_donor",
        "borrowed_trait_count",
        "borrowed_traits",
    ]
    taxon_index = {taxon.taxon: taxon for taxon in taxa}
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for recipient in taxa:
            explanation = explanations[recipient.taxon]
            for donor_name, traits in sorted(explanation.donor_traits.items()):
                donor = taxon_index[donor_name]
                writer.writerow(
                    {
                        "donor": donor_name,
                        "recipient": recipient.taxon,
                        "donor_date": donor.resolved_date.isoformat(),
                        "recipient_date": recipient.resolved_date.isoformat(),
                        "gap_days": (recipient.resolved_date - donor.resolved_date).days,
                        "is_primary_donor": int(donor_name == explanation.primary_donor),
                        "borrowed_trait_count": len(traits),
                        "borrowed_traits": "|".join(traits),
                    }
                )


def write_summary(
    taxa: list[Taxon],
    trait_fields: list[str],
    explanations: dict[str, TaxonExplanation],
    greedy_explanations: dict[str, TaxonExplanation],
    total_score: float,
    greedy_total_score: float,
    output_path: Path,
    donor_cost: float,
    borrow_cost: float,
    innovation_cost: float,
) -> None:
    donor_usage = Counter()
    donor_traits = Counter()
    borrowed_traits = 0
    innovations = 0
    no_donor_taxa = 0
    precision_counts = Counter(taxon.date_precision for taxon in taxa)
    improved_taxa: list[tuple[float, TaxonExplanation, TaxonExplanation]] = []

    for explanation in explanations.values():
        borrowed_traits += explanation.borrowed_trait_count
        innovations += explanation.innovation_count
        if explanation.donor_count == 0:
            no_donor_taxa += 1
        for donor, traits in explanation.donor_traits.items():
            donor_usage[donor] += 1
            donor_traits[donor] += len(traits)
        greedy_explanation = greedy_explanations.get(explanation.taxon.taxon)
        if greedy_explanation is not None:
            improvement = greedy_explanation.total_score - explanation.total_score
            if improvement > 1e-9:
                improved_taxa.append((improvement, explanation, greedy_explanation))

    borrow_heavy = sorted(
        explanations.values(),
        key=lambda explanation: (
            -explanation.borrowed_trait_count,
            -explanation.donor_count,
            -explanation.innovation_count,
            explanation.taxon.resolved_date,
            explanation.taxon.taxon,
        ),
    )
    score_improvement = greedy_total_score - total_score
    percent_improvement = (
        (score_improvement / greedy_total_score) * 100.0 if greedy_total_score else 0.0
    )
    improved_taxa.sort(
        key=lambda item: (
            -item[0],
            item[1].taxon.resolved_date,
            item[1].taxon.taxon,
        )
    )

    lines = [
        "# Canonical Donor Graph",
        "",
        f"- Input matrix: {INPUT_CSV}",
        f"- Taxa: {len(taxa)}",
        f"- Trait fields scored: {len(trait_fields)}",
        "- Optimizer: exact per-taxon MILP via SciPy HiGHS; global optimum is the sum of those exact local optima because taxa decouple under the current dated donor-only objective.",
        "- Objective: minimize donor activation cost + per-trait borrow cost + innovation cost.",
        "- Constraint: donors must have strictly earlier release dates than recipients.",
        "- Model: no primary parent; every applicable trait is either borrowed from an older donor or treated as an innovation.",
        "- Display parent rule: donor with the most borrowed traits; ties break by shortest temporal gap, then taxon name.",
        f"- Secondary donor activation cost: {donor_cost:.2f} per donor taxon",
        f"- Borrowed trait cost: {borrow_cost:.2f} per trait",
        f"- Innovation cost: {innovation_cost:.2f} per trait",
        f"- Total score: {total_score:.2f}",
        f"- Greedy baseline total score: {greedy_total_score:.2f}",
        f"- Improvement vs greedy: {score_improvement:.2f} ({percent_improvement:.2f}%)",
        f"- Taxa improved vs greedy: {len(improved_taxa)}",
        f"- Donor edges used: {sum(explanation.donor_count for explanation in explanations.values())}",
        f"- Borrowed traits explained: {borrowed_traits}",
        f"- Innovations required: {innovations}",
        f"- Taxa with no donor selected: {no_donor_taxa}",
        f"- Release-date precision mix: "
        + ", ".join(f"{precision}={precision_counts[precision]}" for precision in sorted(precision_counts)),
        "",
        "## Applicability Filters",
        "",
        "- MoE downstream fields are ignored when `has_moe=0`.",
        "- Shared-expert merge details are ignored when `has_shared_expert=0`.",
        "- Rope-detail fields are ignored when the positional encoding is not RoPE-based.",
        "- `qk_norm_detail` is ignored when `qk_norm=none`.",
        "",
        "## Top Donors",
        "",
    ]

    for donor, recipient_count in donor_usage.most_common(12):
        lines.append(
            f"- {donor}: donor to {recipient_count} taxa, {donor_traits[donor]} borrowed traits"
        )

    lines.extend(["", "## Borrow-Heavy Taxa", ""])
    for explanation in borrow_heavy[:12]:
        donor_preview = ", ".join(
            f"{donor}({len(traits)})"
            for donor, traits in sorted(
                explanation.donor_traits.items(),
                key=lambda item: (-len(item[1]), item[0]),
            )
        ) or "none"
        innovation_preview = ", ".join(explanation.innovations[:8]) if explanation.innovations else "none"
        lines.append(
            f"- {explanation.taxon.taxon}: donors {donor_preview}, innovations {innovation_preview}, score {explanation.total_score:.2f}"
        )

    lines.extend(["", "## Largest Improvements Vs Greedy", ""])
    if improved_taxa:
        for improvement, explanation, greedy_explanation in improved_taxa[:12]:
            lines.append(
                f"- {explanation.taxon.taxon}: exact {explanation.total_score:.2f} vs greedy {greedy_explanation.total_score:.2f}, improvement {improvement:.2f}"
            )
    else:
        lines.append("- None. The exact solve matched the greedy heuristic on every taxon.")

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- {EXPLANATIONS_OUTPUT.name}",
            f"- {EDGES_OUTPUT.name}",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Infer a donor-only chronological graph over canonical_features.csv."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_CSV,
        help=f"Canonical feature matrix (default: {INPUT_CSV})",
    )
    parser.add_argument(
        "--donor-cost",
        type=float,
        default=DEFAULT_DONOR_COST,
        help=f"Activation cost for each donor taxon (default: {DEFAULT_DONOR_COST})",
    )
    parser.add_argument(
        "--borrow-cost",
        type=float,
        default=DEFAULT_BORROW_COST,
        help=f"Per-trait cost for borrowing (default: {DEFAULT_BORROW_COST})",
    )
    parser.add_argument(
        "--innovation-cost",
        type=float,
        default=DEFAULT_INNOVATION_COST,
        help=f"Per-trait cost for innovation (default: {DEFAULT_INNOVATION_COST})",
    )
    args = parser.parse_args()

    taxa, trait_fields = load_taxa(args.input)
    greedy_explanations, greedy_total_score = solve_greedy(
        taxa=taxa,
        trait_fields=trait_fields,
        donor_cost=args.donor_cost,
        borrow_cost=args.borrow_cost,
        innovation_cost=args.innovation_cost,
    )
    explanations, total_score = solve_exact(
        taxa=taxa,
        trait_fields=trait_fields,
        donor_cost=args.donor_cost,
        borrow_cost=args.borrow_cost,
        innovation_cost=args.innovation_cost,
    )
    write_explanations(
        taxa=taxa,
        explanations=explanations,
        greedy_explanations=greedy_explanations,
        output_path=EXPLANATIONS_OUTPUT,
        donor_cost=args.donor_cost,
        borrow_cost=args.borrow_cost,
        innovation_cost=args.innovation_cost,
    )
    write_edges(taxa=taxa, explanations=explanations, output_path=EDGES_OUTPUT)
    write_summary(
        taxa=taxa,
        trait_fields=trait_fields,
        explanations=explanations,
        greedy_explanations=greedy_explanations,
        total_score=total_score,
        greedy_total_score=greedy_total_score,
        output_path=SUMMARY_OUTPUT,
        donor_cost=args.donor_cost,
        borrow_cost=args.borrow_cost,
        innovation_cost=args.innovation_cost,
    )

    print(f"Wrote taxon explanations: {EXPLANATIONS_OUTPUT}")
    print(f"Wrote donor edges: {EDGES_OUTPUT}")
    print(f"Wrote summary: {SUMMARY_OUTPUT}")
    print(f"Greedy baseline total score: {greedy_total_score:.2f}")
    print(f"Exact total score: {total_score:.2f}")
    print(f"Improvement vs greedy: {greedy_total_score - total_score:.2f}")


if __name__ == "__main__":
    main()
