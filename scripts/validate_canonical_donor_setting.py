#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import statistics
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from build_canonical_donor_graph import DEFAULT_INNOVATION_COST, Taxon, TaxonExplanation, load_taxa, solve_exact
from donor_validation_utils import heldout_metrics, split_trait_folds, stability_trait_subsets, summarize_solution


ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT / "canonical_features.csv"
BASE_EXPLANATIONS_CSV = ROOT / "canonical_donor_taxon_explanations.csv"
BASE_EDGES_CSV = ROOT / "canonical_donor_edges.csv"
VALIDATION_REPORT_MD = ROOT / "canonical_donor_fixed_cost_validation.md"
CONSENSUS_SUPPORT_CSV = ROOT / "canonical_donor_consensus_support.csv"
CONSENSUS_EXPLANATIONS_CSV = ROOT / "canonical_donor_consensus_explanations.csv"
CONSENSUS_EDGES_CSV = ROOT / "canonical_donor_consensus_edges.csv"

DEFAULT_DONOR_COST = 2.0
DEFAULT_BORROW_COST = 0.10
DEFAULT_HOLDOUT_FOLDS = 5
DEFAULT_STABILITY_REPS = 20
DEFAULT_STABILITY_FRACTION = 0.8


@dataclass(frozen=True)
class HoldoutFoldResult:
    fold_index: int
    heldout_trait_count: int
    heldout_cost_mean: float
    heldout_coverage: float


@dataclass(frozen=True)
class StabilityReplicateResult:
    replicate_index: int
    agreement_with_full: float
    primary_map: dict[str, str]


@dataclass(frozen=True)
class ConsensusSupport:
    taxon: str
    consensus_primary: str
    vote_count: int
    support: float
    nonempty_support: float
    nonempty_vote_count: int
    full_primary: str
    full_primary_support: float
    matches_full_primary: bool
    in_full_solution: bool
    vote_breakdown: str


def load_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open() as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if not reader.fieldnames:
            raise ValueError(f"No header found in {path}")
        return rows, list(reader.fieldnames)


def primary_map_stability(
    full_primary_map: dict[str, str],
    subset_primary_map: dict[str, str],
) -> float:
    comparisons = 0
    matches = 0
    for taxon_name, donor_name in full_primary_map.items():
        if not donor_name:
            continue
        comparisons += 1
        if subset_primary_map.get(taxon_name, "") == donor_name:
            matches += 1
    if comparisons == 0:
        return 1.0
    return matches / comparisons


def run_holdout_fold(
    input_path: Path,
    donor_cost: float,
    borrow_cost: float,
    innovation_cost: float,
    fold_index: int,
    heldout_fields: tuple[str, ...],
) -> HoldoutFoldResult:
    taxa, trait_fields = load_taxa(input_path)
    train_fields = [field for field in trait_fields if field not in set(heldout_fields)]
    explanations, _ = solve_exact(
        taxa=taxa,
        trait_fields=train_fields,
        donor_cost=donor_cost,
        borrow_cost=borrow_cost,
        innovation_cost=innovation_cost,
    )
    heldout_cost_mean, heldout_coverage, heldout_trait_count = heldout_metrics(
        taxa=taxa,
        heldout_fields=heldout_fields,
        explanations=explanations,
        borrow_cost=borrow_cost,
        innovation_cost=innovation_cost,
    )
    return HoldoutFoldResult(
        fold_index=fold_index,
        heldout_trait_count=heldout_trait_count,
        heldout_cost_mean=heldout_cost_mean,
        heldout_coverage=heldout_coverage,
    )


def run_stability_replicate(
    input_path: Path,
    donor_cost: float,
    borrow_cost: float,
    innovation_cost: float,
    replicate_index: int,
    subset_fields: tuple[str, ...],
    full_primary_map: dict[str, str],
) -> StabilityReplicateResult:
    taxa, _ = load_taxa(input_path)
    explanations, _ = solve_exact(
        taxa=taxa,
        trait_fields=list(subset_fields),
        donor_cost=donor_cost,
        borrow_cost=borrow_cost,
        innovation_cost=innovation_cost,
    )
    primary_map = {
        taxon_name: explanation.primary_donor or ""
        for taxon_name, explanation in explanations.items()
    }
    return StabilityReplicateResult(
        replicate_index=replicate_index,
        agreement_with_full=primary_map_stability(full_primary_map, primary_map),
        primary_map=primary_map,
    )


def build_consensus_support(
    taxa: list[Taxon],
    full_explanations: dict[str, TaxonExplanation],
    stability_results: list[StabilityReplicateResult],
) -> list[ConsensusSupport]:
    replicate_count = len(stability_results)
    taxon_index = {taxon.taxon: taxon for taxon in taxa}
    support_rows: list[ConsensusSupport] = []

    for taxon in taxa:
        full_explanation = full_explanations[taxon.taxon]
        full_primary = full_explanation.primary_donor or ""
        vote_counts = Counter(
            result.primary_map.get(taxon.taxon, "")
            for result in stability_results
            if result.primary_map.get(taxon.taxon, "")
        )
        nonempty_vote_count = sum(vote_counts.values())
        if vote_counts:
            consensus_primary, vote_count = min(
                vote_counts.items(),
                key=lambda item: (
                    -item[1],
                    -len(full_explanation.donor_traits.get(item[0], [])),
                    (taxon.resolved_date - taxon_index[item[0]].resolved_date).days,
                    item[0],
                ),
            )
        else:
            consensus_primary = ""
            vote_count = 0

        vote_breakdown = ";".join(
            f"{donor}:{count}"
            for donor, count in sorted(vote_counts.items(), key=lambda item: (-item[1], item[0]))
        )
        support_rows.append(
            ConsensusSupport(
                taxon=taxon.taxon,
                consensus_primary=consensus_primary,
                vote_count=vote_count,
                support=(vote_count / replicate_count) if replicate_count else 0.0,
                nonempty_support=(vote_count / nonempty_vote_count) if nonempty_vote_count else 0.0,
                nonempty_vote_count=nonempty_vote_count,
                full_primary=full_primary,
                full_primary_support=(
                    vote_counts.get(full_primary, 0) / replicate_count if replicate_count and full_primary else 0.0
                ),
                matches_full_primary=bool(consensus_primary and consensus_primary == full_primary),
                in_full_solution=bool(consensus_primary and consensus_primary in full_explanation.donor_traits),
                vote_breakdown=vote_breakdown,
            )
        )

    return support_rows


def write_consensus_support_csv(
    taxa: list[Taxon],
    support_rows: list[ConsensusSupport],
    full_explanations: dict[str, TaxonExplanation],
    output_path: Path,
) -> None:
    fieldnames = [
        "taxon",
        "resolved_date",
        "full_primary_donor",
        "consensus_primary_donor",
        "consensus_vote_count",
        "consensus_primary_support",
        "consensus_primary_nonempty_support",
        "consensus_nonempty_vote_count",
        "full_primary_support",
        "consensus_matches_full_primary",
        "consensus_primary_in_full_solution",
        "full_donor_count",
        "full_borrowed_trait_count",
        "full_innovation_count",
        "vote_breakdown",
    ]
    support_index = {row.taxon: row for row in support_rows}
    taxon_index = {taxon.taxon: taxon for taxon in taxa}

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for taxon in taxa:
            support = support_index[taxon.taxon]
            full_explanation = full_explanations[taxon.taxon]
            writer.writerow(
                {
                    "taxon": taxon.taxon,
                    "resolved_date": taxon_index[taxon.taxon].resolved_date.isoformat(),
                    "full_primary_donor": support.full_primary,
                    "consensus_primary_donor": support.consensus_primary,
                    "consensus_vote_count": support.vote_count,
                    "consensus_primary_support": f"{support.support:.4f}",
                    "consensus_primary_nonempty_support": f"{support.nonempty_support:.4f}",
                    "consensus_nonempty_vote_count": support.nonempty_vote_count,
                    "full_primary_support": f"{support.full_primary_support:.4f}",
                    "consensus_matches_full_primary": int(support.matches_full_primary),
                    "consensus_primary_in_full_solution": int(support.in_full_solution),
                    "full_donor_count": full_explanation.donor_count,
                    "full_borrowed_trait_count": full_explanation.borrowed_trait_count,
                    "full_innovation_count": full_explanation.innovation_count,
                    "vote_breakdown": support.vote_breakdown,
                }
            )


def write_consensus_explanations(
    taxa: list[Taxon],
    base_rows: list[dict[str, str]],
    base_fieldnames: list[str],
    support_rows: list[ConsensusSupport],
    full_explanations: dict[str, TaxonExplanation],
    output_path: Path,
) -> None:
    support_index = {row.taxon: row for row in support_rows}
    taxon_index = {taxon.taxon: taxon for taxon in taxa}
    fieldnames = list(base_fieldnames)
    extra_fields = [
        "full_primary_donor",
        "full_primary_donor_date",
        "full_primary_donor_gap_days",
        "full_primary_donor_borrowed_trait_count",
        "primary_donor_support",
        "primary_donor_nonempty_support",
        "primary_donor_vote_count",
        "primary_donor_nonempty_vote_count",
        "primary_donor_in_full_solution",
        "primary_donor_matches_full_primary",
        "full_primary_donor_support",
        "primary_donor_vote_breakdown",
    ]
    for field in extra_fields:
        if field not in fieldnames:
            fieldnames.append(field)

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in base_rows:
            updated = dict(row)
            taxon_name = updated["taxon"]
            support = support_index[taxon_name]
            taxon = taxon_index[taxon_name]
            full_explanation = full_explanations[taxon_name]

            updated["full_primary_donor"] = row.get("primary_donor", "")
            updated["full_primary_donor_date"] = row.get("primary_donor_date", "")
            updated["full_primary_donor_gap_days"] = row.get("primary_donor_gap_days", "")
            updated["full_primary_donor_borrowed_trait_count"] = row.get(
                "primary_donor_borrowed_trait_count",
                "",
            )
            updated["primary_donor"] = support.consensus_primary
            if support.consensus_primary:
                donor_taxon = taxon_index[support.consensus_primary]
                updated["primary_donor_date"] = donor_taxon.resolved_date.isoformat()
                updated["primary_donor_gap_days"] = (
                    taxon.resolved_date - donor_taxon.resolved_date
                ).days
                if support.consensus_primary in full_explanation.donor_traits:
                    updated["primary_donor_borrowed_trait_count"] = len(
                        full_explanation.donor_traits[support.consensus_primary]
                    )
                else:
                    updated["primary_donor_borrowed_trait_count"] = ""
            else:
                updated["primary_donor_date"] = ""
                updated["primary_donor_gap_days"] = ""
                updated["primary_donor_borrowed_trait_count"] = ""

            updated["primary_donor_support"] = f"{support.support:.4f}" if support.consensus_primary else ""
            updated["primary_donor_nonempty_support"] = (
                f"{support.nonempty_support:.4f}" if support.consensus_primary else ""
            )
            updated["primary_donor_vote_count"] = support.vote_count if support.consensus_primary else ""
            updated["primary_donor_nonempty_vote_count"] = (
                support.nonempty_vote_count if support.consensus_primary else ""
            )
            updated["primary_donor_in_full_solution"] = int(support.in_full_solution)
            updated["primary_donor_matches_full_primary"] = int(support.matches_full_primary)
            updated["full_primary_donor_support"] = (
                f"{support.full_primary_support:.4f}" if support.full_primary else ""
            )
            updated["primary_donor_vote_breakdown"] = support.vote_breakdown
            writer.writerow(updated)


def write_consensus_edges(
    base_rows: list[dict[str, str]],
    base_fieldnames: list[str],
    support_rows: list[ConsensusSupport],
    output_path: Path,
) -> None:
    support_index = {row.taxon: row for row in support_rows}
    fieldnames = list(base_fieldnames)
    extra_fields = [
        "consensus_primary_donor",
        "consensus_primary_support",
    ]
    for field in extra_fields:
        if field not in fieldnames:
            fieldnames.append(field)

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in base_rows:
            updated = dict(row)
            support = support_index[updated["recipient"]]
            is_primary = bool(support.consensus_primary and updated["donor"] == support.consensus_primary)
            updated["is_primary_donor"] = int(is_primary)
            updated["consensus_primary_donor"] = support.consensus_primary
            updated["consensus_primary_support"] = f"{support.support:.4f}" if is_primary else ""
            writer.writerow(updated)


def write_report(
    taxa: list[Taxon],
    donor_cost: float,
    borrow_cost: float,
    innovation_cost: float,
    full_score: float,
    donor_edges: int,
    mean_donors: float,
    borrowed_traits: int,
    innovations: int,
    holdout_results: list[HoldoutFoldResult],
    stability_results: list[StabilityReplicateResult],
    support_rows: list[ConsensusSupport],
    output_path: Path,
    holdout_folds: int,
    stability_reps: int,
    stability_fraction: float,
) -> None:
    heldout_trait_total = sum(result.heldout_trait_count for result in holdout_results)
    heldout_cost_mean = (
        sum(result.heldout_cost_mean * result.heldout_trait_count for result in holdout_results)
        / heldout_trait_total
        if heldout_trait_total
        else float("inf")
    )
    heldout_coverage = (
        sum(result.heldout_coverage * result.heldout_trait_count for result in holdout_results)
        / heldout_trait_total
        if heldout_trait_total
        else 0.0
    )
    fold_costs = [result.heldout_cost_mean for result in holdout_results]
    fold_coverages = [result.heldout_coverage for result in holdout_results]
    stability_scores = [result.agreement_with_full for result in stability_results]

    consensus_nonempty = [row for row in support_rows if row.consensus_primary]
    majority_supported = [row for row in consensus_nonempty if row.support >= 0.5]
    strong_supported = [row for row in consensus_nonempty if row.support >= 0.75]
    differs_from_full = [
        row
        for row in consensus_nonempty
        if row.full_primary and row.consensus_primary != row.full_primary
    ]
    outside_full_solution = [row for row in consensus_nonempty if not row.in_full_solution]

    strongest = sorted(
        consensus_nonempty,
        key=lambda row: (-row.support, row.taxon),
    )[:12]
    ambiguous = sorted(
        consensus_nonempty,
        key=lambda row: (row.support, row.taxon),
    )[:12]

    lines = [
        "# Canonical Donor Fixed-Cost Validation",
        "",
        f"- Input matrix: {INPUT_CSV}",
        f"- Costs: donor={donor_cost:.2f}, borrow={borrow_cost:.2f}, innovation={innovation_cost:.2f}",
        f"- Full-data exact score: {full_score:.2f}",
        f"- Full-data donor edges: {donor_edges}",
        f"- Full-data mean donors per taxon: {mean_donors:.2f}",
        f"- Full-data borrowed traits: {borrowed_traits}",
        f"- Full-data innovations: {innovations}",
        f"- Held-out evaluation: {holdout_folds}-fold trait-column CV",
        f"- Held-out cost per applicable trait: {heldout_cost_mean:.4f}",
        f"- Held-out donor coverage: {heldout_coverage:.2%}",
        f"- Held-out fold spread: min {min(fold_costs):.4f}, median {statistics.median(fold_costs):.4f}, max {max(fold_costs):.4f}",
        f"- Stability evaluation: {stability_reps} feature-subsample refits at {stability_fraction:.0%} of trait columns",
        f"- Mean primary-donor agreement with full fit: {statistics.mean(stability_scores):.2%}",
        f"- Stability spread: min {min(stability_scores):.2%}, median {statistics.median(stability_scores):.2%}, max {max(stability_scores):.2%}",
        f"- Consensus parents chosen: {len(consensus_nonempty)} / {len(taxa)} taxa",
        f"- Consensus-parent support: mean {statistics.mean(row.support for row in consensus_nonempty):.2%}, median {statistics.median(row.support for row in consensus_nonempty):.2%}",
        f"- Majority-backed consensus edges (>=50% of replicates): {len(majority_supported)}",
        f"- Strong consensus edges (>=75% of replicates): {len(strong_supported)}",
        f"- Consensus differs from full primary donor: {len(differs_from_full)} taxa",
        f"- Consensus parent absent from full donor solution: {len(outside_full_solution)} taxa",
        "",
        "## Holdout Folds",
        "",
    ]

    for result in sorted(holdout_results, key=lambda item: item.fold_index):
        lines.append(
            f"- Fold {result.fold_index + 1}: held-out cost {result.heldout_cost_mean:.4f}, coverage {result.heldout_coverage:.2%}, traits {result.heldout_trait_count}"
        )

    lines.extend(["", "## Strongest Consensus Parents", ""])
    for row in strongest:
        lines.append(
            f"- {row.taxon}: {row.consensus_primary} with support {row.support:.2%} (full primary {row.full_primary or 'none'})"
        )

    lines.extend(["", "## Most Ambiguous Consensus Parents", ""])
    for row in ambiguous:
        lines.append(
            f"- {row.taxon}: {row.consensus_primary or 'none'} with support {row.support:.2%}, votes {row.vote_breakdown or 'none'}"
        )

    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate one fixed donor-cost regime and build a consensus-parent scaffold."
    )
    parser.add_argument("--input", type=Path, default=INPUT_CSV)
    parser.add_argument("--base-explanations", type=Path, default=BASE_EXPLANATIONS_CSV)
    parser.add_argument("--base-edges", type=Path, default=BASE_EDGES_CSV)
    parser.add_argument("--donor-cost", type=float, default=DEFAULT_DONOR_COST)
    parser.add_argument("--borrow-cost", type=float, default=DEFAULT_BORROW_COST)
    parser.add_argument("--innovation-cost", type=float, default=DEFAULT_INNOVATION_COST)
    parser.add_argument("--holdout-folds", type=int, default=DEFAULT_HOLDOUT_FOLDS)
    parser.add_argument("--stability-reps", type=int, default=DEFAULT_STABILITY_REPS)
    parser.add_argument("--stability-fraction", type=float, default=DEFAULT_STABILITY_FRACTION)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, (os.cpu_count() or 1) - 1)),
    )
    args = parser.parse_args()

    taxa, trait_fields = load_taxa(args.input)
    base_explanation_rows, base_explanation_fields = load_csv_rows(args.base_explanations)
    base_edge_rows, base_edge_fields = load_csv_rows(args.base_edges)

    print(
        f"Running full exact fit at donor={args.donor_cost:.2f}, borrow={args.borrow_cost:.2f}, innovation={args.innovation_cost:.2f}."
    )
    full_explanations, full_score = solve_exact(
        taxa=taxa,
        trait_fields=trait_fields,
        donor_cost=args.donor_cost,
        borrow_cost=args.borrow_cost,
        innovation_cost=args.innovation_cost,
    )
    donor_edges, mean_donors, borrowed_traits, innovations = summarize_solution(full_explanations, len(taxa))
    full_primary_map = {
        taxon_name: explanation.primary_donor or ""
        for taxon_name, explanation in full_explanations.items()
    }

    holdout_folds = split_trait_folds(trait_fields, args.holdout_folds, args.seed)
    stability_sets = stability_trait_subsets(
        trait_fields,
        reps=args.stability_reps,
        fraction=args.stability_fraction,
        seed=args.seed + 101,
    )

    holdout_results: list[HoldoutFoldResult] = []
    stability_results: list[StabilityReplicateResult] = []
    total_tasks = len(holdout_folds) + len(stability_sets)
    completed_tasks = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_map = {}
        for fold_index, heldout_fields in enumerate(holdout_folds):
            future = executor.submit(
                run_holdout_fold,
                args.input,
                args.donor_cost,
                args.borrow_cost,
                args.innovation_cost,
                fold_index,
                heldout_fields,
            )
            future_map[future] = ("holdout", fold_index)
        for replicate_index, subset_fields in enumerate(stability_sets):
            future = executor.submit(
                run_stability_replicate,
                args.input,
                args.donor_cost,
                args.borrow_cost,
                args.innovation_cost,
                replicate_index,
                subset_fields,
                full_primary_map,
            )
            future_map[future] = ("stability", replicate_index)

        for future in as_completed(future_map):
            kind, index = future_map[future]
            completed_tasks += 1
            if kind == "holdout":
                result = future.result()
                holdout_results.append(result)
                print(
                    f"[{completed_tasks}/{total_tasks}] holdout fold {index + 1}: cost={result.heldout_cost_mean:.4f}, coverage={result.heldout_coverage:.2%}"
                )
            else:
                result = future.result()
                stability_results.append(result)
                print(
                    f"[{completed_tasks}/{total_tasks}] stability replicate {index + 1}: agreement={result.agreement_with_full:.2%}"
                )

    support_rows = build_consensus_support(
        taxa=taxa,
        full_explanations=full_explanations,
        stability_results=stability_results,
    )
    write_consensus_support_csv(
        taxa=taxa,
        support_rows=support_rows,
        full_explanations=full_explanations,
        output_path=CONSENSUS_SUPPORT_CSV,
    )
    write_consensus_explanations(
        taxa=taxa,
        base_rows=base_explanation_rows,
        base_fieldnames=base_explanation_fields,
        support_rows=support_rows,
        full_explanations=full_explanations,
        output_path=CONSENSUS_EXPLANATIONS_CSV,
    )
    write_consensus_edges(
        base_rows=base_edge_rows,
        base_fieldnames=base_edge_fields,
        support_rows=support_rows,
        output_path=CONSENSUS_EDGES_CSV,
    )
    write_report(
        taxa=taxa,
        donor_cost=args.donor_cost,
        borrow_cost=args.borrow_cost,
        innovation_cost=args.innovation_cost,
        full_score=full_score,
        donor_edges=donor_edges,
        mean_donors=mean_donors,
        borrowed_traits=borrowed_traits,
        innovations=innovations,
        holdout_results=holdout_results,
        stability_results=stability_results,
        support_rows=support_rows,
        output_path=VALIDATION_REPORT_MD,
        holdout_folds=args.holdout_folds,
        stability_reps=args.stability_reps,
        stability_fraction=args.stability_fraction,
    )

    print(f"Wrote validation report: {VALIDATION_REPORT_MD}")
    print(f"Wrote consensus support CSV: {CONSENSUS_SUPPORT_CSV}")
    print(f"Wrote consensus explanations: {CONSENSUS_EXPLANATIONS_CSV}")
    print(f"Wrote consensus edges: {CONSENSUS_EDGES_CSV}")


if __name__ == "__main__":
    main()
