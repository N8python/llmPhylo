#!/usr/bin/env python3
from __future__ import annotations

import math
import random

from build_canonical_donor_graph import Taxon, TaxonExplanation, applicable_traits


def split_trait_folds(
    trait_fields: list[str],
    folds: int,
    seed: int,
) -> list[tuple[str, ...]]:
    if folds < 2:
        raise ValueError("Need at least 2 folds for held-out evaluation.")
    shuffled = list(trait_fields)
    random.Random(seed).shuffle(shuffled)
    return [tuple(shuffled[index::folds]) for index in range(folds)]


def stability_trait_subsets(
    trait_fields: list[str],
    reps: int,
    fraction: float,
    seed: int,
) -> list[tuple[str, ...]]:
    if reps < 1:
        raise ValueError("Need at least one stability replicate.")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("Stability fraction must be in (0, 1].")
    subset_size = max(1, math.ceil(len(trait_fields) * fraction))
    rng = random.Random(seed)
    return [
        tuple(sorted(rng.sample(trait_fields, subset_size), key=trait_fields.index))
        for _ in range(reps)
    ]


def heldout_metrics(
    taxa: list[Taxon],
    heldout_fields: tuple[str, ...],
    explanations: dict[str, TaxonExplanation],
    borrow_cost: float,
    innovation_cost: float,
) -> tuple[float, float, int]:
    taxon_index = {taxon.taxon: taxon for taxon in taxa}
    total_cost = 0.0
    covered = 0
    total_traits = 0

    for taxon in taxa:
        heldout_applicable = applicable_traits(taxon, list(heldout_fields))
        if not heldout_applicable:
            continue
        donor_names = tuple(explanations[taxon.taxon].donor_traits)
        for field in heldout_applicable:
            total_traits += 1
            if any(
                taxon_index[donor_name].traits.get(field, "?") == taxon.traits.get(field, "?")
                for donor_name in donor_names
            ):
                total_cost += borrow_cost
                covered += 1
            else:
                total_cost += innovation_cost

    if total_traits == 0:
        return float("inf"), 0.0, 0
    return total_cost / total_traits, covered / total_traits, total_traits


def summarize_solution(
    explanations: dict[str, TaxonExplanation],
    taxa_count: int,
) -> tuple[int, float, int, int]:
    donor_edges = sum(explanation.donor_count for explanation in explanations.values())
    borrowed_traits = sum(explanation.borrowed_trait_count for explanation in explanations.values())
    innovations = sum(explanation.innovation_count for explanation in explanations.values())
    mean_donors = donor_edges / taxa_count if taxa_count else 0.0
    return donor_edges, mean_donors, borrowed_traits, innovations
