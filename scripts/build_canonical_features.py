#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PARSIMONY_MATRIX = ROOT / "native_architecture_parsimony_matrix.csv"
RELEASE_DATES = ROOT / "native_architecture_release_dates.csv"
OUTPUT_CSV = ROOT / "canonical_features.csv"


def load_release_dates(path: Path) -> dict[str, str]:
    with path.open() as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"No CSV header found in {path}")
        required = {"model", "date"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing columns in {path}: {', '.join(sorted(missing))}")

        release_dates: dict[str, str] = {}
        for row in reader:
            model = row["model"].strip()
            raw_date = row["date"].strip()
            if not model or not raw_date:
                continue
            existing = release_dates.get(model)
            if existing is not None and existing != raw_date:
                raise ValueError(
                    f"Conflicting release dates for {model}: {existing} vs {raw_date}"
                )
            release_dates[model] = raw_date
    return release_dates


def build_canonical_features(
    parsimony_path: Path,
    release_dates_path: Path,
    output_path: Path,
) -> tuple[int, int]:
    release_dates = load_release_dates(release_dates_path)

    with parsimony_path.open() as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"No CSV header found in {parsimony_path}")
        if "taxon" not in reader.fieldnames:
            raise ValueError(f"Expected a 'taxon' column in {parsimony_path}")

        trait_fields = [field for field in reader.fieldnames if field != "taxon"]
        fieldnames = ["taxon", "release_date", *trait_fields]
        rows_to_write: list[dict[str, str]] = []
        missing_dates: list[str] = []
        seen_taxa: set[str] = set()

        for row in reader:
            taxon = row["taxon"].strip()
            if not taxon:
                continue
            if taxon in seen_taxa:
                raise ValueError(f"Duplicate taxon in {parsimony_path}: {taxon}")
            seen_taxa.add(taxon)
            release_date = release_dates.get(taxon, "").strip()
            if not release_date:
                missing_dates.append(taxon)
                continue
            rows_to_write.append(
                {
                    "taxon": taxon,
                    "release_date": release_date,
                    **{field: row[field].strip() for field in trait_fields},
                }
            )

    if missing_dates:
        raise ValueError(
            "Missing release dates for taxa: " + ", ".join(sorted(missing_dates))
        )

    unused_release_dates = sorted(set(release_dates).difference({row["taxon"] for row in rows_to_write}))

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_to_write)

    return len(rows_to_write), len(unused_release_dates)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join release dates onto the normalized architecture matrix to build canonical_features.csv."
    )
    parser.add_argument("--parsimony-matrix", type=Path, default=PARSIMONY_MATRIX)
    parser.add_argument("--release-dates", type=Path, default=RELEASE_DATES)
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    args = parser.parse_args()

    row_count, unused_release_dates = build_canonical_features(
        parsimony_path=args.parsimony_matrix,
        release_dates_path=args.release_dates,
        output_path=args.output,
    )
    print(
        f"Wrote {row_count} taxa to {args.output} using release dates from {args.release_dates}."
    )
    if unused_release_dates:
        print(
            f"Warning: {unused_release_dates} release-date rows were unused because those models are absent from the parsimony matrix."
        )


if __name__ == "__main__":
    main()
