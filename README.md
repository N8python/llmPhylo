# llmPhylogeny

Minimal extracted repo for the open-source LLM faux-phylogeny workflow.

## Contents

- `models/`: canonical architecture metadata used to build the feature matrix
- `scripts/build_native_architecture_character_matrix.py`: expands `models/` into the raw architecture matrix
- `scripts/build_native_parsimony_artifacts.py`: normalizes the raw matrix into the 63-character parsimony matrix
- `scripts/build_canonical_features.py`: joins release dates onto the normalized matrix to produce `canonical_features.csv`
- `scripts/build_canonical_donor_graph.py`: exact donor-only solver using the tuned default costs
- `scripts/validate_canonical_donor_setting.py`: fixed-cost validation plus consensus-parent scaffold
- `scripts/render_canonical_donor_network.py`: static donor network renderer
- `scripts/render_canonical_donor_timeline_video.py`: timeline-style MP4 renderer

## Tuned Defaults

The donor graph defaults are set to the tuned regime used in the original work:

- `donor_cost = 2.0`
- `borrow_cost = 0.10`
- `innovation_cost = 1.0`

## End-to-End Build

From the repo root:

```bash
python3 scripts/build_native_architecture_character_matrix.py
python3 scripts/build_native_parsimony_artifacts.py --skip-search
python3 scripts/build_canonical_features.py
python3 scripts/build_canonical_donor_graph.py
python3 scripts/validate_canonical_donor_setting.py
python3 scripts/render_canonical_donor_network.py --explanations canonical_donor_consensus_explanations.csv --edges canonical_donor_consensus_edges.csv --output canonical_donor_consensus_network.png --title "Canonical Donor Consensus Network"
python3 scripts/render_canonical_donor_timeline_video.py --output canonical_donor_consensus_evolution.mp4
```

`ffmpeg` must be available on `PATH` for the MP4 render step.
