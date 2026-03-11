# Canonical Donor Graph

- Input matrix: /Users/natebreslow/Documents/llmPhylogeny/canonical_features.csv
- Taxa: 94
- Trait fields scored: 63
- Optimizer: exact per-taxon MILP via SciPy HiGHS; global optimum is the sum of those exact local optima because taxa decouple under the current dated donor-only objective.
- Objective: minimize donor activation cost + per-trait borrow cost + innovation cost.
- Constraint: donors must have strictly earlier release dates than recipients.
- Model: no primary parent; every applicable trait is either borrowed from an older donor or treated as an innovation.
- Display parent rule: donor with the most borrowed traits; ties break by shortest temporal gap, then taxon name.
- Secondary donor activation cost: 2.00 per donor taxon
- Borrowed trait cost: 0.10 per trait
- Innovation cost: 1.00 per trait
- Total score: 1099.70
- Greedy baseline total score: 1109.20
- Improvement vs greedy: 9.50 (0.86%)
- Taxa improved vs greedy: 10
- Donor edges used: 154
- Borrowed traits explained: 4767
- Innovations required: 315
- Taxa with no donor selected: 1
- Release-date precision mix: exact_day=93, month_midpoint=1

## Applicability Filters

- MoE downstream fields are ignored when `has_moe=0`.
- Shared-expert merge details are ignored when `has_shared_expert=0`.
- Rope-detail fields are ignored when the positional encoding is not RoPE-based.
- `qk_norm_detail` is ignored when `qk_norm=none`.

## Top Donors

- llama: donor to 14 taxa, 391 borrowed traits
- phi: donor to 7 taxa, 144 borrowed traits
- qwen: donor to 7 taxa, 171 borrowed traits
- mixtral: donor to 7 taxa, 242 borrowed traits
- olmoe: donor to 6 taxa, 213 borrowed traits
- deepseek_v3: donor to 6 taxa, 219 borrowed traits
- gpt_oss: donor to 5 taxa, 122 borrowed traits
- gpt2: donor to 4 taxa, 176 borrowed traits
- glm: donor to 4 taxa, 145 borrowed traits
- gpt_neox: donor to 4 taxa, 162 borrowed traits
- minicpm: donor to 4 taxa, 151 borrowed traits
- phixtral: donor to 4 taxa, 75 borrowed traits

## Borrow-Heavy Taxa

- dots1: donors deepseek(35), minimax(28), innovations none, score 10.30
- exaone_moe: donors afmoe(36), ministral3(27), innovations none, score 10.30
- Klear: donors dots1(33), nemotron_h(28), innovations shared_branch_merge_mode, coefficient_mix_shared_routed, score 12.10
- mimo_v2_flash: donors glm4_moe(37), gpt_oss(24), innovations rope_parameterization, score 11.10
- ernie4_5_moe: donors olmoe(41), nemotron_h(20), innovations none, score 10.10
- glm4_moe_lite: donors deepseek_v3(61), innovations tie_embeddings_configurable, score 9.10
- qwen3_next: donors bailing_moe_linear(49), nemotron_h(6), qwen2_moe(5), innovations state_update_rule, attention_family, gated_delta_net, score 15.00
- hunyuan: donors bailing_moe(36), dbrx(24), innovations attention_projection, cross_layer_kv_sharing, positional_encoding, score 13.00
- qwen3_5: donors qwen3_next(38), gpt_oss(22), innovations modality, rope_partition_detail, rope_parameterization, score 13.00
- dbrx: donors minicpm(32), phixtral(28), innovations none, score 10.00
- qwen3_moe: donors llama4(31), mixtral(29), innovations none, score 10.00
- glm4_moe: donors dots1(60), innovations positional_encoding, output_head_connection, score 10.00

## Largest Improvements Vs Greedy

- qwen: exact 12.60 vs greedy 14.20, improvement 1.60
- granitemoehybrid: exact 12.90 vs greedy 14.00, improvement 1.10
- qwen2: exact 9.00 vs greedy 9.90, improvement 0.90
- qwen3_moe: exact 10.00 vs greedy 10.90, improvement 0.90
- apertus: exact 9.90 vs greedy 10.80, improvement 0.90
- nemotron: exact 10.80 vs greedy 11.70, improvement 0.90
- cohere: exact 11.80 vs greedy 12.70, improvement 0.90
- phimoe: exact 11.80 vs greedy 12.70, improvement 0.90
- internlm2: exact 10.80 vs greedy 11.50, improvement 0.70
- qwen3_next: exact 15.00 vs greedy 15.70, improvement 0.70

## Output Files

- canonical_donor_taxon_explanations.csv
- canonical_donor_edges.csv
