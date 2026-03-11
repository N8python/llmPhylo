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
- Total score: 1100.10
- Greedy baseline total score: 1108.90
- Improvement vs greedy: 8.80 (0.79%)
- Taxa improved vs greedy: 10
- Donor edges used: 156
- Borrowed traits explained: 4771
- Innovations required: 311
- Taxa with no donor selected: 1
- Release-date precision mix: exact_day=94

## Applicability Filters

- MoE downstream fields are ignored when `has_moe=0`.
- Shared-expert merge details are ignored when `has_shared_expert=0`.
- Rope-detail fields are ignored when the positional encoding is not RoPE-based.
- `qk_norm_detail` is ignored when `qk_norm=none`.

## Top Donors

- llama: donor to 15 taxa, 440 borrowed traits
- qwen: donor to 7 taxa, 172 borrowed traits
- mixtral: donor to 6 taxa, 210 borrowed traits
- deepseek: donor to 6 taxa, 165 borrowed traits
- olmoe: donor to 6 taxa, 194 borrowed traits
- deepseek_v3: donor to 6 taxa, 209 borrowed traits
- gpt2: donor to 5 taxa, 200 borrowed traits
- gpt_neox: donor to 5 taxa, 123 borrowed traits
- dots1: donor to 5 taxa, 188 borrowed traits
- qwen2_moe: donor to 4 taxa, 95 borrowed traits
- nemotron_h: donor to 4 taxa, 95 borrowed traits
- gpt_oss: donor to 4 taxa, 81 borrowed traits

## Borrow-Heavy Taxa

- dots1: donors bailing_moe(34), granitemoe(29), innovations none, score 10.30
- exaone_moe: donors afmoe(35), ministral3(28), innovations none, score 10.30
- Klear: donors nemotron_h(31), dots1(30), innovations shared_branch_merge_mode, coefficient_mix_shared_routed, score 12.10
- mimo_v2_flash: donors glm4_moe(31), gpt_oss(30), innovations rope_parameterization, score 11.10
- ernie4_5_moe: donors olmoe(32), nemotron_h(29), innovations none, score 10.10
- minimax: donors dots1(31), phixtral(30), innovations none, score 10.10
- glm4_moe_lite: donors deepseek_v3(61), innovations tie_embeddings_configurable, score 9.10
- qwen3_next: donors bailing_moe_linear(49), qwen2_moe(7), nemotron_h(4), innovations state_update_rule, attention_family, gated_delta_net, score 15.00
- hunyuan: donors dots1(33), qwen2_moe(27), innovations attention_projection, cross_layer_kv_sharing, positional_encoding, score 13.00
- qwen3_5: donors qwen3_next(36), gpt_oss(24), innovations modality, rope_partition_detail, rope_parameterization, score 13.00
- dbrx: donors phixtral(33), deepseek(27), innovations none, score 10.00
- qwen3_moe: donors granitemoe(31), llama4(29), innovations none, score 10.00

## Largest Improvements Vs Greedy

- granitemoehybrid: exact 12.90 vs greedy 14.00, improvement 1.10
- qwen2: exact 9.00 vs greedy 9.90, improvement 0.90
- llama4_text: exact 9.00 vs greedy 9.90, improvement 0.90
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
