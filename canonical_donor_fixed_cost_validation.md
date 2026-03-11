# Canonical Donor Fixed-Cost Validation

- Input matrix: /Users/natebreslow/Documents/llmPhylogeny/canonical_features.csv
- Costs: donor=2.00, borrow=0.10, innovation=1.00
- Full-data exact score: 1100.10
- Full-data donor edges: 156
- Full-data mean donors per taxon: 1.66
- Full-data borrowed traits: 4771
- Full-data innovations: 311
- Held-out evaluation: 5-fold trait-column CV
- Held-out cost per applicable trait: 0.2052
- Held-out donor coverage: 88.31%
- Held-out fold spread: min 0.1513, median 0.2151, max 0.2294
- Stability evaluation: 20 feature-subsample refits at 80% of trait columns
- Mean primary-donor agreement with full fit: 54.95%
- Stability spread: min 44.09%, median 54.84%, max 63.44%
- Consensus parents chosen: 93 / 94 taxa
- Consensus-parent support: mean 64.09%, median 65.00%
- Majority-backed consensus edges (>=50% of replicates): 64
- Strong consensus edges (>=75% of replicates): 32
- Consensus differs from full primary donor: 30 taxa
- Consensus parent absent from full donor solution: 12 taxa

## Holdout Folds

- Fold 1: held-out cost 0.1513, coverage 94.30%, traits 1018
- Fold 2: held-out cost 0.2123, coverage 87.52%, traits 1098
- Fold 3: held-out cost 0.2294, coverage 85.63%, traits 974
- Fold 4: held-out cost 0.2192, coverage 86.76%, traits 944
- Fold 5: held-out cost 0.2151, coverage 87.21%, traits 1048

## Strongest Consensus Parents

- bailing_moe_linear: bailing_moe with support 100.00% (full primary bailing_moe)
- deepseek: mixtral with support 100.00% (full primary mixtral)
- deepseek_v2: deepseek with support 100.00% (full primary deepseek)
- deepseek_v3: deepseek_v2 with support 100.00% (full primary deepseek_v2)
- deepseek_v32: deepseek_v3 with support 100.00% (full primary deepseek_v3)
- glm4_moe: dots1 with support 100.00% (full primary dots1)
- glm4_moe_lite: deepseek_v3 with support 100.00% (full primary deepseek_v3)
- gpt_bigcode: gpt2 with support 100.00% (full primary gpt2)
- gpt_neox: gpt2 with support 100.00% (full primary gpt2)
- mamba2: mamba with support 100.00% (full primary mamba)
- minicpm3: deepseek_v2 with support 100.00% (full primary deepseek_v2)
- ministral3: llama with support 100.00% (full primary llama)

## Most Ambiguous Consensus Parents

- glm: llama with support 20.00%, votes llama:4;bitnet:3;mamba:2;openelm:2;phi3:2;phi3small:2;gemma:1;minicpm:1;olmo:1;phi:1;starcoder2:1
- lille-130m: afm7 with support 20.00%, votes afm7:4;nemotron-nas:4;olmo:3;glm:2;helium:2;exaone:1;openelm:1;phi3:1;phimoe:1;plamo2:1
- gemma: llama with support 25.00%, votes gpt_bigcode:5;llama:5;stablelm:4;gpt2:2;deepseek:1;gpt_neox:1;minicpm:1;plamo:1
- hunyuan_v1_dense: qwen3 with support 25.00%, votes hunyuan:5;internlm3:5;qwen3:5;dots1:4;olmo2:1
- minicpm: llama with support 25.00%, votes deepseek:5;llama:5;mixtral:4;stablelm:3;internlm2:2;qwen:1
- minimax: dots1 with support 25.00%, votes dots1:5;phixtral:5;bailing_moe:3;Klear:2;mixtral:2;bailing_moe_linear:1;glm4_moe:1;granitemoe:1
- seed_oss: qwen2 with support 25.00%, votes qwen2:5;granite:4;minicpm:3;bitnet:1;ernie4_5:1;helium:1;internlm3:1;llama:1;olmo2:1;olmoe:1;qwen3:1
- apertus: glm4_moe with support 30.00%, votes glm4_moe:6;smollm3:3;olmoe:2;qwen3:2;starcoder2:2;llama4:1;nemotron:1;olmo:1;olmo2:1;qwen3_moe:1
- exaone: llama with support 30.00%, votes bitnet:6;llama:6;internlm2:2;qwen:2;glm:1;minicpm:1;mixtral:1;qwen2:1
- internlm2: nemotron with support 30.00%, votes nemotron:6;plamo:5;mixtral:4;llama:2;mamba:2;qwen:1
- telechat3: helium with support 30.00%, votes helium:6;internlm3:5;hunyuan_v1_dense:2;minicpm:2;dots1:1;ernie4_5:1;glm:1;mixtral:1;seed_oss:1
- cohere: phi with support 35.00%, votes phi:7;llama:4;stablelm:3;gemma:2;gpt2:2;deepseek:1;plamo:1
