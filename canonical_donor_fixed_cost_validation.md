# Canonical Donor Fixed-Cost Validation

- Input matrix: /Users/natebreslow/Documents/llmPhylogeny/canonical_features.csv
- Costs: donor=2.00, borrow=0.10, innovation=1.00
- Full-data exact score: 1100.10
- Full-data donor edges: 156
- Full-data mean donors per taxon: 1.66
- Full-data borrowed traits: 4771
- Full-data innovations: 311
- Held-out evaluation: 5-fold trait-column CV
- Held-out cost per applicable trait: 0.2055
- Held-out donor coverage: 88.27%
- Held-out fold spread: min 0.1530, median 0.2177, max 0.2275
- Stability evaluation: 20 feature-subsample refits at 80% of trait columns
- Mean primary-donor agreement with full fit: 56.51%
- Stability spread: min 47.31%, median 55.91%, max 67.74%
- Consensus parents chosen: 93 / 94 taxa
- Consensus-parent support: mean 64.35%, median 65.00%
- Majority-backed consensus edges (>=50% of replicates): 64
- Strong consensus edges (>=75% of replicates): 32
- Consensus differs from full primary donor: 26 taxa
- Consensus parent absent from full donor solution: 12 taxa

## Holdout Folds

- Fold 1: held-out cost 0.1530, coverage 94.11%, traits 1018
- Fold 2: held-out cost 0.2115, coverage 87.61%, traits 1098
- Fold 3: held-out cost 0.2275, coverage 85.83%, traits 974
- Fold 4: held-out cost 0.2192, coverage 86.76%, traits 944
- Fold 5: held-out cost 0.2177, coverage 86.93%, traits 1048

## Strongest Consensus Parents

- bailing_moe_linear: bailing_moe with support 100.00% (full primary bailing_moe)
- deepseek: mixtral with support 100.00% (full primary mixtral)
- deepseek_v2: deepseek with support 100.00% (full primary deepseek)
- deepseek_v3: deepseek_v2 with support 100.00% (full primary deepseek_v2)
- deepseek_v32: deepseek_v3 with support 100.00% (full primary deepseek_v3)
- glm4_moe_lite: deepseek_v3 with support 100.00% (full primary deepseek_v3)
- gpt_bigcode: gpt2 with support 100.00% (full primary gpt2)
- gpt_neox: gpt2 with support 100.00% (full primary gpt2)
- mamba2: mamba with support 100.00% (full primary mamba)
- minicpm3: deepseek_v2 with support 100.00% (full primary deepseek_v2)
- ministral3: llama with support 100.00% (full primary llama)
- olmo2: olmoe with support 100.00% (full primary olmoe)

## Most Ambiguous Consensus Parents

- glm: llama with support 20.00%, votes llama:4;bitnet:3;mamba:2;openelm:2;phi3:2;phi3small:2;gemma:1;minicpm:1;olmo:1;phi:1;starcoder2:1
- lille-130m: nemotron-nas with support 20.00%, votes nemotron-nas:4;afm7:3;glm:3;olmo:3;exaone:1;falcon_h1:1;helium:1;llama4_text:1;phimoe:1;plamo:1;plamo2:1
- gemma: llama with support 25.00%, votes gpt_bigcode:5;llama:5;stablelm:4;gpt2:2;deepseek:1;gpt_neox:1;minicpm:1;plamo:1
- minicpm: llama with support 25.00%, votes deepseek:5;llama:5;mixtral:4;stablelm:3;internlm2:2;qwen:1
- seed_oss: minicpm with support 25.00%, votes minicpm:5;qwen2:4;granite:3;bitnet:1;deepseek:1;granitemoe:1;helium:1;internlm3:1;llama:1;olmo2:1;qwen3:1
- apertus: glm4_moe with support 30.00%, votes glm4_moe:6;openelm:4;smollm3:3;bailing_moe:1;dots1:1;nemotron:1;olmo2:1;olmoe:1;qwen3:1;qwen3_moe:1
- exaone: llama with support 30.00%, votes bitnet:6;llama:6;internlm2:2;qwen:2;glm:1;minicpm:1;mixtral:1;qwen2:1
- hunyuan: dots1 with support 30.00%, votes dots1:6;qwen2_moe:5;bailing_moe:3;llama4:1;mixtral:1;olmoe:1;openelm:1;phixtral:1;qwen3_moe:1
- internlm2: nemotron with support 30.00%, votes nemotron:6;plamo:5;mixtral:4;llama:2;mamba:2;qwen:1
- telechat3: helium with support 30.00%, votes helium:6;internlm3:5;glm:2;hunyuan_v1_dense:2;mixtral:2;ernie4_5:1;minicpm:1;seed_oss:1
- cohere: phi with support 35.00%, votes phi:7;llama:4;stablelm:3;gemma:2;gpt2:2;deepseek:1;plamo:1
- minimax: dots1 with support 35.00%, votes dots1:7;phixtral:3;Klear:2;dbrx:2;glm4_moe:2;olmoe:2;bailing_moe:1;mixtral:1
