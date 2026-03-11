# Canonical Donor Fixed-Cost Validation

- Input matrix: /Users/natebreslow/Documents/llmPhylogeny/canonical_features.csv
- Costs: donor=2.00, borrow=0.10, innovation=1.00
- Full-data exact score: 1099.70
- Full-data donor edges: 154
- Full-data mean donors per taxon: 1.64
- Full-data borrowed traits: 4767
- Full-data innovations: 315
- Held-out evaluation: 5-fold trait-column CV
- Held-out cost per applicable trait: 0.2031
- Held-out donor coverage: 88.55%
- Held-out fold spread: min 0.1530, median 0.2098, max 0.2358
- Stability evaluation: 20 feature-subsample refits at 80% of trait columns
- Mean primary-donor agreement with full fit: 55.65%
- Stability spread: min 45.16%, median 55.38%, max 64.52%
- Consensus parents chosen: 93 / 94 taxa
- Consensus-parent support: mean 65.65%, median 65.00%
- Majority-backed consensus edges (>=50% of replicates): 68
- Strong consensus edges (>=75% of replicates): 40
- Consensus differs from full primary donor: 28 taxa
- Consensus parent absent from full donor solution: 19 taxa

## Holdout Folds

- Fold 1: held-out cost 0.1530, coverage 94.11%, traits 1018
- Fold 2: held-out cost 0.2098, coverage 87.80%, traits 1098
- Fold 3: held-out cost 0.2358, coverage 84.91%, traits 974
- Fold 4: held-out cost 0.2077, coverage 88.03%, traits 944
- Fold 5: held-out cost 0.2099, coverage 87.79%, traits 1048

## Strongest Consensus Parents

- afm7: openelm with support 100.00% (full primary openelm)
- bailing_moe_linear: bailing_moe with support 100.00% (full primary bailing_moe)
- deepseek_v2: deepseek with support 100.00% (full primary deepseek)
- deepseek_v3: deepseek_v2 with support 100.00% (full primary deepseek_v2)
- glm: gpt2 with support 100.00% (full primary gpt2)
- glm4_moe_lite: deepseek_v3 with support 100.00% (full primary deepseek_v3)
- gpt_bigcode: gpt2 with support 100.00% (full primary gpt2)
- gpt_neox: gpt2 with support 100.00% (full primary gpt2)
- granitemoe: mixtral with support 100.00% (full primary mixtral)
- llama: glm with support 100.00% (full primary glm)
- mamba2: mamba with support 100.00% (full primary mamba)
- minicpm3: deepseek_v2 with support 100.00% (full primary deepseek_v2)

## Most Ambiguous Consensus Parents

- lille-130m: nemotron-nas with support 15.00%, votes nemotron-nas:3;afm7:2;glm:2;helium:2;olmo:2;qwen:2;exaone:1;falcon_h1:1;hunyuan:1;mamba:1;phimoe:1;plamo:1;plamo2:1
- minicpm: llama with support 20.00%, votes deepseek:4;internlm2:4;llama:4;mixtral:4;nemotron:3;glm:1
- baichuan_m1: internlm2 with support 25.00%, votes internlm2:5;minicpm:4;cohere2:3;helium:2;qwen2_moe:2;mamba:1;mixtral:1;rwkv7:1;starcoder2:1
- nemotron-nas: exaone with support 25.00%, votes exaone:5;mixtral:4;mamba2:3;bitnet:1;deepseek:1;granite:1;llama:1;olmoe:1;phi:1;phi3:1;qwen:1
- Klear: dots1 with support 30.00%, votes dots1:6;minimax:5;qwen3_moe:4;glm4_moe:2;llama4:2;nemotron_h:1
- cohere: phi with support 30.00%, votes phi:6;llama:4;stablelm:4;bitnet:2;gemma:1;gpt2:1;mixtral:1;plamo:1
- gemma3_text: olmo2 with support 30.00%, votes olmo2:6;cohere2:5;gemma2:5;bailing_moe:4
- hunyuan_v1_dense: internlm3 with support 30.00%, votes internlm3:6;qwen3:6;dots1:3;hunyuan:2;minimax:1;olmo2:1;qwen3_moe:1
- gemma: glm with support 35.00%, votes glm:7;stablelm:5;gpt2:2;gpt_bigcode:2;plamo:2;deepseek:1;llama:1
- gemma3n: gemma3_text with support 35.00%, votes gemma3_text:7;gemma2:5;llama:5;llama4_text:2;bitnet:1
- hunyuan: qwen2_moe with support 35.00%, votes qwen2_moe:7;dots1:5;mixtral:3;afm7:1;bailing_moe:1;bailing_moe_linear:1;granitemoe:1;openelm:1
- internlm2: plamo with support 35.00%, votes plamo:7;mixtral:5;nemotron:5;llama:2;mamba:1
