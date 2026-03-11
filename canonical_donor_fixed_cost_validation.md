# Canonical Donor Fixed-Cost Validation

- Input matrix: /Users/natebreslow/Documents/llmPhylogeny/canonical_features.csv
- Costs: donor=2.00, borrow=0.10, innovation=1.00
- Full-data exact score: 1100.10
- Full-data donor edges: 156
- Full-data mean donors per taxon: 1.66
- Full-data borrowed traits: 4771
- Full-data innovations: 311
- Held-out evaluation: 5-fold trait-column CV
- Held-out cost per applicable trait: 0.2045
- Held-out donor coverage: 88.39%
- Held-out fold spread: min 0.1477, median 0.2142, max 0.2368
- Stability evaluation: 20 feature-subsample refits at 80% of trait columns
- Mean primary-donor agreement with full fit: 58.60%
- Stability spread: min 46.24%, median 59.14%, max 70.97%
- Consensus parents chosen: 93 / 94 taxa
- Consensus-parent support: mean 64.62%, median 65.00%
- Majority-backed consensus edges (>=50% of replicates): 65
- Strong consensus edges (>=75% of replicates): 35
- Consensus differs from full primary donor: 19 taxa
- Consensus parent absent from full donor solution: 10 taxa

## Holdout Folds

- Fold 1: held-out cost 0.1477, coverage 94.70%, traits 1018
- Fold 2: held-out cost 0.2090, coverage 87.89%, traits 1098
- Fold 3: held-out cost 0.2368, coverage 84.80%, traits 974
- Fold 4: held-out cost 0.2163, coverage 87.08%, traits 944
- Fold 5: held-out cost 0.2142, coverage 87.31%, traits 1048

## Strongest Consensus Parents

- afm7: openelm with support 100.00% (full primary openelm)
- bailing_moe_linear: bailing_moe with support 100.00% (full primary bailing_moe)
- deepseek: mixtral with support 100.00% (full primary mixtral)
- deepseek_v2: deepseek with support 100.00% (full primary deepseek)
- deepseek_v3: deepseek_v2 with support 100.00% (full primary deepseek_v2)
- deepseek_v32: deepseek_v3 with support 100.00% (full primary deepseek_v3)
- glm4_moe_lite: deepseek_v3 with support 100.00% (full primary deepseek_v3)
- gpt_bigcode: gpt2 with support 100.00% (full primary gpt2)
- gpt_neox: gpt2 with support 100.00% (full primary gpt2)
- granitemoe: mixtral with support 100.00% (full primary mixtral)
- mamba2: mamba with support 100.00% (full primary mamba)
- minicpm3: deepseek_v2 with support 100.00% (full primary deepseek_v2)

## Most Ambiguous Consensus Parents

- glm: bitnet with support 20.00%, votes bitnet:4;mamba:3;phi3:3;phi3small:3;jamba:2;llama:2;gemma:1;minicpm:1;olmo:1
- nemotron-nas: mixtral with support 20.00%, votes deepseek:4;exaone:4;mixtral:4;granite:1;granitemoe:1;llama:1;mamba2:1;minicpm:1;phi:1;phi3:1;qwen:1
- lille-130m: olmo with support 25.00%, votes olmo:5;nemotron-nas:3;plamo2:3;afm7:2;glm:2;helium:2;openelm:1;phi3:1;phimoe:1
- minimax: dots1 with support 25.00%, votes Klear:5;dots1:5;bailing_moe_linear:2;glm4_moe:2;mixtral:2;bailing_moe:1;dbrx:1;granitemoe:1;openelm:1
- seed_oss: qwen2 with support 25.00%, votes minicpm:5;qwen2:5;granite:2;bitnet:1;ernie4_5:1;exaone:1;granitemoe:1;hunyuan_v1_dense:1;internlm3:1;olmo2:1;qwen3:1
- apertus: glm4_moe with support 30.00%, votes glm4_moe:6;smollm3:3;nemotron:2;olmo2:2;openelm:2;olmo:1;olmoe:1;phi:1;qwen3:1;qwen3_moe:1
- gemma3_text: cohere2 with support 30.00%, votes cohere2:6;olmo2:6;bailing_moe:3;gemma2:3;olmoe:1;recurrent_gemma:1
- minicpm: deepseek with support 30.00%, votes deepseek:6;llama:3;mixtral:3;nemotron:3;stablelm:3;internlm2:2
- afmoe: dots1 with support 35.00%, votes dots1:7;bailing_moe_linear:4;gpt_oss:4;exaone4:1;glm4_moe:1;mixtral:1;phimoe:1;qwen3_next:1
- cohere: phi with support 35.00%, votes phi:7;llama:5;gemma:3;stablelm:3;deepseek:1;gpt2:1
- gemma: stablelm with support 35.00%, votes stablelm:7;gpt2:6;gpt_bigcode:3;llama:2;plamo:2
- hunyuan_v1_dense: qwen3 with support 35.00%, votes internlm3:7;qwen3:7;hunyuan:3;dots1:1;olmo2:1;olmoe:1
