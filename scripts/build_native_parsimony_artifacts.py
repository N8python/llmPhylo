#!/usr/bin/env python3

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
INPUT_MATRIX = ROOT / "native_architecture_character_matrix.csv"
NORMALIZED_CSV = ROOT / "native_architecture_parsimony_matrix.csv"
CHARACTER_MANIFEST = ROOT / "native_architecture_parsimony_characters.csv"
NEXUS_OUTPUT = ROOT / "native_architecture_parsimony_matrix.nex"
TREE_OUTPUT = ROOT / "native_architecture_parsimony_tree.newick"
SUMMARY_OUTPUT = ROOT / "native_architecture_parsimony_summary.md"

EXPECTED_TAXA = 94
DEFAULT_REPLICATES = 16
DEFAULT_SEED = 13

CHARACTER_DESCRIPTIONS = {
    "modality": "Primary model modality coverage.",
    "stack_topology": "High-level arrangement of transformer, state-space, or recurrent blocks.",
    "state_update_rule": "Underlying recurrent/state-space update kernel used when the architecture is not pure dense attention.",
    "attention_family": "Core token mixing mechanism used for contextualization.",
    "attention_projection": "How Q/K/V or latent state projections are parameterized.",
    "projection_bias_scheme": "Bias-usage regime across the architecture's main attention/FFN/expert linear projections.",
    "q_lora_present": "Whether the architecture uses Q-side low-rank LoRA-style attention factorization.",
    "kv_lora_present": "Whether the architecture uses K/V-side low-rank LoRA-style attention factorization.",
    "kv_layout": "Key/value head-sharing layout or latent cache organization.",
    "cross_layer_kv_sharing": "Whether later layers reuse key/value states projected in other layers.",
    "attention_pattern": "Context-window or hybrid-attention scheduling pattern.",
    "hybrid_block_schedule": "How heterogeneous mixer block types are arranged across depth.",
    "window_schedule_detail": "Finer structure of sliding/local/global attention scheduling when present.",
    "secondary_token_mixer": "Whether some layers replace attention with a secondary non-attention token mixer.",
    "attention_output_gating": "Whether the attention or linear-attention output is multiplicatively gated after mixing.",
    "positional_encoding": "Primary positional encoding family.",
    "windowed_rope_scope": "How RoPE is scoped across local/global attention layers in windowed architectures.",
    "rope_partition_detail": "Detailed way RoPE is partitioned, split, or disabled across subspaces or layers.",
    "rope_parameterization": "Whether RoPE behavior is controlled by direct scaling or a rope_parameters object.",
    "qk_norm": "Normalization applied to Q/K (and optionally V) inside attention.",
    "qk_norm_detail": "More specific Q/K/V normalization variant when the coarse qk_norm class is insufficient.",
    "norm_scheme": "Primary normalization family used in blocks.",
    "block_norm_variant": "Specific normalization variant used at block boundaries beyond the broad RMSNorm/LayerNorm family.",
    "residual_scheme": "Residual scheduling around attention and feedforward blocks.",
    "residual_update_rule": "Whether residual updates are plain, scaled, or clipped on addition.",
    "ffn_core": "Base feedforward block family after removing MoE dispatch details.",
    "ffn_projection_style": "How the feedforward or expert cell projects and gates its hidden states.",
    "ffn_activation_core": "Primary nonlinearity used in the feedforward pathway.",
    "has_aux_silu_gate": "Whether a secondary SiLU-like gate appears alongside the core FFN activation.",
    "has_aux_relu2_gate": "Whether a secondary ReLU2-style gate appears alongside the core FFN activation.",
    "has_moe": "Whether the architecture includes routed experts.",
    "moe_distribution": "Whether MoE appears in all layers or only selected layers.",
    "moe_schedule_type": "Layer scheduling motif that determines where MoE blocks appear.",
    "moe_subtype": "Subtype of MoE block beyond simple presence and layer distribution.",
    "router_activation": "Router score transformation before top-k expert selection.",
    "router_activation_configurable": "Whether the router activation family is exposed as a configurable architecture choice.",
    "has_shared_expert": "Whether routed MoE is supplemented with shared dense experts.",
    "shared_branch_merge_mode": "How a shared dense branch, when present, is combined with routed expert output.",
    "shared_expert_configurable": "Whether the presence of shared experts is itself configurable.",
    "expert_bias_correction": "Whether router bias correction is part of expert selection.",
    "zero_expert_option": "Whether a zero/identity expert option is part of routing.",
    "tie_embeddings_default": "Default input/output embedding tying state.",
    "tie_embeddings_configurable": "Whether embedding tying is an exposed architecture knob.",
    "output_head_connection": "How logits connect to embeddings or output head weights.",
    "lm_head_bias": "Whether the LM head includes an explicit bias term.",
    "uses_logit_scale": "Whether logits are globally scaled after projection.",
    "uses_logit_softcap": "Whether logits are softcapped after projection.",
    "bitlinear_weights": "Uses BitLinear-style low-bit parameterization in the main stack.",
    "chunked_kv_cache": "Uses chunked KV caching as an architectural feature.",
    "laurel": "Includes Laurel-style corrective blocks.",
    "altup": "Includes alternating update correction blocks.",
    "conv_on_kv": "Applies short convolution to K/V states.",
    "periodic_no_rope_layers": "Contains layers that intentionally disable RoPE on a schedule.",
    "rglru_recurrence": "Uses Griffin-style RGLRU recurrence.",
    "gated_delta_net": "Includes a Gated Delta Net style linear-attention block.",
    "ngram_memory": "Adds explicit n-gram memory augmentation.",
    "same_block_attention_ssm": "Combines attention and state-space mixing inside the same block.",
    "dual_attention_modules": "Includes two distinct attention modules in the same architecture family.",
    "kv_reuse_attention": "Reuses cached KV structure beyond standard causal attention.",
    "coefficient_mix_shared_routed": "Learns a coefficient mix between shared and routed expert outputs.",
    "global_plus_linear_attention_mix": "Mixes global attention with recurrent linear attention layers.",
    "grouped_expert_selection": "Uses grouped expert selection rather than flat top-k alone.",
    "optional_nope": "Can disable RoPE in favor of NoPE-style positional handling.",
}

CHARACTER_FIELDS = [
    "modality",
    "stack_topology",
    "state_update_rule",
    "attention_family",
    "attention_projection",
    "projection_bias_scheme",
    "q_lora_present",
    "kv_lora_present",
    "kv_layout",
    "cross_layer_kv_sharing",
    "attention_pattern",
    "hybrid_block_schedule",
    "window_schedule_detail",
    "secondary_token_mixer",
    "attention_output_gating",
    "positional_encoding",
    "windowed_rope_scope",
    "rope_partition_detail",
    "rope_parameterization",
    "qk_norm",
    "qk_norm_detail",
    "norm_scheme",
    "block_norm_variant",
    "residual_scheme",
    "residual_update_rule",
    "ffn_core",
    "ffn_projection_style",
    "ffn_activation_core",
    "has_aux_silu_gate",
    "has_aux_relu2_gate",
    "has_moe",
    "moe_distribution",
    "moe_schedule_type",
    "moe_subtype",
    "router_activation",
    "router_activation_configurable",
    "has_shared_expert",
    "shared_branch_merge_mode",
    "shared_expert_configurable",
    "expert_bias_correction",
    "zero_expert_option",
    "tie_embeddings_default",
    "tie_embeddings_configurable",
    "output_head_connection",
    "lm_head_bias",
    "uses_logit_scale",
    "uses_logit_softcap",
    "bitlinear_weights",
    "chunked_kv_cache",
    "laurel",
    "altup",
    "conv_on_kv",
    "periodic_no_rope_layers",
    "rglru_recurrence",
    "gated_delta_net",
    "ngram_memory",
    "same_block_attention_ssm",
    "dual_attention_modules",
    "kv_reuse_attention",
    "coefficient_mix_shared_routed",
    "global_plus_linear_attention_mix",
    "grouped_expert_selection",
    "optional_nope",
]

BINARY_FIELDS = {
    "has_aux_silu_gate",
    "has_aux_relu2_gate",
    "has_moe",
    "q_lora_present",
    "kv_lora_present",
    "router_activation_configurable",
    "has_shared_expert",
    "shared_expert_configurable",
    "expert_bias_correction",
    "zero_expert_option",
    "tie_embeddings_configurable",
    "lm_head_bias",
    "uses_logit_scale",
    "uses_logit_softcap",
    "bitlinear_weights",
    "chunked_kv_cache",
    "laurel",
    "altup",
    "conv_on_kv",
    "periodic_no_rope_layers",
    "rglru_recurrence",
    "gated_delta_net",
    "ngram_memory",
    "same_block_attention_ssm",
    "dual_attention_modules",
    "kv_reuse_attention",
    "coefficient_mix_shared_routed",
    "global_plus_linear_attention_mix",
    "grouped_expert_selection",
    "optional_nope",
}

STATE_UPDATE_RULE_OVERRIDES = {
    "recurrent_gemma": "rg_lru",
    "rwkv7": "wkv_recurrence",
    "bailing_moe_linear": "gla_decay",
    "qwen3_next": "gated_delta",
    "qwen3_5": "gated_delta",
    "kimi_linear": "gated_delta",
}

HYBRID_BLOCK_SCHEDULE_OVERRIDES = {
    "falcon_h1": "same_block_parallel",
    "jamba": "periodic_global_insertion",
    "recurrent_gemma": "alternating_layer_types",
    "granitemoehybrid": "alternating_layer_types",
    "nemotron_h": "alternating_layer_types",
    "lfm2": "sparse_injected_layers",
    "lfm2_moe": "sparse_injected_layers",
    "bailing_moe_linear": "periodic_global_insertion",
    "plamo2": "periodic_global_insertion",
    "qwen3_next": "periodic_global_insertion",
    "qwen3_5": "periodic_global_insertion",
    "kimi_linear": "sparse_injected_layers",
}

MOE_SCHEDULE_TYPE_OVERRIDES = {
    "step3p5": "explicit_layer_list",
    "kimi_linear": "all_layers",
    "lfm2_moe": "dense_prefix_then_periodic",
    "nemotron_h": "periodic_with_exclusions",
    "qwen3_5": "all_layers",
}

CROSS_LAYER_KV_SHARING_OVERRIDES = {
    "afm7": "reuse_tail_layers",
    "hunyuan": "stride_shared_projection",
    "gemma3n": "terminal_shared_layers",
}

WINDOWED_ROPE_SCOPE_OVERRIDES = {
    "afmoe": "local_only",
    "cohere2": "local_only",
    "exaone4": "local_only",
    "exaone_moe": "local_only",
    "gemma3_text": "local_global_dual_base",
    "gemma3n": "local_global_dual_base",
}

SECONDARY_TOKEN_MIXER_OVERRIDES = {
    "lfm2": "short_depthwise_conv",
    "lfm2_moe": "short_depthwise_conv",
}

ATTENTION_OUTPUT_GATING_TAXA = {
    "afmoe",
    "bailing_moe_linear",
    "kimi_linear",
    "qwen3_5",
    "qwen3_next",
    "step3p5",
}

SHARED_BRANCH_MERGE_MODE_OVERRIDES = {
    "Klear": "learned_coefficient_mix",
    "kimi_linear": "none",
    "qwen2_moe": "sigmoid_gated_shared",
    "qwen3_5": "sigmoid_gated_shared",
    "qwen3_next": "sigmoid_gated_shared",
}

UNGATED_TWO_PROJ_TAXA = {
    "apertus",
    "gpt2",
    "gpt_bigcode",
    "gpt_neox",
    "nanochat",
    "nemotron",
    "nemotron_h",
    "phi",
    "phixtral",
    "starcoder2",
}

GATED_TWO_PROJ_FUSED_TAXA = {
    "glm",
    "glm4",
    "openelm",
    "phi3",
    "phi3small",
    "plamo2",
}

FUSED_SPLIT_HIDDEN_TAXA = {
    "olmo",
}

PROJECTION_BIAS_SCHEME_OVERRIDES = {
    "gpt2": "full_linear",
    "gpt_bigcode": "full_linear",
    "gpt_neox": "full_linear",
    "gpt_oss": "full_linear",
    "phi": "full_linear",
    "phi3small": "full_linear",
    "starcoder2": "full_linear",
    "mimo": "attention_only",
    "phimoe": "attention_only",
    "qwen": "attention_only",
    "qwen2": "attention_only",
    "qwen2_moe": "attention_only",
    "recurrent_gemma": "mixed_custom",
}

BLOCK_NORM_VARIANT_OVERRIDES = {
    "gemma": "gemma_rmsnorm_1plus",
    "gemma2": "gemma_rmsnorm_1plus",
    "gemma3_text": "gemma_rmsnorm_1plus",
    "recurrent_gemma": "gemma_rmsnorm_1plus",
    "step3p5": "zero_centered_rmsnorm",
    "nemotron": "layernorm_1p",
    "olmo": "affine_free_layernorm",
    "phimoe": "standard_layernorm",
}

RESIDUAL_UPDATE_RULE_OVERRIDES = {
    "gemma3_text": "clipped_add",
    "granite": "scaled_add",
    "granitemoe": "scaled_add",
    "granitemoehybrid": "scaled_add",
    "minicpm": "scaled_add",
    "minicpm3": "scaled_add",
}


def read_native_matrix(path: Path) -> list[dict[str, str]]:
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != EXPECTED_TAXA:
        raise ValueError(f"Expected {EXPECTED_TAXA} native taxa, found {len(rows)} in {path}.")
    return rows


def split_semicolon_field(value: str) -> set[str]:
    if not value:
        return set()
    return {item for item in value.split(";") if item}


def has_prefixed_token(tokens: set[str], prefix: str) -> bool:
    return prefix in tokens or any(item.startswith(prefix + "=") for item in tokens)


def parse_ffn_activation_core(value: str) -> str:
    if "swiglu" in value:
        return "swiglu"
    if "gegelu" in value:
        return "gegelu"
    if "relu2_glu" in value:
        return "relu2_glu"
    if "gelu_approx" in value:
        return "gelu_approx"
    if value == "gelu":
        return "gelu"
    if "silu" in value:
        return "silu"
    return "custom"


def parse_ffn_core(value: str) -> str:
    if value in {"dense", "moe", "hybrid_dense_plus_moe"}:
        return "dense"
    if value == "state_space_gated":
        return "state_space_gated"
    return value


def parse_moe_distribution(value: str) -> str:
    if value == "moe":
        return "all_layers"
    if value == "hybrid_dense_plus_moe":
        return "interleaved_or_periodic"
    return "none"


def parse_router_activation(value: str) -> str:
    if value == "none":
        return "none"
    return value.split("+", 1)[0]


def parse_tie_embeddings_default(value: str) -> str | None:
    if value in {"tied", "configurable_default_tied"}:
        return "tied"
    if value in {"untied_or_custom", "configurable_default_untied"}:
        return "untied"
    return None


def parse_output_head_connection(value: str) -> str:
    if value in {"tied_embedding_head", "tied_plus_logit_scale"}:
        return "tied"
    if value in {"untied_linear", "linear_bias"}:
        return "untied_linear"
    return "custom"


def parse_rope_parameterization(value: str) -> str:
    knobs = split_semicolon_field(value)
    if has_prefixed_token(knobs, "rope_parameters"):
        return "rope_parameters"
    if has_prefixed_token(knobs, "rope_scaling"):
        return "rope_scaling"
    return "none"


def parse_window_schedule_detail(
    attention_pattern: str, traits: set[str], knobs: set[str]
) -> str:
    if "swa_head_split" in traits:
        return "head_split_swa"
    if "alternating_local_and_global_attention" in traits:
        return "alternating_local_global"
    if "layerwise_sliding_window_set" in traits or "sliding_window_layers" in knobs:
        return "layerwise_sliding"
    if has_prefixed_token(knobs, "sliding_window") or any(
        label in attention_pattern
        for label in ("sliding_window", "mixed_global_plus_sliding_window", "periodic_global_plus_sliding_window")
    ):
        return "fixed_sliding"
    return "none"


def parse_rope_partition_detail(traits: set[str]) -> str:
    if "manual_rope_score_split" in traits:
        return "manual_rope_score_split"
    if "mrope_sections" in traits:
        return "mrope_sections"
    if "rope_free_dense_layers" in traits:
        return "rope_free_dense_layers"
    if "rope_nope_split" in traits:
        return "rope_nope_split"
    return "none"


def parse_qk_norm_detail(qk_norm: str, traits: set[str]) -> str:
    if "per_head_layernorm_qk" in traits:
        return "per_head_layernorm_qk"
    return qk_norm


def parse_moe_subtype(mlp_type: str, moe_routing: str, traits: set[str]) -> str:
    if mlp_type not in {"moe", "hybrid_dense_plus_moe"}:
        return "none"
    if "switch_mlp_moe" in traits:
        return "switch_mlp"
    if "periodic_sparse_moe" in traits:
        return "periodic_sparse"
    if moe_routing.startswith("topk_custom") or "router_can_use_sigmoid_or_softmax" in traits:
        return "custom"
    return "standard_topk"


def parse_state_update_rule(row: dict[str, str], traits: set[str]) -> str:
    taxon = row["model"]
    if taxon in STATE_UPDATE_RULE_OVERRIDES:
        return STATE_UPDATE_RULE_OVERRIDES[taxon]
    if "gated_delta" in row["attention_family"] or "gated_delta_net" in traits:
        return "gated_delta"
    if row["stack_topology"] in {"ssm_only", "hybrid_attention_ssm"}:
        return "selective_ssm"
    return "none"


def parse_hybrid_block_schedule(row: dict[str, str], traits: set[str]) -> str:
    taxon = row["model"]
    if taxon in HYBRID_BLOCK_SCHEDULE_OVERRIDES:
        return HYBRID_BLOCK_SCHEDULE_OVERRIDES[taxon]
    if "attention_and_mamba_in_same_block" in traits:
        return "same_block_parallel"
    return "none"


def parse_cross_layer_kv_sharing(row: dict[str, str]) -> str:
    return CROSS_LAYER_KV_SHARING_OVERRIDES.get(row["model"], "none")


def parse_windowed_rope_scope(row: dict[str, str]) -> str:
    return WINDOWED_ROPE_SCOPE_OVERRIDES.get(row["model"], "uniform_all_layers")


def parse_secondary_token_mixer(row: dict[str, str]) -> str:
    return SECONDARY_TOKEN_MIXER_OVERRIDES.get(row["model"], "none")


def parse_attention_output_gating(row: dict[str, str]) -> str:
    return "post_attention_gate" if row["model"] in ATTENTION_OUTPUT_GATING_TAXA else "none"


def parse_moe_schedule_type(row: dict[str, str], knobs: set[str]) -> str:
    taxon = row["model"]
    if row["mlp_type"] not in {"moe", "hybrid_dense_plus_moe"}:
        return "none"
    if taxon in MOE_SCHEDULE_TYPE_OVERRIDES:
        return MOE_SCHEDULE_TYPE_OVERRIDES[taxon]
    if has_prefixed_token(knobs, "moe_layers_enum"):
        return "explicit_layer_list"
    if (
        has_prefixed_token(knobs, "decoder_sparse_step")
        or has_prefixed_token(knobs, "interleave_moe_layer_step")
        or has_prefixed_token(knobs, "expert_layer_period")
    ):
        return "periodic_with_exclusions"
    if (
        has_prefixed_token(knobs, "first_k_dense_replace")
        or has_prefixed_token(knobs, "moe_layer_freq")
    ):
        return "dense_prefix_then_periodic"
    return "all_layers"


def parse_shared_branch_merge_mode(
    row: dict[str, str], moe_routing: str, traits: set[str]
) -> str:
    taxon = row["model"]
    if taxon in SHARED_BRANCH_MERGE_MODE_OVERRIDES:
        return SHARED_BRANCH_MERGE_MODE_OVERRIDES[taxon]
    if taxon in {"granitemoehybrid", "step3p5"}:
        return "additive"
    if "+shared" in moe_routing or "shared_expert_toggle" in traits:
        return "additive"
    return "none"


def parse_ffn_projection_style(row: dict[str, str]) -> str:
    taxon = row["model"]
    if taxon in UNGATED_TWO_PROJ_TAXA:
        return "ungated_two_proj"
    if taxon in GATED_TWO_PROJ_FUSED_TAXA:
        return "gated_two_proj_fused"
    if taxon in FUSED_SPLIT_HIDDEN_TAXA:
        return "fused_split_hidden"
    return "gated_three_proj_separate"


def parse_projection_bias_scheme(row: dict[str, str]) -> str:
    return PROJECTION_BIAS_SCHEME_OVERRIDES.get(row["model"], "none")


def parse_block_norm_variant(row: dict[str, str]) -> str:
    taxon = row["model"]
    if taxon in BLOCK_NORM_VARIANT_OVERRIDES:
        return BLOCK_NORM_VARIANT_OVERRIDES[taxon]
    if row["norm_scheme"] == "layernorm":
        return "standard_layernorm"
    return "standard_rmsnorm"


def parse_residual_update_rule(row: dict[str, str]) -> str:
    return RESIDUAL_UPDATE_RULE_OVERRIDES.get(row["model"], "plain_add")


def normalize_row(row: dict[str, str]) -> dict[str, object]:
    traits = split_semicolon_field(row["special_traits"])
    knobs = split_semicolon_field(row["architecture_knobs"])
    activation = row["mlp_activation"]
    ffn_activation_core = parse_ffn_activation_core(activation)
    moe_routing = row["moe_routing"]
    embedding_tying = row["embedding_tying"]
    logits_head = row["logits_head"]
    shared_branch_merge_mode = parse_shared_branch_merge_mode(row, moe_routing, traits)
    state_update_rule = parse_state_update_rule(row, traits)

    return {
        "taxon": row["model"],
        "modality": row["modality"],
        "stack_topology": row["stack_topology"],
        "state_update_rule": state_update_rule,
        "attention_family": row["attention_family"],
        "attention_projection": row["attention_projection"],
        "projection_bias_scheme": parse_projection_bias_scheme(row),
        "q_lora_present": int("q_lora" in traits or has_prefixed_token(knobs, "q_lora_rank")),
        "kv_lora_present": int("kv_lora" in traits or has_prefixed_token(knobs, "kv_lora_rank")),
        "kv_layout": row["kv_layout"],
        "cross_layer_kv_sharing": parse_cross_layer_kv_sharing(row),
        "attention_pattern": row["attention_pattern"],
        "hybrid_block_schedule": parse_hybrid_block_schedule(row, traits),
        "window_schedule_detail": parse_window_schedule_detail(
            row["attention_pattern"], traits, knobs
        ),
        "secondary_token_mixer": parse_secondary_token_mixer(row),
        "attention_output_gating": parse_attention_output_gating(row),
        "positional_encoding": row["positional_encoding"],
        "windowed_rope_scope": parse_windowed_rope_scope(row),
        "rope_partition_detail": parse_rope_partition_detail(traits),
        "rope_parameterization": parse_rope_parameterization(row["architecture_knobs"]),
        "qk_norm": row["qk_norm"],
        "qk_norm_detail": parse_qk_norm_detail(row["qk_norm"], traits),
        "norm_scheme": row["norm_scheme"],
        "block_norm_variant": parse_block_norm_variant(row),
        "residual_scheme": row["residual_scheme"],
        "residual_update_rule": parse_residual_update_rule(row),
        "ffn_core": parse_ffn_core(row["mlp_type"]),
        "ffn_projection_style": parse_ffn_projection_style(row),
        "ffn_activation_core": ffn_activation_core,
        "has_aux_silu_gate": int("silu" in activation and ffn_activation_core != "silu"),
        "has_aux_relu2_gate": int(
            "relu2_glu" in activation and ffn_activation_core != "relu2_glu"
        ),
        "has_moe": int(row["mlp_type"] in {"moe", "hybrid_dense_plus_moe"}),
        "moe_distribution": parse_moe_distribution(row["mlp_type"]),
        "moe_schedule_type": parse_moe_schedule_type(row, knobs),
        "moe_subtype": parse_moe_subtype(row["mlp_type"], moe_routing, traits),
        "router_activation": parse_router_activation(moe_routing),
        "router_activation_configurable": int("router_can_use_sigmoid_or_softmax" in traits),
        "has_shared_expert": int(shared_branch_merge_mode != "none"),
        "shared_branch_merge_mode": shared_branch_merge_mode,
        "shared_expert_configurable": int("shared_expert_toggle" in traits),
        "expert_bias_correction": int("bias_correction" in moe_routing),
        "zero_expert_option": int("zero_expert" in moe_routing),
        "tie_embeddings_default": parse_tie_embeddings_default(embedding_tying),
        "tie_embeddings_configurable": int("configurable" in embedding_tying),
        "output_head_connection": parse_output_head_connection(logits_head),
        "lm_head_bias": int(logits_head == "linear_bias"),
        "uses_logit_scale": int(logits_head == "tied_plus_logit_scale"),
        "uses_logit_softcap": int(logits_head == "linear_or_tied_plus_softcap"),
        "bitlinear_weights": int("bitlinear_weights" in traits),
        "chunked_kv_cache": int("chunked_kv_cache" in traits),
        "laurel": int("laurel" in traits or "laurel_blocks" in traits),
        "altup": int("altup" in traits or "altup_correction" in traits),
        "conv_on_kv": int("conv_on_kv" in traits or "conv2_preconditioned_kv" in traits),
        "periodic_no_rope_layers": int("periodic_no_rope_layers" in traits),
        "rglru_recurrence": int("rglru_recurrence" in traits),
        "gated_delta_net": int(
            "gated_delta_net" in traits or state_update_rule == "gated_delta"
        ),
        "ngram_memory": int("ngram_memory" in traits or "ngram_memory_adapter" in traits),
        "same_block_attention_ssm": int("attention_and_mamba_in_same_block" in traits),
        "dual_attention_modules": int("dual_attention_modules" in traits),
        "kv_reuse_attention": int("kv_reuse_attention" in traits),
        "coefficient_mix_shared_routed": int(
            "coefficient_mix_shared_and_routed_experts" in traits
        ),
        "global_plus_linear_attention_mix": int("global_plus_linear_attention_mix" in traits),
        "grouped_expert_selection": int("grouped_expert_selection" in traits),
        "optional_nope": int("optional_nope" in traits),
    }


def build_normalized_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    normalized = [normalize_row(row) for row in rows]
    duplicate_signatures = Counter(
        tuple((field, row[field]) for field in CHARACTER_FIELDS) for row in normalized
    )
    duplicate_groups = sum(1 for count in duplicate_signatures.values() if count > 1)
    if duplicate_groups:
        raise ValueError(
            f"Normalized matrix still contains {duplicate_groups} duplicate character bundles."
        )
    return normalized


def write_normalized_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = ["taxon", *CHARACTER_FIELDS]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {}
            for field in fieldnames:
                value = row[field]
                out[field] = "?" if value is None else value
            writer.writerow(out)


def build_character_manifest(rows: list[dict[str, object]]) -> list[dict[str, str]]:
    manifest = []
    for field in CHARACTER_FIELDS:
        states = sorted(
            {str(row[field]) for row in rows if row[field] is not None},
            key=lambda value: (len(value), value),
        )
        manifest.append(
            {
                "character": field,
                "kind": "binary" if field in BINARY_FIELDS else "multistate",
                "description": CHARACTER_DESCRIPTIONS[field],
                "states": "|".join(states),
            }
        )
    return manifest


def write_character_manifest(rows: list[dict[str, str]], path: Path) -> None:
    fieldnames = ["character", "kind", "description", "states"]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def encode_character_matrix(
    rows: list[dict[str, object]],
) -> tuple[list[str], list[np.ndarray], list[list[str]], list[str]]:
    taxa = [str(row["taxon"]) for row in rows]
    state_lists = []
    for field in CHARACTER_FIELDS:
        if field in BINARY_FIELDS:
            state_lists.append(["0", "1"])
            continue
        states = sorted({str(row[field]) for row in rows if row[field] is not None})
        state_lists.append(states)

    mask_library = []
    encoded_strings = []
    nchar = len(CHARACTER_FIELDS)

    for row in rows:
        masks = np.zeros(nchar, dtype=np.uint64)
        symbols = []
        for index, field in enumerate(CHARACTER_FIELDS):
            states = state_lists[index]
            if field in BINARY_FIELDS:
                value = str(int(row[field]))
            else:
                value = None if row[field] is None else str(row[field])

            if value is None:
                mask = (np.uint64(1) << np.uint64(len(states))) - np.uint64(1)
                masks[index] = mask
                symbols.append("?")
            else:
                state_index = states.index(value)
                masks[index] = np.uint64(1) << np.uint64(state_index)
                symbols.append(value)
        mask_library.append(masks)
        encoded_strings.append(symbols)

    return taxa, mask_library, state_lists, encoded_strings


def build_nexus_sequences(
    rows: list[dict[str, object]], state_lists: list[list[str]]
) -> dict[str, str]:
    symbols = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    if any(len(states) > len(symbols) for states in state_lists):
        raise ValueError("One of the characters has too many states for the NEXUS symbol map.")

    sequences = {}
    for row in rows:
        chars = []
        for index, field in enumerate(CHARACTER_FIELDS):
            value = row[field]
            if value is None:
                chars.append("?")
                continue
            if field in BINARY_FIELDS:
                chars.append(str(int(value)))
                continue
            chars.append(symbols[state_lists[index].index(str(value))])
        sequences[str(row["taxon"])] = "".join(chars)
    return sequences


def write_nexus(
    rows: list[dict[str, object]],
    state_lists: list[list[str]],
    output_path: Path,
) -> None:
    sequences = build_nexus_sequences(rows, state_lists)
    symbol_alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    max_states = max(len(states) for states in state_lists)
    symbols = symbol_alphabet[: max(max_states, 2)]

    with output_path.open("w") as fh:
        fh.write("#NEXUS\n\n")
        fh.write("BEGIN DATA;\n")
        fh.write(f"    DIMENSIONS NTAX={len(rows)} NCHAR={len(CHARACTER_FIELDS)};\n")
        fh.write(
            f'    FORMAT DATATYPE=STANDARD MISSING=? GAP=- SYMBOLS="{symbols}";\n'
        )
        fh.write("    MATRIX\n")
        for taxon in sorted(sequences):
            fh.write(f"    '{taxon}' {sequences[taxon]}\n")
        fh.write("    ;\n")
        fh.write("END;\n")


def clone_adjacency(adjacency: dict[int, set[int]]) -> dict[int, set[int]]:
    return {node: set(neighbors) for node, neighbors in adjacency.items()}


def tree_edges(adjacency: dict[int, set[int]]) -> list[tuple[int, int]]:
    edges = []
    for node, neighbors in adjacency.items():
        for neighbor in neighbors:
            if node < neighbor:
                edges.append((node, neighbor))
    return edges


def internal_edges(adjacency: dict[int, set[int]], ntax: int) -> list[tuple[int, int]]:
    return [
        edge
        for edge in tree_edges(adjacency)
        if edge[0] >= ntax and edge[1] >= ntax
    ]


def add_edge(adjacency: dict[int, set[int]], left: int, right: int) -> None:
    adjacency.setdefault(left, set()).add(right)
    adjacency.setdefault(right, set()).add(left)


def remove_edge(adjacency: dict[int, set[int]], left: int, right: int) -> None:
    adjacency[left].remove(right)
    adjacency[right].remove(left)


def make_initial_tree(order: list[int], next_internal: int) -> tuple[dict[int, set[int]], int]:
    adjacency: dict[int, set[int]] = {}
    center = next_internal
    add_edge(adjacency, center, order[0])
    add_edge(adjacency, center, order[1])
    add_edge(adjacency, center, order[2])
    return adjacency, next_internal + 1


def insert_taxon_on_edge(
    adjacency: dict[int, set[int]],
    edge: tuple[int, int],
    taxon: int,
    next_internal: int,
) -> dict[int, set[int]]:
    candidate = clone_adjacency(adjacency)
    left, right = edge
    remove_edge(candidate, left, right)
    add_edge(candidate, left, next_internal)
    add_edge(candidate, right, next_internal)
    add_edge(candidate, taxon, next_internal)
    return candidate


def orient_tree(adjacency: dict[int, set[int]], root: int) -> tuple[dict[int, int | None], list[int]]:
    parent: dict[int, int | None] = {root: None}
    order = [root]
    stack = [root]

    while stack:
        node = stack.pop()
        for neighbor in adjacency[node]:
            if neighbor == parent[node]:
                continue
            parent[neighbor] = node
            order.append(neighbor)
            stack.append(neighbor)

    return parent, order


def fitch_score(
    adjacency: dict[int, set[int]],
    leaf_masks: list[np.ndarray],
    ntax: int,
) -> int:
    internal_nodes = [node for node in adjacency if node >= ntax]
    root = min(internal_nodes) if internal_nodes else 0
    parent, order = orient_tree(adjacency, root)
    states: dict[int, np.ndarray] = {}
    score = 0

    for node in reversed(order):
        children = [child for child in adjacency[node] if parent.get(child) == node]
        if node < ntax:
            states[node] = leaf_masks[node]
            continue

        current = states[children[0]].copy()
        for child in children[1:]:
            child_state = states[child]
            intersection = current & child_state
            disjoint = intersection == 0
            score += int(np.count_nonzero(disjoint))
            current = np.where(disjoint, current | child_state, intersection)
        states[node] = current

    return score


def best_insertion(
    adjacency: dict[int, set[int]],
    taxon: int,
    next_internal: int,
    leaf_masks: list[np.ndarray],
    ntax: int,
    rng: random.Random,
) -> tuple[dict[int, set[int]], int]:
    best_score = None
    best_candidates: list[dict[int, set[int]]] = []

    for edge in tree_edges(adjacency):
        candidate = insert_taxon_on_edge(adjacency, edge, taxon, next_internal)
        score = fitch_score(candidate, leaf_masks, ntax)
        if best_score is None or score < best_score:
            best_score = score
            best_candidates = [candidate]
        elif score == best_score:
            best_candidates.append(candidate)

    return rng.choice(best_candidates), best_score if best_score is not None else 0


def swap_subtrees(
    adjacency: dict[int, set[int]],
    node_a: int,
    node_b: int,
    subtree_a: int,
    subtree_b: int,
) -> dict[int, set[int]]:
    candidate = clone_adjacency(adjacency)
    remove_edge(candidate, node_a, subtree_a)
    remove_edge(candidate, node_b, subtree_b)
    add_edge(candidate, node_a, subtree_b)
    add_edge(candidate, node_b, subtree_a)
    return candidate


def nni_neighbors(
    adjacency: dict[int, set[int]],
    edge: tuple[int, int],
    ntax: int,
) -> list[dict[int, set[int]]]:
    left, right = edge
    left_neighbors = [node for node in adjacency[left] if node != right]
    right_neighbors = [node for node in adjacency[right] if node != left]

    if len(left_neighbors) != 2 or len(right_neighbors) != 2:
        return []

    a, b = left_neighbors
    c, d = right_neighbors
    return [
        swap_subtrees(adjacency, left, right, b, c),
        swap_subtrees(adjacency, left, right, b, d),
    ]


def improve_with_nni(
    adjacency: dict[int, set[int]],
    leaf_masks: list[np.ndarray],
    ntax: int,
    rng: random.Random,
) -> tuple[dict[int, set[int]], int]:
    current = adjacency
    current_score = fitch_score(current, leaf_masks, ntax)

    improved = True
    while improved:
        improved = False
        edges = internal_edges(current, ntax)
        rng.shuffle(edges)

        for edge in edges:
            candidates = []
            for candidate in nni_neighbors(current, edge, ntax):
                candidates.append((fitch_score(candidate, leaf_masks, ntax), candidate))

            if not candidates:
                continue

            best_score, best_candidate = min(candidates, key=lambda item: item[0])
            if best_score < current_score:
                current = best_candidate
                current_score = best_score
                improved = True
                break

    return current, current_score


def heuristic_parsimony_search(
    taxa: list[str],
    leaf_masks: list[np.ndarray],
    replicates: int,
    seed: int,
) -> tuple[dict[int, set[int]], int]:
    ntax = len(taxa)
    overall_best_tree = None
    overall_best_score = None

    for replicate in range(replicates):
        rng = random.Random(seed + replicate)
        order = list(range(ntax))
        rng.shuffle(order)

        adjacency, next_internal = make_initial_tree(order, ntax)
        for taxon in order[3:]:
            adjacency, _ = best_insertion(
                adjacency, taxon, next_internal, leaf_masks, ntax, rng
            )
            next_internal += 1

        adjacency, score = improve_with_nni(adjacency, leaf_masks, ntax, rng)

        if overall_best_score is None or score < overall_best_score:
            overall_best_tree = adjacency
            overall_best_score = score

    if overall_best_tree is None or overall_best_score is None:
        raise ValueError("Parsimony search failed to produce a tree.")

    return overall_best_tree, overall_best_score


def quote_newick_label(label: str) -> str:
    return "'" + label.replace("'", "''") + "'"


def subtree_taxa(
    adjacency: dict[int, set[int]],
    start: int,
    parent: int,
    ntax: int,
    taxa: list[str],
) -> list[str]:
    collected = []
    stack = [(start, parent)]

    while stack:
        node, node_parent = stack.pop()
        if node < ntax:
            collected.append(taxa[node])
            continue
        for neighbor in adjacency[node]:
            if neighbor != node_parent:
                stack.append((neighbor, node))

    return sorted(collected)


def to_newick(
    adjacency: dict[int, set[int]],
    taxa: list[str],
) -> str:
    ntax = len(taxa)
    internal = [node for node in adjacency if node >= ntax]
    root = min(internal) if internal else 0

    def recurse(node: int, parent: int | None) -> str:
        if node < ntax:
            return quote_newick_label(taxa[node])

        children = [neighbor for neighbor in adjacency[node] if neighbor != parent]
        rendered = ",".join(recurse(child, node) for child in children)
        return f"({rendered})"

    return recurse(root, None) + ";"


def write_tree(adjacency: dict[int, set[int]], taxa: list[str], path: Path) -> None:
    path.write_text(to_newick(adjacency, taxa) + "\n")


def write_summary(
    normalized_rows: list[dict[str, object]],
    score: int,
    replicates: int,
    seed: int,
    adjacency: dict[int, set[int]],
    taxa: list[str],
    output_path: Path,
) -> None:
    ntax = len(taxa)
    internal = [node for node in adjacency if node >= ntax]
    root = min(internal) if internal else 0
    root_children = [neighbor for neighbor in adjacency[root]]
    clades = []

    for child in root_children:
        members = subtree_taxa(adjacency, child, root, ntax, taxa)
        preview = ", ".join(members[:8])
        if len(members) > 8:
            preview += ", ..."
        clades.append((len(members), preview))

    lines = [
        "# Native Architecture Maximum Parsimony",
        "",
        f"- Taxa: {len(normalized_rows)}",
        f"- Characters: {len(CHARACTER_FIELDS)}",
        f"- Duplicate normalized profiles: 0",
        f"- Heuristic search: {replicates} random-addition replicate(s) with NNI hill-climbing",
        f"- Random seed base: {seed}",
        f"- Best parsimony score: {score}",
        "",
        "## Output Files",
        "",
        f"- {NORMALIZED_CSV.name}",
        f"- {CHARACTER_MANIFEST.name}",
        f"- {NEXUS_OUTPUT.name}",
        f"- {TREE_OUTPUT.name}",
        "",
        "## Rooted View Preview",
        "",
    ]

    for size, preview in sorted(clades, reverse=True):
        lines.append(f"- {size} taxa: {preview}")

    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a normalized native-only parsimony matrix and maximum parsimony tree."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_MATRIX,
        help=f"Input native matrix CSV (default: {INPUT_MATRIX})",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=DEFAULT_REPLICATES,
        help=f"Random-addition parsimony replicates (default: {DEFAULT_REPLICATES})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Base RNG seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--skip-search",
        action="store_true",
        help="Only rebuild the normalized matrix, manifest, and NEXUS export; skip parsimony tree search.",
    )
    args = parser.parse_args()

    raw_rows = read_native_matrix(args.input)
    normalized_rows = build_normalized_rows(raw_rows)
    manifest = build_character_manifest(normalized_rows)

    write_normalized_csv(normalized_rows, NORMALIZED_CSV)
    write_character_manifest(manifest, CHARACTER_MANIFEST)

    taxa, leaf_masks, state_lists, _ = encode_character_matrix(normalized_rows)
    write_nexus(normalized_rows, state_lists, NEXUS_OUTPUT)

    print(f"Wrote normalized matrix: {NORMALIZED_CSV}")
    print(f"Wrote character manifest: {CHARACTER_MANIFEST}")
    print(f"Wrote NEXUS matrix: {NEXUS_OUTPUT}")
    print(f"Taxa: {len(normalized_rows)}")
    print(f"Characters: {len(CHARACTER_FIELDS)}")
    if args.skip_search:
        print("Skipped parsimony search.")
        return

    adjacency, score = heuristic_parsimony_search(
        taxa, leaf_masks, args.replicates, args.seed
    )
    write_tree(adjacency, taxa, TREE_OUTPUT)
    write_summary(
        normalized_rows,
        score,
        args.replicates,
        args.seed,
        adjacency,
        taxa,
        SUMMARY_OUTPUT,
    )

    print(f"Wrote Newick tree: {TREE_OUTPUT}")
    print(f"Wrote summary: {SUMMARY_OUTPUT}")
    print(f"Best parsimony score: {score}")


if __name__ == "__main__":
    main()
