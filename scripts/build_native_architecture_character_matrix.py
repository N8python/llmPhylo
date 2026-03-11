#!/usr/bin/env python3

import argparse
import ast
import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
DEFAULT_OUTPUT = ROOT / "native_architecture_character_matrix.csv"

# These files are wrappers, inheritance adapters, or re-exports rather than
# native architecture implementations. The matrix is intentionally scoped to
# native architectures only.
NON_NATIVE_EXCLUSIONS = {
    "gemma3": "wrapper around gemma3_text",
    "glm_moe_dsa": "inheritance adapter over deepseek_v32",
    "kimi_k25": "wrapper around deepseek_v3",
    "kimi_vl": "wrapper around deepseek_v3",
    "lfm2-vl": "wrapper around lfm2",
    "mistral3": "wrapper around llama or ministral3",
    "pixtral": "wrapper around llama",
    "qwen2_vl": "wrapper around qwen2",
    "qwen3_5_moe": "inheritance adapter over qwen3_5",
    "qwen3_vl": "wrapper around qwen3",
    "qwen3_vl_moe": "wrapper around qwen3_moe",
    "solar_open": "re-export of glm4_moe.Model",
}

EXPECTED_NATIVE_COUNT = 106 - len(NON_NATIVE_EXCLUSIONS)

IMPORTANT_KNOBS = (
    "num_attention_heads",
    "num_key_value_heads",
    "multi_query",
    "head_dim",
    "q_lora_rank",
    "kv_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
    "partial_rotary_factor",
    "rotary_pct",
    "rotary_dim",
    "rope_scaling",
    "rope_parameters",
    "rope_local_base_freq",
    "sliding_window",
    "sliding_window_pattern",
    "sliding_window_layers",
    "attention_window_size",
    "block_types",
    "layer_types",
    "layers_block_type",
    "interleave_moe_layer_step",
    "decoder_sparse_step",
    "expert_layer_period",
    "expert_layer_offset",
    "moe_layer_freq",
    "first_k_dense_replace",
    "num_experts",
    "n_routed_experts",
    "num_local_experts",
    "num_experts_per_tok",
    "num_experts_per_token",
    "n_shared_experts",
    "num_shared_experts",
    "shared_expert_intermediate_size",
    "moe_shared_expert_intermediate_size",
    "zero_expert_num",
    "zero_expert_type",
    "position_embedding_type",
    "use_qk_norm",
    "normalize_qk_projections",
    "attn_temperature_tuning",
    "laurel_rank",
    "altup_num_inputs",
    "conv_window",
    "mamba_enabled",
    "mamba_d_state",
    "mamba_n_heads",
    "final_logit_softcapping",
    "tie_word_embeddings",
)

MARKER_KNOBS = (
    ("BitLinear", "BitLinear"),
    ("ChunkedKVCache", "ChunkedKVCache"),
    ("DeepseekV2YarnRotaryEmbedding", "YaRN"),
    ("DynamicNTKScalingRoPE", "DynamicNTKRoPE"),
    ("DynamicNTKAlphaRoPE", "DynamicNTKAlphaRoPE"),
    ("GatedDeltaNet", "GatedDeltaNet"),
    ("RGLRU", "RGLRU"),
    ("SuScaledRoPE", "SuScaledRoPE"),
)

MANUAL_SPECIAL_TRAITS = {
    "Klear": ["coefficient_mix_shared_and_routed_experts"],
    "afm7": ["kv_reuse_attention", "fused_lora_or_quantized_linear"],
    "afmoe": ["alternating_local_and_global_attention", "grouped_expert_selection"],
    "baichuan_m1": ["conv2_preconditioned_kv", "layerwise_sliding_window_set"],
    "bailing_moe": ["router_can_use_sigmoid_or_softmax", "shared_expert_toggle"],
    "bailing_moe_linear": [
        "global_plus_linear_attention_mix",
        "router_can_use_sigmoid_or_softmax",
        "shared_expert_toggle",
    ],
    "deepseek_v2": ["mla_decoupled_rope", "softmax_router_with_shared_experts"],
    "deepseek_v3": ["mla_decoupled_rope", "sigmoid_router_with_shared_experts"],
    "deepseek_v32": ["dual_attention_modules", "mla_decoupled_rope"],
    "falcon_h1": ["attention_and_mamba_in_same_block"],
    "gemma3n": [
        "laurel_blocks",
        "altup_correction",
        "gelu_topk",
        "dual_base_rope",
        "softcapped_logits",
    ],
    "granitemoehybrid": ["attention_or_mamba_layer_types", "optional_nope"],
    "jamba": ["layerwise_attention_or_mamba", "periodic_sparse_moe"],
    "kimi_linear": ["mla_plus_gated_delta", "manual_rope_score_split"],
    "llama4": ["periodic_no_rope_layers", "chunked_kv_cache"],
    "longcat_flash": ["mla_decoupled_rope", "zero_expert_identity_option"],
    "longcat_flash_ngram": ["ngram_memory_adapter"],
    "nemotron_h": ["switch_mlp_moe"],
    "qwen3_5": ["mixed_full_attention_and_gated_delta", "mrope_sections"],
    "recurrent_gemma": ["griffin_blocks", "local_attention_plus_rglru"],
    "rwkv7": ["recurrent_time_mixing", "custom_state_update"],
}

MANUAL_COLUMN_OVERRIDES = {
    "bailing_moe_linear": {
        "mixer_signature": "hybrid_sdpa_plus_linear_attention",
    },
    "qwen3_5": {
        "mixer_signature": "hybrid_sdpa_plus_gated_delta",
        "mlp_activation": "swiglu+silu",
    },
}


def read(path: Path) -> str:
    return path.read_text()


def parse_model_args(path: Path) -> dict[str, object]:
    tree = ast.parse(read(path))
    fields: dict[str, object] = {}

    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name not in {"ModelArgs", "TextArgs", "TextModelArgs", "TextConfig"}:
            continue
        for stmt in node.body:
            if not isinstance(stmt, ast.AnnAssign) or not isinstance(stmt.target, ast.Name):
                continue
            name = stmt.target.id
            fields[name] = parse_default_value(stmt.value)

    return fields


def parse_default_value(node: ast.AST | None) -> object:
    if node is None:
        return None
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        if isinstance(node.operand, ast.Constant):
            return -node.operand.value
    return ast.unparse(node)


def has_any(text: str, *needles: str) -> bool:
    return any(needle in text for needle in needles)


def has_all(text: str, *needles: str) -> bool:
    return all(needle in text for needle in needles)


def classify_modality(text: str, stem: str) -> str:
    has_audio = has_any(text, "audio_tower", "embed_audio")
    has_vision = has_any(
        text,
        "vision_tower",
        "vision_model",
        "embed_vision",
        "multi_modal_projector",
        "mm_projector",
    ) or stem in {"gemma3n", "llama4"}

    if has_audio and has_vision:
        return "audio+vision+text"
    if has_vision:
        return "vision+text"
    return "text"


def classify_stack_topology(text: str, stem: str) -> str:
    if stem == "rwkv7":
        return "recurrent_rwkv"
    if has_all(text, "LocalAttentionBlock", "RGLRU"):
        return "hybrid_recurrent_attention"
    if "gated_delta_update" in text:
        return "hybrid_attention_linear"
    if has_all(text, "create_ssm_mask", "scaled_dot_product_attention") and has_any(
        text, "recurrent_gla", "class LinearAttention", "layers_block_type", "JambaMambaMixer"
    ):
        return "hybrid_attention_ssm" if "JambaMambaMixer" in text else "hybrid_attention_linear"
    if has_all(text, "create_ssm_mask", "scaled_dot_product_attention", "ssm_update"):
        return "hybrid_attention_ssm"
    if has_any(text, "class MambaBlock", "class Mamba2Block", "ssm_update"):
        return "ssm_only"
    return "transformer"


def classify_attention_family(text: str, stem: str) -> str:
    if stem == "rwkv7":
        return "rwkv7_state_mixing"
    if "gated_delta_update" in text:
        if has_any(text, "kv_lora_rank", "q_lora_rank", "qk_nope_head_dim", "MultiLinear"):
            return "mla_plus_gated_delta"
        return "sdpa_plus_gated_delta"
    if has_any(text, "recurrent_gla", "class LinearAttention") and "scaled_dot_product_attention" in text:
        return "sdpa_plus_linear_attention"
    if has_any(text, "kv_lora_rank", "q_lora_rank", "qk_nope_head_dim", "MultiLinear"):
        return "mla"
    if has_all(text, "create_ssm_mask", "scaled_dot_product_attention") and "JambaMambaMixer" in text:
        return "sdpa_plus_ssm"
    if has_all(text, "create_ssm_mask", "scaled_dot_product_attention", "ssm_update"):
        return "sdpa_plus_ssm"
    if "scaled_dot_product_attention" in text:
        return "sdpa"
    if has_any(text, "class MambaBlock", "class Mamba2Block", "ssm_update"):
        return "ssm"
    return "custom"


def classify_mixer_signature(stack_topology: str, attention_family: str, mlp_type: str) -> str:
    if stack_topology == "transformer":
        if mlp_type == "moe":
            return f"transformer_{attention_family}_moe"
        if mlp_type == "hybrid_dense_plus_moe":
            return f"transformer_{attention_family}_mixed_moe"
        return f"transformer_{attention_family}_dense"
    if stack_topology == "hybrid_attention_ssm":
        return "hybrid_attention_ssm_blocks"
    if stack_topology == "hybrid_attention_linear":
        return "hybrid_mla_linear_attention"
    if stack_topology == "hybrid_recurrent_attention":
        return "griffin_local_attention_plus_recurrence"
    if stack_topology == "recurrent_rwkv":
        return "rwkv7_recurrent_state_mixing"
    if stack_topology == "ssm_only":
        return "mamba_style_ssm"
    return stack_topology


def classify_attention_projection(text: str) -> str:
    if has_any(text, "c_attn = nn.Linear", "query_key_value = nn.Linear", "Wqkv"):
        return "fused_qkv"
    if "qkv_proj" in text:
        return "fused_qkvkv"
    if has_any(text, "kv_lora_rank", "q_lora_rank"):
        return "lora_factored_q_or_kv"
    if "MultiLinear" in text:
        return "latent_multilinear"
    if has_all(text, "q_proj", "k_proj", "v_proj"):
        return "separate_q_k_v"
    if has_any(text, "ssm_update", "RGLRU"):
        return "state_space"
    return "custom"


def classify_kv_layout(text: str, fields: dict[str, object], attention_family: str) -> str:
    if attention_family in {"mla", "mla_plus_gated_delta"}:
        return "latent_kv_mla"
    if fields.get("multi_query") is True:
        return "mqa"

    n_heads = fields.get("num_attention_heads") or fields.get("n_head") or fields.get("num_heads")
    n_kv = fields.get("num_key_value_heads")

    if isinstance(n_heads, int) and isinstance(n_kv, int):
        if n_kv == 1:
            return "mqa"
        if n_kv < n_heads:
            return "gqa"
        if n_kv == n_heads:
            return "mha"

    if "num_key_value_heads" in fields:
        return "kv_head_parameterized"
    if attention_family == "ssm" or "RGLRU" in text:
        return "state_cache"
    return "mha"


def classify_attention_pattern(text: str, stem: str, stack_topology: str) -> str:
    if stack_topology == "hybrid_recurrent_attention":
        return "alternating_local_attention_and_recurrence"
    if "ChunkedKVCache" in text:
        return "chunked_attention_cache"
    if has_any(text, "sliding_window_pattern", "sliding_window_layers"):
        return "periodic_global_plus_sliding_window"
    if has_all(text, "RotatingKVCache", "window_size="):
        return "mixed_global_plus_sliding_window"
    if "conv_window" in text and "scaled_dot_product_attention" in text:
        return "conv_preconditioned_attention"
    if "ngram" in stem.lower():
        return "ngram_memory_hybrid"
    if stack_topology == "hybrid_attention_ssm":
        return "attention_and_ssm_masks"
    if stack_topology == "hybrid_attention_linear":
        return "full_attention_and_linear_state"
    if stack_topology == "ssm_only":
        return "recurrent_ssm"
    return "full_causal"


def classify_positional_encoding(text: str) -> str:
    if "wpe = nn.Embedding" in text:
        return "learned_absolute"
    if has_all(text, "qk_nope_head_dim", "pe_scores ="):
        return "decoupled_rope_split"
    if has_any(text, "DeepseekV2YarnRotaryEmbedding", "yarn_"):
        return "yarn_rope"
    if "SuScaledRoPE" in text:
        return "su_scaled_rope"
    if has_any(text, "DynamicNTKScalingRoPE", "DynamicNTKAlphaRoPE"):
        return "dynamic_ntk_rope"
    if "rope_local_base_freq" in text:
        return "dual_base_rope"
    if has_any(text, "partial_rotary_factor", "rotary_pct", "rotary_dim", "mrope_section"):
        return "partial_rope"
    if has_any(text, "nn.RoPE", "initialize_rope", "rotary_emb"):
        return "rope"
    return "none"


def classify_qk_norm(text: str) -> str:
    if has_any(text, "v_norm", "RMSNoScale"):
        return "qk_plus_v_norm"
    if has_any(text, "q_norm", "k_norm", "normalize_qk_projections", "use_qk_norm"):
        if has_any(text, "LayerNorm2D", "nn.LayerNorm(head_dim", "self.k_norm = nn.LayerNorm"):
            return "layernorm_qk"
        return "rmsnorm_qk"
    return "none"


def classify_mlp_type(text: str) -> str:
    has_moe = has_any(
        text,
        "SwitchGLU",
        "SwitchMLP",
        "switch_mlp",
        "block_sparse_moe",
        "SparseMoe",
        "shared_expert",
        "shared_experts",
        "n_routed_experts",
        "num_experts",
        "num_local_experts",
    )

    if has_moe:
        if has_any(
            text,
            "first_k_dense_replace",
            "expert_layer_period",
            "expert_layer_offset",
            "decoder_sparse_step",
            "moe_layer_freq",
            "interleave_moe_layer_step",
            "use_moe",
        ):
            return "hybrid_dense_plus_moe"
        return "moe"

    if has_any(text, "class MLP", "class FeedForward", "class MLPBlock"):
        return "dense"
    if has_any(text, "ssm_update", "RGLRU"):
        return "state_space_gated"
    return "custom"


def classify_mlp_activation(text: str) -> str:
    activations = []
    if "swiglu" in text:
        activations.append("swiglu")
    if "SwitchGLU" in text and "swiglu" not in activations:
        activations.append("swiglu")
    if "gegelu" in text:
        activations.append("gegelu")
    if "relu2" in text:
        activations.append("relu2_glu")
    if "nn.gelu_approx" in text:
        activations.append("gelu_approx")
    if re.search(r"\bnn\.gelu\(", text):
        activations.append("gelu")
    if "nn.silu" in text:
        activations.append("silu")

    if not activations:
        return "custom_or_none"

    return "+".join(dict.fromkeys(activations))


def classify_moe_routing(text: str) -> str:
    if not has_any(
        text,
        "num_experts",
        "n_routed_experts",
        "SwitchGLU",
        "SwitchMLP",
        "switch_mlp",
        "block_sparse_moe",
    ):
        return "none"

    if has_any(text, "mx.sigmoid(gates", "mx.sigmoid(self.gate"):
        routing = "sigmoid_topk"
    elif has_any(text, "mx.softmax(gates", "mx.softmax(scores", "top_k_gates = mx.softmax"):
        routing = "softmax_topk"
    else:
        routing = "topk_custom"

    if has_any(text, "shared_expert", "shared_experts"):
        routing += "+shared"
    if "e_score_correction_bias" in text or "expert_bias" in text:
        routing += "+bias_correction"
    if "zero_expert" in text:
        routing += "+zero_expert"

    return routing


def classify_norm_scheme(text: str) -> str:
    has_rms = "RMSNorm" in text or "rms_norm" in text
    has_ln = "LayerNorm" in text or "layer_norm" in text

    if has_rms and has_ln:
        return "mixed_rms_layernorm"
    if has_rms:
        if has_any(text, "ZeroCenteredRMSNorm", "RMSNoScale", "RMSNormGated", "class RMSNorm("):
            return "custom_rmsnorm"
        return "rmsnorm"
    if has_ln:
        return "layernorm"
    return "custom"


def classify_residual_scheme(text: str) -> str:
    if "use_parallel_residual" in text or has_all(
        text, "attn_h = self.self_attn(h", "ff_h = self.mlp(h)"
    ):
        return "parallel_residual"
    if has_any(
        text,
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
        "post_mlp_layernorm",
        "post_mixer_norm",
    ):
        return "sandwich_or_multi_norm"
    if has_any(text, "input_layernorm", "self.norm(x)", "pre_mixer_norm", "ln_1 = nn.LayerNorm"):
        return "prenorm_sequential"
    return "custom"


def classify_embedding_tying(text: str, fields: dict[str, object]) -> str:
    if "tie_word_embeddings" in fields:
        value = fields["tie_word_embeddings"]
        if value is True:
            return "configurable_default_tied"
        if value is False:
            return "configurable_default_untied"
        return "configurable"

    if "as_linear(out)" in text:
        return "tied"
    return "untied_or_custom"


def classify_logits_head(text: str) -> str:
    if has_any(text, "final_logit_softcapping", "logit_softcap"):
        return "linear_or_tied_plus_softcap"
    if "logit_scale" in text:
        return "tied_plus_logit_scale"
    if "lm_head = nn.Linear" in text and "bias=True" in text:
        return "linear_bias"
    if "as_linear(out)" in text:
        return "tied_embedding_head"
    if "lm_head = nn.Linear" in text:
        return "untied_linear"
    return "custom"


def architecture_knobs(text: str, fields: dict[str, object]) -> str:
    knobs = []

    for key in IMPORTANT_KNOBS:
        if key not in fields:
            continue
        value = fields[key]
        if value is None:
            knobs.append(key)
        else:
            knobs.append(f"{key}={value}")

    for needle, label in MARKER_KNOBS:
        if needle in text:
            knobs.append(label)

    return ";".join(knobs)


def special_traits(text: str, stem: str) -> str:
    traits = []

    for needle, label in (
        ("BitLinear", "bitlinear_weights"),
        ("ChunkedKVCache", "chunked_kv_cache"),
        ("DeepseekV2YarnRotaryEmbedding", "yarn_rotary"),
        ("DynamicNTKScalingRoPE", "dynamic_ntk_rope"),
        ("DynamicNTKAlphaRoPE", "dynamic_ntk_alpha_rope"),
        ("GatedDeltaNet", "gated_delta_net"),
        ("RGLRU", "rglru_recurrence"),
        ("SuScaledRoPE", "su_scaled_rope"),
        ("LayerNorm2D", "per_head_layernorm_qk"),
        ("q_norm", "qk_norm"),
        ("v_norm", "value_norm"),
        ("final_logit_softcapping", "final_logit_softcap"),
        ("attn_temperature_tuning", "rope_free_dense_layers"),
        ("e_score_correction_bias", "expert_bias_correction"),
        ("expert_bias", "expert_bias_correction"),
        ("q_lora_rank", "q_lora"),
        ("kv_lora_rank", "kv_lora"),
        ("qk_nope_head_dim", "rope_nope_split"),
        ("num_swa_attention_heads", "swa_head_split"),
        ("conv_window", "conv_on_kv"),
        ("gelu_topk", "gelu_topk"),
        ("Laurel", "laurel"),
        ("AltUp", "altup"),
    ):
        if needle in text:
            traits.append(label)

    traits.extend(MANUAL_SPECIAL_TRAITS.get(stem, []))
    return ";".join(dict.fromkeys(traits))


def build_row(path: Path) -> dict[str, str]:
    text = read(path)
    fields = parse_model_args(path)

    mlp_type = classify_mlp_type(text)
    stack_topology = classify_stack_topology(text, path.stem)
    attention_family = classify_attention_family(text, path.stem)

    row = {
        "model": path.stem,
        "source_file": path.name,
        "modality": classify_modality(text, path.stem),
        "stack_topology": stack_topology,
        "mixer_signature": classify_mixer_signature(stack_topology, attention_family, mlp_type),
        "attention_family": attention_family,
        "attention_projection": classify_attention_projection(text),
        "kv_layout": classify_kv_layout(text, fields, attention_family),
        "attention_pattern": classify_attention_pattern(text, path.stem, stack_topology),
        "positional_encoding": classify_positional_encoding(text),
        "qk_norm": classify_qk_norm(text),
        "mlp_type": mlp_type,
        "mlp_activation": classify_mlp_activation(text),
        "moe_routing": classify_moe_routing(text),
        "norm_scheme": classify_norm_scheme(text),
        "residual_scheme": classify_residual_scheme(text),
        "embedding_tying": classify_embedding_tying(text, fields),
        "logits_head": classify_logits_head(text),
        "architecture_knobs": architecture_knobs(text, fields),
        "special_traits": special_traits(text, path.stem),
    }
    row.update(MANUAL_COLUMN_OVERRIDES.get(path.stem, {}))
    return row


def native_model_files() -> list[Path]:
    files = sorted(MODEL_DIR.glob("*.py"))
    return [path for path in files if path.stem not in NON_NATIVE_EXCLUSIONS]


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    if not rows:
        raise ValueError("No native model rows were generated.")

    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a native-only architecture character matrix for models/."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"CSV output path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    native_files = native_model_files()
    rows = [build_row(path) for path in native_files]

    if len(rows) != EXPECTED_NATIVE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_NATIVE_COUNT} native architectures, found {len(rows)}."
        )

    write_csv(rows, args.output)
    print(f"Wrote {len(rows)} native architecture rows to {args.output}")


if __name__ == "__main__":
    main()
