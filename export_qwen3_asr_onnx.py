from __future__ import annotations

import argparse
import gc
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx import TensorProto, helper
from transformers.cache_utils import DynamicCache


ASR_ONNX_OPSET = 18
ASR_TRACE_PAST_LEN = 8
ASR_TRACE_SEQ_LEN = 1
ASR_TRACE_EMBED_SEQ_LEN = 16
ASR_TRACE_MEL_FRAMES = 2997
ASR_SEED = 20260615
ASR_ATOL = 5.0e-3
ASR_RTOL = 5.0e-3
ASR_LOGITS_ATOL = 1.0e-1
ASR_LOGITS_RTOL = 1.0e-2
ASR_TEXT_ATOL = 1.5e-1
ASR_TEXT_RTOL = 1.0e-2
ASR_LOGITS_TOP_K = 10
ASR_LOGITS_MIN_TOP_K_OVERLAP = 8

COMPONENTS = (
    "audio_encoder",
    "token_embedding",
    "text_core",
)

DTYPE_ALIASES = {
    "float": torch.float32,
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}

COMPONENT_ALIASES = {
    "all": "all",
    "audio": "audio_encoder",
    "audio_encoder": "audio_encoder",
    "audio-encoder": "audio_encoder",
    "encoder": "audio_encoder",
    "embed": "token_embedding",
    "embedding": "token_embedding",
    "token_embedding": "token_embedding",
    "token-embedding": "token_embedding",
    "text": "text_core",
    "core": "text_core",
    "text_core": "text_core",
    "text-core": "text_core",
}


def parse_dtype(value: str) -> torch.dtype:
    key = value.strip().lower()
    try:
        return DTYPE_ALIASES[key]
    except KeyError as exc:
        valid = ", ".join(sorted(DTYPE_ALIASES))
        raise argparse.ArgumentTypeError(f"unsupported dtype {value!r}; choose one of: {valid}") from exc


def parse_components(value: str) -> tuple[str, ...]:
    raw_components = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not raw_components:
        raise argparse.ArgumentTypeError("components cannot be empty")

    normalized = []
    for item in raw_components:
        component = COMPONENT_ALIASES.get(item)
        if component is None:
            valid = ", ".join(COMPONENTS + ("all",))
            raise argparse.ArgumentTypeError(f"unsupported component {item!r}; choose from: {valid}")
        if component == "all":
            return COMPONENTS
        normalized.append(component)

    selected = set(normalized)
    return tuple(component for component in COMPONENTS if component in selected)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export Qwen3-ASR ordinary ASR ONNX files for batch_size=1: "
            "audio_encoder, token_embedding, and shared text_core."
        )
    )
    parser.add_argument(
        "--model-path",
        default=Path("/nfs5/models/Qwen35A"),
        type=Path,
        help="Path to the Qwen3-ASR model directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("./onnx_asr"),
        type=Path,
        help="Root ONNX output directory.",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        type=parse_dtype,
        help=(
            "Model/export dtype: float32, float16, or bfloat16. Aliases fp32/fp16/bf16 are accepted. "
            "float16 is the default because ONNXRuntime does not accept bfloat16 Conv in audio_encoder."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for loading/export. Use auto, cpu, cuda, or cuda:0. auto prefers cuda:0.",
    )
    parser.add_argument(
        "--components",
        default=COMPONENTS,
        type=parse_components,
        help="Comma-separated components to export. Use all, audio_encoder, token_embedding, text_core.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run ONNXRuntime verification after exporting each selected component.",
    )
    parser.add_argument(
        "--audio-trace-mel-frames",
        default=ASR_TRACE_MEL_FRAMES,
        type=int,
        help=(
            "Mel frame count used only for tracing audio_encoder. "
            "The exported audio_encoder keeps this axis dynamic and pads internally to n_window * 2 frames."
        ),
    )
    parser.add_argument(
        "--text-trace-past-len",
        default=ASR_TRACE_PAST_LEN,
        type=int,
        help="Past KV length used when tracing text_core.",
    )
    parser.add_argument(
        "--text-trace-seq-len",
        default=ASR_TRACE_SEQ_LEN,
        type=int,
        help="Current sequence length used when tracing text_core.",
    )
    parser.add_argument(
        "--embed-trace-seq-len",
        default=ASR_TRACE_EMBED_SEQ_LEN,
        type=int,
        help="Sequence length used when tracing token_embedding.",
    )
    parser.add_argument(
        "--external-data",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Save large ONNX weights as external data.",
    )
    parser.add_argument(
        "--merge-external-data",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="When using external data, merge tensors into one .onnx.data file per component.",
    )
    return parser


def print_header(message: str) -> None:
    print(f"\n==> {message}", flush=True)


def resolve_device(device: str, fallback: str | None = None) -> str:
    value = str(device).strip().lower()
    if value == "auto":
        if fallback and fallback != "auto":
            return fallback
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if value == "cuda":
        return "cuda:0"
    return value


def validate_dtype_device(dtype: torch.dtype, device: str, component: str) -> None:
    if dtype in (torch.float16, torch.bfloat16) and device.startswith("cpu"):
        raise RuntimeError(f"{component} {dtype} export requires CUDA. Use --device cuda:0 or --dtype float32.")


def maybe_to_device(module: nn.Module, device: str) -> nn.Module:
    if device.startswith("cpu"):
        return module
    if not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is not available")
    return module.to(device)


def maybe_eval(module: Any) -> None:
    if hasattr(module, "eval"):
        module.eval()


def force_eager_attention(module: nn.Module) -> None:
    """Force every nested Transformers config/module config to use eager attention.

    Passing ``attn_implementation="eager"`` to ``from_pretrained`` is not always
    propagated to nested configs that were already materialized inside composite
    models. ONNX export cannot handle PyTorch SDPA with GQA in this stack, so we
    set the private/public attention implementation fields explicitly.
    """

    seen: set[int] = set()

    def _set_config(config: Any) -> None:
        if config is None or id(config) in seen:
            return
        seen.add(id(config))

        for attr in ("_attn_implementation", "_attn_implementation_internal", "attn_implementation"):
            try:
                setattr(config, attr, "eager")
            except Exception:
                pass

        for child_attr in ("thinker_config", "audio_config", "text_config"):
            _set_config(getattr(config, child_attr, None))

    _set_config(getattr(module, "config", None))
    for submodule in module.modules():
        _set_config(getattr(submodule, "config", None))


def force_greedy_generation_config(module: Any) -> None:
    """ASR export/decode is greedy only; remove sampling-only generation knobs."""

    targets = [module, getattr(module, "generation_config", None), getattr(module, "config", None)]
    for target in targets:
        if target is None:
            continue
        for attr, value in (
            ("do_sample", False),
            ("temperature", None),
            ("top_k", None),
            ("top_p", None),
            ("typical_p", None),
        ):
            if hasattr(target, attr):
                try:
                    setattr(target, attr, value)
                except Exception:
                    pass


@contextmanager
def quiet_generation_config_sampling_warnings():
    """Suppress checkpoint sampling warnings while loading for greedy-only export."""

    from transformers.utils import logging as hf_logging

    previous_verbosity = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        yield
    finally:
        hf_logging.set_verbosity(previous_verbosity)


def get_module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def get_module_dtype(module: nn.Module) -> torch.dtype:
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return torch.float32


def inline_tensor_to_array(tensor: onnx.TensorProto) -> np.ndarray | None:
    if tensor.data_location == TensorProto.EXTERNAL:
        return None
    return onnx.numpy_helper.to_array(tensor)


def constant_tensor_values(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for initializer in model.graph.initializer:
        value = inline_tensor_to_array(initializer)
        if value is not None:
            values[initializer.name] = value

    for node in model.graph.node:
        if node.op_type != "Constant" or not node.output:
            continue
        for attr in node.attribute:
            if attr.name == "value":
                value = inline_tensor_to_array(attr.t)
                if value is not None:
                    values[node.output[0]] = value
                break
    return values


def patch_dynamic_range_reshape(onnx_path: str | Path) -> int:
    """Patch traced Range->Reshape([trace_len, 1]) masks into dynamic Unsqueeze(axis=1)."""
    model = onnx.load(str(onnx_path), load_external_data=False)
    if any(node.name.endswith("_range_unsqueeze_axis1") for node in model.graph.node):
        print("dynamic range reshape patch skipped: already patched")
        return 0

    constants = constant_tensor_values(model)
    patched = 0
    new_nodes = []
    replaced_outputs: dict[str, str] = {}

    for node in model.graph.node:
        is_range_reshape = node.op_type == "Reshape" and len(node.input) >= 2 and "Range" in node.input[0]
        shape_value = constants.get(node.input[1]) if len(node.input) >= 2 else None
        if is_range_reshape and shape_value is not None and shape_value.shape == (2,) and int(shape_value[1]) == 1:
            node_prefix = node.name or f"/model/RangeReshapePatch_{patched}"
            axes_name = f"{node_prefix}_unsqueeze_axis1_const_output_0"
            unsqueeze_out = f"{node_prefix}_range_unsqueeze_axis1_output_0"
            new_nodes.append(
                helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=[axes_name],
                    name=f"{node_prefix}_unsqueeze_axis1_const",
                    value=helper.make_tensor("value", TensorProto.INT64, [1], [1]),
                )
            )
            new_nodes.append(
                helper.make_node(
                    "Unsqueeze",
                    inputs=[node.input[0], axes_name],
                    outputs=[unsqueeze_out],
                    name=f"{node_prefix}_range_unsqueeze_axis1",
                )
            )
            replaced_outputs[node.output[0]] = unsqueeze_out
            patched += 1
            continue

        for input_index, input_name in enumerate(node.input):
            if input_name in replaced_outputs:
                node.input[input_index] = replaced_outputs[input_name]
        new_nodes.append(node)

    if patched:
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)
        onnx.save(model, str(onnx_path))
        print(f"patched dynamic range reshape nodes: {patched}")
    else:
        print("dynamic range reshape patch skipped: no Range->Reshape([N,1]) node found")
        return patched


def patch_cache_position_dynamic_reshape(onnx_path: str | Path) -> int:
    """Patch traced cache_position.reshape([1, 1]) into dynamic Unsqueeze(axis=1)."""
    model = onnx.load(str(onnx_path), load_external_data=False)
    if any(node.name.endswith("_cache_position_unsqueeze_axis1") for node in model.graph.node):
        print("text_core cache_position reshape patch skipped: already patched")
        return 0

    constants = constant_tensor_values(model)
    patched = 0
    new_nodes = []
    replaced_outputs: dict[str, str] = {}

    for node in model.graph.node:
        shape_value = constants.get(node.input[1]) if node.op_type == "Reshape" and len(node.input) >= 2 else None
        if (
            node.op_type == "Reshape"
            and len(node.input) >= 2
            and node.input[0] == "cache_position"
            and shape_value is not None
            and shape_value.shape == (2,)
            and int(shape_value[0]) == 1
            and int(shape_value[1]) == 1
        ):
            node_prefix = node.name or f"/model/CachePositionReshapePatch_{patched}"
            axes_name = f"{node_prefix}_cache_position_unsqueeze_axis1_const_output_0"
            unsqueeze_out = f"{node_prefix}_cache_position_unsqueeze_axis1_output_0"
            new_nodes.append(
                helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=[axes_name],
                    name=f"{node_prefix}_cache_position_unsqueeze_axis1_const",
                    value=helper.make_tensor("value", TensorProto.INT64, [1], [1]),
                )
            )
            new_nodes.append(
                helper.make_node(
                    "Unsqueeze",
                    inputs=[node.input[0], axes_name],
                    outputs=[unsqueeze_out],
                    name=f"{node_prefix}_cache_position_unsqueeze_axis1",
                )
            )
            replaced_outputs[node.output[0]] = unsqueeze_out
            patched += 1
            continue

        for input_index, input_name in enumerate(node.input):
            if input_name in replaced_outputs:
                node.input[input_index] = replaced_outputs[input_name]
        new_nodes.append(node)

    if patched:
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)
        onnx.save(model, str(onnx_path))
        print(f"patched text_core cache_position reshape nodes: {patched}")
    else:
        print("text_core cache_position reshape patch skipped: no cache_position Reshape([1,1]) node found")
    return patched


def save_onnx_with_single_external_data(
    staged_onnx_path: Path,
    output_path: Path,
    data_file_name: str,
) -> None:
    model = onnx.load(str(staged_onnx_path), load_external_data=True)
    data_path = output_path.with_name(data_file_name)
    if data_path.exists():
        data_path.unlink()
    onnx.save_model(
        model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_file_name,
        size_threshold=0,
        convert_attribute=False,
    )


def export_onnx(
    wrapper: nn.Module,
    dummy_inputs: tuple[torch.Tensor, ...],
    output_path: Path,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]] | None,
    external_data: bool,
    merge_external_data: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if external_data and merge_external_data:
        with tempfile.TemporaryDirectory(prefix=f"{output_path.stem}_onnx_") as tmp_dir:
            staged_path = Path(tmp_dir) / output_path.name
            torch.onnx.export(
                wrapper,
                dummy_inputs,
                str(staged_path),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=ASR_ONNX_OPSET,
                do_constant_folding=False,
                dynamo=False,
                external_data=True,
            )
            save_onnx_with_single_external_data(staged_path, output_path, f"{output_path.name}.data")
    else:
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=ASR_ONNX_OPSET,
            do_constant_folding=False,
            dynamo=False,
            external_data=external_data,
        )


def _flatten_cache(cache: Any, num_layers: int) -> tuple[torch.Tensor, ...]:
    if cache is None:
        raise ValueError("past_key_values is None; call the model with use_cache=True.")
    if hasattr(cache, "layers"):
        return tuple(tensor for i in range(num_layers) for tensor in (cache.layers[i].keys, cache.layers[i].values))
    return tuple(tensor for layer in cache for tensor in layer)


def _legacy_cache_from_flat(past_kv_flat: tuple[torch.Tensor, ...], num_layers: int) -> DynamicCache:
    if len(past_kv_flat) != 2 * num_layers:
        raise ValueError(f"expected {2 * num_layers} KV tensors, got {len(past_kv_flat)}")
    legacy_cache = tuple((past_kv_flat[2 * i], past_kv_flat[2 * i + 1]) for i in range(num_layers))
    return DynamicCache.from_legacy_cache(legacy_cache)


class ASRAudioEncoder(nn.Module):
    """ONNX-friendly batch-1 dynamic-length wrapper around Qwen3ASRAudioEncoder.

    The upstream audio tower uses ``pad_sequence`` after Python-side chunk splitting.
    PyTorch does not export ``aten::pad_sequence`` to ONNX, so this wrapper expects
    arbitrary time length. It pads internally to a multiple of ``n_window * 2``
    and uses ``feature_lens`` to slice the real output length.
    """

    def __init__(self, audio_tower: nn.Module) -> None:
        super().__init__()
        self.audio_tower = audio_tower
        self.chunk_size = int(audio_tower.n_window) * 2

    @staticmethod
    def _feat_extract_output_length(input_length: torch.Tensor) -> torch.Tensor:
        input_lengths_leave = input_length % 100
        feat_lengths = torch.div(input_lengths_leave - 1, 2, rounding_mode="floor") + 1
        output_lengths = (
            torch.div(torch.div(feat_lengths - 1, 2, rounding_mode="floor") + 1 - 1, 2, rounding_mode="floor")
            + 1
            + torch.div(input_length, 100, rounding_mode="floor") * 13
        )
        return output_lengths

    def forward(self, input_features: torch.Tensor, feature_lens: torch.Tensor) -> torch.Tensor:
        x = input_features[0]
        time_len = x.shape[1]
        pad_frames = (self.chunk_size - (time_len % self.chunk_size)) % self.chunk_size
        x = torch.nn.functional.pad(x, (0, pad_frames), mode="constant", value=0.0)

        padded_feature = (
            x.transpose(0, 1)
            .reshape(-1, self.chunk_size, int(self.audio_tower.num_mel_bins))
            .transpose(1, 2)
            .unsqueeze(1)
        )

        padded_embed = torch.nn.functional.gelu(self.audio_tower.conv2d1(padded_feature))
        padded_embed = torch.nn.functional.gelu(self.audio_tower.conv2d2(padded_embed))
        padded_embed = torch.nn.functional.gelu(self.audio_tower.conv2d3(padded_embed))

        bsz, channels, freq, time = padded_embed.size()
        padded_embed = self.audio_tower.conv_out(
            padded_embed.permute(0, 3, 1, 2).contiguous().view(bsz, time, channels * freq)
        )
        positional_embedding = (
            self.audio_tower.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
            .unsqueeze(0)
            .to(padded_embed.dtype)
        )
        padded_embed = padded_embed + positional_embedding

        hidden_states = padded_embed.reshape(-1, padded_embed.shape[-1])
        output_len = self._feat_extract_output_length(feature_lens[0]).to(torch.long)
        hidden_states = hidden_states[:output_len]
        cu_seqlens = torch.stack(
            [
                torch.zeros((), dtype=torch.int32, device=hidden_states.device),
                output_len.to(torch.int32),
            ],
            dim=0,
        )

        for encoder_layer in self.audio_tower.layers:
            layer_outputs = encoder_layer(hidden_states, cu_seqlens)
            hidden_states = layer_outputs[0]

        hidden_states = self.audio_tower.ln_post(hidden_states)
        hidden_states = self.audio_tower.proj1(hidden_states)
        hidden_states = self.audio_tower.act(hidden_states)
        hidden_states = self.audio_tower.proj2(hidden_states)
        return hidden_states


class ASRTokenEmbedding(nn.Module):
    """Token id to hidden embedding wrapper."""

    def __init__(self, embedding: nn.Module) -> None:
        super().__init__()
        self.embedding = embedding

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)


class ASRTextCore(nn.Module):
    """Shared ASR transformer step used by both prompt prefill and token decode."""

    def __init__(self, thinker: nn.Module) -> None:
        super().__init__()
        self.text_model = thinker.model
        self.lm_head = thinker.lm_head
        self.num_layers = int(self.text_model.config.num_hidden_layers)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        cache_position: torch.Tensor,
        *past_kv_flat: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        past_key_values = _legacy_cache_from_flat(past_kv_flat, self.num_layers)
        out = self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )
        hidden = out.last_hidden_state[:, -1:, :]
        logits = self.lm_head(hidden)
        valid_cache_len = cache_position[-1] + 1
        new_kv = tuple(tensor[:, :, :valid_cache_len, :] for tensor in _flatten_cache(out.past_key_values, self.num_layers))
        return (logits, hidden) + new_kv


def load_qwen3_asr_model(model_path: Path, dtype: torch.dtype, device: str):
    from qwen_asr import Qwen3ASRModel

    print_header(f"Loading Qwen3-ASR model from {model_path}")
    with quiet_generation_config_sampling_warnings():
        model = Qwen3ASRModel.from_pretrained(
            str(model_path),
            dtype=dtype,
            device_map=device if not device.startswith("cpu") else None,
            attn_implementation="eager",
            max_inference_batch_size=1,
            max_new_tokens=2048,
        )
    qwen_model = model.model
    if device.startswith("cpu"):
        qwen_model = qwen_model.to(device)
    force_greedy_generation_config(model)
    force_greedy_generation_config(qwen_model)
    force_eager_attention(qwen_model)
    maybe_eval(qwen_model)
    maybe_eval(getattr(qwen_model, "thinker", None))
    return qwen_model


def _prepare_audio_encoder_inputs(
    audio_tower: nn.Module,
    mel_frames: int,
    batch_size: int = 1,
    seed: int = ASR_SEED,
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_size != 1:
        raise ValueError("Qwen3-ASR audio encoder export currently supports batch_size=1 only.")
    device = get_module_device(audio_tower)
    dtype = get_module_dtype(audio_tower)
    num_mel_bins = int(audio_tower.config.num_mel_bins)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    input_features = torch.randn(
        batch_size,
        num_mel_bins,
        mel_frames,
        dtype=torch.float32,
        generator=generator,
    ).to(device=device, dtype=dtype)
    feature_lens = torch.tensor([mel_frames], dtype=torch.long, device=device)
    return input_features, feature_lens


def _prepare_token_embedding_inputs(
    embedding: nn.Module,
    seq_len: int = ASR_TRACE_EMBED_SEQ_LEN,
    batch_size: int = 1,
    seed: int = ASR_SEED,
) -> tuple[torch.Tensor]:
    device = get_module_device(embedding)
    vocab_size = int(embedding.num_embeddings)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        dtype=torch.long,
        generator=generator,
    ).to(device=device)
    return (input_ids,)


def _prepare_text_core_inputs(
    thinker: nn.Module,
    past_len: int = ASR_TRACE_PAST_LEN,
    seq_len: int = ASR_TRACE_SEQ_LEN,
    batch_size: int = 1,
    seed: int = ASR_SEED,
) -> tuple[torch.Tensor, ...]:
    text_model = thinker.model
    device = get_module_device(text_model)
    dtype = get_module_dtype(text_model)
    config = text_model.config
    hidden_size = int(config.hidden_size)
    num_layers = int(config.num_hidden_layers)
    num_kv_heads = int(getattr(config, "num_key_value_heads", config.num_attention_heads))
    head_dim = int(getattr(config, "head_dim", hidden_size // int(config.num_attention_heads)))

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    inputs_embeds = torch.randn(
        batch_size,
        seq_len,
        hidden_size,
        dtype=torch.float32,
        generator=generator,
    ).to(device=device, dtype=dtype)
    attention_mask = torch.ones(batch_size, past_len + seq_len, dtype=torch.long, device=device)
    cache_position = torch.arange(past_len, past_len + seq_len, dtype=torch.long, device=device)
    past_kv = []
    for _ in range(num_layers):
        key = torch.randn(
            batch_size,
            num_kv_heads,
            past_len,
            head_dim,
            dtype=torch.float32,
            generator=generator,
        ).to(device=device, dtype=dtype)
        value = torch.randn(
            batch_size,
            num_kv_heads,
            past_len,
            head_dim,
            dtype=torch.float32,
            generator=generator,
        ).to(device=device, dtype=dtype)
        past_kv.extend([key, value])
    return (inputs_embeds, attention_mask, cache_position, *past_kv)


def _text_core_io_names(thinker: nn.Module) -> tuple[list[str], list[str]]:
    num_layers = int(thinker.model.config.num_hidden_layers)
    input_names = ["inputs_embeds", "attention_mask", "cache_position"]
    output_names = ["logits", "last_hidden"]
    for i in range(num_layers):
        input_names += [f"past_key_{i}", f"past_value_{i}"]
        output_names += [f"new_past_key_{i}", f"new_past_value_{i}"]
    return input_names, output_names


def _text_core_dynamic_axes(thinker: nn.Module) -> dict[str, dict[int, str]]:
    num_layers = int(thinker.model.config.num_hidden_layers)
    dynamic_axes = {
        "inputs_embeds": {1: "seq_len"},
        "attention_mask": {1: "full_len"},
        "cache_position": {0: "seq_len"},
    }
    for i in range(num_layers):
        dynamic_axes[f"past_key_{i}"] = {2: "past_len"}
        dynamic_axes[f"past_value_{i}"] = {2: "past_len"}
        dynamic_axes[f"new_past_key_{i}"] = {2: "new_len"}
        dynamic_axes[f"new_past_value_{i}"] = {2: "new_len"}
    return dynamic_axes


def export_audio_encoder(
    thinker: nn.Module,
    output_dir: str | Path,
    trace_mel_frames: int = ASR_TRACE_MEL_FRAMES,
    external_data: bool = True,
    merge_external_data: bool = True,
) -> Path:
    output_path = Path(output_dir) / "audio_encoder.onnx"
    wrapper = ASRAudioEncoder(thinker.audio_tower).eval()
    dummy_inputs = _prepare_audio_encoder_inputs(thinker.audio_tower, mel_frames=trace_mel_frames)
    export_onnx(
        wrapper=wrapper,
        dummy_inputs=dummy_inputs,
        output_path=output_path,
        input_names=["input_features", "feature_lens"],
        output_names=["audio_features"],
        dynamic_axes={
            "input_features": {2: "mel_frames"},
            "audio_features": {0: "audio_seq_len"},
        },
        external_data=external_data,
        merge_external_data=merge_external_data,
    )
    return output_path


def export_token_embedding(
    thinker: nn.Module,
    output_dir: str | Path,
    trace_seq_len: int = ASR_TRACE_EMBED_SEQ_LEN,
    external_data: bool = True,
    merge_external_data: bool = True,
) -> Path:
    output_path = Path(output_dir) / "token_embedding.onnx"
    embedding = thinker.model.embed_tokens
    wrapper = ASRTokenEmbedding(embedding).eval()
    dummy_inputs = _prepare_token_embedding_inputs(embedding, seq_len=trace_seq_len)
    export_onnx(
        wrapper=wrapper,
        dummy_inputs=dummy_inputs,
        output_path=output_path,
        input_names=["input_ids"],
        output_names=["inputs_embeds"],
        dynamic_axes={"input_ids": {1: "seq_len"}, "inputs_embeds": {1: "seq_len"}},
        external_data=external_data,
        merge_external_data=merge_external_data,
    )
    return output_path


def export_text_core(
    thinker: nn.Module,
    output_dir: str | Path,
    trace_past_len: int = ASR_TRACE_PAST_LEN,
    trace_seq_len: int = ASR_TRACE_SEQ_LEN,
    external_data: bool = True,
    merge_external_data: bool = True,
) -> Path:
    output_path = Path(output_dir) / "asr_text_core.onnx"
    wrapper = ASRTextCore(thinker).eval()
    dummy_inputs = _prepare_text_core_inputs(thinker, past_len=trace_past_len, seq_len=trace_seq_len)
    input_names, output_names = _text_core_io_names(thinker)
    dynamic_axes = _text_core_dynamic_axes(thinker)
    export_onnx(
        wrapper=wrapper,
        dummy_inputs=dummy_inputs,
        output_path=output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        external_data=external_data,
        merge_external_data=merge_external_data,
    )
    patch_dynamic_range_reshape(output_path)
    patch_cache_position_dynamic_reshape(output_path)
    return output_path


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    return tensor.detach().cpu().numpy()


def _to_numpy_feed(input_names: list[str], tensors: tuple[torch.Tensor, ...]) -> dict[str, np.ndarray]:
    feed = {}
    for name, tensor in zip(input_names, tensors):
        array = _as_numpy(tensor)
        if tensor.dtype == torch.long:
            array = array.astype(np.int64, copy=False)
        feed[name] = array
    return feed


def _default_providers() -> list[str]:
    import onnxruntime as ort

    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider"]
    return ["CPUExecutionProvider"]


def _compare_outputs(
    names: list[str],
    onnx_outputs: list[np.ndarray],
    pytorch_outputs: list[np.ndarray],
    atol: float = ASR_ATOL,
    rtol: float = ASR_RTOL,
) -> None:
    max_abs_diff = 0.0
    worst_name = None
    for name, onnx_output, pytorch_output in zip(names, onnx_outputs, pytorch_outputs):
        if name == "logits":
            output_atol = ASR_LOGITS_ATOL
            output_rtol = ASR_LOGITS_RTOL
        elif name == "last_hidden" or name.startswith("new_past_"):
            output_atol = ASR_TEXT_ATOL
            output_rtol = ASR_TEXT_RTOL
        else:
            output_atol = atol
            output_rtol = rtol
        if onnx_output.shape != pytorch_output.shape:
            raise AssertionError(f"{name} shape mismatch: onnx={onnx_output.shape}, pytorch={pytorch_output.shape}")
        diff = onnx_output.astype(np.float64) - pytorch_output.astype(np.float64)
        current_max = float(np.abs(diff).max()) if diff.size else 0.0
        if current_max >= max_abs_diff:
            max_abs_diff = current_max
            worst_name = name
        if name in ("audio_features", "inputs_embeds", "logits", "last_hidden"):
            print(
                f"{name} compare stats: shape={onnx_output.shape}, "
                f"max_abs_diff={current_max:.8f}, mean_abs_diff={np.abs(diff).mean():.8f}, "
                f"atol={output_atol:.1e}, rtol={output_rtol:.1e}"
            )
        if not np.allclose(onnx_output, pytorch_output, atol=output_atol, rtol=output_rtol):
            raise AssertionError(
                f"ONNX output mismatch: {name} "
                f"(max_abs_diff={current_max:.8f}, atol={output_atol:.1e}, rtol={output_rtol:.1e})"
            )
    print(f"compare stats: outputs={len(names)}, max_abs_diff={max_abs_diff:.8f}, worst={worst_name}")


def _top_k_indices(values: np.ndarray, k: int) -> np.ndarray:
    flat = np.asarray(values).reshape(-1)
    top_k = min(int(k), flat.size)
    if top_k <= 0:
        return np.empty((0,), dtype=np.int64)
    indices = np.argpartition(-flat, top_k - 1)[:top_k]
    return indices[np.argsort(-flat[indices])]


def _compare_text_core_outputs(
    names: list[str],
    onnx_outputs: list[np.ndarray],
    pytorch_outputs: list[np.ndarray],
) -> None:
    max_abs_diff = 0.0
    worst_name = None
    logits_checked = False

    for name, onnx_output, pytorch_output in zip(names, onnx_outputs, pytorch_outputs):
        if onnx_output.shape != pytorch_output.shape:
            raise AssertionError(f"{name} shape mismatch: onnx={onnx_output.shape}, pytorch={pytorch_output.shape}")
        if not np.isfinite(onnx_output).all():
            raise AssertionError(f"ONNX output contains non-finite values: {name}")

        diff = onnx_output.astype(np.float64) - pytorch_output.astype(np.float64)
        current_max = float(np.abs(diff).max()) if diff.size else 0.0
        current_mean = float(np.abs(diff).mean()) if diff.size else 0.0
        if current_max >= max_abs_diff:
            max_abs_diff = current_max
            worst_name = name

        if name in ("logits", "last_hidden"):
            print(
                f"{name} compare stats: shape={onnx_output.shape}, "
                f"max_abs_diff={current_max:.8f}, mean_abs_diff={current_mean:.8f}"
            )

        if name != "logits":
            continue

        logits_checked = True
        if np.allclose(onnx_output, pytorch_output, atol=ASR_LOGITS_ATOL, rtol=ASR_LOGITS_RTOL):
            print(
                f"logits allclose ok: atol={ASR_LOGITS_ATOL:.1e}, "
                f"rtol={ASR_LOGITS_RTOL:.1e}"
            )
            continue

        onnx_top_k = _top_k_indices(onnx_output, ASR_LOGITS_TOP_K)
        pytorch_top_k = _top_k_indices(pytorch_output, ASR_LOGITS_TOP_K)
        overlap = len(set(onnx_top_k.tolist()) & set(pytorch_top_k.tolist()))
        onnx_argmax = int(onnx_top_k[0]) if onnx_top_k.size else -1
        pytorch_argmax = int(pytorch_top_k[0]) if pytorch_top_k.size else -1
        print(
            "logits semantic compare: "
            f"argmax onnx={onnx_argmax}, pytorch={pytorch_argmax}, "
            f"top{ASR_LOGITS_TOP_K}_overlap={overlap}/{ASR_LOGITS_TOP_K}, "
            f"onnx_top5={onnx_top_k[:5].tolist()}, pytorch_top5={pytorch_top_k[:5].tolist()}"
        )
        if onnx_argmax != pytorch_argmax or overlap < ASR_LOGITS_MIN_TOP_K_OVERLAP:
            raise AssertionError(
                "ONNX logits ranking mismatch: "
                f"argmax onnx={onnx_argmax}, pytorch={pytorch_argmax}, "
                f"top{ASR_LOGITS_TOP_K}_overlap={overlap}/{ASR_LOGITS_TOP_K}"
            )

    if not logits_checked:
        raise AssertionError("text_core verification did not receive logits output")
    print(f"text_core compare stats: outputs={len(names)}, max_abs_diff={max_abs_diff:.8f}, worst={worst_name}")


def verify_audio_encoder(
    thinker: nn.Module,
    onnx_path: Path | str,
    trace_mel_frames: int = ASR_TRACE_MEL_FRAMES,
    providers: list[str] | None = None,
) -> None:
    import onnxruntime as ort

    wrapper = ASRAudioEncoder(thinker.audio_tower).eval()
    inputs = _prepare_audio_encoder_inputs(thinker.audio_tower, mel_frames=trace_mel_frames)
    with torch.inference_mode():
        pytorch_outputs = [wrapper(*inputs)]
    session = ort.InferenceSession(str(onnx_path), providers=providers or _default_providers())
    onnx_outputs = session.run(["audio_features"], _to_numpy_feed(["input_features", "feature_lens"], inputs))
    _compare_outputs(["audio_features"], onnx_outputs, [_as_numpy(output) for output in pytorch_outputs])
    print(f"onnx audio_encoder ok: mel_frames={trace_mel_frames}")


def verify_token_embedding(
    thinker: nn.Module,
    onnx_path: Path | str,
    trace_seq_len: int = ASR_TRACE_EMBED_SEQ_LEN,
    providers: list[str] | None = None,
) -> None:
    import onnxruntime as ort

    embedding = thinker.model.embed_tokens
    wrapper = ASRTokenEmbedding(embedding).eval()
    inputs = _prepare_token_embedding_inputs(embedding, seq_len=trace_seq_len)
    with torch.inference_mode():
        pytorch_outputs = [wrapper(*inputs)]
    session = ort.InferenceSession(str(onnx_path), providers=providers or _default_providers())
    onnx_outputs = session.run(["inputs_embeds"], _to_numpy_feed(["input_ids"], inputs))
    _compare_outputs(["inputs_embeds"], onnx_outputs, [_as_numpy(output) for output in pytorch_outputs])
    print(f"onnx token_embedding ok: seq_len={trace_seq_len}")


def verify_text_core(
    thinker: nn.Module,
    onnx_path: Path | str,
    past_len: int = ASR_TRACE_PAST_LEN,
    seq_len: int = ASR_TRACE_SEQ_LEN,
    providers: list[str] | None = None,
) -> None:
    import onnxruntime as ort

    wrapper = ASRTextCore(thinker).eval()
    inputs = _prepare_text_core_inputs(thinker, past_len=past_len, seq_len=seq_len)
    input_names, output_names = _text_core_io_names(thinker)
    with torch.inference_mode():
        pytorch_outputs = wrapper(*inputs)
    session = ort.InferenceSession(str(onnx_path), providers=providers or _default_providers())
    onnx_outputs = session.run(output_names, _to_numpy_feed(input_names, inputs))
    _compare_text_core_outputs(output_names, onnx_outputs, [_as_numpy(output) for output in pytorch_outputs])
    print(f"onnx text_core ok: past_len={past_len}, seq_len={seq_len}")


def export_selected(args: argparse.Namespace) -> dict[str, Path]:
    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    selected = tuple(args.components)

    print("Qwen3-ASR ONNX export config:")
    print(f"  model_path: {args.model_path}")
    print(f"  output_dir: {output_root}")
    print(f"  dtype: {args.dtype}")
    print(f"  device: {device}")
    print(f"  components: {', '.join(selected)}")
    print(f"  verify: {args.verify}")
    print(f"  audio_trace_mel_frames: {args.audio_trace_mel_frames}")
    print(f"  text_trace_past_len: {args.text_trace_past_len}")
    print(f"  text_trace_seq_len: {args.text_trace_seq_len}")
    print(f"  external_data: {args.external_data}")
    print(f"  merge_external_data: {args.merge_external_data}")

    validate_dtype_device(args.dtype, device, "Qwen3-ASR model")
    qwen_model = load_qwen3_asr_model(args.model_path, args.dtype, device)
    thinker = qwen_model.thinker
    exported: dict[str, Path] = {}

    if "audio_encoder" in selected:
        audio_dir = output_root / "audio_encoder"
        print_header("Exporting audio_encoder")
        audio_path = export_audio_encoder(
            thinker=thinker,
            output_dir=audio_dir,
            trace_mel_frames=args.audio_trace_mel_frames,
            external_data=args.external_data,
            merge_external_data=args.merge_external_data,
        )
        exported["audio_encoder"] = audio_path
        if args.verify:
            print_header("Verifying audio_encoder")
            verify_audio_encoder(thinker, audio_path, trace_mel_frames=args.audio_trace_mel_frames)

    if "token_embedding" in selected:
        embed_dir = output_root / "token_embedding"
        print_header("Exporting token_embedding")
        embed_path = export_token_embedding(
            thinker=thinker,
            output_dir=embed_dir,
            trace_seq_len=args.embed_trace_seq_len,
            external_data=args.external_data,
            merge_external_data=args.merge_external_data,
        )
        exported["token_embedding"] = embed_path
        if args.verify:
            print_header("Verifying token_embedding")
            verify_token_embedding(thinker, embed_path, trace_seq_len=args.embed_trace_seq_len)

    if "text_core" in selected:
        text_dir = output_root / "text_core"
        print_header("Exporting text_core")
        text_path = export_text_core(
            thinker=thinker,
            output_dir=text_dir,
            trace_past_len=args.text_trace_past_len,
            trace_seq_len=args.text_trace_seq_len,
            external_data=args.external_data,
            merge_external_data=args.merge_external_data,
        )
        exported["text_core"] = text_path
        if args.verify:
            print_header("Verifying text_core decode-shaped input")
            verify_text_core(
                thinker,
                text_path,
                past_len=args.text_trace_past_len,
                seq_len=args.text_trace_seq_len,
            )
            print_header("Verifying text_core zero-past prefill-shaped input")
            verify_text_core(
                thinker,
                text_path,
                past_len=0,
                seq_len=max(args.text_trace_seq_len, 8),
            )

    del qwen_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return exported


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    exported = export_selected(args)

    print("\nExport complete:")
    for component, path in exported.items():
        print(f"  {component}: {path}")


if __name__ == "__main__":
    main()
