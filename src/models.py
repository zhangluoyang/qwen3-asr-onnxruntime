from __future__ import annotations

"""Qwen3-ASR 的 ONNXRuntime 推理封装。

这个文件只负责运行已经导出的三个 ONNX：
1. audio_encoder.onnx：把 mel 特征编码成 audio hidden states。
2. token_embedding.onnx：把文本 token id 转成 embedding。
3. asr_text_core.onnx：共享的文本 Transformer，prefill/decode 都用它。

当前实现有两条路径：
- transcribe(): 离线 ASR，一次性处理音频或音频 chunk。
- streaming_transcribe(): 伪流式 ASR，按音频块触发重算，不复用 audio encoder cache。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from src.OnnxSessionRunner import ORT_INPUT_DTYPES, OnnxSessionRunner, is_ortvalue


DEFAULT_MODEL_PATH = Path("/nfs5/models/Qwen35A")
DEFAULT_ONNX_DIR = Path("./onnx_asr")
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_EOS_TOKEN_IDS = (151645, 151643)
STREAM_SAMPLE_RATE = 16000


@dataclass
class ASROnnxTranscription:
    """一次离线 ASR 的输出。raw_text 是未清洗的模型原始文本。"""

    language: str
    text: str
    raw_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ASRTextCoreOutputs:
    """text_core.onnx 的输出。

    logits 会回到 CPU 做 greedy argmax；past_key_values 在 IO binding 时通常是 OrtValue，
    会直接作为下一步 decode 的输入继续绑定，避免 KV cache 每步来回拷贝。
    """

    logits: np.ndarray
    past_key_values: tuple[tuple[Any, Any], ...]
    raw_outputs: dict[str, Any] = field(default_factory=dict)

    @property
    def next_token_logits(self) -> np.ndarray:
        return self.logits[:, -1, :]


@dataclass
class ASROnnxStreamingState:
    """伪流式 ASR 的可变状态。

    buffer 保存还没凑够一个流式 chunk 的 PCM；audio_accum 保存当前要重跑 ASR 的音频窗口。
    注意这里没有 audio encoder cache，所以每次触发 decode 都会重新跑 audio_accum。
    """

    unfixed_chunk_num: int
    unfixed_token_num: int
    chunk_size_sec: float
    chunk_size_samples: int
    max_audio_history_samples: Optional[int]
    chunk_id: int
    total_audio_samples: int
    total_trimmed_samples: int
    last_trimmed_samples: int
    trim_events: int
    buffer: np.ndarray
    audio_accum: np.ndarray
    audio_storage: np.ndarray
    audio_storage_start: int
    audio_accum_samples: int
    last_appended_start_samples: int
    last_appended_end_samples: int
    context: str
    force_language: Optional[str]
    hotwords: list[str]
    language: str = ""
    text: str = ""
    confirmed_delta_text: str = ""
    confirmed_text: str = ""
    pending_text: str = ""
    delta_raw_text: str = ""
    _raw_decoded: str = ""
    _committed_raw_decoded: str = ""
    _emitted_text: str = ""
    _pending_text: str = ""
    _emitted_raw_decoded: str = ""
    _raw_decoded_token_ids: list[int] = field(default_factory=list)
    last_raw_generated: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def raw_text(self) -> str:
        return self._raw_decoded

    @property
    def committed_raw_text(self) -> str:
        return self._committed_raw_decoded


class Qwen3ASROnnxModel:
    """batch=1 的 Qwen3-ASR ONNXRuntime 推理器。

    设计目标：
    - 普通 ASR 使用三个 ONNX 串起来完成。
    - 文本 decode 阶段启用 IO binding，KV cache 尽量留在设备上。
    - 只做 greedy decode，不采样。
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        onnx_dir: str | Path = DEFAULT_ONNX_DIR,
        providers: Optional[list[str]] = None,
        audio_providers: Optional[list[str]] = None,
        embedding_providers: Optional[list[str]] = None,
        text_providers: Optional[list[str]] = None,
        use_iobinding: Optional[bool] = True,
        audio_use_iobinding: Optional[bool] = None,
        embedding_use_iobinding: Optional[bool] = None,
        text_use_iobinding: Optional[bool] = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> None:
        self.model_path = Path(model_path)
        self.onnx_dir = Path(onnx_dir)
        self.providers = list(providers) if providers is not None else self._default_providers()
        self.audio_providers = list(audio_providers) if audio_providers is not None else list(self.providers)
        # token_embedding 只是查表，decode 时每步只产生 [1, 1, hidden] 的小张量。
        # 默认放 CPU，节省 GPU 显存；真正重的 audio_encoder/text_core 仍按主 provider 跑。
        self.embedding_providers = (
            list(embedding_providers) if embedding_providers is not None else ["CPUExecutionProvider"]
        )
        self.text_providers = list(text_providers) if text_providers is not None else list(self.providers)
        self.audio_use_iobinding = use_iobinding if audio_use_iobinding is None else audio_use_iobinding
        # CPU embedding session 不需要 IO binding；如果想强行放 CUDA，可以显式传 embedding_* 参数。
        self.embedding_use_iobinding = False if embedding_use_iobinding is None else embedding_use_iobinding
        self.text_use_iobinding = use_iobinding if text_use_iobinding is None else text_use_iobinding
        self.max_new_tokens = int(max_new_tokens)

        # 三段 ONNX 的固定目录约定，和 export_qwen3_asr_onnx.py 的输出保持一致。
        self.audio_encoder_onnx_path = self.onnx_dir / "audio_encoder" / "audio_encoder.onnx"
        self.token_embedding_onnx_path = self.onnx_dir / "token_embedding" / "token_embedding.onnx"
        self.text_core_onnx_path = self.onnx_dir / "text_core" / "asr_text_core.onnx"

        # processor 仍然复用原始 qwen-asr：负责 chat template、tokenizer 和 mel feature extraction。
        self.processor = self._load_processor()
        self.audio_token_id = int(self.processor.tokenizer.convert_tokens_to_ids(self.processor.audio_token))
        self.eos_token_ids = tuple(int(item) for item in DEFAULT_EOS_TOKEN_IDS)

        # session 延迟初始化；第一次访问对应 runner 时才真正加载 ONNX。
        self._audio_encoder_runner: OnnxSessionRunner | None = None
        self._token_embedding_runner: OnnxSessionRunner | None = None
        self._text_core_runner: OnnxSessionRunner | None = None
        self._text_core_output_names: list[str] | None = None
        self._text_core_kv_output_names: tuple[tuple[str, str], ...] | None = None
        self._empty_past_key_values_cache: tuple[tuple[np.ndarray, np.ndarray], ...] | None = None
        self._decode_attention_mask_cache: dict[int, np.ndarray] = {}
        self._decode_cache_position_cache: dict[int, np.ndarray] = {}

        self._validate_sessions()

    @staticmethod
    def _default_providers() -> list[str]:
        import onnxruntime as ort

        # 有 CUDA EP 就优先 CUDA，保留 CPU fallback；没有 CUDA 就纯 CPU。
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _load_processor(self):
        from qwen_asr.core.transformers_backend import Qwen3ASRProcessor

        return Qwen3ASRProcessor.from_pretrained(self.model_path, fix_mistral_regex=True)

    @property
    def audio_encoder_runner(self) -> OnnxSessionRunner:
        if self._audio_encoder_runner is None:
            self._audio_encoder_runner = OnnxSessionRunner(
                self.audio_encoder_onnx_path,
                providers=self.audio_providers,
                name="asr_audio_encoder",
                use_iobinding=self.audio_use_iobinding,
            )
        return self._audio_encoder_runner

    @property
    def token_embedding_runner(self) -> OnnxSessionRunner:
        if self._token_embedding_runner is None:
            self._token_embedding_runner = OnnxSessionRunner(
                self.token_embedding_onnx_path,
                providers=self.embedding_providers,
                name="asr_token_embedding",
                use_iobinding=self.embedding_use_iobinding,
            )
        return self._token_embedding_runner

    @property
    def text_core_runner(self) -> OnnxSessionRunner:
        if self._text_core_runner is None:
            self._text_core_runner = OnnxSessionRunner(
                self.text_core_onnx_path,
                providers=self.text_providers,
                name="asr_text_core",
                use_iobinding=self.text_use_iobinding,
            )
        return self._text_core_runner

    def _validate_sessions(self) -> None:
        audio_required_inputs = {"input_features", "feature_lens"}
        audio_required_outputs = {"audio_features"}
        self._check_runner_io(self.audio_encoder_runner, audio_required_inputs, audio_required_outputs)

        embedding_required_inputs = {"input_ids"}
        embedding_required_outputs = {"inputs_embeds"}
        self._check_runner_io(self.token_embedding_runner, embedding_required_inputs, embedding_required_outputs)

        text_required_inputs = {"inputs_embeds", "attention_mask", "cache_position", "past_key_0", "past_value_0"}
        text_required_outputs = {"logits", "new_past_key_0", "new_past_value_0"}
        self._check_runner_io(self.text_core_runner, text_required_inputs, text_required_outputs)

    @staticmethod
    def _check_runner_io(
        runner: OnnxSessionRunner,
        required_inputs: set[str],
        required_outputs: set[str],
    ) -> None:
        missing_inputs = sorted(required_inputs - set(runner.input_names))
        missing_outputs = sorted(required_outputs - set(runner.output_names))
        if missing_inputs or missing_outputs:
            raise ValueError(
                f"{runner.name} ONNX IO mismatch: "
                f"missing_inputs={missing_inputs}, missing_outputs={missing_outputs}, path={runner.path}"
            )

    def _get_text_core_output_names(self) -> list[str]:
        if self._text_core_output_names is None:
            # text_core 输出很多 KV cache，这里只取 logits + new_past_key/value_*。
            # last_hidden 当前 ASR 路径不使用，所以不请求，减少一点输出绑定开销。
            names = ["logits"]
            for key_name, value_name in self._get_text_core_kv_output_names():
                names.extend([key_name, value_name])
            self._text_core_output_names = names
        return self._text_core_output_names

    def _get_text_core_kv_output_names(self) -> tuple[tuple[str, str], ...]:
        if self._text_core_kv_output_names is None:
            key_prefix = "new_past_key_"
            value_prefix = "new_past_value_"
            layer_ids = sorted(
                int(name.removeprefix(key_prefix))
                for name in self.text_core_runner.output_names
                if name.startswith(key_prefix)
            )
            self._text_core_kv_output_names = tuple(
                (f"{key_prefix}{layer_id}", f"{value_prefix}{layer_id}") for layer_id in layer_ids
            )
        return self._text_core_kv_output_names

    def _collect_past_key_values(self, raw_outputs: dict[str, Any]) -> tuple[tuple[Any, Any], ...]:
        return tuple(
            (raw_outputs[key_name], raw_outputs[value_name])
            for key_name, value_name in self._get_text_core_kv_output_names()
        )

    @staticmethod
    def _to_cpu_numpy(value: Any, name: str) -> np.ndarray:
        if is_ortvalue(value):
            try:
                return value.numpy()
            except Exception as exc:
                raise RuntimeError(
                    f"{name} must be CPU-bound before converting to numpy. "
                    "Bind this output to CPU with output_device_overrides."
                ) from exc
        return np.asarray(value)

    @staticmethod
    def _value_shape(value: Any) -> tuple[int, ...]:
        if is_ortvalue(value):
            return tuple(int(dim) for dim in value.shape())
        return tuple(int(dim) for dim in np.asarray(value).shape)

    @staticmethod
    def _past_kv_length(past_key_values: tuple[tuple[Any, Any], ...]) -> int:
        if not past_key_values:
            return 0
        key_shape = Qwen3ASROnnxModel._value_shape(past_key_values[0][0])
        if len(key_shape) < 3:
            raise ValueError(f"past_key shape must be [batch, heads, past_len, dim], got {key_shape}")
        return int(key_shape[2])

    def _empty_past_key_values(self) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        if self._empty_past_key_values_cache is None:
            # prefill 第一步没有历史 KV，但 text_core.onnx 仍然要求传入每层 past。
            # 这里构造长度为 0 的 KV：[1, num_kv_heads, 0, head_dim]。
            input_meta = self.text_core_runner.input_metas["past_key_0"]
            input_dtype = ORT_INPUT_DTYPES.get(input_meta.type, np.float16)
            shape = list(input_meta.shape)
            if len(shape) != 4:
                raise ValueError(f"past_key_0 must be rank 4, got shape={input_meta.shape}")
            batch = int(shape[0])
            heads = int(shape[1])
            head_dim = int(shape[3])
            empty_shape = (batch, heads, 0, head_dim)
            layer_count = len(self._get_text_core_kv_output_names())
            self._empty_past_key_values_cache = tuple(
                (
                    np.zeros(empty_shape, dtype=input_dtype),
                    np.zeros(empty_shape, dtype=input_dtype),
                )
                for _ in range(layer_count)
            )
        return self._empty_past_key_values_cache

    def _decode_attention_mask(self, past_len: int) -> np.ndarray:
        past_len = int(past_len)
        cached = self._decode_attention_mask_cache.get(past_len)
        if cached is None:
            cached = np.ones((1, past_len + 1), dtype=np.int64)
            self._decode_attention_mask_cache[past_len] = cached
        return cached

    def _decode_cache_position(self, past_len: int) -> np.ndarray:
        past_len = int(past_len)
        cached = self._decode_cache_position_cache.get(past_len)
        if cached is None:
            cached = np.array([past_len], dtype=np.int64)
            self._decode_cache_position_cache[past_len] = cached
        return cached

    def _build_messages(self, context: str, audio_payload: Any) -> list[dict[str, Any]]:
        # 对齐 qwen-asr 官方 Qwen3ASRModel：context 放 system，音频占位放 user。
        return [
            {"role": "system", "content": context or ""},
            {"role": "user", "content": [{"type": "audio", "audio": audio_payload}]},
        ]

    @staticmethod
    def _normalize_hotwords(hotwords: Optional[str | Sequence[str]]) -> list[str]:
        if hotwords is None:
            return []
        if isinstance(hotwords, str):
            items = hotwords.replace("，", ",").split(",")
        else:
            items = hotwords
        return [str(item).strip() for item in items if str(item).strip()]

    @classmethod
    def _context_with_hotwords(cls, context: str = "", hotwords: Optional[str | Sequence[str]] = None) -> str:
        # 热词不是模型的单独输入，只能作为 system context 注入。
        # 这里的措辞尽量温和，避免模型无条件把热词硬塞进结果。
        words = cls._normalize_hotwords(hotwords)
        if not words:
            return str(context or "")
        hint = "热词：" + "、".join(words) + "。仅在语音内容匹配时优先使用这些词。"
        context = str(context or "").strip()
        return f"{context}\n{hint}" if context else hint

    def _build_text_prompt(
        self,
        context: str = "",
        language: Optional[str] = None,
        hotwords: Optional[str | Sequence[str]] = None,
        output_prefix: str = "",
    ) -> str:
        context = self._context_with_hotwords(context=context, hotwords=hotwords)
        messages = self._build_messages(context=context, audio_payload="")
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        if language:
            # 强制语言时，官方实现会在 assistant prompt 后追加这个生成前缀。
            prompt += f"language {language}{'<asr_text>'}"
        if output_prefix:
            # 伪流式会把上一轮确认的文本作为 assistant prefix，帮助下一轮续写。
            prompt += str(output_prefix)
        return prompt

    def _prepare_inputs(
        self,
        wav: np.ndarray,
        context: str = "",
        language: Optional[str] = None,
        hotwords: Optional[str | Sequence[str]] = None,
        output_prefix: str = "",
    ) -> dict[str, np.ndarray]:
        prompt = self._build_text_prompt(
            context=context,
            language=language,
            hotwords=hotwords,
            output_prefix=output_prefix,
        )
        inputs = self.processor(text=[prompt], audio=[wav], return_tensors="np", padding=True)
        return {key: np.asarray(value) for key, value in inputs.items()}

    def _run_audio_encoder(self, input_features: np.ndarray, feature_attention_mask: np.ndarray) -> np.ndarray:
        # feature_attention_mask 的有效长度就是真实 mel 帧数；右侧 padding 不送进 audio_encoder。
        feature_len = int(np.asarray(feature_attention_mask).sum(axis=-1).reshape(-1)[0])
        trimmed_features = np.ascontiguousarray(input_features[:, :, :feature_len])
        outputs = self.audio_encoder_runner.run(
            output_names=["audio_features"],
            feed={
                "input_features": trimmed_features,
                "feature_lens": np.array([feature_len], dtype=np.int64),
            },
            use_iobinding=self.audio_encoder_runner.use_iobinding,
            copy_outputs_to_cpu=True,
        )
        return np.asarray(outputs[0])

    def _run_token_embedding_cpu(self, input_ids: np.ndarray) -> np.ndarray:
        outputs = self.token_embedding_runner.run(
            output_names=["inputs_embeds"],
            feed={"input_ids": np.ascontiguousarray(input_ids.astype(np.int64, copy=False))},
            use_iobinding=self.token_embedding_runner.use_iobinding,
            copy_outputs_to_cpu=True,
        )
        return np.asarray(outputs[0])

    def _run_token_embedding_for_decode(self, token_id: int) -> Any:
        outputs = self.token_embedding_runner.run(
            output_names=["inputs_embeds"],
            feed={"input_ids": np.array([[int(token_id)]], dtype=np.int64)},
            use_iobinding=self.token_embedding_runner.use_iobinding,
            copy_outputs_to_cpu=not self.token_embedding_runner.use_iobinding,
        )
        return outputs[0]

    def _merge_audio_features(
        self,
        input_ids: np.ndarray,
        inputs_embeds: np.ndarray,
        audio_features: np.ndarray,
    ) -> np.ndarray:
        # processor 会把一个 <|audio_pad|> 展开成和 audio_features 等长的一串占位 token。
        # 这里模拟原始 PyTorch 的 masked_scatter：把占位 token 的 embedding 替换成音频特征。
        input_ids = np.asarray(input_ids)
        inputs_embeds = np.asarray(inputs_embeds).copy()
        audio_features = np.asarray(audio_features)
        audio_positions = input_ids[0] == self.audio_token_id
        placeholder_count = int(audio_positions.sum())
        audio_feature_count = int(audio_features.shape[0])
        if placeholder_count != audio_feature_count:
            raise ValueError(
                "audio placeholder count does not match audio_encoder output length: "
                f"placeholders={placeholder_count}, audio_features={audio_feature_count}, "
                f"input_ids_shape={input_ids.shape}, audio_features_shape={audio_features.shape}"
            )
        inputs_embeds[0, audio_positions, :] = audio_features.reshape(audio_feature_count, inputs_embeds.shape[-1])
        return np.ascontiguousarray(inputs_embeds)

    def _prepare_prefill_embeds(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        # prefill 的输入 embedding = 文本 token embedding + audio_encoder 输出替换 audio placeholder。
        # 这一步当前在 CPU/Numpy 合并；后续高频 decode 阶段才主要依赖 IO binding。
        input_ids = np.asarray(inputs["input_ids"], dtype=np.int64)
        token_embeds = self._run_token_embedding_cpu(input_ids)
        audio_features = self._run_audio_encoder(inputs["input_features"], inputs["feature_attention_mask"])
        return self._merge_audio_features(input_ids, token_embeds, audio_features)

    def _run_text_core(self, feed: dict[str, Any]) -> ASRTextCoreOutputs:
        output_names = self._get_text_core_output_names()
        outputs = self.text_core_runner.run(
            output_names=output_names,
            feed=feed,
            use_iobinding=self.text_core_runner.use_iobinding,
            copy_outputs_to_cpu=not self.text_core_runner.use_iobinding,
            # logits 需要回 CPU 做 greedy argmax；KV cache 不指定 CPU，IO binding 时会留在设备上。
            output_device_overrides={"logits": "cpu"},
        )
        raw_outputs = {name: output for name, output in zip(output_names, outputs)}
        logits = self._to_cpu_numpy(raw_outputs["logits"], "logits")
        raw_outputs["logits"] = logits
        return ASRTextCoreOutputs(
            logits=logits,
            past_key_values=self._collect_past_key_values(raw_outputs),
            raw_outputs=raw_outputs,
        )

    def run_prefill(self, inputs: dict[str, np.ndarray]) -> ASRTextCoreOutputs:
        # prefill 一次性处理完整 prompt：system/user/audio placeholders/assistant 前缀。
        # cache_position 从 0 到 seq_len-1，past KV 为空。
        inputs_embeds = self._prepare_prefill_embeds(inputs)
        seq_len = int(inputs_embeds.shape[1])
        feed: dict[str, Any] = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": np.ascontiguousarray(np.asarray(inputs["attention_mask"], dtype=np.int64)),
            "cache_position": np.arange(seq_len, dtype=np.int64),
        }
        for layer_id, (key, value) in enumerate(self._empty_past_key_values()):
            feed[f"past_key_{layer_id}"] = key
            feed[f"past_value_{layer_id}"] = value
        return self._run_text_core(feed)

    def run_decode_step(
        self,
        token_id: int,
        past_key_values: tuple[tuple[Any, Any], ...],
    ) -> ASRTextCoreOutputs:
        # decode 每次只喂上一个 token 的 embedding，并带上前一步返回的新 KV。
        # token_embedding 默认 CPU 查表；KV cache 在 text_core IO binding 路径下通常是 OrtValue。
        past_len = self._past_kv_length(past_key_values)
        inputs_embeds = self._run_token_embedding_for_decode(token_id)
        feed: dict[str, Any] = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": self._decode_attention_mask(past_len),
            "cache_position": self._decode_cache_position(past_len),
        }
        for layer_id, (key, value) in enumerate(past_key_values):
            feed[f"past_key_{layer_id}"] = key
            feed[f"past_value_{layer_id}"] = value
        return self._run_text_core(feed)

    @staticmethod
    def _greedy_next_token(logits: np.ndarray) -> int:
        next_token_logits = np.asarray(logits)[:, -1, :]
        return int(np.argmax(next_token_logits, axis=-1).reshape(-1)[0])

    def generate_ids(
        self,
        inputs: dict[str, np.ndarray],
        max_new_tokens: Optional[int] = None,
        eos_token_ids: Sequence[int] | None = None,
    ) -> np.ndarray:
        # 整个 ASR 文本生成固定为 greedy：每步取 logits 最大的 token，遇到 eos 停止。
        max_new_tokens = self.max_new_tokens if max_new_tokens is None else int(max_new_tokens)
        eos_ids = set(int(item) for item in (self.eos_token_ids if eos_token_ids is None else eos_token_ids))
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")

        state = self.run_prefill(inputs)
        generated: list[int] = []
        logits = state.logits
        past_key_values = state.past_key_values

        for _ in range(max_new_tokens):
            token_id = self._greedy_next_token(logits)
            if token_id in eos_ids:
                break
            generated.append(token_id)
            state = self.run_decode_step(token_id, past_key_values)
            logits = state.logits
            past_key_values = state.past_key_values

        return np.asarray(generated, dtype=np.int64)

    def decode_ids(self, token_ids: np.ndarray) -> str:
        token_ids = np.asarray(token_ids, dtype=np.int64).reshape(1, -1)
        decoded = self.processor.batch_decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return str(decoded[0])

    def infer_chunk(
        self,
        wav: np.ndarray,
        context: str = "",
        language: Optional[str] = None,
        hotwords: Optional[str | Sequence[str]] = None,
        output_prefix: str = "",
        max_new_tokens: Optional[int] = None,
    ) -> str:
        # 单个音频 chunk 的最小 ASR 单元：准备 processor 输入 -> 生成 token -> decode 成文本。
        # 离线 transcribe 和伪流式每次重算都会复用这个函数。
        inputs = self._prepare_inputs(
            wav=wav,
            context=context,
            language=language,
            hotwords=hotwords,
            output_prefix=output_prefix,
        )
        token_ids = self.generate_ids(inputs, max_new_tokens=max_new_tokens)
        return self.decode_ids(token_ids)

    def transcribe(
        self,
        audio: Any,
        context: str | list[str] = "",
        language: Optional[str] | list[Optional[str]] = None,
        hotwords: Optional[str | Sequence[str]] = None,
        max_new_tokens: Optional[int] = None,
    ) -> list[ASROnnxTranscription]:
        """离线 ASR 入口。

        这里保持 qwen-asr 原版 transcribe 的习惯：支持路径、(np.ndarray, sr)、列表输入；
        内部会按官方 MAX_ASR_INPUT_SECONDS 切长音频，然后逐块识别再拼接。
        """

        from qwen_asr.inference.utils import (
            MAX_ASR_INPUT_SECONDS,
            SAMPLE_RATE,
            AudioChunk,
            merge_languages,
            normalize_audios,
            normalize_language_name,
            parse_asr_output,
            split_audio_into_chunks,
            validate_language,
        )

        audio = self._normalize_audio_argument(audio, sample_rate=SAMPLE_RATE)
        wavs = normalize_audios(audio)
        num_items = len(wavs)

        contexts = context if isinstance(context, list) else [context]
        if len(contexts) == 1 and num_items > 1:
            contexts = contexts * num_items
        if len(contexts) != num_items:
            raise ValueError(f"Batch size mismatch: audio={num_items}, context={len(contexts)}")

        if language is None:
            languages: list[Optional[str]] = [None] * num_items
        else:
            languages = language if isinstance(language, list) else [language]
            if len(languages) == 1 and num_items > 1:
                languages = languages * num_items
            if len(languages) != num_items:
                raise ValueError(f"Batch size mismatch: audio={num_items}, language={len(languages)}")

        normalized_languages: list[Optional[str]] = []
        for item in languages:
            if item is None or str(item).strip() == "":
                normalized_languages.append(None)
            else:
                normalized = normalize_language_name(str(item))
                validate_language(normalized)
                normalized_languages.append(normalized)

        chunks: list[AudioChunk] = []
        for sample_id, wav in enumerate(wavs):
            for chunk_id, (chunk_wav, offset_sec) in enumerate(
                split_audio_into_chunks(wav=wav, sr=SAMPLE_RATE, max_chunk_sec=MAX_ASR_INPUT_SECONDS)
            ):
                chunks.append(
                    AudioChunk(
                        orig_index=sample_id,
                        chunk_index=chunk_id,
                        wav=chunk_wav,
                        sr=SAMPLE_RATE,
                        offset_sec=offset_sec,
                    )
                )

        per_item_languages: list[list[str]] = [[] for _ in range(num_items)]
        per_item_texts: list[list[str]] = [[] for _ in range(num_items)]
        per_item_raw_texts: list[list[str]] = [[] for _ in range(num_items)]

        for chunk in chunks:
            forced_language = normalized_languages[chunk.orig_index]
            raw_text = self.infer_chunk(
                wav=chunk.wav,
                context=str(contexts[chunk.orig_index] or ""),
                language=forced_language,
                hotwords=hotwords,
                max_new_tokens=max_new_tokens,
            )
            predicted_language, text = parse_asr_output(raw_text, user_language=forced_language)
            per_item_languages[chunk.orig_index].append(predicted_language)
            per_item_texts[chunk.orig_index].append(text)
            per_item_raw_texts[chunk.orig_index].append(raw_text)

        results: list[ASROnnxTranscription] = []
        for item_id in range(num_items):
            merged_text = "".join(text for text in per_item_texts[item_id] if text is not None)
            merged_language = merge_languages(per_item_languages[item_id])
            raw_text = "".join(text for text in per_item_raw_texts[item_id] if text is not None)
            results.append(
                ASROnnxTranscription(
                    language=merged_language,
                    text=merged_text,
                    raw_text=raw_text,
                    metadata={
                        "chunk_count": len(per_item_raw_texts[item_id]),
                        "hotwords": self._normalize_hotwords(hotwords),
                        "use_iobinding": {
                            "audio_encoder": bool(self.audio_encoder_runner.use_iobinding),
                            "token_embedding": bool(self.token_embedding_runner.use_iobinding),
                            "text_core": bool(self.text_core_runner.use_iobinding),
                        },
                        "providers": {
                            "audio_encoder": list(self.audio_encoder_runner.providers),
                            "token_embedding": list(self.token_embedding_runner.providers),
                            "text_core": list(self.text_core_runner.providers),
                        },
                    },
                )
            )
        return results

    @staticmethod
    def _normalize_audio_argument(audio: Any, sample_rate: int) -> Any:
        if isinstance(audio, np.ndarray):
            return (audio, int(sample_rate))
        if isinstance(audio, list):
            return [
                (item, int(sample_rate)) if isinstance(item, np.ndarray) else item
                for item in audio
            ]
        return audio

    def init_streaming_state(
        self,
        context: str = "",
        language: Optional[str] = None,
        hotwords: Optional[str | Sequence[str]] = None,
        unfixed_chunk_num: int = 2,
        unfixed_token_num: int = 5,
        chunk_size_sec: float = 2.0,
        max_audio_history_sec: Optional[float] = None,
    ) -> ASROnnxStreamingState:
        """创建伪流式状态。

        chunk_size_sec 控制“攒够多少秒触发一次识别”。
        max_audio_history_sec 控制每次重算最多保留最近多少秒音频；None 表示保留全部历史。
        """

        from qwen_asr.inference.utils import SAMPLE_RATE, normalize_language_name, validate_language

        if chunk_size_sec is None or float(chunk_size_sec) <= 0:
            raise ValueError(f"chunk_size_sec must be > 0, got: {chunk_size_sec}")
        max_audio_history_samples = None
        if max_audio_history_sec is not None and float(max_audio_history_sec) > 0:
            max_audio_history_samples = max(1, int(round(float(max_audio_history_sec) * SAMPLE_RATE)))

        force_language = None
        if language is not None and str(language).strip() != "":
            force_language = normalize_language_name(str(language))
            validate_language(force_language)

        chunk_size_samples = max(1, int(round(float(chunk_size_sec) * SAMPLE_RATE)))
        return ASROnnxStreamingState(
            unfixed_chunk_num=int(unfixed_chunk_num),
            unfixed_token_num=int(unfixed_token_num),
            chunk_size_sec=float(chunk_size_sec),
            chunk_size_samples=int(chunk_size_samples),
            max_audio_history_samples=max_audio_history_samples,
            chunk_id=0,
            total_audio_samples=0,
            total_trimmed_samples=0,
            last_trimmed_samples=0,
            trim_events=0,
            buffer=np.zeros((0,), dtype=np.float32),
            audio_accum=np.zeros((0,), dtype=np.float32),
            audio_storage=np.zeros((0,), dtype=np.float32),
            audio_storage_start=0,
            audio_accum_samples=0,
            last_appended_start_samples=0,
            last_appended_end_samples=0,
            context=str(context or ""),
            force_language=force_language,
            hotwords=self._normalize_hotwords(hotwords),
            metadata={
                "streaming_mode": "pseudo",
                "recompute_audio": True,
            },
        )

    def streaming_transcribe(
        self,
        pcm16k: np.ndarray,
        state: ASROnnxStreamingState,
        max_new_tokens: Optional[int] = None,
    ) -> ASROnnxStreamingState:
        """喂入一段 16 kHz mono PCM，并在 buffer 满一个 chunk 时触发识别。

        注意：这不是 encoder-cache 真流式。每次触发时都会把 state.audio_accum 重新送进
        audio_encoder/text_core 跑一遍，只是用上一轮文本 prefix 帮助续写和稳定输出。
        """

        if state is None:
            raise ValueError("state must not be None. Call init_streaming_state() first.")
        if pcm16k is None:
            raise ValueError("pcm16k must not be None.")

        state.confirmed_delta_text = ""
        state.delta_raw_text = ""
        x = self._normalize_stream_pcm(pcm16k)
        state.metadata.update(
            {
                "input_samples": int(x.shape[0]),
                "input_ms": int(round(x.shape[0] * 1000.0 / STREAM_SAMPLE_RATE)),
                "buffer_samples_before": int(state.buffer.shape[0]),
            }
        )
        if x.shape[0] > 0:
            state.buffer = np.concatenate([state.buffer, x], axis=0)
        state.metadata["buffer_samples_after_append"] = int(state.buffer.shape[0])

        # feed 可以比 chunk 小：不够 chunk_size_samples 时只缓存，不做识别。
        triggered = 0
        while state.buffer.shape[0] >= state.chunk_size_samples:
            chunk = state.buffer[: state.chunk_size_samples]
            state.buffer = state.buffer[state.chunk_size_samples :]
            self._append_stream_audio(state, chunk)
            self._run_streaming_decode(state, max_new_tokens=max_new_tokens)
            triggered += 1

        state.metadata.update(
            {
                "triggered_decode_count": int(triggered),
                "buffer_samples": int(state.buffer.shape[0]),
            }
        )

        return state

    def finish_streaming_transcribe(
        self,
        state: ASROnnxStreamingState,
        max_new_tokens: Optional[int] = None,
    ) -> ASROnnxStreamingState:
        """结束流式输入时调用，强制处理 buffer 里不足一个 chunk 的尾巴。"""

        if state is None:
            raise ValueError("state must not be None.")
        state.confirmed_delta_text = ""
        state.delta_raw_text = ""
        state.metadata.update(
            {
                "input_samples": int(state.buffer.shape[0]) if state.buffer is not None else 0,
                "input_ms": int(round((state.buffer.shape[0] if state.buffer is not None else 0) * 1000.0 / STREAM_SAMPLE_RATE)),
                "buffer_samples_before": int(state.buffer.shape[0]) if state.buffer is not None else 0,
                "finalize_call": True,
            }
        )
        if state.buffer is None or state.buffer.shape[0] == 0:
            self._commit_streaming_raw(state, finalize=True)
            state.metadata.update(
                {
                    "triggered_decode_count": 0,
                    "buffer_samples": 0,
                    "audio_accum_samples": int(state.audio_accum_samples),
                    "asr_window_samples": int(state.audio_accum_samples),
                    "asr_window_ms": int(round(state.audio_accum_samples * 1000.0 / STREAM_SAMPLE_RATE)),
                    "total_audio_samples": int(state.total_audio_samples),
                    "audio_end_ms": int(round(state.total_audio_samples * 1000.0 / STREAM_SAMPLE_RATE)),
                }
            )
            return state

        tail = state.buffer
        state.buffer = np.zeros((0,), dtype=np.float32)
        self._append_stream_audio(state, tail)
        self._run_streaming_decode(state, max_new_tokens=max_new_tokens, finalize=True)
        return state

    @staticmethod
    def _normalize_stream_pcm(pcm16k: np.ndarray) -> np.ndarray:
        # 流式接口约定外部传 16 kHz 单声道 PCM；这里只做 dtype/shape/range 归一化。
        x = np.asarray(pcm16k)
        if x.ndim != 1:
            x = x.reshape(-1)
        if x.dtype == np.int16:
            x = x.astype(np.float32) / 32768.0
        else:
            x = x.astype(np.float32, copy=False)
        if x.size == 0:
            return np.zeros((0,), dtype=np.float32)
        return np.clip(x, -1.0, 1.0)

    @staticmethod
    def _append_stream_audio(state: ASROnnxStreamingState, chunk: np.ndarray) -> None:
        # audio_accum 是下一次重算要喂给模型的音频窗口；可以按 max_audio_history_samples 裁剪。
        # 内部使用可复用 storage，避免每个流式 chunk 都 np.concatenate 整段历史音频。
        chunk = np.ascontiguousarray(np.asarray(chunk, dtype=np.float32).reshape(-1))
        chunk_samples = int(chunk.shape[0])
        state.last_appended_start_samples = int(state.total_audio_samples)
        state.last_appended_end_samples = int(state.total_audio_samples) + chunk_samples
        state.total_audio_samples += chunk_samples
        state.last_trimmed_samples = 0

        if state.max_audio_history_samples is not None and chunk_samples > 0:
            max_samples = int(state.max_audio_history_samples)
            overflow = max(0, int(state.audio_accum_samples) + chunk_samples - max_samples)
            if overflow > 0:
                drop_existing = min(int(state.audio_accum_samples), overflow)
                if drop_existing > 0:
                    state.audio_storage_start += drop_existing
                    state.audio_accum_samples -= drop_existing
                drop_from_chunk = overflow - drop_existing
                if drop_from_chunk > 0:
                    chunk = chunk[drop_from_chunk:]
                    chunk_samples = int(chunk.shape[0])
                state.last_trimmed_samples = int(overflow)
                state.total_trimmed_samples += int(overflow)
                state.trim_events += 1

        if chunk_samples > 0:
            Qwen3ASROnnxModel._ensure_stream_audio_capacity(state, chunk_samples)
            write_start = int(state.audio_storage_start) + int(state.audio_accum_samples)
            write_end = write_start + chunk_samples
            state.audio_storage[write_start:write_end] = chunk
            state.audio_accum_samples += chunk_samples

        state.audio_accum = Qwen3ASROnnxModel._stream_audio_view(state)

    @staticmethod
    def _stream_audio_view(state: ASROnnxStreamingState) -> np.ndarray:
        start = int(state.audio_storage_start)
        end = start + int(state.audio_accum_samples)
        if end <= start:
            return np.zeros((0,), dtype=np.float32)
        return state.audio_storage[start:end]

    @staticmethod
    def _ensure_stream_audio_capacity(state: ASROnnxStreamingState, extra_samples: int) -> None:
        extra_samples = int(extra_samples)
        if extra_samples <= 0:
            return

        current_samples = int(state.audio_accum_samples)
        required = current_samples + extra_samples
        capacity = int(state.audio_storage.shape[0])
        write_end = int(state.audio_storage_start) + current_samples + extra_samples
        if capacity >= required and write_end <= capacity:
            return

        if capacity >= required:
            if current_samples > 0:
                start = int(state.audio_storage_start)
                state.audio_storage[:current_samples] = state.audio_storage[start : start + current_samples].copy()
            state.audio_storage_start = 0
            return

        new_capacity = max(required, max(1, capacity * 2), int(state.chunk_size_samples))
        new_storage = np.zeros((new_capacity,), dtype=np.float32)
        if current_samples > 0:
            start = int(state.audio_storage_start)
            new_storage[:current_samples] = state.audio_storage[start : start + current_samples]
        state.audio_storage = new_storage
        state.audio_storage_start = 0

    def _streaming_prefix(self, state: ASROnnxStreamingState) -> str:
        # 前几轮上下文太短，先不信上一轮文本；之后回滚末尾若干 token 再作为稳定 prefix。
        # 这样上一轮最后几个字如果猜错，下一轮还有机会修正。
        if state.chunk_id < int(state.unfixed_chunk_num):
            return ""
        token_ids = state._raw_decoded_token_ids or self.processor.tokenizer.encode(state._raw_decoded)
        return self._decode_with_token_rollback(token_ids, max(0, int(state.unfixed_token_num)))

    def _decode_with_token_rollback(self, token_ids: Sequence[int], rollback: int) -> str:
        token_ids = list(token_ids)
        rollback = max(0, int(rollback))
        while True:
            end_idx = max(0, len(token_ids) - rollback)
            decoded = self.processor.tokenizer.decode(token_ids[:end_idx]) if end_idx > 0 else ""
            if "\ufffd" not in decoded:
                return decoded
            if end_idx == 0:
                return ""
            rollback += 1

    def _stable_streaming_raw(self, state: ASROnnxStreamingState, *, finalize: bool = False) -> str:
        if finalize:
            return state._raw_decoded
        if state.chunk_id < int(state.unfixed_chunk_num):
            return ""

        token_ids = state._raw_decoded_token_ids or self.processor.tokenizer.encode(state._raw_decoded)
        return self._decode_with_token_rollback(token_ids, max(0, int(state.unfixed_token_num)))

    def _candidate_streaming_raw(self, state: ASROnnxStreamingState, *, finalize: bool = False) -> str:
        if finalize:
            return state._raw_decoded
        token_ids = state._raw_decoded_token_ids or self.processor.tokenizer.encode(state._raw_decoded)
        return self._decode_with_token_rollback(token_ids, 0)

    def _commit_streaming_raw(self, state: ASROnnxStreamingState, *, finalize: bool = False) -> None:
        from qwen_asr.inference.utils import parse_asr_output

        committed_raw = self._stable_streaming_raw(state, finalize=finalize)
        state._committed_raw_decoded = committed_raw
        candidate_raw = self._candidate_streaming_raw(state, finalize=finalize)
        language, candidate_text = parse_asr_output(candidate_raw, user_language=state.force_language)
        state.language = language
        confirmed_delta, pending_text, text_revision = self._consume_confirmed_and_pending_delta(
            candidate_text,
            state._emitted_text,
            state._pending_text,
            finalize=finalize,
        )
        if finalize and text_revision:
            state._emitted_text = candidate_text
            state._pending_text = ""
        elif text_revision:
            state._emitted_text = ""
            state._pending_text = pending_text
        else:
            state._emitted_text += confirmed_delta
            state._pending_text = pending_text
        state.confirmed_delta_text += confirmed_delta
        state.confirmed_text = state._emitted_text
        state.pending_text = state._pending_text
        state.text = f"{state.confirmed_text}{state.pending_text}"
        delta_raw, raw_revision = self._consume_stream_delta(committed_raw, state._emitted_raw_decoded)
        if not raw_revision:
            state.delta_raw_text += delta_raw
            state._emitted_raw_decoded = committed_raw
        state.metadata.update(
            {
                "committed_raw_chars": len(committed_raw),
                "uncommitted_raw_chars": max(0, len(state._raw_decoded) - len(committed_raw)),
                "confirmed_delta_chars": len(state.confirmed_delta_text),
                "confirmed_text_chars": len(state.confirmed_text),
                "pending_chars": len(state.pending_text),
                "delta_raw_chars": len(state.delta_raw_text),
                "emitted_text_chars": len(state._emitted_text),
                "stream_text_revision": bool(text_revision),
                "stream_raw_revision": bool(raw_revision),
                "finalized": bool(finalize),
            }
        )

    @staticmethod
    def _consume_stream_delta(current: str, emitted: str) -> tuple[str, bool]:
        current = str(current or "")
        emitted = str(emitted or "")
        if not current:
            return "", False
        if current.startswith(emitted):
            return current[len(emitted) :], False
        return "", True

    @staticmethod
    def _consume_confirmed_and_pending_delta(
        current: str,
        confirmed: str,
        previous_pending: str,
        *,
        finalize: bool = False,
    ) -> tuple[str, str, bool]:
        current = str(current or "")
        confirmed = str(confirmed or "")
        previous_pending = str(previous_pending or "")
        if not current:
            return "", "", False

        if finalize:
            if current.startswith(confirmed):
                confirmed_delta = current[len(confirmed) :]
                return confirmed_delta, "", False
            return "", "", bool(confirmed or previous_pending)

        if not current.startswith(confirmed):
            return "", current, True

        suffix = current[len(confirmed) :]
        prefix_len = 0
        for left, right in zip(previous_pending, suffix):
            if left != right:
                break
            prefix_len += 1
        confirmed_delta = suffix[:prefix_len]
        pending = suffix[prefix_len:]
        return confirmed_delta, pending, False

    def _run_streaming_decode(
        self,
        state: ASROnnxStreamingState,
        max_new_tokens: Optional[int] = None,
        finalize: bool = False,
    ) -> None:
        # 伪流式的核心：prefix + 当前音频窗口重新做一次普通 ASR。
        # state.raw_text 保留完整候选；state.text 暴露 confirmed + pending 的当前可展示文本。
        prefix = self._streaming_prefix(state)
        generated = self.infer_chunk(
            wav=state.audio_accum,
            context=state.context,
            language=state.force_language,
            hotwords=state.hotwords,
            output_prefix=prefix,
            max_new_tokens=max_new_tokens,
        )
        state.last_raw_generated = generated
        state._raw_decoded = f"{prefix}{generated}" if prefix else generated
        state._raw_decoded_token_ids = list(self.processor.tokenizer.encode(state._raw_decoded))
        self._commit_streaming_raw(state, finalize=finalize)
        decoded_chunk_id = int(state.chunk_id)
        state.chunk_id += 1
        audio_window_start_samples = int(state.total_audio_samples) - int(state.audio_accum_samples)
        state.metadata.update(
            {
                "decoded_chunk_id": decoded_chunk_id,
                "processed_chunks": int(state.chunk_id),
                "buffer_samples": int(state.buffer.shape[0]),
                "audio_accum_samples": int(state.audio_accum_samples),
                "asr_window_samples": int(state.audio_accum_samples),
                "asr_window_ms": int(round(state.audio_accum_samples * 1000.0 / STREAM_SAMPLE_RATE)),
                "audio_storage_capacity_samples": int(state.audio_storage.shape[0]),
                "total_audio_samples": int(state.total_audio_samples),
                "audio_end_ms": int(round(state.total_audio_samples * 1000.0 / STREAM_SAMPLE_RATE)),
                "trigger_chunk_start_ms": int(
                    round(state.last_appended_start_samples * 1000.0 / STREAM_SAMPLE_RATE)
                ),
                "trigger_chunk_end_ms": int(
                    round(state.last_appended_end_samples * 1000.0 / STREAM_SAMPLE_RATE)
                ),
                "audio_window_start_ms": int(round(audio_window_start_samples * 1000.0 / STREAM_SAMPLE_RATE)),
                "audio_window_end_ms": int(round(state.total_audio_samples * 1000.0 / STREAM_SAMPLE_RATE)),
                "max_audio_history_samples": state.max_audio_history_samples,
                "trim_events": int(state.trim_events),
                "last_trimmed_samples": int(state.last_trimmed_samples),
                "total_trimmed_samples": int(state.total_trimmed_samples),
                "prefix_chars": len(prefix),
                "hotwords": list(state.hotwords),
                "use_iobinding": {
                    "audio_encoder": bool(self.audio_encoder_runner.use_iobinding),
                    "token_embedding": bool(self.token_embedding_runner.use_iobinding),
                    "text_core": bool(self.text_core_runner.use_iobinding),
                },
            }
        )
