from __future__ import annotations

"""ONNXRuntime wrapper for FSMN-VAD."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml

from src.OnnxSessionRunner import OnnxSessionRunner


DEFAULT_FSMN_MODEL_DIR = Path("./fsmn")
DEFAULT_FSMN_ONNX_PATH = Path("./onnx_fsmn/fsmn_vad_encoder.onnx")
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_STREAM_CHUNK_MS = 200


@dataclass
class VadResult:
    """Normalized VAD result.

    segments are finalized speech ranges in milliseconds: [[start_ms, end_ms], ...].
    events are the raw streaming events:
    [beg, -1] for speech start, [-1, end] for speech end, [beg, end] for a complete segment.
    """

    segments: list[list[int]]
    events: list[list[int]] = field(default_factory=list)
    raw: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class FsmnVadStreamingState:
    """Mutable state for one streaming VAD session."""

    cache: dict[str, Any]
    chunk_size_ms: int
    sample_rate: int
    processed_chunks: int = 0
    total_samples: int = 0
    current_start_ms: int | None = None
    finalized_segments: list[list[int]] = field(default_factory=list)
    raw_events: list[list[int]] = field(default_factory=list)

    @property
    def total_ms(self) -> float:
        return self.total_samples * 1000.0 / self.sample_rate

    @property
    def in_speech(self) -> bool:
        return self.current_start_ms is not None


@dataclass
class FsmnVadOnnxEncoderOutputs:
    scores: np.ndarray
    caches: list[np.ndarray]


class FsmnVadOnnxEncoderSession:
    """Runs the exported FSMN encoder ONNX with streaming caches."""

    def __init__(
        self,
        onnx_path: str | Path = DEFAULT_FSMN_ONNX_PATH,
        encoder_conf: Optional[dict[str, Any]] = None,
        providers: Optional[list[str]] = None,
        use_iobinding: Optional[bool] = None,
    ) -> None:
        self.onnx_path = Path(onnx_path)
        self.encoder_conf = dict(encoder_conf or {})
        self.num_layers = int(self.encoder_conf.get("fsmn_layers", 4))
        self.proj_dim = int(self.encoder_conf.get("proj_dim", 128))
        self.cache_frames = int(
            (int(self.encoder_conf.get("lorder", 20)) - 1)
            * int(self.encoder_conf.get("lstride", 1))
        )
        self.runner = OnnxSessionRunner(
            self.onnx_path,
            providers=providers or ["CPUExecutionProvider"],
            name="fsmn_vad_encoder",
            use_iobinding=use_iobinding,
        )
        self._validate_io()

    def _validate_io(self) -> None:
        required_inputs = {"speech"} | {f"in_cache{i}" for i in range(self.num_layers)}
        required_outputs = {"logits"} | {f"out_cache{i}" for i in range(self.num_layers)}
        missing_inputs = sorted(required_inputs - set(self.runner.input_names))
        missing_outputs = sorted(required_outputs - set(self.runner.output_names))
        if missing_inputs or missing_outputs:
            raise RuntimeError(
                f"FSMN ONNX IO mismatch: missing_inputs={missing_inputs}, "
                f"missing_outputs={missing_outputs}"
            )

    def empty_cache(self, batch_size: int = 1) -> list[np.ndarray]:
        return [
            np.zeros((batch_size, self.proj_dim, self.cache_frames, 1), dtype=np.float32)
            for _ in range(self.num_layers)
        ]

    def run(
        self,
        speech: np.ndarray,
        caches: Optional[list[np.ndarray]] = None,
    ) -> FsmnVadOnnxEncoderOutputs:
        speech = np.asarray(speech, dtype=np.float32)
        if speech.ndim != 3:
            raise ValueError(f"speech must have shape [B, T, D], got {speech.shape}")

        batch_size = int(speech.shape[0])
        caches = caches if caches is not None else self.empty_cache(batch_size=batch_size)
        if len(caches) != self.num_layers:
            raise ValueError(f"expected {self.num_layers} caches, got {len(caches)}")

        feed: dict[str, np.ndarray] = {"speech": np.ascontiguousarray(speech)}
        for i, cache in enumerate(caches):
            feed[f"in_cache{i}"] = np.ascontiguousarray(np.asarray(cache, dtype=np.float32))

        output_names = ["logits"] + [f"out_cache{i}" for i in range(self.num_layers)]
        outputs = self.runner.run(output_names=output_names, feed=feed)
        return FsmnVadOnnxEncoderOutputs(
            scores=np.asarray(outputs[0], dtype=np.float32),
            caches=[np.asarray(item, dtype=np.float32) for item in outputs[1:]],
        )


class FsmnVadOnnxEncoderAdapter(torch.nn.Module):
    """Drop-in replacement for FunASR's torch FSMN encoder."""

    def __init__(self, session: FsmnVadOnnxEncoderSession) -> None:
        super().__init__()
        self.session = session
        # The surrounding FunASR module may inspect encoder.parameters() to infer
        # device placement. The real encoder is in ONNXRuntime, so this tiny
        # parameter only keeps that compatibility path working.
        self._dummy_device_parameter = torch.nn.Parameter(
            torch.empty(0), requires_grad=False
        )

    def forward(
        self,
        input: torch.Tensor,
        cache: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        if cache is None:
            cache = {}

        input_device = input.device
        speech = input.detach().cpu().numpy().astype(np.float32, copy=False)
        caches = self._read_caches(cache, batch_size=int(speech.shape[0]))
        outputs = self.session.run(speech, caches=caches)
        self._write_caches(cache, outputs.caches)
        return torch.from_numpy(outputs.scores).to(input_device)

    def output_size(self) -> int:
        return int(self.session.encoder_conf.get("output_dim", 248))

    def _read_caches(self, cache: dict[str, Any], batch_size: int) -> list[np.ndarray]:
        values: list[np.ndarray] = []
        defaults = self.session.empty_cache(batch_size=batch_size)
        for i in range(self.session.num_layers):
            key = f"cache_layer_{i}"
            value = cache.get(key)
            if value is None:
                values.append(defaults[i])
            elif isinstance(value, torch.Tensor):
                values.append(value.detach().cpu().numpy().astype(np.float32, copy=False))
            else:
                values.append(np.asarray(value, dtype=np.float32))
        return values

    @staticmethod
    def _write_caches(cache: dict[str, Any], caches: list[np.ndarray]) -> None:
        for i, value in enumerate(caches):
            cache[f"cache_layer_{i}"] = value


class FsmnVadOnnxModel:
    """FSMN-VAD inference with ONNXRuntime encoder and FunASR postprocess."""

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_FSMN_MODEL_DIR,
        onnx_path: str | Path = DEFAULT_FSMN_ONNX_PATH,
        providers: Optional[list[str]] = None,
        use_iobinding: Optional[bool] = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        **runtime_kwargs: Any,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.onnx_path = Path(onnx_path)
        self.sample_rate = int(sample_rate)
        self.config = self._load_config()
        self.frontend = self._build_frontend()
        self.model = self._build_vad_model().eval()
        self.device = "cpu"
        self.runtime_kwargs = {
            "device": self.device,
            **runtime_kwargs,
        }
        self.encoder_session = FsmnVadOnnxEncoderSession(
            onnx_path=self.onnx_path,
            encoder_conf=self.config.get("encoder_conf", {}),
            providers=providers,
            use_iobinding=use_iobinding,
        )
        self.model.encoder = FsmnVadOnnxEncoderAdapter(self.encoder_session)

    def _load_config(self) -> dict[str, Any]:
        candidates = [
            self.onnx_path.parent / "config.yaml",
            self.model_dir / "config.yaml",
        ]
        for path in candidates:
            if path.exists():
                with path.open("r", encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
                self.config_path = path
                return dict(config)
        raise FileNotFoundError(
            "missing FSMN-VAD config.yaml; checked "
            + ", ".join(str(path) for path in candidates)
        )

    def _build_frontend(self):
        from funasr.frontends.wav_frontend import WavFrontendOnline

        frontend_name = self.config.get("frontend", "WavFrontendOnline")
        if frontend_name != "WavFrontendOnline":
            raise ValueError(f"unsupported FSMN-VAD frontend: {frontend_name}")

        frontend_conf = dict(self.config.get("frontend_conf", {}) or {})
        if frontend_conf.get("cmvn_file") is None:
            for cmvn_path in (self.config_path.parent / "am.mvn", self.model_dir / "am.mvn"):
                if cmvn_path.exists():
                    frontend_conf["cmvn_file"] = str(cmvn_path)
                    break
        return WavFrontendOnline(**frontend_conf)

    def _build_vad_model(self) -> torch.nn.Module:
        import funasr.models.fsmn_vad_streaming.encoder  # noqa: F401
        from funasr.models.fsmn_vad_streaming.model import FsmnVADStreaming

        model_name = self.config.get("model", "FsmnVADStreaming")
        if model_name != "FsmnVADStreaming":
            raise ValueError(f"unsupported FSMN-VAD model: {model_name}")

        model_conf = dict(self.config.get("model_conf", {}) or {})
        return FsmnVADStreaming(
            encoder=self.config.get("encoder", "FSMN"),
            encoder_conf=dict(self.config.get("encoder_conf", {}) or {}),
            **model_conf,
        )

    def init_streaming_state(
        self,
        chunk_size_ms: int = DEFAULT_STREAM_CHUNK_MS,
    ) -> FsmnVadStreamingState:
        return FsmnVadStreamingState(
            cache={},
            chunk_size_ms=int(chunk_size_ms),
            sample_rate=self.sample_rate,
        )

    def detect_offline(
        self,
        audio: str | Path | np.ndarray,
        sample_rate: int | None = None,
        **generate_kwargs: Any,
    ) -> VadResult:
        audio_arg = self._normalize_audio_arg(audio, sample_rate=sample_rate)
        result = self._run_model_inference(
            audio_arg,
            cache={},
            is_final=True,
            **generate_kwargs,
        )
        segments = self._collect_values(result)
        return VadResult(segments=segments, events=segments, raw=result)

    def detect_streaming(
        self,
        audio_chunk: np.ndarray,
        state: FsmnVadStreamingState,
        is_final: bool = False,
        **generate_kwargs: Any,
    ) -> VadResult:
        chunk = self._normalize_pcm16k(audio_chunk)
        state.total_samples += int(chunk.shape[0])
        state.processed_chunks += 1
        result = self._run_model_inference(
            chunk,
            cache=state.cache,
            is_final=bool(is_final),
            chunk_size=state.chunk_size_ms,
            **generate_kwargs,
        )
        events = self._collect_values(result)
        segments = self._update_streaming_segments(state, events)
        return VadResult(segments=segments, events=events, raw=result)

    def finish_streaming(
        self,
        state: FsmnVadStreamingState,
        **generate_kwargs: Any,
    ) -> VadResult:
        tail = np.zeros(max(1, int(self.sample_rate * 0.01)), dtype=np.float32)
        return self.detect_streaming(tail, state, is_final=True, **generate_kwargs)

    def _run_model_inference(
        self,
        audio: str | Path | np.ndarray,
        cache: dict[str, Any],
        is_final: bool,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        key = kwargs.pop("key", None)
        if key is None:
            key = Path(audio).stem if isinstance(audio, (str, Path)) else "vad"
        if isinstance(key, str):
            key = [key]

        inference_kwargs = {
            **self.runtime_kwargs,
            **kwargs,
            "frontend": self.frontend,
            "cache": cache,
            "is_final": bool(is_final),
        }
        with torch.no_grad():
            result, _meta = self.model.inference(
                data_in=[audio],
                key=key,
                **inference_kwargs,
            )
        return result

    @staticmethod
    def _collect_values(result: list[dict[str, Any]]) -> list[list[int]]:
        values: list[list[int]] = []
        for item in result or []:
            for segment in item.get("value", []) or []:
                if len(segment) >= 2:
                    values.append([int(segment[0]), int(segment[1])])
        return values

    def _normalize_audio_arg(
        self,
        audio: str | Path | np.ndarray,
        sample_rate: int | None = None,
    ) -> str | np.ndarray:
        if isinstance(audio, (str, Path)):
            return str(audio)
        if sample_rate is not None and int(sample_rate) != self.sample_rate:
            raise ValueError(
                f"audio ndarray must be {self.sample_rate} Hz, got {sample_rate}; "
                "resample before calling this helper"
            )
        return self._normalize_pcm16k(audio)

    @staticmethod
    def _normalize_pcm16k(audio: np.ndarray) -> np.ndarray:
        x = np.asarray(audio)
        if x.ndim > 1:
            x = x.mean(axis=-1)
        if x.dtype == np.int16:
            x = x.astype(np.float32) / 32768.0
        else:
            x = x.astype(np.float32, copy=False)
        return np.clip(x.reshape(-1), -1.0, 1.0)

    @staticmethod
    def _update_streaming_segments(
        state: FsmnVadStreamingState,
        events: list[list[int]],
    ) -> list[list[int]]:
        new_segments: list[list[int]] = []
        for event in events:
            state.raw_events.append(event)
            start_ms, end_ms = event
            if start_ms >= 0 and end_ms == -1:
                state.current_start_ms = start_ms
            elif start_ms == -1 and end_ms >= 0:
                segment = [state.current_start_ms or 0, end_ms]
                state.finalized_segments.append(segment)
                new_segments.append(segment)
                state.current_start_ms = None
            elif start_ms >= 0 and end_ms >= 0:
                segment = [start_ms, end_ms]
                state.finalized_segments.append(segment)
                new_segments.append(segment)
                state.current_start_ms = None
        return new_segments
