from __future__ import annotations

"""Offline VAD + ASR pipeline."""

import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import numpy as np

from src.fsmn_vad_onnx import FsmnVadOnnxModel, FsmnVadStreamingState, VadResult
from src.models import ASROnnxStreamingState, Qwen3ASROnnxModel


DEFAULT_SAMPLE_RATE = 16000


@dataclass
class OfflineVadAsrSegment:
    start_ms: int
    end_ms: int
    vad_start_ms: int
    vad_end_ms: int
    language: str
    text: str
    raw_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    asr_elapsed_ms: float = 0.0


@dataclass
class OfflineVadAsrResult:
    sample_rate: int
    duration_ms: int
    vad: VadResult
    segments: list[OfflineVadAsrSegment]
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    vad_elapsed_ms: float = 0.0
    asr_elapsed_ms: float = 0.0


class OfflineVadAsrPipeline:
    """Runs offline VAD first, then ASR on each detected speech segment."""

    def __init__(
        self,
        vad_model: FsmnVadOnnxModel,
        asr_model: Qwen3ASROnnxModel,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        pad_start_ms: int = 120,
        pad_end_ms: int = 180,
        merge_gap_ms: int = 200,
        min_segment_ms: int = 200,
    ) -> None:
        self.vad_model = vad_model
        self.asr_model = asr_model
        self.sample_rate = int(sample_rate)
        self.pad_start_ms = max(0, int(pad_start_ms))
        self.pad_end_ms = max(0, int(pad_end_ms))
        self.merge_gap_ms = max(0, int(merge_gap_ms))
        self.min_segment_ms = max(0, int(min_segment_ms))

    def transcribe(
        self,
        audio: str | Path | np.ndarray,
        sample_rate: Optional[int] = None,
        context: str = "",
        language: Optional[str] = None,
        hotwords: Optional[str | Sequence[str]] = None,
        max_new_tokens: Optional[int] = None,
    ) -> OfflineVadAsrResult:
        wav, sr = self._load_audio(audio, sample_rate=sample_rate)
        vad_started_at = time.perf_counter()
        vad = self.vad_model.detect_offline(wav, sample_rate=sr)
        vad_elapsed_ms = self._elapsed_ms(vad_started_at)
        segments_ms = self._prepare_segments(vad.segments, duration_ms=self._duration_ms(wav))

        results: list[OfflineVadAsrSegment] = []
        asr_elapsed_total_ms = 0.0
        for segment in segments_ms:
            start_ms, end_ms, vad_start_ms, vad_end_ms = segment
            clip = self._slice_ms(wav, start_ms=start_ms, end_ms=end_ms)
            if clip.size == 0:
                continue
            asr_started_at = time.perf_counter()
            asr_result = self.asr_model.transcribe(
                audio=clip,
                context=context,
                language=language,
                hotwords=hotwords,
                max_new_tokens=max_new_tokens,
            )[0]
            asr_elapsed_ms = self._elapsed_ms(asr_started_at)
            asr_elapsed_total_ms += asr_elapsed_ms
            metadata = dict(asr_result.metadata)
            metadata["asr_elapsed_ms"] = asr_elapsed_ms
            results.append(
                OfflineVadAsrSegment(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    vad_start_ms=vad_start_ms,
                    vad_end_ms=vad_end_ms,
                    language=asr_result.language,
                    text=asr_result.text,
                    raw_text=asr_result.raw_text,
                    metadata=metadata,
                    asr_elapsed_ms=asr_elapsed_ms,
                )
            )

        return OfflineVadAsrResult(
            sample_rate=sr,
            duration_ms=self._duration_ms(wav),
            vad=vad,
            segments=results,
            text="".join(item.text for item in results),
            metadata={
                "speech_segment_count": len(results),
                "vad_segment_count": len(vad.segments),
                "pad_start_ms": self.pad_start_ms,
                "pad_end_ms": self.pad_end_ms,
                "merge_gap_ms": self.merge_gap_ms,
                "min_segment_ms": self.min_segment_ms,
                "vad_elapsed_ms": vad_elapsed_ms,
                "asr_elapsed_ms": asr_elapsed_total_ms,
                "total_elapsed_ms": vad_elapsed_ms + asr_elapsed_total_ms,
            },
            vad_elapsed_ms=vad_elapsed_ms,
            asr_elapsed_ms=asr_elapsed_total_ms,
        )

    def _load_audio(
        self,
        audio: str | Path | np.ndarray,
        sample_rate: Optional[int] = None,
    ) -> tuple[np.ndarray, int]:
        if isinstance(audio, (str, Path)):
            import librosa
            import soundfile as sf

            wav, sr = sf.read(str(audio), dtype="float32", always_2d=False)
            wav = self._to_mono_float32(wav)
            if int(sr) != self.sample_rate:
                wav = librosa.resample(wav, orig_sr=int(sr), target_sr=self.sample_rate).astype(np.float32)
            return wav, self.sample_rate

        if sample_rate is None:
            sample_rate = self.sample_rate
        wav = self._to_mono_float32(audio)
        if int(sample_rate) != self.sample_rate:
            import librosa

            wav = librosa.resample(wav, orig_sr=int(sample_rate), target_sr=self.sample_rate).astype(np.float32)
        return wav, self.sample_rate

    @staticmethod
    def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
        wav = np.asarray(audio)
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        if wav.dtype == np.int16:
            wav = wav.astype(np.float32) / 32768.0
        else:
            wav = wav.astype(np.float32, copy=False)
        return np.clip(wav.reshape(-1), -1.0, 1.0)

    def _prepare_segments(self, segments: list[list[int]], duration_ms: int) -> list[tuple[int, int, int, int]]:
        padded: list[tuple[int, int, int, int]] = []
        for segment in segments:
            if len(segment) < 2:
                continue
            vad_start_ms = max(0, int(segment[0]))
            vad_end_ms = min(duration_ms, int(segment[1]))
            if vad_end_ms - vad_start_ms < self.min_segment_ms:
                continue
            start_ms = max(0, vad_start_ms - self.pad_start_ms)
            end_ms = min(duration_ms, vad_end_ms + self.pad_end_ms)
            if end_ms > start_ms:
                padded.append((start_ms, end_ms, vad_start_ms, vad_end_ms))

        if not padded:
            return []

        merged: list[tuple[int, int, int, int]] = [padded[0]]
        for start_ms, end_ms, vad_start_ms, vad_end_ms in padded[1:]:
            last_start, last_end, last_vad_start, last_vad_end = merged[-1]
            if start_ms - last_end <= self.merge_gap_ms:
                merged[-1] = (
                    last_start,
                    max(last_end, end_ms),
                    last_vad_start,
                    max(last_vad_end, vad_end_ms),
                )
            else:
                merged.append((start_ms, end_ms, vad_start_ms, vad_end_ms))
        return merged

    def _slice_ms(self, wav: np.ndarray, start_ms: int, end_ms: int) -> np.ndarray:
        start = max(0, int(round(start_ms * self.sample_rate / 1000.0)))
        end = min(wav.shape[0], int(round(end_ms * self.sample_rate / 1000.0)))
        return np.asarray(wav[start:end], dtype=np.float32)

    def _duration_ms(self, wav: np.ndarray) -> int:
        return int(round(wav.shape[0] * 1000.0 / self.sample_rate))

    @staticmethod
    def _elapsed_ms(started_at: float) -> float:
        return (time.perf_counter() - started_at) * 1000.0


@dataclass(slots=True)
class StreamingVadAsrResult:
    event_type: Literal["speech_start", "partial", "final", "speech_end", "pipeline_done", "error"]
    utterance_id: int = -1
    result_id: int = -1
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    audio_end_ms: Optional[int] = None
    is_final: bool = False
    text: str = ""
    confirmed_delta_text: str = ""
    confirmed_text: str = ""
    pending_text: str = ""
    full_text: str = ""
    raw_text: str = ""
    language: Optional[str] = None
    vad_elapsed_ms: float = 0.0
    vad_elapsed_ms_total: float = 0.0
    asr_elapsed_ms: float = 0.0
    asr_elapsed_ms_total: float = 0.0
    emit_latency_ms: float = 0.0
    audio_queue_size: int = 0
    speech_queue_size: int = 0
    result_queue_size: int = 0
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def type(self) -> str:
        """兼容旧调用方的 event.type 写法。"""
        return self.event_type


StreamingVadAsrEvent = StreamingVadAsrResult


@dataclass
class StreamingVadAsrState:
    vad_state: FsmnVadStreamingState
    asr_state: Optional[ASROnnxStreamingState] = None
    vad_in_speech: bool = False
    vad_start_ms: Optional[int] = None
    asr_start_ms: Optional[int] = None
    current_utterance_id: int = -1
    next_utterance_id: int = 0
    next_result_id: int = 0
    last_partial_text: str = ""
    finalized_segments: list[StreamingVadAsrResult] = field(default_factory=list)
    vad_elapsed_ms_total: float = 0.0
    asr_elapsed_ms_total: float = 0.0


@dataclass
class _AudioQueueItem:
    audio: np.ndarray
    timestamp_ms: int
    pushed_at: float


@dataclass
class _SpeechQueueEvent:
    type: str
    audio: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    timestamp_ms: Optional[int] = None
    audio_pushed_at: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class StreamingVadAsrPipeline:
    """Producer-consumer streaming pipeline: audio -> VAD worker -> ASR worker -> result events."""

    def __init__(
        self,
        vad_model: FsmnVadOnnxModel,
        asr_model: Qwen3ASROnnxModel,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        vad_chunk_ms: int = 200,
        asr_chunk_size_sec: float = 1.0,
        asr_max_audio_history_sec: Optional[float] = 30.0,
        pre_roll_ms: int = 300,
        context: str = "",
        language: Optional[str] = None,
        hotwords: Optional[str | Sequence[str]] = None,
        max_new_tokens: Optional[int] = None,
        unfixed_chunk_num: int = 2,
        unfixed_token_num: int = 5,
        audio_queue_size: int = 32,
        speech_queue_size: int = 32,
        result_queue_size: int = 64,
    ) -> None:
        self.vad_model = vad_model
        self.asr_model = asr_model
        self.sample_rate = int(sample_rate)
        self.vad_chunk_ms = int(vad_chunk_ms)
        self.asr_chunk_size_sec = float(asr_chunk_size_sec)
        self.asr_max_audio_history_sec = asr_max_audio_history_sec
        self.pre_roll_ms = max(0, int(pre_roll_ms))
        self.context = str(context or "")
        self.language = language
        self.hotwords = hotwords
        self.max_new_tokens = max_new_tokens
        self.unfixed_chunk_num = int(unfixed_chunk_num)
        self.unfixed_token_num = int(unfixed_token_num)

        self.audio_queue: queue.Queue[_AudioQueueItem | None] = queue.Queue(maxsize=int(audio_queue_size))
        self.speech_queue: queue.Queue[_SpeechQueueEvent | None] = queue.Queue(maxsize=int(speech_queue_size))
        self.result_queue: queue.Queue[StreamingVadAsrResult | None] = queue.Queue(maxsize=int(result_queue_size))
        self.state = StreamingVadAsrState(vad_state=self.vad_model.init_streaming_state(chunk_size_ms=self.vad_chunk_ms))
        self._threads: list[threading.Thread] = []
        self._started = False
        self._finished = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._threads = [
            threading.Thread(target=self._vad_worker, name="vad-worker", daemon=True),
            threading.Thread(target=self._asr_worker, name="asr-worker", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def push_audio(self, audio: np.ndarray, timestamp_ms: int) -> None:
        if not self._started:
            self.start()
        if self._finished:
            raise RuntimeError("cannot push audio after finish()")
        self.audio_queue.put(
            _AudioQueueItem(
                audio=self._normalize_pcm(audio),
                timestamp_ms=int(timestamp_ms),
                pushed_at=time.perf_counter(),
            )
        )

    def read_result(self, timeout: Optional[float] = None) -> Optional[StreamingVadAsrResult]:
        item = self.result_queue.get(timeout=timeout)
        return item

    def finish(self) -> None:
        if not self._started:
            self.start()
        if self._finished:
            return
        self._finished = True
        self.audio_queue.put(None)

    def join(self) -> None:
        for thread in self._threads:
            thread.join()

    def stop(self) -> None:
        if not self._finished:
            self._finished = True
            self.audio_queue.put(None)
        self.join()

    def _make_result(
        self,
        event_type: Literal["speech_start", "partial", "final", "speech_end", "pipeline_done", "error"],
        *,
        utterance_id: Optional[int] = None,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        audio_end_ms: Optional[int] = None,
        is_final: bool = False,
        text: str = "",
        confirmed_delta_text: str = "",
        confirmed_text: str = "",
        pending_text: str = "",
        full_text: str = "",
        raw_text: str = "",
        language: Optional[str] = None,
        vad_elapsed_ms: Optional[float] = None,
        vad_elapsed_ms_total: Optional[float] = None,
        asr_elapsed_ms: Optional[float] = None,
        asr_elapsed_ms_total: Optional[float] = None,
        audio_pushed_at: Optional[float] = None,
        error: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> StreamingVadAsrResult:
        metadata = dict(metadata or {})
        result_id = self.state.next_result_id
        self.state.next_result_id += 1
        resolved_utterance_id = self.state.current_utterance_id if utterance_id is None else int(utterance_id)
        latency_ms = 0.0
        if audio_pushed_at is not None:
            latency_ms = self._elapsed_ms(float(audio_pushed_at))
        vad_ms = float(metadata.get("vad_elapsed_ms", 0.0) if vad_elapsed_ms is None else vad_elapsed_ms)
        vad_total = float(
            metadata.get("vad_elapsed_ms_total", self.state.vad_elapsed_ms_total)
            if vad_elapsed_ms_total is None
            else vad_elapsed_ms_total
        )
        asr_ms = float(metadata.get("asr_elapsed_ms", 0.0) if asr_elapsed_ms is None else asr_elapsed_ms)
        asr_total = float(
            metadata.get("asr_elapsed_ms_total", self.state.asr_elapsed_ms_total)
            if asr_elapsed_ms_total is None
            else asr_elapsed_ms_total
        )
        metadata.update(
            {
                "utterance_id": resolved_utterance_id,
                "result_id": result_id,
                "full_text_chars": len(full_text or text or ""),
                "confirmed_delta_chars": len(confirmed_delta_text or ""),
                "confirmed_text_chars": len(confirmed_text or ""),
                "pending_chars": len(pending_text or ""),
                "emit_latency_ms": latency_ms,
                "audio_queue_size": self.audio_queue.qsize(),
                "speech_queue_size": self.speech_queue.qsize(),
                "result_queue_size": self.result_queue.qsize(),
            }
        )
        return StreamingVadAsrResult(
            event_type=event_type,
            utterance_id=resolved_utterance_id,
            result_id=result_id,
            start_ms=start_ms,
            end_ms=end_ms,
            audio_end_ms=audio_end_ms,
            is_final=is_final,
            text=text,
            confirmed_delta_text=confirmed_delta_text,
            confirmed_text=confirmed_text,
            pending_text=pending_text,
            full_text=full_text,
            raw_text=raw_text,
            language=language,
            vad_elapsed_ms=vad_ms,
            vad_elapsed_ms_total=vad_total,
            asr_elapsed_ms=asr_ms,
            asr_elapsed_ms_total=asr_total,
            emit_latency_ms=latency_ms,
            audio_queue_size=self.audio_queue.qsize(),
            speech_queue_size=self.speech_queue.qsize(),
            result_queue_size=self.result_queue.qsize(),
            error=error,
            metadata=metadata,
        )

    def _vad_worker(self) -> None:
        pre_roll = np.zeros((0,), dtype=np.float32)
        pre_roll_samples = max(0, int(round(self.pre_roll_ms * self.sample_rate / 1000.0)))
        try:
            while True:
                item = self.audio_queue.get()
                if item is None:
                    self._finish_vad_stream()
                    self.speech_queue.put(None)
                    return

                was_in_speech = self.state.vad_in_speech
                vad_started_at = time.perf_counter()
                result = self.vad_model.detect_streaming(item.audio, self.state.vad_state, is_final=False)
                vad_elapsed_ms = self._elapsed_ms(vad_started_at)
                self.state.vad_elapsed_ms_total += vad_elapsed_ms
                chunk_end_ms = self._chunk_end_ms(item.timestamp_ms, item.audio)
                vad_metadata = {
                    "vad_elapsed_ms": vad_elapsed_ms,
                    "vad_elapsed_ms_total": self.state.vad_elapsed_ms_total,
                    "audio_timestamp_ms": item.timestamp_ms,
                    "vad_input_samples": int(item.audio.shape[0]),
                    "vad_input_ms": int(round(item.audio.shape[0] * 1000.0 / self.sample_rate)),
                    "vad_chunk_start_ms": int(item.timestamp_ms),
                    "vad_chunk_end_ms": int(chunk_end_ms),
                }
                fed_current_chunk = False
                ended_this_chunk = False

                for event in result.events:
                    start_ms, end_ms = event
                    if start_ms >= 0 and end_ms == -1:
                        if self.state.vad_in_speech:
                            continue
                        self.state.vad_in_speech = True
                        self.state.vad_start_ms = start_ms
                        start_audio = self._concat_audio(pre_roll, item.audio)
                        metadata = dict(vad_metadata)
                        metadata.update(
                            {
                                "speech_event_type": "speech_start",
                                "speech_input_samples": int(start_audio.shape[0]),
                                "speech_input_ms": int(round(start_audio.shape[0] * 1000.0 / self.sample_rate)),
                                "pre_roll_samples": int(pre_roll.shape[0]),
                                "pre_roll_ms": int(round(pre_roll.shape[0] * 1000.0 / self.sample_rate)),
                            }
                        )
                        self.speech_queue.put(
                            _SpeechQueueEvent(
                                type="speech_start",
                                audio=start_audio,
                                start_ms=start_ms,
                                timestamp_ms=item.timestamp_ms,
                                audio_pushed_at=item.pushed_at,
                                metadata=metadata,
                            )
                        )
                        fed_current_chunk = True
                    elif start_ms == -1 and end_ms >= 0:
                        end_ms = self._normalize_vad_end_ms(end_ms, chunk_end_ms)
                        if self.state.vad_in_speech or was_in_speech:
                            if not fed_current_chunk:
                                metadata = dict(vad_metadata)
                                metadata.update(
                                    {
                                        "speech_event_type": "speech_chunk",
                                        "speech_input_samples": int(item.audio.shape[0]),
                                        "speech_input_ms": int(round(item.audio.shape[0] * 1000.0 / self.sample_rate)),
                                    }
                                )
                                self.speech_queue.put(
                                    _SpeechQueueEvent(
                                        type="speech_chunk",
                                        audio=item.audio,
                                        timestamp_ms=item.timestamp_ms,
                                        audio_pushed_at=item.pushed_at,
                                        metadata=metadata,
                                    )
                                )
                                fed_current_chunk = True
                            metadata = dict(vad_metadata)
                            metadata["speech_event_type"] = "speech_end"
                            self.speech_queue.put(
                                _SpeechQueueEvent(
                                    type="speech_end",
                                    end_ms=end_ms,
                                    timestamp_ms=item.timestamp_ms,
                                    audio_pushed_at=item.pushed_at,
                                    metadata=metadata,
                                )
                            )
                            ended_this_chunk = True
                        self.state.vad_in_speech = False
                        self.state.vad_start_ms = None
                    elif start_ms >= 0 and end_ms >= 0:
                        end_ms = self._normalize_vad_end_ms(end_ms, chunk_end_ms, start_ms=start_ms)
                        if self.state.vad_in_speech or was_in_speech:
                            if not fed_current_chunk:
                                metadata = dict(vad_metadata)
                                metadata.update(
                                    {
                                        "speech_event_type": "speech_chunk",
                                        "speech_input_samples": int(item.audio.shape[0]),
                                        "speech_input_ms": int(round(item.audio.shape[0] * 1000.0 / self.sample_rate)),
                                    }
                                )
                                self.speech_queue.put(
                                    _SpeechQueueEvent(
                                        type="speech_chunk",
                                        audio=item.audio,
                                        timestamp_ms=item.timestamp_ms,
                                        audio_pushed_at=item.pushed_at,
                                        metadata=metadata,
                                    )
                                )
                                fed_current_chunk = True
                            metadata = dict(vad_metadata)
                            metadata["speech_event_type"] = "speech_end"
                            self.speech_queue.put(
                                _SpeechQueueEvent(
                                    type="speech_end",
                                    end_ms=end_ms,
                                    timestamp_ms=item.timestamp_ms,
                                    audio_pushed_at=item.pushed_at,
                                    metadata=metadata,
                                )
                            )
                        else:
                            start_audio = self._concat_audio(pre_roll, item.audio)
                            metadata = dict(vad_metadata)
                            metadata.update(
                                {
                                    "speech_event_type": "speech_start",
                                    "speech_input_samples": int(start_audio.shape[0]),
                                    "speech_input_ms": int(round(start_audio.shape[0] * 1000.0 / self.sample_rate)),
                                    "pre_roll_samples": int(pre_roll.shape[0]),
                                    "pre_roll_ms": int(round(pre_roll.shape[0] * 1000.0 / self.sample_rate)),
                                }
                            )
                            self.speech_queue.put(
                                _SpeechQueueEvent(
                                    type="speech_start",
                                    audio=start_audio,
                                    start_ms=start_ms,
                                    timestamp_ms=item.timestamp_ms,
                                    audio_pushed_at=item.pushed_at,
                                    metadata=metadata,
                                )
                            )
                            fed_current_chunk = True
                            metadata = dict(vad_metadata)
                            metadata["speech_event_type"] = "speech_end"
                            self.speech_queue.put(
                                _SpeechQueueEvent(
                                    type="speech_end",
                                    end_ms=end_ms,
                                    timestamp_ms=item.timestamp_ms,
                                    audio_pushed_at=item.pushed_at,
                                    metadata=metadata,
                                )
                            )
                        self.state.vad_in_speech = False
                        self.state.vad_start_ms = None
                        ended_this_chunk = True

                if self.state.vad_in_speech and not fed_current_chunk:
                    metadata = dict(vad_metadata)
                    metadata.update(
                        {
                            "speech_event_type": "speech_chunk",
                            "speech_input_samples": int(item.audio.shape[0]),
                            "speech_input_ms": int(round(item.audio.shape[0] * 1000.0 / self.sample_rate)),
                        }
                    )
                    self.speech_queue.put(
                        _SpeechQueueEvent(
                            type="speech_chunk",
                            audio=item.audio,
                            timestamp_ms=item.timestamp_ms,
                            audio_pushed_at=item.pushed_at,
                            metadata=metadata,
                        )
                    )
                if self.state.vad_in_speech and not ended_this_chunk:
                    pre_roll = np.zeros((0,), dtype=np.float32)
                else:
                    pre_roll = self._trim_preroll(self._concat_audio(pre_roll, item.audio), pre_roll_samples)
        except Exception as exc:
            self.result_queue.put(self._make_result("error", error=repr(exc), metadata={"worker": "vad"}))
            self.speech_queue.put(None)

    def _finish_vad_stream(self) -> None:
        vad_started_at = time.perf_counter()
        result = self.vad_model.finish_streaming(self.state.vad_state)
        vad_elapsed_ms = self._elapsed_ms(vad_started_at)
        self.state.vad_elapsed_ms_total += vad_elapsed_ms
        vad_metadata = {
            "vad_elapsed_ms": vad_elapsed_ms,
            "vad_elapsed_ms_total": self.state.vad_elapsed_ms_total,
            "vad_is_final": True,
        }
        emitted_end = False
        stream_end_ms = int(round(self.state.vad_state.total_ms))
        for event in result.events:
            start_ms, end_ms = event
            if start_ms >= 0 and end_ms == -1:
                if not self.state.vad_in_speech:
                    self.state.vad_in_speech = True
                    self.state.vad_start_ms = start_ms
                    self.speech_queue.put(
                        _SpeechQueueEvent(type="speech_start", start_ms=start_ms, metadata=vad_metadata)
                    )
            elif start_ms == -1 and end_ms >= 0:
                if self.state.vad_start_ms is not None and int(end_ms) < int(self.state.vad_start_ms):
                    end_ms = stream_end_ms
                self.speech_queue.put(_SpeechQueueEvent(type="speech_end", end_ms=end_ms, metadata=vad_metadata))
                self.state.vad_in_speech = False
                self.state.vad_start_ms = None
                emitted_end = True
            elif start_ms >= 0 and end_ms >= 0:
                if self.state.vad_start_ms is not None and int(end_ms) < int(self.state.vad_start_ms):
                    end_ms = stream_end_ms
                if self.state.vad_in_speech:
                    self.speech_queue.put(_SpeechQueueEvent(type="speech_end", end_ms=end_ms, metadata=vad_metadata))
                else:
                    self.speech_queue.put(
                        _SpeechQueueEvent(type="speech_start", start_ms=start_ms, metadata=vad_metadata)
                    )
                    self.speech_queue.put(_SpeechQueueEvent(type="speech_end", end_ms=end_ms, metadata=vad_metadata))
                self.state.vad_in_speech = False
                self.state.vad_start_ms = None
                emitted_end = True
        if self.state.vad_in_speech and not emitted_end:
            self.speech_queue.put(
                _SpeechQueueEvent(type="speech_end", end_ms=stream_end_ms, metadata=vad_metadata)
            )
            self.state.vad_in_speech = False
            self.state.vad_start_ms = None

    def _asr_worker(self) -> None:
        try:
            while True:
                event = self.speech_queue.get()
                if event is None:
                    self._finalize_asr(end_ms=None)
                    self.result_queue.put(
                        self._make_result(
                            "pipeline_done",
                            is_final=True,
                            vad_elapsed_ms_total=self.state.vad_elapsed_ms_total,
                            asr_elapsed_ms_total=self.state.asr_elapsed_ms_total,
                        )
                    )
                    self.result_queue.put(None)
                    return

                if event.type == "speech_start":
                    self._start_asr(
                        start_ms=event.start_ms,
                        timestamp_ms=event.timestamp_ms,
                        audio_pushed_at=event.audio_pushed_at,
                        vad_metadata=event.metadata,
                    )
                    if event.audio.size > 0:
                        self._feed_asr(event.audio, event.timestamp_ms, event.metadata, event.audio_pushed_at)
                elif event.type == "speech_chunk":
                    self._feed_asr(event.audio, event.timestamp_ms, event.metadata, event.audio_pushed_at)
                elif event.type == "speech_end":
                    self._finalize_asr(
                        end_ms=event.end_ms,
                        vad_metadata=event.metadata,
                        audio_pushed_at=event.audio_pushed_at,
                    )
        except Exception as exc:
            self.result_queue.put(self._make_result("error", error=repr(exc), metadata={"worker": "asr"}))
            self.result_queue.put(None)

    def _start_asr(
        self,
        start_ms: Optional[int],
        timestamp_ms: Optional[int] = None,
        audio_pushed_at: Optional[float] = None,
        vad_metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self.state.asr_state = self.asr_model.init_streaming_state(
            context=self.context,
            language=self.language,
            hotwords=self.hotwords,
            unfixed_chunk_num=self.unfixed_chunk_num,
            unfixed_token_num=self.unfixed_token_num,
            chunk_size_sec=self.asr_chunk_size_sec,
            max_audio_history_sec=self.asr_max_audio_history_sec,
        )
        self.state.asr_start_ms = start_ms
        self.state.current_utterance_id = self.state.next_utterance_id
        self.state.next_utterance_id += 1
        self.state.last_partial_text = ""
        metadata = dict(vad_metadata or {})
        self.result_queue.put(
            self._make_result(
                "speech_start",
                start_ms=start_ms,
                end_ms=start_ms,
                audio_end_ms=timestamp_ms,
                vad_elapsed_ms=metadata.get("vad_elapsed_ms", 0.0),
                vad_elapsed_ms_total=metadata.get("vad_elapsed_ms_total", self.state.vad_elapsed_ms_total),
                audio_pushed_at=audio_pushed_at,
                metadata=metadata,
            )
        )

    def _feed_asr(
        self,
        audio: np.ndarray,
        timestamp_ms: Optional[int],
        vad_metadata: Optional[dict[str, Any]] = None,
        audio_pushed_at: Optional[float] = None,
    ) -> None:
        if self.state.asr_state is None:
            self._start_asr(
                start_ms=timestamp_ms,
                timestamp_ms=timestamp_ms,
                audio_pushed_at=audio_pushed_at,
                vad_metadata=vad_metadata,
            )
        asr_started_at = time.perf_counter()
        self.asr_model.streaming_transcribe(audio, self.state.asr_state, max_new_tokens=self.max_new_tokens)
        asr_elapsed_ms = self._elapsed_ms(asr_started_at)
        self.state.asr_elapsed_ms_total += asr_elapsed_ms
        full_text = self.state.asr_state.text
        confirmed_delta_text = self.state.asr_state.confirmed_delta_text
        pending_text = self.state.asr_state.pending_text
        confirmed_text = self.state.asr_state.confirmed_text
        stream_text_revision = bool(self.state.asr_state.metadata.get("stream_text_revision"))
        if confirmed_delta_text or full_text != self.state.last_partial_text or stream_text_revision:
            self.state.last_partial_text = full_text
            metadata = dict(self.state.asr_state.metadata)
            metadata.update(vad_metadata or {})
            metadata["asr_elapsed_ms"] = asr_elapsed_ms
            metadata["asr_elapsed_ms_total"] = self.state.asr_elapsed_ms_total
            metadata["full_text_chars"] = len(full_text)
            local_end_ms = metadata.get("trigger_chunk_end_ms", metadata.get("audio_end_ms"))
            end_ms = None
            if self.state.asr_start_ms is not None and local_end_ms is not None:
                end_ms = int(self.state.asr_start_ms) + int(local_end_ms)
            self.result_queue.put(
                self._make_result(
                    "partial",
                    start_ms=self.state.asr_start_ms,
                    end_ms=end_ms,
                    audio_end_ms=int(local_end_ms) if local_end_ms is not None else end_ms,
                    is_final=False,
                    text=full_text,
                    confirmed_delta_text=confirmed_delta_text,
                    confirmed_text=confirmed_text,
                    pending_text=pending_text,
                    full_text=full_text,
                    language=self.state.asr_state.language,
                    raw_text=self.state.asr_state.committed_raw_text,
                    vad_elapsed_ms=metadata.get("vad_elapsed_ms", 0.0),
                    vad_elapsed_ms_total=metadata.get("vad_elapsed_ms_total", self.state.vad_elapsed_ms_total),
                    asr_elapsed_ms=asr_elapsed_ms,
                    asr_elapsed_ms_total=self.state.asr_elapsed_ms_total,
                    audio_pushed_at=audio_pushed_at,
                    metadata=metadata,
                )
            )

    def _finalize_asr(
        self,
        end_ms: Optional[int],
        vad_metadata: Optional[dict[str, Any]] = None,
        audio_pushed_at: Optional[float] = None,
    ) -> None:
        if self.state.asr_state is None:
            return
        if end_ms is not None:
            self._trim_asr_to_vad_end(end_ms)
        asr_started_at = time.perf_counter()
        self.asr_model.finish_streaming_transcribe(self.state.asr_state, max_new_tokens=self.max_new_tokens)
        asr_elapsed_ms = self._elapsed_ms(asr_started_at)
        self.state.asr_elapsed_ms_total += asr_elapsed_ms
        if end_ms is not None and self.state.asr_start_ms is not None:
            if int(end_ms) < int(self.state.asr_start_ms):
                end_ms = max(
                    int(self.state.asr_start_ms),
                    int(round(self.state.vad_state.total_ms)),
                )
            else:
                end_ms = int(end_ms)
        metadata = dict(self.state.asr_state.metadata)
        metadata.update(vad_metadata or {})
        metadata["asr_elapsed_ms"] = asr_elapsed_ms
        metadata["asr_elapsed_ms_total"] = self.state.asr_elapsed_ms_total
        metadata["full_text_chars"] = len(self.state.asr_state.text)
        local_audio_end_ms = metadata.get("audio_end_ms")
        event = self._make_result(
            "final",
            start_ms=self.state.asr_start_ms,
            end_ms=end_ms,
            audio_end_ms=int(local_audio_end_ms) if local_audio_end_ms is not None else end_ms,
            is_final=True,
            text=self.state.asr_state.text,
            confirmed_delta_text=self.state.asr_state.confirmed_delta_text,
            confirmed_text=self.state.asr_state.confirmed_text,
            pending_text=self.state.asr_state.pending_text,
            full_text=self.state.asr_state.text,
            language=self.state.asr_state.language,
            raw_text=self.state.asr_state.raw_text,
            vad_elapsed_ms=metadata.get("vad_elapsed_ms", 0.0),
            vad_elapsed_ms_total=metadata.get("vad_elapsed_ms_total", self.state.vad_elapsed_ms_total),
            asr_elapsed_ms=asr_elapsed_ms,
            asr_elapsed_ms_total=self.state.asr_elapsed_ms_total,
            audio_pushed_at=audio_pushed_at,
            metadata=metadata,
        )
        self.state.finalized_segments.append(event)
        self.result_queue.put(event)
        self.state.asr_state = None
        self.state.asr_start_ms = None
        self.state.current_utterance_id = -1
        self.state.last_partial_text = ""

    def _trim_asr_to_vad_end(self, end_ms: int) -> None:
        state = self.state.asr_state
        if state is None or self.state.asr_start_ms is None:
            return
        end_ms = int(end_ms)
        start_ms = int(self.state.asr_start_ms)
        if end_ms < start_ms:
            return

        target_samples = max(0, int(round((end_ms - start_ms) * self.sample_rate / 1000.0)))
        total_samples = int(state.total_audio_samples)
        if target_samples < total_samples:
            drop_samples = total_samples - target_samples
            drop_from_accum = min(drop_samples, int(state.audio_accum_samples))
            if drop_from_accum > 0:
                state.audio_accum_samples -= drop_from_accum
            state.total_audio_samples = target_samples
            state.last_appended_end_samples = min(int(state.last_appended_end_samples), target_samples)
            state.last_appended_start_samples = min(
                int(state.last_appended_start_samples),
                int(state.last_appended_end_samples),
            )
            state.buffer = np.zeros((0,), dtype=np.float32)
            state.audio_accum = self.asr_model._stream_audio_view(state)
            return

        buffered_target = target_samples - total_samples
        if buffered_target < int(state.buffer.shape[0]):
            state.buffer = np.asarray(state.buffer[:buffered_target], dtype=np.float32)

    @staticmethod
    def _elapsed_ms(started_at: float) -> float:
        return (time.perf_counter() - started_at) * 1000.0

    @staticmethod
    def _normalize_pcm(audio: np.ndarray) -> np.ndarray:
        wav = np.asarray(audio)
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        if wav.dtype == np.int16:
            wav = wav.astype(np.float32) / 32768.0
        else:
            wav = wav.astype(np.float32, copy=False)
        return np.clip(wav.reshape(-1), -1.0, 1.0)

    @staticmethod
    def _concat_audio(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        if left.size == 0:
            return np.asarray(right, dtype=np.float32).reshape(-1)
        if right.size == 0:
            return np.asarray(left, dtype=np.float32).reshape(-1)
        return np.concatenate([left.reshape(-1), right.reshape(-1)]).astype(np.float32, copy=False)

    @staticmethod
    def _trim_preroll(audio: np.ndarray, max_samples: int) -> np.ndarray:
        if max_samples <= 0 or audio.size == 0:
            return np.zeros((0,), dtype=np.float32)
        return np.asarray(audio[-max_samples:], dtype=np.float32)

    def _chunk_end_ms(self, timestamp_ms: int, audio: np.ndarray) -> int:
        duration_ms = int(round(np.asarray(audio).reshape(-1).shape[0] * 1000.0 / self.sample_rate))
        return int(timestamp_ms) + max(0, duration_ms)

    def _normalize_vad_end_ms(
        self,
        end_ms: int,
        chunk_end_ms: int,
        start_ms: Optional[int] = None,
    ) -> int:
        end_ms = int(end_ms)
        if self.state.vad_start_ms is not None:
            start_ms = self.state.vad_start_ms
        if start_ms is not None and end_ms < int(start_ms):
            return max(int(start_ms), int(chunk_end_ms))
        return end_ms
