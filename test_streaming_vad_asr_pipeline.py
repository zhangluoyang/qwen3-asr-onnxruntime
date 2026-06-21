from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from src.fsmn_vad_onnx import FsmnVadOnnxModel
from src.models import Qwen3ASROnnxModel
from src.pipeline import StreamingVadAsrPipeline


# ======================== 默认运行配置 ========================
# 直接运行本文件即可使用下面这些配置：
#     python test_streaming_vad_asr_pipeline.py
#
# 这个脚本用 wav 文件模拟实时音频生产者：每 200ms 推一个 PCM chunk。

# Qwen3-ASR 原始模型目录。ONNX 推理仍需要从这里加载 processor / tokenizer，
# 用于 chat template、音频特征提取、文本 decode 和 ASR 输出解析。
ASR_MODEL_PATH = Path("/home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-ASR-0.6B")

# Qwen3-ASR ONNX 导出目录，应包含 audio_encoder、token_embedding、text_core 三个子目录。
ASR_ONNX_DIR = Path("./onnx_asr")

# FSMN-VAD 原始模型目录。当前 VAD 封装会复用原模型配置、frontend 和后处理逻辑。
VAD_MODEL_DIR = Path("./fsmn")

# FSMN-VAD encoder 的 ONNX 文件路径。
VAD_ONNX_PATH = Path("./onnx_fsmn/fsmn_vad_vad_encoder.onnx")

# 用来模拟实时输入的测试音频。脚本会读取 wav，再按 VAD_CHUNK_MS 切块推入 pipeline。
AUDIO_PATH = Path("./data/voice_design_stream.wav")

# ASR 使用的 ONNXRuntime provider。cuda 表示优先 CUDAExecutionProvider，并保留 CPU fallback。
ASR_PROVIDERS = "cuda"

# VAD 默认使用 CPU。FSMN-VAD 计算量较小，放 CPU 可以减少 GPU 资源占用。
VAD_PROVIDERS = "cpu"

# 最多读取测试音频前多少秒；<= 0 表示读取完整音频。
MAX_AUDIO_SECONDS = 3600

# VAD 流式切块大小，也是脚本模拟实时推流时每次 push 的音频长度。
# 值越小，VAD 触发更及时，但调度次数更多；值越大，端到端延迟会增加。
VAD_CHUNK_MS = 250

# 默认按真实音频时间推流。用 wav 文件测试时，每推一个 chunk 后等待到对应时间点，
# 这样日志和真实麦克风输入一样，不会一口气把整段音频塞进队列。
REALTIME_PACING = True

# ASR 伪流式每累计多少秒语音触发一次识别。
# 当前 ASR 不复用 audio encoder cache，而是重跑最近音频窗口，所以太小会增加重复计算。
ASR_CHUNK_SIZE_SEC = 0.5

# ASR 每次重算时最多保留最近多少秒历史语音；<= 0 表示保留全部历史。
# 限制历史窗口可以防止长句子里每次重算越来越慢。
ASR_MAX_AUDIO_HISTORY_SEC = 10

# VAD 检测到语音开始后，额外向前保留多少毫秒音频。
# 这样可以降低句首被 VAD 切掉的概率。
PRE_ROLL_MS = 300

# 强制识别语言；None 表示让 Qwen3-ASR 自行识别语言。
# 已知语言时可填 "Chinese"、"English" 等，减少语言判断的不确定性。
LANGUAGE = None

# ASR system prompt。可放业务上下文，但应避免写成让模型改写或总结的指令。
CONTEXT = ""

# 热词提示，会被追加到 system prompt 中；多个热词用英文逗号分隔。
# 只应作为识别偏置，不能期待模型在无声学证据时强行输出热词。
HOTWORDS = "小医仙,萧炎,斗破苍穹"

# 单次 ASR 生成的最大 token 数。太小可能截断，太大可能让异常输入生成过长。
MAX_NEW_TOKENS = 1024

# 伪流式输出时，末尾保留多少个 chunk 不固定，等待后续音频帮助确认边界文本。
# 值越大越稳定，但 partial 输出延迟越高。
UNFIXED_CHUNK_NUM = 2

# 伪流式输出时，末尾保留多少个 token 不提交，降低末尾文字反复变化的概率。
# 值越大越保守，但 partial 文本会更慢变长。
UNFIXED_TOKEN_NUM = 5

# 推送长音频时每隔多少秒打印一次进度，避免误以为程序卡住。
PROGRESS_INTERVAL_SEC = 60.0


def resolve_providers(name: str) -> list[str] | None:
    if name == "auto":
        return None
    if name == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def load_audio(audio_path: Path, max_audio_seconds: float) -> tuple[np.ndarray, int]:
    import librosa
    import soundfile as sf

    info = sf.info(str(audio_path))
    frames = -1 if max_audio_seconds <= 0 else int(round(float(max_audio_seconds) * int(info.samplerate)))
    wav, sample_rate = sf.read(str(audio_path), frames=frames, dtype="float32", always_2d=False)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1).astype(np.float32)
    if int(sample_rate) != 16000:
        wav = librosa.resample(wav, orig_sr=int(sample_rate), target_sr=16000).astype(np.float32)
        sample_rate = 16000
    return np.clip(wav.reshape(-1), -1.0, 1.0), int(sample_rate)


def validate_paths() -> None:
    missing = [path for path in (ASR_MODEL_PATH, ASR_ONNX_DIR, VAD_ONNX_PATH, AUDIO_PATH) if not path.exists()]
    if missing:
        raise FileNotFoundError("missing test file(s): " + ", ".join(str(path) for path in missing))


def format_timing(metadata: dict) -> str:
    vad_ms = float(metadata.get("vad_elapsed_ms", 0.0))
    asr_ms = float(metadata.get("asr_elapsed_ms", 0.0))
    vad_total_ms = float(metadata.get("vad_elapsed_ms_total", 0.0))
    asr_total_ms = float(metadata.get("asr_elapsed_ms_total", 0.0))
    return (
        f"vad_ms={vad_ms:.3f} asr_ms={asr_ms:.3f} "
        f"vad_total_ms={vad_total_ms:.3f} asr_total_ms={asr_total_ms:.3f}"
    )


def format_metadata_fields(metadata: dict) -> str:
    fields = [
        "vad_input_samples",
        "vad_input_ms",
        "vad_chunk_start_ms",
        "vad_chunk_end_ms",
        "speech_event_type",
        "speech_input_samples",
        "speech_input_ms",
        "pre_roll_samples",
        "pre_roll_ms",
        "input_samples",
        "input_ms",
        "buffer_samples_before",
        "buffer_samples_after_append",
        "buffer_samples",
        "triggered_decode_count",
        "decoded_chunk_id",
        "processed_chunks",
        "trigger_chunk_start_ms",
        "trigger_chunk_end_ms",
        "audio_accum_samples",
        "asr_window_samples",
        "asr_window_ms",
        "audio_window_start_ms",
        "audio_window_end_ms",
        "total_audio_samples",
        "max_audio_history_samples",
        "last_trimmed_samples",
        "total_trimmed_samples",
        "trim_events",
        "prefix_chars",
        "committed_raw_chars",
        "uncommitted_raw_chars",
        "confirmed_delta_chars",
        "confirmed_text_chars",
        "pending_chars",
        "delta_raw_chars",
        "emitted_text_chars",
        "stream_text_revision",
        "stream_raw_revision",
        "finalized",
    ]
    parts = []
    for key in fields:
        if key in metadata:
            value = metadata[key]
            if isinstance(value, bool):
                value = int(value)
            parts.append(f"{key}={value!r}")
    return " ".join(parts)


def format_result_state(event) -> str:
    metadata_text = format_metadata_fields(event.metadata)
    if metadata_text:
        metadata_text = " " + metadata_text
    return (
        f"utt={event.utterance_id} rid={event.result_id} "
        f"start={event.start_ms}ms end={event.end_ms}ms audio_end={event.audio_end_ms}ms "
        f"vad_ms={event.vad_elapsed_ms:.3f} asr_ms={event.asr_elapsed_ms:.3f} "
        f"vad_total_ms={event.vad_elapsed_ms_total:.3f} asr_total_ms={event.asr_elapsed_ms_total:.3f} "
        f"latency_ms={event.emit_latency_ms:.3f} "
        f"q={event.audio_queue_size}/{event.speech_queue_size}/{event.result_queue_size}"
        f"{metadata_text}"
    )


def print_event(event) -> None:
    if event.type == "speech_start":
        print(f"[speech_start] {format_result_state(event)}", flush=True)
    elif event.type == "partial":
        print(
            f"[partial] {format_result_state(event)} "
            f"confirmed_delta={event.confirmed_delta_text!r} "
            f"confirmed_text={event.confirmed_text!r} "
            f"pending={event.pending_text!r} "
            f"full_chars={len(event.full_text)}",
            flush=True,
        )
    elif event.type == "final":
        print(
            f"[final] {format_result_state(event)} "
            f"full_chars={len(event.full_text)} "
            f"language={event.language!r} "
            f"confirmed_delta={event.confirmed_delta_text!r} "
            f"confirmed_text={event.confirmed_text!r} "
            f"pending={event.pending_text!r} "
            f"text={event.text!r}",
            flush=True,
        )
    elif event.type == "error":
        print(f"[error] {format_result_state(event)} error={event.error!r} metadata={event.metadata}", flush=True)
    elif event.type == "pipeline_done":
        print(f"[pipeline_done] {format_result_state(event)}", flush=True)


def drain_results(pipeline: StreamingVadAsrPipeline) -> None:
    while True:
        event = pipeline.read_result(timeout=None)
        if event is None:
            break
        print_event(event)


def main() -> None:
    validate_paths()
    wav, sample_rate = load_audio(AUDIO_PATH, MAX_AUDIO_SECONDS)

    vad_model = FsmnVadOnnxModel(
        model_dir=VAD_MODEL_DIR,
        onnx_path=VAD_ONNX_PATH,
        providers=resolve_providers(VAD_PROVIDERS),
        sample_rate=sample_rate,
    )
    asr_model = Qwen3ASROnnxModel(
        model_path=ASR_MODEL_PATH,
        onnx_dir=ASR_ONNX_DIR,
        providers=resolve_providers(ASR_PROVIDERS),
        use_iobinding=True,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    pipeline = StreamingVadAsrPipeline(
        vad_model=vad_model,
        asr_model=asr_model,
        sample_rate=sample_rate,
        vad_chunk_ms=VAD_CHUNK_MS,
        asr_chunk_size_sec=ASR_CHUNK_SIZE_SEC,
        asr_max_audio_history_sec=ASR_MAX_AUDIO_HISTORY_SEC,
        pre_roll_ms=PRE_ROLL_MS,
        context=CONTEXT,
        language=LANGUAGE,
        hotwords=HOTWORDS,
        max_new_tokens=MAX_NEW_TOKENS,
        unfixed_chunk_num=UNFIXED_CHUNK_NUM,
        unfixed_token_num=UNFIXED_TOKEN_NUM,
    )

    print("== Streaming VAD + ASR Pipeline ==", flush=True)
    print(f"audio: {AUDIO_PATH}", flush=True)
    print(f"loaded_audio_seconds: {wav.shape[0] / sample_rate:.2f}", flush=True)
    print(f"sample_rate: {sample_rate}", flush=True)
    print(f"audio_samples: {wav.shape[0]}", flush=True)
    print(f"audio_duration_ms: {int(round(wav.shape[0] * 1000.0 / sample_rate))}", flush=True)
    print(f"max_audio_seconds: {MAX_AUDIO_SECONDS}", flush=True)
    print(f"vad_chunk_ms: {VAD_CHUNK_MS}", flush=True)
    print(f"vad_chunk_samples: {max(1, int(round(VAD_CHUNK_MS * sample_rate / 1000.0)))}", flush=True)
    print(f"realtime_pacing: {REALTIME_PACING}", flush=True)
    print(f"asr_chunk_size_sec: {ASR_CHUNK_SIZE_SEC}", flush=True)
    print(f"asr_chunk_samples: {max(1, int(round(ASR_CHUNK_SIZE_SEC * sample_rate)))}", flush=True)
    print(f"asr_max_audio_history_sec: {ASR_MAX_AUDIO_HISTORY_SEC}", flush=True)
    print(
        "asr_max_audio_history_samples: "
        f"{int(round(ASR_MAX_AUDIO_HISTORY_SEC * sample_rate)) if ASR_MAX_AUDIO_HISTORY_SEC and ASR_MAX_AUDIO_HISTORY_SEC > 0 else None}",
        flush=True,
    )
    print(f"pre_roll_ms: {PRE_ROLL_MS}", flush=True)
    print(f"pre_roll_samples: {max(0, int(round(PRE_ROLL_MS * sample_rate / 1000.0)))}", flush=True)
    print(f"language: {LANGUAGE!r}", flush=True)
    print(f"context_chars: {len(CONTEXT)}", flush=True)
    print(f"hotwords: {HOTWORDS!r}", flush=True)
    print(f"max_new_tokens: {MAX_NEW_TOKENS}", flush=True)
    print(f"unfixed_chunk_num: {UNFIXED_CHUNK_NUM}", flush=True)
    print(f"unfixed_token_num: {UNFIXED_TOKEN_NUM}", flush=True)
    print(f"vad_onnx: {VAD_ONNX_PATH}", flush=True)
    print(f"vad_model_dir: {VAD_MODEL_DIR}", flush=True)
    print(f"vad_requested_providers: {resolve_providers(VAD_PROVIDERS)}", flush=True)
    print(f"vad_actual_providers: {vad_model.encoder_session.runner.providers}", flush=True)
    print(f"vad_use_iobinding: {vad_model.encoder_session.runner.use_iobinding}", flush=True)
    print(f"asr_onnx_dir: {ASR_ONNX_DIR}", flush=True)
    print(f"asr_model_path: {ASR_MODEL_PATH}", flush=True)
    print(f"asr_requested_providers: {resolve_providers(ASR_PROVIDERS)}", flush=True)
    print(f"asr_audio_encoder_onnx: {asr_model.audio_encoder_onnx_path}", flush=True)
    print(f"asr_token_embedding_onnx: {asr_model.token_embedding_onnx_path}", flush=True)
    print(f"asr_text_core_onnx: {asr_model.text_core_onnx_path}", flush=True)
    print(f"asr_audio_encoder_providers: {asr_model.audio_encoder_runner.providers}", flush=True)
    print(f"asr_token_embedding_providers: {asr_model.token_embedding_runner.providers}", flush=True)
    print(f"asr_text_core_providers: {asr_model.text_core_runner.providers}", flush=True)
    print(
        "asr_use_iobinding: "
        f"audio_encoder={asr_model.audio_encoder_runner.use_iobinding} "
        f"token_embedding={asr_model.token_embedding_runner.use_iobinding} "
        f"text_core={asr_model.text_core_runner.use_iobinding}",
        flush=True,
    )

    pipeline.start()
    reader = threading.Thread(target=drain_results, args=(pipeline,), name="result-reader")
    reader.start()

    chunk_samples = max(1, int(round(VAD_CHUNK_MS * sample_rate / 1000.0)))
    next_progress_samples = int(round(PROGRESS_INTERVAL_SEC * sample_rate))
    producer_started_at = time.perf_counter()
    for start in range(0, wav.shape[0], chunk_samples):
        chunk = wav[start : start + chunk_samples]
        timestamp_ms = int(round(start * 1000.0 / sample_rate))
        pipeline.push_audio(chunk, timestamp_ms=timestamp_ms)
        chunk_end_samples = min(wav.shape[0], start + chunk.shape[0])
        chunk_end_ms = int(round(chunk_end_samples * 1000.0 / sample_rate))
        print(
            f"[audio_push] chunk_id={start // chunk_samples} "
            f"start_ms={timestamp_ms} end_ms={chunk_end_ms} "
            f"samples={chunk.shape[0]} "
            f"duration_ms={int(round(chunk.shape[0] * 1000.0 / sample_rate))} "
            f"audio_queue_size={pipeline.audio_queue.qsize()}",
            flush=True,
        )
        if REALTIME_PACING:
            target_elapsed_sec = chunk_end_samples / float(sample_rate)
            sleep_sec = producer_started_at + target_elapsed_sec - time.perf_counter()
            if sleep_sec > 0:
                time.sleep(sleep_sec)
        if next_progress_samples > 0 and start >= next_progress_samples:
            print(f"[progress] pushed_audio_seconds={start / sample_rate:.1f}", flush=True)
            next_progress_samples += int(round(PROGRESS_INTERVAL_SEC * sample_rate))

    pipeline.finish()
    pipeline.join()
    reader.join()
    print("== done ==", flush=True)


if __name__ == "__main__":
    main()
