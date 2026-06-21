from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from src.models import Qwen3ASROnnxModel


# ======================== 默认运行配置 ========================
# 直接运行本文件即可使用下面这些配置：
#     python test_qwen3_stream_onnx.py
#
# 如需临时覆盖某个配置，也可以继续使用命令行参数，例如：
#     python test_qwen3_stream_onnx.py --providers cpu --max-audio-seconds 30

# 原始 Qwen3-ASR 模型目录，用于加载 tokenizer / processor。
DEFAULT_MODEL_PATH = Path("/home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-ASR-0.6B")

# 已导出的 ONNX 目录，里面需要包含 audio_encoder、token_embedding、text_core 三个子目录。
DEFAULT_ONNX_DIR = Path("./onnx_asr")

# 测试音频路径；脚本会自动读取并重采样到 16 kHz 单声道。
DEFAULT_AUDIO_PATH = Path("./data/base_clone_nonstream.wav")

# 强制识别语言；None 表示让模型自己判断。也可以填 "Chinese" / "English" 等。
DEFAULT_LANGUAGE = None

# 上下文提示词，会放进 system prompt；空字符串表示不额外提示。
DEFAULT_CONTEXT = ""

# 热词提示，多个热词用英文逗号分隔。
DEFAULT_HOTWORDS = "小医仙,萧炎,斗破苍穹"

# 每攒够多少秒音频触发一次伪流式识别。
DEFAULT_CHUNK_SIZE_SEC = 0.5

# 每次向流式接口喂入多少秒音频；小于 chunk_size 时会先缓存，所以日志可能两行输出相同。
DEFAULT_FEED_SIZE_SEC = 0.5

# 最多读取音频前多少秒；<= 0 表示读取完整音频。
DEFAULT_MAX_AUDIO_SECONDS = 1200.0

# 每次伪流式重算最多保留最近多少秒音频；<= 0 表示保留全部历史音频。
DEFAULT_MAX_AUDIO_HISTORY_SECONDS = 30.0

# 文本生成最大 token 数；太小会截断结果，太大可能让异常续写更长。
DEFAULT_MAX_NEW_TOKENS = 2048

# ONNXRuntime provider：cuda 优先使用 GPU，cpu 使用 CPU，auto 交给封装内部自动选择。
DEFAULT_PROVIDERS = "cuda"

# 是否关闭 IO binding；默认 False 表示启用 IO binding。
DEFAULT_NO_IOBINDING = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pseudo-streaming Qwen3-ASR ONNXRuntime test.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, type=Path, help="原始 Qwen3-ASR 模型目录。")
    parser.add_argument("--onnx-dir", default=DEFAULT_ONNX_DIR, type=Path, help="导出的 ONNX 根目录。")
    parser.add_argument("--audio", default=DEFAULT_AUDIO_PATH, type=Path, help="测试音频路径。")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE, help="强制识别语言；默认自动判断。")
    parser.add_argument("--context", default=DEFAULT_CONTEXT, help="ASR 上下文提示词。")
    parser.add_argument("--hotwords", default=DEFAULT_HOTWORDS, help="热词，多个词用英文逗号分隔。")
    parser.add_argument("--chunk-size-sec", default=DEFAULT_CHUNK_SIZE_SEC, type=float, help="每个识别 chunk 的秒数。")
    parser.add_argument("--feed-size-sec", default=DEFAULT_FEED_SIZE_SEC, type=float, help="每次喂入流式接口的秒数。")
    parser.add_argument(
        "--max-audio-seconds",
        default=DEFAULT_MAX_AUDIO_SECONDS,
        type=float,
        help="最多读取音频前 N 秒；<= 0 表示读取完整音频。",
    )
    parser.add_argument(
        "--max-audio-history-seconds",
        default=DEFAULT_MAX_AUDIO_HISTORY_SECONDS,
        type=float,
        help="每次重算最多保留最近 N 秒音频；<= 0 表示保留全部历史。",
    )
    parser.add_argument("--max-new-tokens", default=DEFAULT_MAX_NEW_TOKENS, type=int, help="最大生成 token 数。")
    parser.add_argument("--providers", default=DEFAULT_PROVIDERS, choices=("cuda", "cpu", "auto"), help="推理设备。")
    parser.add_argument(
        "--no-iobinding",
        action="store_true",
        default=DEFAULT_NO_IOBINDING,
        help="关闭 ONNXRuntime IO binding。",
    )
    return parser.parse_args()


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
    wav = np.asarray(wav)
    if wav.ndim != 1:
        wav = wav.mean(axis=-1).astype(np.float32)
    wav = wav.astype(np.float32, copy=False)
    if int(sample_rate) != 16000:
        wav = librosa.resample(wav, orig_sr=int(sample_rate), target_sr=16000).astype(np.float32)
        sample_rate = 16000
    return wav, int(sample_rate)


def main() -> None:
    args = parse_args()
    model = Qwen3ASROnnxModel(
        model_path=args.model_path,
        onnx_dir=args.onnx_dir,
        providers=resolve_providers(args.providers),
        use_iobinding=not args.no_iobinding,
        max_new_tokens=args.max_new_tokens,
    )
    wav, sample_rate = load_audio(args.audio, args.max_audio_seconds)

    state = model.init_streaming_state(
        context=args.context,
        language=args.language,
        hotwords=args.hotwords,
        chunk_size_sec=args.chunk_size_sec,
        max_audio_history_sec=args.max_audio_history_seconds,
    )
    feed_samples = max(1, int(round(float(args.feed_size_sec) * sample_rate)))

    print("== Qwen3-ASR ONNX pseudo streaming test ==")
    print(f"audio: {args.audio}")
    print(f"loaded_audio_seconds: {wav.shape[0] / sample_rate:.2f}")
    print(f"max_audio_seconds: {args.max_audio_seconds}")
    print(f"max_audio_history_seconds: {args.max_audio_history_seconds}")
    print(f"chunk_size_sec: {args.chunk_size_sec}")
    print(f"feed_size_sec: {args.feed_size_sec}")
    print(f"hotwords: {args.hotwords}")

    last_trim_events = state.trim_events
    last_partial_text = ""
    for start in range(0, wav.shape[0], feed_samples):
        piece = wav[start : start + feed_samples]
        state = model.streaming_transcribe(piece, state, max_new_tokens=args.max_new_tokens)
        if state.trim_events > last_trim_events:
            trimmed_sec = state.last_trimmed_samples / sample_rate
            total_trimmed_sec = state.total_trimmed_samples / sample_rate
            kept_sec = state.audio_accum.shape[0] / sample_rate
            print(
                f"[trim] events={state.trim_events} "
                f"trimmed={trimmed_sec:.2f}s "
                f"total_trimmed={total_trimmed_sec:.2f}s "
                f"kept_audio={kept_sec:.2f}s"
            )
            last_trim_events = state.trim_events
        if state.confirmed_delta_text or state.text != last_partial_text or state.metadata.get("stream_text_revision"):
            last_partial_text = state.text
            chunk_start = float(state.metadata.get("trigger_chunk_start_ms", 0)) / 1000.0
            chunk_end = float(state.metadata.get("trigger_chunk_end_ms", 0)) / 1000.0
            window_start = float(state.metadata.get("audio_window_start_ms", 0)) / 1000.0
            window_end = float(state.metadata.get("audio_window_end_ms", 0)) / 1000.0
            print(
                f"[partial] chunk={chunk_start:7.2f}-{chunk_end:7.2f}s "
                f"window={window_start:7.2f}-{window_end:7.2f}s "
                f"feed_samples={start + piece.shape[0]:>7d} "
                f"processed_chunks={state.chunk_id:<3d} "
                f"language={state.language!r} "
                f"confirmed_delta={state.confirmed_delta_text!r} "
                f"confirmed_text={state.confirmed_text!r} "
                f"pending={state.pending_text!r}"
            )

    state = model.finish_streaming_transcribe(state, max_new_tokens=args.max_new_tokens)
    if state.confirmed_delta_text or state.text != last_partial_text:
        chunk_start = float(state.metadata.get("trigger_chunk_start_ms", 0)) / 1000.0
        chunk_end = float(state.metadata.get("trigger_chunk_end_ms", 0)) / 1000.0
        window_start = float(state.metadata.get("audio_window_start_ms", 0)) / 1000.0
        window_end = float(state.metadata.get("audio_window_end_ms", 0)) / 1000.0
        print(
            f"[final] chunk={chunk_start:7.2f}-{chunk_end:7.2f}s "
            f"window={window_start:7.2f}-{window_end:7.2f}s "
            f"confirmed_delta={state.confirmed_delta_text!r} "
            f"confirmed_text={state.confirmed_text!r} "
            f"pending={state.pending_text!r}"
        )
    print("== final ==")
    print(f"language: {state.language}")
    print(f"text: {state.text}")
    print(f"raw_text: {state.raw_text}")
    print("metadata:")
    for key, value in state.metadata.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
