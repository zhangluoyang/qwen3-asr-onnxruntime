from __future__ import annotations

from pathlib import Path

import numpy as np

from src.fsmn_vad_onnx import FsmnVadOnnxModel
from src.models import Qwen3ASROnnxModel
from src.pipeline import OfflineVadAsrPipeline


# ======================== 默认运行配置 ========================
# 直接运行本文件即可使用下面这些配置：
#     python test_offline_vad_asr_pipeline.py
#
# 如需换模型、音频或推理设备，直接改下面的常量即可。

# Qwen3-ASR 原始模型目录，用于加载 tokenizer / processor。
ASR_MODEL_PATH = Path("/home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-ASR-0.6B")

# 已导出的 Qwen3-ASR ONNX 目录。
ASR_ONNX_DIR = Path("./onnx_asr")

# FSMN-VAD 原始模型目录。当前默认测试会优先使用 VAD ONNX 目录里的 config.yaml / am.mvn。
VAD_MODEL_DIR = Path("./fsmn")

# 已导出的 FSMN-VAD ONNX 文件。
VAD_ONNX_PATH = Path("./onnx_fsmn/fsmn_vad_vad_encoder.onnx")

# 测试音频路径；脚本会自动读取并重采样到 16 kHz 单声道。
AUDIO_PATH = Path("./data/base_clone_nonstream.wav")

# ASR provider：cuda / cpu / auto。
ASR_PROVIDERS = "cuda"

# VAD provider：cpu / cuda / auto。FSMN-VAD 较小，默认 CPU 就够用。
VAD_PROVIDERS = "cpu"

# 最多读取音频前多少秒；<= 0 表示读取完整音频。
MAX_AUDIO_SECONDS = 20.0

# ASR 参数。
LANGUAGE = None
CONTEXT = ""
HOTWORDS = "小医仙,萧炎,斗破苍穹"
MAX_NEW_TOKENS = 1024

# VAD 片段送 ASR 前的保护边界，避免切掉开头/结尾。
PAD_START_MS = 120
PAD_END_MS = 180
MERGE_GAP_MS = 200
MIN_SEGMENT_MS = 200


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
    pipeline = OfflineVadAsrPipeline(
        vad_model=vad_model,
        asr_model=asr_model,
        sample_rate=sample_rate,
        pad_start_ms=PAD_START_MS,
        pad_end_ms=PAD_END_MS,
        merge_gap_ms=MERGE_GAP_MS,
        min_segment_ms=MIN_SEGMENT_MS,
    )

    result = pipeline.transcribe(
        audio=wav,
        sample_rate=sample_rate,
        context=CONTEXT,
        language=LANGUAGE,
        hotwords=HOTWORDS,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    print("== Offline VAD + ASR Pipeline ==")
    print(f"audio: {AUDIO_PATH}")
    print(f"loaded_audio_seconds: {wav.shape[0] / sample_rate:.2f}")
    print(f"sample_rate: {sample_rate}")
    print(f"vad_onnx: {VAD_ONNX_PATH}")
    print(f"asr_onnx_dir: {ASR_ONNX_DIR}")
    print(f"vad_providers: {vad_model.encoder_session.runner.providers}")
    print(f"asr_providers: {asr_model.audio_encoder_runner.providers}")
    print(f"vad_elapsed_ms: {result.vad_elapsed_ms:.2f}")
    print(f"asr_elapsed_ms: {result.asr_elapsed_ms:.2f}")
    print(f"total_elapsed_ms: {result.metadata['total_elapsed_ms']:.2f}")
    print(f"vad_segments_ms: {result.vad.segments}")
    print("segments:")
    for i, segment in enumerate(result.segments):
        print(
            f"  [{i}] {segment.start_ms}-{segment.end_ms}ms "
            f"vad={segment.vad_start_ms}-{segment.vad_end_ms}ms "
            f"asr_elapsed_ms={segment.asr_elapsed_ms:.2f} "
            f"language={segment.language!r} text={segment.text!r}"
        )
    print("text:")
    print(result.text)
    print("metadata:")
    for key, value in result.metadata.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
