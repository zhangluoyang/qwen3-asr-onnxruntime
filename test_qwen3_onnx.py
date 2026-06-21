from __future__ import annotations

from pathlib import Path
from typing import Any

from src.models import Qwen3ASROnnxModel


# ======================== 默认运行配置 ========================
# 直接运行本文件即可使用下面这些配置：
#     python test_qwen3_onnx.py
#
# 如需换模型、音频或推理设备，直接改下面的常量即可。

# 原始 Qwen3-ASR 模型目录，用于加载 tokenizer / processor。
MODEL_PATH = Path("/home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-ASR-0.6B")

# 已导出的 ONNX 目录，里面需要包含 audio_encoder、token_embedding、text_core 三个子目录。
ONNX_DIR = Path("./onnx_asr")

# 测试音频路径。
AUDIO_PATH = Path("./data/voice_design_nonstream.wav")

# 强制识别语言；None 表示让模型自己判断。也可以填 "Chinese" / "English" 等。
LANGUAGE = None

# 上下文提示词，会放进 system prompt；空字符串表示不额外提示。
CONTEXT = ""

# 热词提示，多个热词用英文逗号分隔。
HOTWORDS = "小医仙,萧炎,斗破苍穹"

# 文本生成最大 token 数；太小会截断结果，太大可能让异常续写更长。
MAX_NEW_TOKENS = 1024

# ONNXRuntime provider：cuda 优先使用 GPU 并保留 CPU fallback，cpu 使用 CPU，auto 交给封装内部自动选择。
PROVIDERS = "cuda"

# 是否关闭 IO binding；默认 False 表示启用 IO binding。
NO_IOBINDING = False

# 最多读取音频前多少秒；<= 0 表示读取完整音频。
MAX_AUDIO_SECONDS = 1024.0


def resolve_providers(name: str) -> list[str] | None:
    if name == "auto":
        return None
    if name == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def prepare_audio_arg(audio_path: Path, max_audio_seconds: float) -> Any:
    if max_audio_seconds <= 0:
        return str(audio_path)

    import soundfile as sf

    info = sf.info(str(audio_path))
    frames = int(round(float(max_audio_seconds) * int(info.samplerate)))
    wav, sample_rate = sf.read(str(audio_path), frames=frames, dtype="float32", always_2d=False)
    return wav, int(sample_rate)


def main() -> None:
    providers = resolve_providers(PROVIDERS)
    use_iobinding = not NO_IOBINDING

    model = Qwen3ASROnnxModel(
        model_path=MODEL_PATH,
        onnx_dir=ONNX_DIR,
        providers=providers,
        use_iobinding=use_iobinding,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    audio = prepare_audio_arg(AUDIO_PATH, MAX_AUDIO_SECONDS)

    results = model.transcribe(
        audio=audio,
        context=CONTEXT,
        language=LANGUAGE,
        hotwords=HOTWORDS,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    result = results[0]

    print("== Qwen3-ASR ONNX test ==")
    print(f"model_path: {MODEL_PATH}")
    print(f"onnx_dir: {ONNX_DIR}")
    print(f"audio: {AUDIO_PATH}")
    print(f"max_audio_seconds: {MAX_AUDIO_SECONDS}")
    print(f"providers: {PROVIDERS}")
    print(f"hotwords: {HOTWORDS}")
    print(f"language: {result.language}")
    print(f"text: {result.text}")
    print(f"raw_text: {result.raw_text}")
    print("metadata:")
    for key, value in result.metadata.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
