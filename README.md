# Qwen3-ASR ONNX + FSMN-VAD ONNX

ONNX Runtime inference implementation for Qwen3-ASR, providing a lightweight cross-platform speech recognition deployment solution for fast local and offline ASR.

项目介绍：[Qwen3-ASR 流式推理链路拆解：从模型原理到 ONNX Runtime 实现](https://zhuanlan.zhihu.com/p/2051949383655101596)

## 导出 ONNX

### 1. 导出 Qwen3-ASR

```bash
python export_qwen3_asr_onnx.py \
  --model-path /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-ASR-0.6B \
  --output-dir ./onnx_asr \
  --dtype float16 \
  --device cuda \
  --components all
```

导出后目录应包含：

```text
onnx_asr/audio_encoder/audio_encoder.onnx
onnx_asr/token_embedding/token_embedding.onnx
onnx_asr/text_core/asr_text_core.onnx
```

### 2. 导出 FSMN-VAD

```bash
python export_fsmn_vad_onnx.py \
  --model-dir /home/zhang/.cache/modelscope/hub/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch \
  --output-dir ./onnx_fsmn \
  --output-name fsmn_vad_vad_encoder.onnx \
  --opset-version 13 \
  --dummy-frames 30
```

导出后目录应包含：

```text
onnx_fsmn/fsmn_vad_vad_encoder.onnx
onnx_fsmn/config.yaml
onnx_fsmn/am.mvn
```

## 运行

### Python

```bash
# ASR 离线
python test_qwen3_onnx.py

# FSMN-VAD
python test_fsmn_vad_onnxruntime.py

# 离线 VAD + ASR
python test_offline_vad_asr_pipeline.py

# 流水线 VAD + ASR
python test_streaming_vad_asr_pipeline.py
```

### C++ 实时 VAD + ASR

准备 C++ 依赖：

```bash
bash cpp/scripts/download_deps.sh
```

脚本会下载 ONNX Runtime GPU C++ 包，并编译 FFTW3f 到 `cpp/third_party/`。

编译：

```bash
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
```

运行：

```bash
./build/qwen3asr_streaming_run
```

默认配置在 `cpp/apps/qwen3asr_streaming_pipeline_main.cc` 顶部。运行时日志写到：

```text
./qwen3asr_cpp_streaming.log
```

可以用下面命令查看进度：

```bash
tail -f qwen3asr_cpp_streaming.log
```

日志里主要事件：

```text
[speech_start] 新的 VAD 语音段开始
[partial] ASR 实时中间结果
[final] 当前 VAD 语音段最终结果
[pipeline_done] 流水线结束
```

`partial/final` 会输出 `confirmed_delta`、`confirmed_text`、`pending`；`q=a/b/c` 表示 `audio_queue/speech_queue/result_queue` 当前积压长度。

默认路径：

```text
ASR 模型目录: /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-ASR-0.6B
ASR ONNX: ./onnx_asr
VAD ONNX: ./onnx_fsmn/fsmn_vad_vad_encoder.onnx
VAD CMVN: ./onnx_fsmn/am.mvn
默认音频: ./data/voice_design_stream.wav
```

常用参数：

```bash
./build/qwen3asr_streaming_run --audio ./data/voice_design_stream.wav
./build/qwen3asr_streaming_run --cpu
```
