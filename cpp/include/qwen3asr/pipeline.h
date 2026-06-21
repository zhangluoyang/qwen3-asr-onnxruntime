#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "qwen3asr/fsmn_vad.h"
#include "qwen3asr/qwen_model.h"

namespace qwen3asr {

struct EnergyVadOptions {
  // 能量 VAD 阈值，单位 dB。数值越高越严格；环境噪声大时可以适当调高。
  float threshold_db = -45.0f;

  // VAD 计算 RMS 能量时的帧长和帧移。
  int frame_ms = 20;
  int hop_ms = 10;

  // 最短语音长度和最短静音长度，用于抑制毛刺。
  int min_speech_ms = 200;
  int min_silence_ms = 300;

  // VAD 片段送 ASR 前向前/向后保留一点音频，避免切掉字头字尾。
  int pad_start_ms = 120;
  int pad_end_ms = 180;

  // padding 后相邻片段间隔小于该值时合并，减少碎片化 ASR 调用。
  int merge_gap_ms = 200;
};

struct VadAsrSegment {
  int64_t start_ms = 0;
  int64_t end_ms = 0;
  int64_t vad_start_ms = 0;
  int64_t vad_end_ms = 0;
  std::string language;
  std::string text;
  std::string raw_text;
  double asr_elapsed_ms = 0.0;
};

struct VadAsrResult {
  int sample_rate = 16000;
  int64_t duration_ms = 0;
  std::vector<std::pair<int64_t, int64_t>> vad_segments_ms;
  std::vector<VadAsrSegment> segments;
  std::string language;
  std::string text;
  std::string raw_text;
  double vad_elapsed_ms = 0.0;
  double asr_elapsed_ms = 0.0;
  double total_elapsed_ms = 0.0;
};

struct VadAsrProgressEvent {
  std::string type;
  int64_t segment_index = -1;
  int64_t segment_count = 0;
  int64_t duration_ms = 0;
  double elapsed_ms = 0.0;
  double vad_elapsed_ms = 0.0;
  double asr_elapsed_ms = 0.0;
  std::vector<std::pair<int64_t, int64_t>> vad_segments_ms;
  VadAsrSegment segment;
  std::string text;
};

using VadAsrProgressCallback = std::function<void(const VadAsrProgressEvent&)>;

class EnergyVadAsrPipeline {
 public:
  EnergyVadAsrPipeline(const Qwen3ASROnnxModel& asr_model, EnergyVadOptions options = {});

  VadAsrResult Transcribe(const std::vector<float>& samples,
                          const std::string& context = "",
                          const std::optional<std::string>& language = std::nullopt,
                          const std::string& hotwords = "",
                          int max_new_tokens = -1,
                          VadAsrProgressCallback progress = nullptr) const;

 private:
  struct PreparedSegment {
    int64_t start_ms = 0;
    int64_t end_ms = 0;
    int64_t vad_start_ms = 0;
    int64_t vad_end_ms = 0;
  };

  std::vector<std::pair<int64_t, int64_t>> DetectSpeech(const std::vector<float>& samples) const;
  std::vector<PreparedSegment> PrepareSegments(
      const std::vector<std::pair<int64_t, int64_t>>& vad_segments,
      int64_t duration_ms) const;
  std::vector<float> SliceMs(const std::vector<float>& samples, int64_t start_ms, int64_t end_ms) const;

  const Qwen3ASROnnxModel& asr_model_;
  EnergyVadOptions options_;
};

class FsmnVadAsrPipeline {
 public:
  FsmnVadAsrPipeline(const FsmnVadOnnxModel& vad_model, const Qwen3ASROnnxModel& asr_model);

  VadAsrResult Transcribe(const std::vector<float>& samples,
                          const std::string& context = "",
                          const std::optional<std::string>& language = std::nullopt,
                          const std::string& hotwords = "",
                          int max_new_tokens = -1,
                          VadAsrProgressCallback progress = nullptr) const;

 private:
  struct PreparedSegment {
    int64_t start_ms = 0;
    int64_t end_ms = 0;
    int64_t vad_start_ms = 0;
    int64_t vad_end_ms = 0;
  };

  std::vector<PreparedSegment> PrepareSegments(
      const std::vector<std::pair<int64_t, int64_t>>& vad_segments,
      int64_t duration_ms) const;
  std::vector<float> SliceMs(const std::vector<float>& samples, int64_t start_ms, int64_t end_ms) const;

  const FsmnVadOnnxModel& vad_model_;
  const Qwen3ASROnnxModel& asr_model_;
  int pad_start_ms_ = 120;
  int pad_end_ms_ = 180;
  int merge_gap_ms_ = 200;
  int min_segment_ms_ = 200;
};

}  // namespace qwen3asr
