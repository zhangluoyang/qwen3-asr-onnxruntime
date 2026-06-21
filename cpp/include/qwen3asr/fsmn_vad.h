#pragma once

#include <cstdint>
#include <filesystem>
#include <utility>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "qwen3asr/ort_runner.h"
#include "qwen3asr/tensor.h"

namespace qwen3asr {

struct FsmnVadConfig {
  // 已导出的 FunASR FSMN-VAD encoder ONNX。
  std::filesystem::path onnx_path = "./onnx_fsmn/fsmn_vad_vad_encoder.onnx";

  // FunASR 的 CMVN 文件，必须和导出模型来自同一个目录。
  std::filesystem::path cmvn_path = "./onnx_fsmn/am.mvn";

  // VAD 固定使用 16 kHz 单声道输入，和 Python 封装保持一致。
  int sample_rate = 16000;

  // VAD 模型很小，CPU 通常已经足够；需要时可以打开 CUDA。
  bool use_cuda = false;
  int cuda_device_id = 0;
};

struct FsmnVadResult {
  std::vector<std::pair<int64_t, int64_t>> segments_ms;
  std::vector<std::pair<int64_t, int64_t>> events_ms;
  int64_t frame_count = 0;
  double elapsed_ms = 0.0;
};

class FsmnVadOnnxModel {
 public:
  explicit FsmnVadOnnxModel(FsmnVadConfig config = {});

  enum class VadStateMachine {
    kStartPointNotDetected = 1,
    kInSpeechSegment = 2,
    kEndPointDetected = 3,
  };
  enum class FrameState {
    kInvalid = -1,
    kSil = 0,
    kSpeech = 1,
  };
  enum class AudioChangeState {
    kSpeech2Speech = 0,
    kSpeech2Sil = 1,
    kSil2Sil = 2,
    kSil2Speech = 3,
    kNoBegin = 4,
    kInvalid = 5,
  };

  struct WindowDetector {
    explicit WindowDetector(int window_size_ms = 200,
                            int sil_to_speech_time_ms = 150,
                            int speech_to_sil_time_ms = 150,
                            int frame_size_ms = 10);
    void Reset();
    int WinSize() const { return win_size_frame; }
    AudioChangeState DetectOneFrame(FrameState frame_state);

    int win_size_frame = 20;
    int sil_to_speech_frame_threshold = 15;
    int speech_to_sil_frame_threshold = 15;
    int cur_win_pos = 0;
    int win_sum = 0;
    std::vector<int> win_state;
    FrameState pre_frame_state = FrameState::kSil;
  };

  struct SegmentBuf {
    int64_t start_ms = 0;
    int64_t end_ms = 0;
    bool contain_start = false;
    bool contain_end = false;
  };

  struct DetectionState {
    int64_t data_buf_start_frame = 0;
    int64_t latest_confirmed_speech_frame = 0;
    int64_t latest_confirmed_silence_frame = -1;
    int64_t continuous_silence_frame_count = 0;
    VadStateMachine vad_state_machine = VadStateMachine::kStartPointNotDetected;
    int64_t confirmed_start_frame = -1;
    int64_t confirmed_end_frame = -1;
    int number_end_time_detected = 0;
    int64_t sil_frame = 0;
    double noise_average_decibel = -100.0;
    bool pre_end_silence_detected = false;
    bool next_seg = true;
    int max_end_sil_frame_count_threshold_ms = 650;
    float speech_noise_thres = 0.6f;
    bool max_time_out = false;
    int64_t last_drop_frames = 0;
    WindowDetector window;
    std::vector<SegmentBuf> output_data_buf;
  };

  struct StreamingState {
    int chunk_size_ms = 200;
    int64_t total_samples = 0;
    std::vector<float> input_cache;
    std::vector<float> lfr_splice_cache;
    std::vector<float> decibel_cache;
    std::vector<float> fsmn_caches;
    std::vector<float> scores;
    std::vector<float> decibel;
    int64_t score_frames = 0;
    int64_t output_data_buf_offset = 0;
    DetectionState detection;
  };

  FsmnVadResult DetectOffline(const std::vector<float>& samples) const;
  StreamingState InitStreamingState(int chunk_size_ms = 200) const;
  FsmnVadResult DetectStreaming(const std::vector<float>& samples,
                                StreamingState* state,
                                bool is_final = false) const;
  FsmnVadResult FinishStreaming(StreamingState* state) const;
  FloatTensor DebugFbank(const std::vector<float>& samples) const { return ComputeFbank(samples); }
  FloatTensor DebugFeatures(const std::vector<float>& samples) const {
    return ApplyLfrCmvn(ComputeFbank(samples), cmvn_means_, cmvn_vars_);
  }
  FloatTensor DebugScores(const std::vector<float>& samples) const {
    return RunEncoder(DebugFeatures(samples));
  }

 private:
  static std::vector<float> LoadCmvnValues(const std::filesystem::path& path, const char* marker);
  static std::vector<float> ComputeDecibel(const std::vector<float>& samples);
  static FloatTensor ComputeFbank(const std::vector<float>& samples);
  static FloatTensor ApplyLfrCmvn(const FloatTensor& fbank,
                                  const std::vector<float>& means,
                                  const std::vector<float>& vars);
  static std::vector<float> MakeKaldiMelBanks();
  static double MelScale(double freq);
  static double InverseMelScale(double mel_freq);

  FloatTensor RunEncoder(const FloatTensor& feats) const;
  FloatTensor RunEncoderStreaming(const FloatTensor& feats, StreamingState* state) const;
  FloatTensor ComputeStreamingFeatures(const std::vector<float>& samples,
                                       bool is_final,
                                       StreamingState* state) const;
  std::vector<std::pair<int64_t, int64_t>> CollectStreamingEvents(StreamingState* state) const;
  std::vector<std::pair<int64_t, int64_t>> PostProcess(const FloatTensor& scores,
                                                       const std::vector<float>& decibel) const;
  FrameState GetFrameState(int64_t frame_index,
                           const FloatTensor& scores,
                           const std::vector<float>& decibel,
                           DetectionState* state) const;
  void DetectOneFrame(FrameState frame_state,
                      int64_t frame_index,
                      bool is_final_frame,
                      DetectionState* state) const;
  void OnSilenceDetected(int64_t valid_frame, DetectionState* state) const;
  void OnVoiceDetected(int64_t valid_frame, DetectionState* state) const;
  void OnVoiceStart(int64_t start_frame, bool fake_result, DetectionState* state) const;
  void OnVoiceEnd(int64_t end_frame, bool fake_result, bool is_last_frame, DetectionState* state) const;
  void MaybeOnVoiceEndIfLastFrame(bool is_final_frame, int64_t frame_index, DetectionState* state) const;
  void ResetDetection(DetectionState* state) const;
  int LatencyFrameNumAtStartPoint(const DetectionState& state) const;

  FsmnVadConfig config_;
  Ort::Env env_;
  OrtRunner encoder_;
  std::vector<float> cmvn_means_;
  std::vector<float> cmvn_vars_;
};

}  // namespace qwen3asr
