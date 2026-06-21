#include "qwen3asr/fsmn_vad.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <fftw3.h>

namespace qwen3asr {
namespace {

// VAD 固定输入采样率；前端和模型都按 16 kHz 设计。
constexpr int kSampleRate = 16000;
// VAD 后处理的帧移，1 帧对应 10ms。
constexpr int kFrameMs = 10;
// FBank 分析窗长度，25ms 是常见语音特征窗口。
constexpr int kFrameLengthMs = 25;
// 25ms * 16kHz = 400 samples。
constexpr int kFrameSamples = 400;
// 10ms * 16kHz = 160 samples。
constexpr int kFrameShiftSamples = 160;
// FFT 补零后的窗口长度。
constexpr int kPaddedWindowSize = 512;
// Mel 滤波器个数。
constexpr int kNumMelBins = 80;
// LFR 拼帧数量；5 帧拼成一个模型输入步。
constexpr int kLfrM = 5;
// LFR 帧移；每 1 帧滑动一次。
constexpr int kLfrN = 1;
// 模型输入维度，kNumMelBins * kLfrM。
constexpr int kInputDim = 400;
// FSMN 层数，用于 cache 张量布局。
constexpr int kNumFsmnLayers = 4;
// FSMN cache 的投影维度。
constexpr int kProjDim = 128;
// 每层保留的历史 cache 帧数。
constexpr int kCacheFrames = 19;
// 防止概率取 log 时出现 0 或 1 的极值。
constexpr float kFloatEps = 1.1920928955078125e-7f;
// 计算窗函数时使用的圆周率。
constexpr double kPi = 3.14159265358979323846;

// 是否允许一段音频里检测多个语音段；1 表示端点后重置状态，继续找下一段。
constexpr int kDetectModeMultipleUtterance = 1;
// 端点检测的目标静音时长。实际连续静音计数阈值会减去 kSpeechToSilTimeMs，
// 因此当前配置下大约 1150ms 静音后切断当前 utterance。
constexpr int kMaxEndSilenceTimeMs = 1150;
// 开头最长静音容忍时间；当前 streaming 逻辑里 start_timeout 关闭，主要保留为对齐 FunASR 参数。
constexpr int kMaxStartSilenceTimeMs = 3000;
// 后处理平滑窗口大小；窗口越大，起止点越稳但响应越慢。
constexpr int kWindowSizeMs = 200;
// 静音转语音阈值；窗口内累计约 150ms 语音帧后触发 speech_start。
constexpr int kSilToSpeechTimeMs = 150;
// 语音转静音阈值；窗口内语音帧降低到约 150ms 以下后进入静音趋势。
constexpr int kSpeechToSilTimeMs = 150;
// 起点回看时间；检测到 speech_start 后向前回看，尽量保留开头短音。
constexpr int kLookbackTimeStartPointMs = 200;
// 终点前瞻时间；端点回退时保留少量尾音，避免过早截断。
constexpr int kLookaheadTimeEndPointMs = 100;
// 单个语音段最大长度；超过后强制切段。
constexpr int kMaxSingleSegmentTimeMs = 60000;
// speech/noise 概率比较的权重，越大越偏向判噪声。
constexpr float kSpeech2NoiseRatio = 1.0f;
// 信噪比阈值；当前 -100 基本等于不额外限制 SNR。
constexpr float kSnrThres = -100.0f;
// 帧能量阈值；低于该分贝值直接按静音处理，当前 -100 基本很宽松。
constexpr float kDecibelThres = -100.0f;
// 语音概率需要超过噪声概率的 margin；越大越保守，越不容易触发语音。
constexpr float kSpeechNoiseThres = 0.6f;
// 特征先验阈值；当前只作为极小门限保留。
constexpr float kFePriorThres = 0.0001f;
// 更新噪声平均能量时使用的平滑帧数；越大噪声估计变化越慢。
constexpr int kNoiseFrameNumUsedForSnr = 100;

double ElapsedMs(std::chrono::steady_clock::time_point started_at) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started_at).count();
}

int64_t FrameToMs(int64_t frame) {
  return frame * kFrameMs;
}

float ClampProb(float value) {
  return std::min(1.0f - 1.0e-7f, std::max(1.0e-7f, value));
}

}  // namespace

FsmnVadOnnxModel::WindowDetector::WindowDetector(int window_size_ms,
                                                 int sil_to_speech_time_ms,
                                                 int speech_to_sil_time_ms,
                                                 int frame_size_ms) {
  win_size_frame = std::max(1, window_size_ms / frame_size_ms);
  sil_to_speech_frame_threshold = std::max(1, sil_to_speech_time_ms / frame_size_ms);
  speech_to_sil_frame_threshold = std::max(1, speech_to_sil_time_ms / frame_size_ms);
  win_state.assign(static_cast<size_t>(win_size_frame), 0);
}

void FsmnVadOnnxModel::WindowDetector::Reset() {
  cur_win_pos = 0;
  win_sum = 0;
  std::fill(win_state.begin(), win_state.end(), 0);
  pre_frame_state = FrameState::kSil;
}

FsmnVadOnnxModel::AudioChangeState FsmnVadOnnxModel::WindowDetector::DetectOneFrame(FrameState frame_state) {
  int cur_frame_state = 0;
  if (frame_state == FrameState::kSpeech) {
    cur_frame_state = 1;
  } else if (frame_state == FrameState::kSil) {
    cur_frame_state = 0;
  } else {
    return AudioChangeState::kInvalid;
  }

  win_sum -= win_state[static_cast<size_t>(cur_win_pos)];
  win_sum += cur_frame_state;
  win_state[static_cast<size_t>(cur_win_pos)] = cur_frame_state;
  cur_win_pos = (cur_win_pos + 1) % win_size_frame;

  if (pre_frame_state == FrameState::kSil && win_sum >= sil_to_speech_frame_threshold) {
    pre_frame_state = FrameState::kSpeech;
    return AudioChangeState::kSil2Speech;
  }
  if (pre_frame_state == FrameState::kSpeech && win_sum <= speech_to_sil_frame_threshold) {
    pre_frame_state = FrameState::kSil;
    return AudioChangeState::kSpeech2Sil;
  }
  return pre_frame_state == FrameState::kSil ? AudioChangeState::kSil2Sil : AudioChangeState::kSpeech2Speech;
}

FsmnVadOnnxModel::FsmnVadOnnxModel(FsmnVadConfig config)
    : config_(std::move(config)),
      env_(ORT_LOGGING_LEVEL_WARNING, "qwen3asr_fsmn_vad_cpp"),
      encoder_(env_, OrtRunnerOptions{config_.onnx_path, config_.use_cuda, config_.cuda_device_id}),
      cmvn_means_(LoadCmvnValues(config_.cmvn_path, "<AddShift>")),
      cmvn_vars_(LoadCmvnValues(config_.cmvn_path, "<Rescale>")) {
  if (config_.sample_rate != kSampleRate) {
    throw std::runtime_error("FSMN-VAD C++ only supports 16 kHz audio");
  }
  if (cmvn_means_.size() < kInputDim || cmvn_vars_.size() < kInputDim) {
    throw std::runtime_error("invalid FSMN-VAD CMVN file: " + config_.cmvn_path.string());
  }
}

FsmnVadResult FsmnVadOnnxModel::DetectOffline(const std::vector<float>& samples) const {
  const auto started_at = std::chrono::steady_clock::now();
  FsmnVadResult result;
  auto fbank = ComputeFbank(samples);
  if (fbank.empty()) {
    result.elapsed_ms = ElapsedMs(started_at);
    return result;
  }
  auto feats = ApplyLfrCmvn(fbank, cmvn_means_, cmvn_vars_);
  auto scores = RunEncoder(feats);
  auto decibel = ComputeDecibel(samples);
  result.frame_count = scores.shape().size() >= 2 ? scores.shape()[1] : 0;
  result.segments_ms = PostProcess(scores, decibel);
  result.events_ms = result.segments_ms;
  result.elapsed_ms = ElapsedMs(started_at);
  return result;
}

FsmnVadOnnxModel::StreamingState FsmnVadOnnxModel::InitStreamingState(int chunk_size_ms) const {
  StreamingState state;
  state.chunk_size_ms = chunk_size_ms;
  state.fsmn_caches.assign(static_cast<size_t>(kNumFsmnLayers * kProjDim * kCacheFrames), 0.0f);
  state.detection.window = WindowDetector(kWindowSizeMs, kSilToSpeechTimeMs, kSpeechToSilTimeMs, kFrameMs);
  // 对齐 FunASR/Python streaming VAD 配置，避免 C++ 把多个 Python 段粘成一个长段。
  state.detection.max_end_sil_frame_count_threshold_ms = kMaxEndSilenceTimeMs - kSpeechToSilTimeMs;
  state.detection.speech_noise_thres = kSpeechNoiseThres;
  return state;
}

FsmnVadResult FsmnVadOnnxModel::DetectStreaming(const std::vector<float>& samples,
                                                StreamingState* state,
                                                bool is_final) const {
  if (!state) throw std::runtime_error("FSMN-VAD streaming state is null");
  const auto started_at = std::chrono::steady_clock::now();
  FsmnVadResult result;
  state->total_samples += static_cast<int64_t>(samples.size());

  const auto feats = ComputeStreamingFeatures(samples, is_final, state);
  if (!feats.empty() && feats.shape()[1] > 0) {
    const auto scores = RunEncoderStreaming(feats, state);
    const int64_t frames = scores.shape()[1];
    const int64_t dim = scores.shape()[2];
    state->scores.insert(state->scores.end(), scores.values().begin(), scores.values().end());
    state->score_frames += frames;

    FloatTensor all_scores({1, state->score_frames, dim}, state->scores);
    for (int64_t i = 0; i < frames; ++i) {
      const int64_t frame_index = state->score_frames - frames + i;
      const auto frame_state = GetFrameState(frame_index, all_scores, state->decibel, &state->detection);
      DetectOneFrame(frame_state, frame_index, is_final && i == frames - 1, &state->detection);
    }
  } else if (is_final && state->score_frames > 0) {
    FloatTensor all_scores({1, state->score_frames, 248}, state->scores);
    const int64_t frame_index = state->score_frames - 1;
    DetectOneFrame(GetFrameState(frame_index, all_scores, state->decibel, &state->detection),
                   frame_index, true, &state->detection);
  }

  result.events_ms = CollectStreamingEvents(state);
  for (auto event : result.events_ms) {
    if (event.first >= 0 && event.second >= 0) result.segments_ms.push_back(event);
  }
  result.frame_count = state->score_frames;
  result.elapsed_ms = ElapsedMs(started_at);
  return result;
}

FsmnVadResult FsmnVadOnnxModel::FinishStreaming(StreamingState* state) const {
  std::vector<float> tail(static_cast<size_t>(kSampleRate / 100), 0.0f);
  return DetectStreaming(tail, state, true);
}

std::vector<float> FsmnVadOnnxModel::LoadCmvnValues(const std::filesystem::path& path, const char* marker) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("failed to open CMVN file: " + path.string());
  std::string line;
  bool found = false;
  while (std::getline(in, line)) {
    std::istringstream header(line);
    std::string first;
    header >> first;
    if (first != marker) continue;
    if (!std::getline(in, line)) break;
    found = true;
    break;
  }
  if (!found) throw std::runtime_error("missing CMVN marker: " + std::string(marker));

  const auto open = line.find('[');
  const auto close = line.rfind(']');
  if (open == std::string::npos || close == std::string::npos || close <= open) {
    throw std::runtime_error("invalid CMVN vector line");
  }
  std::istringstream values(line.substr(open + 1, close - open - 1));
  std::vector<float> out;
  float value = 0.0f;
  while (values >> value) out.push_back(value);
  return out;
}

std::vector<float> FsmnVadOnnxModel::ComputeDecibel(const std::vector<float>& samples) {
  if (samples.size() < kFrameSamples) return {};
  const int64_t frames = 1 + (static_cast<int64_t>(samples.size()) - kFrameSamples) / kFrameShiftSamples;
  std::vector<float> decibel(static_cast<size_t>(frames));
  for (int64_t t = 0; t < frames; ++t) {
    const size_t offset = static_cast<size_t>(t * kFrameShiftSamples);
    double energy = 0.0;
    for (int i = 0; i < kFrameSamples; ++i) {
      const double v = samples[offset + static_cast<size_t>(i)];
      energy += v * v;
    }
    decibel[static_cast<size_t>(t)] = static_cast<float>(10.0 * std::log10(energy + 1.0e-6));
  }
  return decibel;
}

FloatTensor FsmnVadOnnxModel::ComputeFbank(const std::vector<float>& samples) {
  if (samples.size() < kFrameSamples) return FloatTensor({1, 0, kNumMelBins}, {});
  const int64_t frames = 1 + (static_cast<int64_t>(samples.size()) - kFrameSamples) / kFrameShiftSamples;
  std::vector<float> values(static_cast<size_t>(frames * kNumMelBins), 0.0f);
  const auto mel_banks = MakeKaldiMelBanks();

  std::vector<float> window(kFrameSamples);
  for (int i = 0; i < kFrameSamples; ++i) {
    window[static_cast<size_t>(i)] =
        static_cast<float>(0.54 - 0.46 * std::cos(2.0 * kPi * i / static_cast<double>(kFrameSamples - 1)));
  }

  std::vector<float> fft_in(kPaddedWindowSize, 0.0f);
  std::vector<fftwf_complex> fft_out(static_cast<size_t>(kPaddedWindowSize / 2 + 1));
  fftwf_plan plan = fftwf_plan_dft_r2c_1d(kPaddedWindowSize, fft_in.data(), fft_out.data(), FFTW_ESTIMATE);
  if (!plan) throw std::runtime_error("failed to create FFTW plan for FSMN-VAD fbank");

  for (int64_t t = 0; t < frames; ++t) {
    const size_t offset = static_cast<size_t>(t * kFrameShiftSamples);
    double mean = 0.0;
    for (int i = 0; i < kFrameSamples; ++i) mean += samples[offset + static_cast<size_t>(i)] * 32768.0;
    mean /= kFrameSamples;

    float previous = static_cast<float>(samples[offset] * 32768.0 - mean);
    for (int i = 0; i < kFrameSamples; ++i) {
      const float current = static_cast<float>(samples[offset + static_cast<size_t>(i)] * 32768.0 - mean);
      const float preemphasized = current - 0.97f * (i == 0 ? current : previous);
      fft_in[static_cast<size_t>(i)] = preemphasized * window[static_cast<size_t>(i)];
      previous = current;
    }
    std::fill(fft_in.begin() + kFrameSamples, fft_in.end(), 0.0f);
    fftwf_execute(plan);

    std::vector<float> spectrum(static_cast<size_t>(kPaddedWindowSize / 2), 0.0f);
    for (int b = 0; b < kPaddedWindowSize / 2; ++b) {
      const float real = fft_out[static_cast<size_t>(b)][0];
      const float imag = fft_out[static_cast<size_t>(b)][1];
      spectrum[static_cast<size_t>(b)] = real * real + imag * imag;
    }
    for (int m = 0; m < kNumMelBins; ++m) {
      double energy = 0.0;
      for (int b = 0; b < kPaddedWindowSize / 2; ++b) {
        energy += static_cast<double>(spectrum[static_cast<size_t>(b)]) *
                  mel_banks[static_cast<size_t>(m * (kPaddedWindowSize / 2) + b)];
      }
      values[static_cast<size_t>(t * kNumMelBins + m)] =
          std::log(std::max(static_cast<float>(energy), kFloatEps));
    }
  }
  fftwf_destroy_plan(plan);
  return FloatTensor({1, frames, kNumMelBins}, std::move(values));
}

FloatTensor FsmnVadOnnxModel::ApplyLfrCmvn(const FloatTensor& fbank,
                                           const std::vector<float>& means,
                                           const std::vector<float>& vars) {
  if (fbank.shape().size() != 3 || fbank.shape()[0] != 1 || fbank.shape()[2] != kNumMelBins) {
    throw std::runtime_error("FSMN-VAD fbank must have shape [1,T,80]");
  }
  const int64_t frames = fbank.shape()[1];
  if (frames <= 0) return FloatTensor({1, 0, kInputDim}, {});
  std::vector<float> values(static_cast<size_t>(frames * kInputDim), 0.0f);
  const auto& src = fbank.values();
  for (int64_t t = 0; t < frames; ++t) {
    for (int k = 0; k < kLfrM; ++k) {
      int64_t source_frame = t + k - ((kLfrM - 1) / 2);
      source_frame = std::min<int64_t>(frames - 1, std::max<int64_t>(0, source_frame));
      for (int m = 0; m < kNumMelBins; ++m) {
        const int dim = k * kNumMelBins + m;
        float value = src[static_cast<size_t>(source_frame * kNumMelBins + m)];
        value += means[static_cast<size_t>(dim)];
        value *= vars[static_cast<size_t>(dim)];
        values[static_cast<size_t>(t * kInputDim + dim)] = value;
      }
    }
  }
  return FloatTensor({1, frames, kInputDim}, std::move(values));
}

std::vector<float> FsmnVadOnnxModel::MakeKaldiMelBanks() {
  const int num_fft_bins = kPaddedWindowSize / 2;
  const double nyquist = 0.5 * kSampleRate;
  const double low_freq = 20.0;
  const double high_freq = nyquist;
  const double fft_bin_width = static_cast<double>(kSampleRate) / kPaddedWindowSize;
  const double mel_low = MelScale(low_freq);
  const double mel_high = MelScale(high_freq);
  const double mel_delta = (mel_high - mel_low) / (kNumMelBins + 1);

  std::vector<float> bins(static_cast<size_t>(kNumMelBins * num_fft_bins), 0.0f);
  for (int m = 0; m < kNumMelBins; ++m) {
    const double left_mel = mel_low + m * mel_delta;
    const double center_mel = mel_low + (m + 1) * mel_delta;
    const double right_mel = mel_low + (m + 2) * mel_delta;
    for (int b = 0; b < num_fft_bins; ++b) {
      const double mel = MelScale(fft_bin_width * b);
      const double up = (mel - left_mel) / (center_mel - left_mel);
      const double down = (right_mel - mel) / (right_mel - center_mel);
      bins[static_cast<size_t>(m * num_fft_bins + b)] =
          static_cast<float>(std::max(0.0, std::min(up, down)));
    }
  }
  return bins;
}

double FsmnVadOnnxModel::MelScale(double freq) {
  return 1127.0 * std::log(1.0 + freq / 700.0);
}

double FsmnVadOnnxModel::InverseMelScale(double mel_freq) {
  return 700.0 * (std::exp(mel_freq / 1127.0) - 1.0);
}

FloatTensor FsmnVadOnnxModel::RunEncoder(const FloatTensor& feats) const {
  std::unordered_map<std::string, Ort::Value> inputs;
  inputs.emplace("speech", encoder_.MakeFloatInput(feats, "speech"));

  std::vector<FloatTensor> cache_tensors;
  cache_tensors.reserve(kNumFsmnLayers);
  for (int i = 0; i < kNumFsmnLayers; ++i) {
    cache_tensors.emplace_back(
        std::vector<int64_t>{1, kProjDim, kCacheFrames, 1},
        std::vector<float>(static_cast<size_t>(kProjDim * kCacheFrames), 0.0f));
    inputs.emplace("in_cache" + std::to_string(i),
                   encoder_.MakeFloatInput(cache_tensors.back(), "in_cache" + std::to_string(i)));
  }

  std::vector<std::string> outputs{"logits"};
  for (int i = 0; i < kNumFsmnLayers; ++i) outputs.push_back("out_cache" + std::to_string(i));
  auto result = encoder_.RunIo(inputs, outputs, {});
  return encoder_.CopyFloatTensor(result[0]);
}

FloatTensor FsmnVadOnnxModel::RunEncoderStreaming(const FloatTensor& feats, StreamingState* state) const {
  std::unordered_map<std::string, Ort::Value> inputs;
  inputs.emplace("speech", encoder_.MakeFloatInput(feats, "speech"));

  std::vector<FloatTensor> cache_tensors;
  cache_tensors.reserve(kNumFsmnLayers);
  const size_t one_cache_size = static_cast<size_t>(kProjDim * kCacheFrames);
  if (state->fsmn_caches.size() != static_cast<size_t>(kNumFsmnLayers) * one_cache_size) {
    state->fsmn_caches.assign(static_cast<size_t>(kNumFsmnLayers) * one_cache_size, 0.0f);
  }
  for (int i = 0; i < kNumFsmnLayers; ++i) {
    std::vector<float> cache(
        state->fsmn_caches.begin() + static_cast<std::ptrdiff_t>(i * one_cache_size),
        state->fsmn_caches.begin() + static_cast<std::ptrdiff_t>((i + 1) * one_cache_size));
    cache_tensors.emplace_back(std::vector<int64_t>{1, kProjDim, kCacheFrames, 1}, std::move(cache));
    inputs.emplace("in_cache" + std::to_string(i),
                   encoder_.MakeFloatInput(cache_tensors.back(), "in_cache" + std::to_string(i)));
  }

  std::vector<std::string> outputs{"logits"};
  for (int i = 0; i < kNumFsmnLayers; ++i) outputs.push_back("out_cache" + std::to_string(i));
  auto result = encoder_.RunIo(inputs, outputs, {});
  for (int i = 0; i < kNumFsmnLayers; ++i) {
    auto out_cache = encoder_.CopyFloatTensor(result[static_cast<size_t>(i + 1)]);
    std::copy(out_cache.values().begin(), out_cache.values().end(),
              state->fsmn_caches.begin() + static_cast<std::ptrdiff_t>(i * one_cache_size));
  }
  return encoder_.CopyFloatTensor(result[0]);
}

FloatTensor FsmnVadOnnxModel::ComputeStreamingFeatures(const std::vector<float>& samples,
                                                       bool is_final,
                                                       StreamingState* state) const {
  std::vector<float> input = state->input_cache;
  input.insert(input.end(), samples.begin(), samples.end());
  if (input.size() < kFrameSamples && !is_final) {
    state->input_cache = std::move(input);
    return FloatTensor({1, 0, kInputDim}, {});
  }

  int64_t frame_num = 0;
  if (input.size() >= kFrameSamples) {
    frame_num = 1 + (static_cast<int64_t>(input.size()) - kFrameSamples) / kFrameShiftSamples;
  }
  if (frame_num <= 0) {
    if (is_final && !state->lfr_splice_cache.empty()) {
      const int64_t cache_frames = static_cast<int64_t>(state->lfr_splice_cache.size() / kNumMelBins);
      FloatTensor cached_fbank({1, cache_frames, kNumMelBins}, state->lfr_splice_cache);
      state->lfr_splice_cache.clear();
      auto feats = ApplyLfrCmvn(cached_fbank, cmvn_means_, cmvn_vars_);
      return feats;
    }
    state->input_cache = std::move(input);
    return FloatTensor({1, 0, kInputDim}, {});
  }

  const size_t used_samples = static_cast<size_t>((frame_num - 1) * kFrameShiftSamples + kFrameSamples);
  std::vector<float> used(input.begin(), input.begin() + static_cast<std::ptrdiff_t>(used_samples));
  state->input_cache.assign(input.begin() + static_cast<std::ptrdiff_t>(frame_num * kFrameShiftSamples), input.end());

  auto fbank = ComputeFbank(used);
  auto decibels = ComputeDecibel(used);
  if (state->lfr_splice_cache.empty() && fbank.shape()[1] > 0) {
    state->lfr_splice_cache.reserve(static_cast<size_t>((kLfrM - 1) / 2 * kNumMelBins));
    for (int i = 0; i < (kLfrM - 1) / 2; ++i) {
      state->lfr_splice_cache.insert(state->lfr_splice_cache.end(),
                                     fbank.values().begin(),
                                     fbank.values().begin() + kNumMelBins);
      if (!decibels.empty()) state->decibel_cache.push_back(decibels.front());
    }
  }

  std::vector<float> lfr_input = state->lfr_splice_cache;
  lfr_input.insert(lfr_input.end(), fbank.values().begin(), fbank.values().end());
  std::vector<float> decibel_input = state->decibel_cache;
  decibel_input.insert(decibel_input.end(), decibels.begin(), decibels.end());

  const int64_t total_frames = static_cast<int64_t>(lfr_input.size() / kNumMelBins);
  if (total_frames < kLfrM && !is_final) {
    state->lfr_splice_cache = std::move(lfr_input);
    state->decibel_cache = std::move(decibel_input);
    return FloatTensor({1, 0, kInputDim}, {});
  }

  int64_t output_frames = 0;
  int64_t splice_idx = 0;
  if (is_final) {
    output_frames = std::max<int64_t>(0, total_frames - (kLfrM - 1) / 2);
    splice_idx = total_frames;
  } else {
    output_frames = std::max<int64_t>(0, total_frames - kLfrM + 1);
    splice_idx = output_frames;
  }
  if (output_frames <= 0) {
    state->lfr_splice_cache = std::move(lfr_input);
    state->decibel_cache = std::move(decibel_input);
    return FloatTensor({1, 0, kInputDim}, {});
  }

  std::vector<float> values(static_cast<size_t>(output_frames * kInputDim), 0.0f);
  for (int64_t t = 0; t < output_frames; ++t) {
    for (int k = 0; k < kLfrM; ++k) {
      int64_t src_frame = t + k;
      if (src_frame >= total_frames) src_frame = total_frames - 1;
      for (int m = 0; m < kNumMelBins; ++m) {
        const int dim = k * kNumMelBins + m;
        float value = lfr_input[static_cast<size_t>(src_frame * kNumMelBins + m)];
        value += cmvn_means_[static_cast<size_t>(dim)];
        value *= cmvn_vars_[static_cast<size_t>(dim)];
        values[static_cast<size_t>(t * kInputDim + dim)] = value;
      }
    }
  }

  for (int64_t t = 0; t < output_frames && t < static_cast<int64_t>(decibel_input.size()); ++t) {
    state->decibel.push_back(decibel_input[static_cast<size_t>(t)]);
  }

  const int64_t keep_from = std::min<int64_t>(splice_idx, total_frames);
  const int64_t keep_from_value = keep_from * kNumMelBins;
  state->lfr_splice_cache.assign(lfr_input.begin() + static_cast<std::ptrdiff_t>(keep_from_value), lfr_input.end());
  if (keep_from < static_cast<int64_t>(decibel_input.size())) {
    state->decibel_cache.assign(decibel_input.begin() + static_cast<std::ptrdiff_t>(keep_from), decibel_input.end());
  } else {
    state->decibel_cache.clear();
  }

  return FloatTensor({1, output_frames, kInputDim}, std::move(values));
}

std::vector<std::pair<int64_t, int64_t>> FsmnVadOnnxModel::CollectStreamingEvents(StreamingState* state) const {
  std::vector<std::pair<int64_t, int64_t>> events;
  auto& stats = state->detection;
  for (int64_t i = state->output_data_buf_offset;
       i < static_cast<int64_t>(stats.output_data_buf.size()); ++i) {
    const auto& segment = stats.output_data_buf[static_cast<size_t>(i)];
    if (!segment.contain_start) continue;
    if (!stats.next_seg && !segment.contain_end) continue;
    int64_t start_ms = stats.next_seg ? segment.start_ms : -1;
    int64_t end_ms = -1;
    if (segment.contain_end) {
      end_ms = segment.end_ms;
      stats.next_seg = true;
      state->output_data_buf_offset = i + 1;
    } else {
      stats.next_seg = false;
    }
    events.push_back({start_ms, end_ms});
  }
  return events;
}

std::vector<std::pair<int64_t, int64_t>> FsmnVadOnnxModel::PostProcess(
    const FloatTensor& scores,
    const std::vector<float>& decibel) const {
  if (scores.shape().size() != 3 || scores.shape()[0] != 1) {
    throw std::runtime_error("FSMN-VAD logits must have shape [1,T,D]");
  }
  const int64_t frames = scores.shape()[1];
  DetectionState state;
  state.window = WindowDetector(kWindowSizeMs, kSilToSpeechTimeMs, kSpeechToSilTimeMs, kFrameMs);
  state.max_end_sil_frame_count_threshold_ms = kMaxEndSilenceTimeMs - kSpeechToSilTimeMs;
  state.speech_noise_thres = kSpeechNoiseThres;

  for (int64_t t = 0; t < frames; ++t) {
    const auto frame_state = GetFrameState(t, scores, decibel, &state);
    DetectOneFrame(frame_state, t, t == frames - 1, &state);
  }

  std::vector<std::pair<int64_t, int64_t>> segments;
  for (const auto& item : state.output_data_buf) {
    if (item.contain_start && item.contain_end && item.end_ms >= item.start_ms) {
      segments.push_back({item.start_ms, item.end_ms});
    }
  }
  return segments;
}

FsmnVadOnnxModel::FrameState FsmnVadOnnxModel::GetFrameState(int64_t frame_index,
                                                             const FloatTensor& scores,
                                                             const std::vector<float>& decibel,
                                                             DetectionState* state) const {
  if (frame_index < 0 || frame_index >= static_cast<int64_t>(decibel.size())) return FrameState::kSil;
  const float cur_decibel = decibel[static_cast<size_t>(frame_index)];
  const float cur_snr = static_cast<float>(cur_decibel - state->noise_average_decibel);
  if (cur_decibel < kDecibelThres) {
    DetectOneFrame(FrameState::kSil, frame_index, false, state);
    return FrameState::kSil;
  }

  const int64_t t = std::min<int64_t>(frame_index, scores.shape()[1] - 1);
  const int64_t dim = scores.shape()[2];
  const float silence_score = ClampProb(scores.values()[static_cast<size_t>(t * dim)]);
  const float noise_prob = std::log(silence_score) * kSpeech2NoiseRatio;
  const float speech_score = ClampProb(1.0f - silence_score);
  const float speech_prob = std::log(speech_score);

  if (std::exp(speech_prob) >= std::exp(noise_prob) + state->speech_noise_thres) {
    if (cur_snr >= kSnrThres && cur_decibel >= kDecibelThres) return FrameState::kSpeech;
    return FrameState::kSil;
  }

  if (state->noise_average_decibel < -99.9) {
    state->noise_average_decibel = cur_decibel;
  } else {
    state->noise_average_decibel =
        (cur_decibel + state->noise_average_decibel * (kNoiseFrameNumUsedForSnr - 1)) /
        kNoiseFrameNumUsedForSnr;
  }
  return FrameState::kSil;
}

void FsmnVadOnnxModel::DetectOneFrame(FrameState frame_state,
                                      int64_t frame_index,
                                      bool is_final_frame,
                                      DetectionState* state) const {
  FrameState tmp_state = FrameState::kInvalid;
  if (frame_state == FrameState::kSpeech) {
    tmp_state = std::fabs(1.0) > kFePriorThres ? FrameState::kSpeech : FrameState::kSil;
  } else if (frame_state == FrameState::kSil) {
    tmp_state = FrameState::kSil;
  }

  const auto change = state->window.DetectOneFrame(tmp_state);
  if (change == AudioChangeState::kSil2Speech) {
    state->continuous_silence_frame_count = 0;
    state->pre_end_silence_detected = false;
    if (state->vad_state_machine == VadStateMachine::kStartPointNotDetected) {
      const int64_t start_frame =
          std::max<int64_t>(state->data_buf_start_frame, frame_index - LatencyFrameNumAtStartPoint(*state));
      OnVoiceStart(start_frame, false, state);
      state->vad_state_machine = VadStateMachine::kInSpeechSegment;
      for (int64_t t = start_frame + 1; t <= frame_index; ++t) OnVoiceDetected(t, state);
    } else if (state->vad_state_machine == VadStateMachine::kInSpeechSegment) {
      for (int64_t t = state->latest_confirmed_speech_frame + 1; t < frame_index; ++t) OnVoiceDetected(t, state);
      if (frame_index - state->confirmed_start_frame + 1 > kMaxSingleSegmentTimeMs / kFrameMs) {
        OnVoiceEnd(frame_index, false, false, state);
        state->vad_state_machine = VadStateMachine::kEndPointDetected;
      } else if (!is_final_frame) {
        OnVoiceDetected(frame_index, state);
      } else {
        MaybeOnVoiceEndIfLastFrame(is_final_frame, frame_index, state);
      }
    }
  } else if (change == AudioChangeState::kSpeech2Sil) {
    state->continuous_silence_frame_count = 0;
    if (state->vad_state_machine == VadStateMachine::kInSpeechSegment) {
      if (frame_index - state->confirmed_start_frame + 1 > kMaxSingleSegmentTimeMs / kFrameMs) {
        OnVoiceEnd(frame_index, false, false, state);
        state->vad_state_machine = VadStateMachine::kEndPointDetected;
      } else if (!is_final_frame) {
        OnVoiceDetected(frame_index, state);
      } else {
        MaybeOnVoiceEndIfLastFrame(is_final_frame, frame_index, state);
      }
    }
  } else if (change == AudioChangeState::kSpeech2Speech) {
    state->continuous_silence_frame_count = 0;
    if (state->vad_state_machine == VadStateMachine::kInSpeechSegment) {
      if (frame_index - state->confirmed_start_frame + 1 > kMaxSingleSegmentTimeMs / kFrameMs) {
        state->max_time_out = true;
        OnVoiceEnd(frame_index, false, false, state);
        state->vad_state_machine = VadStateMachine::kEndPointDetected;
      } else if (!is_final_frame) {
        OnVoiceDetected(frame_index, state);
      } else {
        MaybeOnVoiceEndIfLastFrame(is_final_frame, frame_index, state);
      }
    }
  } else if (change == AudioChangeState::kSil2Sil) {
    state->continuous_silence_frame_count += 1;
    if (state->vad_state_machine == VadStateMachine::kStartPointNotDetected) {
      const bool start_timeout = false;
      if ((start_timeout && state->continuous_silence_frame_count * kFrameMs > kMaxStartSilenceTimeMs) ||
          (is_final_frame && state->number_end_time_detected == 0)) {
        for (int64_t t = state->latest_confirmed_silence_frame + 1; t < frame_index; ++t) {
          OnSilenceDetected(t, state);
        }
        OnVoiceStart(0, true, state);
        OnVoiceEnd(0, true, false, state);
        state->vad_state_machine = VadStateMachine::kEndPointDetected;
      } else if (frame_index >= LatencyFrameNumAtStartPoint(*state)) {
        OnSilenceDetected(frame_index - LatencyFrameNumAtStartPoint(*state), state);
      }
    } else if (state->vad_state_machine == VadStateMachine::kInSpeechSegment) {
      if (state->continuous_silence_frame_count * kFrameMs >= state->max_end_sil_frame_count_threshold_ms) {
        int64_t lookback_frame = state->max_end_sil_frame_count_threshold_ms / kFrameMs;
        lookback_frame -= kLookaheadTimeEndPointMs / kFrameMs;
        lookback_frame -= 1;
        lookback_frame = std::max<int64_t>(0, lookback_frame);
        OnVoiceEnd(frame_index - lookback_frame, false, false, state);
        state->vad_state_machine = VadStateMachine::kEndPointDetected;
      } else if (frame_index - state->confirmed_start_frame + 1 > kMaxSingleSegmentTimeMs / kFrameMs) {
        OnVoiceEnd(frame_index, false, false, state);
        state->vad_state_machine = VadStateMachine::kEndPointDetected;
      } else if (state->continuous_silence_frame_count <= kLookaheadTimeEndPointMs / kFrameMs && !is_final_frame) {
        OnVoiceDetected(frame_index, state);
      } else {
        MaybeOnVoiceEndIfLastFrame(is_final_frame, frame_index, state);
      }
    }
  }

  if (state->vad_state_machine == VadStateMachine::kEndPointDetected &&
      kDetectModeMultipleUtterance == 1) {
    ResetDetection(state);
  }
}

void FsmnVadOnnxModel::OnSilenceDetected(int64_t valid_frame, DetectionState* state) const {
  state->latest_confirmed_silence_frame = valid_frame;
  if (state->vad_state_machine == VadStateMachine::kStartPointNotDetected) {
    state->data_buf_start_frame = std::max(state->data_buf_start_frame, valid_frame);
  }
}

void FsmnVadOnnxModel::OnVoiceDetected(int64_t valid_frame, DetectionState* state) const {
  state->latest_confirmed_speech_frame = valid_frame;
  if (state->output_data_buf.empty()) return;
  state->output_data_buf.back().end_ms = FrameToMs(valid_frame + 1);
}

void FsmnVadOnnxModel::OnVoiceStart(int64_t start_frame, bool fake_result, DetectionState* state) const {
  if (state->confirmed_start_frame != -1) {
    throw std::runtime_error("FSMN-VAD state was not reset before OnVoiceStart");
  }
  state->confirmed_start_frame = start_frame;
  if (!fake_result && state->vad_state_machine == VadStateMachine::kStartPointNotDetected) {
    SegmentBuf segment;
    segment.start_ms = FrameToMs(start_frame);
    segment.end_ms = segment.start_ms;
    segment.contain_start = true;
    state->output_data_buf.push_back(segment);
  }
}

void FsmnVadOnnxModel::OnVoiceEnd(int64_t end_frame,
                                  bool fake_result,
                                  bool is_last_frame,
                                  DetectionState* state) const {
  for (int64_t t = state->latest_confirmed_speech_frame + 1; t < end_frame; ++t) OnVoiceDetected(t, state);
  if (state->confirmed_end_frame != -1) {
    throw std::runtime_error("FSMN-VAD state was not reset before OnVoiceEnd");
  }
  state->confirmed_end_frame = end_frame;
  if (!fake_result) {
    state->sil_frame = 0;
    if (!state->output_data_buf.empty()) {
      state->output_data_buf.back().end_ms = FrameToMs(end_frame + 1);
      state->output_data_buf.back().contain_end = true;
    }
    (void)is_last_frame;
  }
  state->number_end_time_detected += 1;
}

void FsmnVadOnnxModel::MaybeOnVoiceEndIfLastFrame(bool is_final_frame,
                                                  int64_t frame_index,
                                                  DetectionState* state) const {
  if (is_final_frame) {
    OnVoiceEnd(frame_index, false, true, state);
    state->vad_state_machine = VadStateMachine::kEndPointDetected;
  }
}

void FsmnVadOnnxModel::ResetDetection(DetectionState* state) const {
  int64_t drop_frames = state->data_buf_start_frame;
  const auto output = state->output_data_buf;
  if (!output.empty()) drop_frames = std::max<int64_t>(drop_frames, output.back().end_ms / kFrameMs);

  state->continuous_silence_frame_count = 0;
  state->latest_confirmed_speech_frame = 0;
  state->latest_confirmed_silence_frame = -1;
  state->confirmed_start_frame = -1;
  state->confirmed_end_frame = -1;
  state->vad_state_machine = VadStateMachine::kStartPointNotDetected;
  state->window.Reset();
  state->sil_frame = 0;
  state->data_buf_start_frame = drop_frames;
  state->output_data_buf = output;
}

int FsmnVadOnnxModel::LatencyFrameNumAtStartPoint(const DetectionState& state) const {
  return state.window.WinSize() + kLookbackTimeStartPointMs / kFrameMs;
}

}  // namespace qwen3asr
