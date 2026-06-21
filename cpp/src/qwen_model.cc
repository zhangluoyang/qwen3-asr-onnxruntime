#include "qwen3asr/qwen_model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <fftw3.h>

#include "qwen3asr/audio_frontend.h"

namespace qwen3asr {
namespace {

constexpr int kSampleRate = 16000;
constexpr int kNfft = 400;
constexpr int kHopLength = 160;
constexpr int kNumMels = 128;
constexpr int64_t kAudioTokenId = 151676;
constexpr int64_t kEosTokenId = 151645;
constexpr int64_t kEndOfTextTokenId = 151643;
constexpr int64_t kHiddenSize = 2048;
constexpr int64_t kNumLayers = 28;
constexpr int64_t kNumKvHeads = 8;
constexpr int64_t kHeadDim = 128;
constexpr double kMaxAsrInputSeconds = 1200.0;
constexpr double kMinAsrInputSeconds = 0.5;
constexpr double kPi = 3.14159265358979323846;

std::string Trim(std::string value) {
  auto not_space = [](unsigned char c) { return !std::isspace(c); };
  value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
  value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
  return value;
}

double HzToMel(double hz) {
  constexpr double f_min = 0.0;
  constexpr double f_sp = 200.0 / 3.0;
  double mel = (hz - f_min) / f_sp;
  constexpr double min_log_hz = 1000.0;
  constexpr double min_log_mel = (min_log_hz - f_min) / f_sp;
  constexpr double logstep = 1.8562979903656263 / 27.0;
  if (hz >= min_log_hz) mel = min_log_mel + std::log(hz / min_log_hz) / logstep;
  return mel;
}

double MelToHz(double mel) {
  constexpr double f_min = 0.0;
  constexpr double f_sp = 200.0 / 3.0;
  double hz = f_min + f_sp * mel;
  constexpr double min_log_hz = 1000.0;
  constexpr double min_log_mel = (min_log_hz - f_min) / f_sp;
  constexpr double logstep = 1.8562979903656263 / 27.0;
  if (mel >= min_log_mel) hz = min_log_hz * std::exp(logstep * (mel - min_log_mel));
  return hz;
}

std::vector<float> ReflectPad(const std::vector<float>& x, int pad) {
  if (x.empty()) return {};
  std::vector<float> out(x.size() + static_cast<size_t>(2 * pad));
  const int n = static_cast<int>(x.size());
  for (int i = 0; i < pad; ++i) {
    int idx = pad - i;
    if (idx >= n) idx = n - 1;
    out[static_cast<size_t>(i)] = x[static_cast<size_t>(idx)];
  }
  std::copy(x.begin(), x.end(), out.begin() + pad);
  for (int i = 0; i < pad; ++i) {
    int idx = n - 2 - i;
    if (idx < 0) idx = 0;
    out[static_cast<size_t>(pad + n + i)] = x[static_cast<size_t>(idx)];
  }
  return out;
}

std::vector<float> MakeWhisperMelFilters() {
  const int bins = kNfft / 2 + 1;
  const double min_mel = HzToMel(0.0);
  const double max_mel = HzToMel(8000.0);
  std::vector<double> mel_hz(static_cast<size_t>(kNumMels + 2));
  for (int i = 0; i < kNumMels + 2; ++i) {
    mel_hz[static_cast<size_t>(i)] = MelToHz(min_mel + (max_mel - min_mel) * i / (kNumMels + 1));
  }

  std::vector<float> filters(static_cast<size_t>(kNumMels * bins), 0.0f);
  for (int m = 0; m < kNumMels; ++m) {
    const double lower_hz = mel_hz[static_cast<size_t>(m)];
    const double center_hz = mel_hz[static_cast<size_t>(m + 1)];
    const double upper_hz = mel_hz[static_cast<size_t>(m + 2)];
    const double enorm = 2.0 / (upper_hz - lower_hz);
    for (int b = 0; b < bins; ++b) {
      const double freq = static_cast<double>(b) * kSampleRate / kNfft;
      const double lower = (freq - lower_hz) / (center_hz - lower_hz);
      const double upper = (upper_hz - freq) / (upper_hz - center_hz);
      filters[static_cast<size_t>(m * bins + b)] = static_cast<float>(std::max(0.0, std::min(lower, upper)) * enorm);
    }
  }
  return filters;
}

const std::vector<float>& HannWindow() {
  static const std::vector<float> window = [] {
    std::vector<float> values(static_cast<size_t>(kNfft));
    for (int i = 0; i < kNfft; ++i) {
      values[static_cast<size_t>(i)] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * kPi * i / kNfft));
    }
    return values;
  }();
  return window;
}

const std::vector<std::vector<std::pair<int, float>>>& SparseMelFilters() {
  static const std::vector<std::vector<std::pair<int, float>>> sparse = [] {
    const auto dense = MakeWhisperMelFilters();
    const int bins = kNfft / 2 + 1;
    std::vector<std::vector<std::pair<int, float>>> out(static_cast<size_t>(kNumMels));
    for (int m = 0; m < kNumMels; ++m) {
      auto& row = out[static_cast<size_t>(m)];
      for (int b = 0; b < bins; ++b) {
        const float value = dense[static_cast<size_t>(m * bins + b)];
        if (value != 0.0f) row.emplace_back(b, value);
      }
    }
    return out;
  }();
  return sparse;
}

struct WhisperFftWorkspace {
  WhisperFftWorkspace() : power(static_cast<size_t>(kNfft / 2 + 1)) {}

  ~WhisperFftWorkspace() {
    if (plan) fftwf_destroy_plan(plan);
  }

  WhisperFftWorkspace(const WhisperFftWorkspace&) = delete;
  WhisperFftWorkspace& operator=(const WhisperFftWorkspace&) = delete;

  void EnsureFrames(int frames) {
    if (frames == planned_frames && plan) return;
    if (plan) {
      fftwf_destroy_plan(plan);
      plan = nullptr;
    }
    planned_frames = frames;
    frame_batch.resize(static_cast<size_t>(frames) * kNfft);
    fft_out_batch.resize(static_cast<size_t>(frames) * (kNfft / 2 + 1) * 2);
    int n[] = {kNfft};
    plan = fftwf_plan_many_dft_r2c(1, n, frames,
                                   frame_batch.data(), nullptr, 1, kNfft,
                                   reinterpret_cast<fftwf_complex*>(fft_out_batch.data()),
                                   nullptr, 1, kNfft / 2 + 1,
                                   FFTW_ESTIMATE);
    if (!plan) throw std::runtime_error("failed to create batched FFTW plan");
  }

  int planned_frames = 0;
  std::vector<float> frame_batch;
  std::vector<float> power;
  std::vector<float> fft_out_batch;
  fftwf_plan plan = nullptr;
};

}  // namespace

Qwen3ASROnnxModel::Qwen3ASROnnxModel(Qwen3ASRConfig config)
    : config_(std::move(config)),
      env_(ORT_LOGGING_LEVEL_WARNING, "qwen3asr_cpp"),
      tokenizer_(config_.model_dir),
      audio_encoder_(env_, OrtRunnerOptions{config_.onnx_dir / "audio_encoder" / "audio_encoder.onnx",
                                            config_.use_cuda, config_.cuda_device_id}),
      token_embedding_(env_, OrtRunnerOptions{config_.onnx_dir / "token_embedding" / "token_embedding.onnx",
                                             false, config_.cuda_device_id}),
      text_core_(env_, OrtRunnerOptions{config_.onnx_dir / "text_core" / "asr_text_core.onnx",
                                        config_.use_cuda, config_.cuda_device_id}) {
  text_core_output_names_.push_back("logits");
  for (int64_t i = 0; i < kNumLayers; ++i) {
    const std::string key_name = "new_past_key_" + std::to_string(i);
    const std::string value_name = "new_past_value_" + std::to_string(i);
    text_core_output_names_.push_back(key_name);
    text_core_output_names_.push_back(value_name);
    text_core_device_output_names_.insert(key_name);
    text_core_device_output_names_.insert(value_name);
  }
}

ASRResult Qwen3ASROnnxModel::TranscribeFile(const std::filesystem::path& audio_path,
                                            const std::string& context,
                                            const std::optional<std::string>& language,
                                            const std::string& hotwords,
                                            int max_new_tokens) {
  auto audio = LoadAudioMono(audio_path, kSampleRate);
  return TranscribeSamples(audio.samples, context, language, hotwords, max_new_tokens);
}

ASRResult Qwen3ASROnnxModel::TranscribeSamples(const std::vector<float>& samples,
                                               const std::string& context,
                                               const std::optional<std::string>& language,
                                               const std::string& hotwords,
                                               int max_new_tokens) const {
  const auto started_at = std::chrono::steady_clock::now();
  const int limit = max_new_tokens > 0 ? max_new_tokens : config_.max_new_tokens;
  const auto chunks = SplitAudioIntoChunks(samples);
  ASRResult merged;
  merged.chunk_count = static_cast<int64_t>(chunks.size());
  std::string previous_language;
  for (const auto& chunk : chunks) {
    auto result = TranscribeChunk(chunk, context, language, hotwords, limit);
    merged.text += result.text;
    merged.raw_text += result.raw_text;
    merged.audio_tokens += result.audio_tokens;
    merged.mel_frames += result.mel_frames;
    const auto lang = Trim(result.language);
    if (!lang.empty() && lang != previous_language) {
      if (!merged.language.empty()) merged.language += ",";
      merged.language += lang;
      previous_language = lang;
    }
  }
  merged.elapsed_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started_at).count();
  return merged;
}

ASRResult Qwen3ASROnnxModel::TranscribeChunk(const std::vector<float>& samples,
                                             const std::string& context,
                                             const std::optional<std::string>& language,
                                             const std::string& hotwords,
                                             int max_new_tokens,
                                             const std::string& output_prefix) const {
  const auto prepare_started_at = std::chrono::steady_clock::now();
  auto inputs = PrepareInputs(samples, context, language, hotwords, output_prefix);
  const double prepare_inputs_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - prepare_started_at).count();
  const int limit = max_new_tokens > 0 ? max_new_tokens : config_.max_new_tokens;
  const auto generate_started_at = std::chrono::steady_clock::now();
  auto generated = GenerateIds(inputs, limit);
  const double generate_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - generate_started_at).count();
  ASRResult result = ParseOutput(output_prefix + tokenizer_.Decode(generated.ids, false), language);
  result.audio_tokens = inputs.audio_tokens;
  result.mel_frames = inputs.mel_frames;
  result.chunk_count = 1;
  result.prepare_inputs_ms = prepare_inputs_ms;
  result.generate_ms = generate_ms;
  result.token_embedding_ms = generated.token_embedding_ms;
  result.audio_encoder_ms = generated.audio_encoder_ms;
  result.merge_audio_features_ms = generated.merge_audio_features_ms;
  result.prefill_ms = generated.prefill_ms;
  result.decode_ms = generated.decode_ms;
  result.decode_steps = generated.decode_steps;
  return result;
}

ASRResult Qwen3ASROnnxModel::TranscribeSamplesWithPrefix(const std::vector<float>& samples,
                                                         const std::string& context,
                                                         const std::optional<std::string>& language,
                                                         const std::string& hotwords,
                                                         const std::string& output_prefix,
                                                         int max_new_tokens) const {
  const auto started_at = std::chrono::steady_clock::now();
  const int limit = max_new_tokens > 0 ? max_new_tokens : config_.max_new_tokens;
  auto result = TranscribeChunk(samples, context, language, hotwords, limit, output_prefix);
  result.elapsed_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started_at).count();
  return result;
}

ASRResult Qwen3ASROnnxModel::ParseRawText(const std::string& raw_text,
                                          const std::optional<std::string>& language) const {
  return ParseOutput(raw_text, language);
}

std::vector<int64_t> Qwen3ASROnnxModel::EncodeText(const std::string& text) const {
  return tokenizer_.Encode(text);
}

std::string Qwen3ASROnnxModel::DecodeTokenIds(const std::vector<int64_t>& ids,
                                              bool skip_special_tokens) const {
  return tokenizer_.Decode(ids, skip_special_tokens);
}

void Qwen3ASROnnxModel::WarmUp() const {
  // Exercise the main streaming shapes before realtime accounting starts:
  // short utterance, mid utterance, and the capped 10s history window.
  for (double seconds : {1.5, 5.0, 10.0}) {
    std::vector<float> silence(static_cast<size_t>(seconds * kSampleRate), 0.0f);
    (void)TranscribeChunk(silence, "", std::nullopt, "", 1);
  }
}

FloatTensor Qwen3ASROnnxModel::DebugPrefillEmbedsForFile(const std::filesystem::path& audio_path,
                                                         const std::string& context,
                                                         const std::optional<std::string>& language,
                                                         const std::string& hotwords) const {
  auto audio = LoadAudioMono(audio_path, kSampleRate);
  auto inputs = PrepareInputs(audio.samples, context, language, hotwords);
  auto token_embeds = RunTokenEmbedding(inputs.input_ids);
  auto audio_features = RunAudioEncoder(inputs.input_features);
  return MergeAudioFeatures(inputs.input_ids, token_embeds, audio_features);
}

std::string Qwen3ASROnnxModel::ContextWithHotwords(const std::string& context, const std::string& hotwords) {
  const auto clean_hotwords = Trim(hotwords);
  if (clean_hotwords.empty()) return context;
  const std::string hint = "热词：" + clean_hotwords + "。仅在语音内容匹配时优先使用这些词。";
  const auto clean_context = Trim(context);
  return clean_context.empty() ? hint : clean_context + "\n" + hint;
}

std::string Qwen3ASROnnxModel::BuildPrompt(const std::string& context,
                                           int64_t audio_tokens,
                                           const std::optional<std::string>& language,
                                           const std::string& output_prefix) {
  std::string prompt = "<|im_start|>system\n" + context + "<|im_end|>\n";
  prompt += "<|im_start|>user\n<|audio_start|>";
  for (int64_t i = 0; i < audio_tokens; ++i) prompt += "<|audio_pad|>";
  prompt += "<|audio_end|><|im_end|>\n<|im_start|>assistant\n";
  if (language && !language->empty()) prompt += "language " + *language + "<asr_text>";
  if (!output_prefix.empty()) prompt += output_prefix;
  return prompt;
}

int64_t Qwen3ASROnnxModel::AudioTokenCount(int64_t mel_frames) {
  auto py_floor_div = [](int64_t a, int64_t b) {
    int64_t q = a / b;
    int64_t r = a % b;
    if (r != 0 && ((r > 0) != (b > 0))) --q;
    return q;
  };
  const int64_t leave = mel_frames % 100;
  const int64_t feat_lengths = py_floor_div(leave - 1, 2) + 1;
  return py_floor_div(py_floor_div(feat_lengths - 1, 2) + 1 - 1, 2) + 1 + (mel_frames / 100) * 13;
}

std::vector<std::vector<float>> Qwen3ASROnnxModel::SplitAudioIntoChunks(const std::vector<float>& samples) {
  const int64_t total_len = static_cast<int64_t>(samples.size());
  const int64_t max_len = static_cast<int64_t>(kMaxAsrInputSeconds * kSampleRate);
  const int64_t min_len = static_cast<int64_t>(kMinAsrInputSeconds * kSampleRate);
  if (total_len <= max_len) {
    std::vector<float> one = samples;
    if (static_cast<int64_t>(one.size()) < min_len) one.resize(static_cast<size_t>(min_len), 0.0f);
    return {std::move(one)};
  }

  const int64_t expand = static_cast<int64_t>(5.0 * kSampleRate);
  const int64_t win = std::max<int64_t>(4, static_cast<int64_t>(0.1 * kSampleRate));
  std::vector<std::vector<float>> chunks;
  int64_t start = 0;
  while (total_len - start > max_len) {
    const int64_t cut = start + max_len;
    const int64_t left = std::max<int64_t>(start, cut - expand);
    const int64_t right = std::min<int64_t>(total_len, cut + expand);
    int64_t boundary = cut;
    if (right - left > win) {
      std::vector<float> prefix(static_cast<size_t>(right - left + 1), 0.0f);
      for (int64_t i = left; i < right; ++i) {
        prefix[static_cast<size_t>(i - left + 1)] = prefix[static_cast<size_t>(i - left)] + std::abs(samples[i]);
      }
      float best = std::numeric_limits<float>::infinity();
      int64_t best_pos = 0;
      for (int64_t pos = 0; pos <= right - left - win; ++pos) {
        const float sum = prefix[static_cast<size_t>(pos + win)] - prefix[static_cast<size_t>(pos)];
        if (sum < best) {
          best = sum;
          best_pos = pos;
        }
      }
      float inner_best = std::numeric_limits<float>::infinity();
      int64_t inner = 0;
      for (int64_t i = 0; i < win; ++i) {
        const float value = std::abs(samples[static_cast<size_t>(left + best_pos + i)]);
        if (value < inner_best) {
          inner_best = value;
          inner = i;
        }
      }
      boundary = left + best_pos + inner;
    }
    boundary = std::max<int64_t>(boundary, start + 1);
    boundary = std::min<int64_t>(boundary, total_len);
    chunks.emplace_back(samples.begin() + start, samples.begin() + boundary);
    start = boundary;
  }
  chunks.emplace_back(samples.begin() + start, samples.end());
  for (auto& chunk : chunks) {
    if (static_cast<int64_t>(chunk.size()) < min_len) chunk.resize(static_cast<size_t>(min_len), 0.0f);
  }
  return chunks;
}

FloatTensor Qwen3ASROnnxModel::WhisperLogMel(const std::vector<float>& audio) {
  if (audio.empty()) throw std::invalid_argument("cannot transcribe empty audio");
  const auto padded = ReflectPad(audio, kNfft / 2);
  const int raw_frames = std::max<int>(0, (static_cast<int>(padded.size()) - kNfft) / kHopLength + 1);
  const int frames = std::max(0, raw_frames - 1);
  const int bins = kNfft / 2 + 1;
  const auto& filters = SparseMelFilters();
  const auto& window = HannWindow();

  FloatTensor features({1, kNumMels, frames});
  thread_local WhisperFftWorkspace workspace;
  workspace.EnsureFrames(frames);
  auto& frame_batch = workspace.frame_batch;
  auto& power = workspace.power;
  auto& fft_out_batch = workspace.fft_out_batch;

  for (int t = 0; t < frames; ++t) {
    const int start = t * kHopLength;
    float* frame = frame_batch.data() + static_cast<size_t>(t) * kNfft;
    for (int i = 0; i < kNfft; ++i) {
      frame[static_cast<size_t>(i)] = padded[static_cast<size_t>(start + i)] * window[static_cast<size_t>(i)];
    }
  }

  fftwf_execute(workspace.plan);

  float max_log = -std::numeric_limits<float>::infinity();
  for (int t = 0; t < frames; ++t) {
    const auto* fft_out = reinterpret_cast<const fftwf_complex*>(fft_out_batch.data()) + static_cast<size_t>(t) * bins;
    for (int b = 0; b < bins; ++b) {
      const float real = fft_out[static_cast<size_t>(b)][0];
      const float imag = fft_out[static_cast<size_t>(b)][1];
      power[static_cast<size_t>(b)] = real * real + imag * imag;
    }
    for (int m = 0; m < kNumMels; ++m) {
      float sum = 0.0f;
      for (const auto& [b, weight] : filters[static_cast<size_t>(m)]) {
        sum += weight * power[static_cast<size_t>(b)];
      }
      const float log_value = std::log10(std::max(sum, 1.0e-10f));
      features.values()[static_cast<size_t>(m * frames + t)] = log_value;
      max_log = std::max(max_log, log_value);
    }
  }
  const float floor = max_log - 8.0f;
  for (float& value : features.values()) value = (std::max(value, floor) + 4.0f) / 4.0f;
  return features;
}

ASRResult Qwen3ASROnnxModel::ParseOutput(const std::string& raw_text,
                                         const std::optional<std::string>& forced_language) {
  ASRResult result;
  result.raw_text = raw_text;
  result.language = forced_language.value_or("");
  result.text = raw_text;
  const auto marker = raw_text.find("<asr_text>");
  if (marker != std::string::npos) {
    result.text = raw_text.substr(marker + std::string("<asr_text>").size());
    const auto language_pos = raw_text.find("language ");
    if (!forced_language && language_pos != std::string::npos && language_pos < marker) {
      result.language = Trim(raw_text.substr(language_pos + 9, marker - (language_pos + 9)));
    }
  }
  return result;
}

int64_t Qwen3ASROnnxModel::GreedyArgmax(const std::vector<float>& logits) {
  return static_cast<int64_t>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
}

Qwen3ASROnnxModel::PreparedInputs Qwen3ASROnnxModel::PrepareInputs(const std::vector<float>& samples,
                                                                   const std::string& context,
                                                                   const std::optional<std::string>& language,
                                                                   const std::string& hotwords,
                                                                   const std::string& output_prefix) const {
  PreparedInputs out;
  out.input_features = WhisperLogMel(samples);
  out.mel_frames = out.input_features.shape()[2];
  out.audio_tokens = AudioTokenCount(out.mel_frames);
  const auto prompt = BuildPrompt(ContextWithHotwords(context, hotwords), out.audio_tokens, language, output_prefix);
  auto ids = tokenizer_.Encode(prompt);
  const int64_t id_count = static_cast<int64_t>(ids.size());
  out.input_ids = Int64Tensor({1, id_count}, std::move(ids));
  out.attention_mask = Int64Tensor({1, out.input_ids.shape()[1]},
                                   std::vector<int64_t>(static_cast<size_t>(out.input_ids.shape()[1]), 1));
  return out;
}

FloatTensor Qwen3ASROnnxModel::RunAudioEncoder(const FloatTensor& input_features) const {
  std::unordered_map<std::string, Ort::Value> feed;
  feed.emplace("input_features", audio_encoder_.MakeFloatInput(input_features, "input_features"));
  const std::vector<int64_t> len_values{input_features.shape()[2]};
  feed.emplace("feature_lens", audio_encoder_.MakeInt64Input({1}, len_values));
  auto outputs = audio_encoder_.RunIo(feed, {"audio_features"}, {});
  return audio_encoder_.CopyFloatTensor(outputs[0]);
}

FloatTensor Qwen3ASROnnxModel::RunTokenEmbedding(const Int64Tensor& input_ids) const {
  std::unordered_map<std::string, Ort::Value> feed;
  feed.emplace("input_ids", token_embedding_.MakeInt64Input(input_ids));
  auto outputs = token_embedding_.RunIo(feed, {"inputs_embeds"}, {});
  return token_embedding_.CopyFloatTensor(outputs[0]);
}

FloatTensor Qwen3ASROnnxModel::MergeAudioFeatures(const Int64Tensor& input_ids,
                                                  const FloatTensor& token_embeds,
                                                  const FloatTensor& audio_features) const {
  if (token_embeds.shape().size() != 3 || audio_features.shape().size() != 2) {
    throw std::runtime_error("unexpected embed/audio feature rank");
  }
  FloatTensor merged(token_embeds.shape(), token_embeds.values());
  int64_t audio_index = 0;
  const int64_t seq_len = token_embeds.shape()[1];
  const int64_t hidden = token_embeds.shape()[2];
  for (int64_t i = 0; i < seq_len; ++i) {
    if (input_ids.values()[static_cast<size_t>(i)] != kAudioTokenId) continue;
    if (audio_index >= audio_features.shape()[0]) {
      throw std::runtime_error("too many audio placeholder tokens: placeholders_seen=" +
                               std::to_string(audio_index + 1) +
                               " audio_features=" + std::to_string(audio_features.shape()[0]) +
                               " input_ids=" + std::to_string(seq_len));
    }
    std::copy(audio_features.values().begin() + audio_index * hidden,
              audio_features.values().begin() + (audio_index + 1) * hidden,
              merged.values().begin() + i * hidden);
    ++audio_index;
  }
  if (audio_index != audio_features.shape()[0]) {
    throw std::runtime_error("audio feature count does not match prompt: placeholders=" + std::to_string(audio_index) +
                             " audio_features=" + std::to_string(audio_features.shape()[0]));
  }
  return merged;
}

std::vector<Ort::Value> Qwen3ASROnnxModel::RunTextCore(std::unordered_map<std::string, Ort::Value>& inputs,
                                                       std::vector<float>* last_logits) const {
  auto outputs = text_core_.RunIo(inputs, text_core_output_names_, text_core_device_output_names_);
  *last_logits = text_core_.CopyLastLogits(outputs[0]);
  std::vector<Ort::Value> past;
  past.reserve(outputs.size() - 1);
  for (size_t i = 1; i < outputs.size(); ++i) past.push_back(std::move(outputs[i]));
  return past;
}

Qwen3ASROnnxModel::GeneratedIds Qwen3ASROnnxModel::GenerateIds(const PreparedInputs& inputs,
                                                               int max_new_tokens) const {
  GeneratedIds result;
  auto started_at = std::chrono::steady_clock::now();
  FloatTensor token_embeds = RunTokenEmbedding(inputs.input_ids);
  result.token_embedding_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started_at).count();
  started_at = std::chrono::steady_clock::now();
  FloatTensor audio_features = RunAudioEncoder(inputs.input_features);
  result.audio_encoder_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started_at).count();
  started_at = std::chrono::steady_clock::now();
  FloatTensor inputs_embeds = MergeAudioFeatures(inputs.input_ids, token_embeds, audio_features);
  result.merge_audio_features_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started_at).count();

  std::unordered_map<std::string, Ort::Value> feed;
  feed.emplace("inputs_embeds", text_core_.MakeFloatInput(inputs_embeds, "inputs_embeds"));
  feed.emplace("attention_mask", text_core_.MakeInt64Input(inputs.attention_mask));
  std::vector<int64_t> cache_position(static_cast<size_t>(inputs_embeds.shape()[1]));
  for (int64_t i = 0; i < inputs_embeds.shape()[1]; ++i) cache_position[static_cast<size_t>(i)] = i;
  feed.emplace("cache_position", text_core_.MakeInt64Input({inputs_embeds.shape()[1]}, cache_position));

  std::vector<std::vector<Ort::Float16_t>> empty_storage(static_cast<size_t>(kNumLayers * 2));
  std::vector<int64_t> empty_shape{1, kNumKvHeads, 0, kHeadDim};
  Ort::MemoryInfo cpu_memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  for (int64_t layer = 0; layer < kNumLayers; ++layer) {
    feed.emplace("past_key_" + std::to_string(layer),
                 Ort::Value::CreateTensor<Ort::Float16_t>(cpu_memory, empty_storage[static_cast<size_t>(layer * 2)].data(),
                                                          0, empty_shape.data(), empty_shape.size()));
    feed.emplace("past_value_" + std::to_string(layer),
                 Ort::Value::CreateTensor<Ort::Float16_t>(
                     cpu_memory, empty_storage[static_cast<size_t>(layer * 2 + 1)].data(), 0, empty_shape.data(),
                     empty_shape.size()));
  }

  std::vector<float> logits;
  started_at = std::chrono::steady_clock::now();
  auto past = RunTextCore(feed, &logits);
  result.prefill_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started_at).count();

  for (int step = 0; step < max_new_tokens; ++step) {
    started_at = std::chrono::steady_clock::now();
    const int64_t token = GreedyArgmax(logits);
    if (token == kEosTokenId || token == kEndOfTextTokenId) {
      result.decode_ms +=
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started_at).count();
      break;
    }
    result.ids.push_back(token);

    Int64Tensor next_id({1, 1}, std::vector<int64_t>{token});
    FloatTensor next_embed = RunTokenEmbedding(next_id);
    std::unordered_map<std::string, Ort::Value> decode_feed;
    decode_feed.emplace("inputs_embeds", text_core_.MakeFloatInput(next_embed, "inputs_embeds"));
    const int64_t past_len = inputs_embeds.shape()[1] + static_cast<int64_t>(result.ids.size()) - 1;
    auto& decode_attention = decode_attention_cache_[past_len];
    if (decode_attention.empty()) decode_attention.assign(static_cast<size_t>(past_len + 1), 1);
    auto& decode_cache_position = decode_cache_position_cache_[past_len];
    if (decode_cache_position.empty()) decode_cache_position.push_back(past_len);
    decode_feed.emplace("attention_mask", text_core_.MakeInt64Input({1, past_len + 1}, decode_attention));
    decode_feed.emplace("cache_position", text_core_.MakeInt64Input({1}, decode_cache_position));
    for (int64_t layer = 0; layer < kNumLayers; ++layer) {
      decode_feed.emplace("past_key_" + std::to_string(layer), std::move(past[static_cast<size_t>(layer * 2)]));
      decode_feed.emplace("past_value_" + std::to_string(layer), std::move(past[static_cast<size_t>(layer * 2 + 1)]));
    }
    past = RunTextCore(decode_feed, &logits);
    result.decode_steps += 1;
    result.decode_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started_at).count();
  }
  return result;
}

}  // namespace qwen3asr
