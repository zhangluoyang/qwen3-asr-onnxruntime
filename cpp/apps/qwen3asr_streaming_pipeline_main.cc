#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "qwen3asr/audio_frontend.h"
#include "qwen3asr/fsmn_vad.h"
#include "qwen3asr/qwen_model.h"

namespace {

// ======================== 默认实时流水线配置 ========================
// 直接运行 qwen3asr_streaming_run 即可：
//
//     ./qwen3asr_streaming_run
//
// 结构和 Python StreamingVadAsrPipeline 一样：
//
//     Audio Producer -> audio_queue -> VAD worker -> speech_queue -> ASR worker -> result_queue

constexpr const char* DEFAULT_MODEL_DIR = "/home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-ASR-0.6B";
constexpr const char* DEFAULT_ONNX_DIR = "./onnx_asr";
constexpr const char* DEFAULT_VAD_ONNX_PATH = "./onnx_fsmn/fsmn_vad_vad_encoder.onnx";
constexpr const char* DEFAULT_VAD_CMVN_PATH = "./onnx_fsmn/am.mvn";
constexpr const char* DEFAULT_AUDIO_PATH = "./data/voice_design_stream.wav";
constexpr const char* DEFAULT_LOG_PATH = "./qwen3asr_cpp_streaming.log";

constexpr int DEFAULT_SAMPLE_RATE = 16000;
constexpr int DEFAULT_VAD_CHUNK_MS = 250;
constexpr int DEFAULT_PRE_ROLL_MS = 300;
constexpr double DEFAULT_ASR_CHUNK_SEC = 0.5;
constexpr int DEFAULT_ASR_MAX_HISTORY_MS = 10000;
constexpr bool DEFAULT_REALTIME_PACING = true;
constexpr bool DEFAULT_ENABLE_PARTIAL_ASR = true;
constexpr int DEFAULT_UNFIXED_CHUNK_NUM = 2;
constexpr int DEFAULT_UNFIXED_TOKEN_NUM = 5;
constexpr int DEFAULT_MAX_NEW_TOKENS = 1024;
constexpr bool DEFAULT_USE_CUDA = true;
constexpr const char* DEFAULT_LANGUAGE = "chinese";
constexpr const char* DEFAULT_CONTEXT = "";
constexpr const char* DEFAULT_HOTWORDS = "小医仙,萧炎,斗破苍穹";

using Clock = std::chrono::steady_clock;

template <typename T>
class BlockingQueue {
 public:
  explicit BlockingQueue(size_t max_size) : max_size_(std::max<size_t>(1, max_size)) {}

  void Push(T item) {
    std::unique_lock<std::mutex> lock(mutex_);
    not_full_.wait(lock, [&] { return queue_.size() < max_size_; });
    queue_.push_back(std::move(item));
    not_empty_.notify_one();
  }

  T Pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    not_empty_.wait(lock, [&] { return !queue_.empty(); });
    T item = std::move(queue_.front());
    queue_.pop_front();
    not_full_.notify_one();
    return item;
  }

  size_t Size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

 private:
  size_t max_size_ = 1;
  mutable std::mutex mutex_;
  std::condition_variable not_empty_;
  std::condition_variable not_full_;
  std::deque<T> queue_;
};

struct Args {
  std::filesystem::path model_dir = DEFAULT_MODEL_DIR;
  std::filesystem::path onnx_dir = DEFAULT_ONNX_DIR;
  std::filesystem::path vad_onnx_path = DEFAULT_VAD_ONNX_PATH;
  std::filesystem::path vad_cmvn_path = DEFAULT_VAD_CMVN_PATH;
  std::filesystem::path audio = DEFAULT_AUDIO_PATH;
  std::filesystem::path log_file = DEFAULT_LOG_PATH;
  std::string context = DEFAULT_CONTEXT;
  std::string language = DEFAULT_LANGUAGE;
  std::string hotwords = DEFAULT_HOTWORDS;
  int max_new_tokens = DEFAULT_MAX_NEW_TOKENS;
  bool use_cuda = DEFAULT_USE_CUDA;
};

struct AudioChunk {
  int64_t chunk_id = 0;
  int64_t start_ms = 0;
  int64_t end_ms = 0;
  std::vector<float> pcm;
  bool is_final = false;
  Clock::time_point pushed_at = Clock::now();
};

enum class SpeechEventType {
  kSpeechStart,
  kSpeechChunk,
  kSpeechEnd,
  kEndOfStream,
  kError,
};

struct SpeechEvent {
  SpeechEventType type = SpeechEventType::kSpeechChunk;
  int64_t utterance_id = -1;
  int64_t start_ms = -1;
  int64_t end_ms = -1;
  int64_t timestamp_ms = -1;
  std::vector<float> pcm;
  int64_t vad_input_samples = 0;
  int64_t vad_input_ms = 0;
  int64_t vad_chunk_start_ms = -1;
  int64_t vad_chunk_end_ms = -1;
  int64_t speech_input_samples = 0;
  int64_t speech_input_ms = 0;
  int64_t pre_roll_samples = 0;
  int64_t pre_roll_ms = 0;
  std::string speech_event_type;
  double vad_elapsed_ms = 0.0;
  double vad_elapsed_ms_total = 0.0;
  std::string error;
  Clock::time_point audio_pushed_at = Clock::now();
};

enum class StreamingAsrEventType {
  kSpeechStart,
  kPartial,
  kFinal,
  kEndOfStream,
  kError,
  kDebug,
};

struct StreamingAsrEvent {
  StreamingAsrEventType type = StreamingAsrEventType::kDebug;
  int64_t utterance_id = -1;
  int64_t result_id = -1;
  int64_t start_ms = -1;
  int64_t end_ms = -1;
  int64_t audio_end_ms = -1;
  bool is_final = false;
  std::string confirmed_delta_text;
  std::string confirmed_text;
  std::string pending_text;
  std::string text;
  std::string full_text;
  std::string raw_text;
  std::string language;
  double vad_elapsed_ms = 0.0;
  double vad_elapsed_ms_total = 0.0;
  double asr_elapsed_ms = 0.0;
  double asr_elapsed_ms_total = 0.0;
  double emit_latency_ms = 0.0;
  size_t audio_queue_size = 0;
  size_t speech_queue_size = 0;
  size_t result_queue_size = 0;
  int64_t vad_input_samples = 0;
  int64_t vad_input_ms = 0;
  int64_t vad_chunk_start_ms = -1;
  int64_t vad_chunk_end_ms = -1;
  int64_t speech_input_samples = 0;
  int64_t speech_input_ms = 0;
  int64_t pre_roll_samples = 0;
  int64_t pre_roll_ms = 0;
  std::string speech_event_type;
  int64_t input_samples = 0;
  int64_t input_ms = 0;
  int64_t buffer_samples_before = 0;
  int64_t buffer_samples_after_append = 0;
  int64_t buffer_samples = 0;
  int64_t triggered_decode_count = 0;
  int64_t decoded_chunk_id = -1;
  int64_t processed_chunks = 0;
  int64_t trigger_chunk_start_ms = -1;
  int64_t trigger_chunk_end_ms = -1;
  int64_t audio_accum_samples = 0;
  int64_t asr_window_samples = 0;
  int64_t asr_window_ms = 0;
  int64_t audio_window_start_ms = -1;
  int64_t audio_window_end_ms = -1;
  int64_t total_audio_samples = 0;
  int64_t max_audio_history_samples = 0;
  int64_t last_trimmed_samples = 0;
  int64_t total_trimmed_samples = 0;
  int64_t trim_events = 0;
  int64_t prefix_chars = 0;
  int64_t committed_raw_chars = 0;
  int64_t uncommitted_raw_chars = 0;
  int64_t confirmed_delta_chars = 0;
  int64_t confirmed_text_chars = 0;
  int64_t pending_chars = 0;
  int64_t delta_raw_chars = 0;
  int64_t emitted_text_chars = 0;
  double asr_prepare_inputs_ms = 0.0;
  double asr_generate_ms = 0.0;
  double asr_token_embedding_ms = 0.0;
  double asr_audio_encoder_ms = 0.0;
  double asr_merge_audio_features_ms = 0.0;
  double asr_prefill_ms = 0.0;
  double asr_decode_ms = 0.0;
  int64_t asr_decode_steps = 0;
  bool stream_text_revision = false;
  bool stream_raw_revision = false;
  bool finalized = false;
  std::string error;
  std::string debug;
};

struct QwenAsrStreamingState {
  int64_t utterance_id = -1;
  int64_t start_ms = -1;
  int64_t end_ms = -1;
  int64_t total_audio_samples = 0;
  int64_t total_trimmed_samples = 0;
  int64_t last_trimmed_samples = 0;
  int64_t trim_events = 0;
  int64_t last_appended_start_samples = 0;
  int64_t last_appended_end_samples = 0;
  int64_t chunk_id = 0;
  int64_t last_input_samples = 0;
  int64_t last_buffer_samples_before = 0;
  int64_t last_buffer_samples_after_append = 0;
  int64_t last_triggered_decode_count = 0;
  std::vector<float> buffer;
  std::vector<float> audio_accum;
  std::string raw_decoded;
  std::string committed_raw_decoded;
  std::string emitted_text;
  std::string pending_text;
  std::string emitted_raw_decoded;
  std::vector<int64_t> raw_decoded_token_ids;
  std::string text;
  std::string pending_event_text;
  std::string confirmed_delta_text;
  std::string confirmed_text;
  std::string raw_text;
  std::string language;
  std::string last_raw_generated;
  std::string last_prefix;
  bool last_text_revision = false;
  bool last_raw_revision = false;
  bool suppress_next_confirmed_delta = false;
  int64_t last_delta_raw_chars = 0;
  double last_asr_prepare_inputs_ms = 0.0;
  double last_asr_generate_ms = 0.0;
  double last_asr_token_embedding_ms = 0.0;
  double last_asr_audio_encoder_ms = 0.0;
  double last_asr_merge_audio_features_ms = 0.0;
  double last_asr_prefill_ms = 0.0;
  double last_asr_decode_ms = 0.0;
  int64_t last_asr_decode_steps = 0;
  double asr_elapsed_ms_total = 0.0;
};

double ElapsedMs(Clock::time_point started_at) {
  return std::chrono::duration<double, std::milli>(Clock::now() - started_at).count();
}

size_t MsToSamples(int64_t ms) {
  return static_cast<size_t>(std::max<int64_t>(0, ms) * DEFAULT_SAMPLE_RATE / 1000);
}

int64_t SamplesToMs(size_t samples) {
  return static_cast<int64_t>(samples * 1000 / DEFAULT_SAMPLE_RATE);
}

std::vector<float> ConcatAudio(const std::vector<float>& left, const std::vector<float>& right) {
  if (left.empty()) return right;
  if (right.empty()) return left;
  std::vector<float> out;
  out.reserve(left.size() + right.size());
  out.insert(out.end(), left.begin(), left.end());
  out.insert(out.end(), right.begin(), right.end());
  return out;
}

std::vector<float> TrimTail(const std::vector<float>& audio, size_t max_samples) {
  if (max_samples == 0 || audio.empty()) return {};
  if (audio.size() <= max_samples) return audio;
  return {audio.end() - static_cast<std::ptrdiff_t>(max_samples), audio.end()};
}

bool StartsWith(const std::string& text, const std::string& prefix) {
  return text.rfind(prefix, 0) == 0;
}

std::string TextDelta(const std::string& previous, const std::string& current) {
  if (previous.empty()) return current;
  if (!StartsWith(current, previous)) return "";
  return current.substr(previous.size());
}

std::pair<std::string, bool> ConsumeStreamDelta(const std::string& current, const std::string& emitted) {
  if (current.empty()) return {"", false};
  if (StartsWith(current, emitted)) return {current.substr(emitted.size()), false};
  return {"", true};
}

size_t Utf8CharByteLen(unsigned char ch) {
  if ((ch & 0x80) == 0) return 1;
  if ((ch & 0xE0) == 0xC0) return 2;
  if ((ch & 0xF0) == 0xE0) return 3;
  if ((ch & 0xF8) == 0xF0) return 4;
  return 1;
}

size_t CommonUtf8PrefixBytes(const std::string& left, const std::string& right) {
  size_t li = 0;
  size_t ri = 0;
  size_t common = 0;
  while (li < left.size() && ri < right.size()) {
    const size_t llen = Utf8CharByteLen(static_cast<unsigned char>(left[li]));
    const size_t rlen = Utf8CharByteLen(static_cast<unsigned char>(right[ri]));
    if (li + llen > left.size() || ri + rlen > right.size()) break;
    if (llen != rlen || left.compare(li, llen, right, ri, rlen) != 0) break;
    li += llen;
    ri += rlen;
    common = li;
  }
  return common;
}

struct ConfirmedPendingDelta {
  std::string confirmed_delta;
  std::string pending;
  bool revision = false;
};

ConfirmedPendingDelta ConsumeConfirmedAndPendingDelta(const std::string& current,
                                                      const std::string& confirmed,
                                                      const std::string& previous_pending,
                                                      bool finalize) {
  ConfirmedPendingDelta out;
  if (current.empty()) return out;
  if (finalize) {
    if (StartsWith(current, confirmed)) {
      out.confirmed_delta = current.substr(confirmed.size());
    } else {
      out.revision = !confirmed.empty() || !previous_pending.empty();
    }
    return out;
  }
  if (!StartsWith(current, confirmed)) {
    out.pending = current;
    out.revision = true;
    return out;
  }
  const std::string suffix = current.substr(confirmed.size());
  const size_t prefix_bytes = CommonUtf8PrefixBytes(previous_pending, suffix);
  out.confirmed_delta = suffix.substr(0, prefix_bytes);
  out.pending = suffix.substr(prefix_bytes);
  return out;
}

std::string BoolText(bool value) {
  return value ? "1" : "0";
}

size_t Utf8CharCount(const std::string& text) {
  size_t count = 0;
  for (unsigned char ch : text) {
    if ((ch & 0xC0) != 0x80) ++count;
  }
  return count;
}

void CopySpeechMetadata(const SpeechEvent& in, StreamingAsrEvent* out) {
  if (!out) return;
  out->vad_input_samples = in.vad_input_samples;
  out->vad_input_ms = in.vad_input_ms;
  out->vad_chunk_start_ms = in.vad_chunk_start_ms;
  out->vad_chunk_end_ms = in.vad_chunk_end_ms;
  out->speech_input_samples = in.speech_input_samples;
  out->speech_input_ms = in.speech_input_ms;
  out->pre_roll_samples = in.pre_roll_samples;
  out->pre_roll_ms = in.pre_roll_ms;
  out->speech_event_type = in.speech_event_type;
}

void FillAsrMetadata(const QwenAsrStreamingState& state,
                     const std::string& prefix,
                     bool text_revision,
                     bool raw_revision,
                     bool finalized,
                     StreamingAsrEvent* out) {
  if (!out) return;
  const int64_t audio_window_start_samples =
      state.total_audio_samples - static_cast<int64_t>(state.audio_accum.size());
  out->input_samples = state.last_input_samples;
  out->input_ms = SamplesToMs(static_cast<size_t>(std::max<int64_t>(0, state.last_input_samples)));
  out->buffer_samples_before = state.last_buffer_samples_before;
  out->buffer_samples_after_append = state.last_buffer_samples_after_append;
  out->triggered_decode_count = state.last_triggered_decode_count;
  out->buffer_samples = static_cast<int64_t>(state.buffer.size());
  out->decoded_chunk_id = std::max<int64_t>(0, state.chunk_id - 1);
  out->processed_chunks = state.chunk_id;
  out->trigger_chunk_start_ms = SamplesToMs(static_cast<size_t>(state.last_appended_start_samples));
  out->trigger_chunk_end_ms = SamplesToMs(static_cast<size_t>(state.last_appended_end_samples));
  out->audio_accum_samples = static_cast<int64_t>(state.audio_accum.size());
  out->asr_window_samples = static_cast<int64_t>(state.audio_accum.size());
  out->asr_window_ms = SamplesToMs(state.audio_accum.size());
  out->audio_window_start_ms = SamplesToMs(static_cast<size_t>(std::max<int64_t>(0, audio_window_start_samples)));
  out->audio_window_end_ms = SamplesToMs(static_cast<size_t>(state.total_audio_samples));
  out->total_audio_samples = state.total_audio_samples;
  out->max_audio_history_samples = static_cast<int64_t>(MsToSamples(DEFAULT_ASR_MAX_HISTORY_MS));
  out->last_trimmed_samples = state.last_trimmed_samples;
  out->total_trimmed_samples = state.total_trimmed_samples;
  out->trim_events = state.trim_events;
  out->prefix_chars = static_cast<int64_t>(Utf8CharCount(prefix));
  out->committed_raw_chars = static_cast<int64_t>(Utf8CharCount(state.committed_raw_decoded));
  out->uncommitted_raw_chars = static_cast<int64_t>(
      Utf8CharCount(state.raw_decoded) > Utf8CharCount(state.committed_raw_decoded)
          ? Utf8CharCount(state.raw_decoded) - Utf8CharCount(state.committed_raw_decoded)
          : 0);
  out->confirmed_delta_chars = static_cast<int64_t>(Utf8CharCount(state.confirmed_delta_text));
  out->confirmed_text_chars = static_cast<int64_t>(Utf8CharCount(state.confirmed_text));
  out->pending_chars = static_cast<int64_t>(Utf8CharCount(state.pending_text));
  out->delta_raw_chars = state.last_delta_raw_chars;
  out->emitted_text_chars = static_cast<int64_t>(Utf8CharCount(state.emitted_text));
  out->asr_prepare_inputs_ms = state.last_asr_prepare_inputs_ms;
  out->asr_generate_ms = state.last_asr_generate_ms;
  out->asr_token_embedding_ms = state.last_asr_token_embedding_ms;
  out->asr_audio_encoder_ms = state.last_asr_audio_encoder_ms;
  out->asr_merge_audio_features_ms = state.last_asr_merge_audio_features_ms;
  out->asr_prefill_ms = state.last_asr_prefill_ms;
  out->asr_decode_ms = state.last_asr_decode_ms;
  out->asr_decode_steps = state.last_asr_decode_steps;
  out->stream_text_revision = text_revision;
  out->stream_raw_revision = raw_revision;
  out->finalized = finalized;
}

void AppendMetadata(std::ostream& os, const StreamingAsrEvent& event) {
  os << " vad_input_samples=" << event.vad_input_samples
     << " vad_input_ms=" << event.vad_input_ms
     << " vad_chunk_start_ms=" << event.vad_chunk_start_ms
     << " vad_chunk_end_ms=" << event.vad_chunk_end_ms
     << " speech_event_type='" << event.speech_event_type << "'"
     << " speech_input_samples=" << event.speech_input_samples
     << " speech_input_ms=" << event.speech_input_ms
     << " pre_roll_samples=" << event.pre_roll_samples
     << " pre_roll_ms=" << event.pre_roll_ms
     << " input_samples=" << event.input_samples
     << " input_ms=" << event.input_ms
     << " buffer_samples_before=" << event.buffer_samples_before
     << " buffer_samples_after_append=" << event.buffer_samples_after_append
     << " buffer_samples=" << event.buffer_samples
     << " triggered_decode_count=" << event.triggered_decode_count
     << " decoded_chunk_id=" << event.decoded_chunk_id
     << " processed_chunks=" << event.processed_chunks
     << " trigger_chunk_start_ms=" << event.trigger_chunk_start_ms
     << " trigger_chunk_end_ms=" << event.trigger_chunk_end_ms
     << " audio_accum_samples=" << event.audio_accum_samples
     << " asr_window_samples=" << event.asr_window_samples
     << " asr_window_ms=" << event.asr_window_ms
     << " audio_window_start_ms=" << event.audio_window_start_ms
     << " audio_window_end_ms=" << event.audio_window_end_ms
     << " total_audio_samples=" << event.total_audio_samples
     << " max_audio_history_samples=" << event.max_audio_history_samples
     << " last_trimmed_samples=" << event.last_trimmed_samples
     << " total_trimmed_samples=" << event.total_trimmed_samples
     << " trim_events=" << event.trim_events
     << " prefix_chars=" << event.prefix_chars
     << " committed_raw_chars=" << event.committed_raw_chars
     << " uncommitted_raw_chars=" << event.uncommitted_raw_chars
     << " confirmed_delta_chars=" << event.confirmed_delta_chars
     << " confirmed_text_chars=" << event.confirmed_text_chars
     << " pending_chars=" << event.pending_chars
     << " delta_raw_chars=" << event.delta_raw_chars
     << " emitted_text_chars=" << event.emitted_text_chars
     << " asr_prepare_inputs_ms=" << event.asr_prepare_inputs_ms
     << " asr_generate_ms=" << event.asr_generate_ms
     << " asr_token_embedding_ms=" << event.asr_token_embedding_ms
     << " asr_audio_encoder_ms=" << event.asr_audio_encoder_ms
     << " asr_merge_audio_features_ms=" << event.asr_merge_audio_features_ms
     << " asr_prefill_ms=" << event.asr_prefill_ms
     << " asr_decode_ms=" << event.asr_decode_ms
     << " asr_decode_steps=" << event.asr_decode_steps
     << " stream_text_revision=" << BoolText(event.stream_text_revision)
     << " stream_raw_revision=" << BoolText(event.stream_raw_revision)
     << " finalized=" << BoolText(event.finalized);
}

std::vector<float> AudioWindow(const std::vector<float>& audio) {
  const size_t max_samples = MsToSamples(DEFAULT_ASR_MAX_HISTORY_MS);
  if (max_samples == 0 || audio.size() <= max_samples) return audio;
  return {audio.end() - static_cast<std::ptrdiff_t>(max_samples), audio.end()};
}

std::vector<float> TakeFront(std::vector<float>* audio, size_t n) {
  if (!audio || audio->empty() || n == 0) return {};
  n = std::min(n, audio->size());
  std::vector<float> out(audio->begin(), audio->begin() + static_cast<std::ptrdiff_t>(n));
  audio->erase(audio->begin(), audio->begin() + static_cast<std::ptrdiff_t>(n));
  return out;
}

void AppendStreamAudio(QwenAsrStreamingState* state, const std::vector<float>& chunk) {
  if (!state || chunk.empty()) return;
  state->last_appended_start_samples = state->total_audio_samples;
  state->last_appended_end_samples = state->total_audio_samples + static_cast<int64_t>(chunk.size());
  state->total_audio_samples += static_cast<int64_t>(chunk.size());
  state->last_trimmed_samples = 0;

  const size_t max_samples = MsToSamples(DEFAULT_ASR_MAX_HISTORY_MS);
  if (max_samples > 0) {
    const size_t overflow =
        state->audio_accum.size() + chunk.size() > max_samples
            ? state->audio_accum.size() + chunk.size() - max_samples
            : 0;
    if (overflow > 0) {
      const size_t drop_existing = std::min(state->audio_accum.size(), overflow);
      state->audio_accum.erase(state->audio_accum.begin(),
                               state->audio_accum.begin() + static_cast<std::ptrdiff_t>(drop_existing));
      state->last_trimmed_samples = static_cast<int64_t>(overflow);
      state->total_trimmed_samples += static_cast<int64_t>(overflow);
      state->trim_events += 1;
    }
  }
  const size_t available = max_samples > 0 && chunk.size() > max_samples ? max_samples : chunk.size();
  state->audio_accum.insert(state->audio_accum.end(),
                            chunk.end() - static_cast<std::ptrdiff_t>(available),
                            chunk.end());
}

std::string DecodeWithTokenRollback(const qwen3asr::Qwen3ASROnnxModel& model,
                                    const std::vector<int64_t>& token_ids,
                                    int rollback) {
  rollback = std::max(0, rollback);
  while (true) {
    const size_t end = token_ids.size() > static_cast<size_t>(rollback)
                           ? token_ids.size() - static_cast<size_t>(rollback)
                           : 0;
    if (end == 0) return "";
    std::vector<int64_t> trimmed(token_ids.begin(), token_ids.begin() + static_cast<std::ptrdiff_t>(end));
    const std::string decoded = model.DecodeTokenIds(trimmed, false);
    if (decoded.find("\xEF\xBF\xBD") == std::string::npos) return decoded;
    ++rollback;
  }
}

std::string StreamingPrefix(const qwen3asr::Qwen3ASROnnxModel& model, const QwenAsrStreamingState& state) {
  if (state.chunk_id < DEFAULT_UNFIXED_CHUNK_NUM) return "";
  const auto token_ids = state.raw_decoded_token_ids.empty() ? model.EncodeText(state.raw_decoded)
                                                             : state.raw_decoded_token_ids;
  return DecodeWithTokenRollback(model, token_ids, DEFAULT_UNFIXED_TOKEN_NUM);
}

std::string StableStreamingRaw(const qwen3asr::Qwen3ASROnnxModel& model,
                               const QwenAsrStreamingState& state,
                               bool finalize) {
  if (finalize) return state.raw_decoded;
  if (state.chunk_id < DEFAULT_UNFIXED_CHUNK_NUM) return "";
  const auto token_ids = state.raw_decoded_token_ids.empty() ? model.EncodeText(state.raw_decoded)
                                                             : state.raw_decoded_token_ids;
  return DecodeWithTokenRollback(model, token_ids, DEFAULT_UNFIXED_TOKEN_NUM);
}

std::string CandidateStreamingRaw(const qwen3asr::Qwen3ASROnnxModel& model,
                                  const QwenAsrStreamingState& state,
                                  bool finalize) {
  if (finalize) return state.raw_decoded;
  const auto token_ids = state.raw_decoded_token_ids.empty() ? model.EncodeText(state.raw_decoded)
                                                             : state.raw_decoded_token_ids;
  return DecodeWithTokenRollback(model, token_ids, 0);
}

void CommitStreamingRaw(const qwen3asr::Qwen3ASROnnxModel& model,
                        QwenAsrStreamingState* state,
                        const std::optional<std::string>& language,
                        bool finalize) {
  if (!state) return;
  const std::string committed_raw = StableStreamingRaw(model, *state, finalize);
  state->committed_raw_decoded = committed_raw;
  const std::string candidate_raw = CandidateStreamingRaw(model, *state, finalize);
  const auto committed_parsed = model.ParseRawText(committed_raw, language);
  const auto parsed = model.ParseRawText(candidate_raw, language);
  state->language = parsed.language;
  const auto text_delta = ConsumeConfirmedAndPendingDelta(
      parsed.text, state->emitted_text, state->pending_text, finalize);
  std::string exposed_confirmed_delta = text_delta.confirmed_delta;
  if (finalize && text_delta.revision) {
    state->emitted_text = parsed.text;
    state->pending_text.clear();
    state->suppress_next_confirmed_delta = false;
    exposed_confirmed_delta.clear();
  } else if (text_delta.revision) {
    if (!committed_parsed.text.empty() && StartsWith(parsed.text, committed_parsed.text)) {
      state->emitted_text = committed_parsed.text;
      state->pending_text = parsed.text.substr(committed_parsed.text.size());
    } else {
      state->emitted_text = parsed.text;
      state->pending_text.clear();
    }
    state->suppress_next_confirmed_delta = false;
    exposed_confirmed_delta.clear();
  } else {
    state->emitted_text += text_delta.confirmed_delta;
    state->pending_text = text_delta.pending;
    if (state->suppress_next_confirmed_delta && !text_delta.confirmed_delta.empty()) {
      exposed_confirmed_delta.clear();
      state->suppress_next_confirmed_delta = false;
    }
    if (finalize) {
      state->suppress_next_confirmed_delta = false;
    }
  }
  state->confirmed_delta_text += exposed_confirmed_delta;
  state->confirmed_text = state->emitted_text;
  state->pending_event_text = state->pending_text;
  state->text = state->confirmed_text + state->pending_text;

  const auto [delta_raw, raw_revision] = ConsumeStreamDelta(committed_raw, state->emitted_raw_decoded);
  if (!raw_revision) {
    state->emitted_raw_decoded = committed_raw;
  }
  state->last_text_revision = text_delta.revision;
  state->last_raw_revision = raw_revision;
  state->last_delta_raw_chars = static_cast<int64_t>(Utf8CharCount(raw_revision ? "" : delta_raw));
}

void PrintUsage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0 << " [options]\n"
      << "不传参数时使用源码顶部默认配置，启动生产者/消费者实时流水线。\n\n"
      << "Options:\n"
      << "  --audio WAV            临时覆盖默认音频路径\n"
      << "  --model-dir PATH       临时覆盖 Qwen3-ASR 原始模型目录\n"
      << "  --onnx-dir PATH        临时覆盖 Qwen3-ASR ONNX 目录\n"
      << "  --vad-onnx PATH        临时覆盖 FSMN-VAD ONNX 文件\n"
      << "  --vad-cmvn PATH        临时覆盖 FSMN-VAD CMVN 文件\n"
      << "  --log-file PATH        临时覆盖实时日志文件\n"
      << "  --max-new-tokens N     临时覆盖 ASR 最大生成 token 数\n"
      << "  --cpu                  强制 CPU\n";
}

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto need_value = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + name);
      return argv[++i];
    };
    if (key == "--audio") args.audio = need_value(key);
    else if (key == "--model-dir") args.model_dir = need_value(key);
    else if (key == "--onnx-dir") args.onnx_dir = need_value(key);
    else if (key == "--vad-onnx") args.vad_onnx_path = need_value(key);
    else if (key == "--vad-cmvn") args.vad_cmvn_path = need_value(key);
    else if (key == "--log-file") args.log_file = need_value(key);
    else if (key == "--max-new-tokens") args.max_new_tokens = std::stoi(need_value(key));
    else if (key == "--cpu") args.use_cuda = false;
    else if (key == "--help" || key == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + key);
    }
  }
  return args;
}

void LogEvent(std::ofstream& log, const StreamingAsrEvent& event) {
  if (event.type == StreamingAsrEventType::kSpeechStart) {
    log << "[speech_start] utt=" << event.utterance_id
        << " rid=" << event.result_id
        << " start=" << event.start_ms << "ms"
        << " end=" << event.end_ms << "ms"
        << " audio_end=" << event.audio_end_ms << "ms"
        << " vad_ms=" << event.vad_elapsed_ms
        << " asr_ms=" << event.asr_elapsed_ms
        << " vad_total_ms=" << event.vad_elapsed_ms_total
        << " asr_total_ms=" << event.asr_elapsed_ms_total
        << " latency_ms=" << event.emit_latency_ms
        << " q=" << event.audio_queue_size << "/" << event.speech_queue_size << "/" << event.result_queue_size;
    AppendMetadata(log, event);
    log << "\n";
  } else if (event.type == StreamingAsrEventType::kPartial) {
    log << "[partial] utt=" << event.utterance_id
        << " rid=" << event.result_id
        << " start=" << event.start_ms << "ms"
        << " end=" << event.end_ms << "ms"
        << " audio_end=" << event.audio_end_ms << "ms"
        << " vad_ms=" << event.vad_elapsed_ms
        << " asr_ms=" << event.asr_elapsed_ms
        << " vad_total_ms=" << event.vad_elapsed_ms_total
        << " asr_total_ms=" << event.asr_elapsed_ms_total
        << " latency_ms=" << event.emit_latency_ms
        << " q=" << event.audio_queue_size << "/" << event.speech_queue_size << "/" << event.result_queue_size;
    AppendMetadata(log, event);
    log
        << " confirmed_delta='" << event.confirmed_delta_text << "'"
        << " confirmed_text='" << event.confirmed_text << "'"
        << " pending='" << event.pending_text << "'"
        << " full_chars=" << Utf8CharCount(event.full_text) << "\n";
  } else if (event.type == StreamingAsrEventType::kFinal) {
    log << "[final] utt=" << event.utterance_id
        << " rid=" << event.result_id
        << " start=" << event.start_ms << "ms"
        << " end=" << event.end_ms << "ms"
        << " audio_end=" << event.audio_end_ms << "ms"
        << " vad_ms=" << event.vad_elapsed_ms
        << " asr_ms=" << event.asr_elapsed_ms
        << " vad_total_ms=" << event.vad_elapsed_ms_total
        << " asr_total_ms=" << event.asr_elapsed_ms_total
        << " latency_ms=" << event.emit_latency_ms
        << " q=" << event.audio_queue_size << "/" << event.speech_queue_size << "/" << event.result_queue_size;
    AppendMetadata(log, event);
    log
        << " full_chars=" << Utf8CharCount(event.full_text)
        << " language='" << event.language << "'"
        << " confirmed_delta='" << event.confirmed_delta_text << "'"
        << " confirmed_text='" << event.confirmed_text << "'"
        << " pending='" << event.pending_text << "'"
        << " text='" << event.text << "'\n";
  } else if (event.type == StreamingAsrEventType::kEndOfStream) {
    log << "[pipeline_done] utt=" << event.utterance_id
        << " rid=" << event.result_id
        << " start=Nonems end=Nonems audio_end=Nonems"
        << " vad_ms=" << event.vad_elapsed_ms
        << " asr_ms=" << event.asr_elapsed_ms
        << " vad_total_ms=" << event.vad_elapsed_ms_total
        << " asr_total_ms=" << event.asr_elapsed_ms_total
        << " latency_ms=" << event.emit_latency_ms
        << " q=" << event.audio_queue_size << "/" << event.speech_queue_size << "/" << event.result_queue_size;
    AppendMetadata(log, event);
    log << "\n";
  } else if (event.type == StreamingAsrEventType::kError) {
    log << "[error] utt=" << event.utterance_id
        << " rid=" << event.result_id
        << " error='" << event.error << "'\n";
  } else if (event.type == StreamingAsrEventType::kDebug) {
    log << event.debug << "\n";
  }
  log.flush();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    std::ofstream log(args.log_file, std::ios::out | std::ios::trunc);
    if (!log) throw std::runtime_error("failed to open log file: " + args.log_file.string());

    std::cout << "streaming_log_file: " << args.log_file << "\n";
    std::cout.flush();
    log << "[program_start] streaming_pipeline=true"
        << " realtime_pacing=" << BoolText(DEFAULT_REALTIME_PACING)
        << " audio=" << args.audio.string() << "\n";
    log << "[config] sample_rate=" << DEFAULT_SAMPLE_RATE
        << " vad_chunk_ms=" << DEFAULT_VAD_CHUNK_MS
        << " vad_chunk_samples=" << MsToSamples(DEFAULT_VAD_CHUNK_MS)
        << " realtime_pacing=" << BoolText(DEFAULT_REALTIME_PACING)
        << " asr_chunk_size_sec=" << DEFAULT_ASR_CHUNK_SEC
        << " asr_chunk_samples=" << static_cast<int64_t>(DEFAULT_ASR_CHUNK_SEC * DEFAULT_SAMPLE_RATE)
        << " asr_max_audio_history_ms=" << DEFAULT_ASR_MAX_HISTORY_MS
        << " asr_max_audio_history_samples=" << MsToSamples(DEFAULT_ASR_MAX_HISTORY_MS)
        << " pre_roll_ms=" << DEFAULT_PRE_ROLL_MS
        << " pre_roll_samples=" << MsToSamples(DEFAULT_PRE_ROLL_MS)
        << " enable_partial_asr=" << BoolText(DEFAULT_ENABLE_PARTIAL_ASR)
        << " unfixed_chunk_num=" << DEFAULT_UNFIXED_CHUNK_NUM
        << " unfixed_token_num=" << DEFAULT_UNFIXED_TOKEN_NUM
        << " max_new_tokens=" << args.max_new_tokens
        << " use_cuda=" << BoolText(args.use_cuda)
        << " language='" << args.language << "'"
        << " context_chars=" << args.context.size()
        << " hotwords='" << args.hotwords << "'\n";
    log << "[model_config] model_dir=" << args.model_dir.string()
        << " asr_onnx_dir=" << args.onnx_dir.string()
        << " audio_encoder_onnx=" << (args.onnx_dir / "audio_encoder" / "audio_encoder.onnx").string()
        << " token_embedding_onnx=" << (args.onnx_dir / "token_embedding" / "token_embedding.onnx").string()
        << " text_core_onnx=" << (args.onnx_dir / "text_core" / "asr_text_core.onnx").string()
        << " vad_onnx=" << args.vad_onnx_path.string()
        << " vad_cmvn=" << args.vad_cmvn_path.string() << "\n";
    log << "[pipeline] Audio Producer -> audio_queue -> VAD worker -> speech_queue -> ASR worker -> result_queue\n";
    log.flush();

    qwen3asr::Qwen3ASRConfig asr_config;
    asr_config.model_dir = args.model_dir;
    asr_config.onnx_dir = args.onnx_dir;
    asr_config.use_cuda = args.use_cuda;
    asr_config.max_new_tokens = args.max_new_tokens;
    qwen3asr::Qwen3ASROnnxModel asr_model(asr_config);
    const auto asr_warmup_started_at = Clock::now();
    asr_model.WarmUp();
    const double asr_warmup_ms = ElapsedMs(asr_warmup_started_at);

    qwen3asr::FsmnVadConfig vad_config;
    vad_config.onnx_path = args.vad_onnx_path;
    vad_config.cmvn_path = args.vad_cmvn_path;
    vad_config.use_cuda = false;
    qwen3asr::FsmnVadOnnxModel vad_model(vad_config);

    const auto audio = qwen3asr::LoadAudioMono(args.audio, DEFAULT_SAMPLE_RATE);
    log << std::fixed << std::setprecision(3);
    log << "[model_loaded]"
        << " asr_audio_encoder_load_s=" << asr_model.AudioEncoderLoadSeconds()
        << " asr_token_embedding_load_s=" << asr_model.TokenEmbeddingLoadSeconds()
        << " asr_text_core_load_s=" << asr_model.TextCoreLoadSeconds()
        << "\n";
    log << "[warmup]"
        << " asr_warmup_ms=" << asr_warmup_ms
        << " warmup_windows_ms='1500,5000,10000'"
        << "\n";
    log << "[audio_loaded] path=" << args.audio.string()
        << " samples=" << audio.samples.size()
        << " sample_rate=" << DEFAULT_SAMPLE_RATE
        << " duration_ms=" << SamplesToMs(audio.samples.size())
        << " duration_sec=" << static_cast<double>(audio.samples.size()) / DEFAULT_SAMPLE_RATE << "\n";
    log.flush();

    BlockingQueue<AudioChunk> audio_queue(32);
    BlockingQueue<SpeechEvent> speech_queue(32);
    BlockingQueue<StreamingAsrEvent> result_queue(128);
    std::atomic<int64_t> next_result_id{0};

    auto fill_result_event = [&](StreamingAsrEvent* event) {
      if (!event) return;
      event->result_id = next_result_id.fetch_add(1);
      event->audio_queue_size = audio_queue.Size();
      event->speech_queue_size = speech_queue.Size();
      event->result_queue_size = result_queue.Size();
    };

    std::thread producer([&] {
      try {
        const size_t chunk_samples = MsToSamples(DEFAULT_VAD_CHUNK_MS);
        int64_t timestamp_ms = 0;
        int64_t chunk_id = 0;
        const auto pacing_started_at = Clock::now();
        for (size_t start = 0; start < audio.samples.size(); start += chunk_samples, ++chunk_id) {
          const size_t end = std::min(audio.samples.size(), start + chunk_samples);
          const int64_t chunk_duration_ms = SamplesToMs(end - start);
          AudioChunk chunk;
          chunk.chunk_id = chunk_id;
          chunk.start_ms = timestamp_ms;
          chunk.end_ms = timestamp_ms + chunk_duration_ms;
          chunk.pcm.assign(audio.samples.begin() + static_cast<std::ptrdiff_t>(start),
                           audio.samples.begin() + static_cast<std::ptrdiff_t>(end));
          chunk.pushed_at = Clock::now();
          audio_queue.Push(std::move(chunk));
          StreamingAsrEvent debug;
          debug.type = StreamingAsrEventType::kDebug;
          std::ostringstream oss;
          oss << "[audio_push] chunk_id=" << chunk_id
              << " start_ms=" << timestamp_ms
              << " end_ms=" << timestamp_ms + chunk_duration_ms
              << " samples=" << (end - start)
              << " duration_ms=" << chunk_duration_ms
              << " audio_queue_size=" << audio_queue.Size();
          debug.debug = oss.str();
          result_queue.Push(std::move(debug));
          timestamp_ms += chunk_duration_ms;
          if (DEFAULT_REALTIME_PACING) {
            std::this_thread::sleep_until(pacing_started_at + std::chrono::milliseconds(timestamp_ms));
          }
        }
        AudioChunk done;
        done.is_final = true;
        done.chunk_id = chunk_id;
        done.start_ms = timestamp_ms;
        done.end_ms = timestamp_ms;
        audio_queue.Push(std::move(done));
      } catch (const std::exception& ex) {
        StreamingAsrEvent event;
        event.type = StreamingAsrEventType::kError;
        event.error = std::string("producer: ") + ex.what();
        fill_result_event(&event);
        result_queue.Push(std::move(event));
        AudioChunk done;
        done.is_final = true;
        audio_queue.Push(std::move(done));
      }
    });

    std::thread vad_worker([&] {
      try {
        auto state = vad_model.InitStreamingState(DEFAULT_VAD_CHUNK_MS);
        bool vad_in_speech = false;
        int64_t vad_start_ms = -1;
        int64_t utterance_id = 0;
        double vad_elapsed_total = 0.0;
        std::vector<float> pre_roll;
        const size_t pre_roll_samples = MsToSamples(DEFAULT_PRE_ROLL_MS);

        auto push_speech_start = [&](int64_t start_ms,
                                     const AudioChunk& chunk,
                                     double vad_elapsed_ms) {
          SpeechEvent speech;
          speech.type = SpeechEventType::kSpeechStart;
          speech.utterance_id = utterance_id++;
          speech.start_ms = start_ms;
          speech.end_ms = chunk.end_ms;
          speech.timestamp_ms = chunk.start_ms;
          speech.pcm = ConcatAudio(pre_roll, chunk.pcm);
          speech.vad_input_samples = static_cast<int64_t>(chunk.pcm.size());
          speech.vad_input_ms = chunk.end_ms - chunk.start_ms;
          speech.vad_chunk_start_ms = chunk.start_ms;
          speech.vad_chunk_end_ms = chunk.end_ms;
          speech.speech_input_samples = static_cast<int64_t>(speech.pcm.size());
          speech.speech_input_ms = SamplesToMs(speech.pcm.size());
          speech.pre_roll_samples = static_cast<int64_t>(pre_roll.size());
          speech.pre_roll_ms = SamplesToMs(pre_roll.size());
          speech.speech_event_type = "speech_start";
          speech.vad_elapsed_ms = vad_elapsed_ms;
          speech.vad_elapsed_ms_total = vad_elapsed_total;
          speech.audio_pushed_at = chunk.pushed_at;
          speech_queue.Push(std::move(speech));
          vad_in_speech = true;
          vad_start_ms = start_ms;
        };

        auto push_speech_chunk = [&](const AudioChunk& chunk,
                                     double vad_elapsed_ms) {
          if (!vad_in_speech || chunk.pcm.empty()) return;
          SpeechEvent speech;
          speech.type = SpeechEventType::kSpeechChunk;
          speech.utterance_id = utterance_id - 1;
          speech.start_ms = chunk.start_ms;
          speech.end_ms = chunk.end_ms;
          speech.timestamp_ms = chunk.start_ms;
          speech.pcm = chunk.pcm;
          speech.vad_input_samples = static_cast<int64_t>(chunk.pcm.size());
          speech.vad_input_ms = chunk.end_ms - chunk.start_ms;
          speech.vad_chunk_start_ms = chunk.start_ms;
          speech.vad_chunk_end_ms = chunk.end_ms;
          speech.speech_input_samples = static_cast<int64_t>(speech.pcm.size());
          speech.speech_input_ms = SamplesToMs(speech.pcm.size());
          speech.speech_event_type = "speech_chunk";
          speech.vad_elapsed_ms = vad_elapsed_ms;
          speech.vad_elapsed_ms_total = vad_elapsed_total;
          speech.audio_pushed_at = chunk.pushed_at;
          speech_queue.Push(std::move(speech));
        };

        auto push_speech_end = [&](int64_t end_ms,
                                   double vad_elapsed_ms,
                                   const AudioChunk& chunk) {
          if (!vad_in_speech) return;
          SpeechEvent speech;
          speech.type = SpeechEventType::kSpeechEnd;
          speech.utterance_id = utterance_id - 1;
          speech.start_ms = vad_start_ms;
          speech.end_ms = std::max<int64_t>(vad_start_ms, end_ms);
          speech.timestamp_ms = end_ms;
          speech.vad_input_samples = static_cast<int64_t>(chunk.pcm.size());
          speech.vad_input_ms = chunk.end_ms - chunk.start_ms;
          speech.vad_chunk_start_ms = chunk.start_ms;
          speech.vad_chunk_end_ms = chunk.end_ms;
          speech.speech_event_type = "speech_end";
          speech.vad_elapsed_ms = vad_elapsed_ms;
          speech.vad_elapsed_ms_total = vad_elapsed_total;
          speech.audio_pushed_at = chunk.pushed_at;
          speech_queue.Push(std::move(speech));
          vad_in_speech = false;
          vad_start_ms = -1;
        };

        while (true) {
          AudioChunk chunk = audio_queue.Pop();
          if (chunk.is_final) {
            const auto started_at = Clock::now();
            auto result = vad_model.FinishStreaming(&state);
            const double vad_elapsed_ms = ElapsedMs(started_at);
            vad_elapsed_total += vad_elapsed_ms;
            for (auto [start_ms, end_ms] : result.events_ms) {
              if (start_ms >= 0 && !vad_in_speech) push_speech_start(start_ms, chunk, vad_elapsed_ms);
              if (end_ms >= 0) push_speech_end(end_ms, vad_elapsed_ms, chunk);
            }
            if (vad_in_speech) push_speech_end(chunk.end_ms, vad_elapsed_ms, chunk);
            SpeechEvent done;
            done.type = SpeechEventType::kEndOfStream;
            done.vad_elapsed_ms_total = vad_elapsed_total;
            speech_queue.Push(std::move(done));
            return;
          }

          const auto started_at = Clock::now();
          auto result = vad_model.DetectStreaming(chunk.pcm, &state, false);
          const double vad_elapsed_ms = ElapsedMs(started_at);
          vad_elapsed_total += vad_elapsed_ms;

          bool had_event = false;
          bool sent_chunk_audio = false;
          for (auto [start_ms, end_ms] : result.events_ms) {
            had_event = true;
            bool started_in_this_chunk = false;
            if (start_ms >= 0) {
              if (vad_in_speech) push_speech_end(start_ms, vad_elapsed_ms, chunk);
              push_speech_start(start_ms, chunk, vad_elapsed_ms);
              sent_chunk_audio = true;
              started_in_this_chunk = true;
            }
            if (end_ms >= 0) {
              if (!vad_in_speech && start_ms >= 0) push_speech_start(start_ms, chunk, vad_elapsed_ms);
              if (!vad_in_speech && start_ms < 0) continue;
              if (!started_in_this_chunk && !sent_chunk_audio) {
                push_speech_chunk(chunk, vad_elapsed_ms);
                sent_chunk_audio = true;
              }
              push_speech_end(end_ms, vad_elapsed_ms, chunk);
            }
          }
          if (vad_in_speech && !had_event) push_speech_chunk(chunk, vad_elapsed_ms);
          if (vad_in_speech) {
            pre_roll.clear();
          } else {
            pre_roll = TrimTail(ConcatAudio(pre_roll, chunk.pcm), pre_roll_samples);
          }
        }
      } catch (const std::exception& ex) {
        StreamingAsrEvent event;
        event.type = StreamingAsrEventType::kError;
        event.error = std::string("vad_worker: ") + ex.what();
        fill_result_event(&event);
        result_queue.Push(std::move(event));
        SpeechEvent done;
        done.type = SpeechEventType::kEndOfStream;
        speech_queue.Push(std::move(done));
      }
    });

    std::thread asr_worker([&] {
      try {
        QwenAsrStreamingState state;
        const size_t asr_chunk_samples = static_cast<size_t>(DEFAULT_ASR_CHUNK_SEC * DEFAULT_SAMPLE_RATE);
        const std::optional<std::string> language =
            args.language.empty() ? std::nullopt : std::optional<std::string>(args.language);
        double asr_elapsed_total = 0.0;
        double last_vad_elapsed_total = 0.0;

        auto trim_asr_to_vad_end = [&](int64_t end_ms) {
          if (state.start_ms < 0 || end_ms < state.start_ms) return;
          const size_t target_samples = MsToSamples(end_ms - state.start_ms);
          if (target_samples < static_cast<size_t>(state.total_audio_samples)) {
            const size_t drop_samples =
                static_cast<size_t>(state.total_audio_samples) - target_samples;
            const size_t drop_from_accum = std::min(drop_samples, state.audio_accum.size());
            if (drop_from_accum > 0) {
              state.audio_accum.erase(
                  state.audio_accum.end() - static_cast<std::ptrdiff_t>(drop_from_accum),
                  state.audio_accum.end());
            }
            state.total_audio_samples = static_cast<int64_t>(target_samples);
            state.last_appended_end_samples =
                std::min<int64_t>(state.last_appended_end_samples, state.total_audio_samples);
            state.last_appended_start_samples =
                std::min<int64_t>(state.last_appended_start_samples, state.last_appended_end_samples);
            state.buffer.clear();
            return;
          }

          const size_t buffered_target = target_samples - static_cast<size_t>(state.total_audio_samples);
          if (buffered_target < state.buffer.size()) {
            state.buffer.resize(buffered_target);
          }
        };

        auto run_streaming_decode = [&](bool finalize,
                                        const SpeechEvent& source_event) -> double {
          if (state.audio_accum.empty() && state.raw_decoded.empty()) return 0.0;
          double asr_elapsed_ms = 0.0;
          if (!state.audio_accum.empty()) {
            const std::string prefix = StreamingPrefix(asr_model, state);
            state.last_prefix = prefix;
            const auto started_at = Clock::now();
            const auto asr = asr_model.TranscribeSamplesWithPrefix(
                state.audio_accum, args.context, language, args.hotwords, prefix, args.max_new_tokens);
            asr_elapsed_ms = ElapsedMs(started_at);
            asr_elapsed_total += asr_elapsed_ms;
            state.asr_elapsed_ms_total += asr_elapsed_ms;
            state.last_raw_generated = prefix.empty() ? asr.raw_text : asr.raw_text.substr(prefix.size());
            state.raw_decoded = asr.raw_text;
            state.raw_text = asr.raw_text;
            state.raw_decoded_token_ids = asr_model.EncodeText(state.raw_decoded);
            state.last_asr_prepare_inputs_ms = asr.prepare_inputs_ms;
            state.last_asr_generate_ms = asr.generate_ms;
            state.last_asr_token_embedding_ms = asr.token_embedding_ms;
            state.last_asr_audio_encoder_ms = asr.audio_encoder_ms;
            state.last_asr_merge_audio_features_ms = asr.merge_audio_features_ms;
            state.last_asr_prefill_ms = asr.prefill_ms;
            state.last_asr_decode_ms = asr.decode_ms;
            state.last_asr_decode_steps = asr.decode_steps;
          }
          CommitStreamingRaw(asr_model, &state, language, finalize);
          const int64_t decoded_chunk_id = state.chunk_id;
          state.chunk_id += 1;
          (void)decoded_chunk_id;
          state.end_ms = finalize ? source_event.end_ms
                                  : state.start_ms + SamplesToMs(static_cast<size_t>(state.last_appended_end_samples));
          return asr_elapsed_ms;
        };

        auto emit_asr_event = [&](bool final_event,
                                  const SpeechEvent& source_event,
                                  double asr_elapsed_ms) {
          if (!final_event && state.pending_event_text.empty() && state.confirmed_delta_text.empty() &&
              !state.last_text_revision) {
            return;
          }
          const int64_t audio_end_ms =
              final_event ? SamplesToMs(static_cast<size_t>(state.total_audio_samples))
                          : SamplesToMs(static_cast<size_t>(state.last_appended_end_samples));
          StreamingAsrEvent out;
          out.type = final_event ? StreamingAsrEventType::kFinal : StreamingAsrEventType::kPartial;
          out.utterance_id = state.utterance_id;
          out.start_ms = state.start_ms;
          out.end_ms = final_event ? source_event.end_ms : state.start_ms + audio_end_ms;
          out.audio_end_ms = audio_end_ms;
          out.is_final = final_event;
          out.confirmed_delta_text = state.confirmed_delta_text;
          out.confirmed_text = state.confirmed_text;
          out.pending_text = state.pending_text;
          out.text = final_event ? state.text : state.pending_text;
          out.full_text = state.text;
          out.raw_text = final_event ? state.raw_decoded : state.committed_raw_decoded;
          out.language = state.language;
          out.vad_elapsed_ms = source_event.vad_elapsed_ms;
          out.vad_elapsed_ms_total = source_event.vad_elapsed_ms_total;
          out.asr_elapsed_ms = asr_elapsed_ms;
          out.asr_elapsed_ms_total = asr_elapsed_total;
          out.emit_latency_ms = ElapsedMs(source_event.audio_pushed_at);
          CopySpeechMetadata(source_event, &out);
          FillAsrMetadata(state, state.last_prefix, state.last_text_revision, state.last_raw_revision, final_event, &out);
          fill_result_event(&out);
          result_queue.Push(std::move(out));
        };

        auto feed_asr = [&](const SpeechEvent& source_event) {
          if (source_event.pcm.empty()) return;
          state.pending_event_text.clear();
          state.confirmed_delta_text.clear();
          state.last_input_samples = static_cast<int64_t>(source_event.pcm.size());
          state.last_buffer_samples_before = static_cast<int64_t>(state.buffer.size());
          state.buffer.insert(state.buffer.end(), source_event.pcm.begin(), source_event.pcm.end());
          state.last_buffer_samples_after_append = static_cast<int64_t>(state.buffer.size());
          state.last_triggered_decode_count = 0;
          double last_asr_elapsed_ms = 0.0;
          while (state.buffer.size() >= asr_chunk_samples) {
            auto chunk = TakeFront(&state.buffer, asr_chunk_samples);
            AppendStreamAudio(&state, chunk);
            last_asr_elapsed_ms = run_streaming_decode(false, source_event);
            state.last_triggered_decode_count += 1;
          }
          if (DEFAULT_ENABLE_PARTIAL_ASR) emit_asr_event(false, source_event, last_asr_elapsed_ms);
        };

        auto finish_asr = [&](const SpeechEvent& source_event) {
          state.pending_event_text.clear();
          state.confirmed_delta_text.clear();
          trim_asr_to_vad_end(source_event.end_ms);
          double asr_elapsed_ms = 0.0;
          state.last_buffer_samples_before = static_cast<int64_t>(state.buffer.size());
          state.last_triggered_decode_count = 0;
          if (!state.buffer.empty()) {
            auto tail = TakeFront(&state.buffer, state.buffer.size());
            state.last_input_samples = static_cast<int64_t>(tail.size());
            AppendStreamAudio(&state, tail);
            state.last_buffer_samples_after_append = static_cast<int64_t>(state.buffer.size());
            asr_elapsed_ms = run_streaming_decode(true, source_event);
            state.last_triggered_decode_count = 1;
          } else {
            state.last_input_samples = 0;
            state.last_buffer_samples_after_append = static_cast<int64_t>(state.buffer.size());
            asr_elapsed_ms = run_streaming_decode(true, source_event);
          }
          emit_asr_event(true, source_event, asr_elapsed_ms);
        };

        while (true) {
          SpeechEvent event = speech_queue.Pop();
          last_vad_elapsed_total = event.vad_elapsed_ms_total;
          if (event.type == SpeechEventType::kEndOfStream) {
            if (!state.audio_accum.empty() || !state.buffer.empty() || !state.raw_decoded.empty()) {
              if (state.start_ms >= 0) event.end_ms = state.start_ms + SamplesToMs(static_cast<size_t>(state.total_audio_samples));
              finish_asr(event);
            }
            StreamingAsrEvent done;
            done.type = StreamingAsrEventType::kEndOfStream;
            done.utterance_id = -1;
            done.vad_elapsed_ms_total = last_vad_elapsed_total;
            done.asr_elapsed_ms_total = asr_elapsed_total;
            fill_result_event(&done);
            result_queue.Push(std::move(done));
            return;
          }
          if (event.type == SpeechEventType::kError) {
            StreamingAsrEvent error;
            error.type = StreamingAsrEventType::kError;
            error.error = event.error;
            fill_result_event(&error);
            result_queue.Push(std::move(error));
            continue;
          }
          if (event.type == SpeechEventType::kSpeechStart) {
            state = QwenAsrStreamingState{};
            state.utterance_id = event.utterance_id;
            state.start_ms = event.start_ms;
            state.end_ms = event.end_ms;
            StreamingAsrEvent start;
            start.type = StreamingAsrEventType::kSpeechStart;
            start.utterance_id = state.utterance_id;
            start.start_ms = state.start_ms;
            start.end_ms = state.start_ms;
            start.audio_end_ms = event.timestamp_ms;
            start.vad_elapsed_ms = event.vad_elapsed_ms;
            start.vad_elapsed_ms_total = event.vad_elapsed_ms_total;
            start.asr_elapsed_ms_total = asr_elapsed_total;
            start.emit_latency_ms = ElapsedMs(event.audio_pushed_at);
            CopySpeechMetadata(event, &start);
            fill_result_event(&start);
            result_queue.Push(std::move(start));
            feed_asr(event);
          } else if (event.type == SpeechEventType::kSpeechChunk) {
            state.end_ms = event.end_ms;
            feed_asr(event);
          } else if (event.type == SpeechEventType::kSpeechEnd) {
            finish_asr(event);
            state = QwenAsrStreamingState{};
          }
        }
      } catch (const std::exception& ex) {
        StreamingAsrEvent event;
        event.type = StreamingAsrEventType::kError;
        event.error = std::string("asr_worker: ") + ex.what();
        fill_result_event(&event);
        result_queue.Push(std::move(event));
        StreamingAsrEvent done;
        done.type = StreamingAsrEventType::kEndOfStream;
        done.utterance_id = -1;
        fill_result_event(&done);
        result_queue.Push(std::move(done));
      }
    });

    while (true) {
      StreamingAsrEvent event = result_queue.Pop();
      LogEvent(log, event);
      if (event.type == StreamingAsrEventType::kEndOfStream) break;
    }

    producer.join();
    vad_worker.join();
    asr_worker.join();
    log << "[program_done]\n";
    log.flush();
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    PrintUsage(argv[0]);
    return 1;
  }
}
