#include "qwen3asr/pipeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace qwen3asr {
namespace {

constexpr int kSampleRate = 16000;

int64_t SamplesToMs(size_t samples) {
  return static_cast<int64_t>(std::llround(static_cast<double>(samples) * 1000.0 / kSampleRate));
}

size_t MsToSamples(int64_t ms) {
  return static_cast<size_t>(std::max<int64_t>(0, std::llround(static_cast<double>(ms) * kSampleRate / 1000.0)));
}

double ElapsedMs(std::chrono::steady_clock::time_point started_at) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started_at).count();
}

}  // namespace

EnergyVadAsrPipeline::EnergyVadAsrPipeline(const Qwen3ASROnnxModel& asr_model, EnergyVadOptions options)
    : asr_model_(asr_model), options_(options) {}

VadAsrResult EnergyVadAsrPipeline::Transcribe(const std::vector<float>& samples,
                                              const std::string& context,
                                              const std::optional<std::string>& language,
                                              const std::string& hotwords,
                                              int max_new_tokens,
                                              VadAsrProgressCallback progress) const {
  (void)progress;
  const auto total_started_at = std::chrono::steady_clock::now();
  VadAsrResult result;
  result.sample_rate = kSampleRate;
  result.duration_ms = SamplesToMs(samples.size());

  const auto vad_started_at = std::chrono::steady_clock::now();
  result.vad_segments_ms = DetectSpeech(samples);
  result.vad_elapsed_ms = ElapsedMs(vad_started_at);

  const auto prepared = PrepareSegments(result.vad_segments_ms, result.duration_ms);
  std::string previous_language;
  const auto asr_started_at = std::chrono::steady_clock::now();
  for (const auto& prepared_segment : prepared) {
    auto clip = SliceMs(samples, prepared_segment.start_ms, prepared_segment.end_ms);
    const auto segment_started_at = std::chrono::steady_clock::now();
    auto asr = asr_model_.TranscribeSamples(clip, context, language, hotwords, max_new_tokens);
    const double segment_asr_ms = ElapsedMs(segment_started_at);
    result.text += asr.text;
    result.raw_text += asr.raw_text;
    const auto lang = asr.language;
    if (!lang.empty() && lang != previous_language) {
      if (!result.language.empty()) result.language += ",";
      result.language += lang;
      previous_language = lang;
    }
    result.segments.push_back(VadAsrSegment{
        prepared_segment.start_ms,
        prepared_segment.end_ms,
        prepared_segment.vad_start_ms,
        prepared_segment.vad_end_ms,
        asr.language,
        asr.text,
        asr.raw_text,
        segment_asr_ms,
    });
  }
  result.asr_elapsed_ms = ElapsedMs(asr_started_at);
  result.total_elapsed_ms = ElapsedMs(total_started_at);
  return result;
}

std::vector<std::pair<int64_t, int64_t>> EnergyVadAsrPipeline::DetectSpeech(
    const std::vector<float>& samples) const {
  if (samples.empty()) return {};
  const int frame_samples = std::max(1, static_cast<int>(MsToSamples(options_.frame_ms)));
  const int hop_samples = std::max(1, static_cast<int>(MsToSamples(options_.hop_ms)));
  const int min_speech_frames = std::max(1, options_.min_speech_ms / std::max(1, options_.hop_ms));
  const int min_silence_frames = std::max(1, options_.min_silence_ms / std::max(1, options_.hop_ms));

  std::vector<std::pair<int64_t, int64_t>> segments;
  bool in_speech = false;
  int speech_count = 0;
  int silence_count = 0;
  int64_t speech_start_ms = 0;

  for (size_t start = 0, frame_id = 0; start < samples.size(); start += hop_samples, ++frame_id) {
    const size_t end = std::min(samples.size(), start + static_cast<size_t>(frame_samples));
    double energy = 0.0;
    for (size_t i = start; i < end; ++i) energy += static_cast<double>(samples[i]) * samples[i];
    energy /= std::max<size_t>(1, end - start);
    const float db = static_cast<float>(10.0 * std::log10(std::max(energy, 1.0e-12)));
    const bool speech = db >= options_.threshold_db;

    if (speech) {
      ++speech_count;
      silence_count = 0;
      if (!in_speech && speech_count >= min_speech_frames) {
        in_speech = true;
        const int64_t start_frame = static_cast<int64_t>(frame_id + 1 - min_speech_frames);
        speech_start_ms = start_frame * options_.hop_ms;
      }
    } else {
      speech_count = 0;
      if (in_speech) {
        ++silence_count;
        if (silence_count >= min_silence_frames) {
          const int64_t end_frame = static_cast<int64_t>(frame_id + 1 - min_silence_frames);
          const int64_t speech_end_ms = std::max<int64_t>(speech_start_ms, end_frame * options_.hop_ms);
          segments.push_back({speech_start_ms, speech_end_ms});
          in_speech = false;
          silence_count = 0;
        }
      }
    }
  }
  if (in_speech) segments.push_back({speech_start_ms, SamplesToMs(samples.size())});
  return segments;
}

std::vector<EnergyVadAsrPipeline::PreparedSegment> EnergyVadAsrPipeline::PrepareSegments(
    const std::vector<std::pair<int64_t, int64_t>>& vad_segments,
    int64_t duration_ms) const {
  std::vector<PreparedSegment> padded;
  for (auto [vad_start_ms, vad_end_ms] : vad_segments) {
    if (vad_end_ms - vad_start_ms < options_.min_speech_ms) continue;
    const int64_t start_ms = std::max<int64_t>(0, vad_start_ms - options_.pad_start_ms);
    const int64_t end_ms = std::min<int64_t>(duration_ms, vad_end_ms + options_.pad_end_ms);
    if (end_ms > start_ms) padded.push_back({start_ms, end_ms, vad_start_ms, vad_end_ms});
  }
  if (padded.empty()) return {};
  std::vector<PreparedSegment> merged{padded.front()};
  for (size_t i = 1; i < padded.size(); ++i) {
    auto& last = merged.back();
    if (padded[i].start_ms - last.end_ms <= options_.merge_gap_ms) {
      last.end_ms = std::max(last.end_ms, padded[i].end_ms);
      last.vad_end_ms = std::max(last.vad_end_ms, padded[i].vad_end_ms);
    } else {
      merged.push_back(padded[i]);
    }
  }
  return merged;
}

std::vector<float> EnergyVadAsrPipeline::SliceMs(const std::vector<float>& samples,
                                                 int64_t start_ms,
                                                 int64_t end_ms) const {
  const size_t start = std::min(samples.size(), MsToSamples(start_ms));
  const size_t end = std::min(samples.size(), MsToSamples(end_ms));
  if (end <= start) return {};
  return {samples.begin() + static_cast<std::ptrdiff_t>(start), samples.begin() + static_cast<std::ptrdiff_t>(end)};
}

FsmnVadAsrPipeline::FsmnVadAsrPipeline(const FsmnVadOnnxModel& vad_model,
                                       const Qwen3ASROnnxModel& asr_model)
    : vad_model_(vad_model), asr_model_(asr_model) {}

VadAsrResult FsmnVadAsrPipeline::Transcribe(const std::vector<float>& samples,
                                            const std::string& context,
                                            const std::optional<std::string>& language,
                                            const std::string& hotwords,
                                            int max_new_tokens,
                                            VadAsrProgressCallback progress) const {
  const auto total_started_at = std::chrono::steady_clock::now();
  VadAsrResult result;
  result.sample_rate = kSampleRate;
  result.duration_ms = SamplesToMs(samples.size());

  if (progress) {
    VadAsrProgressEvent event;
    event.type = "run_start";
    event.duration_ms = result.duration_ms;
    progress(event);
  }

  const auto vad = vad_model_.DetectOffline(samples);
  result.vad_segments_ms = vad.segments_ms;
  result.vad_elapsed_ms = vad.elapsed_ms;

  const auto prepared = PrepareSegments(result.vad_segments_ms, result.duration_ms);
  if (progress) {
    VadAsrProgressEvent event;
    event.type = "vad_done";
    event.duration_ms = result.duration_ms;
    event.vad_elapsed_ms = result.vad_elapsed_ms;
    event.segment_count = static_cast<int64_t>(prepared.size());
    event.vad_segments_ms = result.vad_segments_ms;
    progress(event);
  }

  std::string previous_language;
  const auto asr_started_at = std::chrono::steady_clock::now();
  for (size_t i = 0; i < prepared.size(); ++i) {
    const auto& prepared_segment = prepared[i];
    if (progress) {
      VadAsrProgressEvent event;
      event.type = "asr_segment_start";
      event.segment_index = static_cast<int64_t>(i);
      event.segment_count = static_cast<int64_t>(prepared.size());
      event.segment.start_ms = prepared_segment.start_ms;
      event.segment.end_ms = prepared_segment.end_ms;
      event.segment.vad_start_ms = prepared_segment.vad_start_ms;
      event.segment.vad_end_ms = prepared_segment.vad_end_ms;
      progress(event);
    }
    auto clip = SliceMs(samples, prepared_segment.start_ms, prepared_segment.end_ms);
    const auto segment_started_at = std::chrono::steady_clock::now();
    auto asr = asr_model_.TranscribeSamples(clip, context, language, hotwords, max_new_tokens);
    const double segment_asr_ms = ElapsedMs(segment_started_at);
    result.text += asr.text;
    result.raw_text += asr.raw_text;
    const auto lang = asr.language;
    if (!lang.empty() && lang != previous_language) {
      if (!result.language.empty()) result.language += ",";
      result.language += lang;
      previous_language = lang;
    }
    result.segments.push_back(VadAsrSegment{
        prepared_segment.start_ms,
        prepared_segment.end_ms,
        prepared_segment.vad_start_ms,
        prepared_segment.vad_end_ms,
        asr.language,
        asr.text,
        asr.raw_text,
        segment_asr_ms,
    });
    if (progress) {
      VadAsrProgressEvent event;
      event.type = "asr_segment_done";
      event.segment_index = static_cast<int64_t>(i);
      event.segment_count = static_cast<int64_t>(prepared.size());
      event.segment = result.segments.back();
      event.asr_elapsed_ms = segment_asr_ms;
      event.text = result.text;
      progress(event);
    }
  }
  result.asr_elapsed_ms = ElapsedMs(asr_started_at);
  result.total_elapsed_ms = ElapsedMs(total_started_at);
  if (progress) {
    VadAsrProgressEvent event;
    event.type = "run_done";
    event.duration_ms = result.duration_ms;
    event.vad_elapsed_ms = result.vad_elapsed_ms;
    event.asr_elapsed_ms = result.asr_elapsed_ms;
    event.elapsed_ms = result.total_elapsed_ms;
    event.text = result.text;
    progress(event);
  }
  return result;
}

std::vector<FsmnVadAsrPipeline::PreparedSegment> FsmnVadAsrPipeline::PrepareSegments(
    const std::vector<std::pair<int64_t, int64_t>>& vad_segments,
    int64_t duration_ms) const {
  std::vector<PreparedSegment> padded;
  for (auto [vad_start_ms, vad_end_ms] : vad_segments) {
    vad_start_ms = std::max<int64_t>(0, vad_start_ms);
    vad_end_ms = std::min<int64_t>(duration_ms, vad_end_ms);
    if (vad_end_ms - vad_start_ms < min_segment_ms_) continue;
    const int64_t start_ms = std::max<int64_t>(0, vad_start_ms - pad_start_ms_);
    const int64_t end_ms = std::min<int64_t>(duration_ms, vad_end_ms + pad_end_ms_);
    if (end_ms > start_ms) padded.push_back({start_ms, end_ms, vad_start_ms, vad_end_ms});
  }
  if (padded.empty()) return {};
  std::vector<PreparedSegment> merged{padded.front()};
  for (size_t i = 1; i < padded.size(); ++i) {
    auto& last = merged.back();
    if (padded[i].start_ms - last.end_ms <= merge_gap_ms_) {
      last.end_ms = std::max(last.end_ms, padded[i].end_ms);
      last.vad_end_ms = std::max(last.vad_end_ms, padded[i].vad_end_ms);
    } else {
      merged.push_back(padded[i]);
    }
  }
  return merged;
}

std::vector<float> FsmnVadAsrPipeline::SliceMs(const std::vector<float>& samples,
                                               int64_t start_ms,
                                               int64_t end_ms) const {
  const size_t start = std::min(samples.size(), MsToSamples(start_ms));
  const size_t end = std::min(samples.size(), MsToSamples(end_ms));
  if (end <= start) return {};
  return {samples.begin() + static_cast<std::ptrdiff_t>(start), samples.begin() + static_cast<std::ptrdiff_t>(end)};
}

}  // namespace qwen3asr
