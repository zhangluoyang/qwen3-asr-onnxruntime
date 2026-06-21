#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "qwen3asr/bpe_tokenizer.h"
#include "qwen3asr/ort_runner.h"
#include "qwen3asr/tensor.h"

namespace qwen3asr {

struct Qwen3ASRConfig {
  std::filesystem::path model_dir;
  std::filesystem::path onnx_dir;
  bool use_cuda = true;
  int cuda_device_id = 0;
  int max_new_tokens = 1024;
};

struct ASRResult {
  std::string language;
  std::string text;
  std::string raw_text;
  int64_t audio_tokens = 0;
  int64_t mel_frames = 0;
  int64_t chunk_count = 0;
  double elapsed_ms = 0.0;
  double prepare_inputs_ms = 0.0;
  double generate_ms = 0.0;
  double token_embedding_ms = 0.0;
  double audio_encoder_ms = 0.0;
  double merge_audio_features_ms = 0.0;
  double prefill_ms = 0.0;
  double decode_ms = 0.0;
  int64_t decode_steps = 0;
};

class Qwen3ASROnnxModel {
 public:
  explicit Qwen3ASROnnxModel(Qwen3ASRConfig config);
  static FloatTensor WhisperLogMel(const std::vector<float>& audio);

  ASRResult TranscribeFile(const std::filesystem::path& audio_path,
                           const std::string& context = "",
                           const std::optional<std::string>& language = std::nullopt,
                           const std::string& hotwords = "",
                           int max_new_tokens = -1);
  ASRResult TranscribeSamples(const std::vector<float>& samples,
                              const std::string& context = "",
                              const std::optional<std::string>& language = std::nullopt,
                              const std::string& hotwords = "",
                              int max_new_tokens = -1) const;
  ASRResult TranscribeSamplesWithPrefix(const std::vector<float>& samples,
                                        const std::string& context,
                                        const std::optional<std::string>& language,
                                        const std::string& hotwords,
                                        const std::string& output_prefix,
                                        int max_new_tokens = -1) const;
  ASRResult ParseRawText(const std::string& raw_text,
                         const std::optional<std::string>& language = std::nullopt) const;
  std::vector<int64_t> EncodeText(const std::string& text) const;
  std::string DecodeTokenIds(const std::vector<int64_t>& ids, bool skip_special_tokens = false) const;
  void WarmUp() const;
  const Qwen3ASRConfig& Config() const { return config_; }
  double AudioEncoderLoadSeconds() const { return audio_encoder_.LoadSeconds(); }
  double TokenEmbeddingLoadSeconds() const { return token_embedding_.LoadSeconds(); }
  double TextCoreLoadSeconds() const { return text_core_.LoadSeconds(); }
  FloatTensor DebugPrefillEmbedsForFile(const std::filesystem::path& audio_path,
                                        const std::string& context = "",
                                        const std::optional<std::string>& language = std::nullopt,
                                        const std::string& hotwords = "") const;

 private:
  struct PreparedInputs {
    Int64Tensor input_ids;
    Int64Tensor attention_mask;
    FloatTensor input_features;
    int64_t audio_tokens = 0;
    int64_t mel_frames = 0;
  };
  struct GeneratedIds {
    std::vector<int64_t> ids;
    double token_embedding_ms = 0.0;
    double audio_encoder_ms = 0.0;
    double merge_audio_features_ms = 0.0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    int64_t decode_steps = 0;
  };

  static std::string ContextWithHotwords(const std::string& context, const std::string& hotwords);
  static std::string BuildPrompt(const std::string& context,
                                 int64_t audio_tokens,
                                 const std::optional<std::string>& language,
                                 const std::string& output_prefix = "");
  static int64_t AudioTokenCount(int64_t mel_frames);
  static std::vector<std::vector<float>> SplitAudioIntoChunks(const std::vector<float>& samples);
  static ASRResult ParseOutput(const std::string& raw_text, const std::optional<std::string>& forced_language);
  static int64_t GreedyArgmax(const std::vector<float>& logits);

  PreparedInputs PrepareInputs(const std::vector<float>& samples,
                               const std::string& context,
                               const std::optional<std::string>& language,
                               const std::string& hotwords,
                               const std::string& output_prefix = "") const;
  FloatTensor RunAudioEncoder(const FloatTensor& input_features) const;
  FloatTensor RunTokenEmbedding(const Int64Tensor& input_ids) const;
  FloatTensor MergeAudioFeatures(const Int64Tensor& input_ids,
                                 const FloatTensor& token_embeds,
                                 const FloatTensor& audio_features) const;
  std::vector<Ort::Value> RunTextCore(std::unordered_map<std::string, Ort::Value>& inputs,
                                      std::vector<float>* last_logits) const;
  GeneratedIds GenerateIds(const PreparedInputs& inputs, int max_new_tokens) const;
  ASRResult TranscribeChunk(const std::vector<float>& samples,
                            const std::string& context,
                            const std::optional<std::string>& language,
                            const std::string& hotwords,
                            int max_new_tokens,
                            const std::string& output_prefix = "") const;

  Qwen3ASRConfig config_;
  Ort::Env env_;
  Qwen2BpeTokenizer tokenizer_;
  OrtRunner audio_encoder_;
  OrtRunner token_embedding_;
  OrtRunner text_core_;
  std::vector<std::string> text_core_output_names_;
  std::unordered_set<std::string> text_core_device_output_names_;
  mutable std::unordered_map<int64_t, std::vector<int64_t>> decode_attention_cache_;
  mutable std::unordered_map<int64_t, std::vector<int64_t>> decode_cache_position_cache_;
};

}  // namespace qwen3asr
