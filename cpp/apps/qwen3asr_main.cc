#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>

#include "qwen3asr/audio_frontend.h"
#include "qwen3asr/pipeline.h"
#include "qwen3asr/qwen_model.h"

namespace {

// ======================== 默认运行配置 ========================
// 直接运行 qwen3asr_run 即可使用下面这些配置：
//
//     ./qwen3asr_run
//
// 如需临时覆盖，也可以继续传命令行参数，例如：
//
//     ./qwen3asr_run --audio data/voice_design_nonstream.wav --max-new-tokens 128

// Qwen3-ASR 原始模型目录。C++ 运行时会从这里读取 tokenizer 的 vocab/merges/special tokens。
constexpr const char* DEFAULT_MODEL_DIR = "/home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-ASR-0.6B";

// 已导出的 Qwen3-ASR ONNX 目录，应包含 audio_encoder、token_embedding、text_core 三个子目录。
constexpr const char* DEFAULT_ONNX_DIR = "./onnx_asr";

// 已导出的 FunASR FSMN-VAD ONNX 文件。
constexpr const char* DEFAULT_VAD_ONNX_PATH = "./onnx_fsmn/fsmn_vad_vad_encoder.onnx";

// FunASR FSMN-VAD 的 CMVN 文件，必须和 VAD ONNX 配套。
constexpr const char* DEFAULT_VAD_CMVN_PATH = "./onnx_fsmn/am.mvn";

// 默认测试音频。这里使用前面截好的 10 分钟音频。
constexpr const char* DEFAULT_AUDIO_PATH = "./data/voice_design_stream.wav";

// 默认走 Python 同款 FSMN-VAD + ASR：先由 FSMN-VAD 找语音段，再逐段送 Qwen3-ASR。
constexpr bool DEFAULT_RUN_FSMN_VAD = true;

// 是否默认使用 CUDAExecutionProvider。false 会强制 CPU。
constexpr bool DEFAULT_USE_CUDA = true;

// 默认强制识别语言。为空字符串表示让模型自己判断；已知中文音频时用 Chinese 更稳定。
constexpr const char* DEFAULT_LANGUAGE = "Chinese";

// 默认 ASR 上下文提示词。保持空字符串，避免让模型改写或总结。
constexpr const char* DEFAULT_CONTEXT = "";

// 默认热词。多个热词可以用逗号分隔；这里只作为识别偏置。
constexpr const char* DEFAULT_HOTWORDS = "";

// 单段 ASR 最大生成 token 数。语音段较长时可适当增大。
constexpr int DEFAULT_MAX_NEW_TOKENS = 1024;

// Energy VAD 默认阈值。数值越高越严格，越低越容易把弱声音判为语音。
constexpr float DEFAULT_VAD_THRESHOLD_DB = -45.0f;

// 默认实时日志文件。程序运行时会边跑边写，方便 tail -f 查看进度。
constexpr const char* DEFAULT_LOG_PATH = "./qwen3asr_cpp_run.log";

struct Args {
  std::filesystem::path model_dir = DEFAULT_MODEL_DIR;
  std::filesystem::path onnx_dir = DEFAULT_ONNX_DIR;
  std::filesystem::path vad_onnx_path = DEFAULT_VAD_ONNX_PATH;
  std::filesystem::path vad_cmvn_path = DEFAULT_VAD_CMVN_PATH;
  std::filesystem::path audio = DEFAULT_AUDIO_PATH;
  std::string context = DEFAULT_CONTEXT;
  std::string language = DEFAULT_LANGUAGE;
  std::string hotwords = DEFAULT_HOTWORDS;
  std::filesystem::path log_file = DEFAULT_LOG_PATH;
  int max_new_tokens = DEFAULT_MAX_NEW_TOKENS;
  bool use_cuda = DEFAULT_USE_CUDA;
  bool debug_mel = false;
  bool debug_embeds = false;
  bool debug_vad = false;
  bool debug_vad_streaming = false;
  bool debug_vad_feats = false;
  bool fsmn_vad = DEFAULT_RUN_FSMN_VAD;
  bool energy_vad = false;
  float vad_threshold_db = DEFAULT_VAD_THRESHOLD_DB;
};

void PrintUsage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0 << " [options]\n"
      << "不传参数时会使用源码顶部的默认配置直接运行。\n\n"
      << "Options:\n"
      << "  --audio WAV            临时覆盖默认音频路径\n"
      << "  --model-dir PATH       临时覆盖 Qwen3-ASR 原始模型目录\n"
      << "  --onnx-dir PATH        临时覆盖 ONNX 导出目录\n"
      << "  --vad-onnx PATH        临时覆盖 FSMN-VAD ONNX 文件\n"
      << "  --vad-cmvn PATH        临时覆盖 FSMN-VAD CMVN 文件\n"
      << "  --context TEXT         临时覆盖 ASR 上下文提示词\n"
      << "  --language NAME        临时覆盖强制语言，例如 Chinese / English；传空字符串则自动判断\n"
      << "  --hotwords TEXT        临时覆盖热词提示\n"
      << "  --log-file PATH        临时覆盖实时日志文件路径\n"
      << "  --max-new-tokens N     临时覆盖单段最大生成 token 数\n"
      << "  --debug-mel            只打印 C++ Whisper 特征统计后退出\n"
      << "  --debug-embeds         只打印 prefill embedding 统计后退出\n"
      << "  --debug-vad            只打印 FSMN-VAD 分段后退出\n"
      << "  --debug-vad-streaming  只按 200ms chunk 打印 FSMN-VAD 流式事件后退出\n"
      << "  --debug-vad-feats      只打印 FSMN-VAD fbank/feats/scores 统计后退出\n"
      << "  --fsmn-vad             使用 Python 同款 FSMN-VAD + ASR（默认）\n"
      << "  --energy-vad           调试用：使用简化 Energy VAD + ASR\n"
      << "  --full-asr             不做 VAD，直接整段 ASR\n"
      << "  --vad-threshold-db N   临时覆盖 Energy VAD 阈值\n"
      << "  --cpu                  强制使用 CPUExecutionProvider\n";
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
    else if (key == "--context") args.context = need_value(key);
    else if (key == "--language") args.language = need_value(key);
    else if (key == "--hotwords") args.hotwords = need_value(key);
    else if (key == "--log-file") args.log_file = need_value(key);
    else if (key == "--max-new-tokens") args.max_new_tokens = std::stoi(need_value(key));
    else if (key == "--debug-mel") args.debug_mel = true;
    else if (key == "--debug-embeds") args.debug_embeds = true;
    else if (key == "--debug-vad") args.debug_vad = true;
    else if (key == "--debug-vad-streaming") args.debug_vad_streaming = true;
    else if (key == "--debug-vad-feats") args.debug_vad_feats = true;
    else if (key == "--fsmn-vad") {
      args.fsmn_vad = true;
      args.energy_vad = false;
    } else if (key == "--energy-vad") {
      args.energy_vad = true;
      args.fsmn_vad = false;
    } else if (key == "--full-asr") {
      args.energy_vad = false;
      args.fsmn_vad = false;
    }
    else if (key == "--vad-threshold-db") args.vad_threshold_db = std::stof(need_value(key));
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

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = ParseArgs(argc, argv);
    if (args.debug_mel) {
      const auto audio = qwen3asr::LoadAudioMono(args.audio, 16000);
      const auto mel = qwen3asr::Qwen3ASROnnxModel::WhisperLogMel(audio.samples);
      const double sum = std::accumulate(mel.values().begin(), mel.values().end(), 0.0);
      std::cout << "shape:";
      for (auto d : mel.shape()) std::cout << " " << d;
      std::cout << "\nmean: " << sum / static_cast<double>(mel.size()) << "\nhead:";
      for (size_t i = 0; i < std::min<size_t>(10, mel.size()); ++i) std::cout << " " << mel.values()[i];
      std::cout << "\n";
      return 0;
    }
    if (args.debug_vad || args.debug_vad_streaming || args.debug_vad_feats) {
      const auto audio = qwen3asr::LoadAudioMono(args.audio, 16000);
      qwen3asr::FsmnVadConfig vad_config;
      vad_config.onnx_path = args.vad_onnx_path;
      vad_config.cmvn_path = args.vad_cmvn_path;
      vad_config.use_cuda = false;
      qwen3asr::FsmnVadOnnxModel vad_model(vad_config);
      if (args.debug_vad_feats) {
        auto print_tensor = [](const char* name, const qwen3asr::FloatTensor& tensor) {
          const double sum = std::accumulate(tensor.values().begin(), tensor.values().end(), 0.0);
          std::cout << name << "_shape:";
          for (auto d : tensor.shape()) std::cout << " " << d;
          std::cout << "\n" << name << "_mean: "
                    << (tensor.size() ? sum / static_cast<double>(tensor.size()) : 0.0) << "\n"
                    << name << "_head:";
          for (size_t i = 0; i < std::min<size_t>(10, tensor.size()); ++i) std::cout << " " << tensor.values()[i];
          std::cout << "\n";
        };
        print_tensor("fbank", vad_model.DebugFbank(audio.samples));
        print_tensor("feats", vad_model.DebugFeatures(audio.samples));
        print_tensor("scores", vad_model.DebugScores(audio.samples));
        return 0;
      }
      if (args.debug_vad_streaming) {
        auto state = vad_model.InitStreamingState(200);
        const size_t chunk_samples = static_cast<size_t>(16000 * 200 / 1000);
        for (size_t start = 0, chunk_id = 0; start < audio.samples.size(); start += chunk_samples, ++chunk_id) {
          const size_t end = std::min(audio.samples.size(), start + chunk_samples);
          std::vector<float> chunk(audio.samples.begin() + static_cast<std::ptrdiff_t>(start),
                                   audio.samples.begin() + static_cast<std::ptrdiff_t>(end));
          auto result = vad_model.DetectStreaming(chunk, &state, false);
          if (!result.events_ms.empty()) {
            std::cout << "chunk=" << chunk_id << " t=" << (end * 1000 / 16000) << "ms events=";
            for (auto [beg, fin] : result.events_ms) std::cout << "[" << beg << "," << fin << "]";
            std::cout << "\n";
          }
        }
        auto final = vad_model.FinishStreaming(&state);
        if (!final.events_ms.empty()) {
          std::cout << "final_events=";
          for (auto [beg, fin] : final.events_ms) std::cout << "[" << beg << "," << fin << "]";
          std::cout << "\n";
        }
        return 0;
      }
      const auto vad = vad_model.DetectOffline(audio.samples);
      std::cout << "vad_elapsed_ms: " << vad.elapsed_ms << "\n";
      std::cout << "vad_frame_count: " << vad.frame_count << "\n";
      std::cout << "vad_segments_ms:";
      for (const auto& item : vad.segments_ms) std::cout << " [" << item.first << "," << item.second << "]";
      std::cout << "\n";
      return 0;
    }
    std::ofstream progress_log;
    auto write_log = [&](const std::string& line) {
      if (!progress_log.is_open()) return;
      progress_log << line << "\n";
      progress_log.flush();
    };
    if (args.fsmn_vad) {
      progress_log.open(args.log_file, std::ios::out | std::ios::trunc);
      if (!progress_log) {
        throw std::runtime_error("failed to open log file: " + args.log_file.string());
      }
      std::cout << "realtime_log_file: " << args.log_file << "\n";
      std::cout.flush();
      write_log("[program_start] audio=" + args.audio.string());
      write_log("[model_loading] asr_model_dir=" + args.model_dir.string() +
                " asr_onnx_dir=" + args.onnx_dir.string());
    }
    qwen3asr::Qwen3ASRConfig config;
    config.model_dir = args.model_dir;
    config.onnx_dir = args.onnx_dir;
    config.use_cuda = args.use_cuda;
    config.max_new_tokens = args.max_new_tokens;

    qwen3asr::Qwen3ASROnnxModel model(config);
    std::optional<std::string> language;
    if (!args.language.empty()) language = args.language;
    if (args.debug_embeds) {
      const auto embeds = model.DebugPrefillEmbedsForFile(args.audio, args.context, language, args.hotwords);
      const double sum = std::accumulate(embeds.values().begin(), embeds.values().end(), 0.0);
      std::cout << "shape:";
      for (auto d : embeds.shape()) std::cout << " " << d;
      std::cout << "\nmean: " << sum / static_cast<double>(embeds.size()) << "\nhead:";
      for (size_t i = 0; i < std::min<size_t>(10, embeds.size()); ++i) std::cout << " " << embeds.values()[i];
      std::cout << "\n";
      return 0;
    }
    if (args.fsmn_vad) {
      auto audio = qwen3asr::LoadAudioMono(args.audio, 16000);
      auto progress = [&](const qwen3asr::VadAsrProgressEvent& event) {
        if (event.type == "run_start") {
          write_log("[run_start] audio=" + args.audio.string() +
                    " duration_ms=" + std::to_string(event.duration_ms));
        } else if (event.type == "vad_done") {
          write_log("[vad_done] vad_elapsed_ms=" + std::to_string(event.vad_elapsed_ms) +
                    " vad_segment_count=" + std::to_string(event.vad_segments_ms.size()) +
                    " asr_segment_count=" + std::to_string(event.segment_count));
          std::string line = "[vad_segments_ms]";
          for (const auto& item : event.vad_segments_ms) {
            line += " [" + std::to_string(item.first) + "," + std::to_string(item.second) + "]";
          }
          write_log(line);
        } else if (event.type == "asr_segment_start") {
          write_log("[asr_segment_start] index=" + std::to_string(event.segment_index + 1) +
                    "/" + std::to_string(event.segment_count) +
                    " audio_ms=" + std::to_string(event.segment.start_ms) +
                    "-" + std::to_string(event.segment.end_ms) +
                    " vad_ms=" + std::to_string(event.segment.vad_start_ms) +
                    "-" + std::to_string(event.segment.vad_end_ms));
        } else if (event.type == "asr_segment_done") {
          write_log("[asr_segment_done] index=" + std::to_string(event.segment_index + 1) +
                    "/" + std::to_string(event.segment_count) +
                    " asr_elapsed_ms=" + std::to_string(event.asr_elapsed_ms) +
                    " language=" + event.segment.language +
                    " text=" + event.segment.text);
        } else if (event.type == "run_done") {
          write_log("[run_done] vad_elapsed_ms=" + std::to_string(event.vad_elapsed_ms) +
                    " asr_elapsed_ms=" + std::to_string(event.asr_elapsed_ms) +
                    " total_elapsed_ms=" + std::to_string(event.elapsed_ms));
          write_log("[text] " + event.text);
        }
      };
      write_log("[model_loaded]");
      write_log("[audio_loaded] duration_ms=" +
                std::to_string(static_cast<int64_t>(audio.samples.size() * 1000 / 16000)));
      qwen3asr::FsmnVadConfig vad_config;
      vad_config.onnx_path = args.vad_onnx_path;
      vad_config.cmvn_path = args.vad_cmvn_path;
      vad_config.use_cuda = false;
      qwen3asr::FsmnVadOnnxModel vad_model(vad_config);
      qwen3asr::FsmnVadAsrPipeline pipeline(vad_model, model);
      const auto result = pipeline.Transcribe(
          audio.samples, args.context, language, args.hotwords, args.max_new_tokens, progress);
      std::cout << "language: " << result.language << "\n";
      std::cout << "duration_ms: " << result.duration_ms << "\n";
      std::cout << "vad_elapsed_ms: " << result.vad_elapsed_ms << "\n";
      std::cout << "asr_elapsed_ms: " << result.asr_elapsed_ms << "\n";
      std::cout << "total_elapsed_ms: " << result.total_elapsed_ms << "\n";
      std::cout << "vad_segments_ms:";
      for (const auto& item : result.vad_segments_ms) std::cout << " [" << item.first << "," << item.second << "]";
      std::cout << "\nsegments:\n";
      for (size_t i = 0; i < result.segments.size(); ++i) {
        const auto& segment = result.segments[i];
        std::cout << "  [" << i << "] " << segment.start_ms << "-" << segment.end_ms
                  << "ms vad=" << segment.vad_start_ms << "-" << segment.vad_end_ms
                  << "ms asr_elapsed_ms=" << segment.asr_elapsed_ms
                  << " language=" << segment.language << " text=" << segment.text << "\n";
      }
      std::cout << "text: " << result.text << "\n";
      return 0;
    }
    if (args.energy_vad) {
      auto audio = qwen3asr::LoadAudioMono(args.audio, 16000);
      qwen3asr::EnergyVadOptions vad_options;
      vad_options.threshold_db = args.vad_threshold_db;
      qwen3asr::EnergyVadAsrPipeline pipeline(model, vad_options);
      const auto result = pipeline.Transcribe(
          audio.samples, args.context, language, args.hotwords, args.max_new_tokens);
      std::cout << "language: " << result.language << "\n";
      std::cout << "duration_ms: " << result.duration_ms << "\n";
      std::cout << "vad_elapsed_ms: " << result.vad_elapsed_ms << "\n";
      std::cout << "asr_elapsed_ms: " << result.asr_elapsed_ms << "\n";
      std::cout << "total_elapsed_ms: " << result.total_elapsed_ms << "\n";
      std::cout << "vad_segments_ms:";
      for (const auto& item : result.vad_segments_ms) std::cout << " [" << item.first << "," << item.second << "]";
      std::cout << "\nsegments:\n";
      for (size_t i = 0; i < result.segments.size(); ++i) {
        const auto& segment = result.segments[i];
        std::cout << "  [" << i << "] " << segment.start_ms << "-" << segment.end_ms
                  << "ms vad=" << segment.vad_start_ms << "-" << segment.vad_end_ms
                  << "ms asr_elapsed_ms=" << segment.asr_elapsed_ms
                  << " language=" << segment.language << " text=" << segment.text << "\n";
      }
      std::cout << "text: " << result.text << "\n";
      return 0;
    }
    const auto result = model.TranscribeFile(args.audio, args.context, language, args.hotwords, args.max_new_tokens);

    std::cout << "language: " << result.language << "\n";
    std::cout << "chunk_count: " << result.chunk_count << "\n";
    std::cout << "mel_frames: " << result.mel_frames << "\n";
    std::cout << "audio_tokens: " << result.audio_tokens << "\n";
    std::cout << "elapsed_ms: " << result.elapsed_ms << "\n";
    std::cout << "raw_text: " << result.raw_text << "\n";
    std::cout << "text: " << result.text << "\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    PrintUsage(argv[0]);
    return 1;
  }
}
