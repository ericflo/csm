#include <ccsm/cli_args.h>
#include <ccsm/version.h>
#include <ccsm/utils.h>
#include <iostream>
#include <cstring>
#include <algorithm>

namespace ccsm {

void print_help() {
    std::cout << "CCSM Generator - C++ inference engine for Conversational Speech Model\n";
    std::cout << "Usage: ccsm-generate [options]\n\n";
    
    std::cout << "Model parameters:\n";
    std::cout << "  --model PATH       Path to model weights (default: auto-download CSM-1B)\n";
    std::cout << "  --text TEXT        Text to generate speech for (required)\n";
    std::cout << "  --speaker ID       Speaker ID (0-9, default: 0)\n";
    std::cout << "  --temperature VAL  Sampling temperature (default: 0.9)\n";
    std::cout << "  --topk VAL         Top-k sampling parameter (default: 50)\n";
    std::cout << "  --topp VAL         Top-p nucleus sampling parameter (default: 1.0)\n";
    std::cout << "  --rep-penalty VAL  Repetition penalty (default: 1.0, higher = less repetition)\n";
    std::cout << "  --seed VAL         Random seed (-1 for random, default: -1)\n";
    
    std::cout << "\nAudio parameters:\n";
    std::cout << "  --max-length MS    Maximum audio length in milliseconds (default: 10000)\n";
    std::cout << "  --output PATH      Output audio file path (default: audio.wav)\n";
    std::cout << "  --no-watermark     Disable audio watermarking\n";
    
    std::cout << "\nContext parameters:\n";
    std::cout << "  --context-text T   Text for context segment (can be used multiple times)\n";
    std::cout << "  --context-audio A  Audio file for context segment (can be used multiple times)\n";
    std::cout << "  --context-speaker S Speaker ID for context segment (can be used multiple times)\n";
    
    std::cout << "\nSystem parameters:\n";
    std::cout << "  --threads N        Number of threads to use (default: 4)\n";
    std::cout << "  --cpu-only         Force CPU-only mode\n";
    std::cout << "  --debug            Enable debug output\n";
    std::cout << "  --help             Show this help message\n";
    std::cout << "  --version          Show version information\n";
    
    std::cout << "\nConfiguration system:\n";
    std::cout << "  --backend-load-config=PATH    Load configuration from file or directory\n";
    std::cout << "  --backend-save-config=PATH    Save configuration to directory\n";
    std::cout << "  --backend-cache-dir=PATH      Set cache directory path\n";
    std::cout << "  --backend-models-dir=PATH     Set models directory path\n";
    
    std::cout << "\nExamples:\n";
    std::cout << "  ccsm-generate --text \"Hello, world!\"\n";
    std::cout << "  ccsm-generate --text \"This is a test.\" --speaker 3 --temperature 1.2\n";
    std::cout << "  ccsm-generate --text \"Follow-up response\" --context-text \"Initial query\" --context-speaker 0\n";
    std::cout << "  ccsm-generate --model model.gguf --backend-save-config=my_configs\n";
    std::cout << "  ccsm-generate --backend-load-config=my_configs --text \"Using saved config\"\n";
}

void print_version() {
    std::cout << "CCSM Generator v" << CCSM_VERSION << std::endl;
    std::cout << "C++ inference engine for Conversational Speech Model\n";
    std::cout << "Built with available backends: ";
    
    std::vector<std::string> backends = {"cpu"};
    
#ifdef CCSM_WITH_MLX
    backends.push_back("mlx");
#endif
    
#ifdef CCSM_WITH_CUDA
    backends.push_back("cuda");
#endif
    
#ifdef CCSM_WITH_VULKAN
    backends.push_back("vulkan");
#endif
    
    for (size_t i = 0; i < backends.size(); i++) {
        std::cout << backends[i];
        if (i < backends.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << std::endl;
}

CLIArgs parse_args(int argc, char** argv) {
    CLIArgs args;
    
    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        // Handle help and version first
        if (arg == "--help" || arg == "-h") {
            args.help = true;
            return args;
        }
        
        if (arg == "--version" || arg == "-v") {
            args.version = true;
            return args;
        }
        
        // Handle remaining arguments
        if (arg == "--model" || arg == "-m") {
            if (i + 1 < argc) {
                args.model_path = argv[++i];
            } else {
                CCSM_ERROR("Missing value for --model");
                args.help = true;
                return args;
            }
        }
        else if (arg == "--text" || arg == "-t") {
            if (i + 1 < argc) {
                args.text = argv[++i];
            } else {
                CCSM_ERROR("Missing value for --text");
                args.help = true;
                return args;
            }
        }
        else if (arg == "--speaker" || arg == "-s") {
            if (i + 1 < argc) {
                args.speaker_id = std::stoi(argv[++i]);
                if (args.speaker_id < 0 || args.speaker_id > 9) {
                    CCSM_ERROR("Speaker ID must be between 0 and 9");
                    args.help = true;
                    return args;
                }
            } else {
                CCSM_ERROR("Missing value for --speaker");
                args.help = true;
                return args;
            }
        }
        else if (arg == "--temperature") {
            if (i + 1 < argc) {
                args.temperature = std::stof(argv[++i]);
                if (args.temperature < 0.0f) {
                    CCSM_ERROR("Temperature must be non-negative");
                    args.help = true;
                    return args;
                }
            } else {
                CCSM_ERROR("Missing value for --temperature");
                args.help = true;
                return args;
            }
        }
        else if (arg == "--topk") {
            if (i + 1 < argc) {
                args.top_k = std::stoi(argv[++i]);
                if (args.top_k <= 0) {
                    CCSM_ERROR("Top-k must be positive");
                    args.help = true;
                    return args;
                }
            } else {
                CCSM_ERROR("Missing value for --topk");
                args.help = true;
                return args;
            }
        }
        else if (arg == "--topp") {
            if (i + 1 < argc) {
                args.top_p = std::stof(argv[++i]);
                if (args.top_p <= 0.0f || args.top_p > 1.0f) {
                    CCSM_ERROR("Top-p must be between 0.0 and 1.0");
                    args.help = true;
                    return args;
                }
            } else {
                CCSM_ERROR("Missing value for --topp");
                args.help = true;
                return args;
            }
        }
        else if (arg == "--rep-penalty") {
            if (i + 1 < argc) {
                args.repetition_penalty = std::stof(argv[++i]);
                if (args.repetition_penalty < 1.0f) {
                    CCSM_ERROR("Repetition penalty must be at least 1.0");
                    args.help = true;
                    return args;
                }
            } else {
                CCSM_ERROR("Missing value for --rep-penalty");
                args.help = true;
                return args;
            }
        }
        else if (arg == "--seed") {
            if (i + 1 < argc) {
                args.seed = std::stoi(argv[++i]);
            } else {
                CCSM_ERROR("Missing value for --seed");
                args.help = true;
                return args;
            }
        }
        else if (arg == "--max-length") {
            if (i + 1 < argc) {
                args.max_audio_length_ms = std::stoi(argv[++i]);
                if (args.max_audio_length_ms <= 0) {
                    CCSM_ERROR("Maximum audio length must be positive");
                    args.help = true;
                    return args;
                }
            } else {
                CCSM_ERROR("Missing value for --max-length");
                args.help = true;
                return args;
            }
        }
        else if (arg == "--output" || arg == "-o") {
            if (i + 1 < argc) {
                args.output_path = argv[++i];
            } else {
                CCSM_ERROR("Missing value for --output");
                args.help = true;
                return args;
            }
        }
        else if (arg == "--no-watermark") {
            args.enable_watermark = false;
        }
        else if (arg == "--context-text") {
            if (i + 1 < argc) {
                args.context_text.push_back(argv[++i]);
            } else {
                CCSM_ERROR("Missing value for --context-text");
                args.help = true;
                return args;
            }
        }
        else if (arg == "--context-audio") {
            if (i + 1 < argc) {
                args.context_audio.push_back(argv[++i]);
            } else {
                CCSM_ERROR("Missing value for --context-audio");
                args.help = true;
                return args;
            }
        }
        else if (arg == "--context-speaker") {
            if (i + 1 < argc) {
                int speaker_id = std::stoi(argv[++i]);
                if (speaker_id < 0 || speaker_id > 9) {
                    CCSM_ERROR("Context speaker ID must be between 0 and 9");
                    args.help = true;
                    return args;
                }
                args.context_speaker.push_back(speaker_id);
            } else {
                CCSM_ERROR("Missing value for --context-speaker");
                args.help = true;
                return args;
            }
        }
        else if (arg == "--threads") {
            if (i + 1 < argc) {
                args.num_threads = std::stoi(argv[++i]);
                if (args.num_threads <= 0) {
                    CCSM_ERROR("Number of threads must be positive");
                    args.help = true;
                    return args;
                }
            } else {
                CCSM_ERROR("Missing value for --threads");
                args.help = true;
                return args;
            }
        }
        else if (arg == "--cpu-only") {
            args.cpu_only = true;
        }
        else if (arg == "--debug" || arg == "-d") {
            args.debug = true;
        }
        else if (arg.substr(0, 9) == "--backend-") {
            // Parse backend parameters in format --backend-name=value
            size_t equals_pos = arg.find('=');
            if (equals_pos != std::string::npos) {
                std::string param_name = arg.substr(9, equals_pos - 9);
                std::string param_value = arg.substr(equals_pos + 1);
                
                // Validate specific known backend parameters
                if (param_name == "load-config") {
                    // Check if the specified path exists
                    if (!param_value.empty() && !std::filesystem::exists(param_value)) {
                        CCSM_WARN("Configuration path does not exist: ", param_value);
                        // Not a fatal error, will create the directory if saving
                    }
                } 
                else if (param_name == "save-config") {
                    // Check if the specified directory exists or can be created
                    if (!param_value.empty()) {
                        std::filesystem::path config_path(param_value);
                        if (std::filesystem::exists(param_value) && !std::filesystem::is_directory(param_value)) {
                            CCSM_ERROR("Config save path exists but is not a directory: ", param_value);
                            args.help = true;
                            return args;
                        }
                    }
                }
                else if (param_name == "cache-dir" || param_name == "models-dir") {
                    // Just validate they're not empty
                    if (param_value.empty()) {
                        CCSM_ERROR("Directory path for ", param_name, " cannot be empty");
                        args.help = true;
                        return args;
                    }
                }
                
                // Store the parameter
                args.backend_params[param_name] = param_value;
            } else {
                CCSM_ERROR("Invalid backend parameter format: ", arg);
                args.help = true;
                return args;
            }
        }
        else {
            CCSM_ERROR("Unknown argument: ", arg);
            args.help = true;
            return args;
        }
    }
    
    // Validate arguments
    if (!args.help && !args.version && args.text.empty()) {
        CCSM_ERROR("Text to generate speech for is required (--text)");
        args.help = true;
    }
    
    // Check context parameters consistency
    if (!args.context_text.empty() && args.context_speaker.size() != args.context_text.size()) {
        CCSM_ERROR("Number of context speakers must match number of context texts");
        args.help = true;
    }
    
    if (!args.context_audio.empty() && args.context_audio.size() != args.context_text.size()) {
        CCSM_ERROR("Number of context audio files must match number of context texts");
        args.help = true;
    }
    
    // Enable debug logging if requested
    if (args.debug) {
        Logger::instance().set_level(LogLevel::DEBUG);
    }
    
    return args;
}

} // namespace ccsm