#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <ccsm/version.h>
#include <ccsm/generator.h>
#include <ccsm/cli_args.h>
#include <ccsm/utils.h>
#include <ccsm/config.h>

using namespace ccsm;

int main(int argc, char** argv) {
    try {
        std::cout << "CCSM Generator (CPU) v" << CCSM_VERSION << std::endl;
        
        // Initialize configuration
        auto& config_manager = ConfigManager::instance();
        
        // Parse command-line arguments
        CLIArgs args = parse_args(argc, argv);
        
        // Handle special flags
        if (args.help) {
            print_help();
            return 0;
        }
        
        if (args.version) {
            print_version();
            return 0;
        }
        
        // Load configuration from file if specified
        if (args.backend_params.count("load-config") && !args.backend_params["load-config"].empty()) {
            std::string config_dir = args.backend_params["load-config"];
            if (std::filesystem::is_directory(config_dir)) {
                // Load from directory (model.json, generation.json, system.json)
                if (config_manager.load_from_directory(config_dir)) {
                    CCSM_INFO("Loaded configuration from directory: ", config_dir);
                } else {
                    CCSM_WARN("Failed to load configuration from directory: ", config_dir);
                }
            } else if (std::filesystem::exists(config_dir)) {
                // Load from single JSON file
                std::ifstream file(config_dir);
                if (file.is_open()) {
                    std::string json_content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
                    file.close();
                    
                    if (config_manager.load_from_json_string(json_content)) {
                        CCSM_INFO("Loaded configuration from file: ", config_dir);
                    } else {
                        CCSM_WARN("Failed to parse configuration file: ", config_dir);
                    }
                }
            } else {
                CCSM_WARN("Configuration file or directory not found: ", config_dir);
            }
        }
        
        // Load configurations from command line arguments (these override file settings)
        config_manager.load_from_args(argc, argv);
        
        // Update logging level based on debug setting
        auto& system_config = config_manager.system_config();
        if (system_config.get_debug() || args.debug) {
            Logger::instance().set_level(LogLevel::DEBUG);
            CCSM_DEBUG("Debug mode enabled");
        }
        
        // Validate required arguments
        if (args.text.empty()) {
            CCSM_ERROR("No text provided. Use --text or -t to specify text to generate speech for.");
            return 1;
        }
        
        // Save configurations if requested
        if (args.backend_params.count("save-config") && !args.backend_params["save-config"].empty()) {
            std::string config_dir = args.backend_params["save-config"];
            if (config_manager.save_to_directory(config_dir)) {
                CCSM_INFO("Saved configuration to directory: ", config_dir);
            } else {
                CCSM_ERROR("Failed to save configuration to directory: ", config_dir);
            }
        }
        
        // Initialize timer
        Timer timer;
        
        // Create generator
        auto& model_config = config_manager.model_config();
        auto& generation_config = config_manager.generation_config();
        
        std::string model_path = args.model_path.empty() ? model_config.get_model_path() : args.model_path;
        
        CCSM_INFO("Loading model from ", model_path);
        std::shared_ptr<Generator> generator;
        
        try {
            // Use the factory function to create generator
            if (model_path.empty()) {
                // Use default model
                generator = load_csm_1b();
            } else {
                // Use model loading mechanism with configuration
                generator = ModelLoaderFactory::load_model(model_path, model_config);
                CCSM_INFO("Loaded model from ", model_path);
            }
            
            CCSM_INFO("Model loaded in ", timer.elapsed_ms(), " ms");
        } catch (const std::exception& e) {
            CCSM_ERROR("Failed to load model: ", e.what());
            return 1;
        }
        
        // Prepare generation options
        GenerationOptions options;
        
        // Use configuration values but override with CLI args if specified
        options.temperature = args.temperature > 0 ? args.temperature : generation_config.get_temperature();
        options.top_k = args.top_k > 0 ? args.top_k : generation_config.get_top_k();
        options.top_p = args.top_p > 0 ? args.top_p : generation_config.get_top_p();
        options.repetition_penalty = args.repetition_penalty >= 1.0f ? 
                                  args.repetition_penalty : 
                                  generation_config.get_repetition_penalty();
        options.max_audio_length_ms = args.max_audio_length_ms > 0 ? 
                                     args.max_audio_length_ms : 
                                     generation_config.get_max_audio_length_ms();
        options.seed = args.seed >= 0 ? args.seed : generation_config.get_seed();
        options.enable_watermark = args.enable_watermark; // CLI flag takes precedence
        options.debug = args.debug || system_config.get_debug();
        
        // Log generation settings
        CCSM_DEBUG("Generation settings:");
        CCSM_DEBUG("  Temperature: ", options.temperature);
        CCSM_DEBUG("  Top-k: ", options.top_k);
        CCSM_DEBUG("  Top-p: ", options.top_p);
        CCSM_DEBUG("  Max audio length: ", options.max_audio_length_ms, " ms");
        CCSM_DEBUG("  Seed: ", options.seed);
        CCSM_DEBUG("  Watermark: ", options.enable_watermark ? "enabled" : "disabled");
        CCSM_DEBUG("  Repetition penalty: ", options.repetition_penalty);
        
        // Create progress bar
        int total_steps = args.max_audio_length_ms / 80; // 80ms per frame
        ProgressBar progress_bar(total_steps);
        
        // Prepare context segments
        std::vector<Segment> context;
        
        // Add context text segments if provided
        if (!args.context_text.empty()) {
            for (size_t i = 0; i < args.context_text.size(); ++i) {
                int speaker_id = 0;
                if (i < args.context_speaker.size()) {
                    speaker_id = args.context_speaker[i];
                }
                
                // Add the context segment with just text (no audio)
                context.emplace_back(args.context_text[i], speaker_id);
            }
        }
        
        // Add context audio segments if provided
        if (!args.context_audio.empty()) {
            for (size_t i = 0; i < args.context_audio.size(); ++i) {
                std::string context_text = "";
                if (i < args.context_text.size()) {
                    context_text = args.context_text[i];
                }
                
                int speaker_id = 0;
                if (i < args.context_speaker.size()) {
                    speaker_id = args.context_speaker[i];
                }
                
                // Load audio file
                int sample_rate = 0;
                std::vector<float> audio;
                
                try {
                    audio = FileUtils::load_wav(args.context_audio[i], &sample_rate);
                    CCSM_INFO("Loaded context audio: ", args.context_audio[i], 
                             " (", audio.size() / sample_rate, " seconds)");
                } catch (const std::exception& e) {
                    CCSM_ERROR("Failed to load context audio: ", e.what());
                    return 1;
                }
                
                // Add the context segment with audio
                context.emplace_back(context_text, speaker_id, audio);
            }
        }
        
        // Actual generation
        CCSM_INFO("Generating speech for: ", args.text.substr(0, 60), 
                 (args.text.length() > 60 ? "..." : ""));
        
        timer.reset();
        std::vector<float> audio;
        
        try {
            // Generate speech with progress callback
            audio = generator->generate_speech(
                args.text,
                args.speaker_id,
                context,
                options,
                [&progress_bar](int current, int total) {
                    progress_bar.update(current);
                }
            );
            
            progress_bar.finish();
        } catch (const std::exception& e) {
            CCSM_ERROR("Speech generation failed: ", e.what());
            return 1;
        }
        
        double generation_time = timer.elapsed_s();
        double audio_length = static_cast<double>(audio.size()) / generator->sample_rate();
        
        CCSM_INFO("Generated ", audio_length, " seconds of audio in ", 
                 generation_time, " seconds (", audio_length / generation_time, "x real-time)");
        
        // Save audio to output file
        CCSM_INFO("Saving audio to ", args.output_path);
        
        if (!FileUtils::save_wav(args.output_path, audio, generator->sample_rate())) {
            CCSM_ERROR("Failed to save audio file");
            return 1;
        }
        
        CCSM_INFO("Successfully saved audio to ", args.output_path);
        std::cout << "Done! Output saved to: " << args.output_path << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        CCSM_ERROR("Unhandled exception: ", e.what());
        return 1;
    }
}