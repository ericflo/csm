#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cmath>

// Enable MLX support if available
// Note: This will use a stub implementation if MLX is not detected at build time

#include <ccsm/version.h>
#include <ccsm/generator.h>
#include <ccsm/cli_args.h>
#include <ccsm/utils.h>
#include <ccsm/tokenizer.h>

namespace ccsm {
    // Stub for MLX weight converter when MLX is not available
    struct MLXWeightConversionConfig {
        bool use_bfloat16 = true;
        bool cache_converted_weights = true;
        std::function<void(float)> progress_callback;
    };
    
    class MLXWeightConverter {
    public:
        MLXWeightConverter(const MLXWeightConversionConfig& config) {}
        
        static bool is_mlx_available() { 
            return false;  // Always return false for now 
        }
        
        bool convert_checkpoint(const std::string& input_path, const std::string& output_path) {
            return false;  // Always fail for now
        }
    };
}

// Helper function to check if a string ends with a specific suffix
bool ends_with(const std::string& str, const std::string& suffix) {
    if (suffix.size() > str.size()) return false;
    return std::equal(suffix.rbegin(), suffix.rend(), str.rbegin());
}

int main(int argc, char** argv) {
    try {
        std::cout << "CCSM Generator (MLX) v" << CCSM_VERSION << std::endl;
        
        // Parse command-line arguments
        ccsm::CLIArgs args = ccsm::parse_args(argc, argv);
        
        // Handle special flags
        if (args.help) {
            ccsm::print_help();
            return 0;
        }
        
        if (args.version) {
            ccsm::print_version();
            return 0;
        }
        
        // Set debug mode if requested
        if (args.debug) {
            ccsm::Logger::instance().set_level(ccsm::LogLevel::DEBUG);
            CCSM_DEBUG("Debug mode enabled");
        }
        
        // Validate required arguments
        if (args.text.empty()) {
            CCSM_ERROR("No text provided. Use --text or -t to specify text to generate speech for.");
            return 1;
        }
        
        // First try to load the MLX model if MLX is available
        std::shared_ptr<ccsm::Generator> generator;
        bool using_mlx = false;
        
        try {
            if (ccsm::MLXWeightConverter::is_mlx_available() && !args.model_path.empty()) {
                CCSM_INFO("Attempting to load MLX model from path: ", args.model_path);
                generator = ccsm::load_csm_1b_mlx();
                using_mlx = true;
                CCSM_INFO("Successfully loaded MLX model");
            } else if (ccsm::MLXWeightConverter::is_mlx_available()) {
                CCSM_INFO("Attempting to load default MLX model");
                generator = ccsm::load_csm_1b_mlx();
                using_mlx = true;
                CCSM_INFO("Successfully loaded default MLX model");
            } else {
                CCSM_WARNING("MLX is not available, falling back to CPU implementation");
                generator = ccsm::load_csm_1b("cpu");
                
                if (!args.model_path.empty()) {
                    CCSM_INFO("Note: Custom model path was specified (", args.model_path, 
                            ") but will be ignored in CPU fallback mode");
                }
            }
        } catch (const std::exception& e) {
            CCSM_WARNING("Failed to load MLX model: ", e.what(), ", falling back to CPU implementation");
            generator = ccsm::load_csm_1b("cpu");
            
            if (!args.model_path.empty()) {
                CCSM_INFO("Note: Custom model path was specified (", args.model_path, 
                        ") but will be ignored in CPU fallback mode");
            }
        }
        
        // Prepare generation options
        ccsm::GenerationOptions options;
        options.temperature = args.temperature;
        options.top_k = args.top_k;
        options.max_audio_length_ms = args.max_audio_length_ms;
        options.seed = args.seed;
        options.enable_watermark = args.enable_watermark;
        options.debug = args.debug;
        
        // Create progress bar
        int total_steps = args.max_audio_length_ms / 80; // 80ms per frame
        ccsm::ProgressBar progress_bar(total_steps);
        
        // Prepare context segments
        std::vector<ccsm::Segment> context;
        
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
                    audio = ccsm::FileUtils::load_wav(args.context_audio[i], &sample_rate);
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
        
        // Generate speech
        CCSM_INFO("Generating speech (", (using_mlx ? "MLX" : "CPU fallback"), ") for: ", 
                 args.text.substr(0, 60), (args.text.length() > 60 ? "..." : ""));
        
        ccsm::Timer timer;
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
        double rtf = audio_length / generation_time;
        
        CCSM_INFO("Generated ", audio_length, " seconds of audio in ",
                 generation_time, " seconds (", rtf, "x real-time)");
        
        // Save audio to output file
        CCSM_INFO("Saving audio to ", args.output_path);
        
        if (!ccsm::FileUtils::save_wav(args.output_path, audio, generator->sample_rate())) {
            CCSM_ERROR("Failed to save audio file");
            return 1;
        }
        
        CCSM_INFO("Successfully saved audio to ", args.output_path);
        std::cout << "Done! Output saved to: " << args.output_path << std::endl;
        
        // Report performance
        std::cout << "Performance: " << rtf << "x real-time" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        CCSM_ERROR("Unhandled exception: ", e.what());
        return 1;
    }
}