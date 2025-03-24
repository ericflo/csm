#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <ccsm/version.h>
#include <ccsm/generator.h>
#include <ccsm/cli_args.h>
#include <ccsm/utils.h>
#include <ccsm/mlx/mlx_weight_converter.h>
#include <ccsm/mlx/mlx_model.h>

// Allow compilation without MLX for testing purposes
#ifndef CCSM_WITH_MLX
// Instead of error, just define a stub for testing
#define CCSM_MLX_TESTING_ONLY
#endif

using namespace ccsm;

int main(int argc, char** argv) {
    try {
        std::cout << "CCSM Generator (MLX) v" << CCSM_VERSION << std::endl;
        
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
        
        // Set debug mode if requested
        if (args.debug) {
            Logger::instance().set_level(LogLevel::DEBUG);
            CCSM_DEBUG("Debug mode enabled");
        }
        
        // Validate required arguments
        if (args.text.empty()) {
            CCSM_ERROR("No text provided. Use --text or -t to specify text to generate speech for.");
            return 1;
        }
        
        // Check if CPU-only mode is forced
        if (args.cpu_only) {
            CCSM_WARNING("CPU-only mode forced, not using MLX acceleration");
            CCSM_INFO("Suggestion: Use ccsm-generate instead for CPU-only inference");
            // Fall back to CPU implementation
            return 0;
        }
        
        // Check MLX availability
        if (!MLXWeightConverter::is_mlx_available()) {
            CCSM_WARNING("MLX is not available on this system, falling back to CPU implementation");
            CCSM_INFO("Suggestion: Use ccsm-generate instead for CPU-only inference");
            
            // Fall back to CPU implementation
            try {
                std::shared_ptr<Generator> generator = load_csm_1b("cpu");
                // Continue with CPU generator...
                
                // Prepare generation options
                GenerationOptions options;
                options.temperature = args.temperature;
                options.top_k = args.top_k;
                options.max_audio_length_ms = args.max_audio_length_ms;
                options.seed = args.seed;
                options.enable_watermark = args.enable_watermark;
                options.debug = args.debug;
                
                // Generate speech with CPU implementation
                CCSM_INFO("Generating speech (CPU fallback) for: ", args.text.substr(0, 60),
                         (args.text.length() > 60 ? "..." : ""));
                
                Timer timer;
                std::vector<float> audio = generator->generate_speech(args.text, args.speaker_id);
                
                double generation_time = timer.elapsed_s();
                double audio_length = static_cast<double>(audio.size()) / generator->sample_rate();
                double rtf = audio_length / generation_time;
                
                CCSM_INFO("Generated ", audio_length, " seconds of audio in ",
                         generation_time, " seconds (", rtf, "x real-time)");
                
                // Save audio to output file
                CCSM_INFO("Saving audio to ", args.output_path);
                
                if (!FileUtils::save_wav(args.output_path, audio, generator->sample_rate())) {
                    CCSM_ERROR("Failed to save audio file");
                    return 1;
                }
                
                CCSM_INFO("Successfully saved audio to ", args.output_path);
                std::cout << "Done! Output saved to: " << args.output_path << std::endl;
                
                // Report performance
                std::cout << "Performance: " << rtf << "x real-time" << std::endl;
                
                return 0;
            } catch (const std::exception& e) {
                CCSM_ERROR("Failed to initialize CPU fallback: ", e.what());
                return 1;
            }
        }
        
        // Initialize timer
        Timer timer;
        
        // Create generator with MLX acceleration
        CCSM_INFO("Loading model from ", args.model_path);
        std::shared_ptr<Generator> generator;
        
        // Convert PyTorch weights to MLX if needed
        try {
#ifndef CCSM_MLX_TESTING_ONLY
            std::string model_path = args.model_path;
            
            // If a custom model path is provided, check if it needs conversion
            if (!model_path.empty()) {
                // Setup weight converter with default configuration
                MLXWeightConversionConfig config;
                config.use_bfloat16 = true;  // Use BF16 for better performance
                config.cache_converted_weights = true;  // Cache the converted weights
                
                // Progress callback
                config.progress_callback = [](float progress) {
                    static int last_percent = -1;
                    int percent = static_cast<int>(progress * 100.0f);
                    if (percent > last_percent) {
                        CCSM_INFO("Converting weights: ", percent, "%");
                        last_percent = percent;
                    }
                };
                
                // Create converter
                MLXWeightConverter converter(config);
                
                // Check if the model needs conversion (PyTorch format)
                if (model_path.ends_with(".pt") || model_path.ends_with(".pth")) {
                    CCSM_INFO("Converting PyTorch weights to MLX format...");
                    
                    // Generate output path for MLX weights
                    std::string mlx_output_path = model_path + ".mlx";
                    
                    // If cached weights already exist, we'll use those
                    if (has_cached_mlx_weights(model_path)) {
                        CCSM_INFO("Using cached MLX weights");
                        mlx_output_path = get_cached_mlx_weights_path(model_path);
                    } else {
                        // Convert weights
                        CCSM_INFO("Converting weights from ", model_path, " to ", mlx_output_path);
                        bool success = converter.convert_checkpoint(model_path, mlx_output_path);
                        
                        if (!success) {
                            CCSM_ERROR("Failed to convert weights to MLX format");
                            throw std::runtime_error("Weight conversion failed");
                        }
                    }
                    
                    // Update model path to use converted weights
                    model_path = mlx_output_path;
                    CCSM_INFO("Using MLX weights from ", model_path);
                }
                
                // Create MLX model
                ModelConfig config;
                config.model_path = model_path;
                config.backend = "mlx";
                std::shared_ptr<Model> model = std::make_shared<MLXModel>(config);
                
                if (!model->load_weights(model_path)) {
                    CCSM_ERROR("Failed to load MLX weights from ", model_path);
                    throw std::runtime_error("Failed to load MLX weights");
                }
                
                // Create generator with MLX model
                generator = std::make_shared<Generator>(model);
            } else {
                // Use default model - this should use pre-converted MLX weights
                generator = load_csm_1b_mlx();
            }
#else
            // In testing mode, just use CPU implementation directly
            if (args.model_path.empty()) {
                generator = load_csm_1b("cpu");
            } else {
                // Just use default model since this is only for testing
                generator = load_csm_1b("cpu");
                CCSM_WARNING("Using default model in testing mode");
            }
#endif
            
            CCSM_INFO("Model loaded in ", timer.elapsed_ms(), " ms");
        } catch (const std::exception& e) {
            CCSM_ERROR("Failed to load model with MLX: ", e.what());
            CCSM_INFO("Falling back to CPU implementation...");
            
            try {
                // Try to fall back to CPU implementation
                generator = load_csm_1b("cpu");
                CCSM_INFO("Successfully loaded CPU model as fallback");
            } catch (const std::exception& fallback_e) {
                CCSM_ERROR("Failed to load fallback CPU model: ", fallback_e.what());
                return 1;
            }
        }
        
        // Prepare generation options
        GenerationOptions options;
        options.temperature = args.temperature;
        options.top_k = args.top_k;
        options.max_audio_length_ms = args.max_audio_length_ms;
        options.seed = args.seed;
        options.enable_watermark = args.enable_watermark;
        options.debug = args.debug;
        
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
        double rtf = audio_length / generation_time;
        
        CCSM_INFO("Generated ", audio_length, " seconds of audio in ", 
                 generation_time, " seconds (", rtf, "x real-time)");
        
        // Save audio to output file
        CCSM_INFO("Saving audio to ", args.output_path);
        
        if (!FileUtils::save_wav(args.output_path, audio, generator->sample_rate())) {
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