#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <ccsm/version.h>
#include <ccsm/generator.h>
#include <ccsm/cli_args.h>
#include <ccsm/utils.h>

#ifndef CCSM_WITH_MLX
#error "This file requires MLX support to be enabled"
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
        
        // Initialize timer
        Timer timer;
        
        // Create generator with MLX acceleration
        CCSM_INFO("Loading model from ", args.model_path);
        std::shared_ptr<Generator> generator;
        
        try {
            // Use the MLX factory function to create generator
            if (args.model_path.empty()) {
                // Use default model
                generator = load_csm_1b_mlx();
            } else {
                // TODO: Implement custom model loading
                // For now, we'll just use the factory function
                generator = load_csm_1b_mlx();
                CCSM_WARNING("Custom model paths not fully implemented yet, using default model");
            }
            
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