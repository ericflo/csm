#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <ccsm/version.h>
#include <ccsm/generator.h>
#include <ccsm/cli_args.h>
#include <ccsm/utils.h>

using namespace ccsm;

int main(int argc, char** argv) {
    try {
        std::cout << "CCSM Generator (CPU) v" << CCSM_VERSION << std::endl;
        
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
        
        // Initialize timer
        Timer timer;
        
        // Create generator
        CCSM_INFO("Loading model from ", args.model_path);
        std::shared_ptr<Generator> generator;
        
        try {
            // Use the factory function to create generator
            if (args.model_path.empty()) {
                // Use default model
                generator = load_csm_1b();
            } else {
                // TODO: Implement custom model loading
                // For now, we'll just use the factory function
                generator = load_csm_1b();
                CCSM_WARNING("Custom model paths not fully implemented yet, using default model");
            }
            
            CCSM_INFO("Model loaded in ", timer.elapsed_ms(), " ms");
        } catch (const std::exception& e) {
            CCSM_ERROR("Failed to load model: ", e.what());
            return 1;
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