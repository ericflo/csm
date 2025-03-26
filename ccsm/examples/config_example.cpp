#include <ccsm/config.h>
#include <ccsm/model_loader.h>
#include <ccsm/generator.h>
#include <ccsm/utils.h>
#include <iostream>
#include <filesystem>

using namespace ccsm;

/**
 * This example demonstrates how to use the configuration system programmatically
 * to customize model loading and text-to-speech generation.
 */
int main(int argc, char** argv) {
    try {
        // Access the configuration manager singleton
        auto& config_manager = ConfigManager::instance();
        
        // Option 1: Load configuration from a file
        if (std::filesystem::exists("config.json")) {
            std::ifstream file("config.json");
            std::string json_content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            file.close();
            
            if (config_manager.load_from_json_string(json_content)) {
                std::cout << "Loaded configuration from config.json" << std::endl;
            }
        }
        
        // Option 2: Set configuration options programmatically
        
        // Set model configuration
        auto& model_config = config_manager.model_config();
        model_config.set_model_path("path/to/model.gguf");  // Set model path
        model_config.set_architecture("ccsm");              // Set architecture
        model_config.set_embedding_dim(4096);               // Set embedding dimension
        model_config.set_num_layers(32);                    // Set number of layers
        
        // Set generation configuration
        auto& generation_config = config_manager.generation_config();
        generation_config.set_temperature(0.8f);            // Lower temperature for more predictable output
        generation_config.set_top_p(0.92f);                 // Enable nucleus sampling
        generation_config.set_top_k(40);                    // Limit to top 40 tokens
        generation_config.set_repetition_penalty(1.1f);     // Apply repetition penalty
        generation_config.set_max_audio_length_ms(20000);   // Allow up to 20 seconds of audio
        generation_config.set_seed(42);                     // Set fixed seed for reproducibility
        
        // Set system configuration
        auto& system_config = config_manager.system_config();
        system_config.set_num_threads(4);                   // Use 4 threads for processing
        system_config.set_debug(true);                      // Enable debug output
        system_config.set_cache_dir(".cache");              // Set local cache directory
        
        // Option 3: Save the configuration for future use
        std::string config_dir = "my_configs";
        if (config_manager.save_to_directory(config_dir)) {
            std::cout << "Saved configuration to directory: " << config_dir << std::endl;
        }
        
        // Load the model using the configuration
        std::cout << "Loading model with configured settings..." << std::endl;
        auto generator = ModelLoaderFactory::load_model("", model_config);
        
        if (!generator) {
            std::cerr << "Failed to load model" << std::endl;
            return 1;
        }
        
        // Prepare generation options from configuration
        GenerationOptions options;
        options.temperature = generation_config.get_temperature();
        options.top_k = generation_config.get_top_k();
        options.top_p = generation_config.get_top_p();
        options.repetition_penalty = generation_config.get_repetition_penalty();
        options.max_audio_length_ms = generation_config.get_max_audio_length_ms();
        options.seed = generation_config.get_seed();
        options.enable_watermark = generation_config.get_enable_watermark();
        options.debug = system_config.get_debug();
        
        // Generate audio
        std::cout << "Generating audio..." << std::endl;
        
        std::string text = "Hello, this is an example of speech generation with custom configuration.";
        int speaker_id = 0;
        std::vector<Segment> context;
        
        auto audio = generator->generate_speech(text, speaker_id, context, options,
            [](int current, int total) {
                // Simple progress bar
                int percent = (current * 100) / total;
                std::cout << "\rProgress: " << percent << "% [";
                for (int i = 0; i < 20; i++) {
                    if (i < (current * 20) / total) {
                        std::cout << "=";
                    } else {
                        std::cout << " ";
                    }
                }
                std::cout << "]" << std::flush;
            }
        );
        
        std::cout << "\nSaving audio to output.wav..." << std::endl;
        
        // Save the generated audio
        FileUtils::save_wav("output.wav", audio, generator->sample_rate());
        
        std::cout << "Audio saved to output.wav" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}