#ifndef CCSM_CLI_ARGS_H
#define CCSM_CLI_ARGS_H

#include <string>
#include <vector>
#include <cstdint>
#include <unordered_map>

namespace ccsm {

// Structure to hold command-line arguments
struct CLIArgs {
    // Model parameters
    std::string model_path;                  // Path to model weights
    std::string text;                        // Text to generate speech for
    int speaker_id = 0;                      // Speaker ID (0-9)
    float temperature = 0.9f;                // Sampling temperature
    int top_k = 50;                          // Top-k sampling parameter
    int seed = -1;                           // Random seed (-1 for random)
    
    // Audio parameters
    int max_audio_length_ms = 10000;         // Maximum audio length in milliseconds
    std::string output_path = "audio.wav";   // Output audio file path
    bool enable_watermark = true;            // Whether to watermark the audio
    
    // Context parameters
    std::vector<std::string> context_text;   // Text for context segments
    std::vector<std::string> context_audio;  // Audio files for context segments
    std::vector<int> context_speaker;        // Speaker IDs for context segments
    
    // System parameters
    int num_threads = 4;                     // Number of threads to use
    bool cpu_only = false;                   // Force CPU-only mode
    bool debug = false;                      // Enable debug output
    bool help = false;                       // Show help message
    bool version = false;                    // Show version information
    
    // Backend-specific parameters
    std::unordered_map<std::string, std::string> backend_params;
};

// Function to parse command-line arguments
CLIArgs parse_args(int argc, char** argv);

// Function to display help message
void print_help();

// Function to display version information
void print_version();

} // namespace ccsm

#endif // CCSM_CLI_ARGS_H