#ifndef CCSM_MLX_GENERATOR_H
#define CCSM_MLX_GENERATOR_H

#include <ccsm/generator.h>
#include <ccsm/mlx/mlx_model.h>
#include <ccsm/mlx/mlx_optimizations.h>
#include <memory>
#include <string>
#include <vector>

namespace ccsm {

/**
 * MLX-specific Generator implementation for Apple Silicon
 * 
 * This class extends the basic Generator with MLX-specific optimizations
 * for Apple Silicon devices. It leverages the MLXModel and optimizations
 * to provide faster generation on compatible hardware.
 */
class MLXGenerator : public Generator {
public:
    // Constructor with MLX model and components
    MLXGenerator(
        std::shared_ptr<Model> model,
        std::shared_ptr<TextTokenizer> text_tokenizer,
        std::shared_ptr<AudioCodec> audio_codec,
        std::shared_ptr<Watermarker> watermarker = nullptr,
        const MLXOptimizationConfig& optimization_config = MLXOptimizationConfig());
    
    // Destructor
    ~MLXGenerator() override;
    
    // Override generate_speech to use MLX optimizations
    std::vector<float> generate_speech_from_tokens(
        const std::vector<int>& tokens,
        int speaker_id = -1,
        const std::vector<Segment>& context = {},
        const GenerationOptions& options = {},
        std::function<void(int, int)> progress_callback = nullptr) override;
    
    // MLX-specific methods
    
    // Set optimization config for MLX operations
    void set_optimization_config(const MLXOptimizationConfig& config);
    
    // Get current optimization config
    const MLXOptimizationConfig& optimization_config() const;
    
    // Return whether MLX acceleration is available and being used
    bool is_mlx_accelerated() const;
    
    // Check if the model is compatible with MLX acceleration
    static bool is_model_mlx_compatible(const std::shared_ptr<Model>& model);
    
private:
    // MLXModel reference for direct access to MLX-specific methods
    std::shared_ptr<MLXModel> mlx_model_;
    
    // Optimization configuration
    MLXOptimizationConfig optimization_config_;
    
    // Flag indicating if MLX acceleration is available
    bool mlx_available_;
    
    // Private helper methods
    
    // Optimize for MLX-specific memory management
    void optimize_for_generation();
    
    // Generate a single audio frame with MLX optimizations
    std::vector<int> generate_frame_mlx(
        const std::vector<int>& tokens,
        const std::vector<int>& positions,
        float temperature,
        int top_k);
    
    // Process generation results for audio decoding
    std::vector<float> process_audio_frames(
        const std::vector<std::vector<int>>& frames,
        const GenerationOptions& options);
};

// Factory function to create an MLX generator
std::shared_ptr<Generator> create_mlx_generator(
    const std::string& model_path,
    const std::string& tokenizer_path,
    const std::string& audio_codec_path,
    const std::string& watermarker_path = "",
    const MLXOptimizationConfig& optimization_config = MLXOptimizationConfig());

// Check if MLX acceleration is available on this system
bool is_mlx_available();

} // namespace ccsm

#endif // CCSM_MLX_GENERATOR_H