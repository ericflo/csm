#ifndef CCSM_MLX_MODEL_H
#define CCSM_MLX_MODEL_H

#include <ccsm/model.h>
#include <ccsm/tensor.h>
#include <ccsm/mlx/mlx_tensor.h>
#include <ccsm/mlx/mlx_transformer.h>
#include <string>
#include <memory>
#include <unordered_map>
#include <vector>
#include <random>

namespace ccsm {

// Forward declarations
class MLXTransformer;

/**
 * MLX-specific model implementation
 * 
 * This class implements the Model interface using the MLX backend
 * for Apple Silicon acceleration. It uses the MLX transformer and
 * tensor implementations for efficient execution on Metal-capable
 * hardware.
 */
class MLXModel : public Model {
public:
    // Constructor
    MLXModel(const ModelConfig& config);
    
    // Destructor
    ~MLXModel() override;
    
    // Load model weights
    bool load_weights(const std::string& path) override;
    
    // Load model weights from a loader
    bool load_weights(std::shared_ptr<ModelLoader> loader) override;
    
    // Load model weights directly from a weight map
    bool load_weights(const WeightMap& weights) override;
    
    // Generate a complete audio frame (all codebooks)
    std::vector<int> generate_frame(
        const std::vector<int>& tokens,
        const std::vector<int>& positions,
        float temperature = 0.9f,
        int top_k = 50) override;
    
    // Reset KV caches
    void reset_caches() override;
    
    // Memory optimization methods
    void optimize_memory(size_t max_memory_mb = 0) override;
    void prune_caches(float prune_factor = 0.5f) override;
    
    // Get backbone model logits
    std::vector<float> get_backbone_logits(
        const std::vector<int>& tokens,
        const std::vector<int>& positions) override;
    
    // Get decoder model logits
    std::vector<float> get_decoder_logits(
        const std::vector<int>& tokens,
        const std::vector<int>& positions,
        int codebook) override;
    
    // MLX-specific methods
    
    // Get MLX array for a specific weight
    #ifdef CCSM_WITH_MLX
    mlx_array get_weight_array(const std::string& name) const;
    #endif
    
    // Check if a specific MLX weight exists
    bool has_weight(const std::string& name) const;
    
    // Create a new transformer context for inference
    std::shared_ptr<MLXTransformer> create_transformer_context() const;
    
private:
    // Private implementation
    struct Impl;
    std::unique_ptr<Impl> impl_;
    
    // Helper methods
    bool load_backbone_weights(const WeightMap& weights);
    bool load_decoder_weights(const WeightMap& weights);
    std::vector<int> sample_from_logits(const std::vector<float>& logits, float temperature, int top_k);
    
    // Random number generator for sampling
    std::mt19937 rng_;
};

} // namespace ccsm

#endif // CCSM_MLX_MODEL_H