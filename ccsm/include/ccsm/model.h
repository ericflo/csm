#ifndef CCSM_MODEL_H
#define CCSM_MODEL_H

#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace ccsm {

// Forward declarations
class Context;
class ModelLoader;
using WeightMap = std::unordered_map<std::string, class Tensor>;

// Model configuration
struct ModelConfig {
    int vocab_size = 32000;    // Vocabulary size for text tokens
    int audio_vocab_size = 2051; // Vocabulary size for audio tokens
    int d_model = 4096;        // Model dimension
    int n_heads = 32;          // Number of attention heads
    int n_kv_heads = 8;        // Number of key-value heads (for GQA)
    int n_layers = 32;         // Number of layers in backbone
    int n_audio_layers = 12;   // Number of layers in audio decoder
    float rope_theta = 10000.0f; // RoPE base frequency
    int max_seq_len = 2048;    // Maximum sequence length
    int num_codebooks = 32;    // Number of audio codebooks (including semantic)
    std::string name = "csm";  // Model name
};

// Model interface
class Model {
public:
    // Constructor
    Model(const ModelConfig& config);
    
    // Virtual destructor
    virtual ~Model() = default;
    
    // Load model weights
    virtual bool load_weights(const std::string& path) = 0;
    
    // Load model weights from a loader
    virtual bool load_weights(std::shared_ptr<ModelLoader> loader) = 0;
    
    // Load model weights directly from a weight map
    virtual bool load_weights(const WeightMap& weights) = 0;
    
    // Generate a complete audio frame (all codebooks)
    virtual std::vector<int> generate_frame(
        const std::vector<int>& tokens,
        const std::vector<int>& positions,
        float temperature = 0.9f,
        int top_k = 50) = 0;
    
    // Reset KV caches
    virtual void reset_caches() = 0;
    
    // Memory optimization methods
    virtual void optimize_memory(size_t max_memory_mb = 0) = 0;
    virtual void prune_caches(float prune_factor = 0.5f) = 0;
    
    // Get backbone model logits
    virtual std::vector<float> get_backbone_logits(
        const std::vector<int>& tokens,
        const std::vector<int>& positions) = 0;
    
    // Get decoder model logits
    virtual std::vector<float> get_decoder_logits(
        const std::vector<int>& tokens,
        const std::vector<int>& positions,
        int codebook) = 0;
    
    // Get the model configuration
    const ModelConfig& config() const;
    
protected:
    ModelConfig config_;
};

// Factory for creating models with different backends
class ModelFactory {
public:
    // Create a model with a specific backend
    static std::shared_ptr<Model> create(
        const std::string& backend, 
        const ModelConfig& config = ModelConfig());
    
    // Check if a backend is available
    static bool is_backend_available(const std::string& backend);
    
    // Get all available backends
    static std::vector<std::string> get_available_backends();
};

} // namespace ccsm

#endif // CCSM_MODEL_H