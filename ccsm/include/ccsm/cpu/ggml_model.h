#ifndef CCSM_GGML_MODEL_H
#define CCSM_GGML_MODEL_H

#include <ccsm/model.h>
#include <ccsm/tensor.h>
#include <ccsm/cpu/ggml_tensor.h>
#include <unordered_map>
#include <memory>
#include <vector>
#include <random>

struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;

namespace ccsm {

// KV Cache for transformer models
class KVCache {
public:
    KVCache(size_t n_layers, size_t n_heads, size_t n_kv_heads, size_t head_dim, size_t max_seq_len);
    ~KVCache();
    
    void clear();
    void resize(size_t seq_len);
    
    struct ggml_tensor* k_cache(int layer);
    struct ggml_tensor* v_cache(int layer);
    
    size_t size() const;
    size_t max_seq_len() const;
    size_t current_seq_len() const;
    
private:
    size_t n_layers_;
    size_t n_heads_;
    size_t n_kv_heads_;
    size_t head_dim_;
    size_t max_seq_len_;
    size_t current_seq_len_;
    
    std::vector<struct ggml_tensor*> k_caches_;
    std::vector<struct ggml_tensor*> v_caches_;
    
    struct ggml_context* ctx_;
};

// CPU model implementation using GGML
class GGMLModel : public Model {
public:
    GGMLModel(const ModelConfig& config);
    ~GGMLModel();
    
    // Model interface implementation
    bool load_weights(const std::string& path) override;
    bool load_weights(std::shared_ptr<ModelLoader> loader) override;
    bool load_weights(const WeightMap& weights) override;
    
    std::vector<int> generate_frame(
        const std::vector<int>& tokens,
        const std::vector<int>& positions,
        float temperature = 0.9f,
        int top_k = 50) override;
    
    void reset_caches() override;
    
    std::vector<float> get_backbone_logits(
        const std::vector<int>& tokens,
        const std::vector<int>& positions) override;
    
    std::vector<float> get_decoder_logits(
        const std::vector<int>& tokens,
        const std::vector<int>& positions,
        int codebook) override;
    
private:
    // GGML context for model weights
    struct ggml_context* ctx_;
    
    // Model weights
    std::unordered_map<std::string, struct ggml_tensor*> weights_;
    
    // KV caches for transformer layers
    std::shared_ptr<KVCache> backbone_kv_cache_;
    std::shared_ptr<KVCache> decoder_kv_cache_;
    
    // Random number generator for sampling
    std::mt19937 rng_;
    
    // Helper methods
    struct ggml_cgraph* build_backbone_graph(
        struct ggml_context* ctx,
        const std::vector<int>& tokens,
        const std::vector<int>& positions);
    
    struct ggml_cgraph* build_decoder_graph(
        struct ggml_context* ctx,
        const std::vector<int>& tokens,
        const std::vector<int>& positions,
        int codebook);
    
    // Sampling methods
    int sample_token(const float* logits, int vocab_size, float temperature, int top_k);
    
    // Check if weights exist
    bool has_weight(const std::string& name) const;
    
    // Get a weight tensor by name
    struct ggml_tensor* get_weight(const std::string& name) const;
    
    // Create computation context (temporary context for graph building)
    struct ggml_context* create_computation_context(size_t mem_size);
    
    // Load backbone weights
    bool load_backbone_weights(const WeightMap& weights);
    
    // Load decoder weights
    bool load_decoder_weights(const WeightMap& weights);
};

} // namespace ccsm

#endif // CCSM_GGML_MODEL_H