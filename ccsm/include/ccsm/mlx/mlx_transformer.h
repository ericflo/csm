#ifndef CCSM_MLX_TRANSFORMER_H
#define CCSM_MLX_TRANSFORMER_H

#include <ccsm/tensor.h>
#include <vector>
#include <memory>
#include <string>

#ifdef CCSM_WITH_MLX
#include "mlx/c/array.h"
#include "mlx/c/ops.h"
#include "mlx/c/device.h"
#include "mlx/c/stream.h"
#endif

namespace ccsm {

class MLXTensorImpl;

// Forward declaration for KV Cache
class MLXKVCache;

// MLX-specific transformer layer implementation
class MLXTransformerLayer {
public:
#ifdef CCSM_WITH_MLX
    MLXTransformerLayer(
        const std::string& prefix,
        const std::unordered_map<std::string, mlx_array>& weights,
        int d_model,
        int n_heads,
        int n_kv_heads,
        float rope_theta = 10000.0f);
    
    ~MLXTransformerLayer();
    
    // Process input through the transformer layer
    mlx_array forward(
        mlx_array x,
        MLXKVCache* kv_cache,
        const std::vector<int>& positions,
        mlx_stream stream);
    
    // Getters for parameters
    int d_model() const { return d_model_; }
    int n_heads() const { return n_heads_; }
    int n_kv_heads() const { return n_kv_heads_; }
    int head_dim() const { return head_dim_; }
    
private:
    // Layer prefix for weight names
    std::string prefix_;
    
    // Model dimensions
    int d_model_;
    int n_heads_;
    int n_kv_heads_;
    int head_dim_;
    float rope_theta_;
    
    // Weights
    mlx_array attention_norm_weight_;
    mlx_array attention_norm_bias_;
    mlx_array wq_weight_;
    mlx_array wk_weight_;
    mlx_array wv_weight_;
    mlx_array wo_weight_;
    mlx_array ffn_norm_weight_;
    mlx_array ffn_norm_bias_;
    mlx_array w1_weight_;
    mlx_array w2_weight_;
    mlx_array w3_weight_;
    
    // Helper function for rotary position embeddings
    mlx_array apply_rotary_embeddings(
        mlx_array x, 
        const std::vector<int>& positions,
        mlx_stream stream);
    
    // Helper function for attention mechanism
    mlx_array attention(
        mlx_array q,
        mlx_array k,
        mlx_array v,
        const std::vector<int>& positions,
        MLXKVCache* kv_cache,
        mlx_stream stream);
    
    // Helper function for SwiGLU feed-forward network
    mlx_array feed_forward(
        mlx_array x,
        mlx_stream stream);
#else
    MLXTransformerLayer() = delete;
    ~MLXTransformerLayer() = default;
#endif
};

// KV Cache for transformer models using MLX
class MLXKVCache {
public:
#ifdef CCSM_WITH_MLX
    MLXKVCache(size_t n_layers, size_t n_heads, size_t head_dim, size_t max_seq_len);
    ~MLXKVCache();
    
    void clear();
    void resize(size_t seq_len);
    
    mlx_array k_cache(int layer) const;
    mlx_array v_cache(int layer) const;
    
    size_t size() const;
    size_t max_seq_len() const;
    size_t current_seq_len() const;
    
    // Update KV cache with new key and value tensors
    void update(int layer, mlx_array k, mlx_array v, const std::vector<int>& positions, mlx_stream stream);
    
private:
    size_t n_layers_;
    size_t n_heads_;
    size_t head_dim_;
    size_t max_seq_len_;
    size_t current_seq_len_;
    
    std::vector<mlx_array> k_caches_;
    std::vector<mlx_array> v_caches_;
#else
    MLXKVCache() = delete;
    ~MLXKVCache() = default;
#endif
};

// MLX-specific transformer model implementation
class MLXTransformer {
public:
#ifdef CCSM_WITH_MLX
    MLXTransformer(
        const std::unordered_map<std::string, mlx_array>& weights,
        int d_model,
        int n_layers,
        int n_heads,
        int n_kv_heads,
        int vocab_size,
        float rope_theta = 10000.0f);
    
    ~MLXTransformer();
    
    // Process input through the transformer model
    mlx_array forward(
        const std::vector<int>& tokens,
        const std::vector<int>& positions,
        MLXKVCache* kv_cache,
        mlx_stream stream);
    
    // Reset KV caches
    void reset_caches();
    
    // Getters for parameters
    int d_model() const { return d_model_; }
    int n_layers() const { return n_layers_; }
    int n_heads() const { return n_heads_; }
    int n_kv_heads() const { return n_kv_heads_; }
    int vocab_size() const { return vocab_size_; }
    
private:
    // Model dimensions
    int d_model_;
    int n_layers_;
    int n_heads_;
    int n_kv_heads_;
    int vocab_size_;
    float rope_theta_;
    
    // Transformer layers
    std::vector<std::unique_ptr<MLXTransformerLayer>> layers_;
    
    // Embedding and output weights
    mlx_array tok_embeddings_weight_;
    mlx_array norm_weight_;
    mlx_array norm_bias_;
    mlx_array output_weight_;
    
    // KV cache
    std::unique_ptr<MLXKVCache> kv_cache_;
#else
    MLXTransformer() = delete;
    ~MLXTransformer() = default;
#endif
};

// Helper functions for MLX transformer operations
#ifdef CCSM_WITH_MLX
// Create rotary position embeddings for RoPE
mlx_array mlx_rotary_embedding(
    mlx_array x,
    const std::vector<int>& positions,
    float theta,
    mlx_stream stream);

// Multi-head attention with caching
mlx_array mlx_attention(
    mlx_array query,
    mlx_array key,
    mlx_array value,
    const std::vector<int>& positions,
    float scale,
    mlx_stream stream);

// Feed-forward network with SwiGLU
mlx_array mlx_feed_forward(
    mlx_array x,
    mlx_array w1,
    mlx_array w2,
    mlx_array w3,
    mlx_stream stream);
#endif

} // namespace ccsm

#endif // CCSM_MLX_TRANSFORMER_H