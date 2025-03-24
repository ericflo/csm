#include <ccsm/mlx/mlx_transformer.h>
#include <ccsm/mlx/mlx_tensor.h>
#include <ccsm/utils.h>
#include <stdexcept>
#include <iostream>
#include <cmath>

#ifdef CCSM_WITH_MLX
// For real implementation, use these headers
// #include "mlx/c/array.h"
// #include "mlx/c/ops.h"
// #include "mlx/c/device.h"
// #include "mlx/c/stream.h"
#endif

namespace ccsm {

#ifdef CCSM_WITH_MLX

// MLX transformer helper functions
mlx_array mlx_rotary_embedding(
    mlx_array x,
    const std::vector<int>& positions,
    float theta,
    mlx_stream stream) {
    
    CCSM_DEBUG("Applying rotary position embeddings");
    
    // Get dimensions of input
    uint32_t ndim;
    mlx_array_ndim(x, &ndim);
    if (ndim < 3) {
        CCSM_WARNING("Input to rotary embeddings has fewer than 3 dimensions, returning unchanged");
        return x;
    }
    
    const int* shape = mlx_array_shape(x);
    int batch_size = shape[0];
    int seq_len = shape[1];
    int feat_dim = shape[2];
    int half_dim = feat_dim / 2;
    
    if (positions.empty() || positions.size() != seq_len) {
        CCSM_WARNING("Invalid positions array for rotary embeddings");
        return x;
    }
    
    // Create freq_cis array for the embedding frequencies
    std::vector<float> freqs;
    freqs.resize(half_dim);
    for (int i = 0; i < half_dim; i++) {
        freqs[i] = 1.0f / powf(theta, (2.0f * i) / feat_dim);
    }
    
    // Create arrays for sin and cos values
    std::vector<float> sin_vals;
    std::vector<float> cos_vals;
    sin_vals.reserve(seq_len * half_dim);
    cos_vals.reserve(seq_len * half_dim);
    
    // Compute sin and cos values for each position
    for (int pos : positions) {
        for (int i = 0; i < half_dim; i++) {
            float freq = freqs[i];
            sin_vals.push_back(sinf(pos * freq));
            cos_vals.push_back(cosf(pos * freq));
        }
    }
    
    // Create MLX arrays for sin and cos
    int sin_shape[] = {seq_len, half_dim};
    mlx_array sin_array = mlx_array_new_data(
        sin_vals.data(),
        sin_shape,
        2,
        MLX_FLOAT32
    );
    
    mlx_array cos_array = mlx_array_new_data(
        cos_vals.data(),
        sin_shape,
        2,
        MLX_FLOAT32
    );
    
    // Reshape x to split the last dimension for rotation
    // Original x shape: [batch_size, seq_len, feat_dim]
    int reshaped_x_shape[] = {batch_size, seq_len, 2, half_dim};
    mlx_array reshaped_x;
    mlx_array_reshape(x, reshaped_x_shape, 4, &reshaped_x);
    
    // Apply rotary embedding: rotate the vectors by complex multiplication
    // x1, x2 = x[..., ::2], x[..., 1::2]
    // x_out[..., ::2] = x1 * cos - x2 * sin
    // x_out[..., 1::2] = x1 * sin + x2 * cos
    
    // Extract even and odd indices from reshaped_x
    int x1_indices[] = {0};
    mlx_array x1;
    mlx_array_index(reshaped_x, 2, x1_indices, 1, &x1);
    
    int x2_indices[] = {1};
    mlx_array x2;
    mlx_array_index(reshaped_x, 2, x2_indices, 1, &x2);
    
    // Compute rotated components
    mlx_array x1_cos, x2_sin, x1_sin, x2_cos;
    mlx_multiply(x1, cos_array, &x1_cos);
    mlx_multiply(x2, sin_array, &x2_sin);
    mlx_multiply(x1, sin_array, &x1_sin);
    mlx_multiply(x2, cos_array, &x2_cos);
    
    // x_out_even = x1 * cos - x2 * sin
    mlx_array x_out_even;
    mlx_subtract(x1_cos, x2_sin, &x_out_even);
    
    // x_out_odd = x1 * sin + x2 * cos
    mlx_array x_out_odd;
    mlx_add(x1_sin, x2_cos, &x_out_odd);
    
    // Concatenate the results back together
    mlx_array arrays[] = {x_out_even, x_out_odd};
    mlx_array x_out;
    mlx_concatenate(arrays, 2, 2, &x_out);
    
    // Reshape back to original shape
    mlx_array result;
    int orig_shape[] = {batch_size, seq_len, feat_dim};
    mlx_array_reshape(x_out, orig_shape, 3, &result);
    
    // Clean up intermediate arrays
    mlx_array_free(sin_array);
    mlx_array_free(cos_array);
    mlx_array_free(reshaped_x);
    mlx_array_free(x1);
    mlx_array_free(x2);
    mlx_array_free(x1_cos);
    mlx_array_free(x2_sin);
    mlx_array_free(x1_sin);
    mlx_array_free(x2_cos);
    mlx_array_free(x_out_even);
    mlx_array_free(x_out_odd);
    mlx_array_free(x_out);
    
    return result;
}

mlx_array mlx_attention(
    mlx_array query,
    mlx_array key,
    mlx_array value,
    const std::vector<int>& positions,
    float scale,
    mlx_stream stream) {
    
    CCSM_DEBUG("Performing multi-head attention");
    
    // Get dimensions of inputs
    uint32_t q_ndim, k_ndim, v_ndim;
    mlx_array_ndim(query, &q_ndim);
    mlx_array_ndim(key, &k_ndim);
    mlx_array_ndim(value, &v_ndim);
    
    if (q_ndim != k_ndim || k_ndim != v_ndim || q_ndim < 3) {
        CCSM_WARNING("Invalid dimensions for attention inputs");
        return query;
    }
    
    const int* q_shape = mlx_array_shape(query);
    const int* k_shape = mlx_array_shape(key);
    const int* v_shape = mlx_array_shape(value);
    
    int batch_size = q_shape[0];
    int q_seq_len = q_shape[1];
    int k_seq_len = k_shape[1];
    int head_dim = q_shape[2];
    
    // Compute attention scores: scale * Q @ K.T
    // First transpose K to [batch, head_dim, k_seq_len]
    int k_transpose_dims[] = {0, 2, 1};
    mlx_array k_transposed;
    mlx_array_transpose(key, k_transpose_dims, 3, &k_transposed);
    
    // Then compute Q @ K.T
    mlx_array scores;
    mlx_matmul(query, k_transposed, &scores);
    
    // Scale the scores
    mlx_array scaled_scores;
    mlx_array_float32 scale_array = mlx_array_new_float32(scale);
    mlx_multiply(scores, scale_array, &scaled_scores);
    mlx_array_free(scale_array);
    
    // Apply causal mask if needed (for autoregressive generation)
    if (!positions.empty()) {
        // Create causal mask: positions[i] >= positions[j]
        std::vector<float> mask_data(q_seq_len * k_seq_len, 0.0f);
        for (int i = 0; i < q_seq_len; i++) {
            for (int j = 0; j < k_seq_len; j++) {
                // Set mask to 1 if position i can attend to position j
                mask_data[i * k_seq_len + j] = (positions[i] >= positions[j]) ? 0.0f : -1e9f;
            }
        }
        
        int mask_shape[] = {1, q_seq_len, k_seq_len}; // [1, q_seq_len, k_seq_len]
        mlx_array mask = mlx_array_new_data(
            mask_data.data(),
            mask_shape,
            3,
            MLX_FLOAT32
        );
        
        // Add the mask to scaled_scores
        mlx_array masked_scores;
        mlx_add(scaled_scores, mask, &masked_scores);
        mlx_array_free(scaled_scores);
        mlx_array_free(mask);
        scaled_scores = masked_scores;
    }
    
    // Apply softmax along the last dimension
    mlx_array attention_weights;
    mlx_softmax(scaled_scores, 2, &attention_weights);
    
    // Compute weighted sum: attn_weights @ V
    mlx_array output;
    mlx_matmul(attention_weights, value, &output);
    
    // Clean up intermediate arrays
    mlx_array_free(k_transposed);
    mlx_array_free(scores);
    mlx_array_free(scaled_scores);
    mlx_array_free(attention_weights);
    
    return output;
}

mlx_array mlx_feed_forward(
    mlx_array x,
    mlx_array w1,
    mlx_array w2,
    mlx_array w3,
    mlx_stream stream) {
    
    CCSM_DEBUG("Executing SwiGLU feed-forward network");
    
    // Implementation of SwiGLU feed-forward network:
    // FFN(x) = (x @ w1 * SiLU(x @ w3)) @ w2
    
    // First linear transformation: x @ w1
    mlx_array x_w1;
    mlx_matmul(x, w1, &x_w1);
    
    // Second linear transformation: x @ w3
    mlx_array x_w3;
    mlx_matmul(x, w3, &x_w3);
    
    // Apply SiLU activation to x_w3
    mlx_array silu_x_w3;
    mlx_silu(x_w3, &silu_x_w3);
    mlx_array_free(x_w3);
    
    // Element-wise multiplication: x_w1 * silu_x_w3
    mlx_array gate_output;
    mlx_multiply(x_w1, silu_x_w3, &gate_output);
    mlx_array_free(x_w1);
    mlx_array_free(silu_x_w3);
    
    // Final linear transformation: gate_output @ w2
    mlx_array output;
    mlx_matmul(gate_output, w2, &output);
    mlx_array_free(gate_output);
    
    return output;
}

// MLXKVCache implementation
MLXKVCache::MLXKVCache(size_t n_layers, size_t n_heads, size_t head_dim, size_t max_seq_len)
    : n_layers_(n_layers),
      n_heads_(n_heads),
      head_dim_(head_dim),
      max_seq_len_(max_seq_len),
      current_seq_len_(0) {
    
    CCSM_DEBUG("Initializing MLXKVCache with capacity for ", n_layers, " layers");
    
    // Initialize empty caches for each layer
    k_caches_.resize(n_layers);
    v_caches_.resize(n_layers);
    
    // We'll lazily initialize the actual arrays when they're first used
}

MLXKVCache::~MLXKVCache() {
    CCSM_DEBUG("MLXKVCache destructor called");
    
    // Free up any allocated arrays in the caches
    for (auto& k_cache : k_caches_) {
        if (k_cache.ctx) {
            mlx_array_free(k_cache);
        }
    }
    
    for (auto& v_cache : v_caches_) {
        if (v_cache.ctx) {
            mlx_array_free(v_cache);
        }
    }
}

void MLXKVCache::clear() {
    CCSM_DEBUG("MLXKVCache::clear called");
    
    // Reset the current sequence length
    current_seq_len_ = 0;
    
    // For each allocated cache, reset to zeros
    for (size_t layer = 0; layer < n_layers_; ++layer) {
        if (k_caches_[layer].ctx) {
            // Free existing arrays
            mlx_array_free(k_caches_[layer]);
            mlx_array_free(v_caches_[layer]);
            
            // Set to null
            k_caches_[layer] = mlx_array{};
            v_caches_[layer] = mlx_array{};
        }
    }
}

void MLXKVCache::resize(size_t seq_len) {
    CCSM_DEBUG("MLXKVCache::resize called with seq_len=", seq_len);
    
    if (seq_len > max_seq_len_) {
        throw std::runtime_error("Cannot resize KV cache beyond maximum sequence length");
    }
    
    if (seq_len == current_seq_len_) {
        // No resize needed
        return;
    }
    
    // Resize all allocated caches
    for (size_t layer = 0; layer < n_layers_; ++layer) {
        if (k_caches_[layer].ctx) {
            if (seq_len < current_seq_len_) {
                // Shrinking - create new smaller arrays and copy
                int new_shape[] = {static_cast<int>(seq_len), static_cast<int>(n_heads_), static_cast<int>(head_dim_)};
                
                // Create new arrays
                mlx_array new_k, new_v;
                mlx_array_zeros(new_shape, 3, MLX_FLOAT32, &new_k);
                mlx_array_zeros(new_shape, 3, MLX_FLOAT32, &new_v);
                
                // Copy data from old to new
                for (size_t pos = 0; pos < seq_len; ++pos) {
                    int idx[] = {static_cast<int>(pos)};
                    
                    mlx_array k_item, v_item;
                    mlx_array_index(k_caches_[layer], 0, idx, 1, &k_item);
                    mlx_array_index(v_caches_[layer], 0, idx, 1, &v_item);
                    
                    mlx_array_update(new_k, k_item, idx, 1);
                    mlx_array_update(new_v, v_item, idx, 1);
                    
                    mlx_array_free(k_item);
                    mlx_array_free(v_item);
                }
                
                // Free old arrays
                mlx_array_free(k_caches_[layer]);
                mlx_array_free(v_caches_[layer]);
                
                // Assign new arrays
                k_caches_[layer] = new_k;
                v_caches_[layer] = new_v;
            }
            // If expanding, we don't need to do anything since arrays are already allocated to max_seq_len_
        }
    }
    
    current_seq_len_ = seq_len;
}

mlx_array MLXKVCache::k_cache(int layer) const {
    CCSM_DEBUG("MLXKVCache::k_cache called for layer=", layer);
    if (layer < 0 || layer >= (int)n_layers_) {
        throw std::out_of_range("Layer index out of range in KV cache");
    }
    return k_caches_[layer];
}

mlx_array MLXKVCache::v_cache(int layer) const {
    CCSM_DEBUG("MLXKVCache::v_cache called for layer=", layer);
    if (layer < 0 || layer >= (int)n_layers_) {
        throw std::out_of_range("Layer index out of range in KV cache");
    }
    return v_caches_[layer];
}

size_t MLXKVCache::size() const {
    return n_layers_;
}

size_t MLXKVCache::max_seq_len() const {
    return max_seq_len_;
}

size_t MLXKVCache::current_seq_len() const {
    return current_seq_len_;
}

void MLXKVCache::update(int layer, mlx_array k, mlx_array v, const std::vector<int>& positions, mlx_stream stream) {
    CCSM_DEBUG("MLXKVCache::update called for layer=", layer);
    if (layer < 0 || layer >= (int)n_layers_) {
        throw std::out_of_range("Layer index out of range in KV cache");
    }
    
    // Update current sequence length if needed
    int max_pos = 0;
    for (int pos : positions) {
        max_pos = std::max(max_pos, pos);
    }
    size_t new_seq_len = std::max(current_seq_len_, (size_t)(max_pos + 1));
    
    // Check if we need to initialize or resize the cache
    if (k_caches_[layer].ctx == nullptr) {
        // First-time initialization for this layer
        // Shape: [max_seq_len, n_heads_, head_dim_]
        int k_shape[] = {static_cast<int>(max_seq_len_), static_cast<int>(n_heads_), static_cast<int>(head_dim_)};
        mlx_array_zeros(k_shape, 3, MLX_FLOAT32, &k_caches_[layer]);
        mlx_array_zeros(k_shape, 3, MLX_FLOAT32, &v_caches_[layer]);
    }
    
    // Get dimension information from the input key
    uint32_t k_ndim;
    mlx_array_ndim(k, &k_ndim);
    const int* k_shape = mlx_array_shape(k);
    
    // Iterate through positions and update the cache at each position
    for (size_t i = 0; i < positions.size(); i++) {
        int pos = positions[i];
        if (pos >= (int)max_seq_len_) {
            CCSM_WARNING("Position ", pos, " exceeds max sequence length ", max_seq_len_);
            continue;
        }
        
        // Indices for the current position in cache and current sequence item
        int cache_idx[] = {pos};
        int seq_idx[] = {static_cast<int>(i)};
        
        // Extract this sequence item from k and v
        mlx_array k_item, v_item;
        mlx_array_index(k, 1, seq_idx, 1, &k_item);
        mlx_array_index(v, 1, seq_idx, 1, &v_item);
        
        // Update cache at this position
        mlx_array_update(k_caches_[layer], k_item, cache_idx, 1);
        mlx_array_update(v_caches_[layer], v_item, cache_idx, 1);
        
        // Clean up
        mlx_array_free(k_item);
        mlx_array_free(v_item);
    }
    
    current_seq_len_ = new_seq_len;
}

bool MLXKVCache::prune(size_t target_len, mlx_stream stream) {
    CCSM_DEBUG("MLXKVCache::prune called with target_len=", target_len);
    
    if (target_len >= current_seq_len_) {
        CCSM_DEBUG("Target length >= current length, no pruning needed");
        return true;
    }
    
    if (target_len == 0) {
        // Special case: clear the cache
        clear();
        return true;
    }
    
    try {
        // Create new arrays for each layer with the target length
        for (size_t layer = 0; layer < n_layers_; ++layer) {
            if (k_caches_[layer].ctx == nullptr || v_caches_[layer].ctx == nullptr) {
                // Skip uninitialized layers
                continue;
            }
            
            // Create new arrays with target length
            int new_shape[] = {static_cast<int>(target_len), static_cast<int>(n_heads_), static_cast<int>(head_dim_)};
            mlx_array new_k_cache, new_v_cache;
            mlx_array_zeros(new_shape, 3, MLX_FLOAT32, &new_k_cache);
            mlx_array_zeros(new_shape, 3, MLX_FLOAT32, &new_v_cache);
            
            // Calculate offset to keep most recent entries
            size_t offset = current_seq_len_ - target_len;
            
            // Copy the most recent entries to the new arrays
            for (size_t i = 0; i < target_len; ++i) {
                // Source index in old cache (keeping most recent entries)
                int src_idx[] = {static_cast<int>(offset + i)};
                // Destination index in new cache
                int dst_idx[] = {static_cast<int>(i)};
                
                // Extract items from old cache
                mlx_array k_item, v_item;
                mlx_array_index(k_caches_[layer], 0, src_idx, 1, &k_item);
                mlx_array_index(v_caches_[layer], 0, src_idx, 1, &v_item);
                
                // Update new cache
                mlx_array_update(new_k_cache, k_item, dst_idx, 1);
                mlx_array_update(new_v_cache, v_item, dst_idx, 1);
                
                // Clean up
                mlx_array_free(k_item);
                mlx_array_free(v_item);
            }
            
            // Free old arrays and replace with new ones
            mlx_array_free(k_caches_[layer]);
            mlx_array_free(v_caches_[layer]);
            k_caches_[layer] = new_k_cache;
            v_caches_[layer] = new_v_cache;
        }
        
        // Update current sequence length
        current_seq_len_ = target_len;
        return true;
    } catch (const std::exception& e) {
        CCSM_ERROR("Error during KV cache pruning: ", e.what());
        return false;
    }
}

// MLXTransformerLayer implementation
MLXTransformerLayer::MLXTransformerLayer(
    const std::string& prefix,
    const std::unordered_map<std::string, mlx_array>& weights,
    int d_model,
    int n_heads,
    int n_kv_heads,
    float rope_theta)
    : prefix_(prefix),
      d_model_(d_model),
      n_heads_(n_heads),
      n_kv_heads_(n_kv_heads),
      head_dim_(d_model / n_heads),
      rope_theta_(rope_theta) {
    
    CCSM_DEBUG("Initializing MLXTransformerLayer with prefix=", prefix);
    
    // Load layer weights
    auto load_weight = [&](const std::string& name, mlx_array& dst) {
        std::string full_name = prefix + "." + name;
        
        auto it = weights.find(full_name);
        if (it == weights.end()) {
            // Try alternate naming convention
            full_name = prefix + "_" + name;
            it = weights.find(full_name);
            
            if (it == weights.end()) {
                // Try just the name itself
                it = weights.find(name);
                
                if (it == weights.end()) {
                    CCSM_WARNING("Weight not found: ", full_name);
                    // Return an empty array
                    dst = mlx_array{};
                    return false;
                }
            }
        }
        
        dst = it->second;
        return true;
    };
    
    // Load attention layer weights
    load_weight("attention_norm.weight", attention_norm_weight_);
    load_weight("attention_norm.bias", attention_norm_bias_);
    load_weight("attention.wq.weight", wq_weight_);
    load_weight("attention.wk.weight", wk_weight_);
    load_weight("attention.wv.weight", wv_weight_);
    load_weight("attention.wo.weight", wo_weight_);
    
    // Load feed-forward layer weights
    load_weight("ffn_norm.weight", ffn_norm_weight_);
    load_weight("ffn_norm.bias", ffn_norm_bias_);
    load_weight("ffn.w1.weight", w1_weight_);
    load_weight("ffn.w2.weight", w2_weight_);
    load_weight("ffn.w3.weight", w3_weight_);
    
    CCSM_DEBUG("MLXTransformerLayer initialized successfully");
}

// Helper function for rotary position embeddings
mlx_array MLXTransformerLayer::apply_rotary_embeddings(
    mlx_array x, 
    const std::vector<int>& positions,
    mlx_stream stream) {
    
    return mlx_rotary_embedding(x, positions, rope_theta_, stream);
}

// Helper function for attention mechanism
mlx_array MLXTransformerLayer::attention(
    mlx_array q,
    mlx_array k,
    mlx_array v,
    const std::vector<int>& positions,
    float scale,
    MLXKVCache* kv_cache,
    mlx_stream stream) {
    
    if (kv_cache) {
        // If using KV cache, we need to get the cached keys and values
        int layer_idx = 0; // This should be properly set based on the layer index
        mlx_array cached_k = kv_cache->k_cache(layer_idx);
        mlx_array cached_v = kv_cache->v_cache(layer_idx);
        
        // Check if the cache is valid
        if (cached_k.ctx && cached_v.ctx) {
            // Use the cached keys and values instead
            return mlx_attention(q, cached_k, cached_v, positions, scale, stream);
        }
    }
    
    // No cache or invalid cache, use the current keys and values
    return mlx_attention(q, k, v, positions, scale, stream);
}

MLXTransformerLayer::~MLXTransformerLayer() {
    CCSM_DEBUG("MLXTransformerLayer destructor called");
    
    // Free all MLX arrays
    if (attention_norm_weight_.ctx) mlx_array_free(attention_norm_weight_);
    if (attention_norm_bias_.ctx) mlx_array_free(attention_norm_bias_);
    if (wq_weight_.ctx) mlx_array_free(wq_weight_);
    if (wk_weight_.ctx) mlx_array_free(wk_weight_);
    if (wv_weight_.ctx) mlx_array_free(wv_weight_);
    if (wo_weight_.ctx) mlx_array_free(wo_weight_);
    if (ffn_norm_weight_.ctx) mlx_array_free(ffn_norm_weight_);
    if (ffn_norm_bias_.ctx) mlx_array_free(ffn_norm_bias_);
    if (w1_weight_.ctx) mlx_array_free(w1_weight_);
    if (w2_weight_.ctx) mlx_array_free(w2_weight_);
    if (w3_weight_.ctx) mlx_array_free(w3_weight_);
}

mlx_array MLXTransformerLayer::forward(
    mlx_array x,
    MLXKVCache* kv_cache,
    const std::vector<int>& positions,
    mlx_stream stream) {
    
    CCSM_DEBUG("Executing MLXTransformerLayer forward pass");
    
    // === Attention Block ===
    
    // First apply layer normalization
    mlx_array norm_x;
    if (attention_norm_bias_.ctx) {
        // Apply layer norm with bias
        mlx_layer_norm(x, attention_norm_weight_, attention_norm_bias_, 1e-5, &norm_x);
    } else {
        // Apply layer norm without bias
        mlx_layer_norm(x, attention_norm_weight_, 1e-5, &norm_x);
    }
    
    // Project to query, key, value
    mlx_array query, key, value;
    mlx_matmul(norm_x, wq_weight_, &query);
    mlx_matmul(norm_x, wk_weight_, &key);
    mlx_matmul(norm_x, wv_weight_, &value);
    
    // Get dimensions
    uint32_t ndim;
    mlx_array_ndim(query, &ndim);
    const int* shape = mlx_array_shape(query);
    int batch_size = shape[0];
    int seq_len = shape[1];
    
    // Reshape for multi-head attention
    // [batch, seq_len, d_model] -> [batch, seq_len, n_heads, head_dim]
    int query_shape[] = {batch_size, seq_len, n_heads_, head_dim_};
    mlx_array reshaped_query;
    mlx_array_reshape(query, query_shape, 4, &reshaped_query);
    
    int key_shape[] = {batch_size, seq_len, n_kv_heads_, head_dim_};
    mlx_array reshaped_key;
    mlx_array_reshape(key, key_shape, 4, &reshaped_key);
    
    int value_shape[] = {batch_size, seq_len, n_kv_heads_, head_dim_};
    mlx_array reshaped_value;
    mlx_array_reshape(value, value_shape, 4, &reshaped_value);
    
    // Apply rotary position embeddings
    mlx_array rotary_query = apply_rotary_embeddings(reshaped_query, positions, stream);
    mlx_array rotary_key = apply_rotary_embeddings(reshaped_key, positions, stream);
    
    // Update KV cache if provided
    if (kv_cache) {
        kv_cache->update(0, rotary_key, reshaped_value, positions, stream);
    }
    
    // Compute attention
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim_));
    mlx_array attn_output = attention(rotary_query, rotary_key, reshaped_value, positions, scale, stream);
    
    // Reshape attention output back to [batch, seq_len, d_model]
    int attn_output_shape[] = {batch_size, seq_len, d_model_};
    mlx_array reshaped_attn_output;
    mlx_array_reshape(attn_output, attn_output_shape, 3, &reshaped_attn_output);
    
    // Project attention output
    mlx_array projected_attn;
    mlx_matmul(reshaped_attn_output, wo_weight_, &projected_attn);
    
    // Residual connection
    mlx_array attn_block_output;
    mlx_add(x, projected_attn, &attn_block_output);
    
    // Clean up intermediate attention arrays
    mlx_array_free(norm_x);
    mlx_array_free(query);
    mlx_array_free(key);
    mlx_array_free(value);
    mlx_array_free(reshaped_query);
    mlx_array_free(reshaped_key);
    mlx_array_free(reshaped_value);
    mlx_array_free(rotary_query);
    mlx_array_free(rotary_key);
    mlx_array_free(attn_output);
    mlx_array_free(reshaped_attn_output);
    mlx_array_free(projected_attn);
    
    // === Feed-Forward Block ===
    
    // Apply layer normalization
    mlx_array ffn_norm_x;
    if (ffn_norm_bias_.ctx) {
        // Apply layer norm with bias
        mlx_layer_norm(attn_block_output, ffn_norm_weight_, ffn_norm_bias_, 1e-5, &ffn_norm_x);
    } else {
        // Apply layer norm without bias
        mlx_layer_norm(attn_block_output, ffn_norm_weight_, 1e-5, &ffn_norm_x);
    }
    
    // Apply feed-forward network with SwiGLU
    mlx_array ffn_output = feed_forward(ffn_norm_x, w1_weight_, w2_weight_, w3_weight_, stream);
    
    // Residual connection
    mlx_array output;
    mlx_add(attn_block_output, ffn_output, &output);
    
    // Clean up remaining intermediate arrays
    mlx_array_free(attn_block_output);
    mlx_array_free(ffn_norm_x);
    mlx_array_free(ffn_output);
    
    return output;
}

// MLXTransformer implementation
MLXTransformer::MLXTransformer(
    const std::unordered_map<std::string, mlx_array>& weights,
    int d_model,
    int n_layers,
    int n_heads,
    int n_kv_heads,
    int vocab_size,
    float rope_theta)
    : d_model_(d_model),
      n_layers_(n_layers),
      n_heads_(n_heads),
      n_kv_heads_(n_kv_heads),
      vocab_size_(vocab_size),
      rope_theta_(rope_theta) {
    
    CCSM_DEBUG("Initializing MLXTransformer with", n_layers, "layers");
    
    // Create KV cache with reasonable max sequence length
    int max_seq_len = 2048;
    int head_dim = d_model / n_heads;
    kv_cache_ = std::make_unique<MLXKVCache>(n_layers, n_kv_heads, head_dim, max_seq_len);
    
    // Load the embedding weights
    auto it = weights.find("token_embedding.weight");
    if (it != weights.end()) {
        tok_embeddings_weight_ = it->second;
    } else {
        it = weights.find("embedding.weight");
        if (it != weights.end()) {
            tok_embeddings_weight_ = it->second;
        } else {
            CCSM_WARNING("Token embedding weights not found");
        }
    }
    
    // Load normalization weights
    it = weights.find("norm.weight");
    if (it != weights.end()) {
        norm_weight_ = it->second;
    } else {
        CCSM_WARNING("Norm weights not found");
    }
    
    it = weights.find("norm.bias");
    if (it != weights.end()) {
        norm_bias_ = it->second;
    }
    
    // Load output weights
    it = weights.find("output.weight");
    if (it != weights.end()) {
        output_weight_ = it->second;
    } else {
        it = weights.find("lm_head.weight");
        if (it != weights.end()) {
            output_weight_ = it->second;
        } else {
            CCSM_WARNING("Output weights not found");
        }
    }
    
    // Create transformer layers
    for (int i = 0; i < n_layers; i++) {
        std::string layer_prefix = "layers." + std::to_string(i);
        
        auto layer = std::make_unique<MLXTransformerLayer>(
            layer_prefix,
            weights,
            d_model,
            n_heads,
            n_kv_heads,
            rope_theta
        );
        
        layers_.push_back(std::move(layer));
    }
    
    CCSM_INFO("MLXTransformer initialized with", n_layers, "layers");
}

MLXTransformer::~MLXTransformer() {
    CCSM_DEBUG("MLXTransformer destructor called");
    
    // Free all MLX arrays used by the transformer
    if (tok_embeddings_weight_.ctx) {
        mlx_array_free(tok_embeddings_weight_);
    }
    
    if (norm_weight_.ctx) {
        mlx_array_free(norm_weight_);
    }
    
    if (norm_bias_.ctx) {
        mlx_array_free(norm_bias_);
    }
    
    if (output_weight_.ctx) {
        mlx_array_free(output_weight_);
    }
    
    // The layers and KV cache will be automatically freed by their destructors
}

mlx_array MLXTransformer::forward(
    const std::vector<int>& tokens,
    const std::vector<int>& positions,
    MLXKVCache* kv_cache,
    mlx_stream stream) {
    
    CCSM_DEBUG("MLXTransformer::forward called with ", tokens.size(), " tokens");
    
    if (tokens.empty()) {
        CCSM_WARNING("Empty tokens array passed to MLXTransformer::forward");
        return mlx_array{};
    }
    
    if (positions.empty() || positions.size() != tokens.size()) {
        CCSM_WARNING("Invalid positions array passed to MLXTransformer::forward");
        return mlx_array{};
    }
    
    if (!tok_embeddings_weight_.ctx) {
        CCSM_ERROR("Token embedding weights not initialized");
        return mlx_array{};
    }
    
    // Use the provided KV cache or the built-in one
    MLXKVCache* cache_ptr = kv_cache ? kv_cache : kv_cache_.get();
    
    // Convert tokens to MLX array
    std::vector<int32_t> token_data(tokens.begin(), tokens.end());
    int shape[] = {static_cast<int>(tokens.size())};
    mlx_array token_array = mlx_array_new_data(
        token_data.data(),
        shape,
        1,
        MLX_INT32
    );
    
    // Look up embeddings
    mlx_array embeddings;
    mlx_embedding_lookup(token_array, tok_embeddings_weight_, &embeddings);
    mlx_array_free(token_array);
    
    // Get the batch size (always 1 for now) and sequence length
    uint32_t ndim;
    mlx_array_ndim(embeddings, &ndim);
    const int* embed_shape = mlx_array_shape(embeddings);
    int seq_len = embed_shape[0];
    
    // Process through all transformer layers
    mlx_array layer_input = embeddings;
    for (size_t i = 0; i < layers_.size(); ++i) {
        mlx_array layer_output = layers_[i]->forward(layer_input, cache_ptr, positions, stream);
        
        if (i > 0) {
            mlx_array_free(layer_input); // Free previous layer's output
        }
        layer_input = layer_output;
    }
    
    // Apply final layer normalization
    mlx_array norm_output;
    if (norm_bias_.ctx) {
        mlx_layer_norm(layer_input, norm_weight_, norm_bias_, 1e-5, &norm_output);
    } else {
        mlx_layer_norm(layer_input, norm_weight_, 1e-5, &norm_output);
    }
    
    // We're done with layer_input now
    if (layers_.size() > 0) {
        mlx_array_free(layer_input);
    }
    
    // Get the last token's representation
    int last_token_idx[] = {seq_len - 1};
    mlx_array last_token;
    mlx_array_index(norm_output, 0, last_token_idx, 1, &last_token);
    mlx_array_free(norm_output);
    
    // Project to logits
    mlx_array logits;
    mlx_matmul(last_token, output_weight_, &logits);
    mlx_array_free(last_token);
    
    return logits;
}

void MLXTransformer::reset_caches() {
    CCSM_DEBUG("MLXTransformer::reset_caches called");
    if (kv_cache_) {
        kv_cache_->clear();
    }
}

bool MLXTransformer::prune_kv_cache(size_t target_len) {
    CCSM_DEBUG("MLXTransformer::prune_kv_cache called with target_len=", target_len);
    
    if (!kv_cache_) {
        CCSM_WARNING("Cannot prune KV cache: KV cache is not initialized");
        return false;
    }
    
    // Get current cache parameters
    size_t current_len = kv_cache_->current_seq_len();
    size_t max_len = kv_cache_->max_seq_len();
    
    // Validate target length
    if (target_len > max_len) {
        CCSM_WARNING("Target length exceeds maximum cache length");
        return false;
    }
    
    if (target_len == current_len) {
        CCSM_DEBUG("Target length already matches current length, no pruning needed");
        return true;
    }
    
    if (target_len > current_len) {
        CCSM_DEBUG("Target length is larger than current length, no pruning needed");
        return true;
    }
    
    // Create a stream for operations
    mlx_stream stream = mlx_default_cpu_stream_new();
    bool success = kv_cache_->prune(target_len, stream);
    
    CCSM_DEBUG("KV cache pruning ", success ? "succeeded" : "failed");
    return success;
}

#else // CCSM_WITH_MLX
// Empty implementations for when MLX is not available
#endif // CCSM_WITH_MLX

} // namespace ccsm