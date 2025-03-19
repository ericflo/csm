#include <ccsm/mlx/mlx_transformer.h>
#include <ccsm/mlx/mlx_tensor.h>
#include <ccsm/utils.h>
#include <stdexcept>
#include <iostream>
#include <cmath>

#ifdef CCSM_WITH_MLX
#include <mlx/c/array.h>
#include <mlx/c/ops.h>
#include <mlx/c/device.h>
#include <mlx/c/stream.h>
#endif

namespace ccsm {

#ifdef CCSM_WITH_MLX

// Utility function to check MLX errors (reused from mlx_tensor.cpp)
static void check_mlx_result(int result, const char* operation) {
    if (result != 0) {
        throw std::runtime_error(std::string("MLX operation failed: ") + operation);
    }
}

// MLX transformer helper functions
mlx_array mlx_rotary_embedding(
    mlx_array x,
    const std::vector<int>& positions,
    float theta,
    mlx_stream stream) {
    
    // Get dimensions of x
    int head_dim = mlx_array_dim(x, 0);
    int n_heads = mlx_array_dim(x, 1);
    int seq_len = mlx_array_dim(x, 2);
    
    // Half the head dimension for complex representation
    int half_dim = head_dim / 2;
    
    // Create position frequencies tensor
    mlx_array freqs;
    {
        // Create tensor with arange(0, half_dim)
        std::vector<int> indices(half_dim);
        for (int i = 0; i < half_dim; i++) {
            indices[i] = i;
        }
        
        // Shape of indices: [half_dim]
        int shape[] = {half_dim};
        
        mlx_array indices_tensor;
        check_mlx_result(mlx_array_new_data(
            &indices_tensor,
            indices.data(),
            shape, 1,
            MLX_INT32
        ), "array_new_data(indices)");
        
        // Convert to float32
        mlx_array indices_float;
        check_mlx_result(mlx_astype(
            &indices_float,
            indices_tensor,
            MLX_FLOAT32,
            stream
        ), "astype(indices)");
        
        // Compute freqs = 1.0 / (theta ** (2.0 * indices / half_dim))
        mlx_array div;
        {
            mlx_array two = mlx_array_new_float(2.0f);
            mlx_array half_dim_tensor = mlx_array_new_float((float)half_dim);
            
            // div = 2.0 * indices / half_dim
            mlx_array two_times_indices;
            check_mlx_result(mlx_multiply(
                &two_times_indices,
                indices_float,
                two,
                stream
            ), "multiply(indices, two)");
            
            check_mlx_result(mlx_divide(
                &div,
                two_times_indices,
                half_dim_tensor,
                stream
            ), "divide(two_times_indices, half_dim)");
            
            // Free temporary tensors
            mlx_array_free(two);
            mlx_array_free(half_dim_tensor);
            mlx_array_free(two_times_indices);
        }
        
        // Compute freqs = theta ** div
        mlx_array theta_tensor = mlx_array_new_float(theta);
        check_mlx_result(mlx_power(
            &freqs,
            theta_tensor,
            div,
            stream
        ), "power(theta, div)");
        
        // Compute freqs = 1.0 / freqs
        mlx_array one = mlx_array_new_float(1.0f);
        mlx_array inv_freqs;
        check_mlx_result(mlx_divide(
            &inv_freqs,
            one,
            freqs,
            stream
        ), "divide(one, freqs)");
        freqs = inv_freqs;
        
        // Free temporary tensors
        mlx_array_free(indices_tensor);
        mlx_array_free(indices_float);
        mlx_array_free(div);
        mlx_array_free(theta_tensor);
        mlx_array_free(one);
    }
    
    // Create sin and cos functions for each position
    std::vector<mlx_array> cos_tensors;
    std::vector<mlx_array> sin_tensors;
    
    for (int pos : positions) {
        // Create position tensor (scalar)
        mlx_array pos_tensor = mlx_array_new_float((float)pos);
        
        // Compute freqs * pos
        mlx_array freqs_pos;
        check_mlx_result(mlx_multiply(
            &freqs_pos,
            freqs,
            pos_tensor,
            stream
        ), "multiply(freqs, pos)");
        
        // Compute cos and sin of freqs_pos
        mlx_array cos_tensor, sin_tensor;
        check_mlx_result(mlx_cos(&cos_tensor, freqs_pos, stream), "cos(freqs_pos)");
        check_mlx_result(mlx_sin(&sin_tensor, freqs_pos, stream), "sin(freqs_pos)");
        
        cos_tensors.push_back(cos_tensor);
        sin_tensors.push_back(sin_tensor);
        
        // Free temporary tensors
        mlx_array_free(pos_tensor);
        mlx_array_free(freqs_pos);
    }
    
    // Convert x to a list of query vectors, one per position
    std::vector<mlx_array> x_positions;
    for (size_t i = 0; i < positions.size(); i++) {
        int start[] = {0, 0, (int)i};
        int stop[] = {head_dim, n_heads, (int)i + 1};
        int strides[] = {1, 1, 1};
        
        mlx_array x_pos;
        check_mlx_result(mlx_slice(
            &x_pos,
            x,
            start, 3,
            stop, 3,
            strides, 3,
            stream
        ), "slice(x)");
        
        // Reshape to [head_dim, n_heads]
        int new_shape[] = {head_dim, n_heads};
        mlx_array x_pos_reshaped;
        check_mlx_result(mlx_reshape(
            &x_pos_reshaped,
            x_pos,
            new_shape, 2,
            stream
        ), "reshape(x_pos)");
        
        x_positions.push_back(x_pos_reshaped);
        mlx_array_free(x_pos);
    }
    
    // Apply rotary embeddings to each position
    std::vector<mlx_array> rotated_positions;
    for (size_t i = 0; i < positions.size(); i++) {
        // Split into first and second half (real and imaginary parts)
        int first_half_start[] = {0, 0};
        int first_half_stop[] = {half_dim, n_heads};
        int second_half_start[] = {half_dim, 0};
        int second_half_stop[] = {head_dim, n_heads};
        int strides[] = {1, 1};
        
        mlx_array x_first_half, x_second_half;
        check_mlx_result(mlx_slice(
            &x_first_half,
            x_positions[i],
            first_half_start, 2,
            first_half_stop, 2,
            strides, 2,
            stream
        ), "slice(x_positions[i], first_half)");
        
        check_mlx_result(mlx_slice(
            &x_second_half,
            x_positions[i],
            second_half_start, 2,
            second_half_stop, 2,
            strides, 2,
            stream
        ), "slice(x_positions[i], second_half)");
        
        // Apply rotary embeddings
        // x_first_half_new = x_first_half * cos - x_second_half * sin
        // x_second_half_new = x_first_half * sin + x_second_half * cos
        
        // x_first_half * cos
        mlx_array x_first_half_cos;
        check_mlx_result(mlx_multiply(
            &x_first_half_cos,
            x_first_half,
            cos_tensors[i],
            stream
        ), "multiply(x_first_half, cos)");
        
        // x_second_half * sin
        mlx_array x_second_half_sin;
        check_mlx_result(mlx_multiply(
            &x_second_half_sin,
            x_second_half,
            sin_tensors[i],
            stream
        ), "multiply(x_second_half, sin)");
        
        // x_first_half_new = x_first_half * cos - x_second_half * sin
        mlx_array x_first_half_new;
        check_mlx_result(mlx_subtract(
            &x_first_half_new,
            x_first_half_cos,
            x_second_half_sin,
            stream
        ), "subtract(x_first_half_cos, x_second_half_sin)");
        
        // x_first_half * sin
        mlx_array x_first_half_sin;
        check_mlx_result(mlx_multiply(
            &x_first_half_sin,
            x_first_half,
            sin_tensors[i],
            stream
        ), "multiply(x_first_half, sin)");
        
        // x_second_half * cos
        mlx_array x_second_half_cos;
        check_mlx_result(mlx_multiply(
            &x_second_half_cos,
            x_second_half,
            cos_tensors[i],
            stream
        ), "multiply(x_second_half, cos)");
        
        // x_second_half_new = x_first_half * sin + x_second_half * cos
        mlx_array x_second_half_new;
        check_mlx_result(mlx_add(
            &x_second_half_new,
            x_first_half_sin,
            x_second_half_cos,
            stream
        ), "add(x_first_half_sin, x_second_half_cos)");
        
        // Concatenate first and second half
        std::vector<mlx_array> to_concat = {x_first_half_new, x_second_half_new};
        int axis = 0;
        
        mlx_array x_pos_rotated;
        check_mlx_result(mlx_concatenate(
            &x_pos_rotated,
            to_concat.data(), to_concat.size(),
            axis,
            stream
        ), "concatenate(x_first_half_new, x_second_half_new)");
        
        // Reshape to [head_dim, n_heads, 1]
        int new_shape[] = {head_dim, n_heads, 1};
        mlx_array x_pos_reshaped;
        check_mlx_result(mlx_reshape(
            &x_pos_reshaped,
            x_pos_rotated,
            new_shape, 3,
            stream
        ), "reshape(x_pos_rotated)");
        
        rotated_positions.push_back(x_pos_reshaped);
        
        // Free temporary tensors
        mlx_array_free(x_first_half);
        mlx_array_free(x_second_half);
        mlx_array_free(x_first_half_cos);
        mlx_array_free(x_second_half_sin);
        mlx_array_free(x_first_half_new);
        mlx_array_free(x_first_half_sin);
        mlx_array_free(x_second_half_cos);
        mlx_array_free(x_second_half_new);
        mlx_array_free(x_pos_rotated);
    }
    
    // Concatenate all positions
    mlx_array result;
    check_mlx_result(mlx_concatenate(
        &result,
        rotated_positions.data(), rotated_positions.size(),
        2, // axis
        stream
    ), "concatenate(rotated_positions)");
    
    // Free temporary tensors
    mlx_array_free(freqs);
    for (auto& tensor : cos_tensors) {
        mlx_array_free(tensor);
    }
    for (auto& tensor : sin_tensors) {
        mlx_array_free(tensor);
    }
    for (auto& tensor : x_positions) {
        mlx_array_free(tensor);
    }
    for (auto& tensor : rotated_positions) {
        mlx_array_free(tensor);
    }
    
    return result;
}

mlx_array mlx_attention(
    mlx_array query,
    mlx_array key,
    mlx_array value,
    const std::vector<int>& positions,
    float scale,
    mlx_stream stream) {
    
    // Shape info
    int head_dim = mlx_array_dim(query, 0);
    int n_heads = mlx_array_dim(query, 1);
    int seq_len = mlx_array_dim(query, 2);
    
    // Reshape query for matrix multiplication: [head_dim, n_heads, seq_len] -> [head_dim, n_heads * seq_len]
    int q_shape[] = {head_dim, n_heads * seq_len};
    mlx_array q_reshaped;
    check_mlx_result(mlx_reshape(
        &q_reshaped,
        query,
        q_shape, 2,
        stream
    ), "reshape(query)");
    
    // Reshape key for matrix multiplication: [head_dim, n_heads, key_len] -> [head_dim, n_heads * key_len]
    int key_len = mlx_array_dim(key, 2);
    int k_shape[] = {head_dim, n_heads * key_len};
    mlx_array k_reshaped;
    check_mlx_result(mlx_reshape(
        &k_reshaped,
        key,
        k_shape, 2,
        stream
    ), "reshape(key)");
    
    // Compute attention scores: q @ k.T
    mlx_array attention_scores;
    check_mlx_result(mlx_matmul(
        &attention_scores,
        q_reshaped,
        k_reshaped,
        true, // transpose second input (key)
        stream
    ), "matmul(q_reshaped, k_reshaped, transpose=true)");
    
    // Reshape attention scores: [n_heads * seq_len, n_heads * key_len] -> [n_heads, seq_len, key_len]
    int scores_shape[] = {n_heads, seq_len, key_len};
    mlx_array scores_reshaped;
    check_mlx_result(mlx_reshape(
        &scores_reshaped,
        attention_scores,
        scores_shape, 3,
        stream
    ), "reshape(attention_scores)");
    
    // Scale attention scores
    mlx_array scale_tensor = mlx_array_new_float(scale);
    mlx_array scores_scaled;
    check_mlx_result(mlx_multiply(
        &scores_scaled,
        scores_reshaped,
        scale_tensor,
        stream
    ), "multiply(scores_reshaped, scale)");
    
    // Apply causal mask
    // Create a mask where mask[i, j] = -inf if j > positions[i]
    mlx_array mask;
    {
        // Create a mask tensor filled with zeros
        int mask_shape[] = {seq_len, key_len};
        check_mlx_result(mlx_zeros(
            &mask,
            mask_shape, 2,
            MLX_FLOAT32,
            stream
        ), "zeros(mask_shape)");
        
        // Create a float tensor with -inf value
        mlx_array neg_inf = mlx_array_new_float(-INFINITY);
        
        // For each position, update the corresponding mask row
        for (size_t i = 0; i < positions.size(); i++) {
            int pos = positions[i];
            
            // For each key position
            for (int j = 0; j < key_len; j++) {
                // If this key is from a future position, mask it
                if (j > pos) {
                    // Create tensor for this specific position in the mask
                    int start[] = {(int)i, j};
                    int stop[] = {(int)i + 1, j + 1};
                    int strides[] = {1, 1};
                    
                    mlx_array mask_pos;
                    check_mlx_result(mlx_slice(
                        &mask_pos,
                        mask,
                        start, 2,
                        stop, 2,
                        strides, 2,
                        stream
                    ), "slice(mask)");
                    
                    // Set this position to -inf
                    check_mlx_result(mlx_copy_to(
                        mask_pos,
                        neg_inf,
                        stream
                    ), "copy_to(mask_pos, neg_inf)");
                    
                    mlx_array_free(mask_pos);
                }
            }
        }
        
        // Reshape mask to [1, seq_len, key_len] for broadcasting across heads
        int new_mask_shape[] = {1, seq_len, key_len};
        mlx_array mask_reshaped;
        check_mlx_result(mlx_reshape(
            &mask_reshaped,
            mask,
            new_mask_shape, 3,
            stream
        ), "reshape(mask)");
        
        mlx_array_free(mask);
        mask = mask_reshaped;
        mlx_array_free(neg_inf);
    }
    
    // Apply mask to attention scores
    mlx_array masked_scores;
    check_mlx_result(mlx_add(
        &masked_scores,
        scores_scaled,
        mask,
        stream
    ), "add(scores_scaled, mask)");
    
    // Apply softmax to get attention weights
    int axes[] = {2}; // Apply softmax along the key dimension
    mlx_array attention_weights;
    check_mlx_result(mlx_softmax(
        &attention_weights,
        masked_scores,
        axes, 1,
        true, // keepdims
        stream
    ), "softmax(masked_scores)");
    
    // Reshape value for matrix multiplication: [head_dim, n_heads, key_len] -> [key_len, n_heads, head_dim]
    int v_shape[] = {key_len, n_heads, head_dim};
    mlx_array v_reshaped;
    check_mlx_result(mlx_reshape(
        &v_reshaped,
        value,
        v_shape, 3,
        stream
    ), "reshape(value)");
    
    // Reshape attention weights for matrix multiplication: [n_heads, seq_len, key_len] -> [seq_len, n_heads, key_len]
    int weights_shape[] = {seq_len, n_heads, key_len};
    mlx_array weights_reshaped;
    check_mlx_result(mlx_reshape(
        &weights_reshaped,
        attention_weights,
        weights_shape, 3,
        stream
    ), "reshape(attention_weights)");
    
    // Compute weighted sum: attention_weights @ value
    // First transpose to prepare for batched matmul
    mlx_array weights_transposed, v_transposed;
    check_mlx_result(mlx_transpose(
        &weights_transposed,
        weights_reshaped,
        stream
    ), "transpose(weights_reshaped)");
    check_mlx_result(mlx_transpose(
        &v_transposed,
        v_reshaped,
        stream
    ), "transpose(v_reshaped)");
    
    // Now do a batch matrix multiply:
    // weighted_value shape: [seq_len, n_heads, head_dim]
    mlx_array weighted_value;
    check_mlx_result(mlx_matmul_batched(
        &weighted_value,
        weights_transposed,
        v_transposed,
        stream
    ), "matmul_batched(weights_transposed, v_transposed)");
    
    // Reshape to [head_dim, n_heads, seq_len]
    int result_shape[] = {head_dim, n_heads, seq_len};
    mlx_array result;
    check_mlx_result(mlx_reshape(
        &result,
        weighted_value,
        result_shape, 3,
        stream
    ), "reshape(weighted_value)");
    
    // Free temporary tensors
    mlx_array_free(q_reshaped);
    mlx_array_free(k_reshaped);
    mlx_array_free(attention_scores);
    mlx_array_free(scores_reshaped);
    mlx_array_free(scale_tensor);
    mlx_array_free(scores_scaled);
    mlx_array_free(mask);
    mlx_array_free(masked_scores);
    mlx_array_free(attention_weights);
    mlx_array_free(v_reshaped);
    mlx_array_free(weights_reshaped);
    mlx_array_free(weights_transposed);
    mlx_array_free(v_transposed);
    mlx_array_free(weighted_value);
    
    return result;
}

mlx_array mlx_feed_forward(
    mlx_array x,
    mlx_array w1,
    mlx_array w2,
    mlx_array w3,
    mlx_stream stream) {
    
    // Get dimensions
    int n_embd = mlx_array_dim(x, 0);
    int seq_len = mlx_array_dim(x, 1);
    
    // First linear transformation: x @ w1
    mlx_array xw1;
    check_mlx_result(mlx_matmul(
        &xw1,
        x,
        w1,
        false, // no transpose
        stream
    ), "matmul(x, w1)");
    
    // Second linear transformation: x @ w3
    mlx_array xw3;
    check_mlx_result(mlx_matmul(
        &xw3,
        x,
        w3,
        false, // no transpose
        stream
    ), "matmul(x, w3)");
    
    // Apply SiLU activation to xw1
    mlx_array xw1_silu;
    check_mlx_result(mlx_silu(
        &xw1_silu,
        xw1,
        stream
    ), "silu(xw1)");
    
    // Multiply SiLU(xw1) * xw3
    mlx_array gated;
    check_mlx_result(mlx_multiply(
        &gated,
        xw1_silu,
        xw3,
        stream
    ), "multiply(xw1_silu, xw3)");
    
    // Final linear transformation: gated @ w2
    mlx_array result;
    check_mlx_result(mlx_matmul(
        &result,
        gated,
        w2,
        false, // no transpose
        stream
    ), "matmul(gated, w2)");
    
    // Free temporary tensors
    mlx_array_free(xw1);
    mlx_array_free(xw3);
    mlx_array_free(xw1_silu);
    mlx_array_free(gated);
    
    return result;
}

// MLXKVCache implementation
MLXKVCache::MLXKVCache(size_t n_layers, size_t n_heads, size_t head_dim, size_t max_seq_len)
    : n_layers_(n_layers),
      n_heads_(n_heads),
      head_dim_(head_dim),
      max_seq_len_(max_seq_len),
      current_seq_len_(0) {
    
    // Create caches for each layer
    for (size_t i = 0; i < n_layers_; i++) {
        // K cache: [head_dim, n_heads, max_seq_len]
        int k_shape[] = {(int)head_dim_, (int)n_heads_, (int)max_seq_len_};
        mlx_array k_cache;
        check_mlx_result(mlx_zeros(
            &k_cache,
            k_shape, 3,
            MLX_FLOAT32,
            mlx_stream_null
        ), "zeros(k_cache)");
        k_caches_.push_back(k_cache);
        
        // V cache: [head_dim, n_heads, max_seq_len]
        int v_shape[] = {(int)head_dim_, (int)n_heads_, (int)max_seq_len_};
        mlx_array v_cache;
        check_mlx_result(mlx_zeros(
            &v_cache,
            v_shape, 3,
            MLX_FLOAT32,
            mlx_stream_null
        ), "zeros(v_cache)");
        v_caches_.push_back(v_cache);
    }
}

MLXKVCache::~MLXKVCache() {
    // Free all tensors
    for (auto& cache : k_caches_) {
        mlx_array_free(cache);
    }
    for (auto& cache : v_caches_) {
        mlx_array_free(cache);
    }
}

void MLXKVCache::clear() {
    // Reset sequence length
    current_seq_len_ = 0;
    
    // Zero out all caches
    for (size_t i = 0; i < n_layers_; i++) {
        int k_shape[] = {(int)head_dim_, (int)n_heads_, (int)max_seq_len_};
        check_mlx_result(mlx_zeros_like(
            &k_caches_[i],
            k_caches_[i],
            mlx_stream_null
        ), "zeros_like(k_cache)");
        
        int v_shape[] = {(int)head_dim_, (int)n_heads_, (int)max_seq_len_};
        check_mlx_result(mlx_zeros_like(
            &v_caches_[i],
            v_caches_[i],
            mlx_stream_null
        ), "zeros_like(v_cache)");
    }
}

void MLXKVCache::resize(size_t seq_len) {
    if (seq_len > max_seq_len_) {
        throw std::runtime_error("Cannot resize KV cache beyond maximum sequence length");
    }
    current_seq_len_ = seq_len;
}

mlx_array MLXKVCache::k_cache(int layer) const {
    if (layer < 0 || layer >= (int)n_layers_) {
        throw std::out_of_range("Layer index out of range in KV cache");
    }
    return k_caches_[layer];
}

mlx_array MLXKVCache::v_cache(int layer) const {
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
    if (layer < 0 || layer >= (int)n_layers_) {
        throw std::out_of_range("Layer index out of range in KV cache");
    }
    
    // Update K cache
    for (size_t i = 0; i < positions.size(); i++) {
        int pos = positions[i];
        
        if (pos >= (int)max_seq_len_) {
            throw std::out_of_range("Position index out of range in KV cache");
        }
        
        // Get the slice of k at position i
        int k_start[] = {0, 0, (int)i};
        int k_stop[] = {(int)head_dim_, (int)n_heads_, (int)i + 1};
        int k_strides[] = {1, 1, 1};
        
        mlx_array k_slice;
        check_mlx_result(mlx_slice(
            &k_slice,
            k,
            k_start, 3,
            k_stop, 3,
            k_strides, 3,
            stream
        ), "slice(k)");
        
        // Reshape to [head_dim, n_heads, 1]
        int k_shape[] = {(int)head_dim_, (int)n_heads_, 1};
        mlx_array k_reshaped;
        check_mlx_result(mlx_reshape(
            &k_reshaped,
            k_slice,
            k_shape, 3,
            stream
        ), "reshape(k_slice)");
        
        // Get the slice of the cache at position pos
        int cache_start[] = {0, 0, pos};
        int cache_stop[] = {(int)head_dim_, (int)n_heads_, pos + 1};
        int cache_strides[] = {1, 1, 1};
        
        mlx_array cache_slice;
        check_mlx_result(mlx_slice(
            &cache_slice,
            k_caches_[layer],
            cache_start, 3,
            cache_stop, 3,
            cache_strides, 3,
            stream
        ), "slice(k_cache)");
        
        // Update the cache
        check_mlx_result(mlx_copy_to(
            cache_slice,
            k_reshaped,
            stream
        ), "copy_to(cache_slice, k_reshaped)");
        
        // Free temporary tensors
        mlx_array_free(k_slice);
        mlx_array_free(k_reshaped);
        mlx_array_free(cache_slice);
    }
    
    // Update V cache
    for (size_t i = 0; i < positions.size(); i++) {
        int pos = positions[i];
        
        // Get the slice of v at position i
        int v_start[] = {0, 0, (int)i};
        int v_stop[] = {(int)head_dim_, (int)n_heads_, (int)i + 1};
        int v_strides[] = {1, 1, 1};
        
        mlx_array v_slice;
        check_mlx_result(mlx_slice(
            &v_slice,
            v,
            v_start, 3,
            v_stop, 3,
            v_strides, 3,
            stream
        ), "slice(v)");
        
        // Reshape to [head_dim, n_heads, 1]
        int v_shape[] = {(int)head_dim_, (int)n_heads_, 1};
        mlx_array v_reshaped;
        check_mlx_result(mlx_reshape(
            &v_reshaped,
            v_slice,
            v_shape, 3,
            stream
        ), "reshape(v_slice)");
        
        // Get the slice of the cache at position pos
        int cache_start[] = {0, 0, pos};
        int cache_stop[] = {(int)head_dim_, (int)n_heads_, pos + 1};
        int cache_strides[] = {1, 1, 1};
        
        mlx_array cache_slice;
        check_mlx_result(mlx_slice(
            &cache_slice,
            v_caches_[layer],
            cache_start, 3,
            cache_stop, 3,
            cache_strides, 3,
            stream
        ), "slice(v_cache)");
        
        // Update the cache
        check_mlx_result(mlx_copy_to(
            cache_slice,
            v_reshaped,
            stream
        ), "copy_to(cache_slice, v_reshaped)");
        
        // Free temporary tensors
        mlx_array_free(v_slice);
        mlx_array_free(v_reshaped);
        mlx_array_free(cache_slice);
    }
    
    // Update current sequence length if needed
    int max_pos = 0;
    for (int pos : positions) {
        max_pos = std::max(max_pos, pos);
    }
    current_seq_len_ = std::max(current_seq_len_, (size_t)(max_pos + 1));
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
    
    // Load weights
    auto get_weight = [&weights, &prefix](const std::string& name) -> mlx_array {
        std::string full_name = prefix + name;
        auto it = weights.find(full_name);
        if (it != weights.end()) {
            return it->second;
        }
        
        // Try alternative naming formats
        std::vector<std::string> alternatives = {
            // Try with different separators
            prefix + "." + name,
            prefix + "_" + name,
            // Try with different capitalization
            prefix + ".Attention." + name,
            prefix + ".attention." + name,
            // Try with different field names
            prefix + ".attention_norm." + name,
            prefix + ".attn_norm." + name,
            prefix + ".ffn_norm." + name,
            prefix + ".feed_forward." + name,
            prefix + ".ffn." + name
        };
        
        for (const auto& alt : alternatives) {
            auto it = weights.find(alt);
            if (it != weights.end()) {
                return it->second;
            }
        }
        
        CCSM_WARNING("Weight not found: ", full_name);
        mlx_array empty = {nullptr};
        return empty;
    };
    
    // Attention norm weights
    attention_norm_weight_ = get_weight("attention_norm.weight");
    attention_norm_bias_ = get_weight("attention_norm.bias");
    
    // Attention weights
    wq_weight_ = get_weight("attention.wq.weight");
    wk_weight_ = get_weight("attention.wk.weight");
    wv_weight_ = get_weight("attention.wv.weight");
    wo_weight_ = get_weight("attention.wo.weight");
    
    // FFN norm weights
    ffn_norm_weight_ = get_weight("ffn_norm.weight");
    ffn_norm_bias_ = get_weight("ffn_norm.bias");
    
    // FFN weights
    w1_weight_ = get_weight("feed_forward.w1.weight");
    w2_weight_ = get_weight("feed_forward.w2.weight");
    w3_weight_ = get_weight("feed_forward.w3.weight");
}

MLXTransformerLayer::~MLXTransformerLayer() {
    // No need to free memory for weights as they are owned by the caller
}

mlx_array MLXTransformerLayer::forward(
    mlx_array x,
    MLXKVCache* kv_cache,
    const std::vector<int>& positions,
    mlx_stream stream) {
    
    // Save the input for the residual connection
    mlx_array residual = x;
    
    // Apply attention norm
    mlx_array normalized;
    check_mlx_result(mlx_layer_norm(
        &normalized,
        x,
        attention_norm_weight_,
        attention_norm_bias_,
        1e-5, // epsilon
        stream
    ), "layer_norm(x)");
    
    // Apply QKV projections
    mlx_array q, k, v;
    check_mlx_result(mlx_matmul(
        &q,
        normalized,
        wq_weight_,
        false, // no transpose
        stream
    ), "matmul(normalized, wq)");
    
    check_mlx_result(mlx_matmul(
        &k,
        normalized,
        wk_weight_,
        false, // no transpose
        stream
    ), "matmul(normalized, wk)");
    
    check_mlx_result(mlx_matmul(
        &v,
        normalized,
        wv_weight_,
        false, // no transpose
        stream
    ), "matmul(normalized, wv)");
    
    // Reshape for attention
    int n_tokens = mlx_array_dim(x, 1);
    
    // Reshape Q: [d_model, n_tokens] -> [head_dim, n_heads, n_tokens]
    int q_shape[] = {head_dim_, n_heads_, n_tokens};
    mlx_array q_reshaped;
    check_mlx_result(mlx_reshape(
        &q_reshaped,
        q,
        q_shape, 3,
        stream
    ), "reshape(q)");
    
    // Reshape K: [d_model, n_tokens] -> [head_dim, n_kv_heads, n_tokens]
    int k_shape[] = {head_dim_, n_kv_heads_, n_tokens};
    mlx_array k_reshaped;
    check_mlx_result(mlx_reshape(
        &k_reshaped,
        k,
        k_shape, 3,
        stream
    ), "reshape(k)");
    
    // Reshape V: [d_model, n_tokens] -> [head_dim, n_kv_heads, n_tokens]
    int v_shape[] = {head_dim_, n_kv_heads_, n_tokens};
    mlx_array v_reshaped;
    check_mlx_result(mlx_reshape(
        &v_reshaped,
        v,
        v_shape, 3,
        stream
    ), "reshape(v)");
    
    // Apply rotary position embeddings
    mlx_array q_rotated = mlx_rotary_embedding(q_reshaped, positions, rope_theta_, stream);
    mlx_array k_rotated = mlx_rotary_embedding(k_reshaped, positions, rope_theta_, stream);
    
    // Update KV cache
    int layer_idx = std::stoi(prefix_.substr(prefix_.find_last_of(".") + 1));
    kv_cache->update(layer_idx, k_rotated, v_reshaped, positions, stream);
    
    // Get the full KV sequence from the cache for attention computation
    mlx_array k_cache = kv_cache->k_cache(layer_idx);
    mlx_array v_cache = kv_cache->v_cache(layer_idx);
    
    // Take only the current sequence length from the cache
    int cache_size = kv_cache->current_seq_len();
    
    int cache_start[] = {0, 0, 0};
    int cache_stop[] = {head_dim_, n_kv_heads_, cache_size};
    int cache_strides[] = {1, 1, 1};
    
    mlx_array k_seq, v_seq;
    check_mlx_result(mlx_slice(
        &k_seq,
        k_cache,
        cache_start, 3,
        cache_stop, 3,
        cache_strides, 3,
        stream
    ), "slice(k_cache)");
    
    check_mlx_result(mlx_slice(
        &v_seq,
        v_cache,
        cache_start, 3,
        cache_stop, 3,
        cache_strides, 3,
        stream
    ), "slice(v_cache)");
    
    // If needed, handle grouped-query attention by repeating KV heads
    mlx_array k_seq_repeated = k_seq;
    mlx_array v_seq_repeated = v_seq;
    
    if (n_heads_ > n_kv_heads_) {
        int repeat_factor = n_heads_ / n_kv_heads_;
        
        // Create repeated K
        std::vector<mlx_array> k_repeated;
        for (int i = 0; i < repeat_factor; i++) {
            k_repeated.push_back(k_seq);
        }
        
        int axis = 1; // Repeat along the heads dimension
        check_mlx_result(mlx_concatenate(
            &k_seq_repeated,
            k_repeated.data(), k_repeated.size(),
            axis,
            stream
        ), "concatenate(k_repeated)");
        
        // Create repeated V
        std::vector<mlx_array> v_repeated;
        for (int i = 0; i < repeat_factor; i++) {
            v_repeated.push_back(v_seq);
        }
        
        check_mlx_result(mlx_concatenate(
            &v_seq_repeated,
            v_repeated.data(), v_repeated.size(),
            axis,
            stream
        ), "concatenate(v_repeated)");
    }
    
    // Compute attention
    float scale = 1.0f / sqrtf(head_dim_);
    mlx_array attn_output = mlx_attention(q_rotated, k_seq_repeated, v_seq_repeated, positions, scale, stream);
    
    // Reshape attention output: [head_dim, n_heads, n_tokens] -> [d_model, n_tokens]
    int attn_shape[] = {d_model_, n_tokens};
    mlx_array attn_reshaped;
    check_mlx_result(mlx_reshape(
        &attn_reshaped,
        attn_output,
        attn_shape, 2,
        stream
    ), "reshape(attn_output)");
    
    // Apply output projection
    mlx_array attn_projected;
    check_mlx_result(mlx_matmul(
        &attn_projected,
        attn_reshaped,
        wo_weight_,
        false, // no transpose
        stream
    ), "matmul(attn_reshaped, wo)");
    
    // Add residual connection
    mlx_array attn_residual;
    check_mlx_result(mlx_add(
        &attn_residual,
        attn_projected,
        residual,
        stream
    ), "add(attn_projected, residual)");
    
    // Save for the second residual connection
    residual = attn_residual;
    
    // Apply FFN norm
    mlx_array ffn_normalized;
    check_mlx_result(mlx_layer_norm(
        &ffn_normalized,
        attn_residual,
        ffn_norm_weight_,
        ffn_norm_bias_,
        1e-5, // epsilon
        stream
    ), "layer_norm(attn_residual)");
    
    // Apply feed-forward network
    mlx_array ffn_output = mlx_feed_forward(ffn_normalized, w1_weight_, w2_weight_, w3_weight_, stream);
    
    // Add second residual connection
    mlx_array output;
    check_mlx_result(mlx_add(
        &output,
        ffn_output,
        residual,
        stream
    ), "add(ffn_output, residual)");
    
    // Free temporary tensors
    mlx_array_free(normalized);
    mlx_array_free(q);
    mlx_array_free(k);
    mlx_array_free(v);
    mlx_array_free(q_reshaped);
    mlx_array_free(k_reshaped);
    mlx_array_free(v_reshaped);
    mlx_array_free(q_rotated);
    mlx_array_free(k_rotated);
    mlx_array_free(k_seq);
    mlx_array_free(v_seq);
    if (n_heads_ > n_kv_heads_) {
        mlx_array_free(k_seq_repeated);
        mlx_array_free(v_seq_repeated);
    }
    mlx_array_free(attn_output);
    mlx_array_free(attn_reshaped);
    mlx_array_free(attn_projected);
    mlx_array_free(attn_residual);
    mlx_array_free(ffn_normalized);
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
    
    // Create KV cache with reasonable max sequence length
    int max_seq_len = 2048;
    int head_dim = d_model / n_heads;
    kv_cache_ = std::make_unique<MLXKVCache>(n_layers, n_kv_heads, head_dim, max_seq_len);
    
    // Load embeddings and output weights
    auto it = weights.find("model.tok_embeddings.weight");
    if (it != weights.end()) {
        tok_embeddings_weight_ = it->second;
    } else {
        // Try alternative names
        it = weights.find("tok_embeddings.weight");
        if (it != weights.end()) {
            tok_embeddings_weight_ = it->second;
        } else {
            CCSM_WARNING("Token embedding weights not found");
        }
    }
    
    // Load norm weights
    it = weights.find("model.norm.weight");
    if (it != weights.end()) {
        norm_weight_ = it->second;
    } else {
        it = weights.find("norm.weight");
        if (it != weights.end()) {
            norm_weight_ = it->second;
        } else {
            CCSM_WARNING("Norm weights not found");
        }
    }
    
    it = weights.find("model.norm.bias");
    if (it != weights.end()) {
        norm_bias_ = it->second;
    } else {
        it = weights.find("norm.bias");
        if (it != weights.end()) {
            norm_bias_ = it->second;
        } else {
            CCSM_WARNING("Norm bias not found");
        }
    }
    
    // Load output projection weights
    it = weights.find("lm_head.weight");
    if (it != weights.end()) {
        output_weight_ = it->second;
    } else {
        it = weights.find("output.weight");
        if (it != weights.end()) {
            output_weight_ = it->second;
        } else {
            CCSM_WARNING("Output projection weights not found");
        }
    }
    
    // Create transformer layers
    for (int i = 0; i < n_layers; i++) {
        // Create layer prefix
        std::string prefix = "model.layers." + std::to_string(i);
        
        // Create transformer layer
        layers_.push_back(std::make_unique<MLXTransformerLayer>(
            prefix,
            weights,
            d_model,
            n_heads,
            n_kv_heads,
            rope_theta
        ));
    }
}

MLXTransformer::~MLXTransformer() {
    // Free tensors
    if (tok_embeddings_weight_.ctx) mlx_array_free(tok_embeddings_weight_);
    if (norm_weight_.ctx) mlx_array_free(norm_weight_);
    if (norm_bias_.ctx) mlx_array_free(norm_bias_);
    if (output_weight_.ctx) mlx_array_free(output_weight_);
}

mlx_array MLXTransformer::forward(
    const std::vector<int>& tokens,
    const std::vector<int>& positions,
    MLXKVCache* kv_cache,
    mlx_stream stream) {
    
    // Create input tokens tensor
    int token_shape[] = {(int)tokens.size()};
    mlx_array input_tokens;
    check_mlx_result(mlx_array_new_data(
        &input_tokens,
        tokens.data(),
        token_shape, 1,
        MLX_INT32
    ), "array_new_data(tokens)");
    
    // Embedding lookup
    // Convert to one-hot, shape [tokens.size(), vocab_size]
    int one_hot_shape[] = {(int)tokens.size(), vocab_size_};
    mlx_array one_hot;
    check_mlx_result(mlx_zeros(
        &one_hot,
        one_hot_shape, 2,
        MLX_FLOAT32,
        stream
    ), "zeros(one_hot)");
    
    // Set 1.0 at the indices corresponding to tokens
    for (size_t i = 0; i < tokens.size(); i++) {
        int row = i;
        int col = tokens[i];
        
        if (col >= vocab_size_) {
            // Handle out of bounds tokens
            CCSM_WARNING("Token ", col, " is out of vocabulary bounds (", vocab_size_, ")");
            col = 0; // Use padding token as fallback
        }
        
        // Create a slice for this position
        int start[] = {(int)i, col};
        int stop[] = {(int)i + 1, col + 1};
        int strides[] = {1, 1};
        
        mlx_array slice;
        check_mlx_result(mlx_slice(
            &slice,
            one_hot,
            start, 2,
            stop, 2,
            strides, 2,
            stream
        ), "slice(one_hot)");
        
        // Set to 1.0
        mlx_array one = mlx_array_new_float(1.0f);
        check_mlx_result(mlx_copy_to(
            slice,
            one,
            stream
        ), "copy_to(slice, one)");
        
        mlx_array_free(slice);
        mlx_array_free(one);
    }
    
    // Multiply one-hot with embedding table to get embeddings
    mlx_array embd;
    check_mlx_result(mlx_matmul(
        &embd,
        one_hot,
        tok_embeddings_weight_,
        false, // no transpose
        stream
    ), "matmul(one_hot, tok_embeddings)");
    
    // Transpose to [d_model, tokens.size()]
    mlx_array embd_transposed;
    check_mlx_result(mlx_transpose(
        &embd_transposed,
        embd,
        stream
    ), "transpose(embd)");
    
    // Process through transformer layers
    mlx_array x = embd_transposed;
    for (int i = 0; i < n_layers_; i++) {
        mlx_array layer_output = layers_[i]->forward(x, kv_cache, positions, stream);
        mlx_array_free(x);
        x = layer_output;
    }
    
    // Final normalization
    mlx_array normalized;
    check_mlx_result(mlx_layer_norm(
        &normalized,
        x,
        norm_weight_,
        norm_bias_,
        1e-5, // epsilon
        stream
    ), "layer_norm(x)");
    
    // Take only the last token's embedding
    int last_token_start[] = {0, (int)tokens.size() - 1};
    int last_token_stop[] = {d_model_, (int)tokens.size()};
    int last_token_strides[] = {1, 1};
    
    mlx_array last_token;
    check_mlx_result(mlx_slice(
        &last_token,
        normalized,
        last_token_start, 2,
        last_token_stop, 2,
        last_token_strides, 2,
        stream
    ), "slice(normalized)");
    
    // Reshape to [d_model, 1]
    int last_token_shape[] = {d_model_, 1};
    mlx_array last_token_reshaped;
    check_mlx_result(mlx_reshape(
        &last_token_reshaped,
        last_token,
        last_token_shape, 2,
        stream
    ), "reshape(last_token)");
    
    // Output projection to get logits
    mlx_array logits;
    check_mlx_result(mlx_matmul(
        &logits,
        output_weight_,
        last_token_reshaped,
        false, // no transpose
        stream
    ), "matmul(output_weight, last_token)");
    
    // Free temporary tensors
    mlx_array_free(input_tokens);
    mlx_array_free(one_hot);
    mlx_array_free(embd);
    mlx_array_free(embd_transposed);
    mlx_array_free(x);
    mlx_array_free(normalized);
    mlx_array_free(last_token);
    mlx_array_free(last_token_reshaped);
    
    return logits;
}

void MLXTransformer::reset_caches() {
    kv_cache_->clear();
}

#else // CCSM_WITH_MLX
// Empty implementations for when MLX is not available
#endif // CCSM_WITH_MLX

} // namespace ccsm