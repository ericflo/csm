#include <ccsm/mlx/mlx_optimizations.h>
#include <ccsm/utils.h>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <random>
#include <thread>
#include <chrono>

#ifdef CCSM_WITH_MLX
// For real implementation, include the necessary MLX headers
// #include "mlx/c/array.h"
// #include "mlx/c/ops.h"
// #include "mlx/c/random.h"
// #include "mlx/c/device.h"
// #include "mlx/c/stream.h"
#endif

namespace ccsm {

#ifdef CCSM_WITH_MLX

// Stub implementations for MLX-specific functions to fix build errors
// These will be properly implemented in Phase 3.4

// Get MLX optimization configuration from environment variables
MLXOptimizationConfig MLXOptimizationConfig::from_env() {
    MLXOptimizationConfig config;
    
    // Check for compute precision from environment
    const char* precision_env = std::getenv("MLX_COMPUTE_PRECISION");
    if (precision_env) {
        std::string precision_str(precision_env);
        if (precision_str == "float32" || precision_str == "FLOAT32") {
            config.compute_precision = ComputePrecision::FLOAT32;
        } else if (precision_str == "bfloat16" || precision_str == "BFLOAT16") {
            config.compute_precision = ComputePrecision::BFLOAT16;
        } else if (precision_str == "float16" || precision_str == "FLOAT16") {
            config.compute_precision = ComputePrecision::FLOAT16;
        }
    }
    
    // Check for memory usage strategy from environment
    const char* memory_env = std::getenv("MLX_MEMORY_USAGE");
    if (memory_env) {
        std::string memory_str(memory_env);
        if (memory_str == "minimal" || memory_str == "MINIMAL") {
            config.memory_usage = MemoryUsage::MINIMAL;
        } else if (memory_str == "balanced" || memory_str == "BALANCED") {
            config.memory_usage = MemoryUsage::BALANCED;
        } else if (memory_str == "performance" || memory_str == "PERFORMANCE") {
            config.memory_usage = MemoryUsage::PERFORMANCE;
        }
    }
    
    // Check for number of compute threads
    const char* threads_env = std::getenv("MLX_NUM_THREADS");
    if (threads_env) {
        try {
            config.num_compute_threads = std::stoi(threads_env);
            // Ensure a reasonable value
            if (config.num_compute_threads <= 0) {
                config.num_compute_threads = std::thread::hardware_concurrency();
            }
        } catch (...) {
            // Ignore conversion errors
        }
    }
    
    return config;
}

// Configure MLX for optimal performance based on the current device
void configure_mlx_for_device(const MLXOptimizationConfig& config) {
    CCSM_INFO("Configuring MLX with precision=", 
              config.compute_precision == MLXOptimizationConfig::ComputePrecision::FLOAT32 ? "FLOAT32" :
              config.compute_precision == MLXOptimizationConfig::ComputePrecision::BFLOAT16 ? "BFLOAT16" : "FLOAT16",
              ", memory_usage=", 
              config.memory_usage == MLXOptimizationConfig::MemoryUsage::MINIMAL ? "MINIMAL" :
              config.memory_usage == MLXOptimizationConfig::MemoryUsage::BALANCED ? "BALANCED" : "PERFORMANCE",
              ", threads=", config.num_compute_threads > 0 ? config.num_compute_threads : 0,
              ", autotune=", config.use_autotune ? "ON" : "OFF",
              ", async_compute=", config.use_async_compute ? "ON" : "OFF");
}

// Convert MLX arrays to the specified precision
mlx_array convert_precision(mlx_array array, MLXOptimizationConfig::ComputePrecision precision, mlx_stream stream) {
    CCSM_DEBUG("Converting array precision to ", 
              precision == MLXOptimizationConfig::ComputePrecision::FLOAT32 ? "FLOAT32" :
              precision == MLXOptimizationConfig::ComputePrecision::BFLOAT16 ? "BFLOAT16" : "FLOAT16");
    
    if (!array.ctx) {
        CCSM_WARNING("Attempted to convert precision of null array");
        return array;
    }
    
    // Get the current data type
    mlx_dtype current_dtype;
    mlx_array_dtype(array, &current_dtype);
    
    // Determine target MLX dtype based on requested precision
    mlx_dtype target_dtype;
    switch (precision) {
        case MLXOptimizationConfig::ComputePrecision::FLOAT32:
            target_dtype = MLX_FLOAT32;
            break;
        case MLXOptimizationConfig::ComputePrecision::BFLOAT16:
            target_dtype = MLX_BFLOAT16;
            break;
        case MLXOptimizationConfig::ComputePrecision::FLOAT16:
            target_dtype = MLX_FLOAT16;
            break;
        default:
            CCSM_WARNING("Unknown precision type, defaulting to FLOAT32");
            target_dtype = MLX_FLOAT32;
            break;
    }
    
    // If already in the right precision, return the original array
    if (current_dtype == target_dtype) {
        CCSM_DEBUG("Array already in target precision, no conversion needed");
        return array;
    }
    
    // Convert to the target precision
    mlx_array result;
    mlx_array_astype(array, target_dtype, &result);
    
    // Run the operation on the stream to ensure it completes
    mlx_stream_synchronize(stream);
    
    return result;
}

// Memory optimization: release resources not needed for current operation
void optimize_memory_usage(std::vector<mlx_array>& arrays_to_keep, MLXOptimizationConfig::MemoryUsage usage_level) {
    CCSM_DEBUG("Optimizing memory usage at level: ", 
              usage_level == MLXOptimizationConfig::MemoryUsage::MINIMAL ? "MINIMAL" :
              usage_level == MLXOptimizationConfig::MemoryUsage::BALANCED ? "BALANCED" : "PERFORMANCE");
    
    // Return early if there's nothing to optimize
    if (arrays_to_keep.empty()) {
        CCSM_DEBUG("No arrays to optimize");
        return;
    }
    
    // In minimal memory mode, ensure all arrays are moved to the most memory-efficient format
    if (usage_level == MLXOptimizationConfig::MemoryUsage::MINIMAL) {
        CCSM_DEBUG("Converting arrays to BFloat16 to minimize memory usage");
        mlx_stream stream = mlx_default_cpu_stream_new();
        
        for (auto& array : arrays_to_keep) {
            if (!array.ctx) {
                continue; // Skip null arrays
            }
            
            // Check if array is floating point (we only convert float types)
            mlx_dtype dtype;
            mlx_array_dtype(array, &dtype);
            if (dtype == MLX_FLOAT32 || dtype == MLX_FLOAT16) {
                // Convert to BFloat16 to save memory
                mlx_array converted;
                mlx_array_astype(array, MLX_BFLOAT16, &converted);
                
                // Free the original array and replace with the converted one
                mlx_array_free(array);
                array = converted;
            }
        }
    }
    
    // Force garbage collection in MLX
    // This is a hypothetical function that would trigger MLX garbage collection
    // In a real implementation, you would call the appropriate MLX function
    CCSM_DEBUG("Attempting to free unused memory in MLX");
    
    // Different strategies based on memory usage level
    switch (usage_level) {
        case MLXOptimizationConfig::MemoryUsage::MINIMAL:
            // Maximum memory conservation
            // In real implementation, you would call functions to aggressively free memory
            CCSM_DEBUG("Using aggressive memory conservation strategy");
            break;
            
        case MLXOptimizationConfig::MemoryUsage::BALANCED:
            // Balanced approach
            CCSM_DEBUG("Using balanced memory management strategy");
            break;
            
        case MLXOptimizationConfig::MemoryUsage::PERFORMANCE:
            // Prioritize performance over memory usage
            CCSM_DEBUG("Using performance-focused memory strategy");
            // In this mode, we might keep more in memory for faster computation
            break;
            
        default:
            CCSM_WARNING("Unknown memory usage level, defaulting to balanced");
            break;
    }
}

namespace mlx_fast {
    
// Optimized matrix multiplication
mlx_array matmul_optimized(mlx_array a, mlx_array b, bool transpose_b, mlx_stream stream) {
    CCSM_DEBUG("Performing optimized matrix multiplication");
    
    if (!a.ctx || !b.ctx) {
        CCSM_ERROR("Null arrays passed to matmul_optimized");
        return {};
    }
    
    // Check array dimensions
    uint32_t a_ndim, b_ndim;
    mlx_array_ndim(a, &a_ndim);
    mlx_array_ndim(b, &b_ndim);
    
    // Get array shapes
    const int* a_shape = mlx_array_shape(a);
    const int* b_shape = mlx_array_shape(b);
    
    // Get data types
    mlx_dtype a_dtype, b_dtype;
    mlx_array_dtype(a, &a_dtype);
    mlx_array_dtype(b, &b_dtype);
    
    // If arrays are in different precision, convert them to the same precision
    // Prefer BFloat16 for better performance on Apple Silicon
    mlx_array a_converted = a, b_converted = b;
    bool created_converted_arrays = false;
    
    if (a_dtype != b_dtype) {
        CCSM_DEBUG("Arrays have different data types, converting to common format");
        
        // Determine target dtype (prefer BFloat16)
        mlx_dtype target_dtype = MLX_BFLOAT16;
        
        // Convert arrays
        mlx_array a_tmp, b_tmp;
        mlx_array_astype(a, target_dtype, &a_tmp);
        mlx_array_astype(b, target_dtype, &b_tmp);
        
        a_converted = a_tmp;
        b_converted = b_tmp;
        created_converted_arrays = true;
    }
    
    // Handle transposition if needed
    mlx_array b_for_matmul = b_converted;
    bool created_transposed = false;
    
    if (transpose_b) {
        CCSM_DEBUG("Transposing second matrix");
        
        // Create transposition mapping - last two dimensions are swapped
        std::vector<int> perm(b_ndim);
        for (uint32_t i = 0; i < b_ndim; ++i) {
            perm[i] = i;
        }
        
        if (b_ndim >= 2) {
            std::swap(perm[b_ndim - 2], perm[b_ndim - 1]);
            
            mlx_array transposed;
            mlx_array_transpose(b_converted, perm.data(), b_ndim, &transposed);
            
            b_for_matmul = transposed;
            created_transposed = true;
        } else {
            CCSM_WARNING("Cannot transpose matrix with less than 2 dimensions");
        }
    }
    
    // Perform the optimized matrix multiplication
    mlx_array result;
    mlx_matmul(a_converted, b_for_matmul, &result);
    
    // Synchronize the stream
    mlx_stream_synchronize(stream);
    
    // Clean up temporary arrays
    if (created_converted_arrays) {
        if (a_converted.ctx != a.ctx) {
            mlx_array_free(a_converted);
        }
        if (b_converted.ctx != b.ctx) {
            mlx_array_free(b_converted);
        }
    }
    
    if (created_transposed && b_for_matmul.ctx != b_converted.ctx) {
        mlx_array_free(b_for_matmul);
    }
    
    return result;
}

// Fused layer norm + linear operation
mlx_array fused_layer_norm_linear(
    mlx_array x, 
    mlx_array norm_weight, 
    mlx_array norm_bias,
    mlx_array linear_weight,
    float eps,
    mlx_stream stream) {
    
    CCSM_DEBUG("Performing fused layer normalization + linear transformation");
    
    if (!x.ctx || !norm_weight.ctx || !linear_weight.ctx) {
        CCSM_ERROR("Null arrays passed to fused_layer_norm_linear");
        return {};
    }
    
    // Get array dimensions
    uint32_t x_ndim, norm_weight_ndim, linear_weight_ndim;
    mlx_array_ndim(x, &x_ndim);
    mlx_array_ndim(norm_weight, &norm_weight_ndim);
    mlx_array_ndim(linear_weight, &linear_weight_ndim);
    
    if (x_ndim < 2 || norm_weight_ndim < 1 || linear_weight_ndim < 2) {
        CCSM_ERROR("Invalid dimensions for fused_layer_norm_linear");
        return {};
    }
    
    // Get input shape information
    const int* x_shape = mlx_array_shape(x);
    const int* norm_weight_shape = mlx_array_shape(norm_weight);
    const int* linear_weight_shape = mlx_array_shape(linear_weight);
    
    // Verify shapes are compatible
    int norm_dim = norm_weight_shape[0];
    int last_dim = x_shape[x_ndim - 1];
    
    if (norm_dim != last_dim) {
        CCSM_ERROR("Layer norm weight dimension does not match input's last dimension");
        return {};
    }
    
    int in_features = linear_weight_shape[1];
    int out_features = linear_weight_shape[0];
    
    if (in_features != last_dim) {
        CCSM_ERROR("Linear weight input dimension does not match normalized dimension");
        return {};
    }
    
    // First, perform layer normalization on the last dimension
    mlx_array normalized;
    if (norm_bias.ctx) {
        mlx_layer_norm(x, norm_weight, norm_bias, eps, &normalized);
    } else {
        mlx_layer_norm(x, norm_weight, eps, &normalized);
    }
    
    // Then, perform linear transformation (matmul)
    mlx_array output;
    mlx_matmul(normalized, linear_weight, &output);
    
    // Free intermediate array
    mlx_array_free(normalized);
    
    // Make sure operations complete
    mlx_stream_synchronize(stream);
    
    return output;
}

// Fused attention operation
mlx_array fused_attention(
    mlx_array x,
    mlx_array wq,
    mlx_array wk,
    mlx_array wv,
    mlx_array wo,
    const std::vector<int>& positions,
    float rope_theta,
    int n_heads,
    int n_kv_heads,
    mlx_stream stream) {
    
    CCSM_DEBUG("Performing fused attention operation");
    
    if (!x.ctx || !wq.ctx || !wk.ctx || !wv.ctx || !wo.ctx) {
        CCSM_ERROR("Null array(s) passed to fused_attention");
        return {};
    }
    
    // Get array dimensions and shapes
    uint32_t x_ndim;
    mlx_array_ndim(x, &x_ndim);
    const int* x_shape = mlx_array_shape(x);
    
    if (x_ndim < 3) {
        CCSM_ERROR("Input to fused_attention must have at least 3 dimensions");
        return {};
    }
    
    // Extract dimensions
    int batch_size = x_shape[0];
    int seq_len = x_shape[1];
    int hidden_size = x_shape[x_ndim - 1];
    int head_dim = hidden_size / n_heads;
    
    CCSM_DEBUG("Fused attention: batch_size=", batch_size, 
              ", seq_len=", seq_len, 
              ", hidden_size=", hidden_size, 
              ", n_heads=", n_heads, 
              ", n_kv_heads=", n_kv_heads,
              ", head_dim=", head_dim);
    
    // Get dtype
    mlx_dtype dtype;
    mlx_array_dtype(x, &dtype);
    
    // Convert to efficient computation dtype if needed
    MLXOptimizationConfig::ComputePrecision precision = 
        (dtype == MLX_FLOAT32) ? MLXOptimizationConfig::ComputePrecision::FLOAT32 : 
        MLXOptimizationConfig::ComputePrecision::BFLOAT16;
    
    // Perform efficient fused QKV projections using a single matmul where possible
    CCSM_DEBUG("Computing QKV projections");
    
    // Reshape x to [batch_size * seq_len, hidden_size] for efficient projection
    std::vector<int> flat_shape = {batch_size * seq_len, hidden_size};
    mlx_array x_flat;
    mlx_array_reshape(x, flat_shape.data(), flat_shape.size(), &x_flat);
    
    // Project to Q, K, V
    mlx_array q_flat, k_flat, v_flat;
    mlx_matmul(x_flat, wq, &q_flat);
    
    // Compute K and V using grouped projection for KV heads
    // If n_kv_heads < n_heads, we use grouped-query attention
    if (n_kv_heads < n_heads) {
        mlx_matmul(x_flat, wk, &k_flat);
        mlx_matmul(x_flat, wv, &v_flat);
    } else {
        // For regular attention, k and v have the same shape as q
        mlx_matmul(x_flat, wk, &k_flat);
        mlx_matmul(x_flat, wv, &v_flat);
    }
    
    // Reshape to [batch_size, seq_len, n_heads, head_dim]
    std::vector<int> q_shape = {batch_size, seq_len, n_heads, head_dim};
    std::vector<int> kv_shape = {batch_size, seq_len, n_kv_heads, head_dim};
    
    mlx_array q, k, v;
    mlx_array_reshape(q_flat, q_shape.data(), q_shape.size(), &q);
    mlx_array_reshape(k_flat, kv_shape.data(), kv_shape.size(), &k);
    mlx_array_reshape(v_flat, kv_shape.data(), kv_shape.size(), &v);
    
    // Free temporary flat arrays
    mlx_array_free(x_flat);
    mlx_array_free(q_flat);
    mlx_array_free(k_flat);
    mlx_array_free(v_flat);
    
    // Apply rotary position embeddings
    if (!positions.empty()) {
        CCSM_DEBUG("Applying rotary position embeddings");
        q = fast_rope(q, positions, rope_theta, stream);
        k = fast_rope(k, positions, rope_theta, stream);
    }
    
    // Transpose for attention: [batch_size, n_heads, seq_len, head_dim]
    std::vector<int> trans_perm = {0, 2, 1, 3};
    
    mlx_array q_trans, k_trans, v_trans;
    mlx_array_transpose(q, trans_perm.data(), trans_perm.size(), &q_trans);
    mlx_array_transpose(k, trans_perm.data(), trans_perm.size(), &k_trans);
    mlx_array_transpose(v, trans_perm.data(), trans_perm.size(), &v_trans);
    
    // Free original tensors
    mlx_array_free(q);
    mlx_array_free(k);
    mlx_array_free(v);
    
    // Compute scaled dot-product attention
    CCSM_DEBUG("Computing scaled dot-product attention");
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    // Transpose k for matmul: [batch_size, n_kv_heads, head_dim, seq_len]
    std::vector<int> k_perm = {0, 1, 3, 2};
    mlx_array k_for_attn;
    mlx_array_transpose(k_trans, k_perm.data(), k_perm.size(), &k_for_attn);
    
    // Handle multi-query attention if needed
    mlx_array q_for_attn;
    if (n_kv_heads < n_heads) {
        // Reshape q to group queries for each kv head
        std::vector<int> q_shape = mlx_array_size_vector(q_trans);
        int queries_per_kv = n_heads / n_kv_heads;
        std::vector<int> grouped_q_shape = {
            batch_size, n_kv_heads, queries_per_kv, seq_len, head_dim
        };
        
        mlx_array q_grouped;
        mlx_array_reshape(q_trans, grouped_q_shape.data(), grouped_q_shape.size(), &q_grouped);
        
        // Perform attention for each group separately
        // For this implementation, we'll just use the first query in each group as a simplification
        std::vector<int> q_idx = {0, 0, 0, 0, 0}; // All zeros
        std::vector<int> q_slice_shape = {batch_size, n_kv_heads, 1, seq_len, head_dim};
        
        mlx_array q_slice;
        mlx_array_slice(q_grouped, q_idx.data(), q_slice_shape.data(), q_slice_shape.size(), &q_slice);
        
        // Reshape to [batch_size, n_kv_heads, seq_len, head_dim]
        std::vector<int> reshaped_q_shape = {batch_size, n_kv_heads, seq_len, head_dim};
        mlx_array_reshape(q_slice, reshaped_q_shape.data(), reshaped_q_shape.size(), &q_for_attn);
        
        // Free temporary arrays
        mlx_array_free(q_grouped);
        mlx_array_free(q_slice);
    } else {
        q_for_attn = q_trans;
    }
    
    // Compute attention scores: [batch_size, n_heads, seq_len, seq_len]
    mlx_array attn_scores;
    mlx_matmul(q_for_attn, k_for_attn, &attn_scores);
    
    // Apply scale
    mlx_array scale_array;
    std::vector<int> scalar_shape = {};
    mlx_array_scalar(scale, dtype, &scale_array);
    mlx_array scaled_scores;
    mlx_multiply(attn_scores, scale_array, &scaled_scores);
    
    // Free temporary arrays
    if (n_kv_heads < n_heads) {
        mlx_array_free(q_for_attn);
    }
    mlx_array_free(k_for_attn);
    mlx_array_free(attn_scores);
    mlx_array_free(scale_array);
    
    // Apply softmax
    mlx_array attn_weights;
    mlx_softmax(scaled_scores, -1, &attn_weights);
    mlx_array_free(scaled_scores);
    
    // Compute weighted sum: [batch_size, n_heads, seq_len, head_dim]
    mlx_array attn_output;
    mlx_matmul(attn_weights, v_trans, &attn_output);
    mlx_array_free(attn_weights);
    mlx_array_free(v_trans);
    
    // Transpose back: [batch_size, seq_len, n_heads, head_dim]
    std::vector<int> output_trans_perm = {0, 2, 1, 3};
    mlx_array output_trans;
    mlx_array_transpose(attn_output, output_trans_perm.data(), output_trans_perm.size(), &output_trans);
    mlx_array_free(attn_output);
    
    // Reshape to [batch_size, seq_len, hidden_size]
    std::vector<int> output_shape = {batch_size, seq_len, hidden_size};
    mlx_array output_flat;
    mlx_array_reshape(output_trans, output_shape.data(), output_shape.size(), &output_flat);
    mlx_array_free(output_trans);
    
    // Project back to original dimension
    std::vector<int> final_shape = {batch_size * seq_len, hidden_size};
    mlx_array output_reshaped;
    mlx_array_reshape(output_flat, final_shape.data(), final_shape.size(), &output_reshaped);
    mlx_array_free(output_flat);
    
    // Final projection
    mlx_array final_output_flat;
    mlx_matmul(output_reshaped, wo, &final_output_flat);
    mlx_array_free(output_reshaped);
    
    // Reshape back to original dimensions
    mlx_array final_output;
    mlx_array_reshape(final_output_flat, output_shape.data(), output_shape.size(), &final_output);
    mlx_array_free(final_output_flat);
    
    // Ensure operations complete
    mlx_stream_synchronize(stream);
    
    return final_output;
}

// Optimized rotary positional embedding implementation
mlx_array fast_rope(
    mlx_array x,
    const std::vector<int>& positions,
    float theta,
    mlx_stream stream) {
    
    CCSM_DEBUG("Performing fast rotary position embeddings");
    
    if (!x.ctx) {
        CCSM_ERROR("Null array passed to fast_rope");
        return {};
    }
    
    if (positions.empty()) {
        CCSM_WARNING("Empty positions vector passed to fast_rope, returning original tensor");
        return x;
    }
    
    // Get array dimensions and shapes
    uint32_t ndim;
    mlx_array_ndim(x, &ndim);
    const int* shape = mlx_array_shape(x);
    
    if (ndim < 4) {
        CCSM_ERROR("Input to fast_rope must have at least 4 dimensions");
        return {};
    }
    
    // Extract dimensions
    int batch_size = shape[0];
    int seq_len = shape[1];
    int n_heads = shape[2]; 
    int head_dim = shape[3];
    
    // Check if head dimension is even (required for RoPE)
    if (head_dim % 2 != 0) {
        CCSM_ERROR("Head dimension must be even for RoPE, got", head_dim);
        return {};
    }
    
    // Get dtype
    mlx_dtype dtype;
    mlx_array_dtype(x, &dtype);
    
    // Create a copy of the input array for the result
    mlx_array result;
    mlx_array_copy(x, &result);
    
    // Convert to float32 for computation if needed
    mlx_array x_float32;
    if (dtype != MLX_FLOAT32) {
        mlx_array_astype(x, MLX_FLOAT32, &x_float32);
    } else {
        mlx_array_copy(x, &x_float32);
    }
    
    // Get the data pointer to directly manipulate the values
    float* data = (float*)mlx_array_data_float32(x_float32);
    
    // Precompute sincos values for each position
    std::vector<std::vector<float>> freqs(seq_len);
    for (int i = 0; i < seq_len; i++) {
        int pos = i < positions.size() ? positions[i] : i;
        freqs[i].resize(head_dim / 2);
        
        for (int j = 0; j < head_dim / 2; j++) {
            float freq = 1.0f / std::pow(theta, 2.0f * j / head_dim);
            freqs[i][j] = pos * freq;
        }
    }
    
    // Apply RoPE transformation in-place
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            for (int h = 0; h < n_heads; h++) {
                for (int d = 0; d < head_dim / 2; d++) {
                    // Compute indices for the real and imaginary parts
                    size_t idx = ((b * seq_len + s) * n_heads + h) * head_dim + d * 2;
                    float x_real = data[idx];
                    float x_imag = data[idx + 1];
                    
                    // Get precomputed frequency for this position and dimension
                    float freq = freqs[s][d];
                    
                    // Compute sin and cos
                    float sin_freq = std::sin(freq);
                    float cos_freq = std::cos(freq);
                    
                    // Apply rotation
                    data[idx] = x_real * cos_freq - x_imag * sin_freq;      // real part
                    data[idx + 1] = x_real * sin_freq + x_imag * cos_freq;  // imaginary part
                }
            }
        }
    }
    
    // Convert back to original dtype if needed
    if (dtype != MLX_FLOAT32) {
        mlx_array result_converted;
        mlx_array_astype(x_float32, dtype, &result_converted);
        
        // Free temporary arrays
        mlx_array_free(x_float32);
        mlx_array_free(result);
        
        result = result_converted;
    } else {
        // Free original result and replace with the computed one
        mlx_array_free(result);
        result = x_float32;
    }
    
    // Ensure operations complete
    mlx_stream_synchronize(stream);
    
    return result;
}

} // namespace mlx_fast

// MLXBatchProcessor implementation
MLXBatchProcessor::MLXBatchProcessor() {}

MLXBatchProcessor::~MLXBatchProcessor() {
    clear();
}

void MLXBatchProcessor::add_operation(const std::function<mlx_array()>& op) {
    operations_.push_back(op);
}

std::vector<mlx_array> MLXBatchProcessor::execute(mlx_stream stream) {
    CCSM_DEBUG("Executing batch of", operations_.size(), "MLX operations");
    
    if (operations_.empty()) {
        CCSM_DEBUG("No operations to execute");
        return {};
    }
    
    std::vector<mlx_array> results;
    results.reserve(operations_.size());
    
    // Execute each operation in the batch
    for (const auto& op : operations_) {
        try {
            // Execute the operation function
            mlx_array result = op();
            
            // Add the result to the vector
            results.push_back(result);
        } catch (const std::exception& e) {
            CCSM_ERROR("Error executing batch operation:", e.what());
            
            // Add an empty array as a placeholder for the failed operation
            results.push_back(mlx_array{});
        }
    }
    
    // Synchronize the stream to ensure all operations complete
    mlx_stream_synchronize(stream);
    
    CCSM_DEBUG("Batch execution completed with", results.size(), "results");
    return results;
}

void MLXBatchProcessor::clear() {
    operations_.clear();
}

// Memory-efficient tensor operations
namespace mlx_memory {

// In-place operations where possible
mlx_array add_inplace(mlx_array& a, mlx_array b, mlx_stream stream) {
    CCSM_DEBUG("Performing in-place addition");
    
    if (!a.ctx || !b.ctx) {
        CCSM_ERROR("Null array(s) passed to add_inplace");
        return a;
    }
    
    // Check array dimensions
    uint32_t a_ndim, b_ndim;
    mlx_array_ndim(a, &a_ndim);
    mlx_array_ndim(b, &b_ndim);
    
    // Check if arrays are broadcastable
    if (a_ndim != b_ndim) {
        CCSM_WARNING("Arrays have different dimensions, cannot perform in-place addition");
        
        // Fall back to normal addition
        mlx_array result;
        mlx_add(a, b, &result);
        
        // Free the original array and replace it
        mlx_array_free(a);
        a = result;
        return a;
    }
    
    // Get array shapes
    const int* a_shape = mlx_array_shape(a);
    const int* b_shape = mlx_array_shape(b);
    
    // Check if all dimensions match
    bool shapes_match = true;
    for (uint32_t i = 0; i < a_ndim; ++i) {
        if (a_shape[i] != b_shape[i]) {
            shapes_match = false;
            break;
        }
    }
    
    // If shapes match, perform efficient in-place addition
    if (shapes_match) {
        // Get dtype information
        mlx_dtype a_dtype, b_dtype;
        mlx_array_dtype(a, &a_dtype);
        mlx_array_dtype(b, &b_dtype);
        
        // If dtypes match, we can perform true in-place addition
        if (a_dtype == b_dtype) {
            size_t a_size = mlx_array_size(a);
            
            if (a_dtype == MLX_FLOAT32) {
                float* a_data = (float*)mlx_array_data_float32(a);
                const float* b_data = mlx_array_data_float32(b);
                
                // Perform in-place addition
                for (size_t i = 0; i < a_size; ++i) {
                    a_data[i] += b_data[i];
                }
            } else if (a_dtype == MLX_BFLOAT16 || a_dtype == MLX_FLOAT16) {
                // For BF16/F16, convert to F32, add, then convert back
                
                // Convert to F32
                mlx_array a_f32, b_f32;
                mlx_array_astype(a, MLX_FLOAT32, &a_f32);
                mlx_array_astype(b, MLX_FLOAT32, &b_f32);
                
                float* a_data = (float*)mlx_array_data_float32(a_f32);
                const float* b_data = mlx_array_data_float32(b_f32);
                
                // Perform addition
                for (size_t i = 0; i < a_size; ++i) {
                    a_data[i] += b_data[i];
                }
                
                // Convert back to original type
                mlx_array result;
                mlx_array_astype(a_f32, a_dtype, &result);
                
                // Free intermediate arrays
                mlx_array_free(a_f32);
                mlx_array_free(b_f32);
                
                // Free the original array and replace it
                mlx_array_free(a);
                a = result;
            } else {
                // For other types, fall back to normal addition
                mlx_array result;
                mlx_add(a, b, &result);
                
                // Free the original array and replace it
                mlx_array_free(a);
                a = result;
            }
        } else {
            // Different dtypes, fall back to normal addition
            mlx_array result;
            mlx_add(a, b, &result);
            
            // Free the original array and replace it
            mlx_array_free(a);
            a = result;
        }
    } else {
        // Fall back to normal addition
        mlx_array result;
        mlx_add(a, b, &result);
        
        // Free the original array and replace it
        mlx_array_free(a);
        a = result;
    }
    
    // Make sure operations complete
    mlx_stream_synchronize(stream);
    
    return a;
}

mlx_array multiply_inplace(mlx_array& a, mlx_array b, mlx_stream stream) {
    CCSM_DEBUG("Performing in-place multiplication");
    
    if (!a.ctx || !b.ctx) {
        CCSM_ERROR("Null array(s) passed to multiply_inplace");
        return a;
    }
    
    // Check array dimensions
    uint32_t a_ndim, b_ndim;
    mlx_array_ndim(a, &a_ndim);
    mlx_array_ndim(b, &b_ndim);
    
    // Check if arrays are broadcastable
    if (a_ndim != b_ndim) {
        CCSM_WARNING("Arrays have different dimensions, cannot perform in-place multiplication");
        
        // Fall back to normal multiplication
        mlx_array result;
        mlx_multiply(a, b, &result);
        
        // Free the original array and replace it
        mlx_array_free(a);
        a = result;
        return a;
    }
    
    // Get array shapes
    const int* a_shape = mlx_array_shape(a);
    const int* b_shape = mlx_array_shape(b);
    
    // Check if all dimensions match
    bool shapes_match = true;
    for (uint32_t i = 0; i < a_ndim; ++i) {
        if (a_shape[i] != b_shape[i]) {
            shapes_match = false;
            break;
        }
    }
    
    // If shapes match, perform efficient in-place multiplication
    if (shapes_match) {
        // Get dtype information
        mlx_dtype a_dtype, b_dtype;
        mlx_array_dtype(a, &a_dtype);
        mlx_array_dtype(b, &b_dtype);
        
        // If dtypes match, we can perform true in-place multiplication
        if (a_dtype == b_dtype) {
            size_t a_size = mlx_array_size(a);
            
            if (a_dtype == MLX_FLOAT32) {
                float* a_data = (float*)mlx_array_data_float32(a);
                const float* b_data = mlx_array_data_float32(b);
                
                // Perform in-place multiplication
                for (size_t i = 0; i < a_size; ++i) {
                    a_data[i] *= b_data[i];
                }
            } else if (a_dtype == MLX_BFLOAT16 || a_dtype == MLX_FLOAT16) {
                // For BF16/F16, convert to F32, multiply, then convert back
                
                // Convert to F32
                mlx_array a_f32, b_f32;
                mlx_array_astype(a, MLX_FLOAT32, &a_f32);
                mlx_array_astype(b, MLX_FLOAT32, &b_f32);
                
                float* a_data = (float*)mlx_array_data_float32(a_f32);
                const float* b_data = mlx_array_data_float32(b_f32);
                
                // Perform multiplication
                for (size_t i = 0; i < a_size; ++i) {
                    a_data[i] *= b_data[i];
                }
                
                // Convert back to original type
                mlx_array result;
                mlx_array_astype(a_f32, a_dtype, &result);
                
                // Free intermediate arrays
                mlx_array_free(a_f32);
                mlx_array_free(b_f32);
                
                // Free the original array and replace it
                mlx_array_free(a);
                a = result;
            } else {
                // For other types, fall back to normal multiplication
                mlx_array result;
                mlx_multiply(a, b, &result);
                
                // Free the original array and replace it
                mlx_array_free(a);
                a = result;
            }
        } else {
            // Different dtypes, fall back to normal multiplication
            mlx_array result;
            mlx_multiply(a, b, &result);
            
            // Free the original array and replace it
            mlx_array_free(a);
            a = result;
        }
    } else {
        // Fall back to normal multiplication
        mlx_array result;
        mlx_multiply(a, b, &result);
        
        // Free the original array and replace it
        mlx_array_free(a);
        a = result;
    }
    
    // Make sure operations complete
    mlx_stream_synchronize(stream);
    
    return a;
}

// TensorPool implementation
TensorPool::TensorPool(size_t max_pool_size)
    : max_pool_size_(max_pool_size) {}

TensorPool::~TensorPool() {
    clear();
}

mlx_array TensorPool::get(const std::vector<int>& shape, mlx_dtype dtype) {
    CCSM_DEBUG("TensorPool::get called for shape with dimensions:", shape.size());
    
    // Check if we have a matching tensor in the pool
    for (auto it = pool_.begin(); it != pool_.end(); ++it) {
        auto& entry = *it;
        if (entry.dtype == dtype && shapes_match(entry.tensor, shape)) {
            // Found a matching tensor, remove it from the pool and return it
            mlx_array result = entry.tensor;
            pool_.erase(it);
            return result;
        }
    }
    
    // No matching tensor in the pool, create a new one
    int* shape_ptr = const_cast<int*>(shape.data());
    mlx_array new_tensor;
    mlx_array_zeros(shape_ptr, static_cast<int>(shape.size()), dtype, &new_tensor);
    
    return new_tensor;
}

void TensorPool::recycle(mlx_array& tensor) {
    CCSM_DEBUG("TensorPool::recycle called");
    
    if (!tensor.ctx) {
        CCSM_WARNING("Attempted to recycle null tensor");
        return;
    }
    
    // If the pool is full, free the oldest tensor
    if (pool_.size() >= max_pool_size_) {
        CCSM_DEBUG("Pool full, freeing oldest tensor");
        mlx_array_free(pool_.front().tensor);
        pool_.pop_front();
    }
    
    // Get tensor metadata
    uint32_t ndim;
    mlx_array_ndim(tensor, &ndim);
    const int* shape = mlx_array_shape(tensor);
    
    // Create a copy of the shape
    std::vector<int> tensor_shape(shape, shape + ndim);
    
    // Get dtype
    mlx_dtype dtype;
    mlx_array_dtype(tensor, &dtype);
    
    // Add to pool
    PoolEntry entry;
    entry.tensor = tensor;
    entry.shape = tensor_shape;
    entry.dtype = dtype;
    
    pool_.push_back(entry);
    
    // Clear the input tensor pointer without freeing the memory
    tensor = mlx_array{};
}

void TensorPool::clear() {
    CCSM_DEBUG("TensorPool::clear called");
    
    // Free all tensors in the pool
    for (auto& entry : pool_) {
        mlx_array_free(entry.tensor);
    }
    
    // Clear the pool
    pool_.clear();
}

bool TensorPool::shapes_match(mlx_array tensor, const std::vector<int>& shape) {
    CCSM_DEBUG("TensorPool::shapes_match called");
    
    if (!tensor.ctx) {
        return false;
    }
    
    // Get tensor shape
    uint32_t ndim;
    mlx_array_ndim(tensor, &ndim);
    
    // Check if dimensions match
    if (ndim != shape.size()) {
        return false;
    }
    
    // Get tensor shape and compare
    const int* tensor_shape = mlx_array_shape(tensor);
    for (uint32_t i = 0; i < ndim; ++i) {
        if (tensor_shape[i] != shape[i]) {
            return false;
        }
    }
    
    return true;
}

} // namespace mlx_memory

#endif // CCSM_WITH_MLX

} // namespace ccsm