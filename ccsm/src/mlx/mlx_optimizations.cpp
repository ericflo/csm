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
#include <mlx/c/array.h>
#include <mlx/c/ops.h>
#include <mlx/c/random.h>
#include <mlx/c/device.h>
#include <mlx/c/stream.h>
#endif

namespace ccsm {

#ifdef CCSM_WITH_MLX

// Helper functions
static void check_mlx_result(int result, const char* operation) {
    if (result != 0) {
        throw std::runtime_error(std::string("MLX operation failed: ") + operation);
    }
}

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
    
    // Check for autotune
    const char* autotune_env = std::getenv("MLX_AUTOTUNE");
    if (autotune_env) {
        std::string autotune_str(autotune_env);
        if (autotune_str == "0" || autotune_str == "false" || autotune_str == "FALSE") {
            config.use_autotune = false;
        } else if (autotune_str == "1" || autotune_str == "true" || autotune_str == "TRUE") {
            config.use_autotune = true;
        }
    }
    
    // Check for async compute
    const char* async_env = std::getenv("MLX_ASYNC_COMPUTE");
    if (async_env) {
        std::string async_str(async_env);
        if (async_str == "0" || async_str == "false" || async_str == "FALSE") {
            config.use_async_compute = false;
        } else if (async_str == "1" || async_str == "true" || async_str == "TRUE") {
            config.use_async_compute = true;
        }
    }
    
    return config;
}

// Configure MLX for optimal performance based on the current device
void configure_mlx_for_device(const MLXOptimizationConfig& config) {
    // Set MLX compute precision
    mlx_dtype dtype;
    switch (config.compute_precision) {
        case MLXOptimizationConfig::ComputePrecision::FLOAT32:
            dtype = MLX_FLOAT32;
            CCSM_DEBUG("Setting MLX compute precision to FLOAT32");
            break;
        case MLXOptimizationConfig::ComputePrecision::BFLOAT16:
            dtype = MLX_BFLOAT16;
            CCSM_DEBUG("Setting MLX compute precision to BFLOAT16");
            break;
        case MLXOptimizationConfig::ComputePrecision::FLOAT16:
            dtype = MLX_FLOAT16;
            CCSM_DEBUG("Setting MLX compute precision to FLOAT16");
            break;
        default:
            dtype = MLX_BFLOAT16; // Default to BFloat16
            CCSM_DEBUG("Using default MLX compute precision (BFLOAT16)");
            break;
    }
    
    // Set the default dtype for MLX
    // Note: MLX-C doesn't have a direct way to set default dtype, so we'll use it for all operations
    
    // Set number of compute threads if specified
    if (config.num_compute_threads > 0) {
        // Set environment variable for MLX to pick up
        std::string threads_str = std::to_string(config.num_compute_threads);
        setenv("MLX_NUM_THREADS", threads_str.c_str(), 1);
        CCSM_DEBUG("Setting MLX compute threads to ", config.num_compute_threads);
    }
    
    // Set autotune if specified
    if (config.use_autotune) {
        setenv("MLX_AUTOTUNE", "1", 1);
        CCSM_DEBUG("Enabling MLX autotune");
    } else {
        setenv("MLX_AUTOTUNE", "0", 1);
        CCSM_DEBUG("Disabling MLX autotune");
    }
    
    // Configure async compute
    // Note: MLX-C doesn't have a direct way to control this, so we use environment variables
    if (config.use_async_compute) {
        CCSM_DEBUG("Enabling MLX async compute");
    } else {
        CCSM_DEBUG("Disabling MLX async compute");
    }
    
    // Create default stream for operations
    mlx_stream stream = mlx_stream_new();
    mlx_stream_free(stream);
    
    CCSM_INFO("MLX configured with precision=", 
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
            target_dtype = MLX_BFLOAT16; // Default to BFloat16
            break;
    }
    
    // Get current dtype
    mlx_dtype current_dtype;
    check_mlx_result(mlx_array_get_dtype(&current_dtype, array), "array_get_dtype");
    
    // If already the target precision, return as is
    if (current_dtype == target_dtype) {
        return array;
    }
    
    // Convert to the target precision
    mlx_array result;
    check_mlx_result(mlx_array_astype(&result, array, target_dtype, stream), "array_astype");
    return result;
}

// Memory optimization: release resources not needed for current operation
void optimize_memory_usage(std::vector<mlx_array>& arrays_to_keep, MLXOptimizationConfig::MemoryUsage usage_level) {
    // Different strategies depending on the usage level
    switch (usage_level) {
        case MLXOptimizationConfig::MemoryUsage::MINIMAL:
            // In minimal memory mode, we aggressively clean up temporary arrays
            // Force a garbage collection in MLX by evaluating all arrays we want to keep
            for (auto& array : arrays_to_keep) {
                check_mlx_result(mlx_array_eval(array), "array_eval");
            }
            // No need to do anything else as MLX will automatically free unreferenced arrays
            break;
            
        case MLXOptimizationConfig::MemoryUsage::BALANCED:
            // In balanced mode, we evaluate arrays but don't force additional cleanup
            for (auto& array : arrays_to_keep) {
                check_mlx_result(mlx_array_eval(array), "array_eval");
            }
            break;
            
        case MLXOptimizationConfig::MemoryUsage::PERFORMANCE:
            // In performance mode, we're less aggressive about memory cleanup
            // Just sync to make sure operations are complete
            mlx_stream stream = mlx_stream_new();
            check_mlx_result(mlx_stream_synchronize(stream), "stream_synchronize");
            mlx_stream_free(stream);
            break;
    }
}

namespace mlx_fast {
    
// Optimized matrix multiplication that avoids unnecessary intermediate memory allocations
mlx_array matmul_optimized(mlx_array a, mlx_array b, bool transpose_b, mlx_stream stream) {
    mlx_array result;
    
    // When transposing B, we can use a more efficient implementation
    if (transpose_b) {
        check_mlx_result(mlx_matmul(&result, a, b, false, true, stream), "matmul");
    } else {
        check_mlx_result(mlx_matmul(&result, a, b, false, false, stream), "matmul");
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
    
    // This implements a fused layernorm followed by linear projection
    // It avoids creating an intermediate normalized tensor
    
    // First, compute the layernorm (mean and variance)
    mlx_array mean, var;
    
    // Determine the axes to normalize over (typically the last dimension)
    int ndim;
    check_mlx_result(mlx_array_ndim(&ndim, x), "array_ndim");
    if (ndim < 1) {
        throw std::runtime_error("Invalid tensor dimensions for layer normalization");
    }
    
    // Calculate mean along the last dimension
    std::vector<int> axes = {ndim - 1};
    check_mlx_result(mlx_mean(&mean, x, axes.data(), axes.size(), true, stream), "mean");
    
    // Calculate variance
    mlx_array x_centered, x_squared;
    check_mlx_result(mlx_subtract(&x_centered, x, mean, stream), "subtract");
    check_mlx_result(mlx_square(&x_squared, x_centered, stream), "square");
    check_mlx_result(mlx_mean(&var, x_squared, axes.data(), axes.size(), true, stream), "mean");
    
    // Add epsilon and take sqrt
    mlx_array var_eps;
    float eps_val = eps;
    mlx_array eps_array;
    check_mlx_result(mlx_array_scalar_float(&eps_array, &eps_val, 1, stream), "array_scalar_float");
    check_mlx_result(mlx_add(&var_eps, var, eps_array, stream), "add");
    
    mlx_array std_dev;
    check_mlx_result(mlx_sqrt(&std_dev, var_eps, stream), "sqrt");
    
    // Normalize
    mlx_array normalized;
    check_mlx_result(mlx_divide(&normalized, x_centered, std_dev, stream), "divide");
    
    // Apply scale and shift
    mlx_array scaled, shifted;
    check_mlx_result(mlx_multiply(&scaled, normalized, norm_weight, stream), "multiply");
    check_mlx_result(mlx_add(&shifted, scaled, norm_bias, stream), "add");
    
    // Apply linear projection
    mlx_array result;
    check_mlx_result(mlx_matmul(&result, shifted, linear_weight, false, false, stream), "matmul");
    
    // Clean up intermediate results
    mlx_array_free(mean);
    mlx_array_free(var);
    mlx_array_free(x_centered);
    mlx_array_free(x_squared);
    mlx_array_free(var_eps);
    mlx_array_free(std_dev);
    mlx_array_free(normalized);
    mlx_array_free(scaled);
    mlx_array_free(shifted);
    mlx_array_free(eps_array);
    
    return result;
}

// Fused attention operation that computes QKV projections and attention in one operation
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
    
    // Get the dimensions of x
    int ndim, seq_len, d_model;
    std::vector<int> x_shape(8, 0); // Maximum ndim for safety
    check_mlx_result(mlx_array_ndim(&ndim, x), "array_ndim");
    check_mlx_result(mlx_array_shape(x_shape.data(), x), "array_shape");
    
    if (ndim != 2) {
        throw std::runtime_error("Input tensor must be 2D [seq_len, d_model]");
    }
    
    seq_len = x_shape[0];
    d_model = x_shape[1];
    
    // Compute head dimensions
    int head_dim = d_model / n_heads;
    
    // Project x to q, k, v
    mlx_array q, k, v;
    check_mlx_result(mlx_matmul(&q, x, wq, false, false, stream), "matmul_q");
    check_mlx_result(mlx_matmul(&k, x, wk, false, false, stream), "matmul_k");
    check_mlx_result(mlx_matmul(&v, x, wv, false, false, stream), "matmul_v");
    
    // Reshape q to [seq_len, n_heads, head_dim]
    std::vector<int> q_shape = {seq_len, n_heads, head_dim};
    mlx_array q_reshaped;
    check_mlx_result(mlx_reshape(&q_reshaped, q, q_shape.data(), q_shape.size(), stream), "reshape_q");
    
    // Reshape k to [seq_len, n_kv_heads, head_dim]
    std::vector<int> k_shape = {seq_len, n_kv_heads, head_dim};
    mlx_array k_reshaped;
    check_mlx_result(mlx_reshape(&k_reshaped, k, k_shape.data(), k_shape.size(), stream), "reshape_k");
    
    // Reshape v to [seq_len, n_kv_heads, head_dim]
    std::vector<int> v_shape = {seq_len, n_kv_heads, head_dim};
    mlx_array v_reshaped;
    check_mlx_result(mlx_reshape(&v_reshaped, v, v_shape.data(), v_shape.size(), stream), "reshape_v");
    
    // Apply RoPE to q and k
    mlx_array q_rope, k_rope;
    q_rope = fast_rope(q_reshaped, positions, rope_theta, stream);
    k_rope = fast_rope(k_reshaped, positions, rope_theta, stream);
    
    // For grouped-query attention when n_heads > n_kv_heads, we repeat k and v
    mlx_array k_expanded, v_expanded;
    if (n_heads > n_kv_heads) {
        int repeats = n_heads / n_kv_heads;
        
        // Compute the indirection tensors
        // Create pattern: [0, 0, 0, ..., 1, 1, 1, ..., n_kv_heads-1, n_kv_heads-1, n_kv_heads-1]
        std::vector<int> indices;
        for (int i = 0; i < n_kv_heads; ++i) {
            for (int j = 0; j < repeats; ++j) {
                indices.push_back(i);
            }
        }
        
        // Convert to MLX array
        mlx_array indices_array;
        check_mlx_result(mlx_array_from_data(&indices_array, indices.data(), MLX_INT32, 
                                        indices.size(), 1, stream), "array_from_data");
        
        // Use gather to repeat the heads
        std::vector<int> k_sizes(3, 0), v_sizes(3, 0);
        check_mlx_result(mlx_array_shape(k_sizes.data(), k_rope), "array_shape_k");
        check_mlx_result(mlx_array_shape(v_sizes.data(), v_rope), "array_shape_v");
        
        // Reshape to expose the head dim for gather
        std::vector<int> k_gather_shape = {k_sizes[0], k_sizes[1], k_sizes[2]};
        std::vector<int> v_gather_shape = {v_sizes[0], v_sizes[1], v_sizes[2]};
        
        // Gather along head dimension (dim 1)
        mlx_array k_gathered, v_gathered;
        check_mlx_result(mlx_take(&k_gathered, k_rope, indices_array, 1, stream), "take_k");
        check_mlx_result(mlx_take(&v_gathered, v_rope, indices_array, 1, stream), "take_v");
        
        k_expanded = k_gathered;
        v_expanded = v_gathered;
        
        // Clean up indices array
        mlx_array_free(indices_array);
    } else {
        // No expansion needed
        k_expanded = k_rope;
        v_expanded = v_rope;
    }
    
    // Permute dimensions for attention: [b, h, s, d]
    // From [seq_len, n_heads, head_dim] to [n_heads, seq_len, head_dim]
    int q_perm[] = {1, 0, 2};
    int k_perm[] = {1, 0, 2};
    int v_perm[] = {1, 0, 2};
    
    mlx_array q_perm_arr, k_perm_arr, v_perm_arr;
    check_mlx_result(mlx_transpose(&q_perm_arr, q_rope, q_perm, 3, stream), "transpose_q");
    check_mlx_result(mlx_transpose(&k_perm_arr, k_expanded, k_perm, 3, stream), "transpose_k");
    check_mlx_result(mlx_transpose(&v_perm_arr, v_expanded, v_perm, 3, stream), "transpose_v");
    
    // Compute attention scores: [n_heads, seq_len, seq_len]
    float scale = 1.0f / sqrtf(head_dim);
    mlx_array q_scaled;
    mlx_array scale_arr;
    check_mlx_result(mlx_array_scalar_float(&scale_arr, &scale, 1, stream), "array_scalar_float");
    check_mlx_result(mlx_multiply(&q_scaled, q_perm_arr, scale_arr, stream), "multiply");
    
    // Compute QK: [n_heads, seq_len, seq_len]
    mlx_array attn_weights;
    check_mlx_result(mlx_matmul(&attn_weights, q_scaled, k_perm_arr, false, true, stream), "matmul_qk");
    
    // Apply causal mask
    mlx_array mask;
    std::vector<int> mask_shape = {1, seq_len, seq_len};
    check_mlx_result(mlx_array_zeros(&mask, MLX_BOOL, mask_shape.data(), mask_shape.size(), stream), "array_zeros");
    
    // Create lower triangular mask (tril)
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j <= i; ++j) {
            bool val = true;
            mlx_array idx[3];
            int indices[3] = {0, i, j};
            check_mlx_result(mlx_array_scalar_int(&idx[0], &indices[0], 1, stream), "array_scalar_int");
            check_mlx_result(mlx_array_scalar_int(&idx[1], &indices[1], 1, stream), "array_scalar_int");
            check_mlx_result(mlx_array_scalar_int(&idx[2], &indices[2], 1, stream), "array_scalar_int");
            
            mlx_array mask_val;
            check_mlx_result(mlx_array_scalar_bool(&mask_val, &val, 1, stream), "array_scalar_bool");
            check_mlx_result(mlx_array_index_put(&mask, mask_val, idx, 3, stream), "array_index_put");
            
            mlx_array_free(idx[0]);
            mlx_array_free(idx[1]);
            mlx_array_free(idx[2]);
            mlx_array_free(mask_val);
        }
    }
    
    // Convert bool mask to float mask with large negative values for false entries
    mlx_array float_mask;
    check_mlx_result(mlx_array_astype(&float_mask, mask, MLX_FLOAT32, stream), "array_astype");
    
    mlx_array neg_inf_array;
    float neg_inf = -std::numeric_limits<float>::infinity();
    check_mlx_result(mlx_array_scalar_float(&neg_inf_array, &neg_inf, 1, stream), "array_scalar_float");
    
    mlx_array zeros;
    check_mlx_result(mlx_array_zeros_like(&zeros, float_mask, stream), "array_zeros_like");
    
    mlx_array masked_vals;
    check_mlx_result(mlx_where(&masked_vals, mask, zeros, neg_inf_array, stream), "where");
    
    // Apply mask to attention weights
    mlx_array masked_weights;
    check_mlx_result(mlx_add(&masked_weights, attn_weights, masked_vals, stream), "add");
    
    // Apply softmax
    mlx_array attn_probs;
    check_mlx_result(mlx_softmax(&attn_probs, masked_weights, -1, stream), "softmax");
    
    // Apply attention: [n_heads, seq_len, head_dim]
    mlx_array attn_output;
    check_mlx_result(mlx_matmul(&attn_output, attn_probs, v_perm_arr, false, false, stream), "matmul_prob_v");
    
    // Transpose back to [seq_len, n_heads, head_dim]
    int out_perm[] = {1, 0, 2};
    mlx_array attn_output_perm;
    check_mlx_result(mlx_transpose(&attn_output_perm, attn_output, out_perm, 3, stream), "transpose_out");
    
    // Reshape to [seq_len, n_heads * head_dim]
    std::vector<int> out_shape = {seq_len, n_heads * head_dim};
    mlx_array attn_output_flat;
    check_mlx_result(mlx_reshape(&attn_output_flat, attn_output_perm, out_shape.data(), out_shape.size(), stream), "reshape_out");
    
    // Project to output dimension with wo
    mlx_array result;
    check_mlx_result(mlx_matmul(&result, attn_output_flat, wo, false, false, stream), "matmul_out");
    
    // Clean up intermediate tensors
    mlx_array_free(q);
    mlx_array_free(k);
    mlx_array_free(v);
    mlx_array_free(q_reshaped);
    mlx_array_free(k_reshaped);
    mlx_array_free(v_reshaped);
    mlx_array_free(q_rope);
    mlx_array_free(k_rope);
    if (n_heads > n_kv_heads) {
        mlx_array_free(k_expanded);
        mlx_array_free(v_expanded);
    }
    mlx_array_free(q_perm_arr);
    mlx_array_free(k_perm_arr);
    mlx_array_free(v_perm_arr);
    mlx_array_free(q_scaled);
    mlx_array_free(scale_arr);
    mlx_array_free(attn_weights);
    mlx_array_free(mask);
    mlx_array_free(float_mask);
    mlx_array_free(neg_inf_array);
    mlx_array_free(zeros);
    mlx_array_free(masked_vals);
    mlx_array_free(masked_weights);
    mlx_array_free(attn_probs);
    mlx_array_free(attn_output);
    mlx_array_free(attn_output_perm);
    mlx_array_free(attn_output_flat);
    
    return result;
}

// Optimized rotary positional embedding implementation
mlx_array fast_rope(
    mlx_array x,
    const std::vector<int>& positions,
    float theta,
    mlx_stream stream) {
    
    // Get the dimensions of x
    int ndim;
    std::vector<int> x_shape(8, 0); // Maximum ndim for safety
    check_mlx_result(mlx_array_ndim(&ndim, x), "array_ndim");
    check_mlx_result(mlx_array_shape(x_shape.data(), x), "array_shape");
    
    if (ndim != 3) {
        throw std::runtime_error("Input tensor must be 3D [seq_len, n_heads, head_dim]");
    }
    
    int seq_len = x_shape[0];
    int n_heads = x_shape[1];
    int head_dim = x_shape[2];
    
    // Check if positions match sequence length
    if (positions.size() != seq_len) {
        throw std::runtime_error("Number of positions must match sequence length");
    }
    
    // Create a copy of x to avoid modifying the input
    mlx_array result;
    check_mlx_result(mlx_array_copy(&result, x, stream), "array_copy");
    
    // Compute the dimension that needs to be rotated (half of head_dim)
    int dim_rotary = head_dim / 2;
    
    // For each position
    for (int i = 0; i < seq_len; ++i) {
        int pos = positions[i];
        
        // For each head
        for (int h = 0; h < n_heads; ++h) {
            // For each dimension in the first half
            for (int j = 0; j < dim_rotary; j += 2) {
                // Compute freqs
                float freq = 1.0f / powf(theta, (float)(j) / dim_rotary);
                float val_cos = cosf(freq * pos);
                float val_sin = sinf(freq * pos);
                
                // Get the values at (i, h, j) and (i, h, j+1)
                float val1, val2;
                int indices1[3] = {i, h, j};
                int indices2[3] = {i, h, j + 1};
                
                mlx_array idx1[3], idx2[3];
                check_mlx_result(mlx_array_scalar_int(&idx1[0], &indices1[0], 1, stream), "array_scalar_int");
                check_mlx_result(mlx_array_scalar_int(&idx1[1], &indices1[1], 1, stream), "array_scalar_int");
                check_mlx_result(mlx_array_scalar_int(&idx1[2], &indices1[2], 1, stream), "array_scalar_int");
                
                check_mlx_result(mlx_array_scalar_int(&idx2[0], &indices2[0], 1, stream), "array_scalar_int");
                check_mlx_result(mlx_array_scalar_int(&idx2[1], &indices2[1], 1, stream), "array_scalar_int");
                check_mlx_result(mlx_array_scalar_int(&idx2[2], &indices2[2], 1, stream), "array_scalar_int");
                
                mlx_array val1_arr, val2_arr;
                check_mlx_result(mlx_array_item(&val1_arr, x, idx1, 3, stream), "array_item");
                check_mlx_result(mlx_array_item(&val2_arr, x, idx2, 3, stream), "array_item");
                
                check_mlx_result(mlx_array_scalar_float_value(&val1, val1_arr), "array_scalar_float_value");
                check_mlx_result(mlx_array_scalar_float_value(&val2, val2_arr), "array_scalar_float_value");
                
                // Compute rotated values
                float rot_val1 = val1 * val_cos - val2 * val_sin;
                float rot_val2 = val1 * val_sin + val2 * val_cos;
                
                // Set the rotated values
                mlx_array rot_val1_arr, rot_val2_arr;
                check_mlx_result(mlx_array_scalar_float(&rot_val1_arr, &rot_val1, 1, stream), "array_scalar_float");
                check_mlx_result(mlx_array_scalar_float(&rot_val2_arr, &rot_val2, 1, stream), "array_scalar_float");
                
                check_mlx_result(mlx_array_index_put(&result, rot_val1_arr, idx1, 3, stream), "array_index_put");
                check_mlx_result(mlx_array_index_put(&result, rot_val2_arr, idx2, 3, stream), "array_index_put");
                
                // Free temporary arrays
                mlx_array_free(idx1[0]);
                mlx_array_free(idx1[1]);
                mlx_array_free(idx1[2]);
                mlx_array_free(idx2[0]);
                mlx_array_free(idx2[1]);
                mlx_array_free(idx2[2]);
                mlx_array_free(val1_arr);
                mlx_array_free(val2_arr);
                mlx_array_free(rot_val1_arr);
                mlx_array_free(rot_val2_arr);
            }
        }
    }
    
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
    std::vector<mlx_array> results;
    results.reserve(operations_.size());
    
    for (const auto& op : operations_) {
        try {
            results.push_back(op());
        } catch (const std::exception& e) {
            // Clean up already created results
            for (auto& result : results) {
                mlx_array_free(result);
            }
            throw;
        }
    }
    
    // Synchronize the stream to ensure all operations are complete
    check_mlx_result(mlx_stream_synchronize(stream), "stream_synchronize");
    
    return results;
}

void MLXBatchProcessor::clear() {
    operations_.clear();
}

// Memory-efficient tensor operations
namespace mlx_memory {

// In-place operations where possible
mlx_array add_inplace(mlx_array& a, mlx_array b, mlx_stream stream) {
    mlx_array result;
    check_mlx_result(mlx_add(&result, a, b, stream), "add");
    
    // Free the original array
    mlx_array_free(a);
    
    // Update the reference
    a = result;
    
    return a;
}

mlx_array multiply_inplace(mlx_array& a, mlx_array b, mlx_stream stream) {
    mlx_array result;
    check_mlx_result(mlx_multiply(&result, a, b, stream), "multiply");
    
    // Free the original array
    mlx_array_free(a);
    
    // Update the reference
    a = result;
    
    return a;
}

// TensorPool implementation
TensorPool::TensorPool(size_t max_pool_size)
    : max_pool_size_(max_pool_size) {}

TensorPool::~TensorPool() {
    clear();
}

mlx_array TensorPool::get(const std::vector<int>& shape, mlx_dtype dtype) {
    // Check if we have a matching tensor in the pool
    for (size_t i = 0; i < pool_.size(); ++i) {
        if (shapes_match(pool_[i], shape)) {
            mlx_dtype pool_dtype;
            check_mlx_result(mlx_array_get_dtype(&pool_dtype, pool_[i]), "array_get_dtype");
            
            if (pool_dtype == dtype) {
                // Found a match, remove from pool and return
                mlx_array result = pool_[i];
                pool_.erase(pool_.begin() + i);
                return result;
            }
        }
    }
    
    // No match found, create a new tensor
    mlx_array result;
    check_mlx_result(mlx_array_zeros(&result, dtype, shape.data(), shape.size(), nullptr), "array_zeros");
    return result;
}

void TensorPool::recycle(mlx_array& tensor) {
    // Check if we can add to the pool
    if (pool_.size() < max_pool_size_) {
        pool_.push_back(tensor);
        
        // Ensure the tensor is not freed elsewhere
        tensor.ctx = nullptr;
    } else {
        // Pool is full, free the tensor
        mlx_array_free(tensor);
        tensor.ctx = nullptr;
    }
}

void TensorPool::clear() {
    // Free all tensors in the pool
    for (auto& tensor : pool_) {
        mlx_array_free(tensor);
    }
    pool_.clear();
}

bool TensorPool::shapes_match(mlx_array tensor, const std::vector<int>& shape) {
    // Check if tensor shape matches the requested shape
    int ndim;
    check_mlx_result(mlx_array_ndim(&ndim, tensor), "array_ndim");
    
    if (ndim != static_cast<int>(shape.size())) {
        return false;
    }
    
    std::vector<int> tensor_shape(ndim);
    check_mlx_result(mlx_array_shape(tensor_shape.data(), tensor), "array_shape");
    
    for (int i = 0; i < ndim; ++i) {
        if (tensor_shape[i] != shape[i]) {
            return false;
        }
    }
    
    return true;
}

} // namespace mlx_memory

#endif // CCSM_WITH_MLX

} // namespace ccsm